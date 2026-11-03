"""Upload consent and confirmation: claim-once, validate twice, queue-or-hand-over."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from app import db
from app.admission import admit_job
from app.contracts import (
    META_UPLOAD_MAX_HOURS,
    MetaFileId,
    MetaFilesClient,
    MetaFilesError,
    MetaReadUnsupported,
    MetaUpload,
    MetaUploadExpiryError,
)
from app.meta_files import (
    MetaClientUnavailableError,
    ProductionMetaFilesClient,
    select_meta_files_client,
)
from app.settings import ProviderSettings, Settings
from app.uploads import (
    MAX_DURATION_S,
    ConfirmationConflictError,
    ConfirmationFailedError,
    confirm_and_send,
    prepare_preview,
)
from app.uploads_store import UploadStoreImpl


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start if start is not None else db.utcnow()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> datetime:
        self.now = self.now + timedelta(**kwargs)
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def uploads(lane_db: Path, clock: FakeClock) -> UploadStoreImpl:
    return UploadStoreImpl(lane_db, clock=clock)


@pytest.fixture(autouse=True)
def _use_fake_clock(monkeypatch, clock: FakeClock):
    monkeypatch.setattr(db, "utcnow", clock)


class FakeMetaFiles:
    """In-memory provider: files by id and by idempotency key, scriptable."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.files: dict[str, MetaUpload] = {}
        self.by_key: dict[str, MetaUpload] = {}
        self.upload_calls: list[tuple[Path, int, str]] = []
        self.find_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
        self.sent_bytes: dict[str, bytes] = {}
        self.on_upload = None
        self.on_find = None
        self.on_delete = None
        self.on_get = None
        self._counter = 0

    def _now(self) -> datetime:
        return self.clock() if self.clock is not None else db.utcnow()

    def upload_file(self, path, *, expires_after_hours, idempotency_key):
        if self.on_upload is not None:
            return self.on_upload(path, expires_after_hours, idempotency_key)
        self.upload_calls.append(
            (Path(path), expires_after_hours, idempotency_key)
        )
        if not idempotency_key:
            raise MetaFilesError("rejected key")
        if idempotency_key in self.by_key:
            return self.by_key[idempotency_key]
        self._counter += 1
        found = MetaUpload(
            MetaFileId(f"meta-{self._counter:04d}"),
            self._now() + timedelta(hours=expires_after_hours),
        )
        self.files[found.file_id] = found
        self.by_key[idempotency_key] = found
        self.sent_bytes[found.file_id] = Path(path).read_bytes()
        return found

    def find_upload(self, idempotency_key):
        if self.on_find is not None:
            return self.on_find(idempotency_key)
        self.find_calls.append(idempotency_key)
        return self.by_key.get(idempotency_key)

    def delete_file(self, file_id):
        if self.on_delete is not None:
            return self.on_delete(file_id)
        self.delete_calls.append(file_id)
        self.files.pop(file_id, None)
        for key, value in list(self.by_key.items()):
            if value.file_id == file_id:
                del self.by_key[key]
        return None

    def get_file(self, file_id):
        if self.on_get is not None:
            return self.on_get(file_id)
        self.get_calls.append(file_id)
        return self.files.get(file_id)


assert isinstance(FakeMetaFiles(), MetaFilesClient)


class _StubState:
    def __init__(self) -> None:
        self.handler = None


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet the stub
        pass

    def _run(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        parsed = urlparse(self.path)
        status, payload = self.server.state.handler(
            self.command, parsed.path, parse_qs(parsed.query), body
        )
        raw = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if raw:
            self.wfile.write(raw)

    do_GET = _run
    do_POST = _run
    do_DELETE = _run


@pytest.fixture
def meta_stub():
    """Local scripted Meta HTTP stub; handler set per test, no outside net."""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    server.state = _StubState()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=10)
    server.server_close()


def _stub_client(meta_stub, clock: FakeClock) -> ProductionMetaFilesClient:
    port = meta_stub.server_address[1]
    return ProductionMetaFilesClient(
        "test-key", base_url=f"http://127.0.0.1:{port}", timeout_s=5.0
    )


def _make_clip(path: Path, *, duration: float = 1.0, with_audio: bool = True) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc=size=160x120:rate=10:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-shortest", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory) -> Path:
    return _make_clip(tmp_path_factory.mktemp("media") / "short.mp4")


def _admit(tag: str):
    return admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")


def _prepare(uploads, ctx, clip: Path, *, name: str = "clip.mp4"):
    with clip.open("rb") as handle:
        return prepare_preview(ctx, handle, name, uploads=uploads)


def _ffprobe_streams(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(out.stdout or "{}").get("streams", [])


def _meta_row(lane_db: Path, file_id: str) -> dict | None:
    conn = sqlite3.connect(lane_db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM meta_files WHERE id = ?", (file_id,)
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def test_audio_stripped_preview_is_exact_sent_file(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("exact")
    preview = _prepare(uploads, ctx, short_clip)
    assert preview.audio_removed is True
    assert [s.get("codec_type") for s in _ffprobe_streams(preview.local_video)] == [
        "video"
    ]
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert file_id in fake.sent_bytes
    assert fake.sent_bytes[file_id] == preview.local_video.read_bytes()


def test_real_ffmpeg_fixture_has_no_audio_and_sent_bytes_match_preview(
    uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    source_types = [s.get("codec_type") for s in _ffprobe_streams(short_clip)]
    assert "audio" in source_types  # the fixture really carries audio
    ctx = _admit("realff")
    preview = _prepare(uploads, ctx, short_clip)
    preview_types = [s.get("codec_type") for s in _ffprobe_streams(preview.local_video)]
    assert "audio" not in preview_types
    assert "video" in preview_types
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    from app.uploads import META_UPLOAD_REQUEST_HOURS

    (sent_path, hours, key) = fake.upload_calls[0]
    assert hours == META_UPLOAD_REQUEST_HOURS == META_UPLOAD_MAX_HOURS - 1
    assert sent_path == preview.local_video
    assert fake.sent_bytes[file_id] == preview.local_video.read_bytes()


def test_preview_is_admitted_artifact_and_survives_worker_reload(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("reload")
    preview = _prepare(uploads, ctx, short_clip)
    conn = sqlite3.connect(lane_db)
    try:
        kind = conn.execute(
            "SELECT kind FROM artifacts WHERE job_id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
        state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert kind == "upload_preview"
    assert state == "waiting"
    # A second worker (fresh store, same database) reloads the same record.
    worker_b = UploadStoreImpl(lane_db, clock=clock)
    loaded = worker_b.load_job(ctx.job_id)
    assert loaded is not None
    _, record = loaded
    assert record.preview.local_video == preview.local_video
    assert record.preview.local_video.read_bytes() == preview.local_video.read_bytes()
    assert record.state == "preview"
    assert record.hidden is False


def test_no_meta_call_before_confirm(
    uploads: UploadStoreImpl, short_clip: Path, fake_meta_files
) -> None:
    ctx = _admit("nocall")
    preview = _prepare(uploads, ctx, short_clip)
    assert preview.local_video.is_file()
    # The shared fake raises on any Meta call; preparing, loading, and
    # re-confirming nothing must stay silent.
    assert uploads.load_job(ctx.job_id) is not None
    with pytest.raises(NotImplementedError):
        fake_meta_files.upload_file(preview.local_video, expires_after_hours=24,
                                    idempotency_key="k")


def test_two_workers_confirm_once_and_only_claim_winner_uploads(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("race")
    _prepare(uploads, ctx, short_clip)
    worker_b = UploadStoreImpl(lane_db, clock=clock)
    fake = FakeMetaFiles(clock)
    entered = threading.Event()
    proceed = threading.Event()

    def _gated(path, expires_after_hours, idempotency_key):
        entered.set()
        assert proceed.wait(timeout=10), "confirm race stalled"
        fake.on_upload = None
        try:
            return fake.upload_file(
                path,
                expires_after_hours=expires_after_hours,
                idempotency_key=idempotency_key,
            )
        finally:
            fake.on_upload = _gated

    fake.on_upload = _gated
    outcome: dict = {}

    def _winner() -> None:
        try:
            outcome["id"] = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
        except Exception as error:  # pragma: no cover - winner must win
            outcome["error"] = error

    thread = threading.Thread(target=_winner, daemon=True)
    thread.start()
    assert entered.wait(timeout=10), "winner never reached upload"
    # The loser claims while the winner uploads: conflict, no second call.
    with pytest.raises(ConfirmationConflictError):
        confirm_and_send(ctx.job_id, uploads=worker_b, client=fake)
    proceed.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert outcome.get("id") is not None
    assert len(fake.upload_calls) == 1
    # A late replay after completion returns the stored id, still one call.
    again = confirm_and_send(ctx.job_id, uploads=worker_b, client=fake)
    assert again == outcome["id"]
    assert len(fake.upload_calls) == 1


def test_meta_persist_failure_requests_delete_and_stale_claim_never_reuploads(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path,
    monkeypatch,
) -> None:
    ctx = _admit("persist")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    attempted: list = []

    def _boom(*args, **kwargs):
        attempted.append(args)
        raise RuntimeError("commit lost")

    monkeypatch.setattr(uploads, "complete_confirmation", _boom)
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert attempted, "persist path never attempted the attach"
    # The known id is durably queued and direct deletion was attempted.
    record = uploads.load_job(ctx.job_id)[1]
    assert record.meta_file_id is not None
    row = _meta_row(lane_db, record.meta_file_id)
    assert row is not None and row["delete_state"] == "pending"
    assert row["delete_requested_at"] is not None
    assert record.meta_file_id in fake.delete_calls
    assert record.failure_count >= 1
    # A stale second confirm never uploads again: the failed job is
    # refused at the claim with the settled copy.
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert len(fake.upload_calls) == 1


def _assert_known_id_queued(lane_db, uploads, fake, job_id, file_id):
    """Shared invalid-expiry assertions: queued, delete tried, never handed over."""
    record = uploads.load_job(job_id)[1]
    assert record.meta_file_id == file_id
    row = _meta_row(lane_db, file_id)
    assert row is not None
    assert row["delete_state"] == "pending"
    assert row["delete_requested_at"] is not None
    assert file_id in fake.delete_calls
    assert record.failure_count >= 1
    assert record.hidden is True
    return record


def test_seven_day_meta_expiry_is_rejected_and_queued_for_delete(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("sevenday")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)

    def _seven_day(path, hours, key):
        return MetaUpload(MetaFileId("meta-seven"), clock() + timedelta(days=7))

    fake.on_upload = _seven_day
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    _assert_known_id_queued(lane_db, uploads, fake, ctx.job_id, "meta-seven")


def test_already_expired_response_is_rejected_and_queued_for_delete(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("expired")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)

    def _stale(path, hours, key):
        return MetaUpload(MetaFileId("meta-old"), clock() - timedelta(hours=1))

    fake.on_upload = _stale
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    _assert_known_id_queued(lane_db, uploads, fake, ctx.job_id, "meta-old")


def test_slow_response_expiring_before_confirmation_is_rejected_and_queued_for_delete(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("slow")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)

    def _slow(path, hours, key):
        expiry = clock() + timedelta(hours=hours)
        clock.advance(hours=hours, minutes=1)  # now >= expires before confirm
        return MetaUpload(MetaFileId("meta-slow"), expiry)

    fake.on_upload = _slow
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    _assert_known_id_queued(lane_db, uploads, fake, ctx.job_id, "meta-slow")


def test_expiry_error_without_attributes_still_queues_by_constructor_args(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    """A raw contract expiry error (mirror stores nothing) still queues by args."""
    ctx = _admit("rawerr")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    raw = MetaUploadExpiryError(MetaFileId("meta-raw"), None)
    assert not hasattr(raw, "file_id")

    def _raise_raw(path, hours, key):
        raise raw

    fake.on_upload = _raise_raw
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    record = _assert_known_id_queued(lane_db, uploads, fake, ctx.job_id, "meta-raw")
    row = _meta_row(lane_db, "meta-raw")
    assert row is not None
    # Missing expiry: the conservative cleanup deadline, not a provider claim.
    assert db.parse_dt(row["expires_at"]) == clock()


def test_upload_does_not_start_after_pre_call_deadline(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path,
    monkeypatch,
) -> None:
    ctx = _admit("stale")
    _prepare(uploads, ctx, short_clip)
    # The store claimed at T; the upload module's own clock reads past guard - 24h.
    late = FakeClock(clock() + timedelta(hours=2))
    monkeypatch.setattr(db, "utcnow", late)
    fake = FakeMetaFiles(clock)
    with pytest.raises(ConfirmationFailedError, match="window passed"):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert fake.upload_calls == []
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "failed"
    assert record.claim_id is not None  # retained for lookup-backed recovery
    assert record.failure_count >= 1
    recoverable = [r.preview.job_id for r in uploads.list_recoverable(late())]
    assert ctx.job_id in recoverable


def test_meta_adapter_rejects_missing_naive_or_later_expiry_with_file_id_error(
    meta_stub, clock: FakeClock, short_clip: Path
) -> None:
    client = _stub_client(meta_stub, clock)
    cases = [
        ({"file_id": "meta-missing"}, None),
        ({"file_id": "meta-naive", "expires_at": "2026-10-28T12:00:00"}, None),
        (
            {"file_id": "meta-late",
             "expires_at": (clock() + timedelta(days=7)).isoformat()},
            clock() + timedelta(days=7),
        ),
    ]
    for payload, expected_expiry in cases:
        def _handler(method, path, query, body, _payload=payload):
            assert method == "POST" and path == "/files"
            return 201, dict(_payload)

        meta_stub.state.handler = _handler
        with pytest.raises(MetaUploadExpiryError) as excinfo:
            client.upload_file(short_clip, expires_after_hours=24,
                               idempotency_key="key-1")
        assert excinfo.value.file_id == payload["file_id"]
        assert excinfo.value.expires_at == expected_expiry
    # No parseable id at all: a plain retryable error, never partial.
    meta_stub.state.handler = lambda m, p, q, b: (201, {"noid": True})
    with pytest.raises(MetaFilesError):
        client.upload_file(short_clip, expires_after_hours=24, idempotency_key="k")


def test_private_meta_client_protocol_covers_upload_find_delete_and_get(
    meta_stub, clock: FakeClock, short_clip: Path
) -> None:
    store: dict[str, dict] = {}
    by_key: dict[str, str] = {}

    def _handler(method, path, query, body):
        if method == "POST" and path == "/files":
            key = query.get("key", query.get("idempotency_key", [""]))[0]
            hours = int(query.get("expires_after_hours", ["24"])[0])
            if key in by_key:
                return 200, store[by_key[key]]
            file_id = f"meta-{len(store) + 1:04d}"
            payload = {
                "file_id": file_id,
                "expires_at": (clock() + timedelta(hours=hours)).isoformat(),
            }
            store[file_id] = payload
            by_key[key] = file_id
            return 201, payload
        if method == "GET" and path == "/files" and "key" in query:
            file_id = by_key.get(query["key"][0])
            if file_id is None:
                return 404, None
            return 200, store[file_id]
        if method == "GET" and path == "/files":
            return 200, {"files": sorted(store)}
        if method == "GET" and path.startswith("/files/"):
            file_id = path.rsplit("/", 1)[-1]
            if file_id not in store:
                return 404, None
            return 200, store[file_id]
        if method == "DELETE" and path.startswith("/files/"):
            file_id = path.rsplit("/", 1)[-1]
            store.pop(file_id, None)
            for key, value in list(by_key.items()):
                if value == file_id:
                    del by_key[key]
            return 204, None
        return 404, None

    meta_stub.state.handler = _handler
    client = _stub_client(meta_stub, clock)
    assert isinstance(client, MetaFilesClient)
    first = client.upload_file(short_clip, expires_after_hours=24,
                               idempotency_key="stable-key")
    replay = client.upload_file(short_clip, expires_after_hours=24,
                                idempotency_key="stable-key")
    assert replay == first  # same key replays the same id and expiry
    assert client.find_upload("stable-key") == first
    assert client.find_upload("no-such-key") is None
    assert client.get_file(first.file_id) == first
    client.delete_file(first.file_id)
    client.delete_file(first.file_id)  # absent delete is idempotent success
    assert client.get_file(first.file_id) is None


def test_meta_client_absence_and_error_semantics(meta_stub, clock: FakeClock) -> None:
    client = _stub_client(meta_stub, clock)

    def _empty_provider(method, path, query, body):
        if method == "GET" and path == "/files" and "key" not in query:
            return 200, {"files": []}
        return 404, None

    meta_stub.state.handler = _empty_provider
    assert client.get_file(MetaFileId("gone")) is None
    assert client.delete_file(MetaFileId("gone")) is None
    assert client.find_upload("gone") is None
    # GET 404 with a broken list read fails closed instead of verifying.
    meta_stub.state.handler = lambda m, p, q, b: (404, None)
    with pytest.raises(MetaFilesError):
        client.get_file(MetaFileId("gone"))
    meta_stub.state.handler = lambda m, p, q, b: (500, {"error": "boom"})
    with pytest.raises(MetaFilesError):
        client.get_file(MetaFileId("x"))
    with pytest.raises(MetaFilesError):
        client.delete_file(MetaFileId("x"))
    with pytest.raises(MetaFilesError):
        client.find_upload("x")
    meta_stub.state.handler = lambda m, p, q, b: (401, None)
    with pytest.raises(MetaFilesError):
        client.get_file(MetaFileId("x"))
    meta_stub.state.handler = lambda m, p, q, b: (200, {"files": [{"bogus": 1}]})
    with pytest.raises(MetaFilesError):
        client.get_file(MetaFileId("x"))
    # GET unsupported and list unsupported: genuinely unavailable reads.
    meta_stub.state.handler = lambda m, p, q, b: (405, None)
    with pytest.raises(MetaReadUnsupported):
        client.get_file(MetaFileId("x"))
    # Transport failure stays a retryable error, never absence.
    dead = ProductionMetaFilesClient("k", base_url="http://127.0.0.1:1", timeout_s=2.0)
    with pytest.raises(MetaFilesError):
        dead.get_file(MetaFileId("x"))
    # Empty key and bad file arguments fail fast without I/O.
    with pytest.raises(MetaFilesError):
        client.upload_file(Path("/nonexistent"), expires_after_hours=99,
                           idempotency_key="")
    with pytest.raises(MetaClientUnavailableError):
        ProductionMetaFilesClient("  ")
    with pytest.raises(MetaClientUnavailableError):
        select_meta_files_client(Settings(data_root=Path("var"),
                                          db_path=Path("var/x"), production=False))


def test_production_client_is_selected_by_settings(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path, db_path=tmp_path / "t.sqlite3",
                        production=False, meta_api_key="secret-key")
    client = select_meta_files_client(settings)
    assert isinstance(client, ProductionMetaFilesClient)
    assert isinstance(client, MetaFilesClient)
    assert "secret-key" not in repr(client)
    with pytest.raises(MetaClientUnavailableError):
        select_meta_files_client(
            Settings(data_root=tmp_path, db_path=tmp_path / "t.sqlite3",
                     production=False)
        )


@contextlib.contextmanager
def _overrides(**kwargs):
    from app.routes import uploads as routes_mod

    routes_mod.set_test_overrides(**kwargs)
    try:
        yield routes_mod
    finally:
        routes_mod.clear_test_overrides()


@pytest.fixture
def route_client(lane_db: Path, tmp_path: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routes import uploads as routes_mod

    application = FastAPI()
    application.include_router(routes_mod.router)
    # Replay keeps the pipeline preflight buildable: live mode with replay
    # providers can never start a job, so confirm refuses before upload.
    application.state.settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay",
    )
    with _overrides():
        yield TestClient(application)


def test_upload_consent_confirm_cancel_round_trip(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.routes import uploads as routes_mod

    fake = FakeMetaFiles(clock)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        body = route_client.get("/upload").text
        assert "Continue to preview" in body
        assert "300 seconds or less" in body
        posted = route_client.post(
            "/upload",
            files={"clip": ("clip.mp4", short_clip.read_bytes(), "video/mp4")},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        preview_url = posted.headers["location"]
        assert preview_url.startswith("/preview/")
        shown = route_client.get(preview_url)
        assert shown.status_code == 200
        assert "Check the exact video" in shown.text
        assert "within 24 hours at the latest" in shown.text
        assert "Confirm and send" in shown.text
        video = route_client.get(preview_url + "/video")
        assert video.status_code == 200
        job_id = preview_url.rsplit("/", 1)[-1]
        served = uploads.load_job(job_id)[1].preview.local_video.read_bytes()
        assert video.content == served
        assert len(video.content) > 0
        # Both consent boxes are required.
        missing = route_client.post(f"/preview/{job_id}/confirm",
                                    data={"consent": "on"})
        assert missing.status_code == 400
        assert "both boxes" in missing.text
        confirmed = route_client.post(
            f"/preview/{job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert confirmed.status_code == 200
        assert "Confirmed." in confirmed.text
        assert len(fake.upload_calls) == 1
        # Idempotent re-confirm: same page, still one upload.
        again = route_client.post(
            f"/preview/{job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert again.status_code == 200
        assert len(fake.upload_calls) == 1
        cancelled = route_client.post(f"/preview/{job_id}/cancel")
        assert cancelled.status_code == 200
        assert "Cancelled." in cancelled.text
        assert "Your files were deleted." in cancelled.text
        assert uploads.load_job(job_id)[1].state == "cancelled"
    assert routes_mod._meta_client_override is None


def test_upload_route_refuses_over_limit_before_admission(
    route_client, lane_db: Path, uploads: UploadStoreImpl, short_clip: Path
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.uploads import OVER_LIMIT_MESSAGE

    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles()):
        from app.uploads import MAX_BYTES

        too_big = route_client.post(
            "/upload",
            files={"clip": ("big.mp4", b"\x00" * (MAX_BYTES + 1),
                            "video/mp4")},
        )
        assert too_big.status_code == 400
        assert OVER_LIMIT_MESSAGE in too_big.text
        refused_type = route_client.post(
            "/upload",
            files={"clip": ("clip.mp4", b"not a video" * 64, "video/mp4")},
        )
        assert refused_type.status_code == 400
        assert OVER_LIMIT_MESSAGE in refused_type.text
        conn = sqlite3.connect(lane_db)
        try:
            assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        finally:
            conn.close()


def test_upload_route_blocks_when_failures_reach_threshold(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.meta_files import UPLOAD_BLOCKED_COPY

    ctx = _admit("blocked")
    _prepare(uploads, ctx, short_clip)
    for index in range(3):
        uploads.record_failure(ctx.job_id, f"block-{index}", "meta_upload",
                               "boom", clock())
    assert uploads.public_uploads_blocked() is True
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        formed = route_client.get("/upload")
        assert UPLOAD_BLOCKED_COPY in formed.text
        posted = route_client.post(
            "/upload",
            files={"clip": ("clip.mp4", short_clip.read_bytes(), "video/mp4")},
        )
        assert posted.status_code == 503
        assert UPLOAD_BLOCKED_COPY in posted.text


def test_confirm_route_without_a_key_keeps_the_preview(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.meta_files import META_UNAVAILABLE_COPY

    ctx = _admit("nokey")
    _prepare(uploads, ctx, short_clip)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    # Live mode, no meta_client override and no key: honest 503, no claim.
    route_client.app.state.settings = Settings(
        data_root=db.data_root(), db_path=lane_db, production=False,
        providers=ProviderSettings(mode="live"),
    )
    with _overrides(uploads=uploads, lifecycle=lifecycle):
        denied = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert denied.status_code == 503
        assert META_UNAVAILABLE_COPY in denied.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "preview"
    assert record.claim_id is None


def test_preview_routes_reject_unknown_and_malicious_ids(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        assert route_client.get("/preview/no-such-job").status_code == 404
        evil = route_client.get("/preview/<script>alert(1)</script>")
        assert evil.status_code == 404
        assert "<script>" not in evil.text
        assert route_client.get("/preview/../x/video").status_code in (404, 405)
        gone = route_client.post("/preview/no-such-job/cancel")
        assert gone.status_code == 404


def test_templates_escape_every_dynamic_value() -> None:
    from app.routes import uploads as routes_mod

    evil = "<img src=x onerror=alert(1)>"
    upload_page = routes_mod._upload_page(error=evil)
    assert evil not in upload_page
    assert "&lt;img" in upload_page
    preview_page = routes_mod._preview_page(
        job_id=evil, duration=evil, size=evil, video_url=evil,
        state="consent", show_video=True, notice=evil,
    )
    assert evil not in preview_page
    assert "&lt;img" in preview_page
    # Server-generated JSON blob carries only numbers and settled copy.
    assert "PARTFIDELITY_LIMITS" in upload_page


def test_evil_filename_never_reflects_into_pages(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    lifecycle = LifecycleRepositoryImpl(lane_db)
    evil_name = "<script>alert(1)</script>.mp4"
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        posted = route_client.post(
            "/upload",
            files={"clip": (evil_name, short_clip.read_bytes(), "video/mp4")},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        shown = route_client.get(posted.headers["location"])
        assert shown.status_code == 200
        assert "<script>" not in shown.text


def test_selected_client_reads_endpoint_from_settings(tmp_path: Path) -> None:
    from app.meta_files import DEFAULT_META_FILES_BASE_URL

    settings = Settings(
        data_root=tmp_path, db_path=tmp_path / "t.sqlite3",
        production=False, meta_api_key="secret-key",
        meta_files_base_url="https://meta.example.test/v9/",
    )
    client = select_meta_files_client(settings)
    assert isinstance(client, ProductionMetaFilesClient)
    assert "secret-key" not in repr(client)
    assert "https://meta.example.test/v9" in repr(client)
    assert DEFAULT_META_FILES_BASE_URL not in repr(client)
    # No configured endpoint: the explicit stub URL still wins for tests.
    bare = Settings(
        data_root=tmp_path, db_path=tmp_path / "t.sqlite3",
        production=False, meta_api_key="secret-key",
    )
    stub = select_meta_files_client(bare, base_url="http://127.0.0.1:9")
    assert "http://127.0.0.1:9" in repr(stub)


def test_meta_files_base_url_is_validated_https_and_required_in_prod() -> None:
    from app.settings import load_settings

    good = load_settings({"META_FILES_BASE_URL": "https://files.meta.example/v1/"})
    assert good.meta_files_base_url == "https://files.meta.example/v1"
    for bad in (
        "http://files.meta.example/v1",
        "https://",
        "https://user:pass@files.meta.example/v1",
        "not-a-url",
        "ftp://files.meta.example/v1",
    ):
        with pytest.raises(RuntimeError, match="META_FILES_BASE_URL"):
            load_settings({"META_FILES_BASE_URL": bad})
    with pytest.raises(RuntimeError, match="META_FILES_BASE_URL"):
        load_settings(
            {
                "PARTFIDELITY_PRODUCTION": "1",
                "PARTFIDELITY_DATA_ROOT": "/tmp/x",
                "META_API_KEY": "k",
            }
        )
    prod = load_settings(
        {
            "PARTFIDELITY_PRODUCTION": "1",
            "PARTFIDELITY_DATA_ROOT": "/tmp/x",
            "META_API_KEY": "k",
            "META_FILES_BASE_URL": "https://files.meta.example/v1",
        }
    )
    assert prod.meta_files_base_url == "https://files.meta.example/v1"


class _RedirectState:
    def __init__(self) -> None:
        self.target_hits: list[dict] = []


class _RedirectHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _run(self) -> None:
        if self.path == "/catch":
            self.server.state.target_hits.append(dict(self.headers.items()))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        port = self.server.state.catch_port
        self.send_response(302)
        self.send_header(
            "Location", f"http://127.0.0.1:{port}/catch"
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = _run
    do_POST = _run
    do_DELETE = _run


def test_production_client_never_follows_redirects() -> None:
    first = HTTPServer(("127.0.0.1", 0), _RedirectHandler)
    first.state = _RedirectState()
    catch = HTTPServer(("127.0.0.1", 0), _RedirectHandler)
    catch.state = first.state
    first.state.catch_port = catch.server_address[1]
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (first, catch)
    ]
    for thread in threads:
        thread.start()
    try:
        client = ProductionMetaFilesClient(
            "test-key", base_url=f"http://127.0.0.1:{first.server_address[1]}",
            timeout_s=5.0,
        )
        with pytest.raises(MetaFilesError):
            client.get_file(MetaFileId("m-1"))
        with pytest.raises(MetaFilesError):
            client.find_upload("some-key")
        # The redirect target never saw a request, so the Authorization
        # header never left the configured origin.
        assert first.state.target_hits == []
    finally:
        first.shutdown()
        catch.shutdown()
        for thread in threads:
            thread.join(timeout=10)
        first.server_close()
        catch.server_close()


def test_find_upload_needs_an_explicit_no_match(
    meta_stub, clock: FakeClock,
) -> None:
    client = _stub_client(meta_stub, clock)
    # Empty or malformed 200s stay retryable; they never discard the key.
    meta_stub.state.handler = lambda m, p, q, b: (200, None)
    with pytest.raises(MetaFilesError):
        client.find_upload("k")
    for payload in ({"bogus": 1}, {"file_id": ""}, [1, 2]):
        meta_stub.state.handler = lambda m, p, q, b, pl=payload: (200, pl)
        with pytest.raises(MetaFilesError):
            client.find_upload("k")
    # Only 404 and explicit no-match documents prove absence.
    meta_stub.state.handler = lambda m, p, q, b: (404, None)
    assert client.find_upload("k") is None
    for payload in (
        {"found": False}, {"match": None}, {"file": None}, {"files": []}
    ):
        meta_stub.state.handler = lambda m, p, q, b, pl=payload: (200, pl)
        assert client.find_upload("k") is None


def test_find_upload_rejects_mixed_match_documents_as_retryable(
    meta_stub, clock: FakeClock,
) -> None:
    client = _stub_client(meta_stub, clock)
    expiry = "2026-10-28T12:00:00+00:00"
    mixed = [
        {"found": False, "file_id": "still-held", "expires_at": expiry},
        {"found": False, "file_id": "still-held"},
        {"found": False, "expires_at": expiry},
        {"found": False, "match": {"file_id": "still-held"}},
        {"found": True, "files": []},
        {"match": None, "file_id": "still-held", "expires_at": expiry},
        {"file": None, "files": ["meta-a"]},
        {"files": [], "file_id": "still-held", "expires_at": expiry},
        # Only the four exact singleton shapes prove absence: any error,
        # extra field, malformed id, or marker combination stays retryable.
        {"found": False, "error": "lookup timed out"},
        {"found": False, "error": None},
        {"found": False, "unexpected": 1},
        {"found": False, "file_id": None},
        {"found": False, "file_id": 123},
        {"found": False, "file_id": ""},
        {"found": False, "file_id": "   "},
        {"found": False, "expires_at": None},
        {"match": None, "error": "boom"},
        {"file": None, "error": "boom"},
        {"files": [], "error": "boom"},
        {"files": [], "file_id": None},
        {"match": None, "files": []},
        {"found": False, "match": None},
        {"found": False, "files": []},
        {"found": None},
        {"found": 0},
        {"found": "false"},
        {"found": True},
        {"files": None},
        {"files": {}},
        {"match": False},
        {"file": ""},
    ]
    for payload in mixed:
        meta_stub.state.handler = lambda m, p, q, b, pl=payload: (200, pl)
        with pytest.raises(MetaFilesError):
            client.find_upload("k")


def test_upload_open_failure_finishes_the_admitted_job(
    route_client, lane_db: Path, uploads: UploadStoreImpl,
    tmp_path: Path, short_clip: Path, monkeypatch,
) -> None:
    from pathlib import Path as SysPath

    from app.ledger import LifecycleRepositoryImpl

    real_open = SysPath.open
    rb_opens = 0

    def _counting_open(self, mode="r", *args, **kwargs):
        nonlocal rb_opens
        if mode == "rb" and ".staging" in str(self):
            rb_opens += 1
            if rb_opens > 1:  # hashing passed; the handoff open fails
                raise OSError("disk lost the staged file")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(SysPath, "open", _counting_open)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles()):
        failed = route_client.post(
            "/upload",
            files={"clip": ("clip.mp4", short_clip.read_bytes(), "video/mp4")},
        )
        assert failed.status_code == 500
        assert "could not process" in failed.text
    assert rb_opens == 2
    conn = sqlite3.connect(lane_db)
    try:
        states = [row[0] for row in conn.execute("SELECT state FROM jobs")]
        assert states == ["failed"]
        assert conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM reservations WHERE state = 'held'"
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_upload_interruption_finishes_the_admitted_job(
    route_client, lane_db: Path, uploads: UploadStoreImpl,
    short_clip: Path, monkeypatch,
) -> None:
    from app import uploads as uploads_mod
    from app.ledger import LifecycleRepositoryImpl

    class Boom(BaseException):
        pass

    def _interrupted(ctx, upload, name, *, uploads, max_duration_s=None):
        raise Boom()

    monkeypatch.setattr(uploads_mod, "prepare_preview", _interrupted)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles()):
        with pytest.raises(Boom):
            route_client.post(
                "/upload",
                files={"clip": ("clip.mp4", short_clip.read_bytes(),
                                "video/mp4")},
            )
    conn = sqlite3.connect(lane_db)
    try:
        states = [row[0] for row in conn.execute("SELECT state FROM jobs")]
    finally:
        conn.close()
    assert states == ["failed"]


def test_cancel_with_unattached_claim_renders_pending_not_deleted(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("claimnoclip")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    assert claim.acquired is True
    fake = FakeMetaFiles(clock)  # provider holds nothing for the key
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        pending = route_client.post(f"/preview/{ctx.job_id}/cancel")
        assert pending.status_code == 202
        assert "still running" in pending.text
        assert "Your files were deleted." not in pending.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "cancelling"
    assert record.claim_id == claim.record.claim_id
    assert record.meta_file_id is None
    assert claim.record.claim_id in fake.find_calls  # keyed recovery tried


def test_cancel_whose_second_check_stop_fails_keeps_preview_and_asks_to_retry(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.providers import open_readback

    ctx = _admit("cancelretry")
    preview = _prepare(uploads, ctx, short_clip)

    def broken(job_id):
        raise RuntimeError("injected mark/drop failure")

    monkeypatch.setattr(open_readback, "cancel", broken)
    with _overrides(uploads=uploads, lifecycle=LifecycleRepositoryImpl(lane_db),
                    meta_client=FakeMetaFiles(clock)):
        retry = route_client.post(f"/preview/{ctx.job_id}/cancel")
    assert retry.status_code == 503
    assert "did not go through" in retry.text and "try again" in retry.text
    assert "Cancelled." not in retry.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state != "cancelling" and record.cancel_requested_at is None
    assert preview.local_video.is_file()


def test_cancel_recovers_a_claimed_file_by_key(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("claimkey")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    fake = FakeMetaFiles(clock)
    # The upload reached Meta but the id never attached locally.
    found = MetaUpload(
        MetaFileId("meta-claimed"), clock() + timedelta(hours=23)
    )
    fake.files[found.file_id] = found
    fake.by_key[claim.record.claim_id] = found
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        done = route_client.post(f"/preview/{ctx.job_id}/cancel")
        assert done.status_code == 200
        assert "Your files were deleted." in done.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "cancelled"
    assert record.meta_file_id == found.file_id


def test_confirm_on_waiting_job_renders_a_waiting_state(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.uploads import CONFIRM_WAITING_COPY

    ctx = _admit("waiting")
    _prepare(uploads, ctx, short_clip)
    # Unconfirmed previews no longer hold slots: fill them with real admitted work.
    _admit("running-a")
    _admit("running-b")
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        waiting = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert waiting.status_code == 409
        assert CONFIRM_WAITING_COPY in waiting.text
        assert "Confirm and send" in waiting.text  # retryable, not hidden
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "preview"
    assert record.hidden is False


def test_late_confirm_hides_and_finalizes_atomically(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("lateclaim")
    _prepare(uploads, ctx, short_clip)
    clock.advance(hours=25)  # past the preview expiry and job deadline
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        late = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert late.status_code == 410
        refreshed = route_client.get(f"/preview/{ctx.job_id}")
        assert refreshed.status_code == 410
        assert "Confirm and send" not in refreshed.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "failed"
    assert record.hidden is True


def test_expired_preview_is_hidden_on_read_without_a_sweep(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("readexpiry")
    _prepare(uploads, ctx, short_clip)
    clock.advance(hours=25)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        shown = route_client.get(f"/preview/{ctx.job_id}")
        assert shown.status_code == 410
        assert "Confirm and send" not in shown.text
        video = route_client.get(f"/preview/{ctx.job_id}/video")
        assert video.status_code == 410
        assert video.content != short_clip.read_bytes()
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "failed"
    assert record.hidden is True


@pytest.mark.parametrize("state", ["preview", "confirming", "confirmed"])
def test_preview_routes_expire_at_exact_ttl(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, state: str,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("exact-ttl")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    if state == "confirming":
        assert uploads.claim_confirmation(ctx.job_id, clock()).acquired
    elif state == "confirmed":
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    before = uploads.load_job(ctx.job_id)[1]
    clock.now = before.preview_expires_at
    with _overrides(uploads=uploads, lifecycle=LifecycleRepositoryImpl(lane_db),
                    meta_client=fake):
        page = route_client.get(f"/preview/{ctx.job_id}")
        assert page.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in page.text
        assert route_client.get(f"/preview/{ctx.job_id}/video").status_code == 410
        assert route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        ).status_code == 410
    after = uploads.load_job(ctx.job_id)[1]
    assert after.hidden
    assert after.state == ("failed" if state == "preview" else state)
    assert after.claim_id == before.claim_id
    assert after.meta_file_id == before.meta_file_id
    assert after.failure_count == 0
    if before.meta_file_id is not None:
        assert _meta_row(lane_db, before.meta_file_id)["delete_state"] == "live"
    assert fake.delete_calls == []


def test_confirmed_preview_expiry_hides_locally_without_touching_meta(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.deletion import cancel_and_delete
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("confirmexpiry")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert uploads.load_job(ctx.job_id)[1].state == "confirmed"
    clock.advance(hours=25)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        shown = route_client.get(f"/preview/{ctx.job_id}")
        assert shown.status_code == 410
        assert "Confirm and send" not in shown.text
        video = route_client.get(f"/preview/{ctx.job_id}/video")
        assert video.status_code == 410
        reconfirm = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert reconfirm.status_code == 410
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirmed"
    assert record.meta_file_id == file_id
    assert record.failure_count == 0
    assert uploads.public_uploads_blocked() is False
    assert _meta_row(lane_db, file_id)["delete_state"] == "live"
    # The Meta lifecycle is uncorrupted: cancel still deletes by the id.
    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert result.meta_outcome == "verified_absent"
    assert result.recorded_state == "cancelled"


def test_confirming_preview_expiry_hides_but_keeps_the_claim(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("confirmingexpiry")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    assert claim.acquired is True
    clock.advance(hours=25)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        shown = route_client.get(f"/preview/{ctx.job_id}")
        assert shown.status_code == 410
        video = route_client.get(f"/preview/{ctx.job_id}/video")
        assert video.status_code == 410
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirming"
    assert record.claim_id == claim.record.claim_id
    assert record.meta_file_id is None
    recoverable = uploads.list_recoverable(clock())
    assert [row.preview.job_id for row in recoverable] == [ctx.job_id]


def test_expired_preview_read_fails_closed_when_persistence_fails(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    from app.ledger import LedgerError, LifecycleRepositoryImpl

    confirmed_ctx = _admit("failclosed1")
    _prepare(uploads, confirmed_ctx, short_clip)
    fake = FakeMetaFiles(clock)
    confirm_and_send(confirmed_ctx.job_id, uploads=uploads, client=fake)
    preview_ctx = _admit("failclosed2")
    _prepare(uploads, preview_ctx, short_clip)
    clock.advance(hours=25)

    def _boom(*args, **kwargs):
        raise LedgerError("disk busy")

    monkeypatch.setattr(uploads, "hide_expired_preview", _boom)
    monkeypatch.setattr(uploads, "recover_job", _boom)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        for job_id in (confirmed_ctx.job_id, preview_ctx.job_id):
            shown = route_client.get(f"/preview/{job_id}")
            assert shown.status_code == 410
            video = route_client.get(f"/preview/{job_id}/video")
            assert video.status_code == 410
    # Both rows are still durably visible: the 410s came from fail-closed.
    assert uploads.load_job(confirmed_ctx.job_id)[1].hidden is False
    assert uploads.load_job(preview_ctx.job_id)[1].hidden is False


def test_confirm_contention_renders_retryable_and_keeps_preview(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    from app.ledger import ContentionError, LifecycleRepositoryImpl
    from app.routes.uploads import CONFIRM_RETRY_COPY

    ctx = _admit("confirmrace")
    _prepare(uploads, ctx, short_clip)

    def _locked(job_id, consented_at):
        raise ContentionError("claim_confirmation lost a write race; retry")

    monkeypatch.setattr(uploads, "claim_confirmation", _locked)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        conflicted = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert conflicted.status_code == 503
        assert CONFIRM_RETRY_COPY in conflicted.text
        assert "Confirm and send" in conflicted.text
        refreshed = route_client.get(f"/preview/{ctx.job_id}")
        assert refreshed.status_code == 200
        assert "Confirm and send" in refreshed.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "preview"
    assert record.hidden is False


def test_confirm_expired_preview_fails_closed_when_persistence_fails(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    # Replaces test_confirm_failed_finalization_renders_retryable_and_keeps_preview,
    # which kept the unsafe response: a 503 consent page carrying the expired
    # video URL. An expired direct POST fails closed instead: 410 with no video.
    from app.ledger import LedgerError, LifecycleRepositoryImpl

    ctx = _admit("finalizefail")
    _prepare(uploads, ctx, short_clip)
    clock.advance(hours=25)

    def _boom(*args, **kwargs):
        raise LedgerError("disk busy")

    saved = (
        uploads.hide_expired_preview,
        uploads.recover_job,
        uploads.record_failure,
    )
    uploads.hide_expired_preview = _boom
    uploads.recover_job = _boom
    uploads.record_failure = _boom
    try:
        lifecycle = LifecycleRepositoryImpl(lane_db)
        with _overrides(uploads=uploads, lifecycle=lifecycle,
                        meta_client=FakeMetaFiles(clock)):
            failed = route_client.post(
                f"/preview/{ctx.job_id}/confirm",
                data={"consent": "on", "ownership": "on"},
            )
            assert failed.status_code == 410
            assert "Confirm and send" not in failed.text
            assert f"/preview/{ctx.job_id}/video" not in failed.text
    finally:
        (
            uploads.hide_expired_preview,
            uploads.recover_job,
            uploads.record_failure,
        ) = saved
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "preview"
    assert record.hidden is False
    # A retry finalizes honestly once the writes work.
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        late = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert late.status_code == 410
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "failed"
    assert record.hidden is True


def test_direct_confirm_post_on_expired_confirmed_preview_hides_without_a_get(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    # No prior GET: without the expiry-aware loader this re-confirm renders
    # 200 with the expired video URL and leaves the row visible.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("directexpired")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert uploads.load_job(ctx.job_id)[1].state == "confirmed"
    clock.advance(hours=25)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        reconfirm = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert reconfirm.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in reconfirm.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirmed"
    assert record.meta_file_id == file_id
    assert record.failure_count == 0
    assert uploads.public_uploads_blocked() is False
    assert _meta_row(lane_db, file_id)["delete_state"] == "live"


def test_direct_confirm_post_on_expired_confirming_preview_hides_claim_intact(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    # No prior GET: without the expiry-aware loader this renders a retryable
    # page carrying the expired video URL instead of hiding.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("directconfirming")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    assert claim.acquired is True
    clock.advance(hours=25)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        posted = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert posted.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in posted.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirming"
    assert record.claim_id == claim.record.claim_id
    assert record.meta_file_id is None


def test_confirm_failure_response_hides_expired_preview_before_render(
    uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path, monkeypatch,
) -> None:
    # The failure-path reload is expiry-aware too: an expired preview whose
    # hide writes fail renders 410 with no video, never a retryable consent.
    from app.ledger import LedgerError
    from app.routes.uploads import _confirm_failure_response

    ctx = _admit("failureexpiry")
    _prepare(uploads, ctx, short_clip)
    clock.advance(hours=25)
    loaded = uploads.load_job(ctx.job_id)
    assert loaded is not None

    def _boom(*args, **kwargs):
        raise LedgerError("disk busy")

    monkeypatch.setattr(uploads, "hide_expired_preview", _boom)
    monkeypatch.setattr(uploads, "recover_job", _boom)
    response = _confirm_failure_response(
        uploads, str(ctx.job_id), loaded[1], max_duration_s=MAX_DURATION_S
    )
    assert response.status_code == 410
    assert f"/preview/{ctx.job_id}/video" not in response.body.decode("utf-8")


def test_confirm_failed_write_leaves_visible_claim_retryable(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    from app.ledger import LedgerError, LifecycleRepositoryImpl
    from app.routes.uploads import CONFIRM_RETRY_COPY

    ctx = _admit("confirmingretry")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)

    def _down(path, expires_after_hours, idempotency_key):
        raise MetaFilesError("meta unreachable")

    fake.on_upload = _down

    def _boom(*args, **kwargs):
        raise LedgerError("disk busy")

    monkeypatch.setattr(uploads, "record_failure", _boom)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        conflicted = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert conflicted.status_code == 409
        assert CONFIRM_RETRY_COPY in conflicted.text
        assert "Confirm and send" in conflicted.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "confirming"
    assert record.hidden is False
    assert record.claim_id is not None


def test_invalid_anon_cookie_is_replaced_never_persisted(
    route_client, lane_db: Path, uploads: UploadStoreImpl,
    short_clip: Path,
) -> None:
    import re

    from app.ledger import LifecycleRepositoryImpl

    lifecycle = LifecycleRepositoryImpl(lane_db)
    route_client.cookies.set("pf_anon", "jane@example.com")
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles()):
        posted = route_client.post(
            "/upload",
            files={"clip": ("clip.mp4", short_clip.read_bytes(), "video/mp4")},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        assert "jane@example.com" not in (posted.headers.get("set-cookie") or "")
    conn = sqlite3.connect(lane_db)
    try:
        anon_ids = [row[0] for row in conn.execute("SELECT anon_id FROM jobs")]
    finally:
        conn.close()
    assert len(anon_ids) == 1
    assert re.fullmatch(r"[0-9a-f]{16}", anon_ids[0])
    assert anon_ids[0] != "jane@example.com"
    # A server-shaped cookie is kept as-is with no replacement set.
    route_client.cookies.set("pf_anon", anon_ids[0])
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles()):
        kept = route_client.post(
            "/upload",
            files={"clip": ("clip.mp4", short_clip.read_bytes(), "video/mp4")},
            follow_redirects=False,
        )
        assert kept.status_code == 303
        assert "set-cookie" not in kept.headers


def test_delete_request_route_throttles_and_dedupes(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
) -> None:
    from app.ledger import LifecycleRepositoryImpl
    from app.meta_files import DELETE_REQUEST_THROTTLED_COPY

    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle):
        ids = []
        for index in range(5):
            reply = route_client.post(
                "/delete-request",
                data={"job_id": "", "contact": f"user-{index}@example.com"},
            )
            assert reply.status_code == 200
            ids.append(reply.text)
        sixth = route_client.post(
            "/delete-request",
            data={"job_id": "", "contact": "user-5@example.com"},
        )
        assert sixth.status_code == 429
        assert DELETE_REQUEST_THROTTLED_COPY in sixth.text
        # An idempotent replay returns the original id, not a refusal.
        replay = route_client.post(
            "/delete-request",
            data={"job_id": "", "contact": "user-0@example.com"},
        )
        assert replay.status_code == 200
        assert replay.text == ids[0]
    assert len(uploads.list_delete_requests(100)) == 5


def test_blocking_routes_are_plain_sync_handlers() -> None:
    import inspect

    from app.routes import uploads as routes_mod

    for name in (
        "upload_submit", "preview_confirm", "preview_cancel",
        "delete_request_submit",
    ):
        assert inspect.iscoroutinefunction(getattr(routes_mod, name)) is False


@contextlib.contextmanager
def _expire_on_lookup_call(monkeypatch, clock: FakeClock, *, target_call: int):
    """Jump the clock past preview expiry during the Nth expiry-aware lookup.

    Every preview render reloads through the helper immediately before
    rendering, so expiring on the helper's reload proves the render path is
    wired through it: without the helper the jump never happens and the
    stale page (with video) renders.
    """
    from app.routes import uploads as routes_mod

    real = routes_mod._expire_preview_during_lookup
    calls = {"n": 0}

    def _wrapped(uploads, job_id):
        calls["n"] += 1
        if calls["n"] == target_call:
            loaded = real(uploads, job_id)
            if loaded is not None:
                clock.now = loaded[1].preview_expires_at + timedelta(seconds=1)
            return real(uploads, job_id)
        return real(uploads, job_id)

    monkeypatch.setattr(routes_mod, "_expire_preview_during_lookup", _wrapped)
    yield calls


def _move_clock_to_expiry_minus_1s(uploads, clock: FakeClock, job_id) -> None:
    expires = uploads.load_job(job_id)[1].preview_expires_at
    clock.now = expires - timedelta(seconds=1)


def _shorten_preview_expiry(
    lane_db: Path, job_id, clock: FakeClock, *, seconds_from_now: int
) -> None:
    # Fresh claims need the 240s job deadline, so expiry-crossing tests for
    # the claim-acquiring paths (success, failure) pull the preview expiry
    # inside the deadline instead of waiting out the 24h TTL.
    conn = sqlite3.connect(lane_db)
    try:
        conn.execute(
            "UPDATE uploads SET preview_expires_at = ? WHERE job_id = ?",
            (
                db.format_dt(clock() + timedelta(seconds=seconds_from_now)),
                str(job_id),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_confirm_conflict_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Probe: durable claim at T0, POST at expiry-1s, claim
    # lookup delayed 2s -> the conflict render hides with no video.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("conflictcross")
    _prepare(uploads, ctx, short_clip)
    first = uploads.claim_confirmation(ctx.job_id, clock())
    assert first.acquired is True
    _move_clock_to_expiry_minus_1s(uploads, clock, ctx.job_id)

    real_claim = uploads.claim_confirmation

    def _slow_claim(job_id, consented_at):
        clock.advance(seconds=2)  # the claim lookup waits past expiry
        return real_claim(job_id, consented_at)

    monkeypatch.setattr(uploads, "claim_confirmation", _slow_claim)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        conflicted = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert conflicted.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in conflicted.text
        assert "Confirm and send" not in conflicted.text
        assert route_client.get(f"/preview/{ctx.job_id}").status_code == 410
        assert route_client.get(f"/preview/{ctx.job_id}/video").status_code == 410
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirming"
    assert record.claim_id == first.record.claim_id
    assert record.meta_file_id is None


def test_confirm_success_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    # POST at expiry-1s, Meta upload takes 2s: the confirmation attaches
    # durably but the success render hides with no video.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("successcross")
    _prepare(uploads, ctx, short_clip)
    _shorten_preview_expiry(lane_db, ctx.job_id, clock, seconds_from_now=200)
    clock.advance(seconds=199)  # expiry-1s, inside the 240s job deadline
    fake = FakeMetaFiles(clock)
    real_upload = FakeMetaFiles.upload_file

    def _slow_upload(path, expires_after_hours, idempotency_key):
        clock.advance(seconds=2)
        fake.on_upload = None  # bypass the hook for the real upload below
        try:
            return real_upload(
                fake, path, expires_after_hours=expires_after_hours,
                idempotency_key=idempotency_key,
            )
        finally:
            fake.on_upload = _slow_upload

    fake.on_upload = _slow_upload
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        done = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert done.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in done.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirmed"
    assert record.meta_file_id is not None
    assert record.failure_count == 0
    assert uploads.public_uploads_blocked() is False
    assert _meta_row(lane_db, record.meta_file_id)["delete_state"] == "live"


def test_confirm_consent_required_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Missing checkboxes at expiry-1s, expiry crosses on the render reload.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("consentcross")
    _prepare(uploads, ctx, short_clip)
    _move_clock_to_expiry_minus_1s(uploads, clock, ctx.job_id)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        with _expire_on_lookup_call(monkeypatch, clock, target_call=2):
            missing = route_client.post(
                f"/preview/{ctx.job_id}/confirm", data={},
            )
        assert missing.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in missing.text
    assert uploads.load_job(ctx.job_id)[1].hidden is True


def test_confirm_unavailable_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # No Meta key at expiry-1s, expiry crosses on the render reload: the
    # unavailable page hides instead of carrying the expired video URL.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("unavailcross")
    _prepare(uploads, ctx, short_clip)
    _move_clock_to_expiry_minus_1s(uploads, clock, ctx.job_id)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    # No meta_client override and no key in settings: the 503 branch.
    with _overrides(uploads=uploads, lifecycle=lifecycle):
        with _expire_on_lookup_call(monkeypatch, clock, target_call=2):
            unavailable = route_client.post(
                f"/preview/{ctx.job_id}/confirm",
                data={"consent": "on", "ownership": "on"},
            )
        assert unavailable.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in unavailable.text
    assert uploads.load_job(ctx.job_id)[1].hidden is True


def test_confirm_early_confirmed_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Re-confirm at expiry-1s, expiry crosses on the render reload.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("earlyconfcross")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    _move_clock_to_expiry_minus_1s(uploads, clock, ctx.job_id)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        with _expire_on_lookup_call(monkeypatch, clock, target_call=2):
            reconfirm = route_client.post(
                f"/preview/{ctx.job_id}/confirm",
                data={"consent": "on", "ownership": "on"},
            )
        assert reconfirm.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in reconfirm.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirmed"
    assert record.meta_file_id == file_id
    assert _meta_row(lane_db, file_id)["delete_state"] == "live"


def test_preview_get_consent_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # GET at expiry-1s, expiry crosses on the render reload.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("getconsentcross")
    _prepare(uploads, ctx, short_clip)
    _move_clock_to_expiry_minus_1s(uploads, clock, ctx.job_id)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        with _expire_on_lookup_call(monkeypatch, clock, target_call=2):
            shown = route_client.get(f"/preview/{ctx.job_id}")
        assert shown.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in shown.text
        assert "Confirm and send" not in shown.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "failed"


def test_preview_get_confirmed_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # GET a confirmed preview at expiry-1s, expiry crosses on the reload.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("getconfcross")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    _move_clock_to_expiry_minus_1s(uploads, clock, ctx.job_id)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        with _expire_on_lookup_call(monkeypatch, clock, target_call=2):
            shown = route_client.get(f"/preview/{ctx.job_id}")
        assert shown.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in shown.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirmed"
    assert record.meta_file_id == file_id
    assert _meta_row(lane_db, file_id)["delete_state"] == "live"


def test_confirm_failure_crossing_expiry_hides_without_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Meta down and the failure write down at expiry-1s: the decision sees
    # a live confirming claim (retryable), but expiry crosses on the render
    # reload, so the retry page hides instead of carrying the video URL.
    from app.ledger import LedgerError, LifecycleRepositoryImpl

    ctx = _admit("failurecross")
    _prepare(uploads, ctx, short_clip)
    _shorten_preview_expiry(lane_db, ctx.job_id, clock, seconds_from_now=200)
    clock.advance(seconds=199)  # expiry-1s, inside the 240s job deadline
    fake = FakeMetaFiles(clock)

    def _down(path, expires_after_hours, idempotency_key):
        raise MetaFilesError("meta unreachable")

    fake.on_upload = _down

    def _boom(*args, **kwargs):
        raise LedgerError("disk busy")

    monkeypatch.setattr(uploads, "record_failure", _boom)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        with _expire_on_lookup_call(monkeypatch, clock, target_call=3):
            failed = route_client.post(
                f"/preview/{ctx.job_id}/confirm",
                data={"consent": "on", "ownership": "on"},
            )
        assert failed.status_code == 410
        assert f"/preview/{ctx.job_id}/video" not in failed.text
        assert "Confirm and send" not in failed.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert record.state == "confirming"
    assert record.claim_id is not None


def test_cancel_on_expired_preview_shows_no_video(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path,
) -> None:
    # Cancel outcomes never carry the video URL, even past expiry.
    from app.ledger import LifecycleRepositoryImpl

    ctx = _admit("cancelexpiry")
    _prepare(uploads, ctx, short_clip)
    clock.advance(hours=25)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle,
                    meta_client=FakeMetaFiles(clock)):
        cancelled = route_client.post(f"/preview/{ctx.job_id}/cancel")
        assert cancelled.status_code in (200, 202)
        assert f"/preview/{ctx.job_id}/video" not in cancelled.text
        assert "Confirm and send" not in cancelled.text
    assert uploads.load_job(ctx.job_id)[1].state == "cancelled"


def test_render_helper_unreadable_reload_uses_stale_proof(
    uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path, monkeypatch,
) -> None:
    # When the render reload itself is unreadable, a stale record proving
    # past-expiry fails closed (410, no video) while a stale live record
    # still renders as decided (transient errors stay retryable).
    from app.ledger import LedgerError
    from app.routes.uploads import _render_preview_response

    old_ctx = _admit("staleexpired")
    _prepare(uploads, old_ctx, short_clip)
    clock.now = uploads.load_job(old_ctx.job_id)[1].preview_expires_at
    stale_expired = uploads.load_job(old_ctx.job_id)[1]
    assert stale_expired.hidden is False
    fresh_ctx = _admit("stalelive")
    _prepare(uploads, fresh_ctx, short_clip)
    stale_live = uploads.load_job(fresh_ctx.job_id)[1]

    def _down(job_id):
        raise LedgerError("db down")

    monkeypatch.setattr(uploads, "load_job", _down)
    from app.routes.uploads import CONFIRM_RETRY_COPY, HIDDEN_COPY

    hidden = _render_preview_response(
        uploads, str(old_ctx.job_id), stale_expired, intent="retry",
        max_duration_s=MAX_DURATION_S,
    )
    assert hidden.status_code == 410
    assert HIDDEN_COPY in hidden.body.decode("utf-8")
    assert CONFIRM_RETRY_COPY not in hidden.body.decode("utf-8")
    assert f"/preview/{old_ctx.job_id}/video" not in hidden.body.decode("utf-8")
    retryable = _render_preview_response(
        uploads, str(fresh_ctx.job_id), stale_live, intent="retry",
        max_duration_s=MAX_DURATION_S,
    )
    assert retryable.status_code == 503
    assert CONFIRM_RETRY_COPY in retryable.body.decode("utf-8")
    assert f"/preview/{fresh_ctx.job_id}/video" in retryable.body.decode("utf-8")


def test_confirm_requests_23h_expiry_on_a_real_clock(
    lane_db: Path, short_clip: Path, monkeypatch,
) -> None:
    real_now = datetime.now(timezone.utc)
    monkeypatch.setattr(db, "utcnow", lambda: datetime.now(timezone.utc))
    live_uploads = UploadStoreImpl(lane_db, clock=lambda: datetime.now(timezone.utc))
    ctx = admit_job("anon-live", "ip-live", "video-live")
    with short_clip.open("rb") as handle:
        prepare_preview(ctx, handle, "clip.mp4", uploads=live_uploads)
    fake = FakeMetaFiles()  # no frozen clock: grants from the real clock
    file_id = confirm_and_send(ctx.job_id, uploads=live_uploads, client=fake)
    (_, hours, _) = fake.upload_calls[0]
    assert hours == 23
    granted = fake.files[file_id].expires_at
    # The real-clock grant stays inside the contract's request+24h bound.
    assert granted <= real_now + timedelta(hours=24)
    assert granted > real_now + timedelta(hours=22)
    record = live_uploads.load_job(ctx.job_id)[1]
    assert record.meta_expires_at == granted


# Every caller intent the routes pass to _render_preview_response.
_RENDER_INTENTS = [
    "show",
    "consent_required",
    "conflict",
    "waiting",
    "unavailable",
    "retry",
    "confirmed",
    "cancelled",
    "cancelled_ack",
    "pending",
    "hidden",
]

# Fresh durable states as (label, state, hidden, meta_attached).
_FRESH_STATES = [
    ("preview-visible", "preview", False, False),
    ("confirming-visible", "confirming", False, False),
    ("confirmed-attached", "confirmed", False, True),
    ("confirmed-bare", "confirmed", False, False),
    ("confirmed-hidden", "confirmed", True, True),
    ("confirming-hidden", "confirming", True, False),
    ("preview-hidden", "preview", True, False),
    ("failed-hidden", "failed", True, False),
    ("failed-visible", "failed", False, False),
    ("cancelling-hidden", "cancelling", True, False),
    ("cancelled-hidden", "cancelled", True, False),
]

_CONSENT_RENDERABLE = {
    "preview-visible", "confirming-visible", "confirmed-bare",
}


def _consent_cell_status(intent: str, fresh_state: str) -> int:
    if intent == "retry":
        return 409 if fresh_state == "confirming" else 503
    return {
        "show": 200,
        "consent_required": 400,
        "conflict": 409,
        "waiting": 409,
        "unavailable": 503,
    }.get(intent, 200)


def test_render_helper_derives_page_from_fresh_record(
    uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path, monkeypatch,
) -> None:
    # Table over every (caller outcome x fresh durable state) pair: the
    # copy shown always matches the fresh state. A consent intent picks
    # its variant only on a live record, cancelled_ack picks the
    # acknowledged-only variant only on a cancelled record, and every
    # other mismatch renders the fresh canonical. The decision record is
    # fixed to a live preview throughout, proving the helper ignores it
    # on a readable reload.
    import dataclasses

    from app.routes import uploads as routes_mod
    from app.routes.uploads import (
        CANCEL_PENDING_COPY,
        CANCELLED_ACK_COPY,
        CANCELLED_COPY,
        CONFIRM_RETRY_COPY,
        CONFIRM_UNAVAILABLE_COPY,
        CONSENT_REQUIRED_COPY,
        HIDDEN_COPY,
        _render_preview_response,
    )
    from app.uploads import CONFIRM_CONFLICT_COPY, CONFIRM_WAITING_COPY

    ctx = _admit("tablecell")
    _prepare(uploads, ctx, short_clip)
    job_id = str(ctx.job_id)
    template = uploads.load_job(ctx.job_id)[1]
    assert template.state == "preview" and template.hidden is False
    holder: dict = {"fresh": template}
    monkeypatch.setattr(
        routes_mod, "_expire_preview_during_lookup",
        lambda store, job: (None, holder["fresh"]),
    )
    video_url = f"/preview/{job_id}/video"
    consent_copies = (
        CONSENT_REQUIRED_COPY,
        CONFIRM_CONFLICT_COPY,
        CONFIRM_WAITING_COPY,
        CONFIRM_UNAVAILABLE_COPY,
        CONFIRM_RETRY_COPY,
    )
    expected_consent = {
        "consent_required": CONSENT_REQUIRED_COPY,
        "conflict": CONFIRM_CONFLICT_COPY,
        "waiting": CONFIRM_WAITING_COPY,
        "unavailable": CONFIRM_UNAVAILABLE_COPY,
        "retry": CONFIRM_RETRY_COPY,
    }
    for fresh_label, fresh_state, fresh_hidden, fresh_meta in _FRESH_STATES:
        holder["fresh"] = dataclasses.replace(
            template,
            state=fresh_state,
            hidden=fresh_hidden,
            meta_file_id=MetaFileId("meta-table-0001") if fresh_meta else None,
        )
        for intent in _RENDER_INTENTS:
            cell = f"{intent} x {fresh_label}"
            response = _render_preview_response(
                uploads, job_id, template, intent=intent,
                max_duration_s=MAX_DURATION_S,
            )
            body = response.body.decode("utf-8")
            if fresh_label in _CONSENT_RENDERABLE:
                assert response.status_code == _consent_cell_status(
                    intent, fresh_state), cell
                assert "Confirm and send" in body, cell
                assert video_url in body, cell
                assert "Confirmed." not in body, cell
                assert CANCEL_PENDING_COPY not in body, cell
                assert CANCELLED_COPY not in body, cell
                assert CANCELLED_ACK_COPY not in body, cell
                assert HIDDEN_COPY not in body, cell
                if intent in expected_consent:
                    assert expected_consent[intent] in body, cell
                    for other in consent_copies:
                        if other != expected_consent[intent]:
                            assert other not in body, cell
                else:
                    # show and terminal intents: live canonical, no copy.
                    for copy in consent_copies:
                        assert copy not in body, cell
            elif fresh_label == "confirmed-attached":
                assert response.status_code == 200, cell
                assert "Confirmed." in body, cell
                assert video_url in body, cell
                assert "Confirm and send" not in body, cell
                assert CANCEL_PENDING_COPY not in body, cell
                assert CANCELLED_COPY not in body, cell
                assert CANCELLED_ACK_COPY not in body, cell
                assert HIDDEN_COPY not in body, cell
                for copy in consent_copies:
                    assert copy not in body, cell
            elif fresh_label in ("cancelling-hidden",):
                assert response.status_code == 202, cell
                assert CANCEL_PENDING_COPY in body, cell
                assert CANCELLED_COPY not in body, cell
                assert CANCELLED_ACK_COPY not in body, cell
                assert HIDDEN_COPY not in body, cell
                for copy in consent_copies:
                    assert copy not in body, cell
                assert video_url not in body, cell
                assert "Confirm and send" not in body, cell
            elif fresh_label in ("cancelled-hidden",):
                assert response.status_code == 200, cell
                if intent == "cancelled_ack":
                    assert CANCELLED_ACK_COPY in body, cell
                    assert CANCELLED_COPY not in body, cell
                else:
                    # Fresh cancelled canonical wins, even for a pending
                    # or conflict outcome from a raced operation.
                    assert CANCELLED_COPY in body, cell
                    assert CANCELLED_ACK_COPY not in body, cell
                assert CANCEL_PENDING_COPY not in body, cell
                assert HIDDEN_COPY not in body, cell
                for copy in consent_copies:
                    assert copy not in body, cell
                assert video_url not in body, cell
                assert "Confirm and send" not in body, cell
            else:  # hidden or failed: 410, canonical hidden copy only
                assert response.status_code == 410, cell
                assert HIDDEN_COPY in body, cell
                assert CANCEL_PENDING_COPY not in body, cell
                assert CANCELLED_COPY not in body, cell
                assert CANCELLED_ACK_COPY not in body, cell
                for copy in consent_copies:
                    assert copy not in body, cell
                assert video_url not in body, cell
                assert "Confirm and send" not in body, cell


def test_losing_confirmation_renders_confirmed_when_competitor_wins_first(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Probe: this request loses the claim race, but the
    # competitor completes before the render reload, so the loser renders
    # the confirmed page — never a stale 409 consent form.
    from app.ledger import LifecycleRepositoryImpl
    from app.routes import uploads as routes_mod

    ctx = _admit("racewinner")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    real_confirm = routes_mod.confirm_and_send

    def _lose_after_competitor(job_id, *, uploads, client):
        file_id = real_confirm(job_id, uploads=uploads, client=client)
        assert uploads.load_job(job_id)[1].state == "confirmed"
        raise ConfirmationConflictError("another confirmation is in flight")

    monkeypatch.setattr(routes_mod, "confirm_and_send", _lose_after_competitor)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        lost = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert lost.status_code == 200
        assert "Confirmed." in lost.text
        assert f"/preview/{ctx.job_id}/video" in lost.text
        assert "another confirmation is in flight" not in lost.text
        assert "Confirm and send" not in lost.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "confirmed"
    assert record.meta_file_id is not None
    assert len(fake.upload_calls) == 1
    assert _meta_row(lane_db, record.meta_file_id)["delete_state"] == "live"


def test_cancel_pending_outcome_renders_cancelled_when_sweeper_wins_first(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Probe: cancel reports retry (pending intent) while a
    # concurrent sweeper completes before the final reload. The fresh
    # record is cancelled, so the page shows the cancelled canonical —
    # never "deletion is still running".
    import dataclasses

    from app.ledger import LifecycleRepositoryImpl
    from app.routes import uploads as routes_mod
    from app.routes.uploads import CANCEL_PENDING_COPY, CANCELLED_COPY

    ctx = _admit("sweeperrace")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    assert uploads.load_job(ctx.job_id)[1].state == "confirmed"
    # This cancel's Meta delete needs retry: the file is still present.
    fake.on_get = lambda fid: MetaUpload(
        MetaFileId(fid), clock() + timedelta(hours=23)
    )
    real_lookup = routes_mod._expire_preview_during_lookup

    def _sweeper_won(store, job):
        loaded = real_lookup(store, job)
        assert loaded is not None
        _, rec = loaded
        return loaded[0], dataclasses.replace(
            rec, state="cancelled", hidden=True,
        )

    monkeypatch.setattr(
        routes_mod, "_expire_preview_during_lookup", _sweeper_won
    )
    lifecycle = LifecycleRepositoryImpl(lane_db)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        replied = route_client.post(f"/preview/{ctx.job_id}/cancel")
        assert replied.status_code == 200
        assert CANCELLED_COPY in replied.text
        assert CANCEL_PENDING_COPY not in replied.text
        assert f"/preview/{ctx.job_id}/video" not in replied.text
        assert "Confirm and send" not in replied.text
    # The durable row is still cancelling with retry; only the render
    # saw the sweeper's completed state.
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "cancelling"
    assert record.meta_file_id == file_id


def test_confirm_conflict_renders_cancelled_when_cancellation_wins_first(
    route_client, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock,
    short_clip: Path, monkeypatch,
) -> None:
    # Confirm/cancel race: this confirm loses to a concurrent
    # cancellation that completes before the render reload. The fresh
    # record is cancelled, so the page shows the cancelled canonical —
    # never a stale conflict notice.
    from app.ledger import LifecycleRepositoryImpl
    from app.routes import uploads as routes_mod
    from app.routes.uploads import CANCELLED_COPY
    from app.uploads import CONFIRM_CONFLICT_COPY

    ctx = _admit("confirmcancel")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    lifecycle = LifecycleRepositoryImpl(lane_db)
    real_cancel = routes_mod.cancel_and_delete

    def _lose_to_cancel(job_id, *, uploads, client):
        result = real_cancel(
            job_id, uploads=uploads, lifecycle=lifecycle, client=client
        )
        assert uploads.load_job(job_id)[1].state == "cancelled"
        assert result.meta_outcome in ("none", "verified_absent")
        raise ConfirmationConflictError("another confirmation is in flight")

    monkeypatch.setattr(routes_mod, "confirm_and_send", _lose_to_cancel)
    with _overrides(uploads=uploads, lifecycle=lifecycle, meta_client=fake):
        lost = route_client.post(
            f"/preview/{ctx.job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
        )
        assert lost.status_code == 200
        assert CANCELLED_COPY in lost.text
        assert CONFIRM_CONFLICT_COPY not in lost.text
        assert "another confirmation is in flight" not in lost.text
        assert f"/preview/{ctx.job_id}/video" not in lost.text
        assert "Confirm and send" not in lost.text
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "cancelled"
    assert record.hidden is True
