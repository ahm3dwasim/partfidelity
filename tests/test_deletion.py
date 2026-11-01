"""Upload deletion: cancel both sides, verify honestly, recover per job."""

from __future__ import annotations

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
    MetaFileId,
    MetaFilesClient,
    MetaFilesError,
    MetaReadUnsupported,
    MetaUpload,
)
from app.deletion import cancel_and_delete, request_meta_deletion
from app.ledger import LifecycleRepositoryImpl, register_meta_file
from app.meta_files import (
    ACKNOWLEDGED_ONLY_COPY,
    VERIFIED_ABSENT_COPY,
    MissingMetaClient,
    ProductionMetaFilesClient,
)
from app.uploads import confirm_and_send, prepare_preview
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


@pytest.fixture
def lifecycle(lane_db: Path) -> LifecycleRepositoryImpl:
    return LifecycleRepositoryImpl(lane_db)


@pytest.fixture(autouse=True)
def _use_fake_clock(monkeypatch, clock: FakeClock):
    monkeypatch.setattr(db, "utcnow", clock)


class FakeMetaFiles:
    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.files: dict[str, MetaUpload] = {}
        self.by_key: dict[str, MetaUpload] = {}
        self.upload_calls: list[tuple[Path, int, str]] = []
        self.find_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
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
        self.upload_calls.append((Path(path), expires_after_hours, idempotency_key))
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
assert isinstance(MissingMetaClient(), MetaFilesClient)


class _StubState:
    def __init__(self) -> None:
        self.handler = None


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
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
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    server.state = _StubState()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=10)
    server.server_close()


def _make_clip(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-shortest", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return path


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory) -> Path:
    return _make_clip(tmp_path_factory.mktemp("media") / "short.mp4")


def _admit(tag: str):
    return admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")


def _prepare(uploads, ctx, clip: Path):
    with clip.open("rb") as handle:
        return prepare_preview(ctx, handle, "clip.mp4", uploads=uploads)


def _confirm(uploads, ctx, fake) -> MetaFileId:
    return confirm_and_send(ctx.job_id, uploads=uploads, client=fake)


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


def _artifact_row(lane_db: Path, artifact_id: str) -> dict | None:
    conn = sqlite3.connect(lane_db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def test_cancel_deletes_both_sides(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("cancel")
    preview = _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = _confirm(uploads, ctx, fake)
    assert preview.local_video.is_file()
    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert result.local_deleted is True
    assert result.meta_file_id == file_id
    assert result.meta_outcome == "verified_absent"
    assert result.recorded_state == "cancelled"
    assert result.error is None
    assert not preview.local_video.exists()
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert _artifact_row(lane_db, record.preview_artifact_id)["delete_state"] == (
        "tombstoned"
    )
    row = _meta_row(lane_db, file_id)
    assert row is not None and row["delete_state"] == "verified_absent"
    assert row["verified_absent_at"] is not None


def test_cancel_with_second_check_in_flight_defers_local_files_to_the_sweeper(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path, monkeypatch,
) -> None:
    from app.deletion import sweep_expired
    from app.providers import open_readback

    ctx = _admit("cancel-busy")
    preview = _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    _confirm(uploads, ctx, fake)
    quiet = []  # a second-check call is in flight until this holds True
    monkeypatch.setattr(open_readback, "cancel", lambda job_id: bool(quiet))
    monkeypatch.setattr(open_readback, "wait_quiet", lambda job_id, timeout_s=None: bool(quiet))
    result = cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake)
    assert result.local_deleted is False and result.recorded_state == "cancelling"
    assert preview.local_video.is_file()  # never deleted under a running check
    record = uploads.load_job(ctx.job_id)[1]
    assert record.hidden is True
    assert _artifact_row(lane_db, record.preview_artifact_id)["delete_state"] == "pending_delete"

    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=fake)
    assert preview.local_video.is_file()  # still not quiet: kept for the next sweep
    assert _artifact_row(lane_db, record.preview_artifact_id)["delete_state"] == "retry"

    quiet.append(True)
    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=fake)
    assert not preview.local_video.exists()
    assert _artifact_row(lane_db, record.preview_artifact_id)["delete_state"] == "tombstoned"
    assert uploads.load_job(ctx.job_id)[1].state == "cancelled"


def test_cancel_racing_upload_hides_then_deletes_returned_meta_id(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    from app.deletion import sweep_expired

    ctx = _admit("cancelrace")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    entered = threading.Event()
    proceed = threading.Event()

    def _gated(path, hours, key):
        entered.set()
        assert proceed.wait(timeout=10), "cancel race stalled"
        fake.on_upload = None
        try:
            return fake.upload_file(path, expires_after_hours=hours,
                                    idempotency_key=key)
        finally:
            fake.on_upload = _gated

    fake.on_upload = _gated
    outcome: dict = {}

    def _sender() -> None:
        try:
            outcome["id"] = _confirm(uploads, ctx, fake)
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=_sender, daemon=True)
    thread.start()
    assert entered.wait(timeout=10), "upload never started"
    # Cancel commits hide-first while the upload is in flight.
    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert result.local_deleted is True
    assert uploads.load_job(ctx.job_id)[1].hidden is True
    proceed.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    # The completion attached the returned id already due, then raised
    # instead of handing it over; the confirm path deleted it promptly.
    assert "error" in outcome
    assert len(fake.upload_calls) == 1
    record = uploads.load_job(ctx.job_id)[1]
    assert record.meta_file_id is not None
    assert record.meta_file_id in fake.delete_calls
    row = _meta_row(lane_db, record.meta_file_id)
    assert row is not None and row["delete_state"] == "pending"
    # The pending row keeps it retryable: one sweep verifies and closes.
    swept = sweep_expired(clock(), uploads=uploads, lifecycle=lifecycle, client=fake)
    assert swept.meta_verified == 1
    assert uploads.load_job(ctx.job_id)[1].state == "cancelled"


def test_cancel_race_crash_after_attachment_is_swept(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    from app.deletion import sweep_expired

    ctx = _admit("attachcrash")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    assert claim.acquired is True
    uploads.request_cancellation(ctx.job_id, clock())
    file_id = MetaFileId("meta-race-1")
    # The completion attaches after cancellation committed, already due;
    # the worker then crashes before any direct deletion.
    completed = uploads.complete_confirmation(
        claim.record.claim_id, file_id, clock() + timedelta(hours=24),
        clock(), clock(),
    )
    assert completed.state == "cancelling"
    row = _meta_row(lane_db, file_id)
    assert row is not None and row["delete_state"] == "pending"
    # A fresh sweeper (new instances, same database) finishes the job.
    fake = FakeMetaFiles(clock)
    fake.files[file_id] = MetaUpload(file_id, clock() + timedelta(hours=24))
    fresh_uploads = UploadStoreImpl(lane_db, clock=clock)
    fresh_lifecycle = LifecycleRepositoryImpl(lane_db)
    swept = sweep_expired(clock(), uploads=fresh_uploads,
                          lifecycle=fresh_lifecycle, client=fake)
    assert swept.meta_verified == 1
    assert _meta_row(lane_db, file_id)["delete_state"] == "verified_absent"
    assert fresh_uploads.load_job(ctx.job_id)[1].state == "cancelled"


def test_cancel_crash_after_commit_sweeps_all_existing_meta_rows(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    from app.deletion import sweep_expired

    ctx = _admit("multicrash")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    first = _confirm(uploads, ctx, fake)
    extra = [MetaFileId(f"meta-extra-{index}") for index in range(2)]
    for file_id in extra:
        register_meta_file(ctx.job_id, file_id, clock() + timedelta(hours=24),
                           path=lane_db)
        fake.files[file_id] = MetaUpload(file_id, clock() + timedelta(hours=24))
    # The cancelling worker commits, then crashes before direct deletion.
    uploads.request_cancellation(ctx.job_id, clock())
    conn = sqlite3.connect(lane_db)
    try:
        states = {
            row[0]
            for row in conn.execute(
                "SELECT delete_state FROM meta_files WHERE job_id = ?",
                (str(ctx.job_id),),
            ).fetchall()
        }
    finally:
        conn.close()
    assert states == {"pending"}
    fresh_uploads = UploadStoreImpl(lane_db, clock=clock)
    fresh_lifecycle = LifecycleRepositoryImpl(lane_db)
    swept = sweep_expired(clock(), uploads=fresh_uploads,
                          lifecycle=fresh_lifecycle, client=fake)
    assert swept.meta_checked == 3
    assert swept.meta_verified == 3
    assert {first, *extra} <= set(fake.delete_calls)
    assert fresh_uploads.load_job(ctx.job_id)[1].state == "cancelled"


def test_missing_client_never_records_verified_absent(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("missing")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = _confirm(uploads, ctx, fake)
    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle,
        client=MissingMetaClient(),
    )
    assert result.local_deleted is True
    assert result.meta_outcome == "retry"
    assert result.error is not None
    # The id is preserved for retry, never discarded or verified.
    record = uploads.load_job(ctx.job_id)[1]
    assert record.meta_file_id == file_id
    row = _meta_row(lane_db, file_id)
    assert row is not None and row["delete_state"] == "retry"
    assert record.failure_count >= 1
    assert record.hidden is True
    assert result.recorded_state == "cancelling"


def test_file_id_expiry_and_delete_recorded(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("recorded")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = _confirm(uploads, ctx, fake)
    record = uploads.load_job(ctx.job_id)[1]
    assert record.meta_file_id == file_id
    assert record.meta_expires_at == clock() + timedelta(hours=23)
    row = _meta_row(lane_db, file_id)
    assert row is not None and row["job_id"] == str(ctx.job_id)
    assert row["delete_state"] == "live"
    outcome = request_meta_deletion(file_id, client=fake, lifecycle=lifecycle)
    assert outcome == "verified_absent"
    row = _meta_row(lane_db, file_id)
    assert row is not None and row["delete_state"] == "verified_absent"
    assert row["verified_absent_at"] is not None


def _confirmed_job(uploads, lifecycle, clock, short_clip, tag, *, fake=None):
    ctx = _admit(tag)
    with short_clip.open("rb") as handle:
        prepare_preview(ctx, handle, "clip.mp4", uploads=uploads)
    if fake is None:
        fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    return ctx, fake, file_id


def test_verified_absent_requires_get_and_list_evidence(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path, meta_stub,
) -> None:
    ctx, fake, file_id = _confirmed_job(uploads, lifecycle, clock, short_clip, "ev")
    # GET None after delete: verified, with the absence timestamp.
    assert request_meta_deletion(file_id, client=fake, lifecycle=lifecycle) == (
        "verified_absent"
    )
    assert _meta_row(lane_db, file_id)["delete_state"] == "verified_absent"
    # A provider whose GET 404s but whose list still shows the id is
    # contradictory: retry, never verified. The shared provider mints a
    # distinct second id.
    ctx2, _, file_id2 = _confirmed_job(
        uploads, lifecycle, clock, short_clip, "ev2", fake=fake
    )

    def _contradictory(method, path, query, body):
        if method == "GET" and path == f"/files/{file_id2}":
            return 404, None
        if method == "GET" and path == "/files" and "key" not in query:
            return 200, {"files": [file_id2]}
        if method == "DELETE":
            return 204, None
        return 404, None

    meta_stub.state.handler = _contradictory
    port = meta_stub.server_address[1]
    prod = ProductionMetaFilesClient(
        "test-key", base_url=f"http://127.0.0.1:{port}", timeout_s=5.0
    )
    assert request_meta_deletion(file_id2, client=prod, lifecycle=lifecycle) == "retry"
    assert _meta_row(lane_db, file_id2)["delete_state"] == "retry"


def test_present_or_read_error_beats_unsupported(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path, meta_stub,
) -> None:
    ctx, fake, file_id = _confirmed_job(uploads, lifecycle, clock, short_clip, "prs")
    # Still present after delete: retry, and the id is preserved.
    fake.on_get = lambda file_id: fake.files.get(file_id) or MetaUpload(
        file_id, clock() + timedelta(hours=1)
    )
    assert request_meta_deletion(file_id, client=fake, lifecycle=lifecycle) == "retry"
    assert _meta_row(lane_db, file_id)["delete_state"] == "retry"
    assert uploads.load_job(ctx.job_id)[1].meta_file_id == file_id
    # A read error is retryable too, never acknowledged-only.
    def _boom(file_id):
        raise MetaFilesError("socket reset")

    fake.on_get = _boom
    assert request_meta_deletion(file_id, client=fake, lifecycle=lifecycle) == "retry"
    # Unsupported reads are the only acknowledged-only path.
    def _unsupported(file_id):
        raise MetaReadUnsupported("no reads")

    fake.on_get = _unsupported
    assert request_meta_deletion(file_id, client=fake, lifecycle=lifecycle) == (
        "acknowledged_only"
    )
    # The attempt reports its own unsupported read, but the durable row
    # keeps the earlier retry/presence evidence: an acknowledgment must
    # never erase it, so the next sweep retries instead of closing.
    assert _meta_row(lane_db, file_id)["delete_state"] == "retry"
    # Adapter level: a listed id beats an unsupported GET.
    port = meta_stub.server_address[1]
    prod = ProductionMetaFilesClient(
        "test-key", base_url=f"http://127.0.0.1:{port}", timeout_s=5.0
    )

    def _get_down_list_shows(method, path, query, body):
        if path.startswith("/files/"):
            return 405, None
        if method == "GET" and path == "/files":
            return 200, {"files": ["meta-listed"]}
        return 404, None

    meta_stub.state.handler = _get_down_list_shows
    with pytest.raises(MetaFilesError):
        prod.get_file(MetaFileId("meta-listed"))

    def _both_down(method, path, query, body):
        return 405, None

    meta_stub.state.handler = _both_down
    with pytest.raises(MetaReadUnsupported):
        prod.get_file(MetaFileId("meta-listed"))


def test_delete_not_found_still_checks_get_and_list(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path, meta_stub,
) -> None:
    ctx, fake, file_id = _confirmed_job(uploads, lifecycle, clock, short_clip, "nf")

    def _gone_everywhere(method, path, query, body):
        if method == "GET" and path == "/files" and "key" not in query:
            return 200, {"files": []}
        return 404, None

    meta_stub.state.handler = _gone_everywhere
    port = meta_stub.server_address[1]
    prod = ProductionMetaFilesClient(
        "test-key", base_url=f"http://127.0.0.1:{port}", timeout_s=5.0
    )
    # DELETE already-absent is idempotent; GET+list then prove absence.
    assert prod.delete_file(file_id) is None
    assert request_meta_deletion(file_id, client=prod, lifecycle=lifecycle) == (
        "verified_absent"
    )
    assert _meta_row(lane_db, file_id)["delete_state"] == "verified_absent"


def test_acknowledged_only_copy_is_honest(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx, fake, file_id = _confirmed_job(uploads, lifecycle, clock, short_clip, "ack")

    def _unsupported(file_id):
        raise MetaReadUnsupported("no reads")

    fake.on_get = _unsupported
    assert request_meta_deletion(file_id, client=fake, lifecycle=lifecycle) == (
        "acknowledged_only"
    )
    assert _meta_row(lane_db, file_id)["delete_state"] == "acknowledged_only"
    assert "could not confirm absence" in ACKNOWLEDGED_ONLY_COPY
    assert "expires" in ACKNOWLEDGED_ONLY_COPY
    assert "confirmed absent" not in ACKNOWLEDGED_ONLY_COPY
    assert "physical erasure" not in ACKNOWLEDGED_ONLY_COPY
    assert "not physical erasure" in VERIFIED_ABSENT_COPY


def test_failure_hides_output_and_both_workers_block_public_upload(
    lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock, short_clip: Path
) -> None:
    ctx = _admit("hide")
    _prepare(uploads, ctx, short_clip)
    assert uploads.public_uploads_blocked() is False
    uploads.record_failure(ctx.job_id, "hide-1", "meta_upload", "boom", clock())
    assert uploads.load_job(ctx.job_id)[1].hidden is True
    uploads.record_failure(ctx.job_id, "hide-2", "meta_upload", "boom", clock())
    uploads.record_failure(ctx.job_id, "hide-3", "meta_upload", "boom", clock())
    worker_b = UploadStoreImpl(lane_db, clock=clock)
    assert uploads.public_uploads_blocked() is True
    assert worker_b.public_uploads_blocked() is True
    # Both workers recover together once the responsible job resolves.
    uploads.recover_job(ctx.job_id, clock())
    assert uploads.public_uploads_blocked() is False
    assert worker_b.public_uploads_blocked() is False
    assert uploads.load_job(ctx.job_id)[1].failure_count == 0


def test_successful_retry_recovers_only_its_job(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx_a = _admit("retry-a")
    _prepare(uploads, ctx_a, short_clip)
    ctx_b = _admit("retry-b")
    _prepare(uploads, ctx_b, short_clip)
    fake = FakeMetaFiles(clock)
    id_a = _confirm(uploads, ctx_a, fake)
    id_b = _confirm(uploads, ctx_b, fake)
    uploads.record_failure(ctx_a.job_id, "retry-a-1", "meta_delete",
                           "delete failed once", clock(), meta_file_id=id_a)
    uploads.record_failure(ctx_b.job_id, "retry-b-1", "meta_delete",
                           "delete failed once", clock(), meta_file_id=id_b)
    assert request_meta_deletion(id_a, client=fake, lifecycle=lifecycle) == (
        "verified_absent"
    )
    uploads.recover_job(ctx_a.job_id, clock())
    assert uploads.load_job(ctx_a.job_id)[1].failure_count == 0
    assert uploads.load_job(ctx_b.job_id)[1].failure_count == 1
    assert uploads.load_job(ctx_b.job_id)[1].hidden is True


def _stub_prod(meta_stub) -> ProductionMetaFilesClient:
    port = meta_stub.server_address[1]
    return ProductionMetaFilesClient(
        "test-key", base_url=f"http://127.0.0.1:{port}", timeout_s=5.0
    )


def test_one_sided_absence_proof_is_refused(meta_stub) -> None:
    prod = _stub_prod(meta_stub)
    # GET 404 plus an unsupported list is one-sided: acknowledged-only,
    # never verified.
    def _get_gone_list_down(method, path, query, body):
        if method == "GET" and path == "/files":
            return 405, None
        return 404, None

    meta_stub.state.handler = _get_gone_list_down
    with pytest.raises(MetaReadUnsupported):
        prod.get_file(MetaFileId("meta-one-sided"))
    # An unsupported GET plus an empty list is one-sided too.
    def _get_down_list_empty(method, path, query, body):
        if path.startswith("/files/"):
            return 405, None
        if method == "GET" and path == "/files":
            return 200, {"files": []}
        return 404, None

    meta_stub.state.handler = _get_down_list_empty
    with pytest.raises(MetaReadUnsupported):
        prod.get_file(MetaFileId("meta-one-sided"))
    # Present evidence still beats the unsupported read: retry.
    def _get_down_list_shows(method, path, query, body):
        if path.startswith("/files/"):
            return 405, None
        if method == "GET" and path == "/files":
            return 200, {"files": ["meta-present"]}
        return 404, None

    meta_stub.state.handler = _get_down_list_shows
    with pytest.raises(MetaFilesError):
        prod.get_file(MetaFileId("meta-present"))


def test_list_pagination_is_exhausted_before_absence(meta_stub) -> None:
    prod = _stub_prod(meta_stub)
    seen: list[dict] = []

    def _paged(method, path, query, body):
        if path.startswith("/files/"):
            return 404, None
        seen.append(dict(query))
        if path == "/files" and "cursor" not in query:
            return 200, {"files": [], "next": "/files?cursor=p2"}
        if query.get("cursor") == ["p2"]:
            return 200, {"files": ["meta-target"]}
        return 404, None

    meta_stub.state.handler = _paged
    # The id sits on page two: the empty first page proves nothing.
    with pytest.raises(MetaFilesError):
        prod.get_file(MetaFileId("meta-target"))
    assert seen == [{}, {"cursor": ["p2"]}]

    seen.clear()

    def _clean_pages(method, path, query, body):
        if path.startswith("/files/"):
            return 404, None
        seen.append(dict(query))
        if path == "/files" and "cursor" not in query:
            return 200, {"files": [], "next_cursor": "p2"}
        if query.get("cursor") == ["p2"]:
            return 200, {"files": []}
        return 404, None

    meta_stub.state.handler = _clean_pages
    assert prod.get_file(MetaFileId("meta-gone")) is None
    assert seen == [{}, {"cursor": ["p2"]}]


def test_weaker_outcomes_never_replace_retry_evidence(
    lane_db: Path, lifecycle: LifecycleRepositoryImpl, clock: FakeClock,
) -> None:
    ctx = _admit("cas")
    register_meta_file(
        ctx.job_id, MetaFileId("meta-cas"), clock() + timedelta(hours=24),
        path=lane_db,
    )
    t0 = clock()
    lifecycle.record_meta_file_deletion(
        MetaFileId("meta-cas"), "retry", "present after delete",
        attempt_started_at=t0,
    )
    assert _meta_row(lane_db, "meta-cas")["delete_state"] == "retry"
    clock.advance(hours=1)
    t1 = clock()
    lifecycle.record_meta_file_deletion(
        MetaFileId("meta-cas"), "retry", "still present",
        attempt_started_at=t1,
    )
    assert _meta_row(lane_db, "meta-cas")["delete_state"] == "retry"
    # A stale acknowledged-only (started before the newer retry) is
    # refused: the retry/present evidence stands.
    lifecycle.record_meta_file_deletion(
        MetaFileId("meta-cas"), "acknowledged_only",
        attempt_started_at=t0,
    )
    assert _meta_row(lane_db, "meta-cas")["delete_state"] == "retry"
    # A fresh acknowledged-only is likewise refused: only verified
    # two-sided absence clears retry/presence evidence, so a later
    # unsupported read can never close cancellation over a live file.
    lifecycle.record_meta_file_deletion(
        MetaFileId("meta-cas"), "acknowledged_only",
        attempt_started_at=t1,
    )
    assert _meta_row(lane_db, "meta-cas")["delete_state"] == "retry"
    # Verified is the one outcome that clears retry, and it is final.
    lifecycle.record_meta_file_deletion(MetaFileId("meta-cas"), "verified_absent")
    assert _meta_row(lane_db, "meta-cas")["delete_state"] == "verified_absent"
    lifecycle.record_meta_file_deletion(MetaFileId("meta-cas"), "acknowledged_only")
    lifecycle.record_meta_file_deletion(MetaFileId("meta-cas"), "retry", "late")
    assert _meta_row(lane_db, "meta-cas")["delete_state"] == "verified_absent"
    # A first unsupported read still records acknowledged-only, and a
    # terminal outcome never moves back to retry.
    register_meta_file(
        ctx.job_id, MetaFileId("meta-ack"), clock() + timedelta(hours=24),
        path=lane_db,
    )
    lifecycle.record_meta_file_deletion(
        MetaFileId("meta-ack"), "acknowledged_only",
        attempt_started_at=t1,
    )
    assert _meta_row(lane_db, "meta-ack")["delete_state"] == "acknowledged_only"
    lifecycle.record_meta_file_deletion(MetaFileId("meta-ack"), "retry", "late")
    assert _meta_row(lane_db, "meta-ack")["delete_state"] == "acknowledged_only"
    lifecycle.record_meta_file_deletion(MetaFileId("meta-ack"), "verified_absent")
    assert _meta_row(lane_db, "meta-ack")["delete_state"] == "verified_absent"
    # The attempt clock is strict: naive or wrong-typed values refuse.
    with pytest.raises(ValueError, match="aware datetime"):
        lifecycle.record_meta_file_deletion(
            MetaFileId("meta-cas"), "retry", attempt_started_at=t1.replace(tzinfo=None)
        )
    with pytest.raises(ValueError, match="must be a datetime"):
        lifecycle.record_meta_file_deletion(
            MetaFileId("meta-cas"), "retry", attempt_started_at="not-a-datetime"
        )


def test_failed_delete_attempt_stamps_a_version_for_later_cas(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock, short_clip: Path,
) -> None:
    ctx, fake, file_id = _confirmed_job(
        uploads, lifecycle, clock, short_clip, "stamp"
    )
    t0 = clock()

    def _down(file_id):
        raise MetaFilesError("meta unreachable")

    fake.on_delete = _down
    assert request_meta_deletion(file_id, client=fake, lifecycle=lifecycle) == (
        "retry"
    )
    row = _meta_row(lane_db, file_id)
    assert row["delete_state"] == "retry"
    assert db.parse_dt(row["delete_requested_at"]) == t0
    # The stamped retry now refuses an attempt older than itself.
    clock.advance(hours=1)
    lifecycle.record_meta_file_deletion(
        file_id, "acknowledged_only",
        attempt_started_at=t0 - timedelta(seconds=1),
    )
    assert _meta_row(lane_db, file_id)["delete_state"] == "retry"
