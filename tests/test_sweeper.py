"""Upload sweeper: four independent queues plus the delete-request queue."""

from __future__ import annotations

import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import db
from app.admission import admit_job, job_dir
from app.contracts import MetaFileId, MetaFilesClient, MetaFilesError, MetaUpload
from app.deletion import sweep_expired
from app.ledger import LifecycleRepositoryImpl
from app.pipeline import guide_path
from app.uploads import confirm_and_send, prepare_preview
from app.uploads_store import UploadError, UploadStoreImpl


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


def _artifact_rows(lane_db: Path, job_id: str) -> list[dict]:
    conn = sqlite3.connect(lane_db)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM artifacts WHERE job_id = ?", (str(job_id),)
            ).fetchall()
        ]
    finally:
        conn.close()


def test_sweeper_finds_orphan_by_claim_key_after_crash_before_attach(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("orphan")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    assert claim.acquired is True
    claim_id = claim.record.claim_id
    # The provider created the file; the worker crashed before attaching.
    orphan = MetaUpload(MetaFileId("meta-orphan"), clock() + timedelta(hours=24))
    fake = FakeMetaFiles(clock)
    fake.files[orphan.file_id] = orphan
    fake.by_key[claim_id] = orphan
    clock.advance(hours=1)
    # A fresh sweeper restarts the world and runs one pass.
    swept = sweep_expired(
        clock(),
        uploads=UploadStoreImpl(lane_db, clock=clock),
        lifecycle=LifecycleRepositoryImpl(lane_db),
        client=fake,
    )
    assert claim_id in fake.find_calls
    assert swept.recoverable_recovered == 1
    assert swept.meta_verified == 1
    record = uploads.load_job(ctx.job_id)[1]
    assert record.meta_file_id == orphan.file_id
    assert record.failure_count == 0  # per-job recovery resolved the row
    assert record.state == "cancelled"
    assert _meta_row(lane_db, orphan.file_id)["delete_state"] == "verified_absent"


def test_sweeper_queries_failed_claim_after_upload_created_then_raised(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    from app.uploads import ConfirmationFailedError

    ctx = _admit("createdraise")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)

    def _created_then_raised(path, hours, key):
        created = MetaUpload(MetaFileId("meta-created"),
                             clock() + timedelta(hours=hours))
        fake.files[created.file_id] = created
        fake.by_key[key] = created
        raise MetaFilesError("socket died after create")

    fake.on_upload = _created_then_raised
    with pytest.raises(ConfirmationFailedError):
        confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "failed"
    assert record.claim_id is not None
    assert record.meta_file_id is None
    clock.advance(hours=1)
    swept = sweep_expired(
        clock(),
        uploads=UploadStoreImpl(lane_db, clock=clock),
        lifecycle=LifecycleRepositoryImpl(lane_db),
        client=fake,
    )
    assert record.claim_id in fake.find_calls
    assert swept.recoverable_recovered == 1
    assert swept.meta_verified == 1
    after = uploads.load_job(ctx.job_id)[1]
    assert after.meta_file_id == "meta-created"
    assert after.failure_count == 0
    assert _meta_row(lane_db, "meta-created")["delete_state"] == "verified_absent"


def test_sweeper_queues_found_id_with_expiry_after_guard(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("afterguard")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    claim_id = claim.record.claim_id
    guard = claim.record.guard_deadline_utc
    assert guard is not None
    # The provider answers with an expiry past the persisted guard: the
    # id is still queued, but under an immediate conservative deadline.
    found = MetaUpload(MetaFileId("meta-wild"), guard + timedelta(hours=1))
    fake = FakeMetaFiles(clock)
    fake.files[found.file_id] = found
    fake.by_key[claim_id] = found
    clock.advance(hours=1)
    swept = sweep_expired(
        clock(),
        uploads=UploadStoreImpl(lane_db, clock=clock),
        lifecycle=LifecycleRepositoryImpl(lane_db),
        client=fake,
    )
    assert swept.recoverable_recovered == 1
    assert swept.meta_verified == 1
    row = _meta_row(lane_db, found.file_id)
    assert row is not None
    assert db.parse_dt(row["expires_at"]) == clock()
    assert uploads.load_job(ctx.job_id)[1].meta_file_id == found.file_id


def test_no_match_before_guard_deadline_stays_queued(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("noguard")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    claim_id = claim.record.claim_id
    fake = FakeMetaFiles(clock)  # provider holds nothing for the key
    clock.advance(hours=1)
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert claim_id in fake.find_calls
    assert swept.recoverable_checked == 1
    assert swept.recoverable_recovered == 0
    assert swept.recoverable_retry == 1
    record = uploads.load_job(ctx.job_id)[1]
    assert record.claim_id == claim_id
    assert record.state == "confirming"
    assert [r.preview.job_id for r in uploads.list_recoverable(clock())] == [
        ctx.job_id
    ]


def test_no_match_after_guard_deadline_recovers_claim(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("pastguard")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    claim_id = claim.record.claim_id
    guard = claim.record.guard_deadline_utc
    fake = FakeMetaFiles(clock)
    clock.advance(hours=26)  # past the persisted 25-hour guard
    assert clock() > guard
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert claim_id in fake.find_calls
    assert swept.recoverable_recovered == 1
    record = uploads.load_job(ctx.job_id)[1]
    assert record.claim_id is None
    assert record.state == "failed"
    assert uploads.list_recoverable(clock()) == ()


def test_abandoned_unconfirmed_preview_expires_and_footage_is_deleted(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path, tmp_path: Path,
) -> None:
    ctx = _admit("abandoned")
    preview = _prepare(uploads, ctx, short_clip)
    assert preview.local_video.is_file()
    fake = FakeMetaFiles(clock)
    clock.advance(hours=25)  # past the 24-hour preview expiry
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    # The expiry phase runs first: the idle admitted job is overdue
    # (240 s deadline), so the ledger shows expired while the upload
    # itself still terminalizes to failed+hidden below.
    assert swept.expired_jobs == 1
    assert swept.recoverable_recovered == 1
    assert swept.artifacts_tombstoned == 1
    assert fake.find_calls == []  # claimless rows make no Meta call
    record = uploads.load_job(ctx.job_id)[1]
    assert record.state == "failed"
    assert record.hidden is True
    assert not preview.local_video.exists()
    # Guide-less previews use this existing 24-hour upload/job expiry path.
    assert not guide_path(ctx.job_id).exists()
    rows = _artifact_rows(lane_db, ctx.job_id)
    assert len(rows) == 1 and rows[0]["delete_state"] == "tombstoned"
    conn = sqlite3.connect(lane_db)
    try:
        state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert state == "expired"


def _due_provisional(ctx, name: str, *, make_file: bool = True) -> Path:
    directory = job_dir(str(ctx.job_id))
    target = directory / name
    if make_file:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"frame-bytes")
    ctx.artifacts.register("provisional", target)
    return target


def test_sweeper_queues_and_rows_fail_independently(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path, monkeypatch,
) -> None:
    from app.ledger import register_meta_file

    # Queue 2: one good file plus a '.' row that must be refused.
    ctx_a = _admit("iso-a")
    good = _due_provisional(ctx_a, "good.bin")
    sentinel = job_dir(str(ctx_a.job_id)) / "sentinel.bin"
    sentinel.write_bytes(b"do-not-touch")
    ctx_a.artifacts.register("provisional", job_dir(str(ctx_a.job_id)))
    ctx_a.finish("failed")
    # Queue 3: one good id plus one whose recorder blows up.
    ctx_b = _admit("iso-b")
    _prepare(uploads, ctx_b, short_clip)
    fake = FakeMetaFiles(clock)
    good_id = confirm_and_send(ctx_b.job_id, uploads=uploads, client=fake)
    bad_id = MetaFileId("meta-bad-record")
    register_meta_file(ctx_b.job_id, bad_id, clock() + timedelta(hours=24),
                       path=lane_db)
    fake.files[bad_id] = MetaUpload(bad_id, clock() + timedelta(hours=24))
    uploads.request_cancellation(ctx_b.job_id, clock())
    uploads.submit_delete_request(ctx_b.job_id, "op@example.com", clock())
    real_record = lifecycle.record_meta_file_deletion

    def _flaky(file_id, outcome, error=None, **kwargs):
        if str(file_id) == str(bad_id):
            raise RuntimeError("ledger write lost")
        return real_record(file_id, outcome, error, **kwargs)

    monkeypatch.setattr(lifecycle, "record_meta_file_deletion", _flaky)

    def _list_boom(now):
        raise RuntimeError("recoverable read lost")

    monkeypatch.setattr(uploads, "list_recoverable", _list_boom)
    clock.advance(days=31)  # the delete request also expires
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    # Queue 1 fetch failed, but queues 2-4 still ran.
    assert swept.recoverable_checked == 0
    assert "recoverable-list" in swept.failures
    assert swept.delete_requests_purged == 1
    assert swept.public_upload_blocked is False
    # Queue 2: good file and expired preview footage tombstoned, '.' refused.
    assert swept.artifacts_tombstoned == 2
    assert swept.artifacts_retry == 1
    assert not good.exists()
    assert sentinel.is_file()  # never a recursive delete
    assert job_dir(str(ctx_a.job_id)).is_dir()
    # Queue 3: good id verified, bad row kept pending for retry.
    assert swept.meta_verified == 1
    assert swept.meta_retry == 1
    assert str(bad_id) in swept.failures
    assert _meta_row(lane_db, good_id)["delete_state"] == "verified_absent"
    assert _meta_row(lane_db, bad_id)["delete_state"] == "pending"


def test_sweeper_skips_active_job_records(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("active")
    preview = _prepare(uploads, ctx, short_clip)
    live = _due_provisional(ctx, "live.bin")
    fake = FakeMetaFiles(clock)
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert swept.recoverable_checked == 0
    assert swept.artifacts_checked == 0
    assert swept.meta_checked == 0
    assert swept.failures == ()
    assert preview.local_video.is_file()
    assert live.is_file()
    assert fake.find_calls == [] and fake.delete_calls == []


def test_sweeper_refuses_dot_or_directory_artifact(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock,
) -> None:
    ctx = _admit("dotdir")
    directory = job_dir(str(ctx.job_id))
    subdir = directory / "frames"
    subdir.mkdir(parents=True, exist_ok=True)
    inner = subdir / "inner.bin"
    inner.write_bytes(b"not-yet-due")
    ctx.artifacts.register("provisional", directory)
    ctx.artifacts.register("provisional", subdir)
    ctx.finish("failed")
    fake = FakeMetaFiles(clock)
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert swept.artifacts_checked == 2
    assert swept.artifacts_tombstoned == 0
    assert swept.artifacts_retry == 2
    # Nothing removed, nothing recursed into.
    assert directory.is_dir()
    assert subdir.is_dir()
    assert inner.is_file()
    rows = _artifact_rows(lane_db, ctx.job_id)
    assert {row["delete_state"] for row in rows} == {"retry"}


def test_sweeper_treats_local_enoent_as_success_and_tombstones(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock,
) -> None:
    ctx = _admit("enoent")
    target = _due_provisional(ctx, "ghost.bin", make_file=False)
    assert not target.exists()
    ctx.finish("failed")
    fake = FakeMetaFiles(clock)
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert swept.artifacts_tombstoned == 1
    assert swept.artifacts_retry == 0
    rows = _artifact_rows(lane_db, ctx.job_id)
    assert len(rows) == 1 and rows[0]["delete_state"] == "tombstoned"


def test_sweeper_deletes_expired_registered_provisional_render(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock,
) -> None:
    ctx = _admit("prov")
    target = _due_provisional(ctx, "render.png")
    ctx.finish("failed")
    fake = FakeMetaFiles(clock)
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert swept.artifacts_tombstoned == 1
    assert not target.exists()
    rows = _artifact_rows(lane_db, ctx.job_id)
    assert len(rows) == 1 and rows[0]["delete_state"] == "tombstoned"


def test_sweeper_retries_failed_meta_delete_without_local_artifact(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    ctx = _admit("metaonly")
    _prepare(uploads, ctx, short_clip)
    fake = FakeMetaFiles(clock)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=fake)
    uploads.record_failure(ctx.job_id, "metaonly-1", "meta_delete",
                           "delete failed once", clock(), meta_file_id=file_id)

    def _down(file_id):
        raise MetaFilesError("meta unreachable")

    fake.on_delete = _down
    first = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    # The Meta-only fixture: no due local artifact, one due Meta row.
    assert first.artifacts_checked == 0
    assert first.meta_checked == 1
    assert first.meta_retry == 1
    assert _meta_row(lane_db, file_id)["delete_state"] == "retry"
    # A retry row is due on the next sweep, independent of its far-future
    # provider expiry: the hourly sweep cadence is the backoff bound.
    assert file_id in {row.id for row in lifecycle.list_due_meta_files(clock())}
    fake.on_delete = None
    clock.advance(hours=1)
    second = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle, client=fake
    )
    assert second.meta_checked == 1
    assert second.meta_verified == 1
    assert file_id in fake.delete_calls
    assert _meta_row(lane_db, file_id)["delete_state"] == "verified_absent"
    assert uploads.load_job(ctx.job_id)[1].failure_count == 0


def test_delete_request_queue_is_bounded_listable_resolvable_and_retained(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routes import uploads as routes_mod
    from app.settings import Settings

    ctx = _admit("delreq")
    first = uploads.submit_delete_request(ctx.job_id, "a@example.com", clock())
    clock.advance(seconds=1)
    second = uploads.submit_delete_request(None, "b@example.com", clock())
    listed = uploads.list_delete_requests(10)
    assert [record.id for record in listed] == [first, second]
    assert listed[0].job_id == ctx.job_id
    assert listed[1].job_id is None
    assert listed[0].expires_at == listed[0].requested_at + timedelta(days=30)
    with pytest.raises(ValueError):
        uploads.list_delete_requests(0)
    with pytest.raises(ValueError):
        uploads.list_delete_requests(101)
    with pytest.raises(ValueError):
        uploads.submit_delete_request(None, "  ", clock())
    with pytest.raises(ValueError):
        uploads.submit_delete_request(None, "x" * 255, clock())
    uploads.resolve_delete_request(first)
    assert [record.id for record in uploads.list_delete_requests(10)] == [second]
    with pytest.raises(UploadError):
        uploads.resolve_delete_request(first)
    application = FastAPI()
    application.include_router(routes_mod.router)
    application.state.settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False
    )
    routes_mod.set_test_overrides(uploads=uploads, lifecycle=lifecycle)
    try:
        client = TestClient(application)
        acked = client.post(
            "/delete-request",
            data={"job_id": "", "contact": "c@example.com"},
        )
        assert acked.status_code == 200
        assert "Request received" in acked.text
        assert "compliance claim" in acked.text
        # Fill the bounded queue through the store, then prove the route
        # refuses honestly instead of dropping the request.
        live = len(uploads.list_delete_requests(100))
        for index in range(1000 - live):
            uploads.submit_delete_request(None, f"fill-{index}@example.com",
                                          clock())
        with pytest.raises(UploadError):
            uploads.submit_delete_request(None, "one-more@example.com", clock())
        full = client.post(
            "/delete-request", data={"job_id": "", "contact": "d@example.com"}
        )
        assert full.status_code == 503
        assert "inbox is full" in full.text
        uploads.resolve_delete_request(second)
        uploads.submit_delete_request(None, "after-resolve@example.com", clock())
    finally:
        routes_mod.clear_test_overrides()
    # Retention: past 30 days the sweep purges every row.
    clock.advance(days=31)
    assert uploads.purge_delete_requests(clock()) == 1000
    assert uploads.list_delete_requests(100) == ()


def test_sweep_cleans_stale_staging_but_keeps_fresh(
    tmp_path: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock,
) -> None:
    import os

    from app.deletion import sweep_expired
    from app.uploads import STAGING_DIRNAME

    staging = tmp_path / "data" / STAGING_DIRNAME
    staging.mkdir(parents=True)
    stale = staging / "up-deadbeef.bin"
    stale.write_bytes(b"personal footage from a crashed upload")
    fresh = staging / "up-liveface.bin"
    fresh.write_bytes(b"possibly in flight")
    stale_mtime = (clock() - timedelta(hours=2)).timestamp()
    fresh_mtime = (clock() - timedelta(minutes=10)).timestamp()
    os.utime(stale, (stale_mtime, stale_mtime))
    os.utime(fresh, (fresh_mtime, fresh_mtime))
    nested = staging / "subdir"
    nested.mkdir()
    (nested / "nested.bin").write_bytes(b"not staging")
    link = staging / "link.bin"
    try:
        link.symlink_to(fresh)
        have_link = True
    except OSError:
        have_link = False
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle,
        client=FakeMetaFiles(clock),
    )
    assert swept.staging_cleaned == 1
    assert not stale.exists()
    assert fresh.is_file()  # under the age bound: untouched
    assert (nested / "nested.bin").is_file()  # never recurses
    if have_link:
        assert link.is_symlink()  # never follows symlinks


def test_staging_cleanup_on_missing_directory_cleans_nothing(
    tmp_path: Path, clock: FakeClock,
) -> None:
    from app.uploads import cleanup_stale_staging

    assert cleanup_stale_staging(data_root=tmp_path / "data", now=clock()) == 0


def test_sweep_cleans_killed_prepare_orphans_in_expired_jobs(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock,
) -> None:
    import os

    from app.admission import job_dir
    from app.deletion import sweep_expired

    mid_copy = _admit("killcopy")  # SIGKILL mid-copy: staged bytes only
    mid_ffmpeg = _admit("killffmpeg")  # SIGKILL mid-ffmpeg: staged + partial
    copy_stage = job_dir(str(mid_copy.job_id)) / ".stage-deadbeef01.bin"
    copy_stage.write_bytes(b"personal footage, upload never validated")
    ffmpeg_stage = job_dir(str(mid_ffmpeg.job_id)) / ".stage-deadbeef02.bin"
    ffmpeg_stage.write_bytes(b"personal footage, transcode never finished")
    partial = job_dir(str(mid_ffmpeg.job_id)) / "preview.mp4"
    partial.write_bytes(b"partial transcode output")
    stale_mtime = (clock() - timedelta(hours=2)).timestamp()
    for path in (copy_stage, ffmpeg_stage, partial):
        os.utime(path, (stale_mtime, stale_mtime))
    clock.advance(hours=1)  # past the 240-second job deadline
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle,
        client=FakeMetaFiles(clock),
    )
    assert swept.expired_jobs == 2
    assert swept.staging_cleaned == 3
    assert swept.failures == ()
    assert not copy_stage.exists()
    assert not ffmpeg_stage.exists()
    assert not partial.exists()


def test_orphan_cleanup_keeps_fresh_registered_active_and_foreign_files(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock, short_clip: Path,
) -> None:
    import os

    from app.admission import job_dir
    from app.ledger import expire_overdue_jobs
    from app.uploads import cleanup_orphaned_job_files

    def _age(path: Path, **kwargs) -> None:
        moment = (clock() - timedelta(**kwargs)).timestamp()
        os.utime(path, (moment, moment))

    fresh_ctx = _admit("orphfresh")
    fresh_dir = job_dir(str(fresh_ctx.job_id))
    fresh_stage = fresh_dir / ".stage-inflight.bin"
    fresh_stage.write_bytes(b"possibly in flight")
    foreign = fresh_dir / "notes.txt"
    foreign.write_bytes(b"not a known orphan name")
    link = fresh_dir / ".stage-link.bin"
    try:
        link.symlink_to(fresh_stage)
        have_link = True
    except OSError:
        have_link = False
    # Registered preview, no upload row: a crash between artifact
    # registration and row insert. The artifact queue owns this file.
    reg_ctx = _admit("orphreg")
    reg_target = job_dir(str(reg_ctx.job_id)) / "preview.mp4"
    reg_target.write_bytes(b"registered transcode output")
    artifact_id = reg_ctx.artifacts.register("upload_preview", reg_target)
    # A job with an upload row is owned by the recovery queues.
    owned_ctx = _admit("orphowned")
    _prepare(uploads, owned_ctx, short_clip)
    stray = job_dir(str(owned_ctx.job_id)) / ".stage-stray.bin"
    stray.write_bytes(b"not ours to judge")
    clock.advance(hours=1)
    # Admitted after the advance, so still active with a fresh deadline.
    active_ctx = _admit("orphactive")
    active_stage = job_dir(str(active_ctx.job_id)) / ".stage-old.bin"
    active_stage.write_bytes(b"stale but the job is alive")
    # Age everything against the final clock: only the in-flight copy
    # stays under the crash bound.
    _age(fresh_stage, minutes=10)
    for path in (foreign, reg_target, stray, active_stage):
        _age(path, hours=2)
    expired = expire_overdue_jobs(clock())
    assert str(active_ctx.job_id) not in {str(job_id) for job_id in expired}
    assert cleanup_orphaned_job_files(now=clock()) == 0
    assert fresh_stage.is_file()
    assert foreign.is_file()
    if have_link:
        assert link.is_symlink()
    assert reg_target.is_file()
    assert stray.is_file()
    assert active_stage.is_file()
    conn = sqlite3.connect(lane_db)
    try:
        state = conn.execute(
            "SELECT delete_state FROM artifacts WHERE id = ?",
            (str(artifact_id),),
        ).fetchone()[0]
    finally:
        conn.close()
    assert state == "live"


def test_startup_cleanup_heals_staging_and_orphans(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock,
) -> None:
    import os

    from app.admission import job_dir
    from app.ledger import expire_overdue_jobs
    from app.uploads import STAGING_DIRNAME, run_startup_cleanup

    staging = tmp_path / "data" / STAGING_DIRNAME
    staging.mkdir(parents=True)
    stale = staging / "up-deadbeef.bin"
    stale.write_bytes(b"crashed root staging")
    ctx = _admit("startuporphan")
    orphan = job_dir(str(ctx.job_id)) / ".stage-deadbeef03.bin"
    orphan.write_bytes(b"crashed job-local staging")
    stale_mtime = (clock() - timedelta(hours=2)).timestamp()
    os.utime(stale, (stale_mtime, stale_mtime))
    os.utime(orphan, (stale_mtime, stale_mtime))
    clock.advance(hours=1)
    expire_overdue_jobs(clock())
    assert run_startup_cleanup(now=clock()) == 2
    assert not stale.exists()
    assert not orphan.exists()
    # The hook never raises, so startup never fails on cleanup.
    assert run_startup_cleanup(data_root=tmp_path / "missing") == 0


def test_recovered_expiry_past_guard_uses_immediate_deadline(
    lane_db: Path, uploads: UploadStoreImpl, lifecycle: LifecycleRepositoryImpl,
    clock: FakeClock, short_clip: Path,
) -> None:
    from app.deletion import sweep_expired

    ctx = _admit("pastguard2")
    _prepare(uploads, ctx, short_clip)
    claim = uploads.claim_confirmation(ctx.job_id, clock())
    guard = claim.record.guard_deadline_utc
    assert guard is not None
    found = MetaUpload(MetaFileId("meta-late"), guard + timedelta(minutes=30))
    fake = FakeMetaFiles(clock)
    fake.files[found.file_id] = found
    fake.by_key[claim.record.claim_id] = found

    def _down(file_id):
        raise MetaFilesError("meta unreachable")

    fake.on_delete = _down
    clock.advance(hours=2)
    swept = sweep_expired(
        clock(),
        uploads=UploadStoreImpl(lane_db, clock=clock),
        lifecycle=LifecycleRepositoryImpl(lane_db),
        client=fake,
    )
    assert swept.recoverable_recovered == 1
    assert swept.meta_retry == 1
    row = _meta_row(lane_db, found.file_id)
    # Nonconforming provider expiry is never stored: the id is queued
    # under the immediate conservative deadline instead.
    assert db.parse_dt(row["expires_at"]) == clock()
    assert row["delete_state"] == "retry"
    # ... which is due on the very next sweep, not past the guard.
    fake.on_delete = None
    clock.advance(hours=1)
    second = sweep_expired(
        clock(),
        uploads=UploadStoreImpl(lane_db, clock=clock),
        lifecycle=LifecycleRepositoryImpl(lane_db),
        client=fake,
    )
    assert second.meta_verified == 1
    assert _meta_row(lane_db, found.file_id)["delete_state"] == "verified_absent"


def test_sweep_expires_overdue_jobs_first(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock,
) -> None:
    from app.deletion import sweep_expired

    ctx = _admit("noUploadRow")  # admitted, never uploaded: holds capacity
    clock.advance(hours=1)  # past the 240-second job deadline
    swept = sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle,
        client=FakeMetaFiles(clock),
    )
    assert swept.expired_jobs == 1
    assert swept.failures == ()
    conn = sqlite3.connect(lane_db)
    try:
        state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert state == "expired"


def test_sweep_phases_fail_independently(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clock: FakeClock, short_clip: Path,
    monkeypatch,
) -> None:
    import os

    import app.deletion as deletion_mod
    from app.uploads import STAGING_DIRNAME

    def _boom(now):
        raise RuntimeError("expiry backend down")

    monkeypatch.setattr(deletion_mod, "expire_overdue_jobs", _boom)
    staging = tmp_path / "data" / STAGING_DIRNAME
    staging.mkdir(parents=True)
    stale = staging / "up-deadbeef.bin"
    stale.write_bytes(b"crashed footage")
    ancient = (clock() - timedelta(hours=2)).timestamp()
    os.utime(stale, (ancient, ancient))
    ctx = _admit("isophase")
    _prepare(uploads, ctx, short_clip)
    clock.advance(hours=25)
    swept = deletion_mod.sweep_expired(
        clock(), uploads=uploads, lifecycle=lifecycle,
        client=FakeMetaFiles(clock),
    )
    assert "expire-overdue" in swept.failures
    assert swept.staging_cleaned == 1
    assert swept.recoverable_recovered == 1
    assert swept.artifacts_tombstoned == 1
    assert uploads.load_job(ctx.job_id)[1].state == "failed"


def test_delete_request_dedupes_replays_and_throttles_sources(
    uploads: UploadStoreImpl, clock: FakeClock,
) -> None:
    from app.uploads_store import DeleteRequestThrottled

    ctx = _admit("dedupe")
    first = uploads.submit_delete_request(
        ctx.job_id, "same@example.com", clock(),
        source_hash="ip-a", session_hash="sess-a",
    )
    replay = uploads.submit_delete_request(
        ctx.job_id, "same@example.com", clock(),
        source_hash="ip-a", session_hash="sess-a",
    )
    assert replay == first
    assert len(uploads.list_delete_requests(100)) == 1
    # Five distinct requests fit; the sixth from the same source refuses.
    for index in range(4):
        uploads.submit_delete_request(
            None, f"other-{index}@example.com", clock(),
            source_hash="ip-a", session_hash="sess-a",
        )
    with pytest.raises(DeleteRequestThrottled):
        uploads.submit_delete_request(
            None, "one-too-many@example.com", clock(),
            source_hash="ip-a", session_hash="sess-a",
        )
    # Another source is unaffected, and the window resets with time.
    uploads.submit_delete_request(
        None, "elsewhere@example.com", clock(), source_hash="ip-b",
    )
    clock.advance(hours=25)
    uploads.submit_delete_request(
        None, "next-day@example.com", clock(),
        source_hash="ip-a", session_hash="sess-a",
    )
    assert len(uploads.list_delete_requests(100)) == 7
