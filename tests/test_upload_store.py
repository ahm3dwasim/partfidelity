"""Upload store: durable preview/claim/confirm/cancel/failure/recovery."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from datetime import tzinfo as _tzinfo_base
from decimal import Decimal
from pathlib import Path

import pytest

from app import contracts, db
from app.admission import admit_job
from app.contracts import JobId, MetaFileId, UploadPreview
from app.ledger import LifecycleRepositoryImpl
from app.uploads_store import UploadError, UploadStoreImpl


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


class FakeClock:
    """Injectable UTC clock; frozen until advanced, never sleeps."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start if start is not None else db.utcnow()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> datetime:
        self.now = self.now + timedelta(**kwargs)
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def uploads(lane_db: Path, clock: FakeClock) -> UploadStoreImpl:
    return UploadStoreImpl(lane_db, clock=clock)


def _root(tmp_path: Path, job_id: JobId) -> Path:
    return tmp_path / "data" / "jobs" / str(job_id)


def _admit(tag: str):
    return admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")


def _register(
    uploads: UploadStoreImpl,
    ctx,
    root: Path,
    now: datetime,
    *,
    name: str = "clip.mp4",
    duration_s: float = 12.5,
    size_bytes: int = 1000,
    audio_removed: bool = True,
    ttl_hours: float = 24,
):
    target = root / name
    target.write_bytes(b"video-bytes")
    artifact_id = ctx.artifacts.register("upload_preview", target)
    preview = UploadPreview(
        job_id=ctx.job_id,
        local_video=target,
        duration_s=duration_s,
        size_bytes=size_bytes,
        audio_removed=audio_removed,
        confirmed_at=None,
    )
    record = uploads.register_preview(
        ctx, preview, artifact_id, now + timedelta(hours=ttl_hours)
    )
    return record, artifact_id, target


def _claim(uploads: UploadStoreImpl, ctx, consented_at: datetime):
    return uploads.claim_confirmation(ctx.job_id, consented_at)


def test_upload_preview_registration_requires_admitted_job_and_matching_artifact(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    assert isinstance(uploads, contracts.UploadStore)
    ctx = _admit("reg")
    root = _root(tmp_path, ctx.job_id)
    record, artifact_id, target = _register(uploads, ctx, root, clock())
    assert record.state == "preview"
    assert record.preview.local_video == target
    assert record.preview_artifact_id == artifact_id
    assert record.failure_count == 0
    assert record.hidden is False
    # Exact replay returns the existing row instead of inserting again.
    replay_target = root / "clip.mp4"
    replay = uploads.register_preview(
        ctx,
        UploadPreview(
            job_id=ctx.job_id,
            local_video=replay_target,
            duration_s=12.5,
            size_bytes=1000,
            audio_removed=True,
            confirmed_at=None,
        ),
        artifact_id,
        clock() + timedelta(hours=24),
    )
    assert replay == record
    conn = db.connect(lane_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        shortened = conn.execute(
            "SELECT expires_at FROM artifacts WHERE id = ?", (str(artifact_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1
    assert db.parse_dt(shortened) == clock() + timedelta(hours=24)

    # A different preview for the job is refused with no change.
    other_target = root / "other.mp4"
    other_target.write_bytes(b"other")
    other_artifact = ctx.artifacts.register("upload_preview", other_target)
    with pytest.raises(UploadError, match="different preview"):
        uploads.register_preview(
            ctx,
            UploadPreview(
                job_id=ctx.job_id,
                local_video=other_target,
                duration_s=99.0,
                size_bytes=1000,
                audio_removed=True,
                confirmed_at=None,
            ),
            other_artifact,
            clock() + timedelta(hours=24),
        )

    # Unknown job, finished job, wrong kind, foreign artifact, missing file.
    from app.ledger import JobContextImpl

    ghost = JobContextImpl(lane_db, JobId("no-such-job"), tmp_path)
    with pytest.raises(UploadError, match="unknown job"):
        uploads.register_preview(
            ghost,
            UploadPreview(
                job_id=JobId("no-such-job"),
                local_video=target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            artifact_id,
            clock() + timedelta(hours=1),
        )
    done_ctx = _admit("reg-done")
    done_root = _root(tmp_path, done_ctx.job_id)
    done_target = done_root / "clip.mp4"
    done_target.write_bytes(b"video-bytes")
    done_artifact = done_ctx.artifacts.register("upload_preview", done_target)
    done_ctx.finish("failed")
    with pytest.raises(UploadError, match="not open"):
        uploads.register_preview(
            done_ctx,
            UploadPreview(
                job_id=done_ctx.job_id,
                local_video=done_target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            done_artifact,
            clock() + timedelta(hours=1),
        )
    # The failed job freed a running slot, so this admission is active.
    kind_ctx = _admit("reg-kind")
    provisional_target = _root(tmp_path, kind_ctx.job_id) / "p.png"
    provisional_target.write_bytes(b"p")
    provisional = kind_ctx.artifacts.register("provisional", provisional_target)
    with pytest.raises(UploadError, match="kind"):
        uploads.register_preview(
            kind_ctx,
            UploadPreview(
                job_id=kind_ctx.job_id,
                local_video=provisional_target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            provisional,
            clock() + timedelta(hours=1),
        )
    with pytest.raises(UploadError, match="unknown preview artifact"):
        uploads.register_preview(
            kind_ctx,
            UploadPreview(
                job_id=kind_ctx.job_id,
                local_video=target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            artifact_id,  # belongs to the first job
            clock() + timedelta(hours=1),
        )
    ghost_target = _root(tmp_path, kind_ctx.job_id) / "missing.mp4"
    ghost_artifact = kind_ctx.artifacts.register("upload_preview", ghost_target)
    with pytest.raises(UploadError, match="no regular file"):
        uploads.register_preview(
            kind_ctx,
            UploadPreview(
                job_id=kind_ctx.job_id,
                local_video=ghost_target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            ghost_artifact,
            clock() + timedelta(hours=1),
        )
    # Expiry past the 24h TTL, and naive datetimes, fail closed.
    fresh_target = _root(tmp_path, kind_ctx.job_id) / "fresh.mp4"
    fresh_target.write_bytes(b"video-bytes")
    fresh_artifact = kind_ctx.artifacts.register("upload_preview", fresh_target)
    with pytest.raises(ValueError, match="preview TTL"):
        uploads.register_preview(
            kind_ctx,
            UploadPreview(
                job_id=kind_ctx.job_id,
                local_video=fresh_target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            fresh_artifact,
            clock() + timedelta(hours=25),
        )
    with pytest.raises(ValueError, match="aware UTC"):
        uploads.register_preview(
            kind_ctx,
            UploadPreview(
                job_id=kind_ctx.job_id,
                local_video=fresh_target,
                duration_s=1.0,
                size_bytes=1,
                audio_removed=True,
                confirmed_at=None,
            ),
            fresh_artifact,
            (clock() + timedelta(hours=1)).replace(tzinfo=None),
        )


def test_upload_preview_load_rehydrates_context_after_restart(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("rehydrate")
    root = _root(tmp_path, ctx.job_id)
    record, _, _ = _register(uploads, ctx, root, clock())
    # A restart drops every object: a fresh store reloads context + record.
    restarted = UploadStoreImpl(lane_db, clock=clock)
    loaded = restarted.load_job(ctx.job_id)
    assert loaded is not None
    ctx2, record2 = loaded
    assert record2 == record
    assert ctx2.job_id == ctx.job_id
    assert isinstance(ctx2, contracts.JobContext)
    # The rehydrated context authorizes paid calls (reserve-before-spend).
    reservation = ctx2.reserve("sam", Decimal("0.0025"))
    reservation.settle(Decimal("0.001"), "succeeded")
    assert restarted.load_job(JobId("no-such-job")) is None


def test_two_connections_only_one_confirmation_claim_is_acquired(
    lane_db: Path, tmp_path: Path, clock: FakeClock
) -> None:
    ctx = _admit("race")
    first = UploadStoreImpl(lane_db, clock=clock)
    _register(first, ctx, _root(tmp_path, ctx.job_id), clock())
    second = UploadStoreImpl(lane_db, clock=clock)
    barrier = threading.Barrier(2)
    outcomes: dict[str, object] = {}
    errors: list[BaseException] = []

    def attempt(name: str, store: UploadStoreImpl) -> None:
        try:
            barrier.wait(timeout=10)
            outcomes[name] = store.claim_confirmation(ctx.job_id, clock())
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)

    threads = [
        threading.Thread(target=attempt, args=("a", first)),
        threading.Thread(target=attempt, args=("b", second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert sorted(thread.is_alive() for thread in threads) == [False, False]
    claims = (outcomes["a"], outcomes["b"])
    assert [claim.acquired for claim in claims].count(True) == 1
    assert claims[0].record.claim_id == claims[1].record.claim_id
    assert claims[0].record.claim_id is not None
    assert claims[0].record.state == "confirming"
    assert claims[1].record.state == "confirming"


def test_confirmation_completion_atomically_attaches_meta_row_and_is_idempotent(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("attach")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    assert claim.acquired is True
    started = claim.record.claim_started_at
    assert started is not None
    request_at = started + timedelta(seconds=60)
    done_at = started + timedelta(seconds=120)
    expires = request_at + timedelta(hours=24) - timedelta(seconds=60)
    record = uploads.complete_confirmation(
        claim.record.claim_id, MetaFileId("meta-1"), expires, request_at, done_at
    )
    assert record.state == "confirmed"
    assert record.meta_file_id == MetaFileId("meta-1")
    assert record.meta_expires_at == expires
    assert record.confirmed_at == done_at
    conn = db.connect(lane_db)
    try:
        meta = conn.execute(
            "SELECT job_id, expires_at, delete_state FROM meta_files WHERE id = 'meta-1'"
        ).fetchone()
        job_confirmed = conn.execute(
            "SELECT confirmed_at FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert meta["job_id"] == str(ctx.job_id)
    assert db.parse_dt(meta["expires_at"]) == expires
    assert meta["delete_state"] == "live"
    assert db.parse_dt(job_confirmed) == done_at
    # Exact replay returns the same record; a different id or expiry refuses.
    assert (
        uploads.complete_confirmation(
            claim.record.claim_id, MetaFileId("meta-1"), expires, request_at, done_at
        )
        == record
    )
    with pytest.raises(UploadError, match="conflicts"):
        uploads.complete_confirmation(
            claim.record.claim_id, MetaFileId("meta-2"), expires, request_at, done_at
        )
    with pytest.raises(UploadError, match="conflicts"):
        uploads.complete_confirmation(
            claim.record.claim_id,
            MetaFileId("meta-1"),
            expires - timedelta(seconds=1),
            request_at,
            done_at,
        )


def test_confirmation_claim_commits_provider_query_key_before_external_call(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("querykey")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    assert claim.acquired is True
    key = claim.record.claim_id
    assert key is not None and len(key) == 32
    int(key, 16)  # random 128-bit hex, committed before any Meta I/O exists
    conn = db.connect(lane_db)
    try:
        row = conn.execute(
            "SELECT claim_id, state FROM uploads WHERE job_id = ?", (str(ctx.job_id),)
        ).fetchone()
    finally:
        conn.close()
    assert row["claim_id"] == key
    assert row["state"] == "confirming"


def test_confirmation_claim_persists_absolute_guard_deadline_before_external_call(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("guard")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    assert claim.record.claim_started_at == clock()
    assert claim.record.guard_deadline_utc == clock() + timedelta(hours=25)
    assert claim.record.consented_at == clock()


def _complete_window(claim):
    started = claim.record.claim_started_at
    assert started is not None
    request_at = started + timedelta(seconds=60)
    done_at = started + timedelta(seconds=120)
    return request_at, done_at


def test_confirmation_completion_rejects_expiry_after_request_plus_24_hours(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("late-exp")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    request_at, done_at = _complete_window(claim)
    with pytest.raises(UploadError, match="exceeds request plus 24"):
        uploads.complete_confirmation(
            claim.record.claim_id,
            MetaFileId("meta-late"),
            request_at + timedelta(hours=24, seconds=1),
            request_at,
            done_at,
        )
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM meta_files WHERE id = 'meta-late'"
            ).fetchone()[0]
            == 0
        )
        up = conn.execute(
            "SELECT state, meta_file_id FROM uploads WHERE job_id = ?",
            (str(ctx.job_id),),
        ).fetchone()
    finally:
        conn.close()
    assert (up["state"], up["meta_file_id"]) == ("confirming", None)


def test_confirmation_completion_rejects_expiry_at_or_before_request_or_now(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    first = _admit("exp-past")
    _register(uploads, first, _root(tmp_path, first.job_id), clock())
    claim = _claim(uploads, first, clock())
    request_at, done_at = _complete_window(claim)
    # Already expired at request time.
    with pytest.raises(UploadError, match="already reached"):
        uploads.complete_confirmation(
            claim.record.claim_id,
            MetaFileId("meta-past"),
            request_at - timedelta(seconds=1),
            request_at,
            done_at,
        )
    # Future at request time, but reached before confirmation.
    with pytest.raises(UploadError, match="already reached"):
        uploads.complete_confirmation(
            claim.record.claim_id,
            MetaFileId("meta-slow"),
            done_at,
            request_at,
            done_at,
        )
    second = _admit("exp-past2")
    _register(uploads, second, _root(tmp_path, second.job_id), clock())
    # A request before the claim started is outside the claim window too.
    claim2 = _claim(uploads, second, clock())
    request2, done2 = _complete_window(claim2)
    with pytest.raises(UploadError, match="claim window"):
        uploads.complete_confirmation(
            claim2.record.claim_id,
            MetaFileId("meta-early"),
            request2 + timedelta(hours=1),
            request2 - timedelta(hours=2),
            done2,
        )
    conn = db.connect(lane_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM meta_files").fetchone()[0] == 0
        states = {
            row["job_id"]: (row["state"], row["meta_file_id"])
            for row in conn.execute("SELECT job_id, state, meta_file_id FROM uploads").fetchall()
        }
    finally:
        conn.close()
    assert states[str(first.job_id)] == ("confirming", None)
    assert states[str(second.job_id)] == ("confirming", None)


def test_known_meta_failure_atomically_attaches_id_and_requests_delete(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit("knownfail")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    provider_expires = clock() + timedelta(hours=20)
    record = uploads.record_failure(
        ctx.job_id,
        "fail-1",
        "meta_persist",
        "persist blew up",
        clock(),
        meta_file_id=MetaFileId("meta-known"),
        meta_expires_at=provider_expires,
    )
    assert record.state == "cancelling"
    assert record.hidden is True
    assert record.meta_file_id == MetaFileId("meta-known")
    assert record.meta_expires_at == provider_expires
    assert record.failure_count == 1
    assert record.last_error == "persist blew up"
    conn = db.connect(lane_db)
    try:
        meta = conn.execute(
            "SELECT expires_at, delete_state, delete_requested_at FROM meta_files"
            " WHERE id = 'meta-known'"
        ).fetchone()
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert db.parse_dt(meta["expires_at"]) == provider_expires
    assert meta["delete_state"] == "pending"
    assert db.parse_dt(meta["delete_requested_at"]) == clock()
    assert job_state == "failed"
    # Pending is due regardless of its future expiry: the sweeper sees it now.
    assert {row.id for row in lifecycle.list_due_meta_files(clock())} == {
        MetaFileId("meta-known")
    }
    # Exact replay replays the meta row and returns the record, count held.
    replay = uploads.record_failure(
        ctx.job_id,
        "fail-1",
        "meta_persist",
        "persist blew up",
        clock(),
        meta_file_id=MetaFileId("meta-known"),
        meta_expires_at=provider_expires,
    )
    assert replay.failure_count == 1
    assert replay.meta_file_id == MetaFileId("meta-known")
    assert claim.record.claim_id == replay.claim_id


def test_invalid_meta_expiry_known_id_uses_immediate_cleanup_deadline(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    naive_ctx = _admit("naive-exp")
    _register(uploads, naive_ctx, _root(tmp_path, naive_ctx.job_id), clock())
    _claim(uploads, naive_ctx, clock())
    naive = uploads.record_failure(
        naive_ctx.job_id,
        "fail-naive",
        "meta_upload",
        "bad expiry",
        clock(),
        meta_file_id=MetaFileId("meta-naive"),
        meta_expires_at=(clock() + timedelta(hours=2)).replace(tzinfo=None),
    )
    assert naive.state == "cancelling"
    assert naive.meta_expires_at == clock()  # conservative cleanup deadline
    missing_ctx = _admit("missing-exp")
    _register(uploads, missing_ctx, _root(tmp_path, missing_ctx.job_id), clock())
    _claim(uploads, missing_ctx, clock())
    missing = uploads.record_failure(
        missing_ctx.job_id,
        "fail-missing",
        "meta_upload",
        "no expiry",
        clock(),
        meta_file_id=MetaFileId("meta-noexp"),
    )
    assert missing.meta_expires_at == clock()
    conn = db.connect(lane_db)
    try:
        rows = {
            row["id"]: row["expires_at"]
            for row in conn.execute("SELECT id, expires_at FROM meta_files").fetchall()
        }
    finally:
        conn.close()
    assert db.parse_dt(rows["meta-naive"]) == clock()
    assert db.parse_dt(rows["meta-noexp"]) == clock()
    # An expiry without an id is refused outright.
    with pytest.raises(ValueError, match="without a meta id"):
        uploads.record_failure(
            missing_ctx.job_id,
            "fail-noid",
            "meta_upload",
            "x",
            clock(),
            meta_expires_at=clock() + timedelta(hours=1),
        )


@pytest.mark.parametrize("case", ["expired", "overlong"], ids=["expired", "overlong-aware"])
def test_unusable_meta_expiry_makes_failed_delete_retryable_at_once(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock, case: str
) -> None:
    """an expired or overlong supplied expiry stores ``now``.

    A failed direct delete then lands as a ``retry`` row whose expiry has
    already arrived, so the lifecycle query lists it at once instead of
    waiting out the bad expiry.
    """
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit(f"badexp-{case}")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    supplied = (
        clock() - timedelta(hours=1)
        if case == "expired"
        else clock() + timedelta(hours=25)
    )
    meta_id = MetaFileId(f"meta-{case}")
    record = uploads.record_failure(
        ctx.job_id,
        f"fail-{case}",
        "meta_persist",
        "persist blew up",
        clock(),
        meta_file_id=meta_id,
        meta_expires_at=supplied,
    )
    assert record.meta_expires_at == clock()  # immediate conservative deadline
    # An exact replay carrying the same unusable expiry stays idempotent:
    # the unusable expiry carries no identity, so it never conflicts.
    replay = uploads.record_failure(
        ctx.job_id,
        f"fail-{case}",
        "meta_persist",
        "persist blew up",
        clock(),
        meta_file_id=meta_id,
        meta_expires_at=supplied,
    )
    assert replay.failure_count == 1
    assert replay.meta_expires_at == clock()
    lifecycle.record_meta_file_deletion(meta_id, "retry", "direct delete failed")
    assert {row.id for row in lifecycle.list_due_meta_files(clock())} == {meta_id}


class _NoneOffsetTzinfo(_tzinfo_base):
    """tzinfo that claims a zone but reports no offset: naive in disguise."""

    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return None


class _LyingStr(str):
    """str subclass whose equality always claims True."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


def test_none_offset_tzinfo_expiry_keeps_rows_and_retries_at_once(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    """a tzinfo with ``utcoffset() is None`` stores ``now``.

    The bogus expiry must not raise inside the transaction: the failure
    and meta rows persist and the id is due for retry immediately.
    """
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit("none-offset")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    meta_id = MetaFileId("meta-none-offset")
    supplied = (clock() + timedelta(hours=2)).replace(tzinfo=_NoneOffsetTzinfo())
    assert supplied.tzinfo is not None and supplied.utcoffset() is None
    record = uploads.record_failure(
        ctx.job_id,
        "fail-none-offset",
        "meta_upload",
        "upload blew up",
        clock(),
        meta_file_id=meta_id,
        meta_expires_at=supplied,
    )
    assert record.state == "cancelling"
    assert record.meta_expires_at == clock()
    assert record.failure_count == 1
    conn = db.connect(lane_db)
    try:
        meta_rows = conn.execute("SELECT id FROM meta_files").fetchall()
        failure_rows = conn.execute(
            "SELECT failure_key FROM upload_failures"
        ).fetchall()
    finally:
        conn.close()
    assert [row["id"] for row in meta_rows] == [meta_id]
    assert [row["failure_key"] for row in failure_rows] == ["fail-none-offset"]
    lifecycle.record_meta_file_deletion(meta_id, "retry", "direct delete failed")
    assert {row.id for row in lifecycle.list_due_meta_files(clock())} == {meta_id}


def test_r3_subclass_values_rejected_at_upload_trust_boundaries(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    """Subclasses of validated str/int/float/datetime are refused."""
    ctx = _admit("subclass")
    root = _root(tmp_path, ctx.job_id)

    class _SubFloat(float):
        pass

    class _SubInt(int):
        pass

    class _SubDatetime(datetime):
        pass

    target = root / "clip.mp4"
    target.write_bytes(b"video-bytes")
    artifact_id = ctx.artifacts.register("upload_preview", target)

    def _preview(**over):
        base = dict(
            job_id=ctx.job_id,
            local_video=target,
            duration_s=12.5,
            size_bytes=1000,
            audio_removed=True,
            confirmed_at=None,
        )
        base.update(over)
        return UploadPreview(**base)

    expiry = clock() + timedelta(hours=24)
    with pytest.raises(ValueError, match="must be a number"):
        uploads.register_preview(
            ctx, _preview(duration_s=_SubFloat(12.5)), artifact_id, expiry
        )
    with pytest.raises(ValueError, match="must be an int"):
        uploads.register_preview(
            ctx, _preview(size_bytes=_SubInt(1000)), artifact_id, expiry
        )
    with pytest.raises(ValueError, match="different job"):
        uploads.register_preview(
            ctx,
            _preview(job_id=_LyingStr(str(ctx.job_id))),
            artifact_id,
            expiry,
        )
    _register(uploads, ctx, root, clock())
    _claim(uploads, ctx, clock())
    with pytest.raises(ValueError, match="is unknown"):
        uploads.record_failure(
            ctx.job_id,
            "fail-stage",
            _LyingStr("meta_upload"),  # type: ignore[arg-type]
            "boom",
            clock(),
        )
    with pytest.raises(ValueError, match="must be a string"):
        uploads.record_failure(
            ctx.job_id, "fail-err", "meta_upload", _LyingStr("boom"), clock()
        )
    with pytest.raises(ValueError, match="non-empty string"):
        uploads.submit_delete_request(
            None, _LyingStr("op@example.com"), clock()  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must be an int"):
        uploads.list_delete_requests(_SubInt(5))  # type: ignore[arg-type]
    aware_subclass = _SubDatetime.fromtimestamp(
        clock().timestamp(), tz=timezone.utc
    )
    assert aware_subclass.utcoffset() is not None
    with pytest.raises(ValueError, match="aware UTC"):
        uploads.record_failure(
            ctx.job_id, "fail-now", "meta_upload", "boom", aware_subclass
        )
    with pytest.raises(ValueError, match="aware UTC"):
        uploads.record_failure(
            ctx.job_id,
            "fail-nownaive",
            "meta_upload",
            "boom",
            clock().replace(tzinfo=_NoneOffsetTzinfo()),
        )


def test_meta_expiry_at_exactly_24h_is_accepted_verbatim(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    """the bound is inclusive — exactly now+24h stays valid."""
    ctx = _admit("bound24")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    provider_expires = clock() + timedelta(hours=24)
    record = uploads.record_failure(
        ctx.job_id,
        "fail-bound24",
        "meta_persist",
        "persist blew up",
        clock(),
        meta_file_id=MetaFileId("meta-bound24"),
        meta_expires_at=provider_expires,
    )
    assert record.meta_expires_at == provider_expires


def test_cancellation_winning_completion_marks_meta_due_in_same_transaction(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit("cancelwin")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    request_at, done_at = _complete_window(claim)
    cancelled = uploads.request_cancellation(ctx.job_id, request_at)
    assert cancelled.state == "cancelling"
    assert cancelled.hidden is True
    record = uploads.complete_confirmation(
        claim.record.claim_id,
        MetaFileId("meta-cancelled"),
        request_at + timedelta(hours=23),
        request_at,
        done_at,
    )
    # Cancellation won: attached, but never handed to the readback module as confirmed.
    assert record.state == "cancelling"
    assert record.meta_file_id == MetaFileId("meta-cancelled")
    conn = db.connect(lane_db)
    try:
        meta = conn.execute(
            "SELECT delete_state, delete_requested_at FROM meta_files"
            " WHERE id = 'meta-cancelled'"
        ).fetchone()
    finally:
        conn.close()
    assert meta["delete_state"] == "pending"
    assert db.parse_dt(meta["delete_requested_at"]) == done_at
    assert {row.id for row in lifecycle.list_due_meta_files(done_at)} == {
        MetaFileId("meta-cancelled")
    }


def test_request_cancellation_marks_every_existing_meta_row_due_in_same_transaction(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("cancelall")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    request_at, done_at = _complete_window(claim)
    uploads.complete_confirmation(
        claim.record.claim_id,
        MetaFileId("meta-first"),
        request_at + timedelta(hours=23),
        request_at,
        done_at,
    )
    # A second live Meta row for the job (provider-side duplicate handle).
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "INSERT INTO meta_files (id, job_id, expires_at, delete_state,"
            " delete_requested_at, verified_absent_at, last_error)"
            " VALUES ('meta-second', ?, ?, 'live', NULL, NULL, NULL)",
            (str(ctx.job_id), db.format_dt(request_at + timedelta(hours=23))),
        )
    finally:
        conn.close()
    record = uploads.request_cancellation(ctx.job_id, done_at)
    assert record.state == "cancelling"
    assert record.cancel_requested_at == done_at
    conn = db.connect(lane_db)
    try:
        rows = {
            row["id"]: (row["delete_state"], row["delete_requested_at"])
            for row in conn.execute(
                "SELECT id, delete_state, delete_requested_at FROM meta_files"
                " WHERE job_id = ?",
                (str(ctx.job_id),),
            ).fetchall()
        }
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert rows == {
        "meta-first": ("pending", db.format_dt(done_at)),
        "meta-second": ("pending", db.format_dt(done_at)),
    }
    assert job_state == "failed"
    # Replay keeps the earliest cancellation time and stays due.
    replay = uploads.request_cancellation(ctx.job_id, done_at + timedelta(seconds=5))
    assert replay.cancel_requested_at == done_at
    assert replay.state == "cancelling"


def _state_of(uploads: UploadStoreImpl, job_id: JobId) -> str:
    loaded = uploads.load_job(job_id)
    assert loaded is not None
    return loaded[1].state


def test_upload_state_transition_matrix_refuses_every_other_edge(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    def attempt(fn, job_id: JobId, before: str) -> None:
        with pytest.raises((UploadError, ValueError)):
            fn()
        assert _state_of(uploads, job_id) == before

    preview_ctx = _admit("mx-preview")
    _register(uploads, preview_ctx, _root(tmp_path, preview_ctx.job_id), clock())
    confirmed_ctx = _admit("mx-confirmed")
    _register(uploads, confirmed_ctx, _root(tmp_path, confirmed_ctx.job_id), clock())
    confirming_claim = _claim(uploads, confirmed_ctx, clock())
    # A live claim stands contenders down instead of erroring.
    again = _claim(uploads, confirmed_ctx, clock())
    assert again.acquired is False
    assert again.record.claim_id == confirming_claim.record.claim_id
    request_at, done_at = _complete_window(confirming_claim)
    uploads.complete_confirmation(
        confirming_claim.record.claim_id,
        MetaFileId("meta-mx"),
        request_at + timedelta(hours=23),
        request_at,
        done_at,
    )
    assert _state_of(uploads, confirmed_ctx.job_id) == "confirmed"
    assert _claim(uploads, confirmed_ctx, clock()).acquired is False
    # An upload failure cannot move a confirmed upload back to processing.
    attempt(
        lambda: uploads.record_failure(
            confirmed_ctx.job_id, "mx-first", "meta_upload", "boom", clock()
        ),
        confirmed_ctx.job_id,
        "confirmed",
    )
    # A waiting job cannot claim; cancelling the preview job promotes it.
    waiter = _admit("mx-waiter")
    _register(uploads, waiter, _root(tmp_path, waiter.job_id), clock())
    conn = db.connect(lane_db)
    try:
        waiter_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(waiter.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert waiter_state == "waiting"
    attempt(lambda: _claim(uploads, waiter, clock()), waiter.job_id, "preview")
    uploads.request_cancellation(preview_ctx.job_id, clock())
    assert _state_of(uploads, preview_ctx.job_id) == "cancelling"
    attempt(
        lambda: _claim(uploads, preview_ctx, clock()),
        preview_ctx.job_id,
        "cancelling",
    )
    failed_claim = _claim(uploads, waiter, clock())
    uploads.record_failure(waiter.job_id, "mx-second", "meta_upload", "boom", clock())
    assert _state_of(uploads, waiter.job_id) == "failed"
    attempt(lambda: _claim(uploads, waiter, clock()), waiter.job_id, "failed")
    # Completing a failed claim never reissues it.
    attempt(
        lambda: uploads.complete_confirmation(
            failed_claim.record.claim_id,
            MetaFileId("meta-never"),
            clock() + timedelta(hours=1),
            clock(),
            clock(),
        ),
        waiter.job_id,
        "failed",
    )
    # Cancelling the failed upload frees a running slot for the expiry probe.
    uploads.request_cancellation(waiter.job_id, clock())
    assert _state_of(uploads, waiter.job_id) == "cancelling"
    expiring = _admit("mx-expiring")
    _register(
        uploads, expiring, _root(tmp_path, expiring.job_id), clock(), ttl_hours=0.001
    )
    clock.advance(seconds=10)
    with pytest.raises(UploadError, match="expired"):
        # Expiry passes while the job deadline, on the same clock, holds.
        _claim(uploads, expiring, clock())
    # Cancelled is terminal: guard-pass closes the cancelling claim, and a
    # failure afterwards is refused with no change.
    clock.advance(hours=26)
    # The cancel queued the preview; the sweeper's delete lets it close.
    waiter.artifacts.mark_deleted(uploads.load_job(waiter.job_id)[1].preview_artifact_id)
    closed = uploads.recover_job(
        waiter.job_id, clock(), no_match_claim_id=failed_claim.record.claim_id
    )
    assert closed.state == "cancelled"
    attempt(
        lambda: uploads.record_failure(
            waiter.job_id, "mx-last", "meta_upload", "boom", clock()
        ),
        waiter.job_id,
        "cancelled",
    )
    # Unknown claim, unknown job, bad stage, naive datetimes.
    with pytest.raises(UploadError, match="unknown confirmation claim"):
        uploads.complete_confirmation(
            "no-such-claim",
            MetaFileId("meta-x"),
            clock() + timedelta(hours=1),
            clock(),
            clock(),
        )
    with pytest.raises(UploadError, match="unknown upload job"):
        uploads.recover_job(JobId("no-such-job"), clock())
    with pytest.raises(ValueError, match="unknown"):
        uploads.record_failure(
            confirmed_ctx.job_id, "mx-bad", "bogus-stage", "x", clock()  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="aware UTC"):
        uploads.claim_confirmation(confirmed_ctx.job_id, clock().replace(tzinfo=None))
    with pytest.raises(ValueError, match="aware UTC"):
        uploads.list_recoverable(clock().replace(tzinfo=None))


def test_cancellation_wins_confirmation_race_and_hides_before_io(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("racecancel")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    other = UploadStoreImpl(lane_db, clock=clock)
    barrier = threading.Barrier(2)
    outcomes: dict[str, object] = {}
    errors: list[BaseException] = []

    def claim_side() -> None:
        try:
            barrier.wait(timeout=10)
            outcomes["claim"] = other.claim_confirmation(ctx.job_id, clock())
        except BaseException as error:  # noqa: BLE001 - asserted below
            errors.append(error)

    def cancel_side() -> None:
        try:
            barrier.wait(timeout=10)
            outcomes["cancel"] = uploads.request_cancellation(ctx.job_id, clock())
        except BaseException as error:  # noqa: BLE001 - asserted below
            errors.append(error)

    threads = [
        threading.Thread(target=claim_side),
        threading.Thread(target=cancel_side),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    # Either order is safe: a claim that loses the race is refused, and the
    # committed row is hidden before any deletion I/O runs.
    assert [error for error in errors if not isinstance(error, UploadError)] == []
    assert sorted(thread.is_alive() for thread in threads) == [False, False]
    loaded = uploads.load_job(ctx.job_id)
    assert loaded is not None
    assert loaded[1].hidden is True
    assert loaded[1].state == "cancelling"
    assert loaded[1].cancel_requested_at is not None


def test_failure_key_is_counted_once_and_global_block_is_database_derived(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    assert uploads.public_uploads_blocked() is False
    first = _admit("block1")
    _register(uploads, first, _root(tmp_path, first.job_id), clock())
    record = uploads.record_failure(
        first.job_id, "block-k1", "meta_upload", "boom", clock()
    )
    assert record.failure_count == 1
    replay = uploads.record_failure(
        first.job_id, "block-k1", "meta_upload", "boom", clock()
    )
    assert replay.failure_count == 1
    long_error = "e" * 600
    second_key = uploads.record_failure(
        first.job_id, "block-k2", "recovery", long_error, clock()
    )
    assert second_key.failure_count == 2
    assert second_key.last_error == "e" * 512
    with pytest.raises(UploadError, match="conflicting payload"):
        uploads.record_failure(
            first.job_id, "block-k1", "meta_upload", "different", clock()
        )
    second = _admit("block2")
    _register(uploads, second, _root(tmp_path, second.job_id), clock())
    _claim(uploads, second, clock())
    third = uploads.record_failure(
        second.job_id, "block-k3", "meta_upload", "boom", clock()
    )
    assert third.failure_count == 1
    # A fresh store object agrees: the block derives from durable rows.
    assert UploadStoreImpl(lane_db, clock=clock).public_uploads_blocked() is True
    # Resolving one job's row (claim guard path) un-blocks the workers.
    stored = uploads.load_job(second.job_id)
    assert stored is not None
    clock.advance(hours=26)
    uploads.recover_job(
        second.job_id, clock(), no_match_claim_id=stored[1].claim_id
    )
    assert UploadStoreImpl(lane_db, clock=clock).public_uploads_blocked() is False


def test_recover_job_never_clears_another_jobs_failures(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    first = _admit("iso1")
    _register(uploads, first, _root(tmp_path, first.job_id), clock())
    failed = uploads.record_failure(
        first.job_id, "iso-k1", "meta_upload", "boom", clock()
    )
    assert failed.failure_count == 1
    second = _admit("iso2")
    _register(uploads, second, _root(tmp_path, second.job_id), clock())
    _claim(uploads, second, clock())
    uploads.record_failure(second.job_id, "iso-k2", "meta_upload", "boom", clock())
    stored = uploads.load_job(second.job_id)
    assert stored is not None
    clock.advance(hours=26)
    uploads.recover_job(
        second.job_id, clock(), no_match_claim_id=stored[1].claim_id
    )
    conn = db.connect(lane_db)
    try:
        rows = {
            row["failure_key"]: row["resolved_at"]
            for row in conn.execute(
                "SELECT failure_key, resolved_at FROM upload_failures"
            ).fetchall()
        }
    finally:
        conn.close()
    assert rows["iso-k2"] is not None
    assert rows["iso-k1"] is None
    assert uploads.load_job(first.job_id)[1].failure_count == 1


def test_stale_claim_never_becomes_a_second_upload_claim(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("stale")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    key = claim.record.claim_id
    clock.advance(hours=26)
    # Past the guard the old claim still owns the row: no takeover, no new key.
    again = _claim(uploads, ctx, clock())
    assert again.acquired is False
    assert again.record.claim_id == key
    assert again.record.state == "confirming"
    # Once the stale claim resolves, it is never reissued either.
    uploads.recover_job(ctx.job_id, clock(), no_match_claim_id=key)
    assert _state_of(uploads, ctx.job_id) == "failed"
    with pytest.raises(UploadError):
        _claim(uploads, ctx, clock())


def test_failed_unknown_id_claim_is_still_listed_for_provider_lookup(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    failed_ctx = _admit("listed-failed")
    _register(uploads, failed_ctx, _root(tmp_path, failed_ctx.job_id), clock())
    _claim(uploads, failed_ctx, clock())
    uploads.record_failure(
        failed_ctx.job_id, "listed-k1", "meta_upload", "boom", clock()
    )
    cancelling_ctx = _admit("listed-cancelling")
    _register(
        uploads, cancelling_ctx, _root(tmp_path, cancelling_ctx.job_id), clock()
    )
    _claim(uploads, cancelling_ctx, clock())
    uploads.request_cancellation(cancelling_ctx.job_id, clock())
    by_job = {record.preview.job_id: record for record in uploads.list_recoverable(clock())}
    assert set(by_job) == {failed_ctx.job_id, cancelling_ctx.job_id}
    assert by_job[failed_ctx.job_id].state == "failed"
    assert by_job[cancelling_ctx.job_id].state == "cancelling"
    assert by_job[failed_ctx.job_id].claim_id is not None
    assert by_job[failed_ctx.job_id].meta_file_id is None


def test_definitive_no_match_before_guard_deadline_stays_recoverable(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("nomatch-early")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    key = claim.record.claim_id
    uploads.record_failure(ctx.job_id, "early-k1", "meta_upload", "boom", clock())
    # Omitted key and mismatched key resolve nothing (mismatch refuses).
    assert uploads.recover_job(ctx.job_id, clock()).failure_count == 1
    with pytest.raises(UploadError, match="no-match"):
        uploads.recover_job(ctx.job_id, clock(), no_match_claim_id="wrong-key")
    # Before the guard, and exactly at it, the claim stays queued.
    clock.advance(hours=24)
    early = uploads.recover_job(ctx.job_id, clock(), no_match_claim_id=key)
    assert early.failure_count == 1
    clock.advance(hours=1)
    assert clock() == early.guard_deadline_utc
    at_guard = uploads.recover_job(ctx.job_id, clock(), no_match_claim_id=key)
    assert at_guard.failure_count == 1
    assert at_guard.state == "failed"
    assert {record.preview.job_id for record in uploads.list_recoverable(clock())} == {
        ctx.job_id
    }


def test_definitive_no_match_after_guard_deadline_recovers_claim(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("nomatch-late")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    uploads.record_failure(ctx.job_id, "late-k1", "meta_upload", "boom", clock())
    # A confirming claim with no failure rows resolves the same way; claim
    # it before the clock advances, while the job deadline still holds.
    confirming_ctx = _admit("nomatch-confirming")
    _register(
        uploads, confirming_ctx, _root(tmp_path, confirming_ctx.job_id), clock()
    )
    confirming_claim = _claim(uploads, confirming_ctx, clock())
    clock.advance(hours=26)
    record = uploads.recover_job(
        ctx.job_id, clock(), no_match_claim_id=claim.record.claim_id
    )
    assert record.state == "failed"
    assert record.failure_count == 0
    assert record.last_error is None
    assert {row.preview.job_id for row in uploads.list_recoverable(clock())} == {
        confirming_ctx.job_id
    }
    conn = db.connect(lane_db)
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
        resolved = conn.execute(
            "SELECT resolved_at FROM upload_failures WHERE failure_key = 'late-k1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert job_state == "failed"
    assert db.parse_dt(resolved) == clock()
    resolved_record = uploads.recover_job(
        confirming_ctx.job_id, clock(), no_match_claim_id=confirming_claim.record.claim_id
    )
    assert resolved_record.state == "failed"
    assert uploads.list_recoverable(clock()) == ()


def test_expired_unclaimed_preview_is_recoverable_and_terminalized_before_artifact_cleanup(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit("abandoned")
    root = _root(tmp_path, ctx.job_id)
    record, _, target = _register(uploads, ctx, root, clock(), ttl_hours=1)
    assert uploads.list_recoverable(clock()) == ()
    clock.advance(hours=2)
    assert [row.preview.job_id for row in uploads.list_recoverable(clock())] == [
        ctx.job_id
    ]
    recovered = uploads.recover_job(ctx.job_id, clock())
    assert recovered.state == "failed"
    assert recovered.hidden is True
    conn = db.connect(lane_db)
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
    finally:
        conn.close()
    assert job_state == "failed"
    # Terminalized first: the same pass's artifact queue then deletes footage.
    assert {row.id for row in lifecycle.list_due_artifacts(clock())} == {
        record.preview_artifact_id
    }
    assert target.exists()
    target.unlink()
    lifecycle.record_artifact_deletion(record.preview_artifact_id, "tombstoned")
    assert lifecycle.list_due_artifacts(clock()) == ()


def test_delete_request_queue_is_bounded_listable_and_purged_on_retention(
    tmp_path: Path,
    uploads: UploadStoreImpl,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _admit("delreq")
    first_id = uploads.submit_delete_request(
        ctx.job_id, "operator@example.com", clock()
    )
    clock.advance(seconds=1)
    second_id = uploads.submit_delete_request(None, "lost-id@example.com", clock())
    listed = uploads.list_delete_requests(100)
    assert [row.id for row in listed] == [first_id, second_id]
    assert listed[0].job_id == ctx.job_id
    assert listed[1].job_id is None
    assert listed[0].contact == "operator@example.com"
    assert listed[0].expires_at == listed[0].requested_at + timedelta(days=30)
    with pytest.raises(ValueError, match="page size"):
        uploads.list_delete_requests(0)
    with pytest.raises(ValueError, match="page size"):
        uploads.list_delete_requests(101)
    with pytest.raises(ValueError, match="length bound"):
        uploads.submit_delete_request(ctx.job_id, "x" * 255, clock())
    with pytest.raises(ValueError, match="non-empty"):
        uploads.submit_delete_request(ctx.job_id, "   ", clock())
    # The pending bound refuses with no insert (patched small for speed).
    monkeypatch.setattr(
        "app.uploads_store.DELETE_REQUEST_MAX_PENDING", 2
    )
    with pytest.raises(UploadError, match="queue is full"):
        uploads.submit_delete_request(ctx.job_id, "third@example.com", clock())
    assert len(uploads.list_delete_requests(100)) == 2
    # Resolve deletes immediately; retention purges the expired.
    uploads.resolve_delete_request(first_id)
    assert [row.id for row in uploads.list_delete_requests(100)] == [second_id]
    with pytest.raises(UploadError, match="unknown delete request"):
        uploads.resolve_delete_request(first_id)
    clock.advance(days=31)
    assert uploads.purge_delete_requests(clock()) == 1
    assert uploads.list_delete_requests(100) == ()
    # Submit purges expired rows in the same transaction it counts.
    assert uploads.purge_delete_requests(clock()) == 0


def test_failure_replay_adding_or_changing_meta_identity_is_rejected(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("replaymeta")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    uploads.record_failure(ctx.job_id, "same-key", "meta_upload", "boom", clock())
    # A replay that adds a Meta id is rejected, never a silent success
    # that leaves the remote file unqueued for deletion.
    with pytest.raises(UploadError, match="conflicting meta"):
        uploads.record_failure(
            ctx.job_id,
            "same-key",
            "meta_upload",
            "boom",
            clock(),
            meta_file_id=MetaFileId("remote-orphan"),
            meta_expires_at=clock() + timedelta(hours=1),
        )
    conn = db.connect(lane_db)
    try:
        orphaned = conn.execute(
            "SELECT COUNT(*) FROM meta_files WHERE id = 'remote-orphan'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert orphaned == 0
    stored = uploads.load_job(ctx.job_id)
    assert stored is not None
    assert stored[1].failure_count == 1
    assert stored[1].meta_file_id is None
    # A replay that changes the recorded id or valid expiry is rejected too.
    provider_expires = clock() + timedelta(hours=20)
    uploads.record_failure(
        ctx.job_id,
        "known-key",
        "meta_persist",
        "bust",
        clock(),
        meta_file_id=MetaFileId("meta-attached"),
        meta_expires_at=provider_expires,
    )
    with pytest.raises(UploadError, match="conflicting meta"):
        uploads.record_failure(
            ctx.job_id,
            "known-key",
            "meta_persist",
            "bust",
            clock(),
            meta_file_id=MetaFileId("meta-other"),
            meta_expires_at=provider_expires,
        )
    with pytest.raises(UploadError, match="conflicting meta"):
        uploads.record_failure(
            ctx.job_id,
            "known-key",
            "meta_persist",
            "bust",
            clock(),
            meta_file_id=MetaFileId("meta-attached"),
            meta_expires_at=provider_expires + timedelta(hours=1),
        )
    # An exact replay still succeeds without touching the count, as does a
    # replay that omits the id (the original call already queued it).
    assert (
        uploads.record_failure(
            ctx.job_id,
            "known-key",
            "meta_persist",
            "bust",
            clock(),
            meta_file_id=MetaFileId("meta-attached"),
            meta_expires_at=provider_expires,
        ).failure_count
        == 2
    )
    assert (
        uploads.record_failure(
            ctx.job_id, "known-key", "meta_persist", "bust", clock()
        ).failure_count
        == 2
    )


def test_record_failure_on_unclaimed_preview_terminalizes_job_in_same_transaction(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit("failprev")
    filler = _admit("failprev-filler")
    waiter = _admit("failprev-waiter")
    root = _root(tmp_path, ctx.job_id)
    record, _, _ = _register(uploads, ctx, root, clock())
    hold = ctx.reserve("other", Decimal("0.04"))
    assert hold is not None
    failed = uploads.record_failure(
        ctx.job_id, "prev-k1", "meta_upload", "boom", clock()
    )
    assert failed.state == "failed"
    assert failed.hidden is True
    conn = db.connect(lane_db)
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()[0]
        waiter_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(waiter.job_id),)
        ).fetchone()[0]
        held = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE job_id = ? AND state = 'held'",
            (str(ctx.job_id),),
        ).fetchone()[0]
    finally:
        conn.close()
    assert job_state == "failed"  # no failed upload left with an active job
    assert waiter_state == "active"  # the freed slot promotes the waiter
    assert held == 0  # unsettled holds released with the job
    assert uploads.list_recoverable(clock()) == ()  # nothing stranded
    # The preview footage is due for the same pass's artifact queue.
    assert {row.id for row in lifecycle.list_due_artifacts(clock())} == {
        record.preview_artifact_id
    }
    assert filler.job_id != ctx.job_id


def test_hide_expired_preview_hides_only_past_expiry_and_touches_nothing_else(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    ctx = _admit("hideprev")
    root = _root(tmp_path, ctx.job_id)
    record, _, _ = _register(uploads, ctx, root, clock())
    # Unexpired rows are an exact no-op.
    same = uploads.hide_expired_preview(ctx.job_id, clock())
    assert same.hidden is False
    assert same.state == "preview"
    clock.advance(hours=25)
    hidden = uploads.hide_expired_preview(ctx.job_id, clock())
    assert hidden.hidden is True
    assert hidden.state == "preview"
    assert hidden.claim_id is None
    assert hidden.meta_file_id is None
    assert hidden.failure_count == 0
    assert hidden.preview_expires_at == record.preview_expires_at
    # Already-hidden rows are an exact no-op, and unknown jobs refuse.
    again = uploads.hide_expired_preview(ctx.job_id, clock())
    assert again.hidden is True
    assert again.state == "preview"
    with pytest.raises(UploadError):
        uploads.hide_expired_preview(JobId("no-such-job"), clock())


class _FlakyOffsetTzinfo(_tzinfo_base):
    """tzinfo that reports UTC a fixed number of times, then fails."""

    def __init__(self, good_calls: int) -> None:
        self.remaining = good_calls
        self.calls = 0

    def utcoffset(self, dt):
        self.calls += 1
        if self.remaining <= 0:
            raise TypeError("offset no longer available")
        self.remaining -= 1
        return timedelta(0)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "UTC"


class _StripLyingStr(str):
    """str subclass whose strip() invents content for an empty value."""

    def __init__(self, *args) -> None:
        self.strip_calls = 0

    def strip(self, chars=None):
        self.strip_calls += 1
        return "present"


def test_r4_flaky_expiry_offset_keeps_rows_and_retries_at_once(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    """an offset that dies mid-read still stores ``now``.

    The verdict's repro: ``utcoffset()`` answers UTC for the validation
    calls, then raises. The failure and meta rows must still persist and
    the id must be due for retry immediately.
    """
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = _admit("flaky-offset")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    meta_id = MetaFileId("meta-flaky-offset")
    supplied = (clock() + timedelta(hours=2)).replace(tzinfo=_FlakyOffsetTzinfo(3))
    record = uploads.record_failure(
        ctx.job_id,
        "fail-flaky-offset",
        "meta_upload",
        "upload blew up",
        clock(),
        meta_file_id=meta_id,
        meta_expires_at=supplied,
    )
    assert record.state == "cancelling"
    assert record.meta_expires_at == clock()
    assert record.failure_count == 1
    conn = db.connect(lane_db)
    try:
        meta_rows = conn.execute("SELECT id FROM meta_files").fetchall()
        failure_rows = conn.execute(
            "SELECT failure_key FROM upload_failures"
        ).fetchall()
    finally:
        conn.close()
    assert [row["id"] for row in meta_rows] == [meta_id]
    assert [row["failure_key"] for row in failure_rows] == ["fail-flaky-offset"]
    lifecycle.record_meta_file_deletion(meta_id, "retry", "direct delete failed")
    assert {row.id for row in lifecycle.list_due_meta_files(clock())} == {meta_id}


@pytest.mark.parametrize("good_calls", [0, 1, 2])
def test_r4_short_flaky_expiry_offsets_all_store_now(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock, good_calls: int
) -> None:
    """any failure inside the guarded block stores ``now``.

    A tzinfo that dies on the aware check or inside the window check
    still keeps both rows; only an offset that survives the whole
    guarded canonicalization is persisted verbatim.
    """
    ctx = _admit(f"short-flaky-{good_calls}")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    supplied = (clock() + timedelta(hours=2)).replace(
        tzinfo=_FlakyOffsetTzinfo(good_calls)
    )
    record = uploads.record_failure(
        ctx.job_id,
        f"fail-short-flaky-{good_calls}",
        "meta_persist",
        "persist blew up",
        clock(),
        meta_file_id=MetaFileId(f"meta-short-flaky-{good_calls}"),
        meta_expires_at=supplied,
    )
    assert record.state == "cancelling"
    assert record.meta_expires_at == clock()
    assert record.failure_count == 1


def test_r4_flaky_confirmation_expiry_still_attaches(
    lane_db: Path, tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    """the same flaky offset cannot break ``complete``.

    ``complete_confirmation`` used to format the caller's expiry inside
    its transaction after validating it, so the verdict's offset shape
    (UTC three times, then raising) rolled back the attach. The
    canonical snapshot is now taken once and reused for the insert.
    """
    ctx = _admit("flaky-complete")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    claim = _claim(uploads, ctx, clock())
    request_at, done_at = _complete_window(claim)
    supplied = (done_at + timedelta(hours=2)).replace(tzinfo=_FlakyOffsetTzinfo(3))
    record = uploads.complete_confirmation(
        claim.record.claim_id,
        MetaFileId("meta-flaky-complete"),
        supplied,
        request_at,
        done_at,
    )
    assert record.state == "confirmed"
    assert record.meta_expires_at == done_at + timedelta(hours=2)
    conn = db.connect(lane_db)
    try:
        stored = conn.execute(
            "SELECT expires_at FROM meta_files WHERE id = 'meta-flaky-complete'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert db.parse_dt(stored) == done_at + timedelta(hours=2)


def test_r4_strip_lying_keys_rejected_at_upload_trust_boundaries(
    tmp_path: Path, uploads: UploadStoreImpl, clock: FakeClock
) -> None:
    """a str subclass is rejected even when strip() lies.

    The verdict's repro: a ``str("")`` subclass whose ``strip()``
    returns ``"present"`` must not pass the non-empty check for
    ``failure_key``, ``claim_id``, or ``no_match_claim_id`` — and the
    lying method must never be invoked.
    """
    ctx = _admit("strip-keys")
    _register(uploads, ctx, _root(tmp_path, ctx.job_id), clock())
    _claim(uploads, ctx, clock())
    failure_key = _StripLyingStr("")
    with pytest.raises(ValueError, match="non-empty string"):
        uploads.record_failure(
            ctx.job_id, failure_key, "meta_upload", "boom", clock()
        )
    assert failure_key.strip_calls == 0
    claim_key = _StripLyingStr("")
    with pytest.raises(ValueError, match="non-empty string"):
        uploads.complete_confirmation(
            claim_key,
            MetaFileId("meta-strip"),
            clock() + timedelta(hours=1),
            clock(),
            clock(),
        )
    assert claim_key.strip_calls == 0
    no_match_key = _StripLyingStr("")
    with pytest.raises(ValueError, match="non-empty string"):
        uploads.recover_job(ctx.job_id, clock(), no_match_claim_id=no_match_key)
    assert no_match_key.strip_calls == 0
    loaded = uploads.load_job(ctx.job_id)
    assert loaded is not None
    assert loaded[1].state == "confirming"
    assert loaded[1].failure_count == 0
