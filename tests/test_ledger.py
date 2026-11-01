"""Ledger: atomic reserve/settle, caps, breakers, finish, lifecycle."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

import pytest

from app import contracts, db
from app.admission import admit_job
from app.ledger import (
    AGGREGATE_ROW,
    LiveJudgeSpendRefused,
    ArtifactNotFound,
    ArtifactPathError,
    BudgetExhausted,
    ContentionError,
    JobAlreadyFinished,
    JobNotActive,
    JobTimedOut,
    LedgerError,
    LifecycleRepositoryImpl,
    MetaFileNotFound,
    ReservationNotHeld,
    ReserveRefused,
    breaker_state,
    expire_overdue_jobs,
    get_budget_row,
    insert_call_record,
    insert_event,
    list_call_records,
    register_meta_file,
)
from app.contracts import ArtifactId, JobId, MetaFileId


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


def _call_paid(ctx, kind: str, worst: Decimal, client: Callable[[], object]) -> object:
    """The read-back, SAM and guide protocol: reserve first, invoke the client only if held."""
    reservation = ctx.reserve(kind, worst)
    try:
        result = client()
    except Exception:
        reservation.settle(Decimal("0"), "failed")
        raise
    reservation.settle(worst, "succeeded")
    return result


def _held_count(database: Path, job_id: JobId) -> int:
    conn = db.connect(database)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE job_id = ? AND state = 'held'",
            (str(job_id),),
        ).fetchone()[0]
    finally:
        conn.close()


def test_job_context_reserve_raises_before_client_callback(lane_db: Path) -> None:
    ctx = admit_job("anon-cb", "ip-cb", "video-cb")
    assert isinstance(ctx, contracts.JobContext)
    assert isinstance(ctx.artifacts, contracts.ArtifactStore)
    reservation = ctx.reserve("sam", Decimal("0.0025"))
    assert isinstance(reservation, contracts.CallReservation)
    reservation.settle(Decimal("0.001"), "succeeded")

    spent = ctx.reserve("other", Decimal("0.04"))
    spent.settle(Decimal("0.04"), "succeeded")
    invoked: list[str] = []
    with pytest.raises(ReserveRefused):
        _call_paid(ctx, "other", Decimal("0.01"), lambda: invoked.append("ran"))
    assert invoked == []
    assert _held_count(lane_db, ctx.job_id) == 0


def test_counterfactual_spend_before_check_invokes_client_and_overspends(lane_db: Path) -> None:
    # The old check-after-spend order runs the client and spends money that the
    # ledger would have refused; the new order raises before invocation.
    ctx = admit_job("anon-cf", "ip-cf", "video-cf")
    for _ in range(3):
        held = ctx.reserve("readback", Decimal("0.04"))
        held.settle(Decimal("0.04"), "succeeded")
    legacy_ran: list[str] = []

    def legacy_client() -> str:
        legacy_ran.append("charged")
        return "label"

    with pytest.raises(BudgetExhausted):
        result = legacy_client()
        ctx.reserve("readback", Decimal("0.04"))
        _ = result
    assert legacy_ran == ["charged"]

    new_ran: list[str] = []
    with pytest.raises(BudgetExhausted):
        _call_paid(ctx, "readback", Decimal("0.04"), lambda: new_ran.append("charged"))
    assert new_ran == []


def test_sam_reserve_allows_32_calls_and_refuses_33rd_before_client_invocation(
    lane_db: Path,
) -> None:
    ctx = admit_job("anon-sam", "ip-sam", "video-sam")
    held = [ctx.reserve("sam", Decimal("0.0025")) for _ in range(32)]
    assert _held_count(lane_db, ctx.job_id) == 32
    invoked: list[str] = []
    with pytest.raises(BudgetExhausted, match="0.08 sam reservation"):
        _call_paid(ctx, "sam", Decimal("0.0025"), lambda: invoked.append("ran"))
    assert invoked == []
    # Even a $0 worst case is refused: the 32-call count bound is independent.
    with pytest.raises(BudgetExhausted, match="32 sam calls per job"):
        ctx.reserve("sam", Decimal("0"))
    for reservation in held:
        reservation.settle(Decimal("0.0025"), "succeeded")
    assert Decimal(get_budget_row("sam", path=lane_db)["spent_usd"]) == Decimal("0.08")
    assert Decimal(get_budget_row("sam", path=lane_db)["held_usd"]) == Decimal("0")


def test_row_day_and_aggregate_caps(lane_db: Path) -> None:
    ctx = admit_job("anon-cap", "ip-cap", "video-cap")
    day_row = f"day:{db.utc_day(db.utcnow())}"

    def set_cap(name: str, cap: str) -> None:
        conn = db.connect(lane_db)
        try:
            conn.execute(
                "UPDATE budget_rows SET hard_cap_usd = ? WHERE name = ?", (cap, name)
            )
        finally:
            conn.close()

    # Category row: the $0.08 SAM reservation fits the job but not the row.
    set_cap("sam", "0.04")
    with pytest.raises(BudgetExhausted, match="sam row cap"):
        ctx.reserve("sam", Decimal("0.08"))
    assert _held_count(lane_db, ctx.job_id) == 0
    set_cap("sam", "100")

    # Daily breaker already over cap fails closed without taking new holds.
    set_cap(day_row, "0.40")
    with pytest.raises(BudgetExhausted, match=f"{day_row} breaker already over cap"):
        ctx.reserve("spark", Decimal("0.01"))
    assert _held_count(lane_db, ctx.job_id) == 0
    set_cap(day_row, "100")

    # Aggregate ceiling behaves the same.
    set_cap(AGGREGATE_ROW, "0.40")
    with pytest.raises(BudgetExhausted, match="aggregate breaker already over cap"):
        ctx.reserve("spark", Decimal("0.01"))
    assert _held_count(lane_db, ctx.job_id) == 0
    set_cap(AGGREGATE_ROW, "100")

    # Restored ceilings authorize again.
    ctx.reserve("spark", Decimal("0.01")).settle(Decimal("0.005"), "succeeded")


def test_80_and_100_percent_breaker_transitions(tmp_path: Path) -> None:
    database = tmp_path / "breakers.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    conn = db.connect(database)
    try:
        conn.execute("UPDATE budget_rows SET hard_cap_usd = '0.10' WHERE name = 'sam'")
    finally:
        conn.close()

    first = admit_job("anon-first", "ip-first", "video-first")
    assert breaker_state("sam", path=database) == "ok"
    first.reserve("sam", Decimal("0.08"))
    assert breaker_state("sam", path=database) == "warn"
    assert Decimal(get_budget_row("sam", path=database)["breaker_usd"]) == Decimal("0.08")
    # Releasing the hold clears a warn: holds are live, spent money is not.
    first.finish("done")
    assert breaker_state("sam", path=database) == "ok"
    assert get_budget_row("sam", path=database)["breaker_usd"] is None

    tripped_db = tmp_path / "tripped.sqlite3"
    db.configure(tripped_db, tmp_path / "data2")
    db.init_db(tripped_db)
    conn = db.connect(tripped_db)
    try:
        conn.execute("UPDATE budget_rows SET hard_cap_usd = '0.16' WHERE name = 'sam'")
    finally:
        conn.close()
    one = admit_job("anon-t1", "ip-t1", "video-t1")
    two = admit_job("anon-t2", "ip-t2", "video-t2")
    one.reserve("sam", Decimal("0.08"))
    assert breaker_state("sam", path=tripped_db) == "ok"
    two.reserve("sam", Decimal("0.08"))
    assert breaker_state("sam", path=tripped_db) == "tripped"
    assert Decimal(get_budget_row("sam", path=tripped_db)["breaker_usd"]) == Decimal("0.16")
    # Over a tripped row, a fresh active job's call is refused before invoke.
    conn = db.connect(tripped_db)
    try:
        conn.execute("UPDATE budget_rows SET hard_cap_usd = '0.10' WHERE name = 'sam'")
    finally:
        conn.close()
    one.finish("done")
    three = admit_job("anon-t3", "ip-t3", "video-t3")
    with pytest.raises(BudgetExhausted, match="sam row cap"):
        three.reserve("sam", Decimal("0.08"))


def test_job_releases_unused_hold(lane_db: Path) -> None:
    ctx = admit_job("anon-hold", "ip-hold", "video-hold")
    day_row = f"day:{db.utc_day(db.utcnow())}"
    ctx.reserve("sam", Decimal("0.08")).settle(Decimal("0.01"), "succeeded")
    sam = get_budget_row("sam", path=lane_db)
    assert Decimal(sam["held_usd"]) == Decimal("0")
    assert Decimal(sam["spent_usd"]) == Decimal("0.01")
    for name in (day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["held_usd"]) == Decimal("0.38")
        assert Decimal(row["spent_usd"]) == Decimal("0.01")
    ctx.finish("done")
    for name in (day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["held_usd"]) == Decimal("0")
        assert Decimal(row["spent_usd"]) == Decimal("0.01")


def test_failed_settlement_releases_hold_and_records_failed_call(lane_db: Path) -> None:
    ctx = admit_job("anon-fail", "ip-fail", "video-fail")
    reservation = ctx.reserve("sam", Decimal("0.0025"))
    reservation.settle(Decimal("0.0025"), "failed")
    conn = db.connect(lane_db)
    try:
        entry = conn.execute(
            "SELECT state, estimated_usd, actual_usd FROM reservations"
            " WHERE id = ?",
            (reservation.reservation_id,),
        ).fetchone()
    finally:
        conn.close()
    assert (entry["state"], entry["actual_usd"]) == ("failed", "0.0025")
    sam = get_budget_row("sam", path=lane_db)
    assert Decimal(sam["held_usd"]) == Decimal("0")
    assert Decimal(sam["spent_usd"]) == Decimal("0.0025")
    with pytest.raises(ReservationNotHeld):
        reservation.settle(Decimal("0.0025"), "failed")


def test_finish_done_is_one_atomic_transition_and_tombstones_nonkept_provisionals(
    lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-done", "ip-done", "video-done")
    store = ctx.artifacts
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    doomed = store.register("provisional", root / "p1.png")
    (root / "p1.png").write_bytes(b"provisional")
    kept = store.register("provisional", root / "p2.png")
    (root / "p2.png").write_bytes(b"accepted")
    store.mark_kept(kept)
    final = store.register("final", root / "guide.json")
    pending = ctx.reserve("sam", Decimal("0.0025"))
    settled = ctx.reserve("spark", Decimal("0.01"))
    settled.settle(Decimal("0.005"), "succeeded")

    ctx.finish("done", keep_artifact_ids=[kept])
    conn = db.connect(lane_db)
    try:
        job = conn.execute("SELECT state, completed_at FROM jobs WHERE id = ?", (str(ctx.job_id),)).fetchone()
        states = {
            row["id"]: (row["delete_state"], row["kept_at"])
            for row in conn.execute(
                "SELECT id, delete_state, kept_at FROM artifacts WHERE job_id = ?",
                (str(ctx.job_id),),
            ).fetchall()
        }
        pending_state = conn.execute(
            "SELECT state FROM reservations WHERE id = ?", (pending.reservation_id,)
        ).fetchone()["state"]
        hold_state = conn.execute(
            "SELECT state FROM reservations WHERE job_id = ? AND state LIKE 'job\\_%' ESCAPE '\\'",
            (str(ctx.job_id),),
        ).fetchone()["state"]
    finally:
        conn.close()
    assert job["state"] == "done"
    assert job["completed_at"] is not None
    assert states[str(doomed)][0] == "pending_delete"
    assert states[str(kept)] == ("live", states[str(kept)][1]) and states[str(kept)][1] is not None
    assert states[str(final)][0] == "live"
    assert pending_state == "released"
    assert hold_state == "job_released"
    # Finish only queues rows for deletion; the sweeper deletes bytes later.
    assert (root / "p1.png").exists()
    assert Decimal(get_budget_row("sam", path=lane_db)["held_usd"]) == Decimal("0")

    # Exact same-set replay is idempotent; a different set or outcome refuses.
    ctx.finish("done", keep_artifact_ids=[kept])
    with pytest.raises(JobAlreadyFinished):
        ctx.finish("done")
    with pytest.raises(JobAlreadyFinished):
        ctx.finish("failed")
    with pytest.raises(JobNotActive):
        ctx.reserve("sam", Decimal("0.0025"))


def test_finish_failed_is_one_atomic_transition_and_tombstones_nonkept_provisionals(
    lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-failed", "ip-failed", "video-failed")
    store = ctx.artifacts
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    doomed = store.register("provisional", root / "p1.png")
    kept = store.register("provisional", root / "p2.png")
    store.mark_kept(kept)
    ctx.finish("failed")
    conn = db.connect(lane_db)
    try:
        state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()["state"]
        states = {
            row["id"]: (row["delete_state"], row["kept_at"])
            for row in conn.execute(
                "SELECT id, delete_state, kept_at FROM artifacts WHERE job_id = ?",
                (str(ctx.job_id),),
            ).fetchall()
        }
    finally:
        conn.close()
    assert state == "failed"
    assert states[str(doomed)] == ("pending_delete", None)
    assert states[str(kept)] == ("pending_delete", None)
    with pytest.raises(JobAlreadyFinished):
        ctx.finish("failed")


def test_lifecycle_repository_lists_only_terminal_or_expired_jobs(
    lane_db: Path, tmp_path: Path
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    assert isinstance(lifecycle, contracts.LifecycleRepository)
    past = db.format_dt(db.utcnow() - timedelta(days=1))

    done_ctx = admit_job("anon-l1", "ip-l1", "video-l1")
    root_done = tmp_path / "data" / "jobs" / str(done_ctx.job_id)
    kept_done = done_ctx.artifacts.register("provisional", root_done / "p.png")
    done_ctx.artifacts.mark_kept(kept_done)
    done_ctx.finish("done")
    register_meta_file(done_ctx.job_id, MetaFileId("meta-done"), db.utcnow() - timedelta(days=1))

    failed_ctx = admit_job("anon-l2", "ip-l2", "video-l2")
    root_failed = tmp_path / "data" / "jobs" / str(failed_ctx.job_id)
    kept_failed = failed_ctx.artifacts.register("provisional", root_failed / "p.png")
    failed_ctx.artifacts.mark_kept(kept_failed)
    failed_ctx.finish("failed")
    register_meta_file(failed_ctx.job_id, MetaFileId("meta-failed"), db.utcnow() - timedelta(days=1))

    running_ctx = admit_job("anon-l3", "ip-l3", "video-l3")
    root_running = tmp_path / "data" / "jobs" / str(running_ctx.job_id)
    live_provisional = running_ctx.artifacts.register("provisional", root_running / "p.png")
    register_meta_file(running_ctx.job_id, MetaFileId("meta-live"), db.utcnow() - timedelta(days=1))

    waiter_ctx = admit_job("anon-l4", "ip-l4", "video-l4")
    waiter_ctx2 = admit_job("anon-l5", "ip-l5", "video-l5")
    root_waiter = tmp_path / "data" / "jobs" / str(waiter_ctx2.job_id)
    waiting_provisional = waiter_ctx2.artifacts.register("provisional", root_waiter / "p.png")

    conn = db.connect(lane_db)
    try:
        for artifact in (kept_done, kept_failed, live_provisional, waiting_provisional):
            conn.execute("UPDATE artifacts SET expires_at = ? WHERE id = ?", (past, str(artifact)))
        # running and waiting jobs are overdue but still open: past their
        # deadline, yet persisted as active/waiting until expiry runs.
        conn.execute(
            "UPDATE jobs SET deadline_at = ? WHERE id IN (?, ?)",
            (past, str(running_ctx.job_id), str(waiter_ctx2.job_id)),
        )
    finally:
        conn.close()

    now = db.utcnow()
    # Terminal jobs only: overdue-but-open rows are never listed.
    assert {record.id for record in lifecycle.list_due_artifacts(now)} == {
        kept_done,
        kept_failed,
    }
    assert {record.id for record in lifecycle.list_due_meta_files(now)} == {
        MetaFileId("meta-done"),
        MetaFileId("meta-failed"),
    }
    assert {running_ctx.job_id, waiter_ctx.job_id, waiter_ctx2.job_id}.isdisjoint(
        {record.job_id for record in lifecycle.list_due_artifacts(now)}
    )

    # Only the transactional expiration makes overdue rows discoverable.
    assert expire_overdue_jobs(path=lane_db) == (
        running_ctx.job_id,
        waiter_ctx2.job_id,
    )
    assert {record.id for record in lifecycle.list_due_artifacts(now)} == {
        kept_done,
        kept_failed,
        live_provisional,
        waiting_provisional,
    }
    assert {record.id for record in lifecycle.list_due_meta_files(now)} == {
        MetaFileId("meta-done"),
        MetaFileId("meta-failed"),
        MetaFileId("meta-live"),
    }
    # The still-open waiting job's rows stay absent.
    assert waiter_ctx.job_id not in {
        record.job_id for record in lifecycle.list_due_artifacts(now)
    }

    lifecycle.record_artifact_deletion(kept_done, "tombstoned")
    lifecycle.record_artifact_deletion(kept_failed, "retry", "disk busy")
    lifecycle.record_meta_file_deletion(MetaFileId("meta-done"), "verified_absent")
    lifecycle.record_meta_file_deletion(MetaFileId("meta-failed"), "acknowledged_only")
    lifecycle.record_meta_file_deletion(MetaFileId("meta-live"), "retry", "boom")
    assert {record.id for record in lifecycle.list_due_artifacts(now)} == {
        kept_failed,
        live_provisional,
        waiting_provisional,
    }
    assert {record.id for record in lifecycle.list_due_meta_files(now)} == {
        MetaFileId("meta-live"),
    }
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT last_error FROM meta_files WHERE id = 'meta-live'"
            ).fetchone()["last_error"]
            == "boom"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM events WHERE event = 'artifact_delete_retry'"
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()
    with pytest.raises(ArtifactNotFound):
        lifecycle.record_artifact_deletion(contracts.ArtifactId("nope"), "retry")
    with pytest.raises(MetaFileNotFound):
        lifecycle.record_meta_file_deletion(MetaFileId("nope"), "retry")


def test_per_job_reservation_contract_46_plus_04_headroom(lane_db: Path) -> None:
    ctx = admit_job("anon-46", "ip-46", "video-46")
    sparks = [ctx.reserve("spark", Decimal("0.0866")) for _ in range(3)]
    with pytest.raises(BudgetExhausted, match="0.26 spark reservation"):
        ctx.reserve("spark", Decimal("0.0866"))
    reads = [ctx.reserve("readback", Decimal("0.04")) for _ in range(3)]
    with pytest.raises(BudgetExhausted):
        ctx.reserve("readback", Decimal("0.04"))
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT readback_fired_n FROM jobs WHERE id = ?", (str(ctx.job_id),)
            ).fetchone()["readback_fired_n"]
            == 3
        )
    finally:
        conn.close()
    other = ctx.reserve("custom-export", Decimal("0.04"))
    with pytest.raises(BudgetExhausted, match="uncategorized headroom"):
        ctx.reserve("custom-export", Decimal("0.01"))
    for reservation in (*sparks, *reads, other):
        reservation.settle(Decimal("0.001"), "succeeded")
    spent = sum(
        (
            Decimal("0.001")
            for _ in (*sparks, *reads, other)
        ),
        Decimal("0"),
    )
    assert spent == Decimal("0.007")
    conn = db.connect(lane_db)
    try:
        total = sum(
            (
                Decimal(row["actual_usd"])
                for row in conn.execute(
                    "SELECT actual_usd FROM reservations WHERE job_id = ?"
                    " AND state = 'settled'",
                    (str(ctx.job_id),),
                ).fetchall()
            ),
            Decimal("0"),
        )
    finally:
        conn.close()
    assert total == Decimal("0.007")


def test_job_with_raised_spark_cap_funds_whole_video_runs(
    lane_db: Path,
) -> None:
    # A whole-video chunk child carries the configured window envelope:
    # 3 x the honest 214 s worst case ($0.1362/run) reserves where the
    # $0.26 default would refuse the second run; a further hold still
    # refuses against the raised cap, and a bad cap fails before any row.
    ctx = admit_job(
        "anon-wv", "ip-wv", "video-wv", internal=True,
        spark_cap_usd=Decimal("0.48675"),
    )
    share = Decimal("0.1362")
    holds = [ctx.reserve("spark_steps", share) for _ in range(3)]
    # One more $0.08 hold fits the $0.50 job cap ($0.4886) but breaks
    # the raised $0.48675 Spark reservation.
    with pytest.raises(BudgetExhausted, match="0.48675 spark reservation"):
        ctx.reserve("spark_steps", Decimal("0.08"))
    for hold in holds:
        hold.settle(Decimal("0.10"), "succeeded")
    with pytest.raises(ValueError, match="spark_cap_usd"):
        admit_job(
            "anon-bad", "ip-bad", "video-bad",
            spark_cap_usd=Decimal("0"),
        )


def test_live_judge_spend_is_never_authorized(lane_db: Path) -> None:
    ctx = admit_job("anon-judge", "ip-judge", "video-judge")
    for kind in ("judge", "judge-replay", "judge_live"):
        with pytest.raises(LiveJudgeSpendRefused):
            ctx.reserve(kind, Decimal("0.01"))
    assert _held_count(lane_db, ctx.job_id) == 0


def test_legacy_a8_kinds_are_still_refused(lane_db: Path) -> None:
    """The kind spellings from before the rename stay refused, under the old exception name too."""
    from app.ledger import A8SpendRefused, _reject_a8_kind

    assert A8SpendRefused is LiveJudgeSpendRefused
    ctx = admit_job("anon-a8", "ip-a8", "video-a8")
    for kind in ("a8", "A8", " a8 ", "a8_judge", "a8-judge", "a8_live"):
        with pytest.raises(A8SpendRefused):
            ctx.reserve(kind, Decimal("0.01"))
        with pytest.raises(LiveJudgeSpendRefused):
            _reject_a8_kind(kind)
    assert _held_count(lane_db, ctx.job_id) == 0
    for kind in ("a80", "a8x", "spark_a8"):  # only the exact legacy spellings are refused
        _reject_a8_kind(kind)


def test_artifact_store_scopes_paths_and_tombstones_idempotently(
    lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-art", "ip-art", "video-art")
    store = ctx.artifacts
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    with pytest.raises(ArtifactPathError):
        store.register("provisional", tmp_path / "elsewhere.png")
    with pytest.raises(ArtifactPathError):
        store.register("provisional", root.parent / "sibling.png")
    target = root / "nested" / "p.png"
    artifact = store.register("provisional", target)
    assert target.parent.is_dir()
    with pytest.raises(ArtifactNotFound):
        store.mark_kept(contracts.ArtifactId("nope"))
    with pytest.raises(ArtifactNotFound):
        store.mark_deleted(contracts.ArtifactId("nope"))
    # ENOENT on first delete is success; the second delete is a no-op.
    store.mark_deleted(artifact)
    store.mark_deleted(artifact)
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT delete_state FROM artifacts WHERE id = ?", (str(artifact),)
            ).fetchone()["delete_state"]
            == "tombstoned"
        )
    finally:
        conn.close()
    ctx.finish("done")
    with pytest.raises(JobAlreadyFinished):
        store.register("provisional", root / "late.png")


def test_concurrent_reserve_settle_cycles_keep_exact_books(lane_db: Path) -> None:
    first = admit_job("anon-mt1", "ip-mt1", "video-mt1")
    second = admit_job("anon-mt2", "ip-mt2", "video-mt2")
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []
    successes: dict[str, int] = {str(first.job_id): 0, str(second.job_id): 0}
    tally = threading.Lock()

    def hammer(ctx) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(12):
                try:
                    reservation = ctx.reserve("sam", Decimal("0.0025"))
                except BudgetExhausted:
                    return
                reservation.settle(Decimal("0.001"), "succeeded")
                with tally:
                    successes[str(ctx.job_id)] += 1
        except BaseException as error:  # noqa: BLE001 - re-raised in main thread
            errors.append(error)

    workers = [
        threading.Thread(target=hammer, args=(ctx,))
        for ctx in (first, second)
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert not errors
    for job_id, count in successes.items():
        assert 0 < count <= 32, (job_id, count)
    total = sum(successes.values())
    sam = get_budget_row("sam", path=lane_db)
    assert Decimal(sam["held_usd"]) == Decimal("0")
    assert Decimal(sam["spent_usd"]) == Decimal("0.001") * total


def test_reserve_fails_closed_on_write_contention(lane_db: Path) -> None:
    ctx = admit_job("anon-lock", "ip-lock", "video-lock")
    holder = db.connect(lane_db)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        "INSERT INTO events (job_id, anon_id, session_id, event, detail_json, created_at)"
        " VALUES (NULL, NULL, NULL, 'lock-holder', '{}', '2026-10-27T00:00:00.000000Z')"
    )
    barrier = threading.Barrier(2)
    outcome: list[str] = []

    def attempt() -> None:
        barrier.wait(timeout=10)
        try:
            ctx.reserve("sam", Decimal("0.0025"))
        except ContentionError:
            outcome.append("contention")
        except ReserveRefused:
            outcome.append("refused")

    worker = threading.Thread(target=attempt)
    worker.start()
    barrier.wait(timeout=10)
    worker.join(timeout=60)
    holder.execute("ROLLBACK")
    holder.close()
    assert outcome == ["contention"]
    assert _held_count(lane_db, ctx.job_id) == 0


def test_expire_overdue_jobs_frees_holds_without_running(lane_db: Path) -> None:
    past = db.format_dt(db.utcnow() - timedelta(seconds=1))
    expired_ctx = admit_job("anon-exp", "ip-exp", "video-exp")
    conn = db.connect(lane_db)
    try:
        conn.execute("UPDATE jobs SET deadline_at = ? WHERE id = ?", (past, str(expired_ctx.job_id)))
    finally:
        conn.close()
    assert expire_overdue_jobs(path=lane_db) == (expired_ctx.job_id,)
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT state FROM jobs WHERE id = ?", (str(expired_ctx.job_id),)
            ).fetchone()["state"]
            == "expired"
        )
    finally:
        conn.close()
    day_row = f"day:{db.utc_day(db.utcnow())}"
    for name in (day_row, AGGREGATE_ROW):
        assert Decimal(get_budget_row(name, path=lane_db)["held_usd"]) == Decimal("0")
    with pytest.raises(JobNotActive):
        expired_ctx.reserve("sam", Decimal("0.0025"))


def test_remaining_s_reads_the_job_deadline(lane_db: Path) -> None:
    ctx = admit_job("anon-rem", "ip-rem", "video-rem")
    assert 0 < ctx.remaining_s() <= contracts.JOB_DEADLINE_S
    past = db.format_dt(db.utcnow() - timedelta(seconds=30))
    conn = db.connect(lane_db)
    try:
        conn.execute("UPDATE jobs SET deadline_at = ? WHERE id = ?", (past, str(ctx.job_id)))
    finally:
        conn.close()
    assert ctx.remaining_s() <= -29
    from app.ledger import JobContextImpl

    with pytest.raises(LedgerError, match="unknown job"):
        JobContextImpl(lane_db, JobId("no-such-job"), lane_db.parent).remaining_s()


def test_evidence_seams_round_trip_call_records_and_events(lane_db: Path) -> None:
    from app.contracts import CallRecord

    ctx = admit_job("anon-ev", "ip-ev", "video-ev")
    record = CallRecord(
        job_id=ctx.job_id,
        stage="extract",
        model="spark",
        endpoint="runs",
        latency_ms=1200,
        units_in=10,
        units_out=4,
        estimated_cost_usd=Decimal("0.0866"),
        status="ok",
        evidence={"tokens": 14},
    )
    assert insert_call_record(record, path=lane_db) > 0
    (back,) = list_call_records(ctx.job_id, path=lane_db)
    assert back == record
    assert insert_event("note", {"k": "v"}, job_id=ctx.job_id, path=lane_db) > 0


def test_uncategorized_reservations_hold_day_and_aggregate(lane_db: Path) -> None:
    # Two $0.46 job holds under a $0.96 aggregate cap leave $0.04: the first
    # uncategorized $0.04 call fits, the second is refused before any spend.
    conn = db.connect(lane_db)
    try:
        conn.execute("UPDATE budget_rows SET hard_cap_usd = '0.96' WHERE name = 'aggregate'")
    finally:
        conn.close()
    first = admit_job("anon-u1", "ip-u1", "video-u1")
    second = admit_job("anon-u2", "ip-u2", "video-u2")
    day_row = f"day:{db.utc_day(db.utcnow())}"
    assert Decimal(get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"]) == Decimal("0.92")
    held = first.reserve("custom-export", Decimal("0.04"))
    for name in (day_row, AGGREGATE_ROW):
        assert Decimal(get_budget_row(name, path=lane_db)["held_usd"]) == Decimal("0.96")
    with pytest.raises(BudgetExhausted, match="aggregate ceiling"):
        second.reserve("custom-export", Decimal("0.04"))
    assert _held_count(lane_db, second.job_id) == 0
    # The refusal is atomic: the day hold it took first is rolled back too.
    assert Decimal(get_budget_row(day_row, path=lane_db)["held_usd"]) == Decimal("0.96")
    held.settle(Decimal("0.04"), "succeeded")
    for name in (day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["held_usd"]) == Decimal("0.92")
        assert Decimal(row["spent_usd"]) == Decimal("0.04")
    # Still over the ceiling afterwards: 0.92 held + 0.04 spent + 0.04 worst.
    with pytest.raises(BudgetExhausted, match="aggregate ceiling"):
        second.reserve("custom-export", Decimal("0.04"))


def test_settlement_refuses_overrun_and_records_failed_call(lane_db: Path) -> None:
    ctx = admit_job("anon-ov", "ip-ov", "video-ov")
    day_row = f"day:{db.utc_day(db.utcnow())}"
    thin = ctx.reserve("sam", Decimal("0.01"))
    later = ctx.reserve("sam", Decimal("0.01"))
    with pytest.raises(BudgetExhausted, match="exceeds reserved"):
        thin.settle(Decimal("1.00"), "succeeded")
    conn = db.connect(lane_db)
    try:
        entry = conn.execute(
            "SELECT state, actual_usd FROM reservations WHERE id = ?",
            (thin.reservation_id,),
        ).fetchone()
    finally:
        conn.close()
    assert (entry["state"], entry["actual_usd"]) == ("failed", "1.00")
    # The refusal converts the $0.01 hold into $1.00 of real spend on every
    # applicable row, even past the cap; only the second hold stays held.
    sam = get_budget_row("sam", path=lane_db)
    assert Decimal(sam["held_usd"]) == Decimal("0.01")
    assert Decimal(sam["spent_usd"]) == Decimal("1.00")
    for name in (day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["held_usd"]) == Decimal("0.45")
        assert Decimal(row["spent_usd"]) == Decimal("1.00")
    with pytest.raises(ReservationNotHeld):
        thin.settle(Decimal("0.01"), "succeeded")
    # The recorded $1.00 failure counts against the job: settling the second
    # $0.01 hold now passes the $0.50 job cap and is refused too — and that
    # refusal likewise lands its $0.01 as spend instead of vanishing it.
    with pytest.raises(BudgetExhausted, match="job cap"):
        later.settle(Decimal("0.01"), "succeeded")
    sam = get_budget_row("sam", path=lane_db)
    assert Decimal(sam["held_usd"]) == Decimal("0")
    assert Decimal(sam["spent_usd"]) == Decimal("1.01")
    for name in (day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["held_usd"]) == Decimal("0.44")
        assert Decimal(row["spent_usd"]) == Decimal("1.01")


def test_overrun_refusal_converts_hold_to_spend_on_every_row(lane_db: Path) -> None:
    ctx = admit_job("anon-ov2", "ip-ov2", "video-ov2")
    day_row = f"day:{db.utc_day(db.utcnow())}"
    # Lowered ceilings the $1.00 overrun must trip: the $0.01 reserve and
    # the $0.46 job hold still fit, the escaped $1.00 does not.
    conn = db.connect(lane_db)
    try:
        for name in ("sam", day_row, AGGREGATE_ROW):
            conn.execute(
                "UPDATE budget_rows SET hard_cap_usd = '1.00' WHERE name = ?", (name,)
            )
    finally:
        conn.close()
    thin = ctx.reserve("sam", Decimal("0.01"))
    with pytest.raises(BudgetExhausted, match="exceeds reserved"):
        thin.settle(Decimal("1.00"), "succeeded")
    for name in ("sam", day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["spent_usd"]) == Decimal("1.00"), name
    assert Decimal(get_budget_row("sam", path=lane_db)["held_usd"]) == Decimal("0")
    # The escaped spend trips every applicable breaker ...
    for name in ("sam", day_row, AGGREGATE_ROW):
        assert breaker_state(name, path=lane_db) == "tripped", name
    # ... and a NEW job no longer fits the $0.50 admission envelope.
    with pytest.raises(BudgetExhausted, match="admission refused"):
        admit_job("anon-ov3", "ip-ov3", "video-ov3")
    # The $1.00 of spend counts against the job: a later reserve is refused.
    with pytest.raises(BudgetExhausted):
        ctx.reserve("sam", Decimal("0.01"))
    ctx.finish("done")
    for name in ("sam", day_row, AGGREGATE_ROW):
        row = get_budget_row(name, path=lane_db)
        assert Decimal(row["spent_usd"]) == Decimal("1.00"), name
        assert Decimal(row["held_usd"]) == Decimal("0"), name
    with pytest.raises(JobNotActive):
        ctx.reserve("sam", Decimal("0.01"))


def test_spark_count_cap_refuses_fourth_authorization(lane_db: Path) -> None:
    ctx = admit_job("anon-sp", "ip-sp", "video-sp")
    held = [ctx.reserve("spark", Decimal("0.01")) for _ in range(3)]
    assert _held_count(lane_db, ctx.job_id) == 3
    invoked: list[str] = []
    with pytest.raises(BudgetExhausted, match="3 spark calls per job"):
        _call_paid(ctx, "spark", Decimal("0.01"), lambda: invoked.append("ran"))
    assert invoked == []
    assert _held_count(lane_db, ctx.job_id) == 3
    for reservation in held:
        reservation.settle(Decimal("0.01"), "succeeded")


def test_finish_queues_provisionals_for_sweeper_deletion(
    lane_db: Path, tmp_path: Path
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = admit_job("anon-pd", "ip-pd", "video-pd")
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    doomed = ctx.artifacts.register("provisional", root / "p1.png")
    (root / "p1.png").write_bytes(b"provisional")
    ctx.finish("done")
    conn = db.connect(lane_db)
    try:
        state = conn.execute(
            "SELECT delete_state, deleted_at, expires_at FROM artifacts WHERE id = ?",
            (str(doomed),),
        ).fetchone()
    finally:
        conn.close()
    assert (state["delete_state"], state["deleted_at"]) == ("pending_delete", None)
    assert (root / "p1.png").exists()
    now = db.utcnow()
    # No backdating: finish itself set the expiry to the finish time, so the
    # fresh provisional is due at once.
    assert db.parse_dt(state["expires_at"]) <= now
    assert {record.id for record in lifecycle.list_due_artifacts(now)} == {doomed}
    # The sweeper deletes bytes, then records the tombstone; only then gone.
    (root / "p1.png").unlink()
    lifecycle.record_artifact_deletion(doomed, "tombstoned")
    assert lifecycle.list_due_artifacts(now) == ()


def test_mark_deleted_db_failure_leaves_pending_row_and_retries(
    lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-md", "ip-md", "video-md")
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    target = root / "p.png"
    artifact = ctx.artifacts.register("provisional", target)
    target.write_bytes(b"bytes")
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "CREATE TRIGGER fail_tombstone BEFORE UPDATE ON artifacts"
            " WHEN NEW.delete_state = 'tombstoned'"
            " BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        ctx.artifacts.mark_deleted(artifact)
    conn = db.connect(lane_db)
    try:
        state = conn.execute(
            "SELECT delete_state FROM artifacts WHERE id = ?", (str(artifact),)
        ).fetchone()["delete_state"]
        conn.execute("DROP TRIGGER fail_tombstone")
    finally:
        conn.close()
    # The file is gone but the row is pending, never live; retry converges.
    assert not target.exists()
    assert state == "pending_delete"
    ctx.artifacts.mark_deleted(artifact)
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT delete_state FROM artifacts WHERE id = ?", (str(artifact),)
            ).fetchone()["delete_state"]
            == "tombstoned"
        )
    finally:
        conn.close()


def test_money_boundaries_require_finite_decimal(lane_db: Path) -> None:
    from app.contracts import CallRecord

    ctx = admit_job("anon-mo", "ip-mo", "video-mo")
    for bad in (1, True, 0.01, "0.01", None):
        with pytest.raises(TypeError, match="must be Decimal"):
            ctx.reserve("sam", bad)
    for bad in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(ValueError, match="must be finite"):
            ctx.reserve("sam", bad)
    with pytest.raises(ValueError, match="non-negative"):
        ctx.reserve("sam", Decimal("-0.01"))
    assert _held_count(lane_db, ctx.job_id) == 0
    held = ctx.reserve("sam", Decimal("0.0025"))
    for bad in (1, 0.001, "0.001"):
        with pytest.raises(TypeError, match="must be Decimal"):
            held.settle(bad, "succeeded")
    with pytest.raises(ValueError, match="must be finite"):
        held.settle(Decimal("NaN"), "succeeded")
    # Failed attempts never consume the hold: the honest settle still works.
    held.settle(Decimal("0.001"), "succeeded")
    for bad_cost in (1, 0.01, "0.01", Decimal("NaN")):
        with pytest.raises((TypeError, ValueError)):
            insert_call_record(
                CallRecord(
                    job_id=ctx.job_id,
                    stage="s",
                    model="m",
                    endpoint="e",
                    latency_ms=1,
                    units_in=None,
                    units_out=None,
                    estimated_cost_usd=bad_cost,
                    status="ok",
                    evidence={},
                ),
                path=lane_db,
            )


def test_expire_overdue_jobs_promotes_waiters_fifo(lane_db: Path) -> None:
    running = [admit_job(f"anon-p{i}", f"ip-p{i}", f"video-p{i}") for i in range(3)]
    conn = db.connect(lane_db)
    try:
        states = [
            row[0]
            for row in conn.execute(
                "SELECT state FROM jobs ORDER BY created_at, id"
            ).fetchall()
        ]
        assert states == ["active", "active", "waiting"]
        past = db.format_dt(db.utcnow() - timedelta(seconds=1))
        conn.execute(
            "UPDATE jobs SET deadline_at = ? WHERE id = ?", (past, str(running[0].job_id))
        )
    finally:
        conn.close()
    assert expire_overdue_jobs(path=lane_db) == (running[0].job_id,)
    conn = db.connect(lane_db)
    try:
        states_by_id = {
            row[0]: row[1] for row in conn.execute("SELECT id, state FROM jobs").fetchall()
        }
    finally:
        conn.close()
    assert states_by_id[str(running[0].job_id)] == "expired"
    assert states_by_id[str(running[2].job_id)] == "active"
    # The promoted waiter authorizes calls immediately.
    running[2].reserve("sam", Decimal("0.0025")).settle(Decimal("0.001"), "succeeded")


def test_deadline_equality_refuses_and_expires(
    lane_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(db, "utcnow", lambda: fixed)
    ctx = admit_job("anon-eq", "ip-eq", "video-eq")
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "UPDATE jobs SET deadline_at = ? WHERE id = ?",
            (db.format_dt(fixed), str(ctx.job_id)),
        )
    finally:
        conn.close()
    with pytest.raises(JobTimedOut):
        ctx.reserve("sam", Decimal("0.0025"))
    assert _held_count(lane_db, ctx.job_id) == 0
    assert expire_overdue_jobs(path=lane_db) == (ctx.job_id,)


def test_mark_kept_refuses_finished_jobs_and_deleted_rows(
    lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-mk", "ip-mk", "video-mk")
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    live = ctx.artifacts.register("provisional", root / "live.png")
    gone = ctx.artifacts.register("provisional", root / "gone.png")
    swept = ctx.artifacts.register("provisional", root / "swept.png")
    ctx.artifacts.mark_deleted(gone)
    with pytest.raises(LedgerError, match="not live"):
        ctx.artifacts.mark_kept(gone)
    # Keeping twice is still idempotent on a live row of an open job.
    ctx.artifacts.mark_kept(live)
    ctx.artifacts.mark_kept(live)
    ctx.finish("done")
    with pytest.raises(JobAlreadyFinished):
        ctx.artifacts.mark_kept(live)
    with pytest.raises(JobAlreadyFinished):
        ctx.artifacts.mark_kept(swept)
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT kept_at FROM artifacts WHERE id = ?", (str(swept),)
            ).fetchone()["kept_at"]
            is None
        )
    finally:
        conn.close()


def _receipt_record(job_id: JobId, **overrides):
    from app.contracts import CallRecord

    fields = {
        "job_id": job_id,
        "stage": "extract",
        "model": "spark-1.3",
        "endpoint": "/runs",
        "latency_ms": 1200,
        "units_in": 10,
        "units_out": 4,
        "estimated_cost_usd": Decimal("0.0866"),
        "status": "succeeded",
        "evidence": {"run_index": 0},
    }
    fields.update(overrides)
    return CallRecord(**fields)


def _receipt_counts(database: Path) -> tuple[int, int]:
    conn = db.connect(database)
    try:
        calls = conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()
    return calls, events


def test_record_call_with_event_exact_replay_is_idempotent(lane_db: Path) -> None:
    from app.evidence import call_key

    ctx = admit_job("anon-rc1", "ip-rc1", "video-rc1")
    record = _receipt_record(ctx.job_id)
    key = call_key(record)
    event = ("model.call_recorded", {"stage": "extract", "status": "succeeded"})
    ctx.record_call_with_event(key, record, event)
    ctx.record_call_with_event(key, record, event)
    assert _receipt_counts(lane_db) == (1, 1)
    (back,) = list_call_records(ctx.job_id, path=lane_db)
    assert back == record
    conn = db.connect(lane_db)
    try:
        detail = conn.execute(
            "SELECT job_id, event, detail_json FROM events WHERE id = ?", (key,)
        ).fetchone()
    finally:
        conn.close()
    assert detail["job_id"] == str(ctx.job_id)
    assert detail["event"] == "model.call_recorded"
    assert json.loads(detail["detail_json"]) == {
        "stage": "extract",
        "status": "succeeded",
    }


def test_record_call_with_event_conflicting_replay_raises(lane_db: Path) -> None:
    from app.evidence import call_key

    ctx = admit_job("anon-rc2", "ip-rc2", "video-rc2")
    record = _receipt_record(ctx.job_id)
    key = call_key(record)
    event = ("model.call_recorded", {"stage": "extract", "status": "succeeded"})
    ctx.record_call_with_event(key, record, event)
    # Same key, different call payload: the key no longer matches the record.
    with pytest.raises(ValueError, match="does not match"):
        ctx.record_call_with_event(
            key, _receipt_record(ctx.job_id, latency_ms=1201), event
        )
    # Same key and call, different event: the stored pair conflicts.
    with pytest.raises(ValueError, match="receipt key conflict"):
        ctx.record_call_with_event(
            key, record, ("model.call_failed", {"stage": "extract", "status": "failed"})
        )
    assert _receipt_counts(lane_db) == (1, 1)
    (back,) = list_call_records(ctx.job_id, path=lane_db)
    assert back.latency_ms == 1200


def test_record_call_with_event_invalid_call_or_event_is_rejected_and_writes_nothing(
    lane_db: Path,
) -> None:
    from app.evidence import call_key

    ctx = admit_job("anon-rc3", "ip-rc3", "video-rc3")
    other = admit_job("anon-rc3b", "ip-rc3b", "video-rc3b")
    good = _receipt_record(ctx.job_id)
    event = ("model.call_recorded", {"stage": "extract", "status": "succeeded"})
    bad_calls = [
        _receipt_record(ctx.job_id, stage="judge"),
        _receipt_record(ctx.job_id, model="gpt-4"),
        _receipt_record(ctx.job_id, endpoint="http://host/runs"),
        _receipt_record(ctx.job_id, status="ok"),
        _receipt_record(ctx.job_id, latency_ms=-1),
        _receipt_record(ctx.job_id, estimated_cost_usd=0.01),
        _receipt_record(other.job_id),
    ]
    for bad in bad_calls:
        with pytest.raises(ValueError):
            ctx.record_call_with_event("whatever", bad, event)
    bad_events = [
        ("gallery.click", {"stage": "extract"}),
        ("model.call_recorded", {f"key-{i}": i for i in range(33)}),
        ("model.call_recorded", "not-a-mapping"),
    ]
    for bad_event in bad_events:
        with pytest.raises(ValueError):
            ctx.record_call_with_event(call_key(good), good, bad_event)
    with pytest.raises(ValueError, match="does not match"):
        ctx.record_call_with_event("0" * 64, good, event)
    assert _receipt_counts(lane_db) == (0, 0)


def test_record_call_with_event_event_insert_failure_rolls_back_call(
    lane_db: Path,
) -> None:
    from app.evidence import call_key

    ctx = admit_job("anon-rc4", "ip-rc4", "video-rc4")
    record = _receipt_record(ctx.job_id)
    event = ("model.call_recorded", {"stage": "extract", "status": "succeeded"})
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "CREATE TRIGGER fail_receipt_event BEFORE INSERT ON events"
            " BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        ctx.record_call_with_event(call_key(record), record, event)
    assert _receipt_counts(lane_db) == (0, 0)
    conn = db.connect(lane_db)
    try:
        conn.execute("DROP TRIGGER fail_receipt_event")
    finally:
        conn.close()
    ctx.record_call_with_event(call_key(record), record, event)
    assert _receipt_counts(lane_db) == (1, 1)


def test_finish_done_atomically_keeps_complete_set_or_rolls_back_all_keeps(
    lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-kc", "ip-kc", "video-kc")
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    good = ctx.artifacts.register("provisional", root / "good.png")
    (root / "good.png").write_bytes(b"good")
    extra = ctx.artifacts.register("provisional", root / "extra.png")
    (root / "extra.png").write_bytes(b"extra")
    foreign_ctx = admit_job("anon-kc2", "ip-kc2", "video-kc2")
    foreign_root = tmp_path / "data" / "jobs" / str(foreign_ctx.job_id)
    foreign = foreign_ctx.artifacts.register("provisional", foreign_root / "f.png")
    (foreign_root / "f.png").write_bytes(b"foreign")
    day_row = f"day:{db.utc_day(db.utcnow())}"
    held_before = get_budget_row(day_row, path=lane_db)["held_usd"]

    def _artifact_state(artifact_id) -> tuple:
        conn = db.connect(lane_db)
        try:
            return tuple(
                conn.execute(
                    "SELECT delete_state, kept_at FROM artifacts WHERE id = ?",
                    (str(artifact_id),),
                ).fetchone()
            )
        finally:
            conn.close()

    # Unknown, foreign, and non-provisional ids each fail closed: the job
    # stays active, no keep lands, nothing tombstones, holds stay held.
    final = ctx.artifacts.register("final", root / "guide.json")
    for bad_set in ([good, ArtifactId("no-such-id")], [good, foreign], [final]):
        with pytest.raises((ArtifactNotFound, LedgerError)):
            ctx.finish("done", keep_artifact_ids=bad_set)
    conn = db.connect(lane_db)
    try:
        job_state = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
        ).fetchone()["state"]
    finally:
        conn.close()
    assert job_state == "active"
    assert _artifact_state(good) == ("live", None)
    assert _artifact_state(extra) == ("live", None)
    assert get_budget_row(day_row, path=lane_db)["held_usd"] == held_before
    # A keep id whose file is gone is refused the same way.
    (root / "good.png").unlink()
    with pytest.raises(LedgerError, match="no regular file"):
        ctx.finish("done", keep_artifact_ids=[good])
    assert _artifact_state(good) == ("live", None)
    # The exact set then commits: kept stays live, the other tombstones.
    (root / "good.png").write_bytes(b"good")
    ctx.finish("done", keep_artifact_ids=[good, good])
    assert _artifact_state(good)[0] == "live"
    assert _artifact_state(good)[1] is not None
    assert _artifact_state(extra) == ("pending_delete", None)


def test_finish_failed_clears_prior_provisional_keeps(
    lane_db: Path, tmp_path: Path
) -> None:
    lifecycle = LifecycleRepositoryImpl(lane_db)
    ctx = admit_job("anon-kf", "ip-kf", "video-kf")
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    first = ctx.artifacts.register("provisional", root / "a.png")
    (root / "a.png").write_bytes(b"a")
    second = ctx.artifacts.register("provisional", root / "b.png")
    (root / "b.png").write_bytes(b"b")
    ctx.artifacts.mark_kept(first)
    ctx.artifacts.mark_kept(second)
    with pytest.raises(ValueError, match="keeps nothing"):
        ctx.finish("failed", keep_artifact_ids=[first])
    ctx.finish("failed")
    conn = db.connect(lane_db)
    try:
        states = {
            row["id"]: (row["delete_state"], row["kept_at"])
            for row in conn.execute(
                "SELECT id, delete_state, kept_at FROM artifacts WHERE job_id = ?",
                (str(ctx.job_id),),
            ).fetchall()
        }
    finally:
        conn.close()
    assert states == {
        str(first): ("pending_delete", None),
        str(second): ("pending_delete", None),
    }
    # Queued for the sweeper at once; bytes stay until it deletes them.
    now = db.utcnow()
    assert {record.id for record in lifecycle.list_due_artifacts(now)} == {first, second}
    assert (root / "a.png").exists()


def test_finish_signal_mid_transaction_rolls_back_and_leaves_job_active(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.ledger as ledger_module

    ctx = admit_job("anon-crash", "ip-crash", "video-crash")
    root = tmp_path / "data" / "jobs" / str(ctx.job_id)
    kept = ctx.artifacts.register("provisional", root / "k.png")
    (root / "k.png").write_bytes(b"k")
    doomed = ctx.artifacts.register("provisional", root / "d.png")
    (root / "d.png").write_bytes(b"d")
    day_row = f"day:{db.utc_day(db.utcnow())}"

    def _snapshot() -> tuple:
        conn = db.connect(lane_db)
        try:
            job_state = conn.execute(
                "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
            ).fetchone()["state"]
            arts = tuple(
                (row["id"], row["delete_state"], row["kept_at"])
                for row in conn.execute(
                    "SELECT id, delete_state, kept_at FROM artifacts WHERE job_id = ?"
                    " ORDER BY id",
                    (str(ctx.job_id),),
                ).fetchall()
            )
        finally:
            conn.close()
        held = get_budget_row(day_row, path=lane_db)["held_usd"]
        return job_state, arts, held

    before = _snapshot()
    assert before[0] == "active"
    # Crash 1: a statement failure mid-transaction rolls everything back.
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "CREATE TRIGGER fail_finish_job BEFORE UPDATE ON jobs"
            " WHEN NEW.state = 'done'"
            " BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        ctx.finish("done", keep_artifact_ids=[kept])
    assert _snapshot() == before
    conn = db.connect(lane_db)
    try:
        conn.execute("DROP TRIGGER fail_finish_job")
    finally:
        conn.close()
    # Crash 2: a signal (KeyboardInterrupt) between the keep writes and the
    # commit propagates with the same nothing-committed guarantee.
    def _interrupted(conn_inner, job_id_inner) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        ledger_module, "_release_unsettled_calls", _interrupted
    )
    with pytest.raises(KeyboardInterrupt):
        ctx.finish("done", keep_artifact_ids=[kept])
    monkeypatch.undo()
    assert _snapshot() == before
    # The still-active job then finishes normally.
    ctx.finish("done", keep_artifact_ids=[kept])
    conn = db.connect(lane_db)
    try:
        assert (
            conn.execute(
                "SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)
            ).fetchone()["state"]
            == "done"
        )
    finally:
        conn.close()
