"""Ledger admission: queue slots, deadline, rate fields, atomic money headroom."""

from __future__ import annotations

import sqlite3
import threading
from decimal import Decimal
from pathlib import Path

import pytest

from app import db
from app.admission import (
    QUEUE_FULL_MESSAGE,
    QueueFullError,
    RateLimitedError,
    admit_job,
    job_dir,
)
from app.ledger import (
    AGGREGATE_ROW,
    BudgetExhausted,
    JobTimedOut,
    ReserveRefused,
    get_budget_row,
)


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


def _states(database: Path) -> list[str]:
    conn = db.connect(database)
    try:
        return [
            row[0]
            for row in conn.execute("SELECT state FROM jobs ORDER BY created_at, id").fetchall()
        ]
    finally:
        conn.close()


def test_queue_two_running_ten_waiting_thirteenth_refused(lane_db: Path) -> None:
    contexts = [
        admit_job(f"anon-{i}", f"ip-{i}", f"video-{i}") for i in range(12)
    ]
    assert _states(lane_db) == ["active"] * 2 + ["waiting"] * 10
    with pytest.raises(QueueFullError, match=r"^job queue full: 2 running, 10 waiting$"):
        admit_job("anon-12", "ip-12", "video-12")
    assert str(QueueFullError(QUEUE_FULL_MESSAGE)) == QUEUE_FULL_MESSAGE

    # Finishing a running job promotes the oldest waiting job, FIFO.
    contexts[0].finish("done")
    assert _states(lane_db) == ["done", "active", "active"] + ["waiting"] * 9
    conn = db.connect(lane_db)
    try:
        promoted = conn.execute(
            "SELECT id FROM jobs WHERE state = 'active' ORDER BY created_at, id"
        ).fetchall()
    finally:
        conn.close()
    assert str(contexts[2].job_id) in {row[0] for row in promoted}


def test_two_simultaneous_max_jobs_cannot_overreserve(lane_db: Path) -> None:
    # An aggregate ceiling of $0.50 fits exactly one $0.46 job hold: two
    # barrier-released admissions must leave one winner, never two holds.
    conn = db.connect(lane_db)
    try:
        conn.execute("UPDATE budget_rows SET hard_cap_usd = '0.50' WHERE name = 'aggregate'")
    finally:
        conn.close()
    barrier = threading.Barrier(2)
    outcomes: dict[int, str] = {}

    def attempt(slot: int) -> None:
        barrier.wait(timeout=10)
        try:
            admit_job(f"racer-{slot}", f"racer-ip-{slot}", f"racer-video-{slot}")
        except BudgetExhausted:
            outcomes[slot] = "refused"
        else:
            outcomes[slot] = "admitted"

    workers = [threading.Thread(target=attempt, args=(slot,)) for slot in (0, 1)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert sorted(outcomes.values()) == ["admitted", "refused"]
    assert get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"] == "0.46"
    conn = db.connect(lane_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        conn.close()


def test_timeout_makes_zero_call_authorizations(lane_db: Path) -> None:
    ctx = admit_job("anon-t", "ip-t", "video-t")
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "UPDATE jobs SET deadline_at = '2020-01-01T00:00:00.000000Z' WHERE id = ?",
            (str(ctx.job_id),),
        )
    finally:
        conn.close()
    with pytest.raises(JobTimedOut):
        ctx.reserve("sam", Decimal("0.0025"))
    with pytest.raises(ReserveRefused):
        ctx.reserve("spark", Decimal("0.01"))
    conn = db.connect(lane_db)
    try:
        calls = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE job_id = ? AND state = 'held'",
            (str(ctx.job_id),),
        ).fetchone()[0]
        assert calls == 0
    finally:
        conn.close()


def test_best_effort_rate_limit_refuses_key_flood(lane_db: Path) -> None:
    for i in range(10):
        admit_job("flood-anon", f"flood-ip-{i}", f"flood-video-{i}")
    with pytest.raises(RateLimitedError):
        admit_job("flood-anon", "flood-ip-x", "flood-video-x")
    # A different key still admits while queue slots remain.
    admit_job("other-anon", "other-ip", "other-video")
    assert len(_states(lane_db)) == 11


def test_admission_rolls_back_job_and_hold_when_directory_fails(tmp_path: Path) -> None:
    database = tmp_path / "lane.sqlite3"
    blocker = tmp_path / "data-root-file"
    blocker.write_text("not a directory")
    db.configure(database, blocker)
    db.init_db(database)
    with pytest.raises(NotADirectoryError):
        admit_job("anon-fs", "ip-fs", "video-fs")
    conn = db.connect(database)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
        day_row = f"day:{db.utc_day(db.utcnow())}"
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM budget_rows WHERE name = ?", (day_row,)
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()
    assert Decimal(get_budget_row(AGGREGATE_ROW, path=database)["held_usd"]) == Decimal("0")


def test_admission_rollback_removes_created_job_directory(
    lane_db: Path, tmp_path: Path
) -> None:
    jobs_root = tmp_path / "data" / "jobs"
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "CREATE TRIGGER reject_jobs BEFORE INSERT ON jobs"
            " BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError, match="boom"):
        admit_job("anon-rb", "ip-rb", "video-rb")
    conn = db.connect(lane_db)
    try:
        conn.execute("DROP TRIGGER reject_jobs")
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    finally:
        conn.close()
    assert Decimal(get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"]) == Decimal("0")
    # The database rolled back and the empty directory this call created is
    # gone too: no stranded job folder survives the failed admission.
    entries = list(jobs_root.iterdir()) if jobs_root.exists() else []
    assert entries == []


def test_admission_interrupt_after_mkdir_rolls_back_and_removes_directory(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_root = tmp_path / "data" / "jobs"
    real_mkdir = Path.mkdir

    def mkdir_then_interrupt(self: Path, *args: object, **kwargs: object) -> None:
        # mkdir(parents=True) recurses through Path.mkdir for missing
        # parents, so only the leaf (a direct child of jobs/) trips this:
        # the real job directory exists when the interrupt lands.
        real_mkdir(self, *args, **kwargs)
        if self.parent.name == "jobs":
            raise KeyboardInterrupt("injected after mkdir")

    monkeypatch.setattr(Path, "mkdir", mkdir_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected after mkdir"):
        admit_job("anon-ki", "ip-ki", "video-ki")
    conn = db.connect(lane_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    finally:
        conn.close()
    assert Decimal(get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"]) == Decimal("0")
    # The real mkdir ran, then the interrupt: the owned directory is gone.
    entries = list(jobs_root.iterdir()) if jobs_root.exists() else []
    assert entries == []


def test_admission_interrupt_after_commit_keeps_committed_state(
    lane_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = db.connect

    class CommitThenInterrupt:
        """Proxy that performs the real COMMIT, then injects KeyboardInterrupt."""

        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            result = self._conn.execute(sql, *args)
            if isinstance(sql, str) and sql.strip().upper() == "COMMIT":
                raise KeyboardInterrupt("injected after commit")
            return result

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    def connect_then_interrupt(path: Path | str | None = None) -> CommitThenInterrupt:
        return CommitThenInterrupt(real_connect(path))

    monkeypatch.setattr(db, "connect", connect_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected after commit"):
        admit_job("anon-ka", "ip-ka", "video-ka")
    conn = real_connect(lane_db)
    try:
        jobs = conn.execute("SELECT id, state FROM jobs").fetchall()
        holds = conn.execute(
            "SELECT estimated_usd FROM reservations WHERE state = 'job_hold'"
        ).fetchall()
    finally:
        conn.close()
    # The commit landed before the interrupt: one job, its $0.46 hold, and
    # the directory all survive, and the interrupt itself propagates.
    assert [(job["state"]) for job in jobs] == ["active"]
    assert [hold["estimated_usd"] for hold in holds] == ["0.46"]
    assert job_dir(jobs[0]["id"]).is_dir()
    assert get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"] == "0.46"


def test_admission_rollback_interrupt_preserves_original_error(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = db.connect

    class FailInsertThenRollback:
        """Proxy that fails after the job insert, then interrupts ROLLBACK."""

        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            if isinstance(sql, str) and sql.strip().upper() == "ROLLBACK":
                raise KeyboardInterrupt("second signal during rollback")
            result = self._conn.execute(sql, *args)
            if (
                isinstance(sql, str)
                and sql.strip().upper().startswith("INSERT INTO JOBS")
            ):
                raise RuntimeError("original admission failure")
            return result

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    def connect_with_faults(path: Path | str | None = None) -> FailInsertThenRollback:
        return FailInsertThenRollback(real_connect(path))

    monkeypatch.setattr(db, "connect", connect_with_faults)
    with pytest.raises(RuntimeError, match="original admission failure") as excinfo:
        admit_job("anon-ro", "ip-ro", "video-ro")
    # The rollback KeyboardInterrupt is swallowed, never attached: nothing
    # touches the original exception object, so it escapes unchanged.
    assert not hasattr(excinfo.value, "__notes__")
    conn = real_connect(lane_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    finally:
        conn.close()
    assert Decimal(get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"]) == Decimal("0")
    jobs_root = tmp_path / "data" / "jobs"
    entries = list(jobs_root.iterdir()) if jobs_root.exists() else []
    assert entries == []


def test_admission_hostile_note_error_still_cleans_up(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = db.connect

    class HostileNoteError(RuntimeError):
        def add_note(self, note: str) -> None:
            raise LookupError("note attachment failed")

    class FailInsertThenRollback:
        """Proxy that raises a hostile error, then interrupts ROLLBACK."""

        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            if isinstance(sql, str) and sql.strip().upper() == "ROLLBACK":
                raise KeyboardInterrupt("second signal during rollback")
            result = self._conn.execute(sql, *args)
            if (
                isinstance(sql, str)
                and sql.strip().upper().startswith("INSERT INTO JOBS")
            ):
                raise HostileNoteError("hostile original failure")
            return result

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    def connect_with_faults(path: Path | str | None = None) -> FailInsertThenRollback:
        return FailInsertThenRollback(real_connect(path))

    monkeypatch.setattr(db, "connect", connect_with_faults)
    # The handler never calls add_note on the original, so the hostile
    # LookupError cannot escape and the job directory is still removed.
    with pytest.raises(HostileNoteError, match="hostile original failure"):
        admit_job("anon-hn", "ip-hn", "video-hn")
    conn = real_connect(lane_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    finally:
        conn.close()
    assert Decimal(get_budget_row(AGGREGATE_ROW, path=lane_db)["held_usd"]) == Decimal("0")
    jobs_root = tmp_path / "data" / "jobs"
    entries = list(jobs_root.iterdir()) if jobs_root.exists() else []
    assert entries == []


def test_admission_records_deadline_cap_and_job_directory(lane_db: Path, tmp_path: Path) -> None:
    ctx = admit_job("anon-d", "ip-d", "video-d")
    assert len(str(ctx.job_id)) == 32
    directory = job_dir(str(ctx.job_id))
    assert directory.is_dir()
    assert directory.parent == tmp_path / "data" / "jobs"
    conn = db.connect(lane_db)
    try:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (str(ctx.job_id),)).fetchone()
    finally:
        conn.close()
    assert job["state"] == "active"
    assert job["cost_cap_usd"] == "0.50"
    created = db.parse_dt(job["created_at"])
    deadline = db.parse_dt(job["deadline_at"])
    assert (deadline - created).total_seconds() == 240
    assert (job["anon_id"], job["ip_hash"], job["video_hash"]) == ("anon-d", "ip-d", "video-d")
