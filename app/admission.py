"""Job admission: two running, ten waiting, fail-closed money headroom.

Each admission runs one short ``BEGIN IMMEDIATE`` transaction that expires
overdue open jobs, applies the best-effort anon/IP/video-hash rate limit,
takes a queue slot, checks $0.50 of day/aggregate headroom, creates the job
directory, and records the $0.46 job hold. The returned :class:`JobContext`
is the job's sole path to paid-call authorization.
"""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from app import db
from app.contracts import JOB_COST_CAP_USD, JOB_DEADLINE_S, JobId
from app.settings import DEFAULT_CORRECTIONS_FILENAME, DEFAULT_GUIDE_FILENAME
from app.ledger import (
    AGGREGATE_ROW,
    JOB_HEADROOM_CHECK_USD,
    JOB_HOLD_USD,
    BudgetExhausted,
    ContentionError,
    JobContextImpl,
    LedgerError,
    _expire_overdue,
    _refresh_breaker,
    _require_spark_cap_usd,
    ensure_day_row,
)

MAX_RUNNING = 2
MAX_WAITING = 10
QUEUE_FULL_MESSAGE = "job queue full: 2 running, 10 waiting"

RATE_WINDOW_S = 60
RATE_MAX_PER_KEY = 10


class QueueFullError(LedgerError):
    """A 13th concurrent job found 2 running and 10 waiting."""


class RateLimitedError(LedgerError):
    """Best-effort anon/IP/video-hash rate limit refused an admission."""


def job_dir(job_id: str, *, root: Path | str | None = None) -> Path:
    """Return the random 128-bit job directory for one job id."""
    base = Path(root) if root is not None else db.data_root()
    return base / "jobs" / job_id


def _check_rate_limit(conn: sqlite3.Connection, window_start: str, anon_id: str, ip_hash: str, video_hash: str) -> None:
    for column, value, label in (
        ("anon_id", anon_id, "anon"),
        ("ip_hash", ip_hash, "ip"),
        ("video_hash", video_hash, "video"),
    ):
        recent = conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE {column} = ? AND created_at > ?",
            (value, window_start),
        ).fetchone()[0]
        if recent >= RATE_MAX_PER_KEY:
            raise RateLimitedError(
                f"best-effort {label} rate limit: {recent} admissions in {RATE_WINDOW_S}s"
            )


def admit_job(
    anon_id: str,
    ip_hash: str,
    video_hash: str,
    *,
    internal: bool = False,
    spark_cap_usd: Decimal | None = None,
    admission_check: Callable[[sqlite3.Connection], None] | None = None,
    guide_filename: str = DEFAULT_GUIDE_FILENAME,
    corrections_filename: str = DEFAULT_CORRECTIONS_FILENAME,
) -> JobContextImpl:
    """Admit one job and return its ledger-module context.

    Raises :class:`QueueFullError` when 2 jobs run and 10 wait,
    :class:`RateLimitedError` on best-effort key flood, or
    :class:`BudgetExhausted` when the day/aggregate ceilings cannot fit
    another $0.50 job envelope. An ``internal`` child (one chunk
    sub-job of an already-admitted parent) runs inside its parent's
    slot: it is admitted active without queueing, but keeps the rate
    and budget checks and its own job hold, so a chunk that cannot fit
    still raises :class:`BudgetExhausted`. ``spark_cap_usd`` overrides
    the job's Spark reservation (whole-video chunk children carry the
    configured window envelope); without it the $0.26 default applies.
    The job row keeps its resolved guide and corrections file names so a
    later failed-job rescan finds this job's own outputs.
    """
    for value, label in ((anon_id, "anon_id"), (ip_hash, "ip_hash"), (video_hash, "video_hash")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
    _require_spark_cap_usd(spark_cap_usd)
    database = db.db_path()
    root = db.data_root()
    conn = db.connect(database)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
            created_dir: Path | None = None
            try:
                now = db.utcnow()
                now_text = db.format_dt(now)
                _expire_overdue(conn, now_text)
                _check_rate_limit(
                    conn,
                    db.format_dt(now - timedelta(seconds=RATE_WINDOW_S)),
                    anon_id,
                    ip_hash,
                    video_hash,
                )
                running = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state = 'active'"
                ).fetchone()[0]
                waiting = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state = 'waiting'"
                ).fetchone()[0]
                if internal or running < MAX_RUNNING:
                    state = "active"
                elif waiting < MAX_WAITING:
                    state = "waiting"
                else:
                    raise QueueFullError(QUEUE_FULL_MESSAGE)
                day_row = ensure_day_row(conn, db.utc_day(now))
                for name in (day_row, AGGREGATE_ROW):
                    ceiling = conn.execute(
                        "SELECT hard_cap_usd, spent_usd, held_usd FROM budget_rows"
                        " WHERE name = ?",
                        (name,),
                    ).fetchone()
                    if ceiling is None:  # pragma: no cover - ensured/seeded
                        raise LedgerError(f"missing budget row {name!r}")
                    if (
                        Decimal(ceiling["held_usd"])
                        + Decimal(ceiling["spent_usd"])
                        + JOB_HEADROOM_CHECK_USD
                        > Decimal(ceiling["hard_cap_usd"])
                    ):
                        raise BudgetExhausted(
                            f"admission refused: {name} cannot fit another"
                            f" ${JOB_HEADROOM_CHECK_USD} job envelope"
                        )
                if admission_check is not None:
                    admission_check(conn)
                job_id = secrets.token_hex(16)
                # The directory is created before the commit, while every
                # ledger row is still rollbackable: a filesystem failure here
                # rolls back the job and its hold instead of stranding them.
                # Only a directory this call created is removed on rollback.
                directory = job_dir(job_id, root=root)
                if not directory.exists():
                    # Marked cleanup-owned BEFORE mkdir: arming after
                    # mkdir stranded the directory on an interrupt between
                    # the two. A failed mkdir leaves nothing; rmdir below
                    # then fails closed as a swallowed OSError.
                    created_dir = directory
                directory.mkdir(parents=True, exist_ok=True)
                deadline = db.format_dt(now + timedelta(seconds=JOB_DEADLINE_S))
                conn.execute(
                    "INSERT INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at,"
                    " deadline_at, confirmed_at, completed_at, cost_cap_usd, readback_fired_n,"
                    " guide_filename, corrections_filename)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 0, ?, ?)",
                    (
                        job_id,
                        state,
                        anon_id,
                        ip_hash,
                        video_hash,
                        now_text,
                        deadline,
                        str(JOB_COST_CAP_USD),
                        guide_filename,
                        corrections_filename,
                    ),
                )
                conn.execute(
                    "INSERT INTO reservations (id, job_id, budget_row, call_kind,"
                    " estimated_usd, actual_usd, state, created_at, settled_at)"
                    " VALUES (?, ?, ?, 'job', ?, NULL, 'job_hold', ?, NULL)",
                    (
                        secrets.token_hex(16),
                        job_id,
                        AGGREGATE_ROW,
                        str(JOB_HOLD_USD),
                        now_text,
                    ),
                )
                for name in (day_row, AGGREGATE_ROW):
                    conn.execute(
                        "UPDATE budget_rows SET held_usd = ? WHERE name = ?",
                        (
                            str(
                                Decimal(
                                    conn.execute(
                                        "SELECT held_usd FROM budget_rows WHERE name = ?",
                                        (name,),
                                    ).fetchone()["held_usd"]
                                )
                                + JOB_HOLD_USD
                            ),
                            name,
                        ),
                    )
                    _refresh_breaker(conn, name)
                conn.execute("COMMIT")
            except BaseException:
                # BaseException so an interrupt after mkdir still rolls
                # back the rows and removes the owned directory below.
                # Only while a transaction is active: after COMMIT there
                # is nothing to roll back, and the committed state stays.
                # The exception object itself is never touched: even
                # add_note/repr on it could raise and replace the cause.
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK")
                    except BaseException:
                        pass
                    finally:
                        if created_dir is not None:
                            # Non-recursive empty-directory removal only: a
                            # failure (already gone, non-empty, raced, or an
                            # interrupt) never masks the original cause.
                            try:
                                created_dir.rmdir()
                            except BaseException:
                                pass
                raise
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise ContentionError("admit lost a write race; retry") from error
            raise
    finally:
        conn.close()
    return JobContextImpl(
        database, JobId(job_id), job_dir(job_id, root=root),
        spark_cap_usd=spark_cap_usd,
    )
