"""Atomic reserve-before-spend ledger.

No paid call may start unless its job, budget row, daily breaker, and
aggregate cap are atomically reserved. Every paid-call owner follows one
protocol::

    reservation = ctx.reserve(kind, worst_case_usd)  # raises before any call
    try:
        result = client.invoke(...)                  # client runs only if held
    except Exception:
        reservation.settle(actual_usd, "failed")
        raise
    reservation.settle(actual_usd, "succeeded")

Admission takes a $0.46 job hold against the day and aggregate rows and
requires $0.50 of headroom. Each categorized per-call reservation draws from
that envelope and takes its own hold on its category row; each uncategorized
call holds its own worst case on the day/aggregate rows. Every call settles
actual cost afterwards and unfinished holds are released by ``finish``.
Refusal always raises before the client callback can run, which is what lets
callers degrade (second frames, then labels and read-backs to ``Not checked``,
then return completed work).
"""

from __future__ import annotations

import json
import contextlib
import logging
import math
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence

from app import db
from app.retention_config import guide_lifetime_kinds
from app import evidence as _l5evidence

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def active_job_publication(job_id: JobId | str) -> Iterator[bool]:
    """Hold the job writer lock while a canonical output becomes visible."""
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT j.state, j.deadline_at, u.cancel_requested_at"
                " FROM jobs j LEFT JOIN uploads u ON u.job_id = j.id WHERE j.id = ?",
                (str(job_id),),
            ).fetchone()
            active = (
                row is not None and row["state"] == "active"
                and row["deadline_at"] > db.format_dt(db.utcnow())
                and row["cancel_requested_at"] is None
            )
            yield active
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
from app.contracts import (
    AGGREGATE_SPEND_CEILING_USD,
    GUIDE_CLIP_ARTIFACT_KIND,
    JOB_COST_CAP_USD,
    JOB_DEADLINE_S,
    JOB_RESERVATION_TOTAL_USD,
    MAX_READBACKS_PER_JOB,
    SAM_MAX_CALLS_PER_JOB,
    ArtifactId,
    ArtifactRecord,
    CallRecord,
    JobId,
    JSONValue,
    MetaFileId,
    MetaFileRecord,
    SPARK_RUNS_PER_GUIDE,
)

JOB_HOLD_USD = JOB_RESERVATION_TOTAL_USD
JOB_HEADROOM_CHECK_USD = JOB_COST_CAP_USD

CATEGORY_CAP_USD = {
    "spark": Decimal("0.26"),
    "sam": Decimal("0.08"),
    "readback": Decimal("0.12"),
}
CATEGORY_COUNT_CAP = {
    "spark": SPARK_RUNS_PER_GUIDE,
    "sam": SAM_MAX_CALLS_PER_JOB,
    "readback": MAX_READBACKS_PER_JOB,
}
OTHER_CATEGORY_SPEND_CAP_USD = Decimal("0.04")

AGGREGATE_ROW = "aggregate"
DAY_ROW_PREFIX = "day:"
DAY_SPEND_CAP_USD = Decimal("100")
KIND_ROW_CAP_USD = Decimal("100")

ARTIFACT_RETENTION_DAYS = 7

# Artifact kinds the finish gate treats as provisional renders. The ledger module's own
# rows use 'provisional'; the guide module's renderer registers 'provisional_render'.
# Both literals are pinned by committed tests, so the gate accepts both
# instead of stranding real renders outside every keep set.
PROVISIONAL_ARTIFACT_KINDS = frozenset({"provisional", "provisional_render"})

WARN_FRACTION = Decimal("0.8")

TERMINAL_JOB_STATES = ("done", "failed", "expired")
OPEN_JOB_STATES = ("waiting", "active")


class LedgerError(RuntimeError):
    """Base error for admission/ledger failures (all fail closed)."""


class ReserveRefused(LedgerError):
    """A call authorization was refused; the client must not be invoked."""


class BudgetExhausted(ReserveRefused):
    """A job or call does not fit the row/day/aggregate ceilings."""


class JobNotActive(ReserveRefused):
    """A reservation was attempted on a non-running job."""


class JobTimedOut(ReserveRefused):
    """A reservation was attempted after the job deadline."""


class LiveJudgeSpendRefused(ReserveRefused):
    """Spend on the replay-only sample is never authorized."""


class JobAlreadyFinished(LedgerError):
    """``finish`` was called on a job that already left ``active``."""


class ReservationNotHeld(LedgerError):
    """``settle`` was called on an unknown or already-settled reservation."""


class ArtifactNotFound(LedgerError):
    """An artifact id is unknown to this job."""


class ArtifactPathError(LedgerError):
    """An artifact path escapes its job directory."""


class MetaFileNotFound(LedgerError):
    """A meta-file id is unknown."""


class ContentionError(LedgerError):
    """A short reservation transaction lost a write race; caller retries."""


def _locked(error: sqlite3.OperationalError) -> bool:
    return "locked" in str(error).lower() or "busy" in str(error).lower()


def _require_spark_cap_usd(value: Decimal | None) -> Decimal:
    """Return the job's Spark reservation: the override, else the default.

    Anything explicitly passed must be a finite positive Decimal; the
    check runs before admission opens its transaction, so a bad value
    never leaves a job row behind.
    """
    if value is None:
        return CATEGORY_CAP_USD["spark"]
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or not value > 0
    ):
        raise ValueError(
            "spark_cap_usd must be a finite positive Decimal,"
            f" got {value!r}"
        )
    return value


def category_of(kind: str) -> str:
    """Map a call kind to its spend category (``other`` has no row)."""
    normalized = kind.strip().lower().replace("-", "_")
    if (
        normalized == "spark_readback"
        or normalized.startswith("readback")
        or normalized.startswith("read_back")
    ):
        # The read-back call kind starts with "spark" but spends the
        # read-back reservation ($0.12, 3 calls); matching it as spark
        # would refuse every read-back after the three extraction runs.
        return "readback"
    if normalized.startswith("spark"):
        return "spark"
    if normalized.startswith("sam") or normalized.startswith("segment"):
        return "sam"
    return "other"


# Kind spellings from before the rename stay refused: "a8" and "a8_*" named the
# replay-only sample's kinds, and callers may still send them.
_LEGACY_REFUSED_KIND = "a8"


def _reject_live_judge_kind(kind: str) -> None:
    normalized = kind.strip().lower().replace("-", "_")
    if (
        normalized.startswith("judge")
        or normalized == _LEGACY_REFUSED_KIND
        or normalized.startswith(_LEGACY_REFUSED_KIND + "_")
    ):
        raise LiveJudgeSpendRefused(f"kind {kind!r} targets the replay-only sample; its spend is never authorized")


# Former names, kept so existing imports keep working.
A8SpendRefused = LiveJudgeSpendRefused
_reject_a8_kind = _reject_live_judge_kind


def _money(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"money must be Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise ValueError(f"money must be finite, got {value}")
    if value < 0:
        raise ValueError(f"money must be non-negative, got {value}")
    return value


def _row(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM budget_rows WHERE name = ?", (name,)).fetchone()


def _refresh_breaker(conn: sqlite3.Connection, name: str) -> None:
    """Recompute the 80%/100% breaker level for one budget row.

    ``breaker_usd`` holds the highest crossed threshold: NULL below 80%,
    80% of the cap at warn, the full cap once tripped. Holds release, so the
    level is recomputed live on every mutation; spent money never un-spends.
    """
    current = _row(conn, name)
    if current is None:
        return
    cap = Decimal(current["hard_cap_usd"])
    total = Decimal(current["held_usd"]) + Decimal(current["spent_usd"])
    if cap <= 0 or total >= cap:
        level: Decimal | None = cap
    elif total >= cap * WARN_FRACTION:
        level = cap * WARN_FRACTION
    else:
        level = None
    conn.execute(
        "UPDATE budget_rows SET breaker_usd = ? WHERE name = ?",
        (None if level is None else str(level), name),
    )


def seed_budget_rows(conn: sqlite3.Connection) -> None:
    """Insert the fixed category and aggregate rows (idempotent)."""
    for name, cap in (
        ("spark", KIND_ROW_CAP_USD),
        ("sam", KIND_ROW_CAP_USD),
        ("readback", KIND_ROW_CAP_USD),
        (AGGREGATE_ROW, AGGREGATE_SPEND_CEILING_USD),
    ):
        conn.execute(
            "INSERT OR IGNORE INTO budget_rows"
            " (name, hard_cap_usd, spent_usd, held_usd, day_utc, breaker_usd)"
            " VALUES (?, ?, '0', '0', NULL, NULL)",
            (name, str(cap)),
        )


def ensure_day_row(conn: sqlite3.Connection, day_utc: str) -> str:
    """Return today's day-breaker row name, creating it on first use."""
    name = f"{DAY_ROW_PREFIX}{day_utc}"
    conn.execute(
        "INSERT OR IGNORE INTO budget_rows"
        " (name, hard_cap_usd, spent_usd, held_usd, day_utc, breaker_usd)"
        " VALUES (?, ?, '0', '0', ?, NULL)",
        (name, str(DAY_SPEND_CAP_USD), day_utc),
    )
    return name


def get_budget_row(name: str, *, path: Path | str | None = None) -> dict[str, str | None]:
    """Read one budget row for introspection (tests, audit evidence)."""
    conn = db.connect(path)
    try:
        found = _row(conn, name)
        if found is None:
            raise LedgerError(f"unknown budget row {name!r}")
        return {key: found[key] for key in found.keys()}
    finally:
        conn.close()


def breaker_state(name: str, *, path: Path | str | None = None) -> Literal["ok", "warn", "tripped"]:
    """Report the live 80%/100% breaker state of one budget row."""
    current = get_budget_row(name, path=path)
    cap = Decimal(str(current["hard_cap_usd"]))
    level = current["breaker_usd"]
    if level is None:
        return "ok"
    if cap > 0 and Decimal(str(level)) < cap:
        return "warn"
    return "tripped"


def _job_spend(conn: sqlite3.Connection, job_id: str) -> tuple[Decimal, Decimal, dict[str, Decimal], dict[str, Decimal], dict[str, int]]:
    """Return (spent, held, spent_by_cat, held_by_cat, count_by_cat) for a job."""
    spent = Decimal("0")
    held = Decimal("0")
    spent_by_cat: dict[str, Decimal] = {}
    held_by_cat: dict[str, Decimal] = {}
    count_by_cat: dict[str, int] = {}
    rows = conn.execute(
        "SELECT call_kind, estimated_usd, actual_usd, state FROM reservations"
        " WHERE job_id = ? AND state IN ('held', 'settled', 'failed')",
        (job_id,),
    ).fetchall()
    for entry in rows:
        category = category_of(entry["call_kind"])
        count_by_cat[category] = count_by_cat.get(category, 0) + 1
        if entry["state"] == "held":
            amount = Decimal(entry["estimated_usd"])
            held += amount
            held_by_cat[category] = held_by_cat.get(category, Decimal("0")) + amount
        else:
            amount = Decimal(entry["actual_usd"])
            spent += amount
            spent_by_cat[category] = spent_by_cat.get(category, Decimal("0")) + amount
    return spent, held, spent_by_cat, held_by_cat, count_by_cat


def _settle_money(conn: sqlite3.Connection, names: tuple[str, ...], estimated: Decimal, actual: Decimal) -> None:
    for name in names:
        current = _row(conn, name)
        if current is None:  # pragma: no cover - defensive; rows are seeded/ensured
            raise LedgerError(f"missing budget row {name!r} during settle")
        conn.execute(
            "UPDATE budget_rows SET held_usd = ?, spent_usd = ? WHERE name = ?",
            (
                str(Decimal(current["held_usd"]) - estimated),
                str(Decimal(current["spent_usd"]) + actual),
                name,
            ),
        )
        _refresh_breaker(conn, name)


def _release_money(conn: sqlite3.Connection, names: tuple[str, ...], amount: Decimal) -> None:
    for name in names:
        current = _row(conn, name)
        if current is None:  # pragma: no cover - defensive; rows are seeded/ensured
            raise LedgerError(f"missing budget row {name!r} during release")
        conn.execute(
            "UPDATE budget_rows SET held_usd = ? WHERE name = ?",
            (str(Decimal(current["held_usd"]) - amount), name),
        )
        _refresh_breaker(conn, name)


def _job_hold_remainder(conn: sqlite3.Connection, job_id: str) -> Decimal:
    """Return the unconverted part of the job hold: $0.46 minus settled worsts.

    Only categorized calls draw from the hold; uncategorized calls spend from
    the $0.04 admission headroom and never touch it.
    """
    # Money is fixed-point text; sums stay exact by accumulating in Python.
    rows = conn.execute(
        "SELECT call_kind, estimated_usd FROM reservations"
        " WHERE job_id = ? AND state IN ('settled', 'failed')",
        (job_id,),
    ).fetchall()
    settled_worsts = sum(
        (
            Decimal(entry["estimated_usd"])
            for entry in rows
            if category_of(entry["call_kind"]) in CATEGORY_CAP_USD
        ),
        Decimal("0"),
    )
    return JOB_HOLD_USD - settled_worsts


def _close_job_hold(conn: sqlite3.Connection, job_id: str, day_row: str) -> None:
    """Release the job-hold remainder on the day/aggregate rows, once."""
    hold = conn.execute(
        "SELECT id FROM reservations WHERE job_id = ? AND state = 'job_hold'",
        (job_id,),
    ).fetchone()
    if hold is None:
        return
    remainder = _job_hold_remainder(conn, job_id)
    if remainder != 0:
        _release_money(conn, (day_row, AGGREGATE_ROW), remainder)
    conn.execute(
        "UPDATE reservations SET state = 'job_released', settled_at = ? WHERE id = ?",
        (db.format_dt(db.utcnow()), hold["id"]),
    )


def _day_row_for_job(conn: sqlite3.Connection, created_at: str) -> str:
    return ensure_day_row(conn, db.utc_day(db.parse_dt(created_at)))


def _release_unsettled_calls(conn: sqlite3.Connection, job_id: str) -> None:
    """Mark held per-call reservations released and free their holds.

    Categorized holds sit on the kind row; uncategorized holds sit on the
    day/aggregate rows and are freed symmetrically.
    """
    held = conn.execute(
        "SELECT id, call_kind, estimated_usd FROM reservations"
        " WHERE job_id = ? AND state = 'held'",
        (job_id,),
    ).fetchall()
    now = db.format_dt(db.utcnow())
    day_row: str | None = None
    for entry in held:
        category = category_of(entry["call_kind"])
        if category in CATEGORY_CAP_USD:
            _release_money(conn, (category,), Decimal(entry["estimated_usd"]))
        else:
            if day_row is None:
                created = conn.execute(
                    "SELECT created_at FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                day_row = _day_row_for_job(conn, created["created_at"])
            _release_money(conn, (day_row, AGGREGATE_ROW), Decimal(entry["estimated_usd"]))
        conn.execute(
            "UPDATE reservations SET state = 'released', settled_at = ? WHERE id = ?",
            (now, entry["id"]),
        )


def _finalize_job_failed(
    conn: sqlite3.Connection, job_id: str, now_text: str, max_running: int
) -> bool:
    """Move a waiting/active job to ``failed`` with full hold cleanup.

    Runs on the caller's open transaction: clears every provisional keep,
    tombstones every provisional for sweeper deletion (bytes stay until the
    sweeper deletes them), releases unfinished holds and the job-hold
    remainder, and promotes waiting jobs. Terminal jobs are a no-op
    (``False``). This is the one failed-job finalizer: ``finish("failed")``
    and the upload store's cancellation/failure/recovery paths all reuse
    it instead of growing a second finalizer.
    """
    job = conn.execute(
        "SELECT state, created_at FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if job is None:
        raise LedgerError(f"unknown job {job_id}")
    if job["state"] not in ("waiting", "active"):
        return False
    conn.execute(
        "UPDATE jobs SET state = 'failed', completed_at = ? WHERE id = ?",
        (now_text, job_id),
    )
    conn.execute(
        "UPDATE artifacts SET kept_at = NULL, delete_state = 'pending_delete',"
        " expires_at = ?, deleted_at = NULL, hidden = 1"
        " WHERE job_id = ? AND kind IN (?, ?) AND delete_state != 'tombstoned'",
        (now_text, job_id, *sorted(PROVISIONAL_ARTIFACT_KINDS)),
    )
    _release_unsettled_calls(conn, job_id)
    _close_job_hold(conn, job_id, _day_row_for_job(conn, job["created_at"]))
    _promote_waiting(conn, now_text, max_running)
    return True


def queue_failed_outputs(
    conn: sqlite3.Connection, job_id: str, now_text: str,
    guide_filename: str, corrections_filename: str,
    *, present_in: Path | None = None,
) -> None:
    """Queue canonical files in the caller's transaction before any unlink.

    With ``present_in`` (the job folder), outputs already gone are skipped.
    """
    def present(kind: str) -> bool:
        if present_in is None or (present_in / guide_filename).exists():
            return True
        if kind == "cancel_corrections":
            return (present_in / corrections_filename).exists()
        if kind == "cancel_export_links":
            return conn.execute("SELECT 1 FROM export_links WHERE guide_id = ? LIMIT 1",
                                (job_id,)).fetchone() is not None
        return False

    for kind, name in (
        ("cancel_guide", guide_filename),
        ("cancel_corrections", corrections_filename),
        ("cancel_export_links", guide_filename),
    ):
        if not present(kind):
            continue
        conn.execute(
            "INSERT INTO artifacts"
            " (id, job_id, kind, relative_path, expires_at, hidden,"
            " kept_at, delete_state, deleted_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, 1, NULL, 'pending_delete', NULL, ?)"
            " ON CONFLICT(id) DO UPDATE SET delete_state = 'pending_delete',"
            " expires_at = excluded.expires_at, deleted_at = NULL",
            (f"{kind}:{job_id}", job_id, kind, name, now_text, now_text),
        )
    conn.execute(
        "UPDATE artifacts SET delete_state = 'pending_delete', expires_at = ?,"
        " deleted_at = NULL WHERE job_id = ? AND kind = ?"
        " AND delete_state != 'tombstoned'",
        (now_text, job_id, GUIDE_CLIP_ARTIFACT_KIND),
    )


def queue_guide_images(
    conn: sqlite3.Connection, job_id: str, now_text: str, job_directory: Path,
    guide_filename: str,
) -> None:
    """Queue a deleted guide's renders, frames and masks in the caller's transaction.

    Kept renders return to the normal sweep (due now); frames and masks are
    not registered artifacts, so each referenced file inside the job folder
    gets its own ``cancel_frame``/``cancel_mask`` row. Replays are no-ops.
    """
    conn.execute(
        "UPDATE artifacts SET kept_at = NULL, delete_state = 'pending_delete',"
        " expires_at = ?, deleted_at = NULL, hidden = 1"
        f" WHERE job_id = ? AND kept_at IS NOT NULL AND kind IN ({', '.join('?' for _ in guide_lifetime_kinds())})"
        " AND delete_state != 'tombstoned'",
        (now_text, job_id, *guide_lifetime_kinds()),
    )
    try:
        guide = json.loads((job_directory / guide_filename).read_text(encoding="utf-8"))
        if not isinstance(guide, dict) or not isinstance(guide.get("frame_paths"), list) or not isinstance(guide.get("labels"), list):
            raise ValueError("invalid guide image references")
        refs = [("cancel_frame", p) for p in guide["frame_paths"]]
        for label in guide["labels"]:
            if label.get("detection"):
                refs.append(("cancel_mask", label["detection"]["mask_path"]))
    except FileNotFoundError:
        refs = []
    # An unpublished job has no guide to enumerate its images. Queue the job's
    # PNGs as well, including images created before publication failed.
    def walk_error(error: OSError) -> None:
        raise error
    for directory, _, names in os.walk(job_directory, onerror=walk_error):
        refs.extend(("cancel_frame", Path(directory) / name)
                    for name in names if name.endswith(".png"))
    root = job_directory.resolve()
    for kind, value in refs:
        try:
            relative = Path(value).resolve().relative_to(root)
        except (ValueError, OSError, TypeError) as error:
            raise ValueError(f"invalid guide image path: {value!r}") from error
        conn.execute(
            "INSERT OR IGNORE INTO artifacts"
            " (id, job_id, kind, relative_path, expires_at, hidden,"
            " kept_at, delete_state, deleted_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, 1, NULL, 'pending_delete', NULL, ?)",
            (f"{kind}:{job_id}:{relative}", job_id, kind, str(relative), now_text, now_text),
        )


def _expire_overdue(conn: sqlite3.Connection, now_text: str) -> tuple[JobId, ...]:
    """Move past-deadline waiting/active jobs to ``expired``, freeing holds.

    A job whose deadline has been reached (``deadline_at <= now``) is expired,
    matching the reservation refusal at ``now >= deadline``. Freed running
    slots are filled FIFO so a waiter never sits out while a slot is empty.
    """
    from app.admission import MAX_RUNNING  # deferred: admission imports ledger

    overdue = conn.execute(
        "SELECT id, created_at FROM jobs WHERE state IN ('waiting', 'active') AND deadline_at <= ?"
        " ORDER BY created_at, id",
        (now_text,),
    ).fetchall()
    expired: list[JobId] = []
    for entry in overdue:
        conn.execute(
            "UPDATE jobs SET state = 'expired', completed_at = ? WHERE id = ?",
            (now_text, entry["id"]),
        )
        _release_unsettled_calls(conn, entry["id"])
        _close_job_hold(conn, entry["id"], _day_row_for_job(conn, entry["created_at"]))
        expired.append(JobId(entry["id"]))
    _promote_waiting(conn, now_text, MAX_RUNNING)
    return tuple(expired)


def expire_overdue_jobs(now: datetime | None = None, *, path: Path | str | None = None) -> tuple[JobId, ...]:
    """Expire past-deadline open jobs (the upload module calls this before sweeping)."""
    moment = now or db.utcnow()
    conn = db.connect(path)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                expired = _expire_overdue(conn, db.format_dt(moment))
                conn.execute("COMMIT")
                return expired
            except Exception:
                conn.execute("ROLLBACK")
                raise
        except sqlite3.OperationalError as error:
            if _locked(error):
                raise ContentionError("expire lost a write race; retry") from error
            raise
    finally:
        conn.close()


def extend_job_deadline(
    job_id: JobId,
    extra_s: float,
    *,
    path: Path | str | None = None,
    max_extra_s: float | None = None,
    require_confirmed_upload: bool = False,
) -> bool:
    """Push one active job's deadline out by ``extra_s`` seconds.

    Chunked live runs outlive the single-job deadline, so the service
    extends a confirmed multi-chunk parent before building its live
    clients (and each chunk child carries its own configured Spark
    envelope); reserves, expiry, and client call budgets then share
    the longer window. One shot per job: a deadline already past its
    original ``created_at + 240 s`` is never pushed again, and no push
    passes ``created_at + 240 s + max_extra_s`` (default
    ``app.settings.DEFAULT_DEADLINE_EXTENSION_MAX_S``). Conditional
    on ``active`` with a live deadline: terminal jobs and jobs past
    their deadline keep their row and this returns False, never
    resurrecting one. With ``require_confirmed_upload`` the same
    transaction also requires a confirmed, visible upload row with an
    attached Meta file.
    """
    if (
        isinstance(extra_s, bool)
        or not isinstance(extra_s, (int, float))
        or not math.isfinite(extra_s)
        or not extra_s > 0
    ):
        raise ValueError(
            f"extra_s must be a finite number > 0, got {extra_s!r}"
        )
    if max_extra_s is None:
        # Deferred: settings never imports ledger, but ledger stays
        # importable without it, like the other deferred imports here.
        from app.settings import DEFAULT_DEADLINE_EXTENSION_MAX_S

        max_extra_s = DEFAULT_DEADLINE_EXTENSION_MAX_S
    if (
        isinstance(max_extra_s, bool)
        or not isinstance(max_extra_s, (int, float))
        or not math.isfinite(max_extra_s)
        or not max_extra_s > 0
    ):
        raise ValueError(
            f"max_extra_s must be a finite number > 0, got {max_extra_s!r}"
        )
    if not isinstance(require_confirmed_upload, bool):
        raise ValueError(
            "require_confirmed_upload must be a bool,"
            f" got {require_confirmed_upload!r}"
        )
    conn = db.connect(path)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Judge expiry after the lock is held: a timestamp from
            # before the wait could predate a deadline crossed while
            # blocked, resurrecting an expired job.
            now_text = db.format_dt(db.utcnow())
            try:
                row = conn.execute(
                    "SELECT created_at, deadline_at FROM jobs"
                    " WHERE id = ? AND state = 'active'",
                    (str(job_id),),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return False
                if row["deadline_at"] <= now_text:
                    # Past its deadline: expired in all but the sweep's
                    # UPDATE, so extending would resurrect it.
                    conn.execute("ROLLBACK")
                    return False
                original = db.parse_dt(row["created_at"]) + timedelta(
                    seconds=JOB_DEADLINE_S
                )
                current = db.parse_dt(row["deadline_at"])
                if current > original:
                    # Already extended once; one durable push per job.
                    conn.execute("ROLLBACK")
                    return False
                pushed = current + timedelta(seconds=float(extra_s))
                bound = original + timedelta(
                    seconds=float(max_extra_s)
                )
                if pushed > bound:
                    conn.execute("ROLLBACK")
                    return False
                if require_confirmed_upload:
                    up = conn.execute(
                        "SELECT state, meta_file_id, hidden FROM uploads"
                        " WHERE job_id = ?",
                        (str(job_id),),
                    ).fetchone()
                    if (
                        up is None
                        or up["state"] != "confirmed"
                        or up["meta_file_id"] is None
                        or up["hidden"]
                    ):
                        conn.execute("ROLLBACK")
                        return False
                cursor = conn.execute(
                    "UPDATE jobs SET deadline_at = ? WHERE id = ?"
                    " AND state = 'active' AND deadline_at = ?",
                    (db.format_dt(pushed), str(job_id), row["deadline_at"]),
                )
                if cursor.rowcount == 0:
                    # A concurrent finish or extension won the race
                    # after the SELECT; report it instead of claiming
                    # an extension.
                    conn.execute("ROLLBACK")
                    return False
                conn.execute("COMMIT")
                return True
            except Exception:
                conn.execute("ROLLBACK")
                raise
        except sqlite3.OperationalError as error:
            if _locked(error):
                raise ContentionError("extend lost a write race; retry") from error
            raise
    finally:
        conn.close()


class ArtifactStoreImpl:
    """Job-scoped artifact adapter: only paths under the job directory."""

    def __init__(self, database: Path, job_id: JobId, job_dir: Path) -> None:
        self._db = database
        self._job_id = job_id
        self._job_dir = job_dir

    def _conn(self) -> sqlite3.Connection:
        return db.connect(self._db)

    def _relative(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self._job_dir.resolve())
        except ValueError:
            raise ArtifactPathError(f"artifact path escapes job directory: {path}") from None

    def register(self, kind: str, path: Path) -> ArtifactId:
        """Record one artifact; the path must sit under the job directory."""
        if not kind or not kind.strip():
            raise ValueError("artifact kind must be a non-empty string")
        relative = self._relative(path)
        artifact_id = ArtifactId(secrets.token_hex(16))
        now = db.utcnow()
        expires = now + timedelta(days=ARTIFACT_RETENTION_DAYS)
        conn = self._conn()
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    job = conn.execute(
                        "SELECT state FROM jobs WHERE id = ?", (str(self._job_id),)
                    ).fetchone()
                    if job is None:
                        raise LedgerError(f"unknown job {self._job_id}")
                    if job["state"] not in OPEN_JOB_STATES:
                        raise JobAlreadyFinished(
                            f"cannot register artifacts on {job['state']} job {self._job_id}"
                        )
                    conn.execute(
                        "INSERT INTO artifacts (id, job_id, kind, relative_path, expires_at,"
                        " hidden, kept_at, delete_state, deleted_at, created_at)"
                        " VALUES (?, ?, ?, ?, ?, 0, NULL, 'live', NULL, ?)",
                        (
                            str(artifact_id),
                            str(self._job_id),
                            kind,
                            str(relative),
                            db.format_dt(expires),
                            db.format_dt(now),
                        ),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("register lost a write race; retry") from error
                raise
        finally:
            conn.close()
        absolute = self._job_dir / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        return artifact_id

    def mark_kept(self, artifact_id: ArtifactId) -> None:
        """Retain one live artifact so ``finish`` never deletes it (idempotent).

        One guarded transaction refuses finished jobs and non-live rows, so a
        ``finish`` racing this call cannot leave ``kept_at`` on a deleted row.
        """
        conn = self._conn()
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    found = conn.execute(
                        "SELECT kept_at, delete_state FROM artifacts"
                        " WHERE id = ? AND job_id = ?",
                        (str(artifact_id), str(self._job_id)),
                    ).fetchone()
                    if found is None:
                        raise ArtifactNotFound(
                            f"unknown artifact {artifact_id} for job {self._job_id}"
                        )
                    job = conn.execute(
                        "SELECT state FROM jobs WHERE id = ?", (str(self._job_id),)
                    ).fetchone()
                    if job is None:
                        raise LedgerError(f"unknown job {self._job_id}")
                    if job["state"] not in OPEN_JOB_STATES:
                        raise JobAlreadyFinished(
                            f"cannot keep artifacts on {job['state']} job {self._job_id}"
                        )
                    if found["delete_state"] != "live":
                        raise LedgerError(
                            f"artifact {artifact_id} is {found['delete_state']}, not live"
                        )
                    if found["kept_at"] is None:
                        updated = conn.execute(
                            "UPDATE artifacts SET kept_at = ?"
                            " WHERE id = ? AND delete_state = 'live'",
                            (db.format_dt(db.utcnow()), str(artifact_id)),
                        )
                        if updated.rowcount != 1:  # pragma: no cover - raced delete
                            raise LedgerError(
                                f"artifact {artifact_id} left live state during keep"
                            )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("mark_kept lost a write race; retry") from error
                raise
        finally:
            conn.close()

    def mark_deleted(self, artifact_id: ArtifactId) -> None:
        """Tombstone one artifact idempotently, treating ENOENT as success.

        Two-phase: claim the row as ``pending_delete`` in a guarded
        transaction, unlink, then tombstone in a second guarded transaction.
        A database failure therefore never leaves a live row for a deleted
        file, and retrying converges (the sweeper also lists pending rows).
        """
        conn = self._conn()
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    found = conn.execute(
                        "SELECT relative_path, delete_state FROM artifacts"
                        " WHERE id = ? AND job_id = ?",
                        (str(artifact_id), str(self._job_id)),
                    ).fetchone()
                    if found is None:
                        raise ArtifactNotFound(
                            f"unknown artifact {artifact_id} for job {self._job_id}"
                        )
                    if found["delete_state"] == "tombstoned":
                        conn.execute("COMMIT")
                        return
                    conn.execute(
                        "UPDATE artifacts SET delete_state = 'pending_delete' WHERE id = ?",
                        (str(artifact_id),),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("mark_deleted lost a write race; retry") from error
                raise
            (self._job_dir / found["relative_path"]).unlink(missing_ok=True)
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(
                        "UPDATE artifacts SET delete_state = 'tombstoned', deleted_at = ?,"
                        " hidden = 1 WHERE id = ?",
                        (db.format_dt(db.utcnow()), str(artifact_id)),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("mark_deleted lost a write race; retry") from error
                raise
        finally:
            conn.close()


class CallReservationImpl:
    """One authorized paid call; settled exactly once with its actual cost."""

    def __init__(self, database: Path, reservation_id: str) -> None:
        self._db = database
        self._id = reservation_id

    @property
    def reservation_id(self) -> str:
        """Return the ledger row id behind this handle (audit/testing use)."""
        return self._id

    def settle(self, actual_usd: Decimal, outcome: Literal["succeeded", "failed"]) -> None:
        """Record actual cost, release the unused hold, and close the call.

        The worst-case invariant holds before any balance moves: an actual
        above the reservation, or one that would pass the $0.50 job cap, is
        refused. The refusal still records the reservation as failed and
        converts its hold into the reported actual spend on every applicable
        row — even past the cap — so the attempt is visible and the global
        ceilings see the money that escaped.
        """
        actual = _money(actual_usd)
        if outcome not in ("succeeded", "failed"):
            raise ValueError(f"outcome must be 'succeeded' or 'failed', got {outcome!r}")
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    entry = conn.execute(
                        "SELECT job_id, budget_row, call_kind, estimated_usd, state"
                        " FROM reservations WHERE id = ?",
                        (self._id,),
                    ).fetchone()
                    if entry is None:
                        raise ReservationNotHeld(f"unknown reservation {self._id}")
                    if entry["state"] != "held":
                        raise ReservationNotHeld(
                            f"reservation {self._id} is {entry['state']}, not held"
                        )
                    estimated = Decimal(entry["estimated_usd"])
                    category = category_of(entry["call_kind"])
                    job = conn.execute(
                        "SELECT created_at FROM jobs WHERE id = ?", (entry["job_id"],)
                    ).fetchone()
                    if job is None:  # pragma: no cover - FK RESTRICT prevents this
                        raise LedgerError(f"reservation {self._id} has no job")
                    day_names: tuple[str, ...] = (
                        _day_row_for_job(conn, job["created_at"]),
                        AGGREGATE_ROW,
                    )
                    if actual > estimated:
                        self._refuse_overrun(
                            conn, category, day_names, estimated, actual,
                            f"settle refused: actual ${actual} exceeds"
                            f" reserved ${estimated}",
                        )
                    spent, _, _, _, _ = _job_spend(conn, entry["job_id"])
                    if spent + actual > JOB_COST_CAP_USD:
                        self._refuse_overrun(
                            conn, category, day_names, estimated, actual,
                            f"settle refused: actual ${actual} would pass"
                            f" the ${JOB_COST_CAP_USD} job cap (spent ${spent})",
                        )
                    if category in CATEGORY_CAP_USD:
                        _settle_money(
                            conn,
                            (category, *day_names),
                            estimated,
                            actual,
                        )
                    else:
                        _settle_money(conn, day_names, estimated, actual)
                    conn.execute(
                        "UPDATE reservations SET actual_usd = ?, state = ?, settled_at = ?"
                        " WHERE id = ?",
                        (
                            str(actual),
                            "settled" if outcome == "succeeded" else "failed",
                            db.format_dt(db.utcnow()),
                            self._id,
                        ),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("settle lost a write race; retry") from error
                raise
        finally:
            conn.close()

    def _refuse_overrun(
        self,
        conn: sqlite3.Connection,
        category: str,
        day_names: tuple[str, ...],
        estimated: Decimal,
        actual: Decimal,
        message: str,
    ) -> None:
        """Record an over-cap settlement as failed, keep its spend, then refuse.

        The reported actual already escaped the reservation, so it lands as
        real spend on every applicable row — even past the cap — while the
        hold is released and the breakers recomputed. Commits before raising
        so the failed record survives; the caller-side rollback guard skips
        an already-committed transaction.
        """
        if category in CATEGORY_CAP_USD:
            _settle_money(conn, (category, *day_names), estimated, actual)
        else:
            _settle_money(conn, day_names, estimated, actual)
        conn.execute(
            "UPDATE reservations SET actual_usd = ?, state = 'failed', settled_at = ?"
            " WHERE id = ?",
            (str(actual), db.format_dt(db.utcnow()), self._id),
        )
        conn.execute("COMMIT")
        raise BudgetExhausted(message)


class JobContextImpl:
    """ledger-module job context: the sole path to paid-call authorization."""

    def __init__(
        self,
        database: Path,
        job_id: JobId,
        job_dir: Path,
        *,
        spark_cap_usd: Decimal | None = None,
    ) -> None:
        self._db = database
        self._job_id = job_id
        self._artifacts = ArtifactStoreImpl(database, job_id, job_dir)
        self._spark_cap_usd = _require_spark_cap_usd(spark_cap_usd)

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def artifacts(self) -> ArtifactStoreImpl:
        return self._artifacts

    def reserve(self, kind: str, worst_case_usd: Decimal) -> CallReservationImpl:
        """Authorize one paid call or raise before any client invocation.

        One short ``BEGIN IMMEDIATE`` transaction checks the job state and
        deadline, the $0.50 job cap, the per-category spend and count ceilings
        (the job's Spark reservation / $0.08-32-call SAM / $0.12-3-read-back,
        $0.04 other), the category row, and the day/aggregate ceilings, then
        inserts the hold.
        """
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("call kind must be a non-empty string")
        worst = _money(worst_case_usd)
        _reject_live_judge_kind(kind)
        category = category_of(kind)
        reservation_id = secrets.token_hex(16)
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    job = conn.execute(
                        "SELECT state, created_at, deadline_at, readback_fired_n"
                        " FROM jobs WHERE id = ?",
                        (str(self._job_id),),
                    ).fetchone()
                    if job is None:
                        raise LedgerError(f"unknown job {self._job_id}")
                    if job["state"] != "active":
                        raise JobNotActive(
                            f"job {self._job_id} is {job['state']}, not active"
                        )
                    now = db.utcnow()
                    now_text = db.format_dt(now)
                    if now_text >= job["deadline_at"]:
                        raise JobTimedOut(
                            f"job {self._job_id} passed its {JOB_DEADLINE_S}s deadline"
                        )
                    spent, held, spent_by_cat, held_by_cat, count_by_cat = _job_spend(
                        conn, str(self._job_id)
                    )
                    if spent + held + worst > JOB_COST_CAP_USD:
                        raise BudgetExhausted(
                            f"${worst} {kind} call exceeds the ${JOB_COST_CAP_USD} job cap"
                            f" (spent ${spent}, held ${held})"
                        )
                    if category in CATEGORY_CAP_USD:
                        cat_total = (
                            spent_by_cat.get(category, Decimal("0"))
                            + held_by_cat.get(category, Decimal("0"))
                            + worst
                        )
                        cap = (
                            self._spark_cap_usd
                            if category == "spark"
                            else CATEGORY_CAP_USD[category]
                        )
                        if cat_total > cap:
                            raise BudgetExhausted(
                                f"${worst} {kind} call exceeds the"
                                f" ${cap} {category} reservation"
                            )
                        count_cap = CATEGORY_COUNT_CAP.get(category)
                        if (
                            count_cap is not None
                            and count_by_cat.get(category, 0) >= count_cap
                        ):
                            raise BudgetExhausted(
                                f"{kind} call exceeds the {count_cap} {category} calls per job"
                            )
                        if category == "readback":
                            if job["readback_fired_n"] >= MAX_READBACKS_PER_JOB:
                                raise BudgetExhausted(
                                    "read-back call exceeds the"
                                    f" {MAX_READBACKS_PER_JOB} read-backs per job"
                                )
                            conn.execute(
                                "UPDATE jobs SET readback_fired_n = readback_fired_n + 1"
                                " WHERE id = ?",
                                (str(self._job_id),),
                            )
                    else:
                        other_total = (
                            spent_by_cat.get(category, Decimal("0"))
                            + held_by_cat.get(category, Decimal("0"))
                            + worst
                        )
                        if other_total > OTHER_CATEGORY_SPEND_CAP_USD:
                            raise BudgetExhausted(
                                f"${worst} {kind} call exceeds the"
                                f" ${OTHER_CATEGORY_SPEND_CAP_USD} uncategorized headroom"
                            )
                    day_row = _day_row_for_job(conn, job["created_at"])
                    if category in CATEGORY_CAP_USD:
                        row = _row(conn, category)
                        if row is None:  # pragma: no cover - seeded by init_db
                            raise LedgerError(f"missing budget row {category!r}")
                        if (
                            Decimal(row["held_usd"]) + Decimal(row["spent_usd"]) + worst
                            > Decimal(row["hard_cap_usd"])
                        ):
                            raise BudgetExhausted(
                                f"${worst} {kind} call exceeds the {category} row cap"
                            )
                        conn.execute(
                            "UPDATE budget_rows SET held_usd = ? WHERE name = ?",
                            (
                                str(Decimal(row["held_usd"]) + worst),
                                category,
                            ),
                        )
                        _refresh_breaker(conn, category)
                        # A categorized worst case draws from the $0.46 job hold
                        # admission took, so the day/aggregate check is
                        # fail-closed (refuse when already over cap) without
                        # double-holding.
                        for name in (day_row, AGGREGATE_ROW):
                            ceiling = _row(conn, name)
                            if ceiling is None:  # pragma: no cover - ensured/seeded
                                raise LedgerError(f"missing budget row {name!r}")
                            if Decimal(ceiling["held_usd"]) + Decimal(
                                ceiling["spent_usd"]
                            ) > Decimal(ceiling["hard_cap_usd"]):
                                raise BudgetExhausted(
                                    f"{kind} call refused: {name} breaker already over cap"
                                )
                    else:
                        # Uncategorized calls spend from the $0.04 headroom the
                        # admission checked but never held, so each one holds
                        # its own worst case on the day/aggregate rows (settled
                        # and released symmetrically by settle/finish/expire).
                        for name in (day_row, AGGREGATE_ROW):
                            ceiling = _row(conn, name)
                            if ceiling is None:  # pragma: no cover - ensured/seeded
                                raise LedgerError(f"missing budget row {name!r}")
                            if (
                                Decimal(ceiling["held_usd"])
                                + Decimal(ceiling["spent_usd"])
                                + worst
                                > Decimal(ceiling["hard_cap_usd"])
                            ):
                                raise BudgetExhausted(
                                    f"${worst} {kind} call exceeds the {name} ceiling"
                                )
                            conn.execute(
                                "UPDATE budget_rows SET held_usd = ? WHERE name = ?",
                                (
                                    str(Decimal(ceiling["held_usd"]) + worst),
                                    name,
                                ),
                            )
                            _refresh_breaker(conn, name)
                    conn.execute(
                        "INSERT INTO reservations (id, job_id, budget_row, call_kind,"
                        " estimated_usd, actual_usd, state, created_at, settled_at)"
                        " VALUES (?, ?, ?, ?, ?, NULL, 'held', ?, NULL)",
                        (
                            reservation_id,
                            str(self._job_id),
                            category if category in CATEGORY_CAP_USD else "job",
                            kind,
                            str(worst),
                            now_text,
                        ),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("reserve lost a write race; retry") from error
                raise
        finally:
            conn.close()
        return CallReservationImpl(self._db, reservation_id)

    def record_call_with_event(
        self,
        key: str,
        call: CallRecord,
        event: tuple[str, Mapping[str, JSONValue]],
    ) -> None:
        """Persist one model-call receipt and its funnel event, atomically.

        The call must belong to this job; the call and
        the event are validated and normalized against the exact evidence-module
        declarations in ``app.evidence``; ``key`` must equal the evidence module's
        ``call_key`` of the stored record. Any violation raises ``ValueError``
        before a write. One transaction then runs the keyed ``model_calls``
        ``INSERT OR IGNORE`` first and the keyed ``events`` ``INSERT OR
        IGNORE`` second, reads both rows back, and compares their complete
        normalized payloads (excluding only database-generated
        ``created_at``). An exact replay commits as a no-op; a differing
        payload for the key rolls back and raises ``ValueError("receipt key
        conflict")``. Any statement or commit failure rolls back both rows.
        """
        try:
            if call.job_id != self._job_id:
                raise ValueError("call record belongs to a different job")
            stored = _l5evidence._validated_call_record(call)
            name, detail = event
            valid_name, cleaned = _l5evidence._validated_event(name, detail)
        except (AttributeError, TypeError) as error:
            raise ValueError(f"invalid receipt: {error}") from error
        if key != _l5evidence.call_key(stored):
            raise ValueError("receipt key does not match the stored record")
        call_json = json.dumps(dict(stored.evidence), sort_keys=True)
        detail_json = json.dumps(dict(cleaned), sort_keys=True)
        moment = db.format_dt(db.utcnow())
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    if (
                        conn.execute(
                            "SELECT id FROM jobs WHERE id = ?", (str(self._job_id),)
                        ).fetchone()
                        is None
                    ):
                        raise LedgerError(f"unknown job {self._job_id}")
                    conn.execute(
                        "INSERT OR IGNORE INTO model_calls (id, job_id, stage, model,"
                        " endpoint, latency_ms, units_in, units_out, estimated_cost_usd,"
                        " status, evidence_json, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            key,
                            str(self._job_id),
                            stored.stage,
                            stored.model,
                            stored.endpoint,
                            stored.latency_ms,
                            stored.units_in,
                            stored.units_out,
                            str(stored.estimated_cost_usd),
                            stored.status,
                            call_json,
                            moment,
                        ),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO events (id, job_id, anon_id, session_id,"
                        " event, detail_json, created_at)"
                        " VALUES (?, ?, NULL, NULL, ?, ?, ?)",
                        (key, str(self._job_id), valid_name, detail_json, moment),
                    )
                    call_row = conn.execute(
                        "SELECT job_id, stage, model, endpoint, latency_ms, units_in,"
                        " units_out, estimated_cost_usd, status, evidence_json"
                        " FROM model_calls WHERE id = ?",
                        (key,),
                    ).fetchone()
                    event_row = conn.execute(
                        "SELECT job_id, event, detail_json FROM events WHERE id = ?",
                        (key,),
                    ).fetchone()
                    if call_row is None or event_row is None:
                        raise LedgerError(f"receipt {key} write was lost")
                    back = CallRecord(
                        job_id=JobId(call_row["job_id"]),
                        stage=call_row["stage"],
                        model=call_row["model"],
                        endpoint=call_row["endpoint"],
                        latency_ms=call_row["latency_ms"],
                        units_in=call_row["units_in"],
                        units_out=call_row["units_out"],
                        estimated_cost_usd=Decimal(call_row["estimated_cost_usd"]),
                        status=call_row["status"],
                        evidence=json.loads(call_row["evidence_json"]),
                    )
                    if (
                        back != stored
                        or event_row["job_id"] != str(self._job_id)
                        or event_row["event"] != valid_name
                        or json.loads(event_row["detail_json"]) != cleaned
                    ):
                        raise ValueError("receipt key conflict")
                    conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError(
                        "record_call_with_event lost a write race; retry"
                    ) from error
                raise
        finally:
            conn.close()

    def remaining_s(self) -> float:
        """Seconds left before this job's deadline (<= 0 once it passed).

        The live providers bound every call, retry and backoff by it.
        """
        conn = db.connect(self._db)
        try:
            row = conn.execute(
                "SELECT deadline_at FROM jobs WHERE id = ?", (str(self._job_id),)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise LedgerError(f"unknown job {self._job_id}")
        return (db.parse_dt(row["deadline_at"]) - db.utcnow()).total_seconds()

    def list_call_records(self) -> tuple[CallRecord, ...]:
        """Return this job's evidence rows, oldest first (the evidence module's read seam)."""
        return list_call_records(self._job_id, path=self._db)

    def _kept_provisionals(self, conn: sqlite3.Connection) -> set[str]:
        """Return the ids of this job's provisionals carrying ``kept_at``."""
        return {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM artifacts WHERE job_id = ? AND kind IN (?, ?)"
                " AND kept_at IS NOT NULL",
                (str(self._job_id), *sorted(PROVISIONAL_ARTIFACT_KINDS)),
            ).fetchall()
        }

    def _keep_exact_set(
        self, conn: sqlite3.Connection, unique: list[str], now_text: str
    ) -> None:
        """Validate, keep, and tombstone-around one ``done`` keep set.

        Every id is verified before any write: same job, provisional, live,
        and backed by a regular file. Unknown or foreign ids fail closed
        with no change. The complete set is then marked kept and every other
        provisional is tombstoned for sweeper deletion (any pre-existing
        keep outside the set is cleared).
        """
        for artifact_id in unique:
            found = conn.execute(
                "SELECT job_id, kind, relative_path, delete_state FROM artifacts"
                " WHERE id = ?",
                (artifact_id,),
            ).fetchone()
            if found is None or found["job_id"] != str(self._job_id):
                raise ArtifactNotFound(
                    f"unknown artifact {artifact_id} for job {self._job_id}"
                )
            if found["kind"] not in PROVISIONAL_ARTIFACT_KINDS:
                raise LedgerError(
                    f"keep set holds non-provisional artifact {artifact_id}"
                )
            if found["delete_state"] != "live":
                raise LedgerError(
                    f"keep set holds {found['delete_state']} artifact {artifact_id}"
                )
            absolute = self._artifacts._job_dir / found["relative_path"]
            if absolute.is_symlink() or not absolute.is_file():
                raise LedgerError(
                    f"keep set holds artifact {artifact_id} with no regular file"
                )
        for artifact_id in unique:
            updated = conn.execute(
                "UPDATE artifacts SET kept_at = ? WHERE id = ? AND delete_state = 'live'",
                (now_text, artifact_id),
            )
            if updated.rowcount != 1:  # pragma: no cover - raced delete
                raise LedgerError(
                    f"artifact {artifact_id} left live state during finish"
                )
        tombstone = (
            "UPDATE artifacts SET kept_at = NULL, delete_state = 'pending_delete',"
            " expires_at = ?, deleted_at = NULL, hidden = 1"
            " WHERE job_id = ? AND kind IN (?, ?)"
            " AND delete_state != 'tombstoned'"
        )
        kinds = sorted(PROVISIONAL_ARTIFACT_KINDS)
        if unique:
            placeholders = ", ".join("?" for _ in unique)
            conn.execute(
                f"{tombstone} AND id NOT IN ({placeholders})",
                (now_text, str(self._job_id), *kinds, *unique),
            )
        else:
            conn.execute(tombstone, (now_text, str(self._job_id), *kinds))

    def finish(
        self,
        outcome: Literal["done", "failed"],
        *,
        keep_artifact_ids: Sequence[ArtifactId] = (),
        failed_output_names: tuple[str, str] | None = None,
    ) -> None:
        """Move ``active`` to ``done`` or ``waiting``/``active`` to ``failed``.

        The single transaction marks exactly the requested keep set kept and
        tombstones every other provisional (``pending_delete`` with expiry at
        the finish time, so the sweeper lists them at once; bytes stay until
        the sweeper deletes them), releases unfinished holds and the
        job-hold remainder, and promotes the oldest waiting jobs.
        ``"failed"`` requires an empty keep set, clears any provisional keep,
        and tombstones every provisional. Any validation, statement, or
        pre-commit failure rolls back the keeps and the terminal transition
        together, leaving the job open so the caller can finish failed. An
        exact replay against ``done`` with the identical normalized keep set
        is idempotent; any different set or outcome is refused. Failed output
        names, when supplied, enter the deletion queue in this transaction.
        """
        from app.admission import MAX_RUNNING  # deferred: admission imports ledger

        if outcome not in ("done", "failed"):
            raise ValueError(f"outcome must be 'done' or 'failed', got {outcome!r}")
        try:
            wanted = [str(artifact_id) for artifact_id in keep_artifact_ids]
        except TypeError as error:
            raise ValueError(f"keep_artifact_ids must be a sequence: {error}") from error
        if outcome == "failed" and wanted:
            raise ValueError("finish('failed') keeps nothing; pass an empty keep set")
        if outcome == "done" and failed_output_names is not None:
            raise ValueError("done jobs cannot queue failed outputs")
        unique = list(dict.fromkeys(wanted))
        conn = db.connect(self._db)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    job = conn.execute(
                        "SELECT state, created_at FROM jobs WHERE id = ?",
                        (str(self._job_id),),
                    ).fetchone()
                    if job is None:
                        raise LedgerError(f"unknown job {self._job_id}")
                    if job["state"] != "active" and not (
                        outcome == "failed" and job["state"] == "waiting"
                    ):
                        if (
                            outcome == "done"
                            and job["state"] == "done"
                            and self._kept_provisionals(conn) == set(unique)
                        ):
                            conn.execute("COMMIT")
                            return
                        raise JobAlreadyFinished(
                            f"job {self._job_id} is already {job['state']}"
                        )
                    now_text = db.format_dt(db.utcnow())
                    if outcome == "done":
                        self._keep_exact_set(conn, unique, now_text)
                        conn.execute(
                            "UPDATE jobs SET state = 'done', completed_at = ? WHERE id = ?",
                            (now_text, str(self._job_id)),
                        )
                        _release_unsettled_calls(conn, str(self._job_id))
                        _close_job_hold(
                            conn,
                            str(self._job_id),
                            _day_row_for_job(conn, job["created_at"]),
                        )
                        _promote_waiting(conn, now_text, MAX_RUNNING)
                    else:
                        _finalize_job_failed(
                            conn, str(self._job_id), now_text, MAX_RUNNING
                        )
                        if failed_output_names is not None:
                            queue_failed_outputs(
                                conn, str(self._job_id), now_text,
                                *failed_output_names,
                            )
                            # Generated frames and masks are queued with the
                            # failed state, so a later cleanup error cannot
                            # strand them. A bad listing must not block failure:
                            # the after-failure cleanup lists them again.
                            conn.execute("SAVEPOINT failed_images")
                            try:
                                queue_guide_images(
                                    conn, str(self._job_id), now_text,
                                    self._artifacts._job_dir, failed_output_names[0],
                                )
                            except Exception as error:
                                conn.execute("ROLLBACK TO failed_images")
                                logger.warning(
                                    "failed job=%s images not listed: %s",
                                    self._job_id, type(error).__name__,
                                )
                            conn.execute("RELEASE failed_images")
                    conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except BaseException:
                            pass
                    raise
            except sqlite3.OperationalError as error:
                if _locked(error):
                    raise ContentionError("finish lost a write race; retry") from error
                raise
        finally:
            conn.close()


def _promote_waiting(conn: sqlite3.Connection, now_text: str, max_running: int) -> None:
    """Fill freed running slots with the oldest waiting jobs, FIFO.

    Waiting jobs already past their deadline expire instead of running, so a
    stale queue entry can never start spending after its 240 seconds.
    """
    active = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'active'").fetchone()[0]
    if active >= max_running:
        return
    waiting = conn.execute(
        "SELECT id, created_at, deadline_at FROM jobs WHERE state = 'waiting'"
        " ORDER BY created_at, id"
    ).fetchall()
    for entry in waiting:
        if active >= max_running:
            return
        if entry["deadline_at"] <= now_text:
            conn.execute(
                "UPDATE jobs SET state = 'expired', completed_at = ? WHERE id = ?",
                (now_text, entry["id"]),
            )
            _release_unsettled_calls(conn, entry["id"])
            _close_job_hold(conn, entry["id"], _day_row_for_job(conn, entry["created_at"]))
            continue
        conn.execute("UPDATE jobs SET state = 'active' WHERE id = ?", (entry["id"],))
        active += 1


class LifecycleRepositoryImpl:
    """Terminal-job deletion queue; active jobs are never listed."""

    def __init__(self, database: Path | str | None = None) -> None:
        self._db = Path(database) if database is not None else db.db_path()

    def _due_predicate(self) -> str:
        # Terminal jobs only: an open job's artifacts stay invisible until the
        # transactional expire_overdue_jobs() moves it to 'expired'.
        return "j.state IN ('done', 'failed', 'expired')"

    def list_due_artifacts(self, now: datetime) -> tuple[ArtifactRecord, ...]:
        """Return expired, undeleted artifacts of terminal/expired jobs only.

        A done job's guide clip is never due: it lives with the guide,
        not with the 24-hour preview promise, and only guide deletion
        removes it (a deletion it already claimed, ``pending_delete`` or a
        ``retry`` left by a busy or failed pass, is due like any other). The
        same holds for the exact kept render set of a done job (``kept_at``
        set, kinds from config/retention.toml); guide deletion clears
        ``kept_at``, and discarded renders and every row of a failed or
        expired job still sweep. A failed or expired job's clip row is a crashed
        publish's orphan (no guide was ever committed for it), so it
        sweeps like any other artifact.
        """
        now_text = db.format_dt(now)
        kept_kinds = guide_lifetime_kinds()
        marks = ", ".join("?" for _ in kept_kinds)
        conn = db.connect(self._db)
        try:
            rows = conn.execute(
                "SELECT a.id, a.job_id, a.kind, a.relative_path, a.expires_at"
                " FROM artifacts a JOIN jobs j ON a.job_id = j.id"
                " WHERE a.expires_at <= ?"
                " AND a.delete_state IN ('live', 'retry', 'pending_delete')"
                " AND (a.kind != ? OR j.state != 'done' OR a.delete_state != 'live')"
                f" AND NOT (j.state = 'done' AND a.kept_at IS NOT NULL AND a.kind IN ({marks}))"
                f" AND {self._due_predicate()} ORDER BY a.expires_at, a.id",
                (now_text, GUIDE_CLIP_ARTIFACT_KIND, *kept_kinds),
            ).fetchall()
        finally:
            conn.close()
        return tuple(
            ArtifactRecord(
                id=ArtifactId(entry["id"]),
                job_id=JobId(entry["job_id"]),
                kind=entry["kind"],
                relative_path=Path(entry["relative_path"]),
                expires_at=db.parse_dt(entry["expires_at"]),
            )
            for entry in rows
        )

    def list_due_meta_files(self, now: datetime) -> tuple[MetaFileRecord, ...]:
        """Return expired, unverified meta files of terminal/expired jobs only.

        A Meta retry never depends on a local artifact row: each deletion
        record stands on its own ``delete_state``. A ``pending`` row (queued
        for immediate deletion by the upload store) and a ``retry`` row (a
        failed attempt stamped by the last record) are due on the next
        sweep regardless of provider expiry — the hourly sweep cadence is
        the retry backoff bound, so a failed immediate delete is retried
        within the hour instead of waiting out the 24-hour expiry. Only
        ``live``/``requested`` rows still wait for their expiry.
        """
        now_text = db.format_dt(now)
        conn = db.connect(self._db)
        try:
            rows = conn.execute(
                "SELECT m.id, m.job_id, m.expires_at, m.delete_state"
                " FROM meta_files m JOIN jobs j ON m.job_id = j.id"
                " WHERE (m.delete_state IN ('pending', 'retry')"
                " OR (m.expires_at <= ? AND m.delete_state IN ('live', 'requested')))"
                f" AND {self._due_predicate()} ORDER BY m.expires_at, m.id",
                (now_text,),
            ).fetchall()
        finally:
            conn.close()
        return tuple(
            MetaFileRecord(
                id=MetaFileId(entry["id"]),
                job_id=JobId(entry["job_id"]),
                expires_at=db.parse_dt(entry["expires_at"]),
                delete_state=entry["delete_state"],
            )
            for entry in rows
        )

    def record_artifact_deletion(
        self,
        artifact_id: ArtifactId,
        outcome: Literal["tombstoned", "retry"],
        error: str | None = None,
    ) -> None:
        """Record one local deletion; a retry stays discoverable with its error."""
        if outcome not in ("tombstoned", "retry"):
            raise ValueError(f"outcome must be 'tombstoned' or 'retry', got {outcome!r}")
        conn = db.connect(self._db)
        try:
            found = conn.execute(
                "SELECT job_id FROM artifacts WHERE id = ?", (str(artifact_id),)
            ).fetchone()
            if found is None:
                raise ArtifactNotFound(f"unknown artifact {artifact_id}")
            now_text = db.format_dt(db.utcnow())
            if outcome == "tombstoned":
                conn.execute(
                    "UPDATE artifacts SET delete_state = 'tombstoned', deleted_at = ?,"
                    " hidden = 1 WHERE id = ?",
                    (now_text, str(artifact_id)),
                )
            else:
                conn.execute(
                    "UPDATE artifacts SET delete_state = 'retry' WHERE id = ?",
                    (str(artifact_id),),
                )
                if error:
                    conn.execute(
                        "INSERT INTO events (id, job_id, anon_id, session_id, event,"
                        " detail_json, created_at) VALUES (?, ?, NULL, NULL, ?, ?, ?)",
                        (
                            secrets.token_hex(16),
                            found["job_id"],
                            "artifact_delete_retry",
                            json.dumps({"artifact_id": str(artifact_id), "error": error}),
                            now_text,
                        ),
                    )
        finally:
            conn.close()

    def record_meta_file_deletion(
        self,
        meta_file_id: MetaFileId,
        outcome: Literal["verified_absent", "acknowledged_only", "retry"],
        error: str | None = None,
        *,
        attempt_started_at: datetime | None = None,
    ) -> None:
        """Record one Meta-file deletion; weaker or stale outcomes never win.

        Each outcome applies as one conditional ``UPDATE`` — a
        compare-and-set on the stored row — with two guards. First,
        monotonicity: ``retry``/presence evidence is cleared only by
        ``verified_absent`` (two-sided proof), never by
        ``acknowledged_only``, so a later unsupported read can never
        erase an earlier still-present record and close cancellation
        early; a terminal outcome (``verified_absent``,
        ``acknowledged_only``) never moves back to ``retry``, and
        ``verified_absent`` never downgrades to ``acknowledged_only``;
        exact replays are silent no-ops. Second, the attempt clock:
        every record stamps ``delete_requested_at``, and an attempt
        that started before the row's last stamp is stale and refused,
        so a stale acknowledged-only can never overwrite newer
        retry/present evidence. Callers that cannot date their attempt
        pass nothing and get monotonicity only. A refused outcome is a
        no-op recorded at debug level: the stronger or newer record
        stands, and there is nothing for the stale worker to do.
        """
        if outcome not in ("verified_absent", "acknowledged_only", "retry"):
            raise ValueError(f"unexpected meta-file deletion outcome {outcome!r}")
        attempt_text: str | None = None
        if attempt_started_at is not None:
            if type(attempt_started_at) is not datetime:
                raise ValueError("attempt_started_at must be a datetime")
            try:
                aware = attempt_started_at.utcoffset() is not None
            except Exception:
                aware = False
            if not aware:
                raise ValueError("attempt_started_at must be an aware datetime")
            attempt_text = db.format_dt(attempt_started_at)
        conn = db.connect(self._db)
        try:
            found = conn.execute(
                "SELECT id FROM meta_files WHERE id = ?", (str(meta_file_id),)
            ).fetchone()
            if found is None:
                raise MetaFileNotFound(f"unknown meta file {meta_file_id}")
            now_text = db.format_dt(db.utcnow())
            clock_guard = ""
            clock_args: tuple[str, ...] = ()
            if attempt_text is not None:
                clock_guard = (
                    " AND (delete_requested_at IS NULL"
                    " OR delete_requested_at <= ?)"
                )
                clock_args = (attempt_text,)
            row_id = str(meta_file_id)
            if outcome == "verified_absent":
                cursor = conn.execute(
                    "UPDATE meta_files SET delete_state = 'verified_absent',"
                    " verified_absent_at = ?, delete_requested_at = ?,"
                    " last_error = NULL WHERE id = ?"
                    " AND delete_state != 'verified_absent'" + clock_guard,
                    (now_text, now_text, row_id, *clock_args),
                )
            elif outcome == "acknowledged_only":
                cursor = conn.execute(
                    "UPDATE meta_files SET delete_state = 'acknowledged_only',"
                    " delete_requested_at = ?, last_error = ? WHERE id = ?"
                    " AND delete_state NOT IN"
                    " ('verified_absent', 'acknowledged_only', 'retry')"
                    + clock_guard,
                    (now_text, error, row_id, *clock_args),
                )
            else:
                cursor = conn.execute(
                    "UPDATE meta_files SET delete_state = 'retry',"
                    " delete_requested_at = ?, last_error = ? WHERE id = ?"
                    " AND delete_state NOT IN ('verified_absent', 'acknowledged_only')"
                    + clock_guard,
                    (now_text, error, row_id, *clock_args),
                )
            if cursor.rowcount == 0:
                # Refused by monotonicity or the attempt clock (or an
                # exact replay): the stronger or newer record stands.
                logger.debug(
                    "meta outcome %s for %s refused; kept stored record",
                    outcome,
                    row_id,
                )
        finally:
            conn.close()


def register_meta_file(
    job_id: JobId,
    meta_file_id: MetaFileId,
    expires_at: datetime,
    *,
    path: Path | str | None = None,
) -> None:
    """Record one uploaded Meta file as live (the upload module calls this after upload)."""
    conn = db.connect(path)
    try:
        conn.execute(
            "INSERT INTO meta_files (id, job_id, expires_at, delete_state,"
            " delete_requested_at, verified_absent_at, last_error)"
            " VALUES (?, ?, ?, 'live', NULL, NULL, NULL)",
            (str(meta_file_id), str(job_id), db.format_dt(expires_at)),
        )
    finally:
        conn.close()


def insert_call_record(record: CallRecord, *, path: Path | str | None = None) -> int:
    """Append one evidence row to ``model_calls`` (the evidence module's storage seam).

    Standalone rows keep a generated id; keyed receipts go through
    :meth:`JobContextImpl.record_call_with_event` instead.
    """
    cost = _money(record.estimated_cost_usd)
    conn = db.connect(path)
    try:
        cursor = conn.execute(
            "INSERT INTO model_calls (id, job_id, stage, model, endpoint, latency_ms,"
            " units_in, units_out, estimated_cost_usd, status, evidence_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                secrets.token_hex(16),
                str(record.job_id),
                record.stage,
                record.model,
                record.endpoint,
                record.latency_ms,
                record.units_in,
                record.units_out,
                str(cost),
                record.status,
                json.dumps(dict(record.evidence)),
                db.format_dt(db.utcnow()),
            ),
        )
        return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def job_reservation_spend_usd(
    job_id: JobId, *, path: Path | str | None = None
) -> tuple[Decimal, Decimal]:
    """Return ``(settled actuals, held estimates)`` for one job's calls.

    The durable retry-spend seam: settled rows report what was paid and
    held rows report what is still held, so a resumed run prices prior
    attempts from the ledger instead of re-pricing them at today's
    shares. Released rows stay excluded: a failed settlement freed
    those holds back to the budget, so they are genuinely unspent.
    """
    conn = db.connect(path)
    try:
        spent, held, _, _, _ = _job_spend(conn, str(job_id))
    finally:
        conn.close()
    return spent, held


def list_call_records(job_id: JobId, *, path: Path | str | None = None) -> tuple[CallRecord, ...]:
    """Return the evidence rows for one job, oldest first (the evidence module's read seam)."""
    conn = db.connect(path)
    try:
        rows = conn.execute(
            "SELECT job_id, stage, model, endpoint, latency_ms, units_in, units_out,"
            " estimated_cost_usd, status, evidence_json FROM model_calls"
            " WHERE job_id = ? ORDER BY rowid",
            (str(job_id),),
        ).fetchall()
    finally:
        conn.close()
    return tuple(
        CallRecord(
            job_id=JobId(entry["job_id"]),
            stage=entry["stage"],
            model=entry["model"],
            endpoint=entry["endpoint"],
            latency_ms=entry["latency_ms"],
            units_in=entry["units_in"],
            units_out=entry["units_out"],
            estimated_cost_usd=Decimal(entry["estimated_cost_usd"]),
            status=entry["status"],
            evidence=json.loads(entry["evidence_json"]),
        )
        for entry in rows
    )


def insert_event(
    event: str,
    detail: dict[str, object],
    *,
    job_id: JobId | None = None,
    anon_id: str | None = None,
    session_id: str | None = None,
    path: Path | str | None = None,
) -> int:
    """Append one evidence row to ``events`` (the evidence module's storage seam)."""
    conn = db.connect(path)
    try:
        cursor = conn.execute(
            "INSERT INTO events (id, job_id, anon_id, session_id, event, detail_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                secrets.token_hex(16),
                None if job_id is None else str(job_id),
                anon_id,
                session_id,
                event,
                json.dumps(detail),
                db.format_dt(db.utcnow()),
            ),
        )
        return int(cursor.lastrowid or 0)
    finally:
        conn.close()
