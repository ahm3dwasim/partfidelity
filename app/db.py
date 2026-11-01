"""SQLite plumbing for the admission ledger.

Every connection is opened by :func:`connect` with WAL journal mode, foreign
keys on, and a bounded busy timeout, so no caller can observe or create a
connection with weaker settings.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

BUSY_TIMEOUT_MS = 5000

MIGRATION_001 = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"
MIGRATION_002 = Path(__file__).resolve().parent.parent / "migrations" / "002_uploads.sql"
MIGRATION_003 = Path(__file__).resolve().parent.parent / "migrations" / "003_delete_throttle.sql"
MIGRATION_004 = Path(__file__).resolve().parent.parent / "migrations" / "004_export_links.sql"
MIGRATION_005 = Path(__file__).resolve().parent.parent / "migrations" / "005_extract_retry_claim.sql"
MIGRATION_006 = Path(__file__).resolve().parent.parent / "migrations" / "006_failed_rescan.sql"

_DT_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

_configured_db: Path | None = None
_configured_root: Path | None = None


def configure(db_path: Path | str, data_root: Path | str) -> None:
    """Bind the database file and artifact root used by admission and ledger."""
    global _configured_db, _configured_root
    _configured_db = Path(db_path)
    _configured_root = Path(data_root)


def db_path() -> Path:
    """Return the configured database path, failing closed when unset."""
    if _configured_db is None:
        raise RuntimeError("database is not configured; call app.db.configure first")
    return _configured_db


def data_root() -> Path:
    """Return the configured artifact root, failing closed when unset."""
    if _configured_root is None:
        raise RuntimeError("data root is not configured; call app.db.configure first")
    return _configured_root


def _private_chmod(path: Path, mode: int) -> None:
    """Best-effort chmod; a platform that refuses keeps its own modes."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def ensure_private_storage(target: Path) -> None:
    """Enforce 0700/0600 on the data directory and DB/WAL/SHM files.

    Runs on every connection, so existing installations are repaired
    as well as new ones. (The service unit additionally sets
    ``UMask=0077`` at integration; this closes the gap in code.)
    """
    _private_chmod(target.parent, 0o700)
    _private_chmod(target, 0o600)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(target) + suffix)
        if sidecar.exists():
            _private_chmod(sidecar, 0o600)


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with WAL, foreign keys, and a bounded busy timeout.

    The connection runs in autocommit mode so writers use explicit
    ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` for short reservation
    transactions. This is the only sanctioned way to open the database.
    """
    target = Path(path) if path is not None else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS:d}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_private_storage(target)
    return conn


def init_db(
    path: Path | str | None = None,
    *,
    guide_filename: str | None = None,
    corrections_filename: str | None = None,
) -> None:
    """Create the schema and seed budget rows. Safe to re-run.

    Jobs created before migration 006 get the given output names (the
    deployment's configured ones; the defaults when omitted).
    """
    from app.admission import MAX_RUNNING
    from app.ledger import (  # deferred: ledger imports this module
        _finalize_job_failed,
        seed_budget_rows,
    )
    from app.settings import DEFAULT_CORRECTIONS_FILENAME, DEFAULT_GUIDE_FILENAME

    target = Path(path) if path is not None else db_path()
    conn = connect(target)
    try:
        for migration in (MIGRATION_001, MIGRATION_002, MIGRATION_003, MIGRATION_004):
            conn.executescript(migration.read_text(encoding="utf-8"))
        # Legacy races could admit two children for one retry window. Settle
        # the losing open jobs and install the claim index in one write lock.
        conn.execute("BEGIN IMMEDIATE")
        try:
            duplicate_labels = conn.execute(
                "SELECT anon_id FROM jobs WHERE anon_id LIKE '%/extract/retry/%'"
                " AND ip_hash = anon_id GROUP BY anon_id HAVING COUNT(*) > 1"
            ).fetchall()
            now_text = format_dt(utcnow())
            for label in duplicate_labels:
                rows = conn.execute(
                    "SELECT id FROM jobs WHERE anon_id = ? AND ip_hash = anon_id"
                    " ORDER BY (state = 'done') DESC, created_at, id",
                    (label["anon_id"],),
                ).fetchall()
                for row in rows[1:]:
                    job_id = row["id"]
                    _finalize_job_failed(conn, job_id, now_text, MAX_RUNNING)
                    conn.execute(
                        "UPDATE jobs SET anon_id = anon_id || '/duplicate/' || id WHERE id = ?",
                        (job_id,),
                    )
                    conn.execute(
                        "INSERT INTO events (id, job_id, anon_id, session_id, event, detail_json, created_at)"
                        " VALUES (?, ?, ?, NULL, 'retry_claim_duplicate', ?, ?)",
                        (secrets.token_hex(16), job_id, label["anon_id"],
                         json.dumps({"reason": "duplicate extract retry claim"}), now_text),
                    )
            conn.execute(MIGRATION_005.read_text(encoding="utf-8"))
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "rescan_done_at" not in columns:
                for statement in MIGRATION_006.read_text(encoding="utf-8").split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute(
                    "UPDATE jobs SET guide_filename = ?, corrections_filename = ?"
                    " WHERE guide_filename IS NULL",
                    (guide_filename or DEFAULT_GUIDE_FILENAME,
                     corrections_filename or DEFAULT_CORRECTIONS_FILENAME),
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        seed_budget_rows(conn)
    finally:
        conn.close()


def utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def format_dt(moment: datetime) -> str:
    """Serialize a datetime as UTC ISO-8601 text (naive values mean UTC)."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).strftime(_DT_FORMAT)


def parse_dt(text: str) -> datetime:
    """Parse :func:`format_dt` output back to an aware UTC datetime."""
    return datetime.strptime(text, _DT_FORMAT).replace(tzinfo=timezone.utc)


def utc_day(moment: datetime) -> str:
    """Return the UTC calendar day (``YYYY-MM-DD``) for a datetime."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).strftime("%Y-%m-%d")


def seconds_ago(now: datetime, seconds: int) -> str:
    """Return ``now - seconds`` serialized for timestamp comparisons."""
    aware = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return format_dt(aware - timedelta(seconds=seconds))
