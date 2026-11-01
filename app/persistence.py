"""Snapshot-and-reconcile persistence for a one-container Modal deploy.

The working SQLite database lives on the container's local disk. A Volume
never holds a live WAL: it holds only whole, finished files, so no Modal
commit guarantee is needed for consistency.

* :func:`take_snapshot` copies the live database with the sqlite3 online
  backup API (one read transaction, so the copy is transaction-consistent)
  into a temp file beside the target, fsyncs it, then renames it to
  ``db.snapshot`` and keeps the previous one as ``db.snapshot.prev``.
* :func:`restore_latest` copies the newest snapshot that opens and passes
  ``PRAGMA integrity_check`` onto local disk (else ``.prev``, else empty).
* :func:`reconcile` expires jobs with missing live artifacts and removes
  files unreferenced by surviving artifacts or canonical guide records.

Changes after the last completed snapshot can be lost. Snapshot freshness
is reported by the host's health check; it does not stop writes.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import sqlite3
import stat
import struct
import threading
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from app import db

logger = logging.getLogger(__name__)

SNAPSHOT_NAME = "db.snapshot"
PREV_NAME = "db.snapshot.prev"
JOBS_DIRNAME = "jobs"

class SnapshotError(Exception):
    """A snapshot could not be taken; the previous one is left untouched."""


def atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete, fsynced text file beside its destination."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        _fsync_path(path)
    except OSError:  # a platform that cannot fsync a directory
        pass


def is_good_database(path: Path) -> bool:
    """True when ``path`` opens as SQLite and passes ``integrity_check``."""
    try:
        conn = sqlite3.connect(str(path), timeout=5)
        try:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return rows == [("ok",)]


def take_snapshot(live_db: Path | str, snap_dir: Path | str) -> Path:
    """Write a consistent copy of ``live_db`` as ``snap_dir/db.snapshot``."""
    live = Path(live_db)
    target_dir = Path(snap_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / f".{SNAPSHOT_NAME}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        # mode=rw never creates: a missing live file must fail, not snapshot
        # an empty database over the last good one.
        src = sqlite3.connect(
            f"{live.resolve().as_uri()}?mode=rw", uri=True,
            timeout=db.BUSY_TIMEOUT_MS / 1000,
        )
        try:
            dst = sqlite3.connect(str(tmp))
            try:
                src.backup(dst)
                # A self-contained file: no WAL sidecars appear on the Volume.
                dst.execute("PRAGMA journal_mode=DELETE")
            finally:
                dst.close()
        finally:
            src.close()
        if not is_good_database(tmp):
            raise SnapshotError("fresh snapshot failed integrity_check")
        _fsync_path(tmp)
        current = target_dir / SNAPSHOT_NAME
        # Rotate only a snapshot that is itself good, so a corrupt current
        # one can never replace the last good .prev.
        if current.exists() and is_good_database(current):
            os.replace(current, target_dir / PREV_NAME)
        os.replace(tmp, current)
        _fsync_dir(target_dir)
        return current
    except SnapshotError:
        raise
    except (sqlite3.Error, OSError) as error:
        raise SnapshotError(f"snapshot failed: {type(error).__name__}") from error
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class RestoreResult:
    source: str  # "snapshot" | "prev" | "empty"
    rejected: tuple[str, ...] = ()


def _remove_db_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def restore_latest(live_db: Path | str, snap_dir: Path | str) -> RestoreResult:
    """Put the newest good snapshot at ``live_db`` (local disk); else empty."""
    live = Path(live_db)
    live.parent.mkdir(parents=True, exist_ok=True)
    fd, scratch_name = tempfile.mkstemp(dir=live.parent, prefix=live.name + ".restore-")
    os.close(fd)
    scratch = Path(scratch_name)
    rejected: list[str] = []
    inaccessible = False
    try:
        for name, label in ((SNAPSHOT_NAME, "snapshot"), (PREV_NAME, "prev")):
            candidate = Path(snap_dir) / name
            try:
                shutil.copyfile(candidate, scratch)
            except FileNotFoundError:
                continue
            except OSError:
                inaccessible = True
                rejected.append(name)
                continue
            # Checked on the local copy: the Volume file is never opened.
            if is_good_database(scratch):
                _remove_db_files(live)
                os.replace(scratch, live)
                return RestoreResult(label, tuple(rejected))
            rejected.append(name)
        if inaccessible:
            raise SnapshotError("no usable snapshot and a candidate was inaccessible")
        _remove_db_files(live)
        return RestoreResult("empty", tuple(rejected))
    finally:
        scratch.unlink(missing_ok=True)


@dataclass(frozen=True)
class ReconcileResult:
    expired_rows: int = 0
    orphan_dirs: int = 0
    temp_files: int = 0

    def log_line(self, source: str) -> str:
        return (
            f"reconcile: restored={source} expired_rows={self.expired_rows}"
            f" orphan_dirs={self.orphan_dirs} temp_files={self.temp_files}"
        )


def _artifact_file(data_root: Path, job_id: str, relative: str) -> Path | None:
    base = data_root / JOBS_DIRNAME / job_id
    target = base / relative
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return None
    return target


def _incomplete_png(target: Path) -> bool:
    """True for a public .png the app's own reader cannot decode (crash leftover)."""
    if target.suffix.lower() != ".png":
        return False
    from app.render import read_png_rgb

    try:
        read_png_rgb(target)
    except (ValueError, zlib.error, struct.error):
        return True
    except OSError:
        return False
    return False


def _expire_missing_files(
    conn: sqlite3.Connection, data_root: Path, now_text: str
) -> int:
    from app.admission import MAX_RUNNING
    from app.ledger import _close_job_hold, _day_row_for_job, _promote_waiting, _release_unsettled_calls

    rows = conn.execute(
        "SELECT id, job_id, relative_path FROM artifacts"
        " WHERE delete_state = 'live'"
    ).fetchall()
    expired = 0
    released = False
    for row in rows:
        target = _artifact_file(data_root, row["job_id"], row["relative_path"])
        if target is not None:
            try:
                if stat.S_ISREG(target.stat().st_mode):
                    if not _incomplete_png(target):
                        continue
                    target.unlink()
            except (FileNotFoundError, NotADirectoryError):
                pass
        conn.execute(
            "UPDATE artifacts SET delete_state = 'tombstoned', deleted_at = ?,"
            " hidden = 1 WHERE id = ?",
            (now_text, row["id"]),
        )
        conn.execute(
            "UPDATE uploads SET hidden = 1, updated_at = ?"
            " WHERE preview_artifact_id = ? AND hidden = 0",
            (now_text, row["id"]),
        )
        job = conn.execute("SELECT state, created_at FROM jobs WHERE id = ?", (row["job_id"],)).fetchone()
        if job is not None and job["state"] in ("waiting", "active"):
            _release_unsettled_calls(conn, row["job_id"])
            _close_job_hold(conn, row["job_id"], _day_row_for_job(conn, job["created_at"]))
            released = True
        conn.execute(
            "UPDATE jobs SET state = 'expired', completed_at = ? WHERE id = ?",
            (now_text, row["job_id"]),
        )
        expired += 1
    if released:
        _promote_waiting(conn, now_text, MAX_RUNNING)
    return expired


def _remove_tree(path: Path) -> bool:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        return True
    except OSError as error:
        raise OSError(f"reconcile could not remove {path.name}: {error}") from error


def _remove_orphans(conn: sqlite3.Connection, data_root: Path, canonical: tuple[str, ...],
                    guide_sidecars: tuple[str, ...] = ()) -> tuple[int, int]:
    jobs_dir = data_root / JOBS_DIRNAME
    if not jobs_dir.is_dir():
        return 0, 0
    known = {row["id"] for row in conn.execute("SELECT id FROM jobs")}
    references = set()
    for row in conn.execute("SELECT job_id, relative_path FROM artifacts WHERE delete_state != 'tombstoned'"):
        target = _artifact_file(data_root, row["job_id"], row["relative_path"])
        if target is not None:
            references.add(target.absolute())
    orphan_dirs = temp_files = 0
    for entry in jobs_dir.iterdir():
        if entry.name not in known:
            orphan_dirs += _remove_tree(entry)
            continue
        if not entry.is_dir() or entry.is_symlink():
            continue
        guide_path = entry / canonical[0]
        kept = canonical + (guide_sidecars if guide_path.is_file() else ())
        if guide_path.is_file():
            try:
                guide = json.loads(guide_path.read_text(encoding="utf-8"))
                paths = list(guide.get("frame_paths", []))
                for label in guide.get("labels", []):
                    for section, key in (("detection", "mask_path"), ("provisional", "rendered_image_path")):
                        if label.get(section):
                            paths.append(label[section][key])
                for value in paths:
                    candidate = Path(value)
                    if candidate.is_absolute() and candidate.resolve().is_relative_to(entry.resolve()):
                        references.add(candidate.absolute())
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                logger.warning("reconcile could not read guide references for %s", entry.name)
                continue
        for leftover in entry.rglob("*"):
            if leftover.is_symlink() or (leftover.is_file() and
                    leftover.absolute() not in references and
                    not (leftover.parent == entry and leftover.name in kept)):
                temp_files += _remove_tree(leftover)
    return orphan_dirs, temp_files


def reconcile(
    db_path: Path | str, data_root: Path | str, now: datetime | None = None,
    *, canonical: tuple[str, ...] | None = None, guide_sidecars: tuple[str, ...] = (),
) -> ReconcileResult:
    """Make rows and files agree; idempotent, safe to run on every boot."""
    root = Path(data_root)
    if canonical is None:
        from app.settings import DEFAULT_GUIDE_FILENAME, DEFAULT_PENDING_SETTLEMENT_FILENAME
        canonical = (DEFAULT_GUIDE_FILENAME, DEFAULT_PENDING_SETTLEMENT_FILENAME)
    now_text = db.format_dt(now or db.utcnow())
    conn = db.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            expired = _expire_missing_files(conn, root, now_text)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        # Orphans are judged against the committed rows, after expiry.
        orphan_dirs, temp_files = _remove_orphans(conn, root, canonical, guide_sidecars)
    finally:
        conn.close()
    return ReconcileResult(expired, orphan_dirs, temp_files)


@dataclass
class Snapshotter:
    """Background loop: snapshot every ``interval_s`` and on request."""

    live_db: Path
    snap_dir: Path
    interval_s: float
    settle_s: float
    commit: Callable[[], None] | None = None
    last_ok: float | None = field(default=None, init=False)
    failures: int = field(default=0, init=False)
    _wake: threading.Event = field(default_factory=threading.Event, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def snapshot_now(self) -> bool:
        """Take one snapshot then commit the Volume; False on failure."""
        with self._lock:
            try:
                take_snapshot(self.live_db, self.snap_dir)
                if self.commit is not None:
                    self.commit()
            except Exception as error:
                self.failures += 1
                logger.warning("snapshot failed: %s", type(error).__name__)
                return False
            self.last_ok = time.monotonic()
            return True

    def request(self) -> None:
        """Ask for a snapshot soon (after a guide is published or deleted)."""
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            woken = self._wake.wait(self.interval_s)
            if self._stop.is_set():
                return
            if woken:
                # Let the transaction that triggered the request finish.
                self._stop.wait(self.settle_s)
                self._wake.clear()
            self.snapshot_now()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="pf-snapshot", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and take a final snapshot."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                logger.warning("snapshot worker still running; skipping final snapshot")
                return
        self.snapshot_now()


_active: Snapshotter | None = None


def install(snapshotter: Snapshotter | None) -> None:
    global _active
    _active = snapshotter


def request_snapshot() -> None:
    """Hook for guide publish/delete; a no-op outside a hosted deploy."""
    active = _active
    if active is not None:
        active.request()
