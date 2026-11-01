"""Guide-clip retention: the published guide plays its own clip.

When a job's guide is stored, its confirmed preview is copied beside the
guide as the guide clip and recorded as a ``guide_clip`` artifact row, so
the clip is found by guide id and lives as long as the guide. The sweeper
never lists this kind (see ``LifecycleRepositoryImpl.list_due_artifacts``);
only guide deletion (cancellation) removes the file and tombstones the row.
The 24-hour preview promise is untouched: preview rows keep their own
expiry and the preview page still hides on it, with or without a guide.

Copies are crash-safe: the row is registered and marked pending before
the copy, then promoted live after it, so a crash leaves a tracked
pending row the sweeper deletes, never an untracked file.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

from app import db
from app.admission import job_dir
from app.contracts import (
    GUIDE_CLIP_ARTIFACT_KIND,
    ArtifactId,
    ArtifactRecord,
    ArtifactStore,
    JobId,
    UploadStore,
)
from app.ledger import ARTIFACT_RETENTION_DAYS, ArtifactNotFound, active_job_publication
from app.settings import DEFAULT_GUIDE_CLIP_FILENAME

logger = logging.getLogger(__name__)


class GuideClipError(Exception):
    """The guide clip could not be retained; the caller fails the job."""


def _valid_clip_filename(name: str | None) -> str:
    """Return one validated clip filename, or raise :class:`GuideClipError`.

    Plain filenames only: a separator, dot name, or null byte would
    escape the job directory or crash the copy with a ValueError the
    pipeline does not catch.
    """
    if name is None:
        return DEFAULT_GUIDE_CLIP_FILENAME
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise GuideClipError(f"invalid guide-clip filename: {name!r}")
    return name


def _copy_atomic(source: Path, dest: Path, job_id: JobId | str) -> None:
    """Copy one file atomically; a reader never sees a truncated clip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=".guide-clip-", suffix=".mp4"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp)
        with active_job_publication(job_id) as active:
            if not active:
                raise GuideClipError(f"job {job_id} ended before clip publication")
            tmp.replace(dest)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _clip_records(entries) -> tuple[ArtifactRecord, ...]:
    """Map raw artifact rows onto records, oldest first as selected."""
    return tuple(
        ArtifactRecord(
            id=ArtifactId(entry["id"]),
            job_id=JobId(entry["job_id"]),
            kind=entry["kind"],
            relative_path=Path(entry["relative_path"]),
            expires_at=db.parse_dt(entry["expires_at"]),
        )
        for entry in entries
    )


def find_live_clips(
    job_id: JobId | str, *, db_path: Path | str | None = None
) -> tuple[ArtifactRecord, ...]:
    """Return one job's live guide-clip rows, oldest first."""
    conn = db.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, job_id, kind, relative_path, expires_at FROM artifacts"
            " WHERE job_id = ? AND kind = ? AND delete_state = 'live'"
            " ORDER BY created_at, id",
            (str(job_id), GUIDE_CLIP_ARTIFACT_KIND),
        ).fetchall()
    finally:
        conn.close()
    return _clip_records(rows)


def find_deletable_clips(
    job_id: JobId | str, *, db_path: Path | str | None = None
) -> tuple[ArtifactRecord, ...]:
    """Return one job's clip rows due for immediate cleanup, oldest first.

    Live rows plus rows a failed delete left behind: ``pending_delete``
    from a ``mark_deleted`` whose unlink raised, ``retry`` from a recorded
    failure. A transient unlink failure never strands the clip until its
    expiry; the next cancellation retries it at once.
    """
    conn = db.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, job_id, kind, relative_path, expires_at FROM artifacts"
            " WHERE job_id = ? AND kind = ?"
            " AND delete_state IN ('live', 'pending_delete', 'retry')"
            " ORDER BY created_at, id",
            (str(job_id), GUIDE_CLIP_ARTIFACT_KIND),
        ).fetchall()
    finally:
        conn.close()
    return _clip_records(rows)


def resolve_clip_path(
    record: ArtifactRecord, *, data_root: Path | str | None = None
) -> Path | None:
    """Return one clip row's file, or ``None`` when it is unsafe or gone.

    Refuses absolute, dot, and escaping paths, symlinks, and missing
    files: a hostile or stale row never serves another file.
    """
    relative = record.relative_path
    text = str(relative)
    if not text or text == "." or relative.is_absolute() or ".." in relative.parts:
        return None
    root = Path(data_root) if data_root is not None else db.data_root()
    base = job_dir(str(record.job_id), root=root)
    candidate = base / relative
    try:
        candidate.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    if candidate.is_symlink() or not candidate.is_file():
        return None
    return candidate


def load_guide_clip(
    job_id: JobId | str,
    *,
    db_path: Path | str | None = None,
    data_root: Path | str | None = None,
) -> Path | None:
    """Return one guide's playable clip file, or ``None`` when there is none.

    The newest live row whose file resolves wins; rows whose files are
    gone or unsafe are skipped, so a half-deleted clip 404s instead of
    serving the wrong bytes.
    """
    for record in reversed(find_live_clips(job_id, db_path=db_path)):
        path = resolve_clip_path(record, data_root=data_root)
        if path is not None:
            return path
    return None


def _mark_clip_pending(artifact_id: ArtifactId) -> bool:
    """Mark one fresh clip row pending; return whether it was still live.

    Pending rows wait as ``pending_delete`` with an immediate expiry, so
    the sweeper and the cancel path already treat them as due: a crash
    between here and the promotion leaves a tracked row whose file the
    next sweep removes. Only a live row moves; a concurrent delete wins.
    """
    conn = db.connect()
    try:
        cursor = conn.execute(
            "UPDATE artifacts SET delete_state = 'pending_delete',"
            " expires_at = ? WHERE id = ? AND delete_state = 'live'",
            (db.format_dt(db.utcnow()), str(artifact_id)),
        )
        return cursor.rowcount == 1
    finally:
        conn.close()


def _promote_clip_row(artifact_id: ArtifactId) -> bool:
    """Flip one pending clip row live; return whether it was still pending.

    The promotion carries the standard retention expiry, the same shape
    a fresh registration holds. A concurrent delete wins: when the row
    already left pending, the caller removes its own copy instead.
    """
    expires = db.utcnow() + timedelta(days=ARTIFACT_RETENTION_DAYS)
    conn = db.connect()
    try:
        cursor = conn.execute(
            "UPDATE artifacts SET delete_state = 'live', expires_at = ?"
            " WHERE id = ? AND delete_state = 'pending_delete'"
            " AND EXISTS (SELECT 1 FROM jobs j LEFT JOIN uploads u ON u.job_id = j.id"
            " WHERE j.id = artifacts.job_id AND j.state = 'active'"
            " AND j.deadline_at > ? AND u.cancel_requested_at IS NULL)",
            (db.format_dt(expires), str(artifact_id), db.format_dt(db.utcnow())),
        )
        return cursor.rowcount == 1
    finally:
        conn.close()


def _best_effort_delete(ctx, artifact_id: ArtifactId) -> None:
    """Tombstone one clip row; a failed cleanup never masks its error."""
    try:
        ctx.artifacts.mark_deleted(artifact_id)
    except Exception:
        pass


def _best_effort_unlink(dest: Path) -> None:
    """Remove one copied clip; a failed cleanup never masks its error."""
    try:
        dest.unlink(missing_ok=True)
    except OSError:
        pass


def retain_guide_clip(
    ctx,
    uploads: UploadStore,
    *,
    filename: str | None = None,
) -> bool:
    """Copy one job's confirmed preview into its guide clip; return retained.

    ``False`` is the clean skip for jobs without an upload row. A job
    whose upload row exists but whose preview file is gone raises
    :class:`GuideClipError` — footage the publish promised is missing,
    so the caller fails closed. Already retained clips are idempotent.
    The row is registered and marked pending before the copy, then
    promoted live after it: a crash in between leaves a tracked pending
    row the sweeper deletes, never an untracked file. Failed copies and
    failed or raced promotions remove their row and file before raising,
    so a retry never finds a half-kept clip.
    """
    name = _valid_clip_filename(filename)
    job_id = ctx.job_id
    for record in find_live_clips(job_id):
        if resolve_clip_path(record) is not None:
            return True
    loaded = uploads.load_job(JobId(str(job_id)))
    if loaded is None:
        return False
    _, record = loaded
    source = record.preview.local_video
    if source.is_symlink() or not source.is_file():
        raise GuideClipError(f"job {job_id} preview file is gone")
    dest = job_dir(str(job_id)) / name
    try:
        artifact_id = ctx.artifacts.register(GUIDE_CLIP_ARTIFACT_KIND, dest)
    except Exception as error:
        raise GuideClipError(
            f"job {job_id} clip registration failed: {type(error).__name__}"
        ) from error
    # The register-then-mark window holds a live row with no file yet: a
    # crash there leaves a tracked row the loader skips (no file) and the
    # cancel or expiry paths delete, never an untracked file.
    try:
        pending = _mark_clip_pending(artifact_id)
    except Exception as error:
        _best_effort_delete(ctx, artifact_id)
        raise GuideClipError(
            f"job {job_id} clip registration failed: {type(error).__name__}"
        ) from error
    if not pending:
        _best_effort_delete(ctx, artifact_id)
        raise GuideClipError(f"job {job_id} clip deleted during retain; retry")
    try:
        _copy_atomic(source, dest, job_id)
    except (OSError, GuideClipError) as error:
        _best_effort_delete(ctx, artifact_id)
        raise GuideClipError(f"job {job_id} clip copy failed: {error}") from error
    if dest.is_symlink() or not dest.is_file():  # pragma: no cover - raced unlink
        _best_effort_delete(ctx, artifact_id)
        raise GuideClipError(f"job {job_id} clip copy vanished")
    try:
        promoted = _promote_clip_row(artifact_id)
    except Exception as error:
        _best_effort_unlink(dest)
        _best_effort_delete(ctx, artifact_id)
        raise GuideClipError(
            f"job {job_id} clip registration failed: {type(error).__name__}"
        ) from error
    if not promoted:
        # A concurrent delete tombstoned the pending row; our copy is an
        # orphan, so remove it and fail closed instead of reviving the row.
        _best_effort_unlink(dest)
        raise GuideClipError(f"job {job_id} clip deleted during retain; retry")
    logger.info("guide clip job=%s retained", job_id)
    return True


def delete_job_clips(
    job_id: JobId | str,
    *,
    artifacts: ArtifactStore,
    db_path: Path | str | None = None,
) -> int:
    """Delete one job's guide clips; return the rows tombstoned.

    Each due row goes through the adapter's ``mark_deleted`` (ENOENT
    counts), so files and rows converge together — including rows a
    failed delete left pending, which are due at once, not at expiry.
    A row that vanishes mid-delete was already converged and is skipped.
    """
    removed = 0
    for record in find_deletable_clips(job_id, db_path=db_path):
        try:
            artifacts.mark_deleted(record.id)
        except ArtifactNotFound:
            continue
        removed += 1
    return removed


__all__ = (
    "GuideClipError",
    "delete_job_clips",
    "find_deletable_clips",
    "find_live_clips",
    "load_guide_clip",
    "resolve_clip_path",
    "retain_guide_clip",
)
