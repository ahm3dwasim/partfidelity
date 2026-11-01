"""Cancel, Meta deletion verification, and the hourly expired sweep.

Cancellation commits hide-first through
:meth:`UploadStore.request_cancellation`, deletes the preview through the
job's own artifact adapter and every attached Meta id through the
injected client, records each outcome, and recovers the job. A committed
claim without an attached id is recovered by its provider key before
anything is reported. The sweep runs isolated phases in order —
overdue-job expiry, the startup crash-leftover cleanup (root staging
plus job orphans), then recoverable uploads, due local artifacts, due Meta
files, expired delete requests, and expired export links — and a
fetch, item, or whole-phase failure in one never skips another. All
state is durable; nothing here keeps a process-global lifecycle flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from app import db
from app import export_links, guide_clip, persistence
from app.corrections import CorrectionStore
from app.admission import job_dir
from app.contracts import (
    JobId,
    LifecycleRepository,
    MetaFileId,
    MetaFilesClient,
    MetaFilesError,
    MetaReadUnsupported,
    UploadStore,
)
from app.ledger import expire_overdue_jobs, queue_failed_outputs, queue_guide_images
from app.providers import open_readback
from app.retention_config import failed_rescan_batch
from app.uploads import STAGING_MAX_AGE, run_startup_cleanup
from app.uploads_store import UploadError
from app.settings import DEFAULT_GUIDE_FILENAME, DEFAULT_CORRECTIONS_FILENAME

logger = logging.getLogger(__name__)

ArtifactOutcome = Literal["tombstoned", "retry"]
MetaOutcome = Literal["verified_absent", "acknowledged_only", "retry"]


class CancellationRetryError(RuntimeError):
    """The second check could not be stopped, so nothing was committed; retry."""


@dataclass(frozen=True)
class DeletionResult:
    """Outcome of one cancellation: the actual local and Meta results."""

    job_id: JobId
    local_deleted: bool
    meta_file_id: MetaFileId | None
    meta_outcome: str  # none | verified_absent | acknowledged_only | retry
    recorded_state: str  # final durable upload state
    error: str | None = None


@dataclass(frozen=True)
class SweepResult:
    """Counts for expiry, staging, deletion queues, and export links."""

    expired_jobs: int = 0
    staging_cleaned: int = 0
    recoverable_checked: int = 0
    recoverable_recovered: int = 0
    recoverable_retry: int = 0
    artifacts_checked: int = 0
    artifacts_tombstoned: int = 0
    artifacts_retry: int = 0
    meta_checked: int = 0
    meta_verified: int = 0
    meta_acknowledged: int = 0
    meta_retry: int = 0
    delete_requests_purged: int = 0
    export_links_purged: int = 0
    failures: tuple[str, ...] = ()
    public_upload_blocked: bool = False


def request_meta_deletion(
    meta_file_id: MetaFileId,
    *,
    client: MetaFilesClient,
    lifecycle: LifecycleRepository | None = None,
) -> MetaOutcome:
    """DELETE one Meta id, then prove absence through ``get_file``.

    Centralizes every Meta deletion: after a successful (idempotent)
    delete, ``None`` from a supported read is ``verified_absent``,
    ``MetaReadUnsupported`` is ``acknowledged_only``, and a still-present
    file or any read failure is ``retry`` — present evidence is never
    reported as acknowledged-only and the id is never discarded here.
    Every record carries this attempt's start time, so the ledger's
    compare-and-set refuses it when a newer outcome already landed; the
    returned outcome is still this attempt's own result, and the next
    sweep converges on the durable row.
    """
    attempt_started_at = db.utcnow()

    def _record(
        outcome: MetaOutcome, error: str | None = None,
    ) -> None:
        if lifecycle is not None:
            lifecycle.record_meta_file_deletion(
                meta_file_id,
                outcome,
                error=error,
                attempt_started_at=attempt_started_at,
            )

    try:
        client.delete_file(meta_file_id)
    except MetaFilesError as error:
        _record("retry", error=_short(error))
        return "retry"
    try:
        seen = client.get_file(meta_file_id)
    except MetaReadUnsupported:
        _record("acknowledged_only")
        return "acknowledged_only"
    except MetaFilesError as error:
        _record("retry", error=_short(error))
        return "retry"
    if seen is None:
        _record("verified_absent")
        return "verified_absent"
    _record("retry", error="still present after delete")
    return "retry"


def _recovery_expiry_within_guard(
    expires_at: Any, guard_deadline_utc: Any,
) -> datetime | None:
    """Return a recovered provider expiry only when it honors the guard.

    A lookup match is persisted with its provider expiry only when that
    expiry is aware and at or before the claim's persisted
    ``guard_deadline_utc``; anything else yields ``None`` so the caller
    queues the id under an immediate conservative deadline instead of
    storing a nonconforming provider promise.
    """
    if not isinstance(expires_at, datetime):
        return None
    try:
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            return None
    except Exception:
        return None
    if not isinstance(guard_deadline_utc, datetime):
        return None
    try:
        if guard_deadline_utc.tzinfo is None:
            return None
        if expires_at > guard_deadline_utc:
            return None
    except Exception:
        return None
    return expires_at


def _short(error: BaseException) -> str:
    """One sanitized internal summary line; never rendered as HTML."""
    text = f"{type(error).__name__}: {error}".strip()
    return text[:200] if text else type(error).__name__


def _recover_guarded(uploads: UploadStore, job_id: JobId, now: datetime) -> None:
    """Recover one job; jobs without an upload row are simply skipped."""
    try:
        uploads.recover_job(job_id, now)
    except (UploadError, LookupError):
        pass


def _cancel_claim_recovery(
    job_id: JobId,
    claim_id: str,
    guard_deadline_utc: Any,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    client: MetaFilesClient,
    now: datetime,
) -> tuple[MetaFileId | None, str]:
    """Recover one committed-but-unattached claim by its provider key.

    Returns the attached id and its Meta outcome, or ``(None, "retry")``
    while the claim is still unresolved: a definitive no-match stays
    queued through the guard deadline, and a lookup error stays
    retryable. The committed claim key remains the recovery handle
    either way, so the sweep converges without a second upload.
    """
    try:
        found = client.find_upload(claim_id)
    except MetaFilesError as error:
        logger.info(
            "cancel job=%s claim lookup failed: %s", job_id, type(error).__name__
        )
        return None, "retry"
    if found is None:
        return None, "retry"
    try:
        uploads.record_failure(
            job_id,
            f"cancel-recover:{claim_id}:{found.file_id}",
            "recovery",
            "cancellation recovered the committed claim to a provider file",
            now,
            meta_file_id=found.file_id,
            meta_expires_at=_recovery_expiry_within_guard(
                found.expires_at, guard_deadline_utc
            ),
        )
    except Exception as error:
        logger.warning(
            "cancel job=%s claim attach failed: %s", job_id, type(error).__name__
        )
        return None, "retry"
    return found.file_id, request_meta_deletion(
        found.file_id, client=client, lifecycle=lifecycle
    )


def delete_guide_files(
    job_id: JobId | str,
    *,
    guide_filename: str = DEFAULT_GUIDE_FILENAME,
    corrections_filename: str = DEFAULT_CORRECTIONS_FILENAME,
) -> None:
    """Revoke this guide's exports and remove its guide and correction sidecar."""
    guide_dir = job_dir(str(job_id), root=db.data_root())
    store = CorrectionStore(lambda _: guide_dir, corrections_filename, max_ops=1)
    with store.locked(str(job_id)):
        export_links.delete_guide(str(job_id))
        (guide_dir / guide_filename).unlink(missing_ok=True)
        (guide_dir / corrections_filename).unlink(missing_ok=True)


def cleanup_failed_guide(
    job_id: JobId | str, *, artifacts,
    guide_filename: str = DEFAULT_GUIDE_FILENAME,
    corrections_filename: str = DEFAULT_CORRECTIONS_FILENAME,
) -> None:
    """Queue canonical outputs first, then remove what can be removed now."""
    now = db.format_dt(db.utcnow())
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            guide_dir = job_dir(str(job_id), root=db.data_root())
            # Only what is still there: a no-op row would hold a cancellation open.
            queue_failed_outputs(
                conn, str(job_id), now, guide_filename, corrections_filename,
                present_in=guide_dir,
            )
            queue_guide_images(conn, str(job_id), now, guide_dir, guide_filename)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    for remove in (
        lambda: guide_clip.delete_job_clips(job_id, artifacts=artifacts),
        lambda: delete_guide_files(
            job_id, guide_filename=guide_filename,
            corrections_filename=corrections_filename,
        ),
    ):
        try:
            remove()
        except Exception as error:
            logger.warning("guide cleanup job=%s queued for retry: %s", job_id, type(error).__name__)


def cancel_and_delete(
    job_id: JobId,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    client: MetaFilesClient,
    guide_filename: str = DEFAULT_GUIDE_FILENAME,
    corrections_filename: str = DEFAULT_CORRECTIONS_FILENAME,
) -> DeletionResult:
    """Hide first, then delete the preview and every attached Meta id.

    Commits ``request_cancellation`` before any direct deletion, so a
    crash still leaves a hidden upload with pending, due Meta rows for
    the sweeper. A committed claim without an attached id is recovered
    by its provider key and reported as pending until it resolves —
    never as deleted. Reports the real outcomes: local success plus the
    per-id Meta result, with the final durable state.

    The background second check is stopped before cancellation commits,
    without waiting, so no queued send begins after it; if stopping it
    fails, nothing is committed and ``CancellationRetryError`` asks for a
    retry. The commit itself queues the preview and clips for the sweeper,
    due now, so they go even if the process exits right after it. While a
    second-check call is still in flight, they stay queued (the sweeper
    deletes them once it is quiet) and the result reports pending.
    """
    now = db.utcnow()
    try:
        quiet = open_readback.cancel(job_id)
    except Exception as error:
        logger.warning("cancel job=%s second check cancel failed: %s", job_id, type(error).__name__)
        raise CancellationRetryError("second check could not be stopped; retry") from error
    uploads.request_cancellation(
        job_id, now, guide_filename=guide_filename,
        corrections_filename=corrections_filename,
    )
    loaded = uploads.load_job(job_id)
    if loaded is None:  # pragma: no cover - cancellation just committed it
        raise UploadError(f"unknown upload job {job_id}")
    ctx, record = loaded
    errors: list[str] = []
    local_deleted = quiet
    if not quiet:
        errors.append("second check still running; local delete deferred")
        try:
            # Downloads are revoked now; the files are already queued for the sweeper.
            export_links.delete_guide(str(job_id))
        except Exception as error:
            logger.warning("cancel job=%s link revoke failed: %s", job_id, type(error).__name__)
    else:
        try:
            # The job's own adapter unlinks (ENOENT counts) and tombstones.
            ctx.artifacts.mark_deleted(record.preview_artifact_id)
        except Exception as error:
            local_deleted = False
            errors.append(f"preview delete failed: {_short(error)}")
            logger.warning("cancel job=%s preview delete failed", job_id)
            try:
                uploads.record_failure(
                    job_id,
                    f"cancel-local:{job_id}",
                    "local_delete",
                    errors[-1],
                    db.utcnow(),
                )
            except Exception as write_error:
                logger.warning(
                    "cancel job=%s failure write failed: %s",
                    job_id,
                    type(write_error).__name__,
                )
        try:
            # A published guide's clip goes with the footage: deleting the
            # guide deletes the clip, file and row together. Independent of
            # the preview delete above, so a preview failure never strands
            # the clip behind it.
            guide_clip.delete_job_clips(job_id, artifacts=ctx.artifacts)
        except Exception as error:
            local_deleted = False
            errors.append(f"clip delete failed: {_short(error)}")
            logger.warning("cancel job=%s clip delete failed", job_id)
            try:
                uploads.record_failure(
                    job_id,
                    f"cancel-clip:{job_id}",
                    "local_delete",
                    errors[-1],
                    db.utcnow(),
                )
            except Exception as write_error:
                logger.warning(
                    "cancel job=%s failure write failed: %s",
                    job_id,
                    type(write_error).__name__,
                )
        # Revoke downloads before removing the guide and its correction log.
        # A retry is harmless when a prior pass removed only some of these.
        try:
            delete_guide_files(job_id, guide_filename=guide_filename,
                               corrections_filename=corrections_filename)
            with db.connect() as conn:
                queued = conn.execute(
                    "SELECT id FROM artifacts WHERE job_id = ? AND kind IN"
                    " ('cancel_guide', 'cancel_corrections', 'cancel_export_links')"
                    " AND delete_state IN ('pending_delete', 'retry')", (str(job_id),)
                ).fetchall()
            for row in queued:
                lifecycle.record_artifact_deletion(row["id"], "tombstoned")
        except Exception as error:
            local_deleted = False
            errors.append(f"guide delete failed: {_short(error)}")
    meta_id = record.meta_file_id
    meta_outcome = "none"
    if meta_id is None and record.claim_id is not None:
        # Committed claim, no attached id: the provider may still hold
        # the file, so recover by key before reporting anything.
        meta_id, meta_outcome = _cancel_claim_recovery(
            job_id,
            record.claim_id,
            record.guard_deadline_utc,
            uploads=uploads,
            lifecycle=lifecycle,
            client=client,
            now=db.utcnow(),
        )
        if meta_outcome == "retry":
            errors.append("meta claim recovery needs retry")
    if meta_id is not None and meta_outcome == "none":
        try:
            meta_outcome = request_meta_deletion(
                meta_id, client=client, lifecycle=lifecycle
            )
        except Exception as error:
            meta_outcome = "retry"
            errors.append(f"meta delete failed: {_short(error)}")
            logger.warning("cancel job=%s meta delete failed", job_id)
            try:
                lifecycle.record_meta_file_deletion(
                    meta_id,
                    "retry",
                    error=errors[-1],
                    attempt_started_at=now,
                )
            except Exception as record_error:
                logger.warning(
                    "cancel job=%s meta record failed: %s",
                    job_id,
                    type(record_error).__name__,
                )
        if meta_outcome == "retry":
            errors.append("meta delete needs retry")
            try:
                uploads.record_failure(
                    job_id,
                    f"cancel-meta:{job_id}:{meta_id}",
                    "meta_delete",
                    "meta delete failed during cancellation",
                    db.utcnow(),
                    meta_file_id=meta_id,
                )
            except Exception as write_error:
                logger.warning(
                    "cancel job=%s failure write failed: %s",
                    job_id,
                    type(write_error).__name__,
                )
    try:
        final = uploads.recover_job(job_id, db.utcnow())
        recorded_state = final.state
    except Exception as error:
        logger.warning(
            "cancel job=%s recover failed: %s", job_id, type(error).__name__
        )
        recorded_state = "cancelling"
    error = "; ".join(errors) if errors else None
    persistence.request_snapshot()
    return DeletionResult(
        job_id=job_id,
        local_deleted=local_deleted,
        meta_file_id=meta_id,
        meta_outcome=meta_outcome,
        recorded_state=recorded_state,
        error=error,
    )


def _safe_job_dir(root: Path, job_id: str) -> Path | None:
    """Return ``<root>/jobs/<job_id>`` or ``None`` for an unsafe id."""
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        return None
    base = root / "jobs"
    candidate = base / job_id
    try:
        candidate.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    return candidate


def _safe_artifact_path(base_dir: Path, relative: Path) -> Path | None:
    """Join one artifact path, refusing absolute, dot, or escaping paths."""
    text = str(relative)
    if not text or text == "." or relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = base_dir / relative
    try:
        candidate.resolve().relative_to(base_dir.resolve())
    except (ValueError, OSError):
        return None
    return candidate


def _delete_one_file(path: Path) -> bool:
    """Unlink one regular file. ENOENT succeeds; directories are refused."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except IsADirectoryError:
        return False
    except OSError:
        return False


def _sweep_recoverable(
    now: datetime,
    *,
    uploads: UploadStore,
    client: MetaFilesClient,
    failures: list[str],
) -> tuple[int, int, int]:
    """Queue 1: expired previews and unknown-id claims, one row at a time."""
    try:
        rows = uploads.list_recoverable(now)
    except Exception as error:
        logger.warning("sweep recoverable list failed: %s", type(error).__name__)
        failures.append("recoverable-list")
        return 0, 0, 0
    checked = len(rows)
    recovered = 0
    retry = 0
    for row in rows:
        job_id = row.preview.job_id
        try:
            if row.claim_id is None:
                # Expired unclaimed preview: terminalize first; the same
                # pass's artifact queue then deletes and tombstones the
                # footage. No Meta call is made for a claimless row.
                uploads.recover_job(job_id, now)
                recovered += 1
                continue
            try:
                found = client.find_upload(row.claim_id)
            except MetaFilesError as error:
                logger.info(
                    "sweep lookup job=%s failed: %s",
                    job_id,
                    type(error).__name__,
                )
                retry += 1
                failures.append(f"lookup:{job_id}")
                continue
            if found is None:
                after = uploads.recover_job(
                    job_id, now, no_match_claim_id=row.claim_id
                )
                if after.claim_id is None:
                    recovered += 1
                else:
                    retry += 1
                continue
            # A returned id is always queued through known-id failure
            # recording. Its provider expiry is kept only when aware and
            # at or before the persisted claim guard; anything else is
            # queued under an immediate conservative deadline. The same
            # pass's Meta queue then deletes and verifies it.
            uploads.record_failure(
                job_id,
                f"sweep-recover:{row.claim_id}:{found.file_id}",
                "recovery",
                "recoverable claim resolved to a provider file",
                now,
                meta_file_id=found.file_id,
                meta_expires_at=_recovery_expiry_within_guard(
                    found.expires_at, row.guard_deadline_utc
                ),
            )
            uploads.recover_job(job_id, now)
            recovered += 1
        except Exception as error:
            logger.warning(
                "sweep recoverable job=%s failed: %s",
                job_id,
                type(error).__name__,
            )
            retry += 1
            failures.append(str(job_id))
    return checked, recovered, retry


def _sweep_artifacts(
    now: datetime,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    failures: list[str],
) -> tuple[int, int, int]:
    """Queue 2: due local artifacts, one regular file at a time."""
    try:
        rows = lifecycle.list_due_artifacts(now)
    except Exception as error:
        logger.warning("sweep artifact list failed: %s", type(error).__name__)
        failures.append("artifact-list")
        return 0, 0, 0
    root = db.data_root()
    checked = len(rows)
    tombstoned = 0
    retry = 0
    quiet: dict[JobId, bool] = {}
    for record in rows:
        try:
            if record.job_id not in quiet:
                quiet[record.job_id] = False  # stays False if the wait raises
                # Mark and drop at once; the deferred wait happens here, off the user's path.
                quiet[record.job_id] = (open_readback.cancel(record.job_id)
                                        or open_readback.wait_quiet(record.job_id))
            if not quiet[record.job_id]:
                # A second-check call may still read this job's files: keep them for the next sweep.
                lifecycle.record_artifact_deletion(
                    record.id, "retry", error="second check still running"
                )
                retry += 1
                failures.append(str(record.id))
                continue
            base_dir = _safe_job_dir(root, str(record.job_id))
            target = (
                _safe_artifact_path(base_dir, record.relative_path)
                if base_dir is not None
                else None
            )
            if target is None:
                lifecycle.record_artifact_deletion(
                    record.id, "retry", error="unsafe artifact path"
                )
                retry += 1
                failures.append(str(record.id))
                continue
            if record.kind.startswith("cancel_"):
                with db.connect() as conn:
                    sidecar = conn.execute(
                        "SELECT relative_path FROM artifacts WHERE job_id = ?"
                        " AND kind = 'cancel_corrections' LIMIT 1",
                        (str(record.job_id),),
                    ).fetchone()
                store = CorrectionStore(
                    lambda _: base_dir,
                    sidecar["relative_path"] if sidecar else DEFAULT_CORRECTIONS_FILENAME,
                    max_ops=1,
                )
                with store.locked(str(record.job_id)):
                    if record.kind == "cancel_export_links":
                        export_links.delete_guide(str(record.job_id))
                        deleted = True
                    else:
                        deleted = _delete_one_file(target)
            else:
                deleted = _delete_one_file(target)
            if deleted:
                lifecycle.record_artifact_deletion(record.id, "tombstoned")
                tombstoned += 1
            else:
                lifecycle.record_artifact_deletion(
                    record.id, "retry", error="local delete failed"
                )
                retry += 1
                failures.append(str(record.id))
        except Exception as error:
            logger.warning(
                "sweep artifact=%s failed: %s", record.id, type(error).__name__
            )
            try:
                lifecycle.record_artifact_deletion(
                    record.id, "retry", error="local delete failed"
                )
            except Exception as record_error:
                logger.warning(
                    "sweep artifact=%s record failed: %s",
                    record.id,
                    type(record_error).__name__,
                )
            retry += 1
            failures.append(str(record.id))
        _recover_guarded(uploads, record.job_id, now)
    return checked, tombstoned, retry


def _sweep_meta_files(
    now: datetime,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    client: MetaFilesClient,
    failures: list[str],
) -> tuple[int, int, int, int]:
    """Queue 3: due Meta files, one id at a time, ids never discarded."""
    try:
        rows = lifecycle.list_due_meta_files(now)
    except Exception as error:
        logger.warning("sweep meta list failed: %s", type(error).__name__)
        failures.append("meta-list")
        return 0, 0, 0, 0
    checked = len(rows)
    verified = 0
    acknowledged = 0
    retry = 0
    for record in rows:
        try:
            try:
                outcome = request_meta_deletion(
                    record.id, client=client, lifecycle=lifecycle
                )
            except Exception as error:
                logger.warning(
                    "sweep meta=%s delete failed: %s",
                    record.id,
                    type(error).__name__,
                )
                try:
                    lifecycle.record_meta_file_deletion(
                        record.id,
                        "retry",
                        error=_short(error),
                        attempt_started_at=now,
                    )
                except Exception as record_error:
                    logger.warning(
                        "sweep meta=%s record failed: %s",
                        record.id,
                        type(record_error).__name__,
                    )
                outcome = "retry"
            if outcome == "verified_absent":
                verified += 1
            elif outcome == "acknowledged_only":
                acknowledged += 1
            else:
                retry += 1
                failures.append(str(record.id))
                try:
                    uploads.record_failure(
                        record.job_id,
                        f"sweep-meta:{record.id}",
                        "meta_delete",
                        "meta delete retry; see lifecycle last_error",
                        now,
                        meta_file_id=record.id,
                    )
                except Exception as write_error:
                    logger.warning(
                        "sweep meta=%s failure write failed: %s",
                        record.id,
                        type(write_error).__name__,
                    )
        except Exception as error:
            logger.warning(
                "sweep meta=%s failed: %s", record.id, type(error).__name__
            )
            retry += 1
            failures.append(str(record.id))
        _recover_guarded(uploads, record.job_id, now)
    return checked, verified, acknowledged, retry


def _run_phase(
    label: str, sink: list[str], default: Any, func: Any, *args: Any, **kwargs: Any
) -> Any:
    """Run one sweep phase; a whole-phase failure never stops the others."""
    try:
        return func(*args, **kwargs)
    except Exception as error:
        logger.warning("sweep phase %s failed: %s", label, type(error).__name__)
        sink.append(label)
        return default


def _expire_phase(now: datetime) -> int:
    """Expire past-deadline open jobs so later phases can see them."""
    return len(expire_overdue_jobs(now))


def _rescan_failed_jobs(now: datetime) -> int:
    """Queue leftovers of failed jobs that no queue row covers.

    A lost image listing plus a process exit leaves frames and masks with no
    row. A guide still on disk is queued first; its images follow on a later
    pass once it is gone, so masks never go before their guide. Each job
    uses the file names its row recorded at admission. Once every stage is
    queued the job is marked done and never rescanned; unfinished jobs are
    taken in bounded batches, least recently scanned first, so a job that
    keeps failing never starves the rest.
    """
    # A grace hour so an in-flight after-failure cleanup is not raced.
    cutoff = db.format_dt(now - STAGING_MAX_AGE)
    now_text = db.format_dt(now)
    root = db.data_root()
    conn = db.connect()
    try:
        jobs = conn.execute(
            "SELECT id, guide_filename, corrections_filename FROM jobs"
            " WHERE state = 'failed' AND rescan_done_at IS NULL"
            " AND completed_at <= ?"
            " ORDER BY rescan_seen_at IS NOT NULL, rescan_seen_at, completed_at, id"
            " LIMIT ?",
            (cutoff, failed_rescan_batch()),
        ).fetchall()
        queued = 0
        for job in jobs:
            job_id = job["id"]
            guide_name = job["guide_filename"] or DEFAULT_GUIDE_FILENAME
            corrections_name = job["corrections_filename"] or DEFAULT_CORRECTIONS_FILENAME
            try:
                conn.execute("UPDATE jobs SET rescan_seen_at = ? WHERE id = ?",
                             (now_text, job_id))
                base_dir = _safe_job_dir(root, job_id)
                if base_dir is None or base_dir.is_symlink() or not base_dir.is_dir():
                    # Nothing this rescan may touch is left.
                    conn.execute("UPDATE jobs SET rescan_done_at = ? WHERE id = ?",
                                 (now_text, job_id))
                    continue
                conn.execute("BEGIN IMMEDIATE")
                try:
                    before = conn.total_changes
                    queue_failed_outputs(
                        conn, job_id, now_text, guide_name, corrections_name,
                        present_in=base_dir,
                    )
                    done = not (base_dir / guide_name).exists()
                    if done:
                        queue_guide_images(conn, job_id, now_text, base_dir, guide_name)
                    queued += conn.total_changes - before
                    if done:
                        conn.execute("UPDATE jobs SET rescan_done_at = ? WHERE id = ?",
                                     (now_text, job_id))
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            except Exception as error:
                logger.warning(
                    "failed job=%s rescan failed: %s", job_id, type(error).__name__
                )
    finally:
        conn.close()
    return queued


def sweep_expired(
    now: datetime,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    client: MetaFilesClient,
) -> SweepResult:
    """One hourly pass: expiry, crash leftovers, then five queues.

    Overdue-job expiry runs first (admitted jobs without upload rows
    must not hold capacity forever), then the shared startup cleanup
    (crashed uploads leave personal footage in root staging and in
    job-local orphan files outside every queue), then recoverable
    uploads (expired previews terminalized, unknown-id claims looked
    up by their committed key), due local artifacts, due Meta files,
    the bounded delete-request retention purge, and expired export links. Each phase is
    isolated: a fetch, item, or whole-phase failure in one never skips
    another.
    """
    failures: list[str] = []
    expired = _run_phase("expire-overdue", failures, 0, _expire_phase, now)
    staging = _run_phase(
        "staging-cleanup", failures, 0, run_startup_cleanup, now=now
    )
    _run_phase("failed-rescan", failures, 0, _rescan_failed_jobs, now)
    recoverable = _run_phase(
        "recoverable",
        failures,
        (0, 0, 0),
        _sweep_recoverable,
        now,
        uploads=uploads,
        client=client,
        failures=failures,
    )
    artifacts = _run_phase(
        "artifacts",
        failures,
        (0, 0, 0),
        _sweep_artifacts,
        now,
        uploads=uploads,
        lifecycle=lifecycle,
        failures=failures,
    )
    meta = _run_phase(
        "meta-files",
        failures,
        (0, 0, 0, 0),
        _sweep_meta_files,
        now,
        uploads=uploads,
        lifecycle=lifecycle,
        client=client,
        failures=failures,
    )
    try:
        purged = uploads.purge_delete_requests(now)
    except Exception as error:
        logger.warning("sweep delete-request purge failed: %s", type(error).__name__)
        purged = 0
        failures.append("delete-request-purge")
    links_purged = _run_phase(
        "export-links", failures, 0, export_links.purge_expired, now
    )
    try:
        blocked = uploads.public_uploads_blocked()
    except Exception as error:
        logger.warning(
            "sweep public-upload gate failed closed: %s", type(error).__name__
        )
        blocked = True
        failures.append("public-upload-gate")
    return SweepResult(
        expired_jobs=expired,
        staging_cleaned=staging,
        recoverable_checked=recoverable[0],
        recoverable_recovered=recoverable[1],
        recoverable_retry=recoverable[2],
        artifacts_checked=artifacts[0],
        artifacts_tombstoned=artifacts[1],
        artifacts_retry=artifacts[2],
        meta_checked=meta[0],
        meta_verified=meta[1],
        meta_acknowledged=meta[2],
        meta_retry=meta[3],
        delete_requests_purged=purged,
        export_links_purged=links_purged,
        failures=tuple(failures),
        public_upload_blocked=blocked,
    )
