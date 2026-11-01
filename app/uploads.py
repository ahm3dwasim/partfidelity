"""Bounded owned-footage ingest: staging, validation, preview, confirm.

``prepare_preview`` runs only on an admitted context and writes only
beneath that context's job directory; every post-admission failure
finishes the job failed and leaves no live artifact behind.
``confirm_and_send`` claims first (exactly one winner calls Meta),
enforces the pre-call deadline and the request/response expiry bounds,
and durably queues every known id for deletion instead of handing a bad
id to the readback module. Time comes from :func:`app.db.utcnow` so deployments and the
ledger module's store share one clock; tests point it at a fake clock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Literal

from app import db
from app.admission import job_dir as _job_dir
from app.contracts import (
    META_UPLOAD_MAX_HOURS,
    UPLOAD_PREVIEW_TTL_HOURS,
    ArtifactId,
    ConfirmationClaim,
    JobContext,
    JobId,
    MetaFileId,
    MetaFilesClient,
    MetaFilesError,
    MetaUpload,
    MetaUploadExpiryError,
    UploadPreview,
    UploadStore,
)
from app.meta_files import expiry_details
from app.settings import DEFAULT_MAX_VIDEO_S
from app.uploads_store import (
    ClaimExpiredError,
    ClaimTerminalError,
    ClaimWaitingError,
    UploadError,
)

logger = logging.getLogger(__name__)

MAX_DURATION_S = DEFAULT_MAX_VIDEO_S
MAX_BYTES = 100 * 1024 * 1024
# Client metadata-probe ceiling: a probe that never resolves settles as
# unmeasurable (the server still enforces the limits) instead of wedging
# the form. Served to upload.js through PARTFIDELITY_LIMITS.
PROBE_TIMEOUT_MS = 10_000


def over_limit_message(max_duration_s: float) -> str:
    """Settled refusal copy naming the active duration cap."""
    return (
        f"Trim it to {max_duration_s:g} seconds or less and try again. "
        "A 720p phone video is usually under 100 MB."
    )


OVER_LIMIT_MESSAGE = over_limit_message(DEFAULT_MAX_VIDEO_S)
PROCESSED_FILENAME = "preview.mp4"
STAGING_DIRNAME = ".staging"
CHUNK_BYTES = 1024 * 1024

# Staged raw footage older than this is a crashed upload's leftover: a
# live request stages, validates, and discards within seconds, so an
# hour-old staged file can only come from a SIGKILL or reboot between
# staging and cleanup. The sweep and first-request cleanup delete only
# past this age, never in-flight bytes.
STAGING_MAX_AGE = timedelta(hours=1)

# ffprobe container names accepted as owned phone footage. The check uses
# the probed container plus a real video stream, never the filename or the
# client-supplied content type.
ALLOWED_CONTAINERS = frozenset({"mp4", "mov", "m4v"})

# ffprobe video codec names accepted as real footage. Anything else —
# still-image codecs smuggled in a video container, unknown codecs —
# is refused, since transcoding it would not prove real video.
ALLOWED_VIDEO_CODECS = frozenset({"h264", "hevc", "mpeg4", "vp8", "vp9", "av1"})

# A clip shorter or emptier than this is not meaningful footage: one
# frame, a rounding-error duration, or a non-positive/non-finite probe
# value is refused with the settled message.
MIN_DURATION_S = 0.5
MIN_VIDEO_FRAMES = 2

CONFIRM_CONFLICT_COPY = (
    "This upload was already confirmed, cancelled, or failed. "
    "Start a new upload to try again."
)
CONFIRM_FAILED_COPY = (
    "We could not send your video to Meta. Your preview is hidden; "
    "start a new upload to try again."
)
CONFIRM_STALE_COPY = (
    "Confirmation took too long and the upload window passed. "
    "Start a new upload to try again."
)
CONFIRM_CANCELLED_COPY = (
    "This upload was cancelled while sending. Your files are being deleted."
)
CONFIRM_WAITING_COPY = (
    "Your upload is still waiting for a free worker. "
    "Wait a moment and try again."
)

# The provider is asked for one hour less than the contract maximum, so
# a real-clock grant always lands inside the confirmation path's
# request-plus-24-hours bound even across clock skew between us and Meta.
META_UPLOAD_REQUEST_HOURS = META_UPLOAD_MAX_HOURS - 1


class OverLimitError(ValueError):
    """Settled refusal: over the duration cap, over 100 MB, or not real video."""

    def __init__(self, reason: str = "limit", message: str | None = None) -> None:
        text = message if message is not None else OVER_LIMIT_MESSAGE
        super().__init__(text)
        self.message = text
        self.reason = reason


class ConfirmationFailedError(ValueError):
    """Confirmation did not send; the preview is hidden or failed."""

    def __init__(self, message: str = CONFIRM_FAILED_COPY) -> None:
        super().__init__(message)
        self.message = message


class ConfirmationConflictError(ConfirmationFailedError):
    """A lost confirmation race: another worker/attempt already claimed."""

    def __init__(self, message: str = CONFIRM_CONFLICT_COPY) -> None:
        super().__init__(message)


class ConfirmationDeadlineExceeded(ConfirmationFailedError):
    """The parent job deadline passed before confirmation work could start."""

    def __init__(self) -> None:
        super().__init__("job deadline expired before processing")


@dataclass(frozen=True)
class MediaInfo:
    """Probed facts about staged input: duration, container, video stream."""

    duration_s: float
    containers: frozenset[str]
    has_video_stream: bool
    video_codec: str | None = None
    video_frames: int | None = None


def _parse_frame_rate(raw: object) -> float | None:
    """Return frames-per-second for an ffprobe rate like ``"10/1"``."""
    if not isinstance(raw, str) or "/" not in raw:
        return None
    try:
        numerator, denominator = (float(part) for part in raw.split("/", 1))
    except ValueError:
        return None
    # Both components must be finite and strictly positive: checking only
    # the quotient accepts "-30/-1" as 30 fps.
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0
        or denominator <= 0
    ):
        return None
    fps = numerator / denominator
    return fps if math.isfinite(fps) and fps > 0 else None


def _video_stream_facts(
    streams: object, duration_s: float
) -> tuple[bool, str | None, int | None]:
    """Return ``(has_video, codec, frames)`` for the first video stream.

    Frame count prefers the probed ``nb_frames``; when the container
    omits it, a strictly positive ``avg_frame_rate`` times a finite
    positive duration estimates it. Unknown stays ``None`` and the
    bounds check refuses it: without a count or a trustworthy rate
    estimate, two real frames are unproven.
    """
    video: dict | None = None
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict) and stream.get("codec_type") == "video":
                video = stream
                break
    if video is None:
        return False, None, None
    raw_codec = video.get("codec_name")
    codec = raw_codec.strip().lower() if isinstance(raw_codec, str) else None
    frames: int | None = None
    try:
        frames = int(str(video.get("nb_frames", "")))
    except (TypeError, ValueError):
        frames = None
    if frames is None:
        fps = _parse_frame_rate(video.get("avg_frame_rate"))
        if fps is not None and math.isfinite(duration_s) and duration_s > 0:
            frames = int(duration_s * fps)
    return True, codec, frames


def _probe_media(path: Path) -> MediaInfo:
    """Probe duration, container, and streams with ffprobe (one call)."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OverLimitError("unreadable") from error
    if completed.returncode != 0:
        raise OverLimitError("unreadable")
    try:
        payload = json.loads(completed.stdout or "{}")
    except ValueError as error:
        raise OverLimitError("unreadable") from error
    try:
        duration = float((payload.get("format") or {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    raw_format = str((payload.get("format") or {}).get("format_name", "") or "")
    containers = frozenset(
        part.strip().lower() for part in raw_format.split(",") if part.strip()
    )
    has_video, codec, frames = _video_stream_facts(
        payload.get("streams"), duration
    )
    return MediaInfo(
        duration_s=duration,
        containers=containers,
        has_video_stream=has_video,
        video_codec=codec,
        video_frames=frames,
    )


def check_upload_bounds(
    path: Path, *, size_bytes: int, max_duration_s: float | None = None
) -> MediaInfo:
    """Refuse over-limit or non-video input with the settled message.

    Size first (no probe cost), then over-duration, then container plus
    a real allowlisted-codec video stream, then a finite, positive,
    meaningful duration and frame count. Raises
    :class:`OverLimitError` before any persist or Meta call; the caller
    deletes the staged file. The duration cap comes from config; when
    not passed it defaults to the configured maximum.
    """
    cap = DEFAULT_MAX_VIDEO_S if max_duration_s is None else max_duration_s
    message = over_limit_message(cap)
    if size_bytes > MAX_BYTES:
        raise OverLimitError("size", message)
    info = _probe_media(path)
    if info.duration_s > cap:
        raise OverLimitError("duration", message)
    if not (info.containers & ALLOWED_CONTAINERS) or not info.has_video_stream:
        raise OverLimitError("type", message)
    if info.video_codec not in ALLOWED_VIDEO_CODECS:
        raise OverLimitError("type", message)
    if (
        not math.isfinite(info.duration_s)
        or info.duration_s <= 0
        or info.duration_s < MIN_DURATION_S
    ):
        raise OverLimitError("duration", message)
    if info.video_frames is None or info.video_frames < MIN_VIDEO_FRAMES:
        raise OverLimitError("duration", message)
    return info


def _transcode(src: Path, dst: Path, *, max_duration_s: float) -> None:
    """Trim/cap to 1080p, strip audio, and write the exact preview file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-t",
            f"{max_duration_s:g}",
            "-vf",
            "scale=w=min(iw\\,1920):h=min(ih\\,1080)"
            ":eval=frame:force_original_aspect_ratio=decrease",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or not dst.is_file():
        raise RuntimeError("could not process clip")


def hash_file(path: Path) -> str:
    """Return the hex SHA-256 of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staging_dir(data_root: Path) -> Path:
    """Return the private staging directory, creating it mode 0700."""
    staging = data_root / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)
    try:
        staging.chmod(0o700)
    except OSError:
        pass
    return staging


def stage_upload(
    upload: BinaryIO, *, data_root: Path, prefix: str = "up"
) -> tuple[Path, int]:
    """Stream a bounded upload into private staging.

    Reads at most ``MAX_BYTES + 1`` so an over-limit clip is refused
    without loading it all; the caller unlinks the staged file.
    """
    try:
        upload.seek(0)
    except Exception:
        pass
    staged = _staging_dir(data_root) / f"{prefix}-{secrets.token_hex(8)}.bin"
    size = 0
    try:
        with staged.open("wb") as handle:
            while True:
                chunk = upload.read(CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise OverLimitError("size")
                handle.write(chunk)
    except BaseException:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return staged, size


def discard_staged(path: Path) -> None:
    """Best-effort unlink of one staged file; never raises."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _is_stale_entry(entry: Path, cutoff: datetime) -> bool:
    """True when a non-symlink regular file is past the crash bound."""
    try:
        if entry.is_symlink() or not entry.is_file():
            return False
        mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    try:
        return bool(mtime <= cutoff)
    except Exception:
        return False


def cleanup_stale_staging(
    *, data_root: Path | None = None, now: datetime | None = None
) -> int:
    """Delete crashed-upload leftovers from private staging.

    Unlinks regular files directly under the staging directory that are
    older than :data:`STAGING_MAX_AGE`, and returns the count. Never
    follows symlinks, never recurses, never touches fresh (possibly
    in-flight) files, and never raises for one bad entry — a missing
    staging directory simply cleans zero files.
    """
    root = data_root if data_root is not None else db.data_root()
    staging = root / STAGING_DIRNAME
    try:
        entries = list(staging.iterdir())
    except OSError:
        return 0
    moment = now if now is not None else db.utcnow()
    try:
        cutoff = moment - STAGING_MAX_AGE
    except Exception:
        return 0
    cleaned = 0
    for entry in entries:
        if not _is_stale_entry(entry, cutoff):
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        cleaned += 1
    return cleaned


# Shape of the job-local staged copy ``prepare_preview`` writes before
# validation (``.stage-<hex>.bin``). Only that writer creates these
# names, and it always unlinks them on return — any surviving file is
# a SIGKILL-or-reboot orphan holding personal footage.
STAGED_COPY_PREFIX = ".stage-"
STAGED_COPY_SUFFIX = ".bin"


def _artifact_owns_preview(conn, job_id: str) -> bool:
    """True when an artifact row already owns this job's preview path.

    A crash between artifact registration and upload-row insert leaves
    a registered ``preview.mp4`` with no upload row; the artifact
    queue owns that file, not the orphan cleanup.
    """
    row = conn.execute(
        "SELECT 1 FROM artifacts WHERE job_id = ? AND relative_path = ?",
        (job_id, PROCESSED_FILENAME),
    ).fetchone()
    return row is not None


def _clean_one_orphaned_job(conn, job_dir: Path, cutoff: datetime) -> int:
    """Unlink known orphans in one expired no-upload job directory."""
    if job_dir.is_symlink() or not job_dir.is_dir():
        return 0
    job_id = job_dir.name
    if not job_id or job_id in (".", "..") or "/" in job_id or "\\" in job_id:
        return 0
    job = conn.execute(
        "SELECT state FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if job is None or job["state"] != "expired":
        return 0
    if (
        conn.execute(
            "SELECT 1 FROM uploads WHERE job_id = ?", (job_id,)
        ).fetchone()
        is not None
    ):
        return 0
    try:
        entries = list(job_dir.iterdir())
    except OSError:
        return 0
    cleaned = 0
    for entry in entries:
        name = entry.name
        if name != PROCESSED_FILENAME and not (
            name.startswith(STAGED_COPY_PREFIX)
            and name.endswith(STAGED_COPY_SUFFIX)
        ):
            continue
        if not _is_stale_entry(entry, cutoff):
            continue
        if name == PROCESSED_FILENAME and _artifact_owns_preview(conn, job_id):
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        cleaned += 1
    return cleaned


def cleanup_orphaned_job_files(
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Delete crashed-prepare leftovers from expired no-upload job dirs.

    A SIGKILL or reboot mid-copy/mid-ffmpeg leaves ``.stage-*.bin``
    and/or a partial ``preview.mp4`` in the job directory with no
    upload row, so no recovery queue ever sees them. For each expired
    job with no upload row, this unlinks exactly those known names
    when they are older than :data:`STAGING_MAX_AGE` — never fresh
    (possibly in-flight) files, never symlinks, never anything else.
    A ``preview.mp4`` with an artifact row belongs to the artifact
    queue and is left alone. Returns the count; never raises.
    """
    try:
        root = data_root if data_root is not None else db.data_root()
        jobs = root / "jobs"
        try:
            job_dirs = list(jobs.iterdir())
        except OSError:
            return 0
        moment = now if now is not None else db.utcnow()
        try:
            cutoff = moment - STAGING_MAX_AGE
        except Exception:
            return 0
        conn = db.connect(db_path)
    except Exception:
        return 0
    cleaned = 0
    try:
        for job_dir in job_dirs:
            try:
                cleaned += _clean_one_orphaned_job(conn, job_dir, cutoff)
            except Exception as error:
                logger.warning(
                    "orphan cleanup job=%s failed: %s",
                    job_dir.name,
                    type(error).__name__,
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return cleaned


def run_startup_cleanup(
    *,
    data_root: Path | None = None,
    now: datetime | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Heal every crashed-upload leftover: root staging plus job orphans.

    The worker-startup hook (registered by ``app/main.py`` in the settings module;
    see this module's integration items): a SIGKILL or reboot between
    staging and registration leaves personal footage that no queue
    owns. Runs the age-bounded root-staging cleanup and the expired
    no-upload job-orphan cleanup, and returns the total files removed.
    Never raises, so startup never fails on cleanup. The hourly sweep
    phase and the lazy per-upload call reuse this same hook.
    """
    total = 0
    try:
        total += cleanup_stale_staging(data_root=data_root, now=now)
    except Exception as error:
        logger.warning(
            "startup staging cleanup failed: %s", type(error).__name__
        )
    try:
        total += cleanup_orphaned_job_files(
            data_root=data_root, now=now, db_path=db_path
        )
    except Exception as error:
        logger.warning("startup orphan cleanup failed: %s", type(error).__name__)
    return total


def _copy_stream_to(source: BinaryIO, dst: Path) -> int:
    """Stream ``source`` into ``dst``, bounded; return the byte count."""
    size = 0
    with dst.open("wb") as handle:
        while True:
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                raise OverLimitError("size")
            handle.write(chunk)
    return size


def prepare_preview(
    ctx: JobContext,
    upload: BinaryIO,
    declared_name: str,
    *,
    uploads: UploadStore,
    max_duration_s: float | None = None,
) -> UploadPreview:
    """Validate, transcode, register, and commit one preview for display.

    ``ctx`` is already admitted; everything is written beneath that
    context's job directory. ``declared_name`` is never trusted for
    limits or type. Any failure after admission finishes the job failed
    and leaves no live artifact, and the preview is never exposed from
    an uncommitted row. The duration cap comes from config; when not
    passed it defaults to the configured maximum.
    """
    _ = declared_name  # never trusted; the probed bytes decide
    cap = DEFAULT_MAX_VIDEO_S if max_duration_s is None else max_duration_s
    directory = _job_dir(str(ctx.job_id))
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    staged = directory / f".stage-{secrets.token_hex(8)}.bin"
    target = directory / PROCESSED_FILENAME
    artifact_id: ArtifactId | None = None
    try:
        try:
            upload.seek(0)
        except Exception:
            pass
        size = _copy_stream_to(upload, staged)
        info = check_upload_bounds(staged, size_bytes=size, max_duration_s=cap)
        _transcode(staged, target, max_duration_s=cap)
        if not target.is_file():
            raise RuntimeError("could not process clip")
        processed_size = target.stat().st_size
        if processed_size > MAX_BYTES:
            raise OverLimitError("size")
        hash_file(target)  # proves the bytes hash before registration
        artifact_id = ctx.artifacts.register("upload_preview", target)
        preview = UploadPreview(
            job_id=ctx.job_id,
            local_video=target,
            duration_s=min(info.duration_s, cap),
            size_bytes=processed_size,
            audio_removed=True,
            confirmed_at=None,
        )
        record = uploads.register_preview(
            ctx, preview, artifact_id, db.utcnow() + timedelta(hours=UPLOAD_PREVIEW_TTL_HOURS)
        )
        return record.preview
    except BaseException as failure:
        if artifact_id is not None:
            try:
                ctx.artifacts.mark_deleted(artifact_id)
            except Exception as error:
                logger.warning(
                    "prepare cleanup job=%s artifact=%s failed: %s",
                    ctx.job_id,
                    artifact_id,
                    type(error).__name__,
                )
        for path in (staged, target):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            ctx.finish("failed")
        except Exception as error:
            logger.warning(
                "prepare finish-failed job=%s failed: %s",
                ctx.job_id,
                type(error).__name__,
            )
        logger.info(
            "prepare job=%s failed: %s", ctx.job_id, type(failure).__name__
        )
        raise
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def _expiry_bounds_hold(
    expires_at: datetime | None, request_time_utc: datetime, now: datetime
) -> bool:
    """True when the Meta expiry passes all three confirmation bounds."""
    if expires_at is None or not isinstance(expires_at, datetime):
        return False
    try:
        return bool(
            expires_at.tzinfo is not None
            and expires_at.utcoffset() is not None
            and request_time_utc < expires_at
            and now < expires_at
            and expires_at <= request_time_utc + timedelta(hours=META_UPLOAD_MAX_HOURS)
        )
    except Exception:
        return False


FailureStage = Literal["meta_upload", "meta_persist", "local_delete", "meta_delete", "recovery"]


def confirm_and_send(
    job_id: JobId, *, uploads: UploadStore, client: MetaFilesClient
) -> MetaFileId:
    """Claim once, upload once, attach atomically, and hand the id to the readback module.

    Only the claim winner calls Meta; a replay of a confirmed upload
    returns the stored id without a second upload, and every other
    losing or stale claim raises without invoking ``upload_file``. A
    bad-expiry or persist failure durably queues the known id for
    deletion and attempts immediate deletion; that id is never handed
    to the readback module. When cancellation wins mid-call the id attaches already due
    and this raises instead of returning it.
    """
    claim = claim_for_send(job_id, uploads=uploads)
    return _send_claim(job_id, claim, uploads=uploads, client=client)


def claim_for_send(job_id: JobId, *, uploads: UploadStore) -> ConfirmationClaim:
    """Take the durable confirmation claim before scheduling Meta I/O."""
    try:
        claim = uploads.claim_confirmation(job_id, db.utcnow())
    except ClaimWaitingError as error:
        # The job is honestly still queued: report a retryable waiting
        # state, never a permanent-looking hidden page.
        raise ConfirmationConflictError(CONFIRM_WAITING_COPY) from error
    except (ClaimExpiredError, ClaimTerminalError) as error:
        # Irrecoverable late claim: atomically hide and finalize the
        # upload in the same transaction family, so a refresh can never
        # resurrect the preview the hidden page just replaced.
        try:
            uploads.record_failure(
                job_id,
                f"confirm-late:{job_id}",
                "meta_upload",
                "confirmation arrived after the upload window",
                db.utcnow(),
            )
        except Exception as finalize_error:
            logger.warning(
                "confirm job=%s late-claim finalize failed: %s",
                job_id,
                type(finalize_error).__name__,
            )
        raise ConfirmationFailedError() from error
    except UploadError as error:
        # Unknown, hidden, or cancelled upload: the settled copy says
        # to start over; nothing was ever uploaded.
        raise ConfirmationFailedError() from error
    return claim


def send_claimed_confirmation(
    job_id: JobId, claim_id: str, *, uploads: UploadStore, client: MetaFilesClient
) -> MetaFileId:
    """Send only a still-active claim won by the submitting request."""
    loaded = uploads.load_job(job_id)
    if loaded is None:
        raise ConfirmationFailedError()
    _, record = loaded
    if (
        record.claim_id != claim_id or record.state != "confirming"
        or record.hidden or record.cancel_requested_at is not None
    ):
        raise ConfirmationFailedError()
    return _send_claim(
        job_id, ConfirmationClaim(record=record, acquired=True),
        uploads=uploads, client=client,
    )


def ensure_active_confirmation_job(job_id: JobId) -> None:
    """Refuse Meta and pipeline work after the parent stops or expires."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT j.state, j.deadline_at, u.cancel_requested_at"
            " FROM jobs j JOIN uploads u ON u.job_id = j.id WHERE j.id = ?",
            (str(job_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ConfirmationFailedError()
    if row["deadline_at"] <= db.format_dt(db.utcnow()) or row["state"] == "expired":
        raise ConfirmationDeadlineExceeded()
    if row["state"] != "active" or row["cancel_requested_at"] is not None:
        raise ConfirmationFailedError()


def _send_claim(
    job_id: JobId, claim: ConfirmationClaim, *, uploads: UploadStore,
    client: MetaFilesClient,
) -> MetaFileId:
    if not claim.acquired:
        record = claim.record
        if record.state == "confirmed" and record.meta_file_id is not None:
            return record.meta_file_id
        raise ConfirmationConflictError()
    record = claim.record
    claim_id = record.claim_id
    guard = record.guard_deadline_utc
    if not claim_id or guard is None:
        raise ConfirmationFailedError()
    pre_call_deadline = guard - timedelta(hours=META_UPLOAD_MAX_HOURS)
    request_time_utc = db.utcnow()
    if request_time_utc > pre_call_deadline:
        # Stale claim: never upload. The committed key stays for
        # lookup-backed bounded recovery instead of a second upload.
        try:
            uploads.record_failure(
                job_id,
                f"confirm-stale:{claim_id}",
                "meta_upload",
                "confirmation passed its pre-call deadline",
                request_time_utc,
            )
        except Exception as error:
            logger.warning(
                "confirm stale job=%s claim=%s failure write failed: %s",
                job_id,
                claim_id,
                type(error).__name__,
            )
        logger.info("confirm job=%s refused: stale claim", job_id)
        raise ConfirmationFailedError(CONFIRM_STALE_COPY)

    def _fail_known(
        failure_key: str,
        stage: FailureStage,
        error: str,
        file_id: MetaFileId | None,
        expires_at: datetime | None,
        *,
        now: datetime,
    ) -> None:
        if file_id is not None:
            try:
                uploads.record_failure(
                    job_id,
                    failure_key,
                    stage,
                    error,
                    now,
                    meta_file_id=file_id,
                    meta_expires_at=expires_at,
                )
            except Exception as write_error:
                logger.warning(
                    "confirm job=%s key=%s failure write failed: %s",
                    job_id,
                    failure_key,
                    type(write_error).__name__,
                )
            try:
                client.delete_file(file_id)
            except Exception as delete_error:
                logger.warning(
                    "confirm job=%s compensating delete of %s failed: %s",
                    job_id,
                    file_id,
                    type(delete_error).__name__,
                )
        else:
            try:
                uploads.record_failure(job_id, failure_key, stage, error, now)
            except Exception as write_error:
                logger.warning(
                    "confirm job=%s key=%s failure write failed: %s",
                    job_id,
                    failure_key,
                    type(write_error).__name__,
                )

    preview_path = record.preview.local_video
    ensure_active_confirmation_job(job_id)
    try:
        uploaded = client.upload_file(
            preview_path,
            expires_after_hours=META_UPLOAD_REQUEST_HOURS,
            idempotency_key=claim_id,
        )
    except MetaUploadExpiryError as error:
        now = db.utcnow()
        file_id, expires_at = expiry_details(error)
        if not isinstance(expires_at, datetime):
            expires_at = None
        _fail_known(
            f"confirm-expiry:{claim_id}:{file_id or 'unknown'}",
            "meta_upload",
            "meta upload returned an invalid expiry",
            file_id,
            expires_at,
            now=now,
        )
        raise ConfirmationFailedError() from error
    except MetaFilesError as error:
        # Unknown id: the committed claim key remains the recovery handle.
        now = db.utcnow()
        try:
            uploads.record_failure(
                job_id,
                f"confirm-upload:{claim_id}",
                "meta_upload",
                "meta upload failed",
                now,
            )
        except Exception as write_error:
            logger.warning(
                "confirm job=%s claim=%s failure write failed: %s",
                job_id,
                claim_id,
                type(write_error).__name__,
            )
        logger.info(
            "confirm job=%s upload failed: %s", job_id, type(error).__name__
        )
        raise ConfirmationFailedError() from error
    now = db.utcnow()
    if not isinstance(uploaded, MetaUpload) or not _expiry_bounds_hold(
        uploaded.expires_at, request_time_utc, now
    ):
        # A faulty adapter cannot bypass the second check: fail the
        # confirmation and queue any parseable id for deletion.
        file_id = uploaded.file_id if isinstance(uploaded, MetaUpload) else None
        expires_at = uploaded.expires_at if isinstance(uploaded, MetaUpload) else None
        if not isinstance(expires_at, datetime):
            expires_at = None
        _fail_known(
            f"confirm-expiry:{claim_id}:{file_id or 'unknown'}",
            "meta_upload",
            "meta upload returned an invalid expiry",
            file_id,
            expires_at,
            now=now,
        )
        raise ConfirmationFailedError()
    try:
        completed = uploads.complete_confirmation(
            claim_id, uploaded.file_id, uploaded.expires_at, request_time_utc, now
        )
    except Exception as error:
        _fail_known(
            f"confirm-persist:{claim_id}:{uploaded.file_id}",
            "meta_persist",
            "confirmation persist failed after meta upload",
            uploaded.file_id,
            uploaded.expires_at,
            now=db.utcnow(),
        )
        raise ConfirmationFailedError() from error
    if completed.state == "cancelling" or completed.meta_file_id is None:
        # Cancellation won mid-call: the id attached already due, so it
        # is never handed to the readback module. Delete promptly; the pending row keeps
        # it retryable if this attempt fails.
        try:
            client.delete_file(uploaded.file_id)
        except Exception as error:
            logger.warning(
                "confirm job=%s cancelling delete of %s failed: %s",
                job_id,
                uploaded.file_id,
                type(error).__name__,
            )
        logger.info("confirm job=%s lost to cancellation", job_id)
        raise ConfirmationFailedError(CONFIRM_CANCELLED_COPY)
    logger.info("confirm job=%s attached %s", job_id, completed.meta_file_id)
    return completed.meta_file_id
