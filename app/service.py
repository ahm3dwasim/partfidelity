"""Orchestrator service: starts pipeline jobs from HTTP flows.

The upload-confirmation route and the sample route start ``run_job``
through this service instead of touching the pipeline directly. Replay
mode runs the canonical recorded client set with zero network calls;
live mode builds its client set from the provider factory
(:func:`live_pipeline_clients`: ``select_providers`` for Spark steps,
hosted SAM 3.1 and Spark read-back, plus the existing Meta Files
client, every receipt bound to the job's own context) and raises
:class:`PipelineUnavailable` when no client set can run. Tests may
override the live set with :func:`set_client_factory`.
"""

from __future__ import annotations

import contextlib
import fcntl
import filecmp
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from threading import Event
from typing import Any, BinaryIO

import cv2
import numpy as np

from app import db, persistence, proof_links
from app.admission import job_dir
from app.contracts import (
    JOB_DEADLINE_S,
    JobContext,
    JobId,
    LifecycleRepository,
    UploadStore,
)
from app.ledger import (
    ContentionError,
    LifecycleRepositoryImpl,
    extend_job_deadline,
)
from app.deletion import request_meta_deletion
from app.meta_files import MetaFilesError, select_meta_files_client
from app.pipeline import (
    PipelineClients,
    PipelineResult,
    REFUSAL_MARKER_FILENAME,
    default_replay_clients,
    fail_job,
    load_guide,
    plan_extraction_windows,
    refusal_text_for_job,
    _takes_chunk_path,
    run_confirmed_job,
    run_job,
)
from app.providers import select_providers
from app.providers._transport import HttpTransport
from app.settings import Settings
from app.uploads_store import UploadStoreImpl

logger = logging.getLogger("partfidelity.service")

ClientFactory = Callable[[Settings], PipelineClients]

_client_factory: ClientFactory | None = None


class PipelineUnavailable(RuntimeError):
    """No client set can run this job (live mode cannot build one)."""


class JobStartDeadlineExceeded(PipelineUnavailable):
    """A queued job could not start before its admission deadline."""


def safe_unavailable_cause(error: PipelineUnavailable) -> str:
    """Name known configuration causes without logging provider exception text."""
    message = str(error)
    if "META_API_KEY" in message:
        return "META_API_KEY unavailable"
    if "provider mode" in message:
        return "provider mode mismatch"
    if "job context" in message:
        return "job context unavailable"
    return "PipelineUnavailable"


class SampleUnavailable(RuntimeError):
    """The sample guide could not be built on this pass."""


def set_client_factory(factory: ClientFactory | None) -> None:
    """Install a live client-set override (tests only)."""
    global _client_factory
    _client_factory = factory


def clear_client_factory() -> None:
    """Remove the factory installed by :func:`set_client_factory`."""
    global _client_factory
    _client_factory = None


class CombinedLiveSparkClient:
    """One Spark surface over the live steps + read-back pair.

    ``select_providers`` returns separate clients for step runs and
    read-backs; the pipeline invokes both methods on its ``spark``
    client, so this adapter routes each call to its owner. The steps
    client's ``spark_stage_config`` rides as an attribute, which the
    extractor reads from the client it is handed. Holds no key; the
    repr names the pair through their own key-safe reprs.
    """

    def __init__(self, steps: Any, readback: Any) -> None:
        self._steps = steps
        self._readback = readback
        self.spark_stage_config = steps.spark_stage_config
        step_bound = getattr(steps, "bound_job_id", None)
        rb_bound = getattr(readback, "bound_job_id", None)
        if (
            step_bound is not None
            and rb_bound is not None
            and str(step_bound) != str(rb_bound)
        ):
            raise ValueError(
                "live Spark steps and read-back clients are bound to "
                f"different jobs: {step_bound} vs {rb_bound}"
            )

    @property
    def bound_job_id(self) -> Any:
        """The job both halves are bound to (None for unbound doubles)."""
        bound = getattr(self._steps, "bound_job_id", None)
        if bound is None:
            bound = getattr(self._readback, "bound_job_id", None)
        return bound

    def run_steps(
        self,
        video_file_id: str,
        *,
        clip_duration_s: float | None = None,
        scene_cuts: tuple[float, ...] | None = None,
        clocked: bool = False,
    ):
        return self._steps.run_steps(
            video_file_id,
            clip_duration_s=clip_duration_s,
            scene_cuts=scene_cuts,
            clocked=clocked,
        )

    def read_back(self, image_path: Path, phrase: str, position_hint: str):
        return self._readback.read_back(image_path, phrase, position_hint)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(steps={self._steps!r}, "
            f"readback={self._readback!r})"
        )


def live_pipeline_clients(
    settings: Settings,
    ctx: JobContext,
    *,
    transport: HttpTransport | None = None,
    meta_transport: Callable[..., Any] | None = None,
) -> PipelineClients:
    """Build one job's live client set from the provider factory.

    Spark steps, hosted SAM 3.1 and Spark read-back come from
    :func:`select_providers` bound to the job's context, so every
    receipt persists through ``evidence.record_call`` for that job;
    the clip uploader is the existing Meta Files client. The steps and
    read-back clients combine into the pipeline's single ``spark``
    surface. Replay adapters never appear here: live mode never mixes
    per run. Injected transports replace urllib (tests only);
    production always passes none.
    """
    trio = select_providers(settings, ctx, transport=transport)
    return PipelineClients(
        spark=CombinedLiveSparkClient(trio.spark, trio.readback),  # type: ignore[arg-type]
        sam=trio.sam,
        meta=select_meta_files_client(settings, transport=meta_transport),
    )


def resolve_clients(
    settings: Settings,
    ctx: JobContext | None = None,
    *,
    transport: HttpTransport | None = None,
    meta_transport: Callable[..., Any] | None = None,
) -> PipelineClients:
    """Return the client set for one run under these settings.

    Replay mode always returns the canonical recorded set (the context
    is ignored). Live mode asks the installed override factory when
    one is present, else builds the real set bound to the job's
    context; anything that keeps a live set from being built (no
    context, no key, a mode mismatch, an unavailable Meta endpoint)
    raises :class:`PipelineUnavailable`, naming the cause but never a
    secret. Injected transports reach the built-in path only.
    """
    if settings.pipeline_mode == "replay":
        return default_replay_clients()
    if _client_factory is not None:
        return _client_factory(settings)
    if ctx is None:
        raise PipelineUnavailable(
            "live mode needs the job context to bind receipts"
        )
    try:
        return live_pipeline_clients(
            settings, ctx, transport=transport, meta_transport=meta_transport
        )
    except PipelineUnavailable:
        raise
    except (RuntimeError, ValueError, MetaFilesError) as error:
        raise PipelineUnavailable(
            f"live mode cannot build a client set: {error}"
        ) from error


def replay_flag(settings: Settings) -> bool:
    """Return whether these settings select replay mode."""
    return settings.pipeline_mode == "replay"


def write_sample_clip(path: Path, *, seconds: float = 6.0, fps: int = 10) -> Path:
    """Write a deterministic synthetic test clip (tests only; the sample guide ships real footage)."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    if not writer.isOpened():
        raise SampleUnavailable(f"could not write sample clip to {path}")
    count = int(seconds * fps)
    try:
        for index in range(count):
            frame = np.zeros((240, 320, 3), np.uint8)
            x = int(10 + (300 - 60) * index / max(count - 1, 1))
            cv2.rectangle(frame, (x, 60), (x + 60, 180), (240, 240, 240), -1)
            cv2.putText(
                frame, f"frame {index}", (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def _sample_pointer(settings: Settings) -> Path:
    return Path(settings.data_root) / settings.sample_pointer_filename


def sample_job_id(settings: Settings) -> str | None:
    """The shared sample job's id, or None before it is first built."""
    return _read_pointer(_sample_pointer(settings))


def _sample_bundle(settings: Settings) -> Path:
    return Path(__file__).resolve().parent / settings.sample_bundle_dirname


def sample_notes_path(settings: Settings) -> Path:
    return _sample_bundle(settings) / settings.sample_bundle_notes_filename


def sample_clip_path(settings: Settings, job_id: str) -> Path | None:
    """The sample guide's playable clip (the shipped web copy); None for any other job.

    The sample has no upload row, so it never retains a clip the way uploads do.
    """
    if sample_job_id(settings) != job_id:
        return None
    clip = _sample_bundle(settings) / settings.sample_bundle_clip_filename
    return clip if clip.is_file() and not clip.is_symlink() else None


def _read_pointer(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(value, dict):
        return None
    job_id = value.get("job_id")
    return job_id if isinstance(job_id, str) and job_id else None


def _usable_pointer_job(settings: Settings, pointer: Path) -> JobId | None:
    """Return the pointed-to job when its guide still loads, else None."""
    try:
        if json.loads(pointer.read_text(encoding="utf-8")).get("bundle_version") != settings.sample_bundle_version:
            return None
    except (OSError, ValueError, AttributeError):
        return None
    stored = _read_pointer(pointer)
    if stored is not None:
        conn = db.connect(settings.db_path)
        try:
            row = conn.execute("SELECT state FROM jobs WHERE id = ?", (stored,)).fetchone()
        finally:
            conn.close()
        if row is not None and row["state"] == "expired":
            return None
    if stored is not None and (
        load_guide(
            stored, data_root=Path(settings.data_root),
            guide_filename=settings.guide_filename,
        )
        is not None
        and _installed_media_complete(settings, stored)
    ):
        return JobId(stored)
    return None


def _installed_media_complete(settings: Settings, job_id: str) -> bool:
    """False when an older install lacks media the current guide points at (reinstall backfills it)."""
    try:
        raw = json.loads((job_dir(JobId(job_id), root=Path(settings.data_root)) / settings.guide_filename).read_text(encoding="utf-8"))
        paths = [*raw["frame_paths"], *(p for label in raw["labels"] for p in (
            label["detection"]["mask_path"], label["provisional"]["rendered_image_path"]))]
        return all(Path(p).is_file() for p in paths)
    except (OSError, KeyError, TypeError, ValueError):
        return False


@contextlib.contextmanager
def _sample_build_lock(data_root: Path) -> Iterator[None]:
    """Hold the interprocess first-build lock for the sample guide.

    Same shape as the mask-publish locks: the lock file is created once
    and never unlinked, so a later worker can never publish beside a
    worker still holding the old inode.
    """
    handle = open(data_root / ".sample-build.lock", "a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()


def ensure_sample_job(settings: Settings) -> JobId:
    """Return the sample job id, installing the shipped real guide when missing.

    The sample is a real guide made from a real repair video, shipped as a
    bundle in the image (``app/sample_real``). The pointer file records the
    job; a pointer whose guide no longer loads reinstalls. Raises
    :class:`SampleUnavailable` when the bundle cannot be installed.
    Racing first hits serialize on the interprocess lock and recheck the
    pointer inside it, so exactly one install runs.
    """
    bundle = _sample_bundle(settings)
    try:
        raw = json.loads((bundle / settings.guide_filename).read_text(encoding="utf-8"))
        paths = [settings.sample_bundle_clip_filename, *raw["frame_paths"],
                 *(label["provisional"]["rendered_image_path"] for label in raw["labels"])]
        for name in paths:
            path = bundle / name
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"sample media missing: {name}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SampleUnavailable(f"could not read sample media: {error}") from error
    pointer = _sample_pointer(settings)
    usable = _usable_pointer_job(settings, pointer)
    if usable is not None:
        return usable
    data_root = Path(settings.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    with _sample_build_lock(data_root):
        usable = _usable_pointer_job(settings, pointer)
        if usable is not None:
            return usable
        return _build_sample_job(settings, pointer)


def _install_media(bundle: Path, directory: Path) -> None:
    """Copy the bundle's stills over the installed ones where bytes differ (each swap atomic), then prove every file matches."""
    for sub in ("frames", "labels"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
        for source in sorted((bundle / sub).iterdir()):
            target = directory / sub / source.name
            if not (target.is_file() and filecmp.cmp(source, target, shallow=False)):
                staged = target.with_name(f".{target.name}.tmp")
                with source.open("rb") as src, staged.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(staged, target)
                persistence._fsync_dir(target.parent)
            if not filecmp.cmp(source, target, shallow=False):
                raise OSError(f"{sub}/{source.name} does not match the shipped bundle")


def prepare_sample(settings: Settings) -> None:
    """Before traffic: an installed sample is brought up to the shipped (redacted) bundle, or its media is removed.

    Guide, stills and thumbnails are all replaced (a pointer from an older bundle version reinstalls the guide too).
    If that cannot complete, the installed guide and stills are deleted so no route can serve old bytes, and the sample
    stays unavailable (error page) until a later install succeeds. A host with no installed sample does nothing: its first
    build copies the shipped bundle.
    """
    data_root = Path(settings.data_root)
    bundle = _sample_bundle(settings)
    ids = {_read_pointer(_sample_pointer(settings))}
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        ids.add(json.loads((bundle / settings.guide_filename).read_text(encoding="utf-8"))["job_id"])
    installed = [job_dir(JobId(i), root=data_root) for i in ids if isinstance(i, str) and i]
    installed = [d for d in installed if d.is_dir()]
    if not installed:
        return
    try:
        job_id = ensure_sample_job(settings)
        with _sample_build_lock(data_root):
            _install_media(bundle, job_dir(job_id, root=data_root))
    except Exception as error:  # fail closed, whatever went wrong
        logger.error("sample media refresh failed (%s); removing the installed sample", type(error).__name__)
        for directory in installed:
            for name in ("frames", "labels"):
                shutil.rmtree(directory / name, ignore_errors=True)
            with contextlib.suppress(FileNotFoundError):
                (directory / settings.guide_filename).unlink()
            if (directory / "frames").exists() or (directory / settings.guide_filename).exists():
                raise RuntimeError("the installed sample could not be refreshed or removed; refusing to start") from error


def _build_sample_job(settings: Settings, pointer: Path) -> JobId:
    """Install the shipped real guide as the sample job; the caller holds the build lock."""
    bundle = _sample_bundle(settings)
    try:
        raw = json.loads((bundle / settings.guide_filename).read_text(encoding="utf-8"))
        receipts = json.loads((bundle / "receipts.json").read_text(encoding="utf-8"))
        if not isinstance(receipts, list) or {r["stage"] for r in receipts} != {"upload", "extract", "segment", "readback"}:
            raise ValueError("sample receipts are missing required stages")
        job_id = JobId(raw["job_id"])
        directory = job_dir(job_id, root=Path(settings.data_root))
        # The guide stores absolute paths inside its own job folder, as live guides do.
        absolute = lambda rel: str(directory / rel)  # noqa: E731
        raw["frame_paths"] = [absolute(p) for p in raw["frame_paths"]]
        for label in raw["labels"]:
            label["detection"]["mask_path"] = absolute(label["detection"]["mask_path"])
            label["provisional"]["rendered_image_path"] = absolute(label["provisional"]["rendered_image_path"])
        _install_media(bundle, directory)
        persistence.atomic_write_text(directory / settings.guide_filename, json.dumps(raw, sort_keys=True))
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise SampleUnavailable(f"could not install the sample bundle: {error}") from error
    now = db.format_dt(db.utcnow())
    conn = db.connect(settings.db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at,"
            " deadline_at, confirmed_at, completed_at, cost_cap_usd, readback_fired_n)"
            " VALUES (?, 'done', 'sample-anon', 'sample-ip', 'sample', ?, ?, ?, ?, '0', 0)"
            " ON CONFLICT(id) DO UPDATE SET state = 'done'",
            (str(job_id), now, now, now, now),
        )
        for i, receipt in enumerate(receipts):
            conn.execute(
                "INSERT OR IGNORE INTO model_calls (id, job_id, stage, model, endpoint, latency_ms,"
                " units_in, units_out, estimated_cost_usd, status, evidence_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"sample:{job_id}:{i}", str(job_id), receipt["stage"], receipt["model"],
                 receipt["endpoint"], receipt["latency_ms"], receipt["units_in"], receipt["units_out"],
                 receipt["estimated_cost_usd"], receipt["status"], json.dumps(receipt["evidence"]), receipt["created_at"]),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    try:
        persistence.atomic_write_text(pointer, json.dumps({"job_id": str(job_id), "bundle_version": settings.sample_bundle_version}, sort_keys=True))
    except OSError as error:
        logger.warning("sample pointer unwritten: %s", type(error).__name__)
    return job_id


_RECEIPT_TEXT = ("stage", "model", "endpoint", "status", "estimated_cost_usd")


def _check_receipt_values(receipts: list) -> None:
    """Canonical constraints on each row: finite non-negative cost, non-negative latency, known stage and status."""
    from decimal import Decimal, InvalidOperation
    from app.contracts import RECEIPT_STAGES
    from app.evidence import STATUS_VOCABULARY
    for r in receipts:
        try:
            cost = Decimal(r["estimated_cost_usd"])
        except InvalidOperation:
            cost = Decimal(-1)
        if not (cost.is_finite() and cost >= 0) or r["latency_ms"] < 0 \
                or r["stage"] not in RECEIPT_STAGES or r["status"] not in STATUS_VOCABULARY:
            raise RuntimeError("open-weights receipts fixture has an invalid row")


def _check_receipts_shape(receipts: object) -> None:
    ok = isinstance(receipts, list) and bool(receipts) and all(
        isinstance(r, dict)
        and all(isinstance(r.get(k), str) and r[k] for k in _RECEIPT_TEXT)
        and isinstance(r.get("latency_ms"), int) and not isinstance(r["latency_ms"], bool)
        and all(r.get(k) is None or (isinstance(r[k], int) and not isinstance(r[k], bool)) for k in ("units_in", "units_out"))
        and isinstance(r.get("evidence"), dict)
        for r in receipts
    )
    if not ok:
        raise RuntimeError("open-weights receipts fixture has the wrong shape")


def install_open_weights_receipts(settings: Settings) -> None:
    """Install the shipped receipts of the newer job that ran the open-weights second check (read-only, receipts page only).

    Idempotent: the job row and each receipt are inserted once under fixed ids, so a restart changes nothing.
    A job that already holds real (non-fixture) receipts is left alone and loses any fixture rows.
    """
    links = proof_links.load()
    job_id = links["open_weights_job"]
    path = Path(__file__).resolve().parent / settings.sample_bundle_dirname / "open_weights_receipts.json"
    try:
        raw = path.read_bytes()
        receipts = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"open-weights receipts fixture unreadable: {type(error).__name__}") from error
    if hashlib.sha256(raw).hexdigest() != links["open_weights_fixture_sha256"]:
        raise RuntimeError("open-weights receipts fixture does not match its pinned sha256")
    _check_receipts_shape(receipts)
    if len(receipts) != links["open_weights_receipt_count"]:
        raise RuntimeError("open-weights receipts fixture has the wrong row count")
    _check_receipt_values(receipts)
    now = db.format_dt(db.utcnow())
    conn = db.connect(settings.db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        prefix = proof_links.fixture_id_prefix(job_id)
        ids = [row[0] for row in conn.execute("SELECT id FROM model_calls WHERE job_id = ?", (job_id,))]
        if any(not i.startswith(prefix) for i in ids):
            # The job is a real run with its own receipts: drop any fixture rows an earlier boot added beside them.
            conn.execute("DELETE FROM model_calls WHERE job_id = ? AND substr(id, 1, ?) = ?", (job_id, len(prefix), prefix))
            conn.execute("COMMIT")
            return
        conn.execute(
            "INSERT OR IGNORE INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at,"
            " deadline_at, confirmed_at, completed_at, cost_cap_usd, readback_fired_n)"
            " VALUES (?, 'done', 'receipts-anon', 'receipts-ip', 'shipped-receipts', ?, ?, ?, ?, '0', 0)",
            (job_id, now, now, now, now),
        )
        for i, r in enumerate(receipts):
            conn.execute(
                "INSERT OR IGNORE INTO model_calls (id, job_id, stage, model, endpoint, latency_ms,"
                " units_in, units_out, estimated_cost_usd, status, evidence_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"receipts:{job_id}:{i}", job_id, r["stage"], r["model"], r["endpoint"], r["latency_ms"],
                 r["units_in"], r["units_out"], r["estimated_cost_usd"], r["status"], json.dumps(r["evidence"]), now),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _extend_chunked_parent_deadline(
    settings: Settings, ctx: JobContext, duration_s: float
) -> bool:
    """Extend a live parent for every extraction window on the chunk path.

    A 4-chunk extraction alone runs ~5-7 wall minutes; the frozen 240 s
    job deadline would expire the parent mid-run (reserves refuse, the
    next chunk admission reaps it, and the prebuilt clients' call
    budgets lapse). Called before the live clients are built so their
    monotonic deadlines stamp the longer window too. One shot, capped,
    and only for a confirmed, visible upload with an attached Meta
    file (checked in the extension's own transaction).
    """
    try:
        windows = plan_extraction_windows(duration_s, settings)
    except ValueError:
        logger.warning(
            "deadline extension job=%s skipped: bad duration %r",
            ctx.job_id, duration_s,
        )
        return False
    if not _takes_chunk_path(windows, settings):
        return True
    extra_s = len(windows) * settings.chunk_deadline_allowance_s
    with contextlib.closing(db.connect(settings.db_path)) as conn:
        before = conn.execute(
            "SELECT created_at, deadline_at FROM jobs WHERE id = ?",
            (str(ctx.job_id),),
        ).fetchone()
    if before is None:
        return False
    for attempt in range(settings.deadline_extension_max_attempts):
        try:
            extended = extend_job_deadline(
                ctx.job_id, extra_s, path=settings.db_path,
                max_extra_s=settings.deadline_extension_max_s,
                require_confirmed_upload=True,
            )
        except ContentionError as error:
            logger.warning(
                "deadline extension job=%s lost a race (%d/%d): %s",
                ctx.job_id, attempt + 1, settings.deadline_extension_max_attempts,
                type(error).__name__,
            )
            continue
        with contextlib.closing(db.connect(settings.db_path)) as conn:
            after = conn.execute(
                "SELECT j.state, j.deadline_at, u.state AS upload_state,"
                " u.meta_file_id, u.hidden FROM jobs j"
                " LEFT JOIN uploads u ON u.job_id = j.id WHERE j.id = ?",
                (str(ctx.job_id),),
            ).fetchone()
        required = db.parse_dt(
            before["deadline_at"] if extended else before["created_at"]
        ) + timedelta(seconds=extra_s + (0 if extended else JOB_DEADLINE_S))
        if (after is not None and after["state"] == "active"
                and after["upload_state"] == "confirmed"
                and after["meta_file_id"] is not None and not after["hidden"]
                and db.parse_dt(after["deadline_at"]) >= required
                and after["deadline_at"] > db.format_dt(db.utcnow())):
            return True
        break
    logger.warning("deadline extension job=%s was not verified", ctx.job_id)
    return False


def run_upload_job(
    job_id: JobId | str,
    settings: Settings,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
) -> PipelineResult:
    """Run the pipeline for one confirmed upload; live set or replay set.

    Live mode loads the job's context first and binds the provider
    clients' receipts to it; an unknown job fails closed before any
    client is built, exactly as :func:`run_confirmed_job` would.
    A live parent using the chunk path gets its deadline extended before
    the client set is built, including a long single-window clip.
    """
    replay = replay_flag(settings)
    ctx: JobContext | None = None
    while True:
        with contextlib.closing(db.connect(settings.db_path)) as conn:
            queued = conn.execute(
                "SELECT state, deadline_at FROM jobs WHERE id = ?",
                (str(job_id),),
            ).fetchone()
        if queued is None or queued["state"] != "waiting":
            break
        remaining_s = (db.parse_dt(queued["deadline_at"]) - db.utcnow()).total_seconds()
        if remaining_s <= 0:
            raise JobStartDeadlineExceeded("waiting job passed its deadline")
        Event().wait(min(settings.job_status_refresh_s, remaining_s))
    if queued is not None and queued["state"] == "failed":
        loaded = uploads.load_job(JobId(str(job_id)))
        if loaded is not None and loaded[1].cancel_requested_at is not None:
            marker = job_dir(str(job_id), root=settings.data_root) / REFUSAL_MARKER_FILENAME
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                with marker.open("x", encoding="utf-8") as handle:
                    json.dump({"kind": "cancelled"}, handle)
            except FileExistsError:
                pass
            except OSError as error:
                logger.warning("cancellation reason job=%s unwritten: %s", job_id, type(error).__name__)
        return PipelineResult(
            state="failed", job_id=JobId(str(job_id)), guide=None,
            error=refusal_text_for_job(str(job_id), data_root=settings.data_root),
            replay=replay,
        )
    if queued is not None and queued["state"] != "active":
        if queued["state"] == "expired":
            raise JobStartDeadlineExceeded("waiting job expired before processing")
        raise PipelineUnavailable("job is not active")
    if not replay:
        loaded = uploads.load_job(JobId(str(job_id)))
        if loaded is None:
            return PipelineResult(
                state="failed",
                job_id=JobId(str(job_id)),
                guide=None,
                error="unknown job",
                replay=replay,
            )
        ctx, record = loaded
        if not _extend_chunked_parent_deadline(settings, ctx, record.preview.duration_s):
            raise PipelineUnavailable("job deadline extension unavailable")
    return run_confirmed_job(
        job_id,
        clients=resolve_clients(settings, ctx),
        uploads=uploads,
        lifecycle=lifecycle,
        replay=replay,
        settings=settings,
    )


def run_confirmed_upload(
    job_id: JobId | str,
    settings: Settings,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    meta_client: Any,
) -> PipelineResult:
    """Run one confirmed upload (the background entry); never fail silently.

    When no client set can be built, the cause is logged (it names a
    setting, never a secret), the job fails closed, and its Meta copy
    is deleted through :func:`request_meta_deletion`; a deletion that
    does not verify stays queued for the sweeper.
    """
    try:
        return run_upload_job(
            job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    except Exception as error:
        cause = safe_unavailable_cause(error) if isinstance(error, PipelineUnavailable) else type(error).__name__
        logger.warning("job=%s cannot start: %s", job_id, cause)
        return _fail_and_drop_meta(
            JobId(str(job_id)),
            ("job deadline expired before processing" if isinstance(error, JobStartDeadlineExceeded)
             else "pipeline unavailable" if isinstance(error, PipelineUnavailable)
             else "background job failed"),
            settings,
            uploads=uploads, lifecycle=lifecycle, meta_client=meta_client,
        )


def fail_confirmed_upload(
    job_id: JobId, error: str, settings: Settings, *, uploads: UploadStore,
    lifecycle: LifecycleRepository, meta_client: Any,
) -> PipelineResult:
    """Settle a confirmed job that the runner could not accept."""
    return _fail_and_drop_meta(
        job_id, error, settings, uploads=uploads, lifecycle=lifecycle,
        meta_client=meta_client,
    )


def claim_job_runner_lock(settings: Settings, job_id: JobId | str) -> BinaryIO | None:
    """Hold a process lock from confirmation claim through the runner task."""
    lock_path = job_dir(str(job_id), root=settings.data_root) / ".job-run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.fdopen(
        os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+b"
    )
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def _fail_and_drop_meta(
    job_id: JobId,
    error: str,
    settings: Settings,
    *,
    uploads: UploadStore,
    lifecycle: LifecycleRepository,
    meta_client: Any,
) -> PipelineResult:
    """Fail the job, then delete its attached Meta copy."""
    replay = replay_flag(settings)
    with db.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (str(job_id),)
        ).fetchone()
    if row is not None and row["state"] == "done":
        return PipelineResult(
            state="done", job_id=job_id,
            guide=load_guide(
                job_id, data_root=settings.data_root,
                guide_filename=settings.guide_filename,
            ),
            error=None, replay=replay,
        )
    loaded = uploads.load_job(job_id)
    if loaded is None:
        return PipelineResult(
            state="failed", job_id=job_id, guide=None,
            error="unknown job", replay=replay,
        )
    ctx, record = loaded
    result = fail_job(ctx, error, replay=replay, settings=settings)
    if record.meta_file_id is None and record.state == "confirming":
        uploads.record_failure(
            job_id, f"confirm-settle:{job_id}", "recovery", error, db.utcnow()
        )
    if record.meta_file_id is None:
        return result
    try:
        lifecycle.record_meta_file_deletion(
            record.meta_file_id, "retry", error="failed job Meta deletion pending"
        )
    except Exception as queue_error:
        logger.warning("job=%s Meta deletion queue failed: %s", job_id, type(queue_error).__name__)
    if meta_client is None:
        return result
    try:
        outcome = request_meta_deletion(
            record.meta_file_id, client=meta_client, lifecycle=lifecycle
        )
    except Exception as delete_error:
        outcome = f"retry ({type(delete_error).__name__})"
    logger.info("job=%s meta copy deletion: %s", job_id, outcome)
    return result


def fail_interrupted_jobs(settings: Settings, *, meta_client: Any) -> int:
    """Fail every queued, uploading or running job a container stop cut off.

    Hosted boot only (one container, so nothing else can be running
    them): a claimed or confirmed, visible upload whose job is still waiting
    or active had its run in the stopped container's pool. A failed job with
    a still-confirming upload also needs the second half of failure settlement.
    A claimed upload is never retried: its committed key remains for provider lookup, which
    also covers a crash after Meta accepted the bytes but before attachment.
    """
    conn = db.connect(settings.db_path)
    try:
        rows = conn.execute(
            "SELECT j.id FROM jobs j JOIN uploads u ON u.job_id = j.id"
            " WHERE (j.state IN ('waiting', 'active')"
            " OR (j.state = 'failed' AND u.state = 'confirming'))"
            " AND u.state IN ('confirming', 'confirmed')"
            " AND u.hidden = 0 ORDER BY j.id"
        ).fetchall()
    finally:
        conn.close()
    uploads: UploadStore = UploadStoreImpl(settings.db_path)
    lifecycle: LifecycleRepository = LifecycleRepositoryImpl(settings.db_path)
    interrupted = 0
    for row in rows:
        try:
            runner_lock = claim_job_runner_lock(settings, row["id"])
            if runner_lock is None:
                continue
            with runner_lock:
                logger.warning("job=%s interrupted by a restart: failing it", row["id"])
                _fail_and_drop_meta(
                    JobId(row["id"]), "interrupted by a restart", settings,
                    uploads=uploads, lifecycle=lifecycle, meta_client=meta_client,
                )
                interrupted += 1
        except Exception as error:
            logger.warning(
                "job=%s restart recovery failed: %s", row["id"], type(error).__name__
            )
    return interrupted


__all__ = (
    "ClientFactory",
    "CombinedLiveSparkClient",
    "PipelineUnavailable",
    "SampleUnavailable",
    "clear_client_factory",
    "ensure_sample_job",
    "prepare_sample",
    "install_open_weights_receipts",
    "fail_interrupted_jobs",
    "sample_job_id",
    "live_pipeline_clients",
    "replay_flag",
    "resolve_clients",
    "run_confirmed_upload",
    "run_upload_job",
    "set_client_factory",
    "write_sample_clip",
)
