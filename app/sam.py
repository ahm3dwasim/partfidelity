"""Hosted SAM 3.1 still segmentation with pacing, budget, and honest refusal.

Still images go to hosted ``sam-3.1`` with ``metadata.include_confidence``
``"true"``; request starts are paced to at most 55/minute per process, at
most 8 hosted calls run concurrently per process, and each job is
authorized for at most 32 calls (a 12-step guide's 48 slots share the
same budget: the earliest slots win and the rest stay unchecked).
Only measured parts (fan, battery) with confidence at or above their
configured threshold become :class:`Detection` objects. Anything else --
unsupported phrase, provider error, empty result, below-threshold or
missing confidence, or an ambiguous pick -- yields no Detection, so
downstream renders no label and shows ``Not checked``.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Protocol

import cv2
import numpy as np

from app import contracts
from app.contracts import (
    PART_CHECK_THRESHOLDS,
    SAM_COST_PER_CALL_USD,
    SAM_MAX_CALLS_PER_JOB,
    SAM_MAX_REQUEST_STARTS_PER_MINUTE,
    CallRecord,
    Detection,
    JobContext,
)
from app.parts import normalize_phrase, threshold_for

SAM_MODEL = "sam-3.1"
SAM_ENDPOINT = "segment_stills"
SAM_RESERVE_KIND = "sam"
INCLUDE_CONFIDENCE = "true"
AMBIGUITY_MARGIN = 0.05
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SAM_MAX_IN_FLIGHT = 8
_JOB_COUNT_TTL_S = 2 * contracts.JOB_DEADLINE_S

logger = logging.getLogger("partfidelity.sam")


@dataclass(frozen=True)
class SamMask:
    """One hosted-SAM instance: box, confidence ``c``, and PNG mask bytes."""

    box_xyxy: tuple[int, int, int, int]
    c: float | None
    mask_png: bytes


@dataclass(frozen=True)
class SamStillResult:
    """Adapter-level result for one still; masks may be empty."""

    masks: tuple[SamMask, ...]
    status: str = "ok"


@dataclass(frozen=True)
class SamStageConfig:
    """Immutable SAM config one provider carries for its job.

    Per-part thresholds, the per-call price the ledger reserves and
    settles, and the request pace (None: the shared pacer's own rate).
    A client exposes it as ``sam_stage_config`` and :func:`segment` reads
    it from the client it was handed; nothing is installed process-wide.
    """

    thresholds: Mapping[str, float] = PART_CHECK_THRESHOLDS
    price_per_call: Decimal = SAM_COST_PER_CALL_USD
    max_starts_per_minute: int | None = None

    def __post_init__(self) -> None:
        thresholds = dict(self.thresholds)
        for part, value in thresholds.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"SAM threshold for {part!r} must be 0..1, got {value!r}")
        price = self.price_per_call
        if not isinstance(price, Decimal) or not price.is_finite() or not price > 0:
            raise ValueError(f"SAM price must be a finite positive Decimal, got {price!r}")
        rate = self.max_starts_per_minute
        if rate is not None and (
            isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0
        ):
            raise ValueError(f"SAM pace must be a positive int, got {rate!r}")
        object.__setattr__(
            self,
            "thresholds",
            MappingProxyType({part: float(v) for part, v in thresholds.items()}),
        )

    @property
    def interval_s(self) -> float | None:
        if self.max_starts_per_minute is None:
            return None
        return 60.0 / self.max_starts_per_minute


DEFAULT_SAM_CONFIG = SamStageConfig()


def sam_config_of(client: object) -> SamStageConfig:
    """The config a client carries, or the frozen defaults (replay fakes)."""
    config = getattr(client, "sam_stage_config", None)
    return config if isinstance(config, SamStageConfig) else DEFAULT_SAM_CONFIG


class SamClient(contracts.SamClient, Protocol):
    """SAM-module method surface for the injected hosted-SAM adapter.

    Production adapters send the still to hosted sam-3.1 with
    ``metadata.include_confidence`` ``"true"`` and return the instances with
    their confidence values. Tests inject fakes implementing this method.
    Adapters never read keys themselves: wiring passes ``META_API_KEY`` from
    the environment-only settings.
    """

    def segment_still(
        self, image_path: Path, phrase: str, *, include_confidence: bool = True
    ) -> SamStillResult: ...


class SamCallLimitExceeded(Exception):
    """The 33rd SAM call of a job; raised before reserve or client invocation."""

    def __init__(self, job_id: object, limit: int = SAM_MAX_CALLS_PER_JOB) -> None:
        super().__init__(
            f"SAM call budget exhausted for job {job_id}: limit {limit}"
        )
        self.job_id = job_id
        self.limit = limit


class SamDeadlineExceeded(Exception):
    """The job deadline ran out waiting for an in-flight or pacing slot.

    Raised before reserve or client invocation: nothing was sent or held.
    """


class SamPacer:
    """Spaces hosted-SAM request starts to at most 55/minute per process.

    The first start is immediate; each later start waits for its slot. The
    clock is injectable so tests prove spacing on a fake clock without
    sleeping. Lock-guarded: endpoints run threaded.

    :meth:`paced_call` additionally bounds hosted calls to
    ``SAM_MAX_IN_FLIGHT`` process-wide and holds the pacing lock from the
    slot grant through the enclosed client invocation, so a preempted
    thread cannot start late and bunch with a later slot. Given a
    ``deadline`` on the pacer's clock, every wait is bounded by it and
    :class:`SamDeadlineExceeded` is raised instead of starting late.
    """

    def __init__(
        self,
        *,
        max_starts_per_minute: int = SAM_MAX_REQUEST_STARTS_PER_MINUTE,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / max_starts_per_minute
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start: float | None = None

    def now(self) -> float:
        """Current time on the pacer's clock (also used for call latency)."""
        return self._monotonic()

    def _take_slot_unlocked(
        self, interval_s: float | None = None, deadline: float | None = None
    ) -> float:
        now = self._monotonic()
        if self._next_start is None or now >= self._next_start:
            start = now
        else:
            start = self._next_start
        if deadline is not None and start >= deadline:
            # The slot stays free for a caller that can still use it.
            raise SamDeadlineExceeded("job deadline falls before the next SAM slot")
        if start > now:
            self._sleep(start - now)
            if deadline is not None and self._monotonic() >= deadline:
                # The sleep overshot the deadline: the slot stays free.
                raise SamDeadlineExceeded("job deadline reached during the SAM pacing wait")
        self._next_start = start + (self._interval if interval_s is None else interval_s)
        return start

    def wait(self) -> float:
        """Block until a request may start; return the start time."""
        with self._lock:
            return self._take_slot_unlocked()

    def _time_left(self, deadline: float) -> float:
        return max(0.0, deadline - self._monotonic())

    @contextlib.contextmanager
    def paced_call(
        self, interval_s: float | None = None, deadline: float | None = None
    ) -> Iterator[float]:
        """Grant one pacing slot and hold the start-order lock for the block.

        Invoke the hosted client inside the block, so the real request start
        cannot slip past a later thread's slot. At most
        ``SAM_MAX_IN_FLIGHT`` blocks run concurrently per process.
        ``interval_s`` spaces the next start after this one at the caller's
        own configured pace (None: this pacer's rate); slots stay one shared
        process-wide sequence either way. ``deadline`` (this pacer's clock;
        None: unbounded) bounds the in-flight, lock and pacing waits; past
        it :class:`SamDeadlineExceeded` raises before the block runs.
        """
        if deadline is None:
            with _sam_in_flight:
                with self._lock:
                    yield self._take_slot_unlocked(interval_s)
            return
        if not _sam_in_flight.acquire(timeout=self._time_left(deadline)):
            raise SamDeadlineExceeded("job deadline reached waiting for a SAM in-flight slot")
        try:
            # The acquire may have succeeded at or past the deadline.
            if self._monotonic() >= deadline:
                raise SamDeadlineExceeded("job deadline reached waiting for a SAM in-flight slot")
            if not self._lock.acquire(timeout=self._time_left(deadline)):
                raise SamDeadlineExceeded("job deadline reached waiting for the SAM start order")
            try:
                if self._monotonic() >= deadline:
                    raise SamDeadlineExceeded("job deadline reached waiting for the SAM start order")
                yield self._take_slot_unlocked(interval_s, deadline)
            finally:
                self._lock.release()
        finally:
            _sam_in_flight.release()


_sam_in_flight = threading.BoundedSemaphore(SAM_MAX_IN_FLIGHT)

_default_pacer = SamPacer()


def default_pacer() -> SamPacer:
    """The process-wide pacer used by :func:`segment`."""
    return _default_pacer


def install_pacer(pacer: SamPacer) -> None:
    """Swap the process-wide pacer (tests and service wiring)."""
    global _default_pacer
    _default_pacer = pacer


def reset_pacer() -> None:
    """Restore a fresh real-clock pacer."""
    global _default_pacer
    _default_pacer = SamPacer()


_job_call_counts: dict[str, list[object]] = {}
_counts_lock = threading.Lock()


def _authorize_call(job_id: object) -> int:
    """Authorize one SAM call for a job; return its 1-based request index.

    Raises :class:`SamCallLimitExceeded` past 32 calls. Stale job entries
    (older than twice the job deadline) are evicted on the way in.
    """
    now = time.monotonic()
    key = str(job_id)
    with _counts_lock:
        stale = [
            job
            for job, entry in _job_call_counts.items()
            if now - float(entry[1]) > _JOB_COUNT_TTL_S
        ]
        for job in stale:
            del _job_call_counts[job]
        entry = _job_call_counts.get(key)
        count = int(entry[0]) + 1 if entry is not None else 1
        if count > SAM_MAX_CALLS_PER_JOB:
            raise SamCallLimitExceeded(job_id)
        _job_call_counts[key] = [count, now]
        return count


def reset_sam_job_counts() -> None:
    """Clear per-job call counts (tests and maintenance)."""
    with _counts_lock:
        _job_call_counts.clear()


CallSink = Callable[[JobContext, CallRecord], None]


def _log_sink(ctx: JobContext, record: CallRecord) -> None:
    evidence = record.evidence
    logger.info(
        "sam call job=%s status=%s c=%s latency_ms=%d index=%s cost=%s",
        record.job_id,
        record.status,
        evidence.get("c"),
        record.latency_ms,
        evidence.get("request_index"),
        record.estimated_cost_usd,
    )


_call_sink: ContextVar[CallSink] = ContextVar(
    "sam_call_sink", default=_log_sink
)


def install_call_sink(sink: CallSink) -> None:
    """Install the evidence recorder for this thread only.

    The sink is context-local, so overlapping jobs on threads keep
    their own receipts; the default sink is the service log. Either
    way the record carries returned ``c``, latency, status, and
    request count only -- never images or credentials.
    """
    _call_sink.set(sink)


def reset_call_sink() -> None:
    """Restore the default service-log sink for this thread."""
    _call_sink.set(_log_sink)


def _emit(
    ctx: JobContext,
    *,
    phrase: str,
    c: float | None,
    latency_ms: int,
    status: str,
    request_index: int,
    cost: Decimal,
) -> None:
    _call_sink.get()(
        ctx,
        CallRecord(
            job_id=ctx.job_id,
            stage="sam",
            model=SAM_MODEL,
            endpoint=SAM_ENDPOINT,
            latency_ms=latency_ms,
            units_in=None,
            units_out=None,
            estimated_cost_usd=cost,
            status=status,
            evidence={"phrase": phrase, "c": c, "request_index": request_index},
        ),
    )


@dataclass(frozen=True)
class _ParsedResult:
    masks: tuple[SamMask, ...]  # qualifying only
    best_c: float | None
    status: str  # ok | empty | below_threshold | missing_confidence | provider status


def _validate_mask(mask: SamMask) -> None:
    box = mask.box_xyxy
    if (
        len(box) != 4
        or any(not isinstance(v, int) or isinstance(v, bool) for v in box)
        or box[0] < 0
        or box[1] < 0
        or box[0] >= box[2]
        or box[1] >= box[3]
    ):
        raise ValueError(f"malformed SAM box: {box!r}")
    c = mask.c
    if c is not None and (
        isinstance(c, bool)
        or not isinstance(c, (int, float))
        or not math.isfinite(c)
        or not 0.0 <= c <= 1.0
    ):
        raise ValueError(f"SAM confidence out of range: {c!r}")
    raw = mask.mask_png
    if not isinstance(raw, bytes) or not raw.startswith(PNG_MAGIC):
        raise ValueError("SAM mask is not PNG bytes")
    try:
        decoded = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
    except Exception:
        decoded = None
    if decoded is None:
        raise ValueError("SAM mask is not a decodable PNG")


def _parse_result(result: SamStillResult, threshold: float) -> _ParsedResult:
    if result.status != "ok":
        # Provider-level failure: Not checked, but the actual status is kept.
        return _ParsedResult((), None, result.status)
    if not result.masks:
        return _ParsedResult((), None, "empty")
    for mask in result.masks:
        _validate_mask(mask)
    with_c = [m for m in result.masks if m.c is not None]
    if not with_c:
        return _ParsedResult((), None, "missing_confidence")
    best_c = max(m.c for m in with_c if m.c is not None)
    qualifying = tuple(m for m in with_c if m.c is not None and m.c >= threshold)
    if not qualifying:
        return _ParsedResult((), best_c, "below_threshold")
    return _ParsedResult(qualifying, best_c, "ok")


def _mask_dest(image_path: Path, phrase: str, rank: int) -> Path:
    return image_path.with_name(f"{image_path.stem}_{phrase}_{rank}.mask.png")


@contextlib.contextmanager
def _destination_locks(dirs: Sequence[Path]) -> Iterator[None]:
    """Hold one exclusive lock per directory, taken in sorted path order.

    The lock file is created once and left in place for the life of the
    directory: it is never unlinked, on any path, because unlinking a
    released lock lets a later worker create a fresh file and publish
    concurrently with a worker still holding the old inode.
    """
    held: list[BinaryIO] = []
    try:
        for parent in sorted(set(dirs)):
            handle = open(parent / ".publish.lock", "a+b")
            held.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        for handle in reversed(held):
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                handle.close()


def publish_staged(staged: Sequence[tuple[Path, Path]]) -> None:
    """Rename staged temps onto destinations; prior set intact on failure.

    Existing destinations are copied to backup temps first. If any rename
    fails, every backup is restored, every newly published file is removed,
    leftover backups are deleted, and the error propagates. Backups are
    deleted on success. Shared with frame scoring.

    Snapshot, publish, and rollback run under one exclusive lock per
    destination directory, so overlapping publications serialize instead of
    erasing each other's output.
    """
    with _destination_locks([dest.parent for _tmp, dest in staged]):
        _publish_staged_locked(staged)


def _publish_staged_locked(staged: Sequence[tuple[Path, Path]]) -> None:
    backups: dict[Path, Path | None] = {}
    backup_tmps: list[Path] = []
    in_flight: list[tuple[Path, Path]] = []
    try:
        for tmp, dest in staged:
            backup: Path | None = None
            if os.path.lexists(dest):
                fd, backup_name = tempfile.mkstemp(
                    prefix=dest.name + ".", suffix=".bak", dir=dest.parent
                )
                backup = Path(backup_name)
                backup_tmps.append(backup)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(dest.read_bytes())
                backups[dest] = backup
            else:
                backups[dest] = None
            # Recorded before the rename: on rollback a consumed staged
            # file means the rename ran, a surviving one means it did not.
            in_flight.append((tmp, dest))
            os.replace(tmp, dest)
    except BaseException:
        for tmp, dest in reversed(in_flight):
            with contextlib.suppress(OSError):
                if os.path.lexists(tmp):
                    tmp.unlink(missing_ok=True)
                else:
                    backup = backups[dest]
                    if backup is not None:
                        os.replace(backup, dest)
                    else:
                        dest.unlink(missing_ok=True)
        for backup_tmp in backup_tmps:
            with contextlib.suppress(OSError):
                backup_tmp.unlink(missing_ok=True)
        raise
    for backup_tmp in backup_tmps:
        with contextlib.suppress(OSError):
            backup_tmp.unlink(missing_ok=True)


def _write_masks_staged(
    image_path: Path, phrase: str, masks: Sequence[SamMask]
) -> tuple[Path, ...]:
    """Write every mask via a unique temp file, then publish atomically.

    Each temp is recorded before its fd closes, and a completion-flag
    ``finally`` removes staged temps on any failure path -- including
    ``KeyboardInterrupt``, ``SystemExit``, and an ``os.close`` error.
    Destinations change only through :func:`publish_staged`, so a failed
    publication leaves the prior destination set intact. The caller emits
    the call record before propagating the error.
    """
    staged: list[tuple[Path, Path]] = []
    complete = False
    try:
        for rank, mask in enumerate(masks):
            dest = _mask_dest(image_path, phrase, rank)
            fd, tmp_name = tempfile.mkstemp(
                prefix=dest.name + ".", suffix=".tmp", dir=image_path.parent
            )
            tmp = Path(tmp_name)
            staged.append((tmp, dest))
            os.close(fd)
            tmp.write_bytes(mask.mask_png)
        publish_staged(staged)
        complete = True
    finally:
        if not complete:
            for tmp, _dest in staged:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
    return tuple(dest for _tmp, dest in staged)


def _pacer_deadline(client: object, pacer: SamPacer) -> float | None:
    """The client's job deadline moved onto the pacer's clock (None: unbounded)."""
    remaining_s = getattr(client, "remaining_s", None)
    left = remaining_s() if callable(remaining_s) else None
    return None if left is None else pacer.now() + left


def _record_refusal(client: object, phrase: str, latency_ms: int) -> None:
    record = getattr(client, "record_refusal", None)
    if callable(record):
        record(phrase, latency_ms)


def segment(
    ctx: JobContext, image_path: Path, phrase: str, *, client: SamClient
) -> tuple[Detection, ...]:
    """Locate one checked part in a still; ``()`` renders as ``Not checked``.

    Unsupported phrases return ``()`` with no reservation and no client call.
    Otherwise the call is authorized against the 32-call job bound, then
    ``ctx.reserve`` and the single client invocation run inside one paced
    guard: the pacing slot is taken and used under the same lock, and at
    most 8 hosted calls run concurrently per process. The hold settles
    afterward: full cost on success, full cost with ``failed`` on a
    provider-declared error/timeout, $0 with ``failed`` on exception, and
    the full cost with ``failed`` on a cancel (``BaseException``), which
    re-raises. Price, thresholds and pace come from the client's
    :class:`SamStageConfig` (frozen defaults otherwise). A
    failed ``reserve`` propagates before the client can run; it still
    consumes one of the 32 authorizations, since the job cannot proceed
    without the hold. One evidence record is emitted per attempted call.
    The client's job deadline (``remaining_s``) bounds the in-flight and
    pacing waits: past it nothing is reserved or sent, a $0 ``deadline``
    record is emitted (and the client's own $0 receipt via
    ``record_refusal``), and :class:`SamDeadlineExceeded` raises.
    Mask files are staged to unique temp files and renamed into place
    only after every write succeeds; a local write error still emits its
    call record before propagating.
    """
    config = sam_config_of(client)
    price = config.price_per_call
    norm = normalize_phrase(phrase)
    threshold = threshold_for(norm, thresholds=config.thresholds)
    if threshold is None:
        return ()
    index = _authorize_call(ctx.job_id)
    pacer = default_pacer()
    deadline = _pacer_deadline(client, pacer)
    waited_from = pacer.now()
    with contextlib.ExitStack() as guard:
        try:
            guard.enter_context(pacer.paced_call(config.interval_s, deadline))
        except SamDeadlineExceeded:
            latency_ms = int((pacer.now() - waited_from) * 1000)
            _record_refusal(client, norm, latency_ms)
            _emit(
                ctx,
                phrase=norm,
                c=None,
                latency_ms=latency_ms,
                status="deadline",
                request_index=index,
                cost=Decimal("0"),
            )
            raise
        reservation = ctx.reserve(SAM_RESERVE_KIND, price)
        started = pacer.now()
        try:
            result = client.segment_still(
                image_path, norm, include_confidence=True
            )
            parsed = _parse_result(result, threshold)
        except Exception:
            latency_ms = int((pacer.now() - started) * 1000)
            reservation.settle(Decimal("0"), "failed")
            _emit(
                ctx,
                phrase=norm,
                c=None,
                latency_ms=latency_ms,
                status="error",
                request_index=index,
                cost=Decimal("0"),
            )
            raise
        except BaseException:
            # A cancel may land after the request bytes flew: settle the
            # full price as failed, then re-raise the interrupt untouched.
            try:
                reservation.settle(price, "failed")
            except BaseException:
                pass
            raise
        latency_ms = int((pacer.now() - started) * 1000)
        if result.status == "ok":
            reservation.settle(price, "succeeded")
        else:
            reservation.settle(price, "failed")
    try:
        mask_paths = _write_masks_staged(image_path, norm, parsed.masks)
    except Exception:
        _emit(
            ctx,
            phrase=norm,
            c=parsed.best_c,
            latency_ms=latency_ms,
            status=parsed.status,
            request_index=index,
            cost=price,
        )
        raise
    detections = tuple(
        Detection(
            phrase=norm,
            mask_path=path,
            score=mask.c if mask.c is not None else 0.0,
            box_xyxy=mask.box_xyxy,
        )
        for path, mask in zip(mask_paths, parsed.masks)
    )
    _emit(
        ctx,
        phrase=norm,
        c=parsed.best_c,
        latency_ms=latency_ms,
        status=parsed.status,
        request_index=index,
        cost=price,
    )
    return detections


def choose_detection(
    detections: Sequence[Detection], position_hint: str
) -> Detection | None:
    """Pick the single label candidate; None renders as ``Not checked``.

    No candidates -> None. One candidate -> it. Several -> the top score wins
    outright when it clears the runner-up by more than AMBIGUITY_MARGIN;
    otherwise the position hint picks among the tied leaders by relative box
    position, and a tie the hint cannot separate stays None (ambiguous).
    """
    cands = list(detections)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    ordered = sorted(cands, key=lambda d: d.score, reverse=True)
    top = ordered[0].score
    tied = [d for d in ordered if top - d.score <= AMBIGUITY_MARGIN]
    if len(tied) == 1:
        return tied[0]
    return _break_tie_by_hint(tied, position_hint)


def _break_tie_by_hint(
    tied: Sequence[Detection], position_hint: str
) -> Detection | None:
    tokens = set(position_hint.strip().lower().split())
    up = bool({"upper", "top"} & tokens) and not bool({"lower", "bottom"} & tokens)
    down = (
        bool({"lower", "bottom"} & tokens) and not bool({"upper", "top"} & tokens)
    )
    left = "left" in tokens and "right" not in tokens
    right = "right" in tokens and "left" not in tokens
    centered = bool({"center", "middle", "centre"} & tokens)

    def cx(box: tuple[int, int, int, int]) -> float:
        return (box[0] + box[2]) / 2

    def cy(box: tuple[int, int, int, int]) -> float:
        return (box[1] + box[3]) / 2

    keys: list[Callable[[tuple[int, int, int, int]], float]] = []
    if up:
        keys.append(cy)
    elif down:
        keys.append(lambda box: -cy(box))
    if left:
        keys.append(cx)
    elif right:
        keys.append(lambda box: -cx(box))
    if centered:
        mid_x = sum(cx(d.box_xyxy) for d in tied) / len(tied)
        mid_y = sum(cy(d.box_xyxy) for d in tied) / len(tied)
        keys.append(lambda box: abs(cx(box) - mid_x) + abs(cy(box) - mid_y))
    if not keys:
        return None

    def key(d: Detection) -> tuple[float, ...]:
        return tuple(f(d.box_xyxy) for f in keys)

    ranked = sorted(tied, key=key)
    if key(ranked[0]) == key(ranked[1]):
        return None
    return ranked[0]


__all__ = (
    "AMBIGUITY_MARGIN",
    "CallSink",
    "INCLUDE_CONFIDENCE",
    "SAM_ENDPOINT",
    "SAM_MAX_IN_FLIGHT",
    "SAM_MODEL",
    "SAM_RESERVE_KIND",
    "SamCallLimitExceeded",
    "SamDeadlineExceeded",
    "SamClient",
    "SamMask",
    "SamPacer",
    "SamStageConfig",
    "SamStillResult",
    "choose_detection",
    "default_pacer",
    "install_call_sink",
    "install_pacer",
    "publish_staged",
    "reset_call_sink",
    "reset_pacer",
    "reset_sam_job_counts",
    "sam_config_of",
    "segment",
)
