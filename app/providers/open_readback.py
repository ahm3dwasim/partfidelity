"""Background second check of part labels with Meta's open-weights Spark Open.

Runs on Baseten (OpenAI-style chat completions) after the primary
read-back, on a small pool of daemon workers: the guide finish, confirm
and the user never wait for it, and nothing it does changes a label. Each checked
part ends as one ``readback`` receipt under the ``spark-open`` model:
a succeeded receipt carries the model's own strict yes/no, a failed one
(timeout, error, garbled answer, missing key, cap reached) means "not
available". A call sent with no usage back is charged its whole bounded
reservation (``cost: estimate``); one never sent costs $0 (``cost: skipped``).
Deleting a job cancels its checks first (``cancel``). ``/calls/<job>`` turns those into agrees / disagrees /
not available lines; a disagreement only flags the label for the person.
Every receipt names the label image and the label's edit generation; a
label's outcome is its receipt with the highest generation, never simply the
last one to finish. One spend budget covers all of a job's checks: the first
batch and every recheck after a rename (``budget``, kept in the job's
corrections sidecar so it holds across processes and restarts).

All tunables come from ``config/open_readback.toml`` through a strict
loader. The key is read only from the BASETEN_API_KEY environment variable
and is never logged or put in a repr.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import tomllib
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol
from decimal import Decimal
from pathlib import Path

import cv2

from app.contracts import RECEIPT_MODEL_SPARK_OPEN, JobContext, LabelDecision
from app.providers import _records
from app.providers._transport import (
    HttpTransport,
    ProviderError,
    ProviderMalformed,
    UrllibTransport,
    post_with_retry,
    response_json,
)

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "open_readback.toml"
RECORD_ENDPOINT = "/chat-completions"


class OpenReadbackConfigError(ValueError):
    pass


def _int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _num(v: object) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v > 0)


def _nonneg(v: object) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v >= 0)


def _str(v: object) -> bool:
    return isinstance(v, str) and bool(v.strip())


_SPEC = {
    "enabled": lambda v: isinstance(v, bool),
    "model_id": _str,
    "endpoint": lambda v: _str(v) and v.startswith("https://"),
    "effort": _str,
    "image_max_px": _int,
    "jpeg_quality": lambda v: _int(v) and v <= 100,
    "max_tokens": _int,
    "timeout_s": _num,
    "max_calls_per_job": _int,
    "max_cost_usd_per_job": _num,
    "input_usd_per_mtok": _num,
    "output_usd_per_mtok": _num,
    "label_max_chars": _int,
    "prompt_overhead_tokens": _int,
    "image_token_ceiling": _int,
    "cancel_wait_s": _num,
    "concurrency": _int,
    "temperature": _nonneg,
    "flag_text": _str,
    "flag_confirm_label": _str,
    "flag_aria_label": _str,
    "flag_confirmed_text": _str,
    "flag_undo_label": _str,
    "flag_unavailable_text": _str,
}


@dataclass(frozen=True)
class OpenReadbackConfig:
    enabled: bool
    model_id: str
    endpoint: str
    effort: str
    image_max_px: int
    jpeg_quality: int
    max_tokens: int
    timeout_s: float
    max_calls_per_job: int
    max_cost_usd_per_job: Decimal
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    label_max_chars: int
    prompt_overhead_tokens: int
    image_token_ceiling: int
    cancel_wait_s: float
    concurrency: int
    temperature: float
    flag_text: str
    flag_confirm_label: str
    flag_aria_label: str
    flag_confirmed_text: str
    flag_undo_label: str
    flag_unavailable_text: str


def load_config(path: Path = CONFIG) -> OpenReadbackConfig:
    """Read the config; a missing, misspelled or mistyped key raises."""
    with open(path, "rb") as fh:
        top = tomllib.load(fh)
    table = top.get("open_readback")
    if set(top) != {"open_readback"} or not isinstance(table, dict):
        raise OpenReadbackConfigError("expected one [open_readback] table")
    if set(table) != set(_SPEC):
        raise OpenReadbackConfigError(
            f"keys differ (missing {sorted(set(_SPEC) - set(table))}, "
            f"unknown {sorted(set(table) - set(_SPEC))})"
        )
    for key, ok in _SPEC.items():
        if not ok(table[key]):
            raise OpenReadbackConfigError(f"open_readback.{key}: bad value {table[key]!r}")
    money = {k: Decimal(str(table[k])) for k in (
        "max_cost_usd_per_job", "input_usd_per_mtok", "output_usd_per_mtok")}
    floats = {k: float(table[k]) for k in ("timeout_s", "cancel_wait_s", "temperature")}
    return check_config(OpenReadbackConfig(**{**table, **money, **floats}))


def check_config(cfg: OpenReadbackConfig) -> OpenReadbackConfig:
    """Cross-field rules: one bounded call fits the job cap, and a deletion
    waits at least as long as an in-flight call can run."""
    if reservation_usd(cfg) > cfg.max_cost_usd_per_job:
        raise OpenReadbackConfigError(
            f"open_readback.max_cost_usd_per_job {cfg.max_cost_usd_per_job} cannot cover one"
            f" bounded call ({reservation_usd(cfg)})"
        )
    if cfg.cancel_wait_s < cfg.timeout_s:
        raise OpenReadbackConfigError("open_readback.cancel_wait_s must be at least timeout_s")
    return cfg


def phrase_id(phrase: str) -> str:
    """Short digest of the label text a check was about; an edit changes it."""
    return hashlib.sha256(phrase.encode("utf-8")).hexdigest()[:12]


class Outcome(NamedTuple):
    """A label's second-check result: the generation and text it was about, and
    "disagree", "agree" or "unavailable"."""

    gen: int
    phrase_id: str | None
    state: str


def outcomes(calls: Sequence) -> dict[str, Outcome]:
    """{label image name: Outcome} from the receipt with the highest generation.

    Rechecks can finish in any order, so completion order never decides: an
    older generation landing last cannot replace a newer one. Within one
    generation the later receipt wins. The guide uses an outcome only when its
    generation and phrase_id match the label as it stands.
    """
    best: dict[str, Outcome] = {}
    for c in calls:
        label = c.evidence.get("label")
        if c.model != RECEIPT_MODEL_SPARK_OPEN or not isinstance(label, str):
            continue
        gen = c.evidence.get("gen", 0)
        if label in best and best[label].gen > gen:
            continue
        answer = c.evidence.get("decision") if c.status == "succeeded" else None
        state = {"no": "disagree", "yes": "agree"}.get(answer, "unavailable")
        best[label] = Outcome(gen, c.evidence.get("phrase_id"), state)
    return best


def eligible(decision: LabelDecision) -> bool:
    """A kept label with its own label image: the only kind the second check looks at."""
    return decision.checked and decision.status == "part_pixels_unchanged" and decision.provisional is not None


def _question(phrase: str) -> str:
    return (
        f"Is a '{phrase}' clearly visible in this laptop photo? "
        "Answer with one word: yes or no."
    )


def _input_ceiling(cfg: OpenReadbackConfig, label_bytes: int) -> int:
    """Input tokens one request can use at most: a token is at least one UTF-8
    byte of text, plus the chat template overhead and one bounded image."""
    return (cfg.prompt_overhead_tokens + len(_question("").encode("utf-8")) + label_bytes
            + cfg.image_token_ceiling)


def reservation_usd(cfg: OpenReadbackConfig) -> Decimal:
    """The most one call can cost: the longest allowed label (4 UTF-8 bytes a
    character), the image ceiling and ``max_tokens`` out, at config prices."""
    return _cost(cfg, _input_ceiling(cfg, 4 * cfg.label_max_chars), cfg.max_tokens)


def _jpeg_data_url(path: Path, max_px: int, quality: int) -> str:
    import base64

    image = cv2.imread(str(path))
    if image is None:
        raise ProviderError("cannot read the still")
    scale = max_px / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if max(image.shape[:2]) > max_px:  # the image token ceiling assumes this bound
        raise ProviderError("still larger than the bounded size")
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ProviderError("cannot encode the still")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def parse_answer(text: object) -> str | None:
    """Strict yes/no; anything else fails closed (None)."""
    if not isinstance(text, str):
        return None
    word = text.strip().lower()
    return word if word in ("yes", "no") else None


class SpendBook(Protocol):
    """Where one job's second-check budget lives; shared by every batch of the job."""

    def take_spend(self, job_id: str, reserve: Decimal, max_calls: int, max_cost: Decimal) -> bool: ...
    def settle_spend(self, job_id: str, delta: Decimal, breach: bool) -> None: ...


class _MemoryBook:
    """Process-wide per-job budget, for callers with no corrections sidecar (tests, tools)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # shortcut: one entry per job until it is cancelled; production passes the sidecar book
        self._jobs: dict[str, tuple[int, Decimal, bool]] = {}

    def take_spend(self, job_id: str, reserve: Decimal, max_calls: int, max_cost: Decimal) -> bool:
        with self._lock:
            calls, cost, breached = self._jobs.get(job_id, (0, Decimal("0"), False))
            if breached or calls >= max_calls or cost + reserve > max_cost:
                return False
            self._jobs[job_id] = (calls + 1, cost + reserve, False)
            return True

    def settle_spend(self, job_id: str, delta: Decimal, breach: bool) -> None:
        with self._lock:
            calls, cost, breached = self._jobs.get(job_id, (0, Decimal("0"), False))
            self._jobs[job_id] = (calls, cost + delta, breached or breach)

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


_MEMORY_BOOK = _MemoryBook()


class _Spend:
    """Per-job call and cost caps over a shared book. Each call reserves the bounded
    worst case before it is sent, so measured spend never passes the cap."""

    def __init__(self, cfg: OpenReadbackConfig, job_id: str, book: SpendBook) -> None:
        self._cfg = cfg
        self._job_id = job_id
        self._book = book
        self._reserve = reservation_usd(cfg)

    def take(self) -> bool:
        try:
            return self._book.take_spend(
                self._job_id, self._reserve, self._cfg.max_calls_per_job, self._cfg.max_cost_usd_per_job)
        except Exception as error:  # an unreadable budget never lets a call through
            logger.info("open read-back budget unavailable: %s", type(error).__name__)
            return False

    def settle(self, measured: Decimal | None) -> None:
        """Swap the reserve for the measured cost (an unmeasured call keeps its reserve).
        A bill above the reserve means the token ceiling is wrong: stop this job's calls."""
        if measured is None:
            return
        if measured > self._reserve:
            logger.warning("open read-back billed above its bounded reservation")
        try:
            self._book.settle_spend(self._job_id, measured - self._reserve, measured > self._reserve)
        except Exception as error:  # the reservation stays counted: spend only over-reads
            logger.info("open read-back budget not settled: %s", type(error).__name__)


def _cost(cfg: OpenReadbackConfig, units_in: int, units_out: int) -> Decimal:
    return (units_in * cfg.input_usd_per_mtok + units_out * cfg.output_usd_per_mtok) / 1_000_000


def _ask(
    cfg: OpenReadbackConfig, key: str, transport: HttpTransport, image_url: str, phrase: str
) -> tuple[str | None, int | None, int | None]:
    body = json.dumps(
        {
            "model": cfg.model_id,
            "reasoning_effort": cfg.effort,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _question(phrase)},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
    ).encode("utf-8")
    response = post_with_retry(
        transport,
        cfg.endpoint,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        body=body,
        timeout_s=cfg.timeout_s,
        max_attempts=1,
        deadline_s=cfg.timeout_s,
    )
    payload = response_json(response)
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    units_in = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    units_out = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (units_in, units_out)):
        units_in = units_out = None
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderMalformed("no message content") from None
    return parse_answer(text), units_in, units_out


def check_part(
    ctx: JobContext,
    cfg: OpenReadbackConfig,
    key: str | None,
    transport: HttpTransport,
    spend: _Spend,
    image_path: Path,
    phrase: str,
    index: int = 0,
    allowed: Callable[[], bool] = lambda: True,
    gen: int = 0,
) -> str | None:
    """One second check; returns "yes", "no" or None (not available).

    Records one receipt, except when ``allowed()`` says the job was cancelled
    or deleted before anything was sent: then nothing is read, sent or kept.
    """
    if not allowed():
        return None
    started = time.monotonic()
    answer = units_in = units_out = None
    measured: Decimal | None = None
    sent = False
    reason: str | None = None
    if not key:
        reason = "provider key not configured"  # only that it is absent, never the key
    elif len(phrase) > cfg.label_max_chars:
        reason = "label text too long to check"
    else:
        try:
            url = _jpeg_data_url(image_path, cfg.image_max_px, cfg.jpeg_quality)
        except Exception as error:
            url = None
            reason = "label image could not be read"
            logger.info("open read-back unavailable: %s", type(error).__name__)
        if url is not None:
            if not allowed():
                return None
            sent = spend.take()
            if not sent:
                reason = "second-check budget reached"
        if sent:
            try:
                answer, units_in, units_out = _ask(cfg, key, transport, url, phrase)
                if units_in is not None and units_out is not None:
                    measured = _cost(cfg, units_in, units_out)
            except Exception as error:  # timeout, HTTP error, garbage: all "not available"
                reason = "request failed or timed out"
                logger.info("open read-back unavailable: %s", type(error).__name__)
            spend.settle(measured)
            if reason is None and not answer:
                reason = "answer not understood"
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if measured is not None:
        cost, basis = measured, None
    elif sent:  # sent but the bill is unknown: count the whole bounded reservation
        cost, basis = reservation_usd(cfg), "estimate"
    else:
        cost, basis = Decimal("0"), "skipped"
    _records.persist_receipt(
        ctx,
        _records.readback_record(
            job_id=ctx.job_id,
            model=RECEIPT_MODEL_SPARK_OPEN,
            endpoint=RECORD_ENDPOINT,
            phrase=phrase,
            label=image_path.name,
            phrase_id=phrase_id(phrase),
            gen=gen,
            run_index=index,
            decision=answer,
            latency_ms=elapsed_ms if sent else 0,
            units_in=units_in,
            units_out=units_out,
            cost_usd=cost,
            cost_basis=basis,
            status="succeeded" if answer else "failed",
            reason=None if answer else reason,
        ),
    )
    return answer


def record_unavailable(ctx: JobContext, decision: LabelDecision, gen: int) -> None:
    """Record "not available" for one label generation whose check could not be queued,
    so the guide can say so; nothing was sent and it costs $0."""
    _records.persist_receipt(
        ctx,
        _records.readback_record(
            job_id=ctx.job_id,
            model=RECEIPT_MODEL_SPARK_OPEN,
            endpoint=RECORD_ENDPOINT,
            phrase=decision.provisional.detection.phrase,
            label=decision.provisional.rendered_image_path.name,
            phrase_id=phrase_id(decision.provisional.detection.phrase),
            gen=gen,
            decision=None,
            latency_ms=0,
            units_in=None,
            units_out=None,
            cost_usd=Decimal("0"),
            cost_basis="skipped",
            status="failed",
            reason="check could not be queued",
        ),
    )


class Batch:
    """The checks one ``start`` queued for a job; ``join`` waits for them."""

    def __init__(self, job_id: str, count: int) -> None:
        self.job_id = job_id
        self._left = count
        self._lock = threading.Lock()
        self._done = threading.Event()

    def _one_done(self) -> None:
        with self._lock:
            self._left -= 1
            if self._left <= 0:
                self._done.set()

    def join(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def is_alive(self) -> bool:
        return not self._done.is_set()


class _Runner:
    """A fixed pool of daemon workers with a shared queue.

    Daemon threads never hold up interpreter exit, and ``shutdown`` stops
    the pool explicitly. ``cancel`` never waits: it drops a job's queued
    checks and flags its in-flight ones so they send nothing more.
    ``wait_quiet`` is the separate, deferred wait for those in-flight ones;
    a job's stills are deleted only once it returns True.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._queue: deque[tuple[str, Batch, Callable[[Callable[[], bool]], None]]] = deque()
        self._workers: list[threading.Thread] = []
        self._active: dict[str, int] = {}
        self._wait_s: dict[str, float] = {}
        # shortcut: grows by one id per cancelled job for the process lifetime; prune if that matters
        self._cancelled: set[str] = set()
        self._stopped = False

    def _allowed(self, job_id: str) -> bool:
        with self._cond:
            return not self._stopped and job_id not in self._cancelled

    def submit(self, job_id: str, tasks: Sequence[Callable[[Callable[[], bool]], None]],
               concurrency: int, wait_s: float) -> Batch | None:
        with self._cond:
            if self._stopped or job_id in self._cancelled:
                return None
            batch = Batch(job_id, len(tasks))
            self._queue.extend((job_id, batch, task) for task in tasks)
            self._wait_s[job_id] = max(wait_s, self._wait_s.get(job_id, 0.0))
            self._workers = [w for w in self._workers if w.is_alive()]
            while len(self._workers) < concurrency:
                worker = threading.Thread(target=self._work, name="open-readback", daemon=True)
                self._workers.append(worker)
                worker.start()
            self._cond.notify_all()
            return batch

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                job_id, batch, task = self._queue.popleft()
                self._active[job_id] = self._active.get(job_id, 0) + 1
            try:
                task(lambda: self._allowed(job_id))
            except Exception as error:
                logger.info("open read-back failed: %s", type(error).__name__)
            finally:
                with self._cond:
                    self._active[job_id] -= 1
                    if not self._active[job_id]:
                        del self._active[job_id]
                    self._forget_if_idle(job_id)
                    self._cond.notify_all()
                batch._one_done()

    def _forget_if_idle(self, job_id: str) -> None:
        if not self._active.get(job_id) and not any(e[0] == job_id for e in self._queue):
            self._wait_s.pop(job_id, None)

    def _drop(self, keep: Callable[[str], bool]) -> None:
        dropped = [entry for entry in self._queue if not keep(entry[0])]
        self._queue = deque(entry for entry in self._queue if keep(entry[0]))
        for _, batch, _ in dropped:
            batch._one_done()

    def cancel(self, job_id: str) -> bool:
        """Mark and drop, never wait; True when nothing of the job is running."""
        with self._cond:
            self._cancelled.add(job_id)
            self._drop(lambda queued: queued != job_id)
            self._forget_if_idle(job_id)
            self._cond.notify_all()
            return not self._active.get(job_id)

    def wait_quiet(self, job_id: str, timeout_s: float | None = None) -> bool:
        """Wait for a job's in-flight checks, up to ``timeout_s`` (default: its
        ``cancel_wait_s``); True once none is running."""
        with self._cond:
            wait_s = self._wait_s.get(job_id, 0.0) if timeout_s is None else timeout_s
            deadline = time.monotonic() + wait_s
            while self._active.get(job_id):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("open read-back still in flight; deletion deferred")
                    return False
                self._cond.wait(remaining)
            self._forget_if_idle(job_id)
            return True

    def shutdown(self, timeout_s: float) -> bool:
        with self._cond:
            self._stopped = True
            self._drop(lambda _: False)
            self._wait_s = {job: s for job, s in self._wait_s.items() if self._active.get(job)}
            self._cond.notify_all()
            workers = list(self._workers)
        deadline = time.monotonic() + timeout_s
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        return not any(worker.is_alive() for worker in workers)


_RUNNER = _Runner()


def cancel(job_id: object) -> bool:
    """Stop the second check for a job; never waits.

    Queued checks are dropped, new ones refused, and in-flight ones send
    nothing more. True when nothing of the job is left running; otherwise
    its files must wait for ``wait_quiet``.
    """
    _MEMORY_BOOK.forget(str(job_id))
    return _RUNNER.cancel(str(job_id))


def wait_quiet(job_id: object, timeout_s: float | None = None) -> bool:
    """The deferred wait after ``cancel``, for background work only: up to
    ``timeout_s`` (default the job's ``cancel_wait_s``); True once nothing
    of the job is running."""
    return _RUNNER.wait_quiet(str(job_id), timeout_s)


def shutdown(timeout_s: float = 0.0) -> bool:
    """Stop the workers: queued checks are dropped and nothing new starts.
    Never waits longer than ``timeout_s``; True when every worker has ended."""
    return _RUNNER.shutdown(timeout_s)


def start(
    ctx: JobContext,
    decisions: Sequence[LabelDecision],
    *,
    gens: Mapping[str, int] | None = None,
    budget: SpendBook | None = None,
    cfg: OpenReadbackConfig | None = None,
    transport: HttpTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Batch | None:
    """Queue the background check for the kept labels; returns at once.

    ``gens`` maps a label image name to the label generation being checked
    (absent: 0, the guide as first made). ``budget`` holds the job's one
    spend budget (the corrections store in production); without it a
    process-wide per-job book is shared by every batch of the job.
    ``None`` means nothing was queued (disabled, nothing to check, job
    cancelled, workers shut down). Never raises: the user's path must not
    depend on this; a bad config already failed app startup.
    """
    try:
        cfg = check_config(cfg or load_config())
        if not cfg.enabled:
            return None
        targets = [
            (d.provisional.rendered_image_path, d.provisional.detection.phrase)
            for d in decisions
            if eligible(d)
        ]
        if not targets:
            return None
        key = (os.environ if environ is None else environ).get("BASETEN_API_KEY") or None
        transport = transport or UrllibTransport()
        spend = _Spend(cfg, str(ctx.job_id), _MEMORY_BOOK if budget is None else budget)
        gens = gens or {}

        def task(index: int, image: Path, phrase: str) -> Callable[[Callable[[], bool]], None]:
            return lambda allowed: check_part(
                ctx, cfg, key, transport, spend, image, phrase, index, allowed, gens.get(image.name, 0))

        return _RUNNER.submit(
            str(ctx.job_id), [task(i, image, phrase) for i, (image, phrase) in enumerate(targets)],
            cfg.concurrency, cfg.cancel_wait_s)
    except Exception as error:
        logger.warning("open read-back not started: %s", type(error).__name__)
        return None


__all__ = (
    "Batch",
    "OpenReadbackConfig",
    "OpenReadbackConfigError",
    "cancel",
    "check_config",
    "Outcome",
    "SpendBook",
    "check_part",
    "eligible",
    "load_config",
    "outcomes",
    "parse_answer",
    "phrase_id",
    "record_unavailable",
    "reservation_usd",
    "shutdown",
    "start",
    "wait_quiet",
)
