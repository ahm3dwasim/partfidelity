"""the evidence: model-call receipts and the minimum event vocabulary.

The evidence module owns no tables and no connections: the ledger module owns the ``model_calls`` and
``events`` tables and every statement on them. This module owns the
repository functions (the evidence boundary)::

    record_call(ctx, record)
    record_event(ctx | None, event, detail)
    list_calls(ctx)

Production receipts persist only through the passed ``JobContext``'s
``record_call_with_event`` operation (the ledger-module keyed write over
``model_calls`` and ``events``): a context without it is a contract
violation, never a silent memory write. The per-job in-memory store below
is an explicit test fake. If a ``JobContext`` implementation exposes an
``evidence_store`` attribute holding this module's own
``InMemoryEvidenceStore``, it is used; otherwise the module registry is
used. Any other attached object is refused before any write. Reads mirror
the write seam: ``list_calls`` reads through the context's ledger
``list_call_records`` operation when offered, so real contexts never read
the in-memory registry; contexts without it read the attached fake or the
module registry. Records are always keyed by their own job id, so a result
is never transferred between runs.

Callers pass sanitized metadata. This module never reads ``META_API_KEY``.
Evidence and event details are allowlisted per stage and event type: each
declares the exact keys and value shapes it may store (enums, bounded
numbers), and anything else is dropped before storage, so keys, auth
headers, raw media, filenames, IPs, consent text, and personal identifiers
are never stored or rendered.

Summaries report observed run count, median, and worst only — never a
small-sample p95 or percentile claim.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import logging
import math
import re
import statistics
import threading
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Callable, Mapping

from app.contracts import (
    RECEIPT_MODELS,
    RECEIPT_MODEL_SPARK_OPEN,
    RECEIPT_STAGES,
    RECEIPT_STAGE_EXTRACT,
    RECEIPT_STAGE_LOCATE,
    RECEIPT_STAGE_READBACK,
    RECEIPT_STAGE_SEGMENT,
    RECEIPT_STAGE_UPLOAD,
    CallRecord,
    JobContext,
    JSONValue,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event vocabulary: the minimum needed for tiers/outcomes, consent, failures,
# model evidence, queue/refusal/cancel, and read-back counts (stage 1 to 4
# counts). Anything else (gallery, share clicks, public counters,
# analytics) is out of scope and rejected.
# ---------------------------------------------------------------------------

EVENT_VOCABULARY = frozenset(
    {
        "job.queued",
        "job.admitted",
        "job.refused",
        "job.cancelled",
        "job.completed",
        "job.failed",
        "consent.recorded",
        "model.call_recorded",
        "model.call_failed",
        "readback.counted",
    }
)

# ---------------------------------------------------------------------------
# Evidence bounds.
# ---------------------------------------------------------------------------

MAX_EVIDENCE_KEYS = 32
MAX_EVIDENCE_DEPTH = 4
MAX_EVIDENCE_JSON_BYTES = 8192
# Pre-copy budgets, enforced by walking the caller's structure before any
# copy or traversal, so a huge shape is rejected without allocating it.
MAX_EVIDENCE_NODES = 1024
MAX_EVIDENCE_RAW_CHARS = 65536
# Bound for every allowlisted integer (run/request indexes, counts).
MAX_EVIDENCE_INDEX = 1_000_000

# Evidence/detail keys are allowlisted per stage and event type (schemas
# below); there is no forbidden-key list. Unknown keys and misshaped values
# are dropped before storage, never redacted in place.

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]+\b")
_ABS_PATH_RE = re.compile(r"(?<![\w@.-])(?:/[A-Za-z0-9._~%-]+)+/?")
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s\"']*")
_BEARER_RE = re.compile(r"\b[bB]earer\s+[A-Za-z0-9._~+/-]+=*")
_BASIC_RE = re.compile(r"\bBasic\s+[A-Za-z0-9+/]+=*")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PLUS_PHONE_RE = re.compile(r"\+\d[\d .()/-]{6,}\d")
_DASH_PHONE_RE = re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b")
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{32,}\b")
_MEDIA_FILENAME_RE = re.compile(
    r"\b[A-Za-z0-9._~%-]+\.(?:mp4|mov|avi|mkv|webm|m4v|png|jpe?g|gif|webp|bmp"
    r"|tiff?|mp3|wav|m4a|ogg|flac|pdf|txt|md|csv|json|zip|log)\b"
)
_CONSENT_PHRASE_RE = re.compile(
    r"\bI,?\s+(?:[A-Z][\w.'-]*,?\s+){1,4}agree\w*\b[^.!?]*"
)

_VALUE_PATTERNS = (
    _IPV4_RE,
    _IPV6_RE,
    _ABS_PATH_RE,
    _WIN_PATH_RE,
    _BEARER_RE,
    _BASIC_RE,
    _EMAIL_RE,
    _PLUS_PHONE_RE,
    _DASH_PHONE_RE,
    _LONG_TOKEN_RE,
    _MEDIA_FILENAME_RE,
    _CONSENT_PHRASE_RE,
)


def _measure_shape(value: object, depth: int, budget: list[int]) -> None:
    """Walk the caller's structure without copying; reject huge shapes early.

    ``budget`` is ``[nodes, chars]``. Every mapping entry, list item, and
    scalar counts as one node; all string characters count toward the raw
    budget. Values outside the allowlisted shapes are only counted here; the
    schema pass drops them after the bounded copy.
    """
    budget[0] += 1
    if budget[0] > MAX_EVIDENCE_NODES:
        raise ValueError("evidence carries too many values")
    if depth > MAX_EVIDENCE_DEPTH:
        raise ValueError("evidence nesting exceeds bound")
    if isinstance(value, str):
        budget[1] += len(value)
        if budget[1] > MAX_EVIDENCE_RAW_CHARS:
            raise ValueError("evidence text exceeds bound")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("evidence keys must be strings")
            budget[1] += len(key)
            if budget[1] > MAX_EVIDENCE_RAW_CHARS:
                raise ValueError("evidence text exceeds bound")
            _measure_shape(item, depth + 1, budget)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _measure_shape(item, depth + 1, budget)
        return
    return


def sanitize_evidence(
    evidence: Mapping[str, JSONValue],
    schema: Mapping[str, Callable[[object], bool]] | None = None,
) -> dict[str, JSONValue]:
    """Return a bounded copy of an evidence/detail mapping, allowlisted.

    Only keys in ``schema`` whose values pass the key's shape predicate are
    kept; every unknown key and every misshaped value is dropped before
    storage and therefore never rendered. Oversized shapes are rejected with
    ``ValueError`` before anything is copied; a missing schema keeps nothing.
    """
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    if len(evidence) > MAX_EVIDENCE_KEYS:
        raise ValueError("evidence carries too many top-level keys")
    _measure_shape(evidence, 0, [0, 0])
    allow = schema if schema is not None else {}
    cleaned: dict[str, JSONValue] = {}
    for key, value in copy.deepcopy(dict(evidence)).items():
        check = allow.get(key)
        if check is None:
            continue
        if not check(value):
            continue
        cleaned[key] = value
    if len(json.dumps(cleaned, sort_keys=True)) > MAX_EVIDENCE_JSON_BYTES:
        raise ValueError("evidence exceeds size bound")
    return cleaned


# ---------------------------------------------------------------------------
# Stores. Only the evidence-module InMemoryEvidenceStore below may be attached as
# ``ctx.evidence_store``; get_store refuses anything else by exact type, so
# no foreign store can half-write a receipt. Production persistence lives in
# The ledger module (see _persist_receipt); this module holds no tables and no connections.
# ---------------------------------------------------------------------------


def _copy_call(record: CallRecord) -> CallRecord:
    """Deep-copy a record so stored and returned evidence never alias."""
    return CallRecord(
        job_id=record.job_id,
        stage=record.stage,
        model=record.model,
        endpoint=record.endpoint,
        latency_ms=record.latency_ms,
        units_in=record.units_in,
        units_out=record.units_out,
        estimated_cost_usd=record.estimated_cost_usd,
        status=record.status,
        evidence=copy.deepcopy(dict(record.evidence)),
    )


def call_key(record: CallRecord) -> str:
    """Return the deterministic idempotency key for one call record.

    The key is the sha256 of the record's canonical stored form: job id,
    stage, model, endpoint, latency, units, cost, status, and evidence. The
    frozen contract carries no separate call-id field, so the full stored
    content is the identity: a byte-identical retry maps to the same key,
    and any differing record is a distinct call.
    """
    canonical = json.dumps(
        {
            "job_id": str(record.job_id),
            "stage": record.stage,
            "model": record.model,
            "endpoint": record.endpoint,
            "latency_ms": record.latency_ms,
            "units_in": record.units_in,
            "units_out": record.units_out,
            "estimated_cost_usd": str(record.estimated_cost_usd),
            "status": record.status,
            "evidence": record.evidence,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InMemoryEvidenceStore:
    """Fake repository: per-job call and event lists. Tests inject this.

    Records are deep-copied on the way in and on the way out, so neither a
    caller holding the stored object nor a reader mutating a returned one can
    change what the store holds.

    Receipts persist only through :meth:`record_call_with_event`, which
    refuses any key other than the stored record's own idempotency key
    before mutating anything, then upserts by key under one lock around
    the check and the insert: a failure leaves neither half behind (all
    or none), a repeated key with its event written is a no-op, and one
    without it completes the event. ``save_call`` and ``save_event`` are
    plain single-row helpers (vocabulary events); they are not a receipt
    path.

    Every writable entry point enforces the same call and event allowlists
    as :func:`record_call` and :func:`record_event`: field vocabularies
    raise on anything outside them, and evidence/detail keeps only its
    declared keys and shapes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, dict[str, CallRecord]] = {}
        self._events: dict[str, list[tuple[str, dict[str, JSONValue]]]] = {}
        self._keyed_events: dict[str, set[str]] = {}

    def save_call(self, record: CallRecord) -> None:
        stored = _validated_call_record(record)
        key = call_key(stored)
        copied = _copy_call(stored)
        with self._lock:
            self._calls.setdefault(str(stored.job_id), {})[key] = copied

    def calls_for(self, job_id: str) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(
                _copy_call(r) for r in self._calls.get(str(job_id), {}).values()
            )

    def save_event(
        self, job_id: str, event: str, detail: Mapping[str, JSONValue]
    ) -> None:
        name, cleaned = _validated_event(event, detail)
        with self._lock:
            self._events.setdefault(str(job_id), []).append(
                (name, copy.deepcopy(cleaned))
            )

    def events_for(self, job_id: str) -> tuple[tuple[str, dict[str, JSONValue]], ...]:
        with self._lock:
            return tuple(
                (event, copy.deepcopy(detail))
                for event, detail in self._events.get(str(job_id), ())
            )

    def record_call_with_event(
        self,
        key: str,
        record: CallRecord,
        event: tuple[str, Mapping[str, JSONValue]],
    ) -> None:
        name, detail = event
        stored = _validated_call_record(record)
        valid_name, cleaned = _validated_event(name, detail)
        if key != call_key(stored):
            raise ValueError("receipt key does not match the stored record")
        # Copies first: anything that raises does so before any mutation,
        # so a failure persists neither the call nor the event.
        copied_record = _copy_call(stored)
        copied_detail = copy.deepcopy(cleaned)
        with self._lock:
            job = str(stored.job_id)
            self._calls.setdefault(job, {})[key] = copied_record
            written = self._keyed_events.setdefault(job, set())
            if key not in written:
                self._events.setdefault(job, []).append((valid_name, copied_detail))
                written.add(key)


_registry: dict[str, InMemoryEvidenceStore] = {}
_registry_lock = threading.Lock()


def get_store(ctx: JobContext | None) -> InMemoryEvidenceStore:
    """Return the evidence store for a context (an attached evidence store or fake).

    An attached store must be the evidence-module InMemoryEvidenceStore,
    checked by exact type: anything else — even a class with the same
    methods — is refused here, before any write, so no foreign store can
    orphan a call.
    """
    if ctx is not None:
        attached = getattr(ctx, "evidence_store", None)
        if attached is not None:
            if type(attached) is not InMemoryEvidenceStore:
                raise ValueError(
                    "evidence store must be InMemoryEvidenceStore,"
                    f" not {type(attached).__name__}"
                )
            return attached
        key = str(ctx.job_id)
    else:
        key = ""
    with _registry_lock:
        store = _registry.get(key)
        if store is None:
            store = InMemoryEvidenceStore()
            _registry[key] = store
        return store


def reset_stores() -> None:
    """Clear the fake registry. Tests only; production stores are attached."""
    with _registry_lock:
        _registry.clear()


# Strict field schemas: every stored field is allowlisted, not blacklisted.
# Stages, statuses, and models are closed vocabularies; endpoints are a
# tight shape that leaves no room for IPs, headers, or sentences, and any
# value matching a private-data pattern is rejected outright.
# Derived from the contracts-owned receipt vocabulary (single home):
# every emitter imports the same constants, so the gate and the receipts
# cannot drift apart (one shared vocabulary).
STAGE_VOCABULARY = RECEIPT_STAGES
STATUS_VOCABULARY = frozenset({"succeeded", "failed"})
# The only model ids receipts may carry: Spark for extract and read-back
# (live clients record `spark-1.3`; `muse-spark-1.3` stays for replay and
# older rows), SAM 3.1 for locate and segment, and the Meta Files API for
# clip upload.
MODEL_ALLOWLIST = RECEIPT_MODELS
_ENDPOINT_RE = re.compile(r"/[a-z0-9_-]{1,32}(?:/[a-z0-9_-]{1,32}){0,3}\Z")
# Endpoint shapes legitimately start with "/", which the absolute-path
# pattern matches, so endpoints skip only the path patterns.
_ENDPOINT_SKIPPED_PATTERNS = (_ABS_PATH_RE, _WIN_PATH_RE)


# ---------------------------------------------------------------------------
# Evidence/detail allowlists: each stage and event type declares the exact
# keys and value shapes it may store. Unknown keys and misshaped values are
# dropped before storage, never redacted in place.
# ---------------------------------------------------------------------------

# The checked part list is fan + battery only (PART_CHECK_THRESHOLDS in
# contracts), plus the read-back
# phrasings measured for them.
PART_PHRASE_VOCABULARY = frozenset(
    {
        "fan",
        "battery",
        "cooling fan",
        "internal battery",
        "rectangular battery with label",
    }
)

DECISION_VOCABULARY = frozenset({"yes", "no"})


def _is_phrase(value: object) -> bool:
    return isinstance(value, str) and value in PART_PHRASE_VOCABULARY


def _is_label_name(value: object) -> bool:
    """A rendered label image's file name: the stable identity of one label."""
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,96}", value) is not None


def _is_phrase_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{12}", value) is not None


def _is_decision(value: object) -> bool:
    return isinstance(value, str) and value in DECISION_VOCABULARY


def _is_score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _is_index(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_EVIDENCE_INDEX
    )


def _is_stage_value(value: object) -> bool:
    return isinstance(value, str) and value in STAGE_VOCABULARY


def _is_status_value(value: object) -> bool:
    return isinstance(value, str) and value in STATUS_VOCABULARY


# Why a second check could not run: a closed list, so a receipt can never carry a key or provider text.
UNAVAILABLE_REASONS = frozenset({
    "provider key not configured", "label text too long to check", "label image could not be read",
    "second-check budget reached", "request failed or timed out", "answer not understood",
    "check could not be queued", "check started, receipt not yet recorded",
    "interrupted at shutdown, not retried",
})

# Stage -> {key: shape predicate} for record_call evidence.
CALL_EVIDENCE_SCHEMAS: dict[str, dict[str, Callable[[object], bool]]] = {
    RECEIPT_STAGE_EXTRACT: {"run_index": _is_index},
    RECEIPT_STAGE_LOCATE: {"phrase": _is_phrase, "score": _is_score},
    RECEIPT_STAGE_SEGMENT: {"phrase": _is_phrase, "score": _is_score},
    RECEIPT_STAGE_READBACK: {
        "phrase": _is_phrase,
        "decision": _is_decision,
        "score": _is_score,
        "run_index": _is_index,  # tells apart second checks of the same phrase
        "label": _is_label_name,  # which label image was checked (survives edits and reordering)
        "phrase_id": _is_phrase_id,  # digest of the label text checked, so an edit makes the receipt stale
        "gen": _is_index,  # the label's edit generation checked; the highest one is the label's outcome
        "reason": lambda v: v in UNAVAILABLE_REASONS,  # why a second check shows "not available"
        "cost": lambda v: v in ("estimate", "skipped"),  # second check: unknown bill / never sent
    },
    # The upload receipt is the row itself (stage/model/endpoint/latency/
    # cost/status): a file id or expiry would add no checkable evidence.
    RECEIPT_STAGE_UPLOAD: {},
}

# Event -> {key: shape predicate} for record_event details.
_EVENT_STAGE_ONLY: dict[str, Callable[[object], bool]] = {"stage": _is_stage_value}
EVENT_DETAIL_SCHEMAS: dict[str, dict[str, Callable[[object], bool]]] = {
    "job.queued": _EVENT_STAGE_ONLY,
    "job.admitted": _EVENT_STAGE_ONLY,
    "job.refused": _EVENT_STAGE_ONLY,
    "job.cancelled": _EVENT_STAGE_ONLY,
    "job.completed": _EVENT_STAGE_ONLY,
    "job.failed": _EVENT_STAGE_ONLY,
    "consent.recorded": _EVENT_STAGE_ONLY,
    "model.call_recorded": {"stage": _is_stage_value, "status": _is_status_value},
    "model.call_failed": {"stage": _is_stage_value, "status": _is_status_value},
    "readback.counted": {"stage": _is_stage_value, "count": _is_index},
}


def _check_stage(value: str) -> str:
    if not isinstance(value, str) or value not in STAGE_VOCABULARY:
        raise ValueError(f"call stage {value!r} is outside the known stages")
    return value


def _check_status(value: str) -> str:
    if not isinstance(value, str) or value not in STATUS_VOCABULARY:
        raise ValueError(f"call status {value!r} is outside the known statuses")
    return value


def _check_model(value: str) -> str:
    if not isinstance(value, str) or value not in MODEL_ALLOWLIST:
        raise ValueError(f"call model {value!r} is not a supported model id")
    return value


def _check_endpoint(value: str) -> str:
    if not isinstance(value, str) or not _ENDPOINT_RE.match(value):
        raise ValueError(f"call endpoint {value!r} is not a plain endpoint path")
    for pattern in _VALUE_PATTERNS:
        if pattern in _ENDPOINT_SKIPPED_PATTERNS:
            continue
        if pattern.search(value):
            raise ValueError("call endpoint may carry private data")
    return value


def _check_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"call {name} must be a non-negative int or None")
    return value


def _validated_call_record(record: CallRecord) -> CallRecord:
    """Validate one record against the call allowlists; return its stored form.

    Field vocabularies raise on anything outside them; evidence keeps
    only the record stage's declared keys and shapes. Shared by
    :func:`record_call` and every writable store entry point.
    """
    if not isinstance(record.estimated_cost_usd, Decimal):
        raise ValueError("estimated cost must be a Decimal")
    if not record.estimated_cost_usd.is_finite() or record.estimated_cost_usd < 0:
        raise ValueError("estimated cost must be a finite non-negative Decimal")
    if (
        isinstance(record.latency_ms, bool)
        or not isinstance(record.latency_ms, int)
        or record.latency_ms < 0
    ):
        raise ValueError("latency_ms must be a non-negative int")
    _check_stage(record.stage)
    _check_model(record.model)
    _check_endpoint(record.endpoint)
    _check_status(record.status)
    _check_count("units_in", record.units_in)
    _check_count("units_out", record.units_out)
    cleaned = sanitize_evidence(record.evidence, CALL_EVIDENCE_SCHEMAS[record.stage])
    return CallRecord(
        job_id=record.job_id,
        stage=record.stage,
        model=record.model,
        endpoint=record.endpoint,
        latency_ms=record.latency_ms,
        units_in=record.units_in,
        units_out=record.units_out,
        estimated_cost_usd=record.estimated_cost_usd,
        status=record.status,
        evidence=cleaned,
    )


def _validated_event(
    event: str, detail: Mapping[str, JSONValue]
) -> tuple[str, dict[str, JSONValue]]:
    """Validate one event against the event allowlists; return its stored form.

    Anything outside the minimum vocabulary is rejected; detail keeps
    only the event's declared keys and shapes. Shared by
    :func:`record_event` and every writable store entry point.
    """
    if event not in EVENT_VOCABULARY:
        raise ValueError(f"event {event!r} is outside the minimum vocabulary")
    return event, sanitize_evidence(detail, EVENT_DETAIL_SCHEMAS[event])


def _persist_receipt(
    ctx: JobContext,
    key: str,
    record: CallRecord,
    event: tuple[str, Mapping[str, JSONValue]],
) -> None:
    """Persist one validated receipt through the passed context only.

    This is the evidence module's only production-persistence call site. The evidence module holds no
    tables and no connections: the receipt goes through the context's
    ``record_call_with_event`` operation, and a context without it is a
    contract violation, never a silent memory write.
    """
    persist = getattr(ctx, "record_call_with_event", None)
    if persist is None:
        raise ValueError("context offers no record_call_with_event operation")
    if not callable(persist):
        raise ValueError("context persistence must be a callable method")
    persist(key, record, event)


def record_call(ctx: JobContext, record: CallRecord) -> None:
    """Persist one model-call receipt for the context's own job.

    The record's job id must equal the context's job id: results are never
    transferred between runs. Cost stays an exact ``Decimal``. The receipt
    and its funnel event persist through one atomic, key-idempotent
    ``record_call_with_event`` operation (see :func:`_persist_receipt`),
    so a failure leaves neither half behind and a retry — however
    concurrent — never duplicates the pair. There is no separate
    recovery write: retries simply reissue the same keyed operation.
    """
    if record.job_id != ctx.job_id:
        raise ValueError("call record belongs to a different job")
    stored = _validated_call_record(record)
    event = (
        "model.call_recorded" if record.status == "succeeded" else "model.call_failed"
    )
    detail = {"stage": record.stage, "status": record.status}
    _persist_receipt(ctx, call_key(stored), stored, (event, detail))


def record_event(
    ctx: JobContext | None, event: str, detail: Mapping[str, JSONValue]
) -> None:
    """Persist one vocabulary event. Anything outside the minimum vocabulary
    is rejected so unplanned tracking cannot slip in."""
    name, cleaned = _validated_event(event, detail)
    job_id = "" if ctx is None else str(ctx.job_id)
    get_store(ctx).save_event(job_id, name, cleaned)


def _readable_record(record: CallRecord) -> CallRecord | None:
    """Revalidate one adapter-returned record; return its cleaned copy or None.

    Applies the same structured allowlists as the write path (stage, status,
    model, endpoint, counts, cost, evidence schema). Anything failing them is
    dropped, so pre-existing or adapter-bypassed rows can never render.
    """
    try:
        _check_stage(record.stage)
        _check_status(record.status)
        _check_model(record.model)
        _check_endpoint(record.endpoint)
        latency = record.latency_ms
        if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
            return None
        _check_count("units_in", record.units_in)
        _check_count("units_out", record.units_out)
        cost = record.estimated_cost_usd
        if not isinstance(cost, Decimal) or not cost.is_finite() or cost < 0:
            return None
        cleaned = sanitize_evidence(record.evidence, CALL_EVIDENCE_SCHEMAS[record.stage])
        return replace(_copy_call(record), evidence=cleaned)
    except (ValueError, TypeError, AttributeError):
        return None


def _call_rows(ctx: JobContext, job_id: str) -> tuple[CallRecord, ...]:
    """Return the raw rows backing :func:`list_calls`, before filtering.

    An explicitly attached fake store wins; otherwise a context offering the
    ledger module's ``list_call_records`` read operation is read through it, so real
    contexts never touch the in-memory registry; only contexts with neither
    use the module registry.
    """
    if getattr(ctx, "evidence_store", None) is not None:
        return get_store(ctx).calls_for(job_id)
    read = getattr(ctx, "list_call_records", None)
    if read is None:
        return get_store(ctx).calls_for(job_id)
    if not callable(read):
        raise ValueError("context read operation must be a callable method")
    return tuple(read())


def list_calls(ctx: JobContext) -> tuple[CallRecord, ...]:
    """Return this job's recorded calls in insertion order.

    Rows come from the context's ledger ``list_call_records`` read operation
    when it offers one; otherwise the attached fake or module registry
    backs the read. Records whose job id differs from the context's are
    dropped (a count is logged, never the record), and every remaining
    record is revalidated against the write-path allowlists with its
    evidence re-sanitized, so only clean rows reach callers. Every record
    is deep-copied on the way out, whatever adapter backs the store, so
    mutating a returned record can never change stored state.
    """
    job_id = str(ctx.job_id)
    kept: list[CallRecord] = []
    foreign = 0
    invalid = 0
    for record in _call_rows(ctx, job_id):
        try:
            record_job = str(record.job_id)
        except AttributeError:
            invalid += 1
            continue
        if record_job != job_id:
            foreign += 1
            continue
        cleaned = _readable_record(record)
        if cleaned is None:
            invalid += 1
            continue
        kept.append(cleaned)
    if foreign:
        logger.warning("dropped %d call record(s) for other jobs", foreign)
    if invalid:
        logger.warning("dropped %d invalid call record(s)", invalid)
    return tuple(kept)


def list_events(ctx: JobContext | None) -> tuple[tuple[str, dict[str, JSONValue]], ...]:
    """Return this job's recorded events (``""`` key when ``ctx`` is None).

    Every detail mapping is deep-copied on the way out, whatever adapter
    backs the store, so mutating a returned detail can never change storage.
    """
    job_id = "" if ctx is None else str(ctx.job_id)
    return tuple(
        (event, copy.deepcopy(dict(detail)))
        for event, detail in get_store(ctx).events_for(job_id)
    )


# ---------------------------------------------------------------------------
# Summaries: observed run count, median, worst. No percentiles.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallSummary:
    run_count: int
    median_latency_ms: int | float | None
    worst_latency_ms: int | None
    total_cost_usd: Decimal
    succeeded: int
    failed: int


def summarize_calls(calls: tuple[CallRecord, ...] | list[CallRecord]) -> CallSummary:
    """Summarize observed calls. Median/worst only; no p95 on small samples.

    The median is the exact observed median: an int for odd-sized samples,
    a float midpoint for even-sized ones. It is never truncated down.
    """
    rows = list(calls)
    latencies = [c.latency_ms for c in rows]
    total = sum((c.estimated_cost_usd for c in rows), Decimal("0"))
    return CallSummary(
        run_count=len(rows),
        median_latency_ms=statistics.median(latencies) if latencies else None,
        worst_latency_ms=max(latencies) if latencies else None,
        total_cost_usd=total,
        succeeded=sum(1 for c in rows if c.status == "succeeded"),
        failed=sum(1 for c in rows if c.status != "succeeded"),
    )


# ---------------------------------------------------------------------------
# Sample read-back row: the natural first row of the calls page.
# ---------------------------------------------------------------------------

READBACK_STAGE = RECEIPT_STAGE_READBACK
READBACK_DECISIONS = ("yes", "no")


@dataclass(frozen=True)
class ReadbackRow:
    phrase: str
    decision: str  # "yes" | "no"
    latency_ms: int
    cost_usd: Decimal
    badge: str  # "Read-back answered yes" | "Read-back answered no"


def _badge_for(decision: str) -> str:
    return f"Read-back answered {decision}"


def select_sample_readback(
    calls: tuple[CallRecord, ...] | list[CallRecord],
) -> ReadbackRow | None:
    """Return the first qualifying natural read-back row, or None.

    A row qualifies only when it is this job's own successful read-back call
    carrying a phrase and a yes/no decision. Anything else returns None and
    the page takes the separately labelled fallback link instead.
    """
    for call in calls:
        if call.stage != READBACK_STAGE or call.status != "succeeded":
            continue
        if call.model == RECEIPT_MODEL_SPARK_OPEN:  # the second check is not the sample read-back
            continue
        phrase = call.evidence.get("phrase")
        decision = call.evidence.get("decision")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        if phrase not in PART_PHRASE_VOCABULARY:
            continue
        if decision not in READBACK_DECISIONS:
            continue
        assert isinstance(decision, str)
        return ReadbackRow(
            phrase=phrase,
            decision=decision,
            latency_ms=call.latency_ms,
            cost_usd=call.estimated_cost_usd,
            badge=_badge_for(decision),
        )
    return None


def escape_cell(value: object) -> str:
    """HTML-escape one rendered cell. Rendering never trusts stored text."""
    return html.escape(str(value), quote=True)
