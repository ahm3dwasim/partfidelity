"""Call-record builders in the exact vocabulary ``app.evidence`` validates.

Stages, models, endpoints and statuses come from the evidence module's closed allowlists
(the client maps its configured model id to the receipt model name via
``app.settings``);
evidence carries only each stage's declared keys. The live providers
persist one record per attempted HTTP call through
``evidence.record_call``; the wrappers keep owning reserve/settle, so
the integration code leaves the wrapper sinks unwired in live mode and
lets these records be the single receipt per call.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Literal

from app import evidence as _evidence
from app.contracts import (
    RECEIPT_MODEL_SAM,
    RECEIPT_MODEL_SPARK,
    RECEIPT_STAGE_EXTRACT,
    RECEIPT_STAGE_READBACK,
    RECEIPT_STAGE_SEGMENT,
    CallRecord,
    JobContext,
    JobId,
    JSONValue,
)

logger = logging.getLogger(__name__)

RECORD_MODEL_SPARK = RECEIPT_MODEL_SPARK
RECORD_MODEL_SAM = RECEIPT_MODEL_SAM
RECORD_ENDPOINT = "/responses"

Status = Literal["succeeded", "failed"]

# Receipt writes retry once: the keyed write is idempotent, so a retry
# after SQLite contention completes the pair instead of duplicating it.
RECEIPT_ATTEMPTS = 2


def extract_record(
    *,
    job_id: JobId,
    model: str = RECORD_MODEL_SPARK,
    run_index: int,
    latency_ms: int,
    units_in: int | None,
    units_out: int | None,
    cost_usd: Decimal,
    status: Status,
) -> CallRecord:
    """One Spark step-run receipt (stage ``extract``)."""
    evidence: dict[str, JSONValue] = {"run_index": run_index}
    return CallRecord(
        job_id=job_id,
        stage=RECEIPT_STAGE_EXTRACT,
        model=model,
        endpoint=RECORD_ENDPOINT,
        latency_ms=latency_ms,
        units_in=units_in,
        units_out=units_out,
        estimated_cost_usd=cost_usd,
        status=status,
        evidence=evidence,
    )


def segment_record(
    *,
    job_id: JobId,
    model: str = RECORD_MODEL_SAM,
    phrase: str,
    score: float | None,
    latency_ms: int,
    cost_usd: Decimal,
    status: Status,
) -> CallRecord:
    """One hosted-SAM receipt (stage ``segment``). Unknown phrases and a
    missing best score are left for the evidence module's sanitizer to drop, never crash."""
    evidence: dict[str, JSONValue] = {"phrase": phrase}
    if score is not None:
        evidence["score"] = score
    return CallRecord(
        job_id=job_id,
        stage=RECEIPT_STAGE_SEGMENT,
        model=model,
        endpoint=RECORD_ENDPOINT,
        latency_ms=latency_ms,
        units_in=None,
        units_out=None,
        estimated_cost_usd=cost_usd,
        status=status,
        evidence=evidence,
    )


def readback_record(
    *,
    job_id: JobId,
    model: str = RECORD_MODEL_SPARK,
    endpoint: str = RECORD_ENDPOINT,
    phrase: str,
    decision: str | None,
    latency_ms: int,
    units_in: int | None,
    units_out: int | None,
    cost_usd: Decimal,
    status: Status,
    run_index: int | None = None,
    cost_basis: str | None = None,
    label: str | None = None,
    phrase_id: str | None = None,
    gen: int | None = None,
    reason: str | None = None,
) -> CallRecord:
    """One Spark read-back receipt (stage ``readback``). Failed calls carry
    no decision; the detection score stays with the wrapper, not the receipt."""
    evidence: dict[str, JSONValue] = {"phrase": phrase}
    if decision is not None:
        evidence["decision"] = decision
    if run_index is not None:
        evidence["run_index"] = run_index
    if cost_basis is not None:
        evidence["cost"] = cost_basis
    if label is not None:
        evidence["label"] = label
    if phrase_id is not None:
        evidence["phrase_id"] = phrase_id
    if gen:
        evidence["gen"] = gen
    if reason is not None:
        evidence["reason"] = reason
    return CallRecord(
        job_id=job_id,
        stage=RECEIPT_STAGE_READBACK,
        model=model,
        endpoint=endpoint,
        latency_ms=latency_ms,
        units_in=units_in,
        units_out=units_out,
        estimated_cost_usd=cost_usd,
        status=status,
        evidence=evidence,
    )


def persist_receipt(ctx: JobContext, record: CallRecord) -> bool:
    """Persist one receipt through the evidence module; retry once, then report the loss.

    Returns True once the receipt lands. A ``BaseException`` (cancel)
    always propagates untouched — persistence never retries an
    interrupt. Any other failure retries once and, if the receipt still
    will not land, logs the loss (stage only: the log line carries no
    receipt content) and returns False so the caller can settle the
    measured cost first instead of zeroing it behind a persistence error.
    """
    for _ in range(RECEIPT_ATTEMPTS):
        try:
            _evidence.record_call(ctx, record)
        except Exception:
            continue  # one retry; BaseException propagates past it
        return True
    logger.error(
        "live %s receipt lost after a retry; cost stays settled, evidence missing",
        record.stage,
    )
    return False
