"""Confidence read-back: re-check doubtful labels against a render. The readback module owns this.

`read_back_label` takes the guide module's complete ProvisionalRender plus its LabelSlot,
submits the rendered image internally, and returns a LabelDecision keeping
that exact slot. Reserve ($0.04)
happens before the client invocation; the actual cost settles afterward.
A normal `no` immediately tombstones the provisional artifact and removes
the label (`Not checked`). A client failure settles the hold as failed
and finishes the job failed, as does a tombstone failure after a paid
`no` (so the ledger module's finish still collects the provisional). A cancel
(`BaseException`) settles the full hold as failed, finishes the job
failed and re-raises. A reservation
refusal returns `Not checked` without invoking the client (the 0.70 cutoff:
a read-back that cannot be reserved leaves the label Not checked).

`run_readbacks` is the budgeted wrapper: part threshold <= c < the
maintainer-frozen READBACK_CONFIDENCE_CUTOFF (read from app.contracts as config,
never a local literal), lowest c first, at most three firings per job.
Cap overflow, unknown parts, and reservation failures all yield
`Not checked` / `not_run` without a client call.

Evidence goes through the evidence module's shapes: the wrapper accepts `record_call` /
`record_event` hooks matching the frozen contract (None until the evidence module wires them)
and logs phrase, decision, latency, cost, changed badge, and the running
readback_fired_n. The readback module never writes guide-module files.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol

from app.contracts import (
    READBACK_CONFIDENCE_CUTOFF,
    JSONValue,
    MAX_READBACKS_PER_JOB,
    PART_CHECK_THRESHOLDS,
    READBACK_COST_PER_CALL_USD,
    CallRecord,
    JobContext,
    LabelDecision,
    LabelSlot,
    ProvisionalRender,
    SparkClient,
)
from app.spark import SPARK_MODEL_ID

RESERVE_KIND = "spark_readback"
READBACK_EVENT = "readback"

RecordCall = Callable[[JobContext, CallRecord], None]
RecordEvent = Callable[[JobContext | None, str, Mapping[str, JSONValue]], None]


@dataclass(frozen=True)
class ReadbackResult:
    """One Spark read-back answer plus measured cost/latency (fakeable)."""

    answer: str
    cost_usd: Decimal
    latency_ms: int


class SparkReadbackClient(Protocol):
    """readback-module client surface for read-back (see app.spark docstring)."""

    def read_back(
        self, image_path: Path, phrase: str, position_hint: str
    ) -> ReadbackResult: ...


class ReadbackError(RuntimeError):
    """The client returned an unusable read-back answer (fail closed)."""


def part_threshold(phrase: str) -> float | None:
    """Configured SAM threshold for a part phrase, or None when uncheckable."""
    return PART_CHECK_THRESHOLDS.get(phrase.strip().lower())


def eligible_for_readback(provisional: ProvisionalRender) -> bool:
    """True when part threshold <= c < READBACK_CONFIDENCE_CUTOFF (lowest-c-first input)."""
    threshold = part_threshold(provisional.detection.phrase)
    if threshold is None:
        return False
    score = provisional.detection.score
    return threshold <= score < READBACK_CONFIDENCE_CUTOFF


def _not_checked(
    provisional: ProvisionalRender,
    readback: Literal["no", "not_run"],
    slot: LabelSlot,
) -> LabelDecision:
    if readback == "no":
        # Label removed: the provisional file is tombstoned, so drop it;
        # the detection names the part for the `Not checked` row.
        return LabelDecision(
            slot=slot,
            detection=provisional.detection,
            provisional=None,
            checked=False,
            status="not_checked",
            readback="no",
        )
    return LabelDecision(
        slot=slot,
        detection=provisional.detection,
        provisional=provisional,
        checked=False,
        status="not_checked",
        readback="not_run",
    )


def _failed_actual(result: object) -> Decimal:
    cost = getattr(result, "cost_usd", None)
    if isinstance(cost, Decimal) and cost.is_finite() and cost >= 0:
        return cost
    return Decimal("0")


def _require_valid_result(result: object) -> ReadbackResult:
    if not isinstance(result, ReadbackResult):
        raise ReadbackError(f"client must return ReadbackResult, got {type(result).__name__}")
    if result.answer not in ("yes", "no"):
        raise ReadbackError(f"read-back answer must be 'yes' or 'no', got {result.answer!r}")
    if (
        not isinstance(result.cost_usd, Decimal)
        or not result.cost_usd.is_finite()
        or result.cost_usd < 0
    ):
        raise ReadbackError(f"read-back cost must be finite Decimal >= 0, got {result.cost_usd!r}")
    if (
        not isinstance(result.latency_ms, int)
        or isinstance(result.latency_ms, bool)
        or result.latency_ms < 0
    ):
        raise ReadbackError(f"read-back latency must be int >= 0, got {result.latency_ms!r}")
    return result


def _require_complete_provisional(provisional: ProvisionalRender) -> None:
    if not isinstance(provisional, ProvisionalRender):
        raise ValueError(f"read_back_label needs a ProvisionalRender, got {type(provisional).__name__}")
    if not provisional.detection.phrase.strip():
        raise ValueError("provisional detection carries an empty phrase")
    if not str(provisional.rendered_image_path).strip():
        raise ValueError("provisional render has an empty image path")


def _fire(
    ctx: JobContext,
    provisional: ProvisionalRender,
    judge: Callable[[Path, str, str], object],
    slot: LabelSlot,
) -> tuple[LabelDecision, Decimal | None, int | None]:
    """Reserve -> invoke -> settle once; returns decision plus evidence numbers."""
    try:
        hold = ctx.reserve(RESERVE_KIND, READBACK_COST_PER_CALL_USD)
    except Exception:
        # No hold was issued, so nothing settles; the label stays unchecked.
        return _not_checked(provisional, "not_run", slot), None, None
    try:
        result = judge(
            provisional.rendered_image_path,
            provisional.detection.phrase,
            provisional.position_hint,
        )
    except Exception:
        # A settlement error still terminalizes: finish runs in the finally
        # around the settle, both guarded, and the bare raise re-raises the
        # original untouched (never mutated, never chained).
        try:
            hold.settle(Decimal("0"), "failed")
        except BaseException:
            pass
        finally:
            try:
                ctx.finish("failed")
            except BaseException:
                pass
        raise
    except BaseException:
        # A cancel may land after the request bytes flew: settle the full
        # hold as failed and fail the job, then re-raise the interrupt.
        try:
            hold.settle(READBACK_COST_PER_CALL_USD, "failed")
        except BaseException:
            pass
        finally:
            try:
                ctx.finish("failed")
            except BaseException:
                pass
        raise
    try:
        checked = _require_valid_result(result)
    except ReadbackError:
        try:
            hold.settle(_failed_actual(result), "failed")
        except BaseException:
            pass
        finally:
            try:
                ctx.finish("failed")
            except BaseException:
                pass
        raise
    hold.settle(checked.cost_usd, "succeeded")
    if checked.answer == "no":
        try:
            ctx.artifacts.mark_deleted(provisional.artifact_id)
        except Exception:
            # Paid call settled, but the provisional is still live: fail the
            # job so finish() tombstones it for terminal-job cleanup.
            ctx.finish("failed")
            raise
        return (
            _not_checked(provisional, "no", slot),
            checked.cost_usd,
            checked.latency_ms,
        )
    return (
        LabelDecision(
            slot=slot,
            detection=provisional.detection,
            provisional=provisional,
            checked=True,
            status="part_pixels_unchanged",
            readback="yes",
        ),
        checked.cost_usd,
        checked.latency_ms,
    )


def read_back_label(
    ctx: JobContext,
    provisional: ProvisionalRender,
    *,
    slot: LabelSlot,
    client: SparkClient,
) -> LabelDecision:
    """Fire one read-back for a complete provisional render and its slot."""
    _require_complete_provisional(provisional)
    # Resolve the readback-module method before any reservation: a client without
    # it fails before a cent is held.
    judge = client.read_back  # type: ignore[attr-defined]
    return _fire(ctx, provisional, judge, slot)[0]


def _log_fired(
    ctx: JobContext,
    provisional: ProvisionalRender,
    decision: LabelDecision,
    cost_usd: Decimal,
    latency_ms: int,
    fired_n: int,
    record_call: RecordCall | None,
    record_event: RecordEvent | None,
) -> None:
    if record_call is not None:
        record_call(
            ctx,
            CallRecord(
                job_id=ctx.job_id,
                stage="readback",
                model=SPARK_MODEL_ID,
                endpoint="read_back",
                latency_ms=latency_ms,
                units_in=None,
                units_out=None,
                estimated_cost_usd=cost_usd,
                status=str(decision.readback),
                evidence={
                    "phrase": provisional.detection.phrase,
                    "score": provisional.detection.score,
                    "cutoff": READBACK_CONFIDENCE_CUTOFF,
                    "readback_fired_n": fired_n,
                },
            ),
        )
    if record_event is not None:
        record_event(
            ctx,
            READBACK_EVENT,
            {
                "phrase": provisional.detection.phrase,
                "decision": decision.readback,
                "latency_ms": latency_ms,
                "cost_usd": str(cost_usd),
                "badge_changed": True,  # a fired read-back settles the uncertain badge
                "readback_fired_n": fired_n,
            },
        )


def _log_skipped(
    ctx: JobContext,
    provisional: ProvisionalRender,
    reason: str,
    fired_n: int,
    record_event: RecordEvent | None,
) -> None:
    if record_event is not None:
        record_event(
            ctx,
            READBACK_EVENT,
            {
                "phrase": provisional.detection.phrase,
                "decision": "not_run",
                "reason": reason,
                "latency_ms": None,
                "cost_usd": None,
                "badge_changed": False,
                "readback_fired_n": fired_n,
            },
        )


def run_readbacks(
    ctx: JobContext,
    provisionals: Sequence[ProvisionalRender],
    *,
    slots: Sequence[LabelSlot],
    client: SparkClient,
    record_call: RecordCall | None = None,
    record_event: RecordEvent | None = None,
) -> tuple[LabelDecision, ...]:
    """Decide every label under the read-back cutoff band and the 3-per-job cap.

    `slots` runs parallel to `provisionals`: each decision keeps the slot
    at its own position, so same-phrase parts stay distinct. The lowest-c
    eligible labels fire first (so a mid-run reservation refusal spares
    the most uncertain labels); a label at or above the cutoff is kept
    without read-back; anything else (below its part threshold, unknown
    part, cap overflow, reservation refusal) yields `Not checked`.
    Decisions return in input order. Never mutates the input sequences
    or the frozen provisionals.
    """
    if len(slots) != len(provisionals):
        raise ValueError(
            f"slots ({len(slots)}) must run parallel to provisionals ({len(provisionals)})"
        )
    fire_at = sorted(
        (i for i, p in enumerate(provisionals) if eligible_for_readback(p)),
        key=lambda i: provisionals[i].detection.score,
    )[:MAX_READBACKS_PER_JOB]
    fire_positions = set(fire_at)
    firing_order = fire_at + [i for i in range(len(provisionals)) if i not in fire_positions]
    judge = client.read_back  # type: ignore[attr-defined]
    fired_n = 0
    decisions: dict[int, LabelDecision] = {}
    for index in firing_order:
        provisional = provisionals[index]
        slot = slots[index]
        threshold = part_threshold(provisional.detection.phrase)
        score = provisional.detection.score
        if index in fire_positions:
            _require_complete_provisional(provisional)
            decision, cost_usd, latency_ms = _fire(ctx, provisional, judge, slot)
            if decision.readback in ("yes", "no"):
                fired_n += 1
                assert cost_usd is not None and latency_ms is not None
                _log_fired(
                    ctx, provisional, decision, cost_usd, latency_ms, fired_n,
                    record_call, record_event,
                )
            else:
                _log_skipped(ctx, provisional, "reservation_refused", fired_n, record_event)
            decisions[index] = decision
            continue
        if threshold is not None and score >= READBACK_CONFIDENCE_CUTOFF:
            decisions[index] = LabelDecision(
                slot=slot,
                detection=provisional.detection,
                provisional=provisional,
                checked=True,
                status="part_pixels_unchanged",
                readback="not_run",
            )
            _log_skipped(ctx, provisional, "above_cutoff", fired_n, record_event)
        elif threshold is not None and score >= threshold:
            decisions[index] = _not_checked(provisional, "not_run", slot)
            _log_skipped(ctx, provisional, "cap_reached", fired_n, record_event)
        else:
            decisions[index] = _not_checked(provisional, "not_run", slot)
            _log_skipped(ctx, provisional, "below_threshold", fired_n, record_event)
    return tuple(decisions[i] for i in range(len(provisionals)))
