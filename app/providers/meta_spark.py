"""Live Spark step extraction over the Meta Model API.

One ``run_steps`` call is one 1x video pass: the measured fixed
prompt (clock-read wording, no intro merge) plus the strict step
schema the readback module validates, sent with the standard model id — never
contributor mode. Cost settles from the returned usage; a call whose
spend cannot be measured fails instead of under-billing. Each call
persists one ``extract`` receipt in the evidence module's vocabulary; reserve/settle
stay with the ``app.spark`` wrapper, which also owns the three-run
median consensus and the disagreement flag.

Billing outcomes: a determinate refusal (4xx, exhausted 429s or a
pre-send connect failure) raises, and the wrapper settles $0. Anything
past the point of possible charge — a timeout, a post-send failure, a
5xx, or a 200 whose shape is unusable — returns a poisoned result
carrying the measured cost, or the per-run worst case when the spend
cannot be measured: the wrapper's own result validation rejects the
payload and settles that cost as failed, never $0. The failed receipt
carries the same cost, and the reconcile note goes to the log. An
unexpected local ``Exception`` once the request may have flown is
treated the same way: a poisoned result at the per-run worst case,
never a bare raise the wrapper would settle at $0. A cancel
(``KeyboardInterrupt`` or any ``BaseException``) records its receipt
first — the worst case once request bytes may have flown, $0 before
that — then re-raises; the wrapper settles its share as failed. A
receipt that will not persist after a retry is logged as lost and the
measured cost is still returned, so the wrapper settles the actual
spend before the failure surfaces.

Every call runs under the job's remaining deadline (``deadline_at``
on the client's clock): attempts and backoff end by it, and a call
starting after it refuses before send ($0). The client carries its
job's immutable :class:`SparkStageConfig` (run count, spread, run
shares, model id), which the wrapper reads instead of process state.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from app.contracts import (
    MAX_STEPS,
    SPARK_RUNS_PER_GUIDE,
    JSONValue,
    JobContext,
    JobId,
)
from app.providers import _records
from app.providers._transport import (
    HttpTransport,
    ProviderError,
    ProviderConnectError,
    ProviderHttpError,
    ProviderTimeout,
    ProviderMalformed,
    UrllibTransport,
    call_budget_s,
    output_text,
    post_with_retry,
    response_json,
    usage_counts,
)
from app.settings import SPARK_MAX_OUTPUT_TOKENS, SPARK_MODEL_RECORD_NAMES
from app.spark import PHASE_KINDS, SparkFailureKind, SparkRunResult, SparkStageConfig

logger = logging.getLogger(__name__)

# Contiguous-phase prompt (fix 1): the clip is partitioned into
# coarse repair phases tiling [0, end), never sparse micro-action
# moments. The readback module's strict schema carries seconds only, so the m:ss clock
# is a read-then-write instruction rather than an emitted field. This
# is the one place the prompt text lives: the schema below and the readback module's
# validation mirror its bounds (at most MAX_STEPS steps, at most 3
# parts), never restate it, and the per-clip duration sentence below
# extends this constant rather than copying it. The clock sentence has
# two variants: the burned-in wording when the uploaded bytes carry
# the overlaid clock, the plain wording otherwise.
SPARK_PROMPT_CLOCKED = (
    "A large m:ss.s clock is burned into the top-left corner of every "
    "frame: read each boundary from the on-screen clock first, then "
    "write the same time in seconds. "
)
SPARK_PROMPT_PLAIN_CLOCK = (
    "Read each boundary as an m:ss clock time first, then write the "
    "same time in seconds. "
)
SPARK_PROMPT = (
    "This is a laptop repair/teardown video. Partition the whole clip, "
    "from 0 seconds to its end, into ordered coarse repair phases "
    f"(at most {MAX_STEPS}), one phase per card: each phase is a stage of "
    "the repair bounded by its transition moments (flipping the device, "
    "opening the cover, removing or installing a part, closing up, "
    "powering on to verify), not a tool close-up, a part detail, or an "
    "overview cutaway -- never emit those as standalone steps. For each "
    "phase give a whole number from 1 in order, a short action title, "
    "start_s and end_s in seconds from the start of the video, tools used, "
    "up to 3 short part noun phrases visible in that phase each with a "
    "position hint "
    '(a visible cooling fan is always the phrase "fan"; a visible battery '
    'is always the phrase "battery"), and hazard tags. Phases must tile '
    "the clip with no gaps and no overlaps: the first phase starts at 0, "
    "each phase ends where the next begins, and the last phase ends at "
    "the end of the clip. " + SPARK_PROMPT_PLAIN_CLOCK + "Do not merge the "
    "opening intro and the next short step into one. Also give your best "
    'device_guess (or "unknown" when this is not a teardown) and a confidence '
    "from 0 to 1. Hazard tags must come from: battery, adhesive, heat, "
    "display, esd, data, sharp. For each phase also give kind, one of: "
    "intro for the opening introduction, repair for repair work, verify "
    "for powering on and checking, or outro_or_unrelated for a closing "
    "or unrelated segment that is not part of the repair (such as a "
    "parts teaser or showcase). The opening intro is always intro, "
    "never outro_or_unrelated."
)
# The clocked prompt is the same text with the burned-in clock
# sentence in place of the plain one; exactly one clock sentence is
# ever present.
if SPARK_PROMPT.count(SPARK_PROMPT_PLAIN_CLOCK) != 1:
    raise RuntimeError("SPARK_PROMPT must carry exactly one clock sentence")
SPARK_PROMPT_CLOCKED_FULL = SPARK_PROMPT.replace(
    SPARK_PROMPT_PLAIN_CLOCK, SPARK_PROMPT_CLOCKED
)


def spark_prompt_for_clip(
    clip_duration_s: float,
    *,
    clocked: bool = False,
    scene_cuts: Sequence[float] | None = None,
) -> str:
    """The extraction prompt bound to one clip's exact duration.

    The duration arrives per call from the chunk being sent (never a
    literal): the model is told the clip's exact length in seconds and
    that every emitted time must sit within ``[0, duration]``, so a
    run cannot honestly end past the chunk it was shown. ``clocked``
    selects the burned-in-clock sentence (the uploaded bytes carry
    the overlaid clock); ``scene_cuts`` lists the window-relative cut
    candidates, rounded to 0.1 s, with the validated guidance.
    """
    if (
        isinstance(clip_duration_s, bool)
        or not isinstance(clip_duration_s, (int, float))
        or not math.isfinite(clip_duration_s)
        or not clip_duration_s > 0
    ):
        raise ValueError(
            "clip_duration_s must be a finite number > 0,"
            f" got {clip_duration_s!r}"
        )
    if not isinstance(clocked, bool):
        raise ValueError(f"clocked must be a bool, got {clocked!r}")
    cuts: list[float] = []
    if scene_cuts is not None:
        if not isinstance(scene_cuts, (tuple, list)):
            raise ValueError(
                "scene_cuts must be None or a tuple/list of times,"
                f" got {type(scene_cuts).__name__}"
            )
        for cut in scene_cuts:
            if (
                isinstance(cut, bool)
                or not isinstance(cut, (int, float))
                or not math.isfinite(cut)
                or not cut >= 0
            ):
                raise ValueError(
                    f"scene cut times must be finite >= 0, got {cut!r}"
                )
            cuts.append(float(cut))
    duration = float(clip_duration_s)
    prompt = (
        SPARK_PROMPT_CLOCKED_FULL
        if clocked
        else SPARK_PROMPT
    )
    prompt += (
        f" This clip is exactly {duration:g} seconds long: every start_s"
        f" and end_s must be within [0, {duration:g}], and the last phase"
        f" must end at {duration:g}."
    )
    if cuts:
        listed = ", ".join(f"{cut:.1f}" for cut in cuts)
        # The validated wording, verbatim: with it one run
        # reached 7/9 and the consensus held 6/9. The probe's "every
        # step lasts at least 5 seconds" hint is NOT repeated here:
        # the fold pass enforces the configured minimum
        # deterministically, which a hint cannot do.
        prompt += (
            " Visible scene cuts happen at these times (seconds):"
            f" {listed}. A step boundary very often falls on one of"
            " these cuts, but not every cut is a step boundary and a"
            " boundary may fall between cuts. Treat each change of"
            " what is being done to the laptop (e.g. picking it up"
            " and flipping it, removing a part, a switch from hands"
            " to a computer screen) as a new step."
        )
    return prompt

# Strict schema mirroring app.spark.validate_spark_payload exactly: same
# keys, same bounds (at most MAX_STEPS steps, at most 3 parts), kind
# required in both so every new provider response tags each phase.
# Pre-kind payloads validate only through the explicit legacy path.
# Step-number order (1..N) rides in the prompt; the readback module still validates it.
SPARK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["device_guess", "confidence", "steps"],
    "properties": {
        "device_guess": {"type": "string"},
        "confidence": {"type": "number"},
        "steps": {
            "type": "array",
            "maxItems": MAX_STEPS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "number",
                    "start_s",
                    "end_s",
                    "action",
                    "tools",
                    "parts",
                    "hazards",
                    "kind",
                ],
                "properties": {
                    "number": {"type": "integer"},
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "action": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "parts": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["phrase", "position_hint"],
                            "properties": {
                                "phrase": {"type": "string"},
                                "position_hint": {"type": "string"},
                            },
                        },
                    },
                    "hazards": {"type": "array", "items": {"type": "string"}},
                    "kind": {
                        "type": "string",
                        "enum": list(PHASE_KINDS),
                    },
                },
            },
        },
    },
}

# Bounded usage for the worst-case reservation. Input: the spike
# measured ~273 input tokens per video second at 1x, so a 60 s chunk
# lands near 16.4k; 32k doubles it. Longer single windows scale the
# bound by the configured per-second rate instead (see
# :func:`window_input_tokens`). Output: the shared request cap from
# app.settings, binding on the API side, at 3.3x the 1x-measured
# maximum of 2,984 output tokens. The cap also keeps the honest
# per-run worst case inside a third of the $0.26 Spark reservation at
# the docs prices, which a 16k cap cannot do.
SPARK_MAX_INPUT_TOKENS = 32000


def _require_tokens_per_video_s(tokens_per_video_s: float) -> float:
    """Return the per-second input-token rate, or fail loudly."""
    if (
        isinstance(tokens_per_video_s, bool)
        or not isinstance(tokens_per_video_s, (int, float))
        or not math.isfinite(tokens_per_video_s)
        or not tokens_per_video_s > 0
    ):
        raise ValueError(
            "tokens_per_video_s must be a finite number > 0,"
            f" got {tokens_per_video_s!r}"
        )
    return float(tokens_per_video_s)


def window_input_tokens(window_s: float, tokens_per_video_s: float) -> int:
    """Worst-case input tokens for one extraction window.

    The 60 s chunk bound above stays the floor; longer windows scale
    by the configured per-second rate (measured ~273/s at 720p, so
    the configured default carries headroom for 1080p-capped
    previews). The ceiling keeps fractional rates honest.
    """
    if (
        isinstance(window_s, bool)
        or not isinstance(window_s, (int, float))
        or not math.isfinite(window_s)
        or not window_s > 0
    ):
        raise ValueError(
            f"window_s must be a finite number > 0, got {window_s!r}"
        )
    rate = _require_tokens_per_video_s(tokens_per_video_s)
    return max(
        SPARK_MAX_INPUT_TOKENS,
        math.ceil(float(window_s) * rate),
    )


def spark_worst_case(
    input_price: Decimal,
    output_price: Decimal,
    *,
    input_tokens: int = SPARK_MAX_INPUT_TOKENS,
    output_tokens: int = SPARK_MAX_OUTPUT_TOKENS,
) -> Decimal:
    """Return the per-run worst-case reservation for the active prices."""
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens <= 0
    ):
        raise ValueError(
            f"input_tokens must be a positive int, got {input_tokens!r}"
        )
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
    ):
        raise ValueError(
            f"output_tokens must be a positive int, got {output_tokens!r}"
        )
    return (
        Decimal(input_tokens) * input_price
        + Decimal(output_tokens) * output_price
    )


def spark_window_estimate_usd(
    window_s: float | None,
    input_price: Decimal,
    output_price: Decimal,
    runs: int,
    *,
    tokens_per_video_s: float,
    output_tokens: int = SPARK_MAX_OUTPUT_TOKENS,
) -> Decimal:
    """The ONE Spark reservation estimate for one extraction window.

    Run count times the per-run worst case for the window's
    input-token bound (the 60 s chunk floor, scaled by the
    per-second rate past it) plus the output cap. ``None`` sizes the
    chunk floor, exactly like a short window. Every place that
    admits, preflights, reserves, retries or caps a window prices it
    through this function, so planning, selection, admission, client
    holds and retry accounting cannot disagree. Anything misshaped
    fails loudly.
    """
    rate = _require_tokens_per_video_s(tokens_per_video_s)
    if window_s is None:
        input_tokens = SPARK_MAX_INPUT_TOKENS
    else:
        input_tokens = window_input_tokens(window_s, rate)
    for name, price in (
        ("input_price", input_price),
        ("output_price", output_price),
    ):
        if (
            not isinstance(price, Decimal)
            or not price.is_finite()
            or not price > 0
        ):
            raise ValueError(
                f"{name} must be a finite positive Decimal, got {price!r}"
            )
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs must be an int >= 1, got {runs!r}")
    estimate = (
        spark_worst_case(
            input_price,
            output_price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        * runs
    )
    # Canonical money form: strip the arithmetic's trailing zeros so
    # logs and refusal messages read $0.4086, never $0.40860000.
    stripped = estimate.normalize()
    if stripped.as_tuple().exponent >= 0:
        # An integral value normalizes to an exponent (1E+2): render
        # it back to plain fixed notation.
        stripped = Decimal(format(stripped, "f"))
    return stripped


class _Poison(Exception):
    """Internal: the call passed the point of possible charge but failed.

    Carries the cost the wrapper must settle as failed (measured when
    the 200 carried usable usage, else the per-run worst case) plus a
    reconcile note. Never escapes the client: the public method turns it
    into a poisoned result the wrapper's own validation rejects.
    """

    def __init__(self, cost_usd: Decimal, note: str, kind: SparkFailureKind = "malformed") -> None:
        super().__init__(note)
        self.cost_usd = cost_usd
        self.note = note
        self.kind = kind


def _require_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _require_money(value: object, what: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or not value > 0:
        raise ValueError(f"{what} must be a finite positive Decimal")
    return value


def _require_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not value > 0
    ):
        raise ValueError("timeout_s must be a positive number")
    return float(value)


@dataclass(frozen=True)
class _Priced:
    payload: Mapping[str, JSONValue]
    units_in: int
    units_out: int
    cost_usd: Decimal


class MetaSparkClient:
    """Live ``SparkStepsClient``: one HTTP call per ``run_steps`` invocation.

    Instances are per job: the constructor binds the job context the
    ``extract`` receipts persist through, and a call counter supplies
    the receipt ``run_index``. The API key travels only in the
    Authorization header and never reaches a message, receipt, or repr.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        input_price: Decimal,
        output_price: Decimal,
        timeout_s: float,
        ctx: JobContext,
        runs: int = SPARK_RUNS_PER_GUIDE,
        spread_s: float | None = None,
        span_mode: str | None = None,
        alignment_mode: str | None = None,
        generic_tokens: frozenset[str] | None = None,
        verb_synonyms: tuple[frozenset[str], ...] | None = None,
        negation_words: frozenset[str] | None = None,
        qualifiers: frozenset[str] | None = None,
        part_synonyms: tuple[frozenset[str], ...] | None = None,
        transition_verbs: frozenset[str] | None = None,
        verb_less_overlap_fraction: float | None = None,
        boundary_tolerance_s: float | None = None,
        consensus_min_runs: int | None = None,
        reservation_usd: Decimal | None = None,
        max_input_tokens: int | None = None,
        deadline_at: float | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("MetaSparkClient needs a non-empty api_key")
        self._api_key = api_key
        self._base_url = _require_text(base_url, "base_url").rstrip("/")
        self._model_id = _require_text(model_id, "model_id")
        try:
            self._record_model = SPARK_MODEL_RECORD_NAMES[self._model_id]
        except KeyError:
            raise ValueError(f"unknown Spark model id: {self._model_id!r}") from None
        self._input_price = _require_money(input_price, "input_price")
        self._output_price = _require_money(output_price, "output_price")
        self._timeout_s = _require_timeout(timeout_s)
        if deadline_at is not None and (
            isinstance(deadline_at, bool) or not isinstance(deadline_at, (int, float))
        ):
            raise ValueError("deadline_at must be a number on the monotonic clock")
        self._deadline_at = deadline_at
        self._ctx = ctx
        self._transport = transport if transport is not None else UrllibTransport()
        self._sleep = sleep
        self._monotonic = monotonic
        if max_input_tokens is None:
            max_input_tokens = SPARK_MAX_INPUT_TOKENS
        if (
            isinstance(max_input_tokens, bool)
            or not isinstance(max_input_tokens, int)
            or max_input_tokens <= 0
        ):
            raise ValueError(
                "max_input_tokens must be a positive int,"
                f" got {max_input_tokens!r}"
            )
        self._max_input_tokens = max_input_tokens
        self._worst_case = spark_worst_case(
            self._input_price, self._output_price,
            input_tokens=self._max_input_tokens,
        )
        # Each run reserves the honest per-run worst case at these prices;
        # the config refuses a zero share or a total past the reservation.
        defaults = SparkStageConfig()
        self.spark_stage_config = SparkStageConfig(
            runs=runs,
            spread_s=defaults.spread_s if spread_s is None else spread_s,
            run_shares=(self._worst_case,) * runs,
            model_id=self._model_id,
            span_mode=defaults.span_mode if span_mode is None else span_mode,
            alignment_mode=defaults.alignment_mode
            if alignment_mode is None
            else alignment_mode,
            generic_tokens=defaults.generic_tokens
            if generic_tokens is None
            else generic_tokens,
            verb_synonyms=defaults.verb_synonyms
            if verb_synonyms is None
            else verb_synonyms,
            negation_words=defaults.negation_words
            if negation_words is None
            else negation_words,
            qualifiers=defaults.qualifiers
            if qualifiers is None
            else qualifiers,
            part_synonyms=defaults.part_synonyms
            if part_synonyms is None
            else part_synonyms,
            transition_verbs=defaults.transition_verbs
            if transition_verbs is None
            else transition_verbs,
            verb_less_overlap_fraction=defaults.verb_less_overlap_fraction
            if verb_less_overlap_fraction is None
            else verb_less_overlap_fraction,
            boundary_tolerance_s=defaults.boundary_tolerance_s
            if boundary_tolerance_s is None
            else boundary_tolerance_s,
            consensus_min_runs=consensus_min_runs,
            reservation_usd=defaults.reservation_usd
            if reservation_usd is None
            else reservation_usd,
        )
        self._calls = 0

    def __repr__(self) -> str:  # never leak the key
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, "
            f"model_id={self._model_id!r})"
        )

    @property
    def bound_job_id(self) -> JobId:
        """The job whose context this client's receipts persist through."""
        return self._ctx.job_id

    def _request(self, video_file_id: str, prompt: str) -> _Priced:
        body = json.dumps(
            {
                "model": self._model_id,
                "max_output_tokens": SPARK_MAX_OUTPUT_TOKENS,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "steps",
                        "schema": SPARK_SCHEMA,
                        "strict": True,
                    }
                },
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_video", "file_id": video_file_id},
                        ],
                    }
                ],
            }
        ).encode("utf-8")
        budget_s = call_budget_s(self._timeout_s, self._deadline_at, self._monotonic)
        try:
            response = post_with_retry(
                self._transport,
                self._base_url + "/responses",
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                },
                body=body,
                timeout_s=self._timeout_s,
                sleep=self._sleep,
                deadline_s=budget_s,
                monotonic=self._monotonic,
            )
        except ProviderError as error:
            if error.indeterminate:
                kind = ("timeout" if isinstance(error, ProviderTimeout) else
                        "http_5xx" if isinstance(error, ProviderHttpError) else "transport")
                raise _Poison(
                    self._worst_case,
                    f"indeterminate Spark call ({error}); settles "
                    f"{self._worst_case}; reconcile against provider usage",
                    kind,
                ) from error
            if isinstance(error, ProviderConnectError):
                raise _Poison(Decimal("0"), str(error), "transport") from error
            raise
        try:
            payload = response_json(response)
        except ProviderMalformed as error:
            raise _Poison(
                self._worst_case,
                f"indeterminate Spark call (200 {error}); settles "
                f"{self._worst_case}; reconcile against provider usage",
                "empty" if not response.body.strip() else "malformed",
            ) from error
        try:
            units_in, units_out = usage_counts(payload)
        except ProviderMalformed:
            units_in, units_out = None, None
        measured = (
            Decimal(units_in) * self._input_price
            + Decimal(units_out) * self._output_price
            if units_in is not None and units_out is not None
            else None
        )
        try:
            text = output_text(payload)
            steps = json.loads(text)
        except (ProviderMalformed, ValueError) as error:
            raise _Poison(
                measured if measured is not None else self._worst_case,
                "indeterminate Spark call "
                f"(200 with unusable step text: {error}); settles "
                f"{measured if measured is not None else self._worst_case}; "
                "reconcile against provider usage",
                "empty" if isinstance(error, ProviderMalformed)
                and isinstance(payload, Mapping)
                and (not payload or isinstance(payload.get("output"), list))
                else "malformed",
            ) from error
        if not isinstance(steps, dict):
            raise _Poison(
                measured if measured is not None else self._worst_case,
                "indeterminate Spark call (200 with a non-object step "
                f"payload); settles {measured if measured is not None else self._worst_case}; "
                "reconcile against provider usage",
            )
        if measured is None:  # 200, good text, no usable usage: unmeasurable
            raise _Poison(
                self._worst_case,
                "indeterminate Spark call (200 with no usable usage); "
                f"settles {self._worst_case}; reconcile against provider usage",
            )
        assert units_in is not None and units_out is not None
        return _Priced(
            payload=steps, units_in=units_in, units_out=units_out, cost_usd=measured
        )

    def _failed_record(
        self,
        run_index: int,
        latency_ms: int,
        units_in: int | None,
        units_out: int | None,
        cost_usd: Decimal,
    ) -> None:
        _records.persist_receipt(
            self._ctx,
            _records.extract_record(
                job_id=self._ctx.job_id,
                model=self._record_model,
                run_index=run_index,
                latency_ms=latency_ms,
                units_in=units_in,
                units_out=units_out,
                cost_usd=cost_usd,
                status="failed",
            ),
        )

    def _poisoned(self, cost_usd: Decimal, note: str, latency_ms: int, kind: SparkFailureKind) -> SparkRunResult:
        """Return a result the wrapper settles as failed at ``cost_usd``.

        The payload is deliberately not a mapping; the wrapper classifies
        failure_kind before validating its shape. The note names the cause
        for reconciliation.
        """
        logger.warning("poisoned Spark result: %s", note)
        return SparkRunResult(
            payload=cast("Mapping[str, JSONValue]", f"INDETERMINATE: {note}"),
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            failure_kind=kind,
        )

    def run_steps(
        self,
        video_file_id: str,
        *,
        clip_duration_s: float | None = None,
        scene_cuts: Sequence[float] | None = None,
        clocked: bool = False,
    ) -> SparkRunResult:
        """Run one 1x extraction pass; record it; return payload plus actuals.

        With ``clip_duration_s`` the prompt states the clip's exact
        duration and bounds every time to ``[0, duration]``; without
        it the duration-less base prompt applies (legacy direct
        callers only — the pipeline always passes the chunk).
        ``clocked`` selects the burned-in-clock sentence and
        ``scene_cuts`` lists the cut candidates in the prompt.
        """
        _require_text(video_file_id, "video_file_id")
        if clip_duration_s is None:
            if scene_cuts is not None or clocked:
                raise ValueError(
                    "scene_cuts and clocked need clip_duration_s: the"
                    " duration-less prompt carries no per-clip tuning"
                )
            prompt = SPARK_PROMPT
        else:
            prompt = spark_prompt_for_clip(
                clip_duration_s, clocked=clocked, scene_cuts=scene_cuts
            )
        run_index = self._calls
        self._calls += 1
        started = self._monotonic()
        may_have_sent = False
        try:
            may_have_sent = True
            priced = self._request(video_file_id, prompt)
        except _Poison as poison:
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(run_index, latency_ms, None, None, poison.cost_usd)
            return self._poisoned(poison.cost_usd, poison.note, latency_ms, poison.kind)
        except ProviderError:
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(run_index, latency_ms, None, None, Decimal("0"))
            raise
        except Exception as error:
            # Unexpected local failure once bytes may have flown: a
            # poisoned result at the worst case, so the wrapper settles
            # that (never $0); the bug still surfaces in the log and in
            # the wrapper's rejection.
            logger.exception("unexpected Spark client failure")
            latency_ms = int((self._monotonic() - started) * 1000)
            self._failed_record(run_index, latency_ms, None, None, self._worst_case)
            return self._poisoned(
                self._worst_case,
                f"unexpected Spark client failure ({type(error).__name__}); "
                f"settles {self._worst_case}; reconcile against provider usage",
                latency_ms,
                "local",
            )
        except BaseException:
            latency_ms = int((self._monotonic() - started) * 1000)
            cost = self._worst_case if may_have_sent else Decimal("0")
            self._failed_record(run_index, latency_ms, None, None, cost)
            raise
        latency_ms = int((self._monotonic() - started) * 1000)
        if not _records.persist_receipt(
            self._ctx,
            _records.extract_record(
                job_id=self._ctx.job_id,
                model=self._record_model,
                run_index=run_index,
                latency_ms=latency_ms,
                units_in=priced.units_in,
                units_out=priced.units_out,
                cost_usd=priced.cost_usd,
                status="succeeded",
            ),
        ):
            return self._poisoned(
                priced.cost_usd,
                f"extract receipt lost after a retry; settles actual {priced.cost_usd}",
                latency_ms,
                "local",
            )
        return SparkRunResult(
            payload=priced.payload,
            cost_usd=priced.cost_usd,
            latency_ms=latency_ms,
        )
