"""Provider factory: one config switch between replay and live clients.

``select_providers`` reads ``settings.providers`` and returns the
trio the stage wrappers invoke as their injected ``client``. Live mode
binds each client to the calling job's context (receipts persist
through it) and refuses to build without a key; replay mode returns
the integration code's recorded trio untouched. Either way the factory
first proves the configured worst case still fits the job budget and
refuses, before any request, a config that cannot (run count past the
ledger's Spark count cap, or prices whose worst case breaks a
reservation). Nothing is installed process-wide: each client carries
its job's immutable stage config (Spark run count, spread, per-run
shares and model id; SAM thresholds, per-call price and pace), which
the stage wrappers read from the client they are handed, so two jobs
selected with different configs never run under each other's. Replay
clients get the same config through a thin delegating wrapper. Live
clients also carry the job's deadline on the monotonic clock, so every
call, retry and backoff ends by it. The service calls this factory for
every live-mode job (see :func:`app.service.live_pipeline_clients`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.contracts import (
    JOB_DEADLINE_S,
    MAX_READBACKS_PER_JOB,
    READBACK_COST_PER_CALL_USD,
    READBACK_MAX_RESERVATION_USD,
    SAM_MAX_CALLS_PER_JOB,
    SAM_MAX_RESERVATION_USD,
    JobContext,
)
from app.ledger import CATEGORY_CAP_USD, CATEGORY_COUNT_CAP
from app.providers._transport import HttpTransport
from app.providers.meta_readback import MetaReadbackClient, readback_worst_case
from app.providers.meta_sam import MetaSamClient
from app.providers.meta_spark import (
    SPARK_MAX_INPUT_TOKENS,
    MetaSparkClient,
    spark_window_estimate_usd,
    spark_worst_case,
    window_input_tokens,
)
from app.sam import SamStageConfig
from app.settings import ProviderSettings, Settings
from app.spark import SparkStageConfig


@dataclass(frozen=True)
class Providers:
    """One client per stage surface. Replay or live, never mixed per run."""

    spark: Any
    sam: Any
    readback: Any


def _validate_against_budget(
    tuned: ProviderSettings,
    *,
    input_tokens: int = SPARK_MAX_INPUT_TOKENS,
    spark_cap_usd: Decimal,
) -> Decimal:
    """Refuse a config whose worst case cannot fit the ledger.

    Runs before any request and before anything installs: a run count
    below two cannot form a median consensus, a run count past the
    ledger's Spark count cap would die on its fourth reservation, and
    prices whose worst case breaks the Spark, SAM or read-back
    reservation (or the job envelope) would overrun at settle time.
    Every refusal names the cap it would break. Returns the per-run
    Spark worst case the factory installs as the run shares.
    ``input_tokens`` scales the bound past the 60 s chunk floor for
    longer single windows. ``spark_cap_usd`` is the installed Spark
    cap the window's total must fit — the ledger default for
    no-window jobs, the window-sized cap for explicit windows — and
    the job envelope scales with it (Spark plus the frozen SAM and
    read-back maxima).
    """
    for name in (
        "spark_input_usd_per_token",
        "spark_output_usd_per_token",
        "sam_usd_per_call",
    ):
        price = getattr(tuned, name)
        if not isinstance(price, Decimal) or not price.is_finite() or not price > 0:
            raise RuntimeError(f"{name}={price!r} must be a finite positive price")
    runs = tuned.spark_runs
    if not 2 <= tuned.consensus_min_runs <= runs:
        raise RuntimeError(
            f"consensus_min_runs={tuned.consensus_min_runs} must be between 2 and"
            f" spark_runs={runs}"
        )
    if runs < 2:
        raise RuntimeError(
            f"spark_runs={runs} cannot form a median consensus: "
            "at least two runs"
        )
    spark_count_cap = CATEGORY_COUNT_CAP["spark"]
    if runs > spark_count_cap:
        raise RuntimeError(
            f"spark_runs={runs} exceeds the ledger's {spark_count_cap}-call "
            "Spark count cap"
        )
    try:
        spark_worst = spark_worst_case(
            tuned.spark_input_usd_per_token,
            tuned.spark_output_usd_per_token,
            input_tokens=input_tokens,
        )
    except ValueError as error:
        raise RuntimeError(f"invalid Spark input-token bound: {error}") from error
    if (
        not isinstance(spark_cap_usd, Decimal)
        or not spark_cap_usd.is_finite()
        or not spark_cap_usd > 0
    ):
        raise RuntimeError(
            f"spark_cap_usd must be a finite positive Decimal,"
            f" got {spark_cap_usd!r}"
        )
    spark_total = spark_worst * runs
    if spark_total > spark_cap_usd:
        raise RuntimeError(
            f"spark_runs={runs} at ${tuned.spark_input_usd_per_token}/"
            f"${tuned.spark_output_usd_per_token} per token reserves "
            f"${spark_total}, past the ${spark_cap_usd} Spark reservation"
        )
    sam_total = tuned.sam_usd_per_call * SAM_MAX_CALLS_PER_JOB
    sam_cap = CATEGORY_CAP_USD["sam"]
    if sam_total > sam_cap:
        raise RuntimeError(
            f"sam_usd_per_call=${tuned.sam_usd_per_call} over "
            f"{SAM_MAX_CALLS_PER_JOB} calls reserves ${sam_total}, past "
            f"the ${sam_cap} SAM reservation"
        )
    # The read-back worst case is checked against its own $0.04
    # per-call hold: read_back reserves that fixed hold and settles
    # the measured cost, so prices whose worst case exceeds it would
    # pass every other check and then fail on settlement overrun.
    # Read-backs still count toward the job envelope below.
    rb_worst = readback_worst_case(
        tuned.spark_input_usd_per_token, tuned.spark_output_usd_per_token
    )
    if rb_worst > READBACK_COST_PER_CALL_USD:
        raise RuntimeError(
            f"spark prices reserve a ${rb_worst} read-back worst case,"
            f" past the ${READBACK_COST_PER_CALL_USD} per-call"
            " read-back reservation"
        )
    job_total = (
        spark_total + SAM_MAX_RESERVATION_USD + rb_worst * MAX_READBACKS_PER_JOB
    )
    job_envelope = (
        spark_cap_usd
        + SAM_MAX_RESERVATION_USD
        + READBACK_MAX_RESERVATION_USD
    )
    if job_total > job_envelope:
        raise RuntimeError(
            f"worst-case job ${job_total} exceeds the "
            f"${job_envelope} job reservation envelope"
        )
    return spark_worst


class _ConfiguredReplay:
    """A recorded replay client plus its job's stage config.

    The config rides as the attribute the stage wrappers read
    (``spark_stage_config`` / ``sam_stage_config``); every other
    attribute delegates to the recorded client untouched.
    """

    def __init__(self, inner: Any, **config: Any) -> None:
        self._inner = inner
        self.__dict__.update(config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def whole_video_fits(
    window_s: float,
    input_price: Decimal,
    output_price: Decimal,
    runs: int,
    *,
    tokens_per_video_s: float,
    spark_cap_usd: Decimal,
) -> bool:
    """True when one window of ``window_s`` reserves inside the Spark cap.

    The window's shared estimate (see
    :func:`app.providers.meta_spark.spark_window_estimate_usd`) must
    fit the configured Spark reservation — the same comparison the
    factory enforces when it builds the window's clients, so planning
    and selection agree and a window planned single never refuses at
    build time. Anything misshaped fails loudly (planning must not
    silently chunk or single on bad inputs).
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
    if (
        not isinstance(spark_cap_usd, Decimal)
        or not spark_cap_usd.is_finite()
        or not spark_cap_usd > 0
    ):
        raise ValueError(
            "spark_cap_usd must be a finite positive Decimal,"
            f" got {spark_cap_usd!r}"
        )
    estimate = spark_window_estimate_usd(
        window_s,
        input_price,
        output_price,
        runs,
        tokens_per_video_s=tokens_per_video_s,
    )
    return estimate <= spark_cap_usd


def spark_window_cap_usd(
    window_s: float | None,
    input_price: Decimal,
    output_price: Decimal,
    runs: int,
    *,
    tokens_per_video_s: float,
) -> Decimal:
    """The Spark job-context cap for one window: the ledger default,
    raised to the window's shared estimate when the window needs more.

    The pipeline admits each chunk child with this cap and the factory
    installs this same reservation on the window's clients, so a
    client's holds can never exceed the cap its job context enforces.
    ``None`` sizes the chunk floor.
    """
    return max(
        CATEGORY_CAP_USD["spark"],
        spark_window_estimate_usd(
            window_s,
            input_price,
            output_price,
            runs,
            tokens_per_video_s=tokens_per_video_s,
        ),
    )


def _spark_config(
    tuned: ProviderSettings,
    spark_worst: Decimal,
    *,
    span_mode: str,
    reservation_usd: Decimal,
) -> SparkStageConfig:
    return SparkStageConfig(
        runs=tuned.spark_runs,
        spread_s=tuned.time_spread_flag_s,
        run_shares=(spark_worst,) * tuned.spark_runs,
        reservation_usd=reservation_usd,
        model_id=tuned.spark_model_id,
        span_mode=span_mode,
        alignment_mode=tuned.step_alignment_mode,
        generic_tokens=tuned.step_alignment_generic_tokens,
        verb_synonyms=tuned.step_alignment_verb_synonyms,
        negation_words=tuned.step_alignment_negation_words,
        qualifiers=tuned.step_alignment_qualifiers,
        part_synonyms=tuned.step_alignment_part_synonyms,
        transition_verbs=tuned.step_alignment_transition_verbs,
        verb_less_overlap_fraction=(
            tuned.step_alignment_verb_less_overlap_fraction
        ),
        boundary_tolerance_s=tuned.consensus_boundary_tolerance_s,
        consensus_min_runs=tuned.consensus_min_runs,
    )


def _sam_config(tuned: ProviderSettings) -> SamStageConfig:
    return SamStageConfig(
        thresholds={
            "fan": tuned.sam_threshold_fan,
            "battery": tuned.sam_threshold_battery,
        },
        price_per_call=tuned.sam_usd_per_call,
        max_starts_per_minute=tuned.sam_max_starts_per_minute,
    )


def _job_remaining_s(ctx: JobContext) -> float:
    """The job's remaining seconds: the ledger's deadline, else a fresh 240 s."""
    remaining = getattr(ctx, "remaining_s", None)
    return float(remaining()) if callable(remaining) else float(JOB_DEADLINE_S)


def select_providers(
    settings: Settings,
    ctx: JobContext | None = None,
    *,
    transport: HttpTransport | None = None,
    replay: Providers | None = None,
    window_s: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Providers:
    """Return the replay or live trio for one job.

    Live mode needs the job context (receipts persist through it) and
    ``META_API_KEY`` from settings; replay mode needs the caller to
    pass the recorded trio. Anything else fails closed. Live clients
    get the job's deadline as ``monotonic() + remaining`` (the ledger's
    ``remaining_s`` when the context has one, else the full 240 s).
    ``window_s`` sizes the Spark shares and reservation for the window
    these clients will extract (the chunk path passes each window's
    decoded duration); without it the chunk floor and the ledger's
    default Spark cap apply, matching the context a direct job was
    admitted with. An explicit window installs the window-sized cap —
    the same call the pipeline uses to admit the window's child — but
    only when the window's shared estimate fits the configured
    ceiling; anything past it refuses before any request.
    """
    tuned = settings.providers
    rate = settings.spark_input_tokens_per_video_s
    ceiling = settings.spark_reservation_usd
    if window_s is None:
        input_tokens = SPARK_MAX_INPUT_TOKENS
        sizing_window: float | None = None
    else:
        try:
            scaled = window_input_tokens(window_s, rate)
        except ValueError as error:
            raise RuntimeError(f"invalid window_s: {error}") from error
        # Worst-case scaling is live money: replay settles recorded
        # pennies against the floor shares, never scaled holds.
        input_tokens = scaled if tuned.mode == "live" else SPARK_MAX_INPUT_TOKENS
        sizing_window = window_s if tuned.mode == "live" else None
    if tuned.mode == "live" and window_s is None:
        # A direct job's context was admitted without an override, so
        # it enforces the default: validate against and install that.
        spark_cap = CATEGORY_CAP_USD["spark"]
    else:
        # An explicit window (or replay's floor): refuse a window the
        # configured ceiling cannot fund, else install the window cap.
        estimate = spark_window_estimate_usd(
            sizing_window,
            tuned.spark_input_usd_per_token,
            tuned.spark_output_usd_per_token,
            tuned.spark_runs,
            tokens_per_video_s=rate,
        )
        if estimate > ceiling:
            raise RuntimeError(
                f"spark_runs={tuned.spark_runs} at"
                f" ${tuned.spark_input_usd_per_token}/"
                f"${tuned.spark_output_usd_per_token} per token reserves"
                f" ${estimate}, past the ${ceiling} Spark reservation"
            )
        spark_cap = spark_window_cap_usd(
            sizing_window,
            tuned.spark_input_usd_per_token,
            tuned.spark_output_usd_per_token,
            tuned.spark_runs,
            tokens_per_video_s=rate,
        )
    spark_worst = _validate_against_budget(
        tuned, input_tokens=input_tokens, spark_cap_usd=spark_cap,
    )
    spark_config = _spark_config(
        tuned, spark_worst, span_mode=settings.step_span_mode,
        reservation_usd=spark_cap,
    )
    sam_config = _sam_config(tuned)
    mode = tuned.mode
    if mode == "replay":
        if replay is None:
            raise RuntimeError(
                "provider mode 'replay' needs the recorded trio; "
                "the integration code supplies it"
            )
        return Providers(
            spark=_ConfiguredReplay(replay.spark, spark_stage_config=spark_config),
            sam=_ConfiguredReplay(replay.sam, sam_stage_config=sam_config),
            readback=replay.readback,
        )
    if mode != "live":
        raise ValueError(f"unknown provider mode: {mode!r}")
    if ctx is None:
        raise RuntimeError("provider mode 'live' needs the job context")
    key = settings.meta_api_key
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError(
            "provider mode 'live' refuses to start without: META_API_KEY"
        )
    deadline_at = monotonic() + _job_remaining_s(ctx)
    return Providers(
        spark=MetaSparkClient(
            api_key=key,
            base_url=tuned.meta_api_base_url,
            model_id=tuned.spark_model_id,
            input_price=tuned.spark_input_usd_per_token,
            output_price=tuned.spark_output_usd_per_token,
            timeout_s=tuned.spark_timeout_s,
            max_input_tokens=input_tokens,
            ctx=ctx,
            runs=tuned.spark_runs,
            spread_s=tuned.time_spread_flag_s,
            span_mode=settings.step_span_mode,
            alignment_mode=tuned.step_alignment_mode,
            generic_tokens=tuned.step_alignment_generic_tokens,
            verb_synonyms=tuned.step_alignment_verb_synonyms,
            negation_words=tuned.step_alignment_negation_words,
            qualifiers=tuned.step_alignment_qualifiers,
            part_synonyms=tuned.step_alignment_part_synonyms,
            transition_verbs=tuned.step_alignment_transition_verbs,
            verb_less_overlap_fraction=(
                tuned.step_alignment_verb_less_overlap_fraction
            ),
            boundary_tolerance_s=tuned.consensus_boundary_tolerance_s,
            consensus_min_runs=tuned.consensus_min_runs,
            reservation_usd=spark_cap,
            deadline_at=deadline_at,
            transport=transport,
            monotonic=monotonic,
        ),
        sam=MetaSamClient(
            api_key=key,
            base_url=tuned.meta_api_base_url,
            model_id=tuned.sam_model_id,
            price_per_call=tuned.sam_usd_per_call,
            timeout_s=tuned.sam_timeout_s,
            phrases={
                "fan": tuned.sam_phrase_fan,
                "battery": tuned.sam_phrase_battery,
            },
            thresholds=sam_config.thresholds,
            ctx=ctx,
            max_starts_per_minute=tuned.sam_max_starts_per_minute,
            deadline_at=deadline_at,
            transport=transport,
            monotonic=monotonic,
        ),
        readback=MetaReadbackClient(
            api_key=key,
            base_url=tuned.meta_api_base_url,
            model_id=tuned.spark_model_id,
            input_price=tuned.spark_input_usd_per_token,
            output_price=tuned.spark_output_usd_per_token,
            timeout_s=tuned.readback_timeout_s,
            ctx=ctx,
            deadline_at=deadline_at,
            transport=transport,
            monotonic=monotonic,
        ),
    )


__all__ = ("Providers", "select_providers")
