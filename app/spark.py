"""Spark step extraction: N 1x runs, strict schema, median consensus. The readback module owns this.

Pipeline per guide: reserve -> invoke -> settle, once per configured run
against the standard runtime (never contributor mode), then validate
each raw payload against the strict schema, cluster samples with the
shared temporal alignment (overlap/order first, action/part tokens as
conflict guards; never by ordinal), and take median start/end times
over each linked component holding samples from a majority of runs. A
step whose run spread strictly exceeds the flag line is flagged
(`time_needs_check`), as is each of two kept steps whose windows
honestly disagree. StepsDisagreement when no component holds a
majority of runs, or when more candidates survive than the 12-step
bundle holds: real disagreement is reported, never consensus. The old
exact semantic-key grouping stays available only as the configured
`exact` rollback mode.
All inferred times are approximate. The run count, flag line and
per-run reservation come from the client's own `SparkStageConfig`
(see `spark_config_of`) and default to the frozen contracts.
In contiguous phases mode (`span_mode="contiguous"`,
`alignment_mode="align") consensus matches runs by TIME, not
wording: each valid run is a contiguous partition of the chunk, so
internal boundary times cluster across runs (tolerance
`boundary_tolerance_s`, defaulting to the flag line; support from a
majority of runs; median time) and the kept boundaries tile the
known chunk extent (else the observed span) with no gaps by
construction; a run missing the known extent beyond tolerance is
excluded like any invalid run. Wording only chooses
each segment's label (the most-agreed `same_action` group, then the
largest time overlap) and flags disagreement (`time_needs_check`):
it never drops time. Legacy/sparse mode and `exact` mode keep the
wording-grouped median consensus exactly as before.

Refusal (before any SAM call): unknown device, below-cutoff Spark
confidence, or zero consensus steps raise NotATeardown carrying a
StepsRefusal (`not_a_teardown` + sample CTA). A reservation refusal from
the ledger module propagates before that client invocation; the run count is never
silently reduced.

The readback module owns the SparkClient method surface used here: the injected client
must provide `run_steps(video_file_id, *, clip_duration_s=None, scene_cuts=None, clocked=False) ->
SparkRunResult` (and, for read-back, `read_back(...)` in
app.readback). Production adapters read META_API_KEY from the process
environment only (app.settings); this module ships no network adapter,
fakes only.

Caching: `extract_steps` always runs live. `extract_steps_cached` layers
the replayable stage cache on top: raw runs and consensus are keyed by
job/input/prompt/schema/model/run-count/spread/chunk-extent digest in
time-consensus mode (no cross-job, cross-model, cross-config or
cross-extent reuse; legacy/sparse and exact modes keep the pre-extent
digest shape), and a cache hit returns the bundle with replay=True
and zero context/client calls.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal, Protocol

from app import stage_cache
from app import step_alignment
from app.chunks import DEFAULT_SPAN_MODE, SPAN_MODES, ChunkWindow
from app.contracts import (
    BoundaryInfo,
    ExcludedSpan,
    JSONValue,
    MAX_PARTS_PER_STEP,
    MAX_STEPS,
    SAM_MAX_CALLS_PER_JOB,
    SPARK_3RUNS_RESERVATION_USD,
    SPARK_RUNS_PER_GUIDE,
    TIME_SPREAD_FLAG_S,
    JobContext,
    PartRequest,
    SparkClient,
    Step,
    StepBundle,
)
from app.step_times import median_time, time_spread

SPARK_MODEL_ID = "muse-spark-1.3"
SPARK_PROMPT_VERSION = "l2-v6"
SPARK_SCHEMA_VERSION = "l2-v4"

# Closed phase-kind set Spark tags each phase with: the
# opening introduction, repair work, power-on verification, or a
# closing/unrelated segment that is not part of the repair. The prompt
# and the strict provider schema carry this set with kind required, so
# every new provider response tags each phase; pre-kind payloads (old
# fixtures, recorded runs) validate only through the explicit legacy
# path below, as repair.
PHASE_KINDS: tuple[str, ...] = ("intro", "repair", "verify", "outro_or_unrelated")
DEFAULT_PHASE_KIND = "repair"
OUTRO_PHASE_KIND = "outro_or_unrelated"

RESERVE_KIND = "spark_steps"

# Default per-run worst-case shares; the penny split sums exactly to the
# $0.26 three-run reservation (this replaces an earlier $0.20
# ceiling). Live mode carries shares derived from the active prices on
# its client's SparkStageConfig; these stay the default.
SPARK_RUN_SHARES: tuple[Decimal, Decimal, Decimal] = (
    Decimal("0.09"),
    Decimal("0.09"),
    Decimal("0.08"),
)
if sum(SPARK_RUN_SHARES, Decimal("0")) != SPARK_3RUNS_RESERVATION_USD:
    raise RuntimeError("SPARK_RUN_SHARES must total SPARK_3RUNS_RESERVATION_USD")


@dataclass(frozen=True)
class SparkStageConfig:
    """Immutable extraction config one provider carries for its job.

    Run count, disagreement flag line, per-run reservation shares and the
    model id (cache keys). Nothing is installed process-wide: a client
    exposes its config as ``spark_stage_config`` and the wrappers read it
    from the client they were handed, so two jobs with different configs
    never see each other's. The shape is checked on construction (at
    least two runs for a median consensus, a finite positive flag line,
    one finite positive share per run totaling within the reservation),
    so a bad config fails before any reservation. The reservation
    defaults to the frozen $0.26 chunk envelope; the provider factory
    carries the configured window envelope for longer single windows.
    """

    runs: int = SPARK_RUNS_PER_GUIDE
    spread_s: float = float(TIME_SPREAD_FLAG_S)
    run_shares: tuple[Decimal, ...] = SPARK_RUN_SHARES
    model_id: str = SPARK_MODEL_ID
    span_mode: str = DEFAULT_SPAN_MODE
    alignment_mode: str = step_alignment.DEFAULT_ALIGNMENT_MODE
    generic_tokens: frozenset[str] = step_alignment.GENERIC_TOKENS
    verb_synonyms: tuple[frozenset[str], ...] = step_alignment.VERB_SYNONYM_GROUPS
    negation_words: frozenset[str] = step_alignment.NEGATION_WORDS
    qualifiers: frozenset[str] = step_alignment.QUALIFIERS
    part_synonyms: tuple[frozenset[str], ...] = step_alignment.PART_SYNONYM_GROUPS
    transition_verbs: frozenset[str] = step_alignment.TRANSITION_VERBS
    verb_less_overlap_fraction: float = (
        step_alignment.DEFAULT_VERB_LESS_OVERLAP_FRACTION
    )
    boundary_tolerance_s: float | None = None
    reservation_usd: Decimal = SPARK_3RUNS_RESERVATION_USD

    def __post_init__(self) -> None:
        runs = self.runs
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 2:
            raise ValueError(f"spark runs must be an int >= 2, got {runs!r}")
        spread_s = self.spread_s
        if (
            isinstance(spread_s, bool)
            or not isinstance(spread_s, (int, float))
            or not math.isfinite(spread_s)
            or not spread_s > 0
        ):
            raise ValueError(f"spark spread must be a finite positive number, got {spread_s!r}")
        shares = tuple(self.run_shares)
        if len(shares) != runs:
            raise ValueError(
                f"spark shares ({len(shares)}) must run parallel to runs ({runs})"
            )
        for share in shares:
            if not isinstance(share, Decimal) or not share.is_finite() or not share > 0:
                raise ValueError(f"spark shares must be finite positive Decimals, got {share!r}")
        reservation = self.reservation_usd
        if (
            not isinstance(reservation, Decimal)
            or not reservation.is_finite()
            or not reservation > 0
        ):
            raise ValueError(
                "spark reservation_usd must be a finite positive Decimal,"
                f" got {reservation!r}"
            )
        if sum(shares, Decimal("0")) > reservation:
            raise ValueError(
                f"spark shares total {sum(shares, Decimal('0'))}, past the "
                f"${reservation} Spark reservation"
            )
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("spark model_id must be a non-empty string")
        if self.span_mode not in SPAN_MODES:
            raise ValueError(
                "spark span_mode must be one of "
                f"{', '.join(SPAN_MODES)}, got {self.span_mode!r}"
            )
        if self.alignment_mode not in step_alignment.ALIGNMENT_MODES:
            raise ValueError(
                "spark alignment_mode must be one of "
                f"{', '.join(step_alignment.ALIGNMENT_MODES)}, got {self.alignment_mode!r}"
            )
        if not isinstance(self.generic_tokens, frozenset):
            raise ValueError(
                "spark generic_tokens must be a frozenset, "
                f"got {type(self.generic_tokens).__name__}"
            )
        step_alignment.normalize_generic_tokens(self.generic_tokens)
        if not isinstance(self.verb_synonyms, tuple) or any(
            not isinstance(group, frozenset)
            for group in self.verb_synonyms
        ):
            raise ValueError(
                "spark verb_synonyms must be a tuple of frozensets, "
                f"got {type(self.verb_synonyms).__name__}"
            )
        step_alignment.normalize_verb_synonyms(self.verb_synonyms)
        if not isinstance(self.negation_words, frozenset):
            raise ValueError(
                "spark negation_words must be a frozenset, "
                f"got {type(self.negation_words).__name__}"
            )
        step_alignment.normalize_negation_words(self.negation_words)
        if not isinstance(self.qualifiers, frozenset):
            raise ValueError(
                "spark qualifiers must be a frozenset, "
                f"got {type(self.qualifiers).__name__}"
            )
        step_alignment.normalize_qualifiers(self.qualifiers)
        if not isinstance(self.part_synonyms, tuple) or any(
            not isinstance(group, frozenset)
            for group in self.part_synonyms
        ):
            raise ValueError(
                "spark part_synonyms must be a tuple of frozensets, "
                f"got {type(self.part_synonyms).__name__}"
            )
        step_alignment.normalize_part_synonyms(self.part_synonyms)
        if not isinstance(self.transition_verbs, frozenset):
            raise ValueError(
                "spark transition_verbs must be a frozenset, "
                f"got {type(self.transition_verbs).__name__}"
            )
        step_alignment.normalize_transition_verbs(self.transition_verbs)
        fraction = self.verb_less_overlap_fraction
        step_alignment.normalize_verb_less_overlap_fraction(fraction)
        boundary = self.boundary_tolerance_s
        if boundary is not None and (
            isinstance(boundary, bool)
            or not isinstance(boundary, (int, float))
            or not math.isfinite(boundary)
            or not boundary > 0
        ):
            raise ValueError(
                "spark boundary_tolerance_s must be None (use spread_s) or a"
                f" finite positive number, got {boundary!r}"
            )
        object.__setattr__(self, "spread_s", float(spread_s))
        object.__setattr__(self, "run_shares", shares)
        object.__setattr__(
            self, "verb_less_overlap_fraction", float(fraction)
        )
        if boundary is not None:
            object.__setattr__(self, "boundary_tolerance_s", float(boundary))


DEFAULT_SPARK_CONFIG = SparkStageConfig()


def spark_config_of(client: object) -> SparkStageConfig:
    """The config a client carries, or the frozen defaults (replay fakes)."""
    config = getattr(client, "spark_stage_config", None)
    return config if isinstance(config, SparkStageConfig) else DEFAULT_SPARK_CONFIG


@dataclass(frozen=True)
class ChunkExtent:
    """The known chunk window time consensus tiles: ``[0, duration_s]``.

    Per chunk, never per job: the pipeline builds one from the cut
    being sent (probed decoded duration) and the active contiguity
    tolerance, so the consensus extent and the exclusion bound always
    match the gate the bundle must pass. A run that starts past
    ``tolerance_s`` or ends outside ``duration_s`` plus or minus
    ``tolerance_s`` is excluded from time consensus, never clamped.
    """

    duration_s: float
    tolerance_s: float

    def __post_init__(self) -> None:
        duration = self.duration_s
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not duration > 0
        ):
            raise ValueError(
                f"chunk duration must be a finite number > 0, got {duration!r}"
            )
        tolerance = self.tolerance_s
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or not tolerance >= 0
        ):
            raise ValueError(
                f"chunk tolerance must be a finite number >= 0, got {tolerance!r}"
            )
        object.__setattr__(self, "duration_s", float(duration))
        object.__setattr__(self, "tolerance_s", float(tolerance))


def _uses_chunk_extent(config: SparkStageConfig) -> bool:
    """True when this config's extraction path binds the chunk extent.

    Only time-consensus extraction (contiguous span plus align mode)
    sends the duration-bound prompt, tiles the known extent, and
    extent-keys its cache digests; legacy/sparse and exact modes keep
    the duration-less prompt and the pre-extent call path and digest
    shape, exactly as before.
    """
    return (
        config.span_mode == "contiguous"
        and config.alignment_mode == "align"
    )


# readback-local config: a device guess below even odds is not a teardown claim.
# (Distinct from the maintainer-frozen read-back confidence cutoff, which the readback module only reads.)
SPARK_CONFIDENCE_CUTOFF = 0.50

SAMPLE_CTA = "Try the sample guide"


SparkFailureKind = Literal["timeout", "transport", "http_5xx", "empty", "malformed", "local"]


@dataclass(frozen=True)
class SparkRunResult:
    """One raw Spark run plus its measured cost/latency (fakeable)."""

    payload: Mapping[str, JSONValue]
    cost_usd: Decimal
    latency_ms: int
    failure_kind: SparkFailureKind | None = None


class SparkStepsClient(Protocol):
    """readback-module client surface for step extraction (see module docstring)."""

    def run_steps(
        self,
        video_file_id: str,
        *,
        clip_duration_s: float | None = None,
        scene_cuts: tuple[float, ...] | None = None,
        clocked: bool = False,
    ) -> SparkRunResult: ...


class SparkSchemaError(ValueError):
    """A raw Spark payload violates the strict schema (fail closed)."""


class SparkClientError(RuntimeError):
    """The client returned an unusable run result (bad cost/latency/shape)."""


class SparkServiceError(SparkClientError):
    """A provider service failure; only this failure may be retried."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


class StepsDisagreement(ValueError):
    """The three runs disagree beyond any two-run majority: no honest consensus."""


@dataclass(frozen=True)
class StepsRefusal:
    """Non-teardown outcome: no SAM call is authorized, show the sample CTA."""

    reason: Literal["unknown_device", "low_confidence", "zero_steps"]
    verdict: Literal["not_a_teardown"] = "not_a_teardown"
    sample_cta: str = SAMPLE_CTA


class NotATeardown(Exception):
    """Raised by extract_steps when the video is refused before SAM."""

    def __init__(self, refusal: StepsRefusal) -> None:
        super().__init__(f"{refusal.verdict}: {refusal.reason}")
        self.refusal = refusal


@dataclass(frozen=True)
class CachedExtraction:
    bundle: StepBundle
    replay: bool  # True only for a cache hit (zero model calls)


@dataclass(frozen=True)
class _ValidPart:
    phrase: str
    position_hint: str


@dataclass(frozen=True)
class _ValidStep:
    number: int
    start_s: float
    end_s: float
    action: str
    tools: tuple[str, ...]
    parts: tuple[_ValidPart, ...]
    hazards: tuple[str, ...]
    kind: str = DEFAULT_PHASE_KIND


@dataclass(frozen=True)
class _ValidRun:
    device_guess: str
    confidence: float
    steps: tuple[_ValidStep, ...]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_keys(mapping: Mapping[str, object], keys: set[str], where: str) -> None:
    extra = set(mapping) - keys
    missing = keys - set(mapping)
    if extra or missing:
        raise SparkSchemaError(f"{where}: expected keys {sorted(keys)}, got {sorted(mapping)}")


def _require_text(value: object, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SparkSchemaError(f"{where}: expected non-empty string, got {value!r}")
    return value


def _stored_at_s(value: object) -> float | None:
    """Finite store time for a bundle ``stored_at`` value, else None.

    JSON integers have arbitrary precision, so ``float()`` on a huge
    one raises ``OverflowError``: convert defensively and fail closed
    to None (rerun the chunk) instead of failing the whole pipeline.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        moment = float(value)
    except (TypeError, OverflowError, ValueError):
        return None
    if not math.isfinite(moment):
        return None
    return moment


def _require_number(value: object, where: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise SparkSchemaError(f"{where}: expected finite number, got {value!r}")
    assert isinstance(value, (int, float))
    return float(value)


def _require_str_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SparkSchemaError(f"{where}: expected list, got {value!r}")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_require_text(item, f"{where}[{index}]"))
    return tuple(items)


def _validate_part(value: object, where: str) -> _ValidPart:
    if not isinstance(value, Mapping):
        raise SparkSchemaError(f"{where}: expected mapping, got {value!r}")
    _require_keys(value, {"phrase", "position_hint"}, where)
    phrase = _require_text(value["phrase"], f"{where}.phrase")
    hint = value["position_hint"]
    if not isinstance(hint, str):
        raise SparkSchemaError(f"{where}.position_hint: expected string, got {hint!r}")
    return _ValidPart(phrase=phrase, position_hint=hint)


def _validate_kind(value: object, where: str) -> str:
    """One phase kind: exactly one of the closed set (no null, no free text)."""
    if not isinstance(value, str) or value not in PHASE_KINDS:
        raise SparkSchemaError(
            f"{where}: expected one of {', '.join(PHASE_KINDS)}, got {value!r}"
        )
    return value


def _validate_step(value: object, where: str, *, require_kind: bool) -> _ValidStep:
    if not isinstance(value, Mapping):
        raise SparkSchemaError(f"{where}: expected mapping, got {value!r}")
    required = {"number", "start_s", "end_s", "action", "tools", "parts", "hazards"}
    if require_kind:
        required = required | {"kind"}
    extra = set(value) - (required | {"kind"})
    missing = required - set(value)
    if extra or missing:
        kind_note = "" if require_kind else " plus optional 'kind'"
        raise SparkSchemaError(
            f"{where}: expected keys {sorted(required)}{kind_note},"
            f" got {sorted(value)}"
        )
    number = value["number"]
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise SparkSchemaError(f"{where}.number: expected int >= 1, got {number!r}")
    start_s = _require_number(value["start_s"], f"{where}.start_s")
    end_s = _require_number(value["end_s"], f"{where}.end_s")
    if not (start_s >= 0 and start_s < end_s):
        raise SparkSchemaError(f"{where}: expected 0 <= start_s < end_s, got {start_s}/{end_s}")
    action = _require_text(value["action"], f"{where}.action")
    tools = _require_str_list(value["tools"], f"{where}.tools")
    parts_raw = value["parts"]
    if not isinstance(parts_raw, list) or len(parts_raw) > MAX_PARTS_PER_STEP:
        raise SparkSchemaError(
            f"{where}.parts: expected list of at most {MAX_PARTS_PER_STEP}, got {parts_raw!r}"
        )
    parts = tuple(_validate_part(item, f"{where}.parts[{i}]") for i, item in enumerate(parts_raw))
    hazards = _require_str_list(value["hazards"], f"{where}.hazards")
    kind = (
        _validate_kind(value["kind"], f"{where}.kind")
        if "kind" in value
        else DEFAULT_PHASE_KIND
    )
    return _ValidStep(
        number=number,
        start_s=start_s,
        end_s=end_s,
        action=action,
        tools=tools,
        parts=parts,
        hazards=hazards,
        kind=kind,
    )


def _validate_run_payload(payload: Mapping[str, JSONValue], *, require_kind: bool) -> _ValidRun:
    if not isinstance(payload, Mapping):
        raise SparkSchemaError(f"run payload: expected mapping, got {payload!r}")
    _require_keys(payload, {"device_guess", "confidence", "steps"}, "run")
    device_guess = _require_text(payload["device_guess"], "run.device_guess")
    confidence = _require_number(payload["confidence"], "run.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise SparkSchemaError(f"run.confidence: expected 0..1, got {confidence}")
    steps_raw = payload["steps"]
    if not isinstance(steps_raw, list) or len(steps_raw) > MAX_STEPS:
        raise SparkSchemaError(
            f"run.steps: expected list of at most {MAX_STEPS}, got {steps_raw!r}"
        )
    steps = tuple(
        _validate_step(item, f"run.steps[{i}]", require_kind=require_kind)
        for i, item in enumerate(steps_raw)
    )
    numbers = [step.number for step in steps]
    if sorted(numbers) != list(range(1, len(steps) + 1)):
        raise SparkSchemaError(f"run.steps: numbers must be exactly 1..{len(steps)}, got {numbers}")
    by_number = {step.number: step for step in steps}
    ordered = tuple(by_number[i] for i in range(1, len(steps) + 1))
    return _ValidRun(device_guess=device_guess, confidence=confidence, steps=ordered)


def validate_spark_payload(payload: Mapping[str, JSONValue]) -> _ValidRun:
    """Strict-validate one raw run payload (unknown keys rejected).

    This is the new-provider path: every step must carry a ``kind``
    from the closed set, matching the strict provider schema. Raw
    steps may be 0..12: a run that found no teardown is valid input,
    and zero *consensus* steps become a NotATeardown refusal downstream.
    The StepBundle itself always carries 1..12 steps.
    """
    return _validate_run_payload(payload, require_kind=True)


def validate_legacy_spark_payload(payload: Mapping[str, JSONValue]) -> _ValidRun:
    """Validate one pre-kind run payload (old fixtures, recorded runs).

    The explicit compatibility path for payloads stored before kinds:
    a step without ``kind`` validates as repair; a present kind must
    still be exactly one of the closed set, and every other rule is
    the strict one. Live provider responses never reach this path —
    they validate through :func:`validate_spark_payload`.
    """
    return _validate_run_payload(payload, require_kind=False)


def _consensus_device(runs: Sequence[_ValidRun]) -> str:
    counts = Counter(run.device_guess for run in runs)
    top = max(counts.values())
    for run in runs:  # ties resolve to the earliest run: deterministic
        if counts[run.device_guess] == top:
            return run.device_guess
    raise AssertionError("unreachable: runs is non-empty")


def _normalize_action(action: str) -> str:
    return " ".join(action.strip().lower().split())


def _parts_signature(parts: tuple[_ValidPart, ...]) -> tuple[str, ...]:
    return tuple(sorted(" ".join(part.phrase.strip().lower().split()) for part in parts))


def _semantic_key(step: _ValidStep) -> tuple[str, tuple[str, ...]]:
    return (_normalize_action(step.action), _parts_signature(step.parts))


def _linked(left: _ValidStep, right: _ValidStep, spread_s: float) -> bool:
    """Same-key link: windows overlap, or the gap is at most the flag line."""
    gap = max(left.start_s, right.start_s) - min(left.end_s, right.end_s)
    return gap <= spread_s


def _key_components(
    runs: Sequence[_ValidRun], spread_s: float
) -> list[list[tuple[int, _ValidStep]]]:
    """Connected components of samples linked within one semantic key only.

    Union-find over every sample; different keys are never merged, and a
    chain of pairwise links (single linkage) lands in one component even
    when its ends never meet.
    """
    samples: list[tuple[int, _ValidStep]] = [
        (index, step) for index, run in enumerate(runs) for step in run.steps
    ]
    keys = [_semantic_key(step) for _, step in samples]
    parent = list(range(len(samples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if keys[i] == keys[j] and _linked(samples[i][1], samples[j][1], spread_s):
                parent[find(i)] = find(j)
    grouped: dict[int, list[tuple[int, _ValidStep]]] = {}
    for i, sample in enumerate(samples):
        grouped.setdefault(find(i), []).append(sample)
    return list(grouped.values())


def _consensus_step(
    number: int, samples: Sequence[tuple[int, _ValidStep]], spread_s: float
) -> Step:
    starts = [step.start_s for _, step in samples]
    ends = [step.end_s for _, step in samples]
    median_start = median_time(starts)
    median_end = median_time(ends)
    spread = time_spread(starts, ends)
    # Non-time fields come from the median-start sample (ties: lowest run).
    _, chosen = min(samples, key=lambda item: (abs(item[1].start_s - median_start), item[0]))
    return Step(
        number=number,
        start_s=median_start,
        end_s=median_end,
        action=chosen.action,
        tools=chosen.tools,
        parts=tuple(PartRequest(phrase=p.phrase, position_hint=p.position_hint) for p in chosen.parts),
        hazards=chosen.hazards,
        time_spread_s=spread,
        time_needs_check=spread > spread_s,
        kind=chosen.kind,
    )


def _align_components(
    runs: Sequence[_ValidRun],
    spread_s: float,
    generic_tokens: frozenset[str],
    verb_synonyms: tuple[frozenset[str], ...],
    negation_words: frozenset[str],
    qualifiers: frozenset[str],
    part_synonyms: tuple[frozenset[str], ...],
    transition_verbs: frozenset[str],
    phases_mode: bool,
    verb_less_overlap_fraction: float,
) -> list[list[tuple[int, _ValidStep]]]:
    """Phase samples with the shared monotonic alignment (fix 2)."""
    samples: list[tuple[int, _ValidStep]] = [
        (index, step) for index, run in enumerate(runs) for step in run.steps
    ]
    windows = [step_alignment.window_of(step) for _, step in samples]
    return [
        [samples[i] for i in group]
        for group in step_alignment.cluster_windows(
            windows,
            tolerance_s=spread_s,
            generic_tokens=generic_tokens,
            verb_synonyms=verb_synonyms,
            negation_words=negation_words,
            qualifiers=qualifiers,
            part_synonyms=part_synonyms,
            transition_verbs=transition_verbs,
            phases_mode=phases_mode,
            verb_less_overlap_fraction=verb_less_overlap_fraction,
            run_of=[run for run, _ in samples],
        )
    ]


def _representative_identity(
    component: Sequence[tuple[int, _ValidStep]],
) -> tuple[str, tuple[str, ...]]:
    """Action + part evidence of the median-start sample (the fields' source)."""
    median_start = median_time([step.start_s for _, step in component])
    _, chosen = min(
        component, key=lambda item: (abs(item[1].start_s - median_start), item[0])
    )
    return (
        step_alignment.normalize_action(chosen.action),
        tuple(
            sorted(
                step_alignment.part_tokens(
                    part.phrase for part in chosen.parts
                )
            )
        ),
    )


def cluster_time_boundaries(
    boundaries: Sequence[tuple[float, int]],
    *,
    tolerance_s: float,
    min_support: int,
) -> list[tuple[float, float, int]]:
    """Cluster phase-boundary times across runs (time consensus).

    ``boundaries`` holds ``(time, run_index)`` pairs, one per run
    phase edge. Times sort ascending and cluster greedily: each time
    joins the open cluster while the cluster's whole span (latest
    minus earliest) stays within ``tolerance_s`` and the run is not
    already in the cluster, else it opens a new one -- no chaining
    across a wider gap and at most one boundary per run per cluster,
    so a short phase never collapses into its neighbour. A cluster
    is kept only with edges from at least ``min_support`` distinct
    runs (one-run-only boundaries drop); kept time is the median,
    spread the max-minus-min, support the distinct-run count.
    Returned sorted by time. Deterministic.
    """
    if (
        isinstance(tolerance_s, bool)
        or not isinstance(tolerance_s, (int, float))
        or not math.isfinite(tolerance_s)
        or not tolerance_s > 0
    ):
        raise ValueError(
            "tolerance_s must be a finite number > 0,"
            f" got {tolerance_s!r}"
        )
    if (
        isinstance(min_support, bool)
        or not isinstance(min_support, int)
        or min_support < 1
    ):
        raise ValueError(f"min_support must be an int >= 1, got {min_support!r}")
    for time, run in boundaries:
        if (
            isinstance(time, bool)
            or not isinstance(time, (int, float))
            or not math.isfinite(time)
        ):
            raise ValueError(f"boundary time must be finite, got {time!r}")
        if isinstance(run, bool) or not isinstance(run, int) or run < 0:
            raise ValueError(f"boundary run must be an int >= 0, got {run!r}")
    ordered = sorted(
        ((float(time), run) for time, run in boundaries),
        key=lambda item: (item[0], item[1]),
    )
    clusters: list[list[tuple[float, int]]] = []
    open_runs: set[int] = set()
    for time, run in ordered:
        if (
            clusters
            and time - clusters[-1][0][0] <= tolerance_s
            and run not in open_runs
        ):
            clusters[-1].append((time, run))
            open_runs.add(run)
        else:
            clusters.append([(time, run)])
            open_runs = {run}
    kept: list[tuple[float, float, int]] = []
    for cluster in clusters:
        support = len({run for _, run in cluster})
        if support < min_support:
            continue
        times = [time for time, _ in cluster]
        kept.append((median_time(times), max(times) - min(times), support))
    kept.sort(key=lambda item: (item[0], item[1]))
    return kept


def _overlap_s(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    """Nonnegative overlap of two windows (0 when apart or touching)."""
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _is_contiguous_partition(steps: Sequence[_ValidStep]) -> bool:
    """True when number-ordered steps tile their span (time consensus).

    Starts strictly increase and each phase end equals the next
    phase start exactly; anything else (reordering, gap, overlap)
    is not a partition and the run is excluded from time consensus.
    """
    for prev, nxt in zip(steps, steps[1:]):
        if not nxt.start_s > prev.start_s:
            return False
        if nxt.start_s != prev.end_s:
            return False
    return True


def _within_chunk_extent(
    steps: Sequence[_ValidStep], chunk_extent: ChunkExtent | None
) -> bool:
    """True when the run covers the known chunk extent within tolerance.

    Without a known extent every run passes (the observed span
    rules); with one, the run must start within tolerance of 0 and
    end within tolerance of the duration. A run that overshoots,
    falls short, or starts late is excluded, never clamped or
    stretched: tiling ``[0, duration]`` by construction must not
    fabricate coverage the runs never described, so a missing phase
    still fails closed. All bounds are inclusive: starting exactly
    at the tolerance, or ending exactly at ``duration`` plus or
    minus the tolerance, still passes.

    When the tolerance is greater than or equal to the duration it
    swallows the whole clip (short last chunks under a wide
    tolerance), so slack is meaningless: disjoint fragments would
    pass and stretch to ``[0, duration]``. That regime requires
    literal coverage of both endpoints instead, or the run is
    excluded; overshoot past the duration plus tolerance still
    excludes.
    """
    if chunk_extent is None:
        return True
    start = min(step.start_s for step in steps)
    end = max(step.end_s for step in steps)
    if chunk_extent.tolerance_s >= chunk_extent.duration_s:
        return (
            start <= 0.0
            and chunk_extent.duration_s
            <= end
            <= chunk_extent.duration_s + chunk_extent.tolerance_s
        )
    return (
        start <= chunk_extent.tolerance_s
        and chunk_extent.duration_s - chunk_extent.tolerance_s
        <= end
        <= chunk_extent.duration_s + chunk_extent.tolerance_s
    )


def _time_consensus_bundle(
    runs: Sequence[_ValidRun],
    config: SparkStageConfig,
    device_guess: str,
    confidence: float,
    majority: int,
    chunk_extent: ChunkExtent | None = None,
) -> StepBundle:
    """Contiguous-phases consensus by time (wording never drops a segment).

    Every non-empty run that is an ordered contiguous partition
    contributes its internal phase edges (a reordered, gapped, or
    overlapping run is excluded, never normalized); a run that
    misses the known chunk extent beyond tolerance — overshooting,
    short, or late-starting — is excluded the same way, so one
    over-long run cannot stretch the tail and fragment runs cannot
    gain fabricated coverage. Edges cluster within the boundary
    tolerance (default: the flag line) and survive with
    majority-run support, so the kept medians plus the span edges
    tile the chunk with no gaps by construction. The span edges are
    the KNOWN chunk extent ``[0, duration]`` when the caller passes
    it, else the observed span over the usable runs. Each segment's
    label comes from the
    run phase with the largest overlap, preferring the wording most
    runs agree on via :func:`step_alignment.same_action`; wording
    disagreement only flags the kept segment. Each segment's kind is
    the kind a majority of usable runs voted for it (each run votes
    its max-overlap phase's kind); with no majority the label
    winner's kind stands unless it is outro, which needs a majority
    outright. Each internal boundary carries a record with its run
    support (cut evidence and snap outcome are filled in by the later
    refine pass). StepsDisagreement when fewer than a majority of
    runs remain usable, or when more than 12 segments survive.
    """
    usable = [
        (index, run)
        for index, run in enumerate(runs)
        if run.steps
        and _is_contiguous_partition(run.steps)
        and _within_chunk_extent(run.steps, chunk_extent)
    ]
    if len(usable) < majority:
        raise StepsDisagreement(
            f"only {len(usable)} of {len(runs)} runs are usable partitions,"
            f" need {majority}"
        )
    tolerance = config.spread_s if config.boundary_tolerance_s is None else config.boundary_tolerance_s
    if chunk_extent is None:
        chunk_start = min(step.start_s for _, run in usable for step in run.steps)
        chunk_end = max(step.end_s for _, run in usable for step in run.steps)
    else:
        chunk_start = 0.0
        chunk_end = chunk_extent.duration_s
    clustered = [
        (time, spread, support)
        for time, spread, support in cluster_time_boundaries(
            [
                (step.end_s, index)
                for index, run in usable
                for step in run.steps[:-1]
            ],
            tolerance_s=tolerance,
            min_support=majority,
        )
        if chunk_start < time < chunk_end
    ]
    kept = [(time, spread) for time, spread, _ in clustered]
    supports = [support for _, _, support in clustered]
    edges: list[tuple[float, float]] = (
        [(chunk_start, 0.0)] + kept + [(chunk_end, 0.0)]
    )
    edge_starts = [min(step.start_s for step in run.steps) for _, run in usable]
    edge_ends = [max(step.end_s for step in run.steps) for _, run in usable]
    edges[0] = (chunk_start, max(edge_starts) - min(edge_starts))
    edges[-1] = (chunk_end, max(edge_ends) - min(edge_ends))
    if len(edges) - 1 > MAX_STEPS:
        raise StepsDisagreement(
            f"{len(edges) - 1} consensus segments exceed"
            f" the {MAX_STEPS}-step limit"
        )
    part_tokens_of = {
        index: [
            step_alignment.part_tokens(part.phrase for part in step.parts)
            for step in run.steps
        ]
        for index, run in usable
    }
    steps: list[Step] = []
    for number, ((start, left_spread), (end, right_spread)) in enumerate(
        zip(edges, edges[1:]), start=1
    ):
        candidates: list[tuple[int, int, _ValidStep, float]] = []
        for index, run in usable:
            pick = min(
                range(len(run.steps)),
                key=lambda i: (
                    -_overlap_s(start, end, run.steps[i].start_s, run.steps[i].end_s),
                    run.steps[i].start_s,
                    i,
                ),
            )
            candidates.append(
                (
                    index,
                    pick,
                    run.steps[pick],
                    _overlap_s(start, end, run.steps[pick].start_s, run.steps[pick].end_s),
                )
            )
        width = len(candidates)
        match = [[True] * width for _ in range(width)]
        for i in range(width):
            for j in range(i + 1, width):
                run_i, step_index_i, step_i, _ = candidates[i]
                run_j, step_index_j, step_j, _ = candidates[j]
                linked = step_alignment.same_action(
                    step_i.action, step_j.action,
                    part_tokens_of[run_i][step_index_i],
                    part_tokens_of[run_j][step_index_j],
                    generic_tokens=config.generic_tokens,
                    verb_synonyms=config.verb_synonyms,
                    negation_words=config.negation_words,
                    qualifiers=config.qualifiers,
                    part_synonyms=config.part_synonyms,
                    transition_verbs=config.transition_verbs,
                    phases_mode=True,
                )
                match[i][j] = match[j][i] = linked
        # Complete-link: a candidate joins the earliest group whose
        # every member it matches, so no bridge merges endpoints that
        # disagree directly.
        groups: list[list[int]] = []
        for i in range(width):
            placed = False
            for group in groups:
                if all(match[i][member] for member in group):
                    group.append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])
        winner = max(
            groups,
            key=lambda members: (
                len(members),
                sum(candidates[i][3] for i in members),
                -min(candidates[i][0] for i in members),
            ),
        )
        _, _, chosen, _ = max(
            (candidates[i] for i in winner),
            key=lambda item: (item[3], -item[0]),
        )
        # Segment kind: each usable run votes the kind of its
        # max-overlap phase. Outro needs a run majority outright; any
        # other kind wins on a majority too; with no majority the
        # label winner's kind stands unless it is outro (a lone outro
        # vote never drops a phase -- repair keeps it).
        votes = Counter(step.kind for _, _, step, _ in candidates)
        majority_kinds = [
            kind for kind, count in votes.items() if count >= majority
        ]
        if OUTRO_PHASE_KIND in majority_kinds:
            segment_kind = OUTRO_PHASE_KIND
        elif majority_kinds:
            # At most one: majority > runs / 2, so two majorities
            # would need more votes than usable runs cast.
            (segment_kind,) = majority_kinds
        elif chosen.kind != OUTRO_PHASE_KIND:
            segment_kind = chosen.kind
        else:
            segment_kind = DEFAULT_PHASE_KIND
        spread = max(left_spread, right_spread)
        incompatible = any(
            not match[i][j]
            for i in range(width)
            for j in range(i + 1, width)
        )
        steps.append(
            Step(
                number=number,
                start_s=start,
                end_s=end,
                action=chosen.action,
                tools=chosen.tools,
                parts=tuple(
                    PartRequest(phrase=p.phrase, position_hint=p.position_hint)
                    for p in chosen.parts
                ),
                hazards=chosen.hazards,
                time_spread_s=spread,
                time_needs_check=spread > config.spread_s or incompatible,
                kind=segment_kind,
            )
        )
    total_runs = len(runs)
    boundaries = tuple(
        BoundaryInfo(
            time_s=time,
            support_runs=support,
            total_runs=total_runs,
            nearest_cut_s=None,
            cut_distance_s=None,
            snapped=False,
            snap_delta_s=0.0,
            needs_check=support < total_runs,
        )
        for (time, _), support in zip(kept, supports)
    )
    return StepBundle(
        device_guess=device_guess,
        confidence=confidence,
        steps=tuple(steps),
        source_runs=config.runs,
        timestamps_approximate=True,
        boundaries=boundaries,
    )


def consensus_bundle(
    runs: Sequence[_ValidRun],
    config: SparkStageConfig = DEFAULT_SPARK_CONFIG,
    *,
    chunk_extent: ChunkExtent | None = None,
) -> StepBundle:
    """Group samples by alignment, then take the per-group median consensus.

    In contiguous phases mode (``span_mode="contiguous"`` with
    ``alignment_mode="align"``) consensus matches runs by TIME (see
    :func:`cluster_time_boundaries`): kept boundaries tile the chunk
    with no gaps by construction, wording only chooses each
    segment's label and flags disagreement. The tiled span is the
    known chunk extent when ``chunk_extent`` is passed (a run that
    misses it beyond tolerance is excluded like any invalid run),
    else the observed span over the usable runs; other modes ignore
    the extent and keep model spans. Legacy/sparse mode keeps
    the wording-grouped consensus: default ``align`` mode phases
    with the shared monotonic alignment (:mod:`app.step_alignment`):
    paraphrases merge, disjoint parts veto, one-to-many segmentation
    splits align to one phase, and no bridge merges endpoints that do
    not link directly. ``exact`` mode keeps the old semantic-key
    (exact action plus exact part tuple) grouping as a configured
    rollback. Either wording-grouped way, a component is a step
    candidate only with samples from a majority of distinct runs.
    Candidates order by window start. Two candidates that overlap by
    more than half of the shorter window are both kept and flagged
    (honest disagreement); StepsDisagreement when no candidate holds
    a majority of runs, or when more than 12 candidates survive (the
    bundle never exceeds 12 steps). Tools, hints and hazards follow
    the median-start sample (wording-grouped) or the winning overlap
    sample (time consensus).
    """
    if len(runs) != config.runs:
        raise ValueError(f"consensus needs exactly {config.runs} runs, got {len(runs)}")
    if chunk_extent is not None and not isinstance(chunk_extent, ChunkExtent):
        raise ValueError(
            "chunk_extent must be None or a ChunkExtent,"
            f" got {type(chunk_extent).__name__}"
        )
    if all(len(run.steps) == 0 for run in runs):
        raise NotATeardown(StepsRefusal(reason="zero_steps"))
    device_guess = _consensus_device(runs)
    if device_guess.strip().lower() == "unknown":
        raise NotATeardown(StepsRefusal(reason="unknown_device"))
    confidence = float(statistics.median([run.confidence for run in runs]))
    if confidence < SPARK_CONFIDENCE_CUTOFF:
        raise NotATeardown(StepsRefusal(reason="low_confidence"))
    majority = step_alignment.majority_of(config.runs)
    if config.span_mode == "contiguous" and config.alignment_mode == "align":
        return _time_consensus_bundle(
            runs, config, device_guess, confidence, majority, chunk_extent
        )
    if config.alignment_mode == "exact":
        components = _key_components(runs, config.spread_s)
        sort_keys = [_semantic_key(comp[0][1]) for comp in components]
    else:
        components = _align_components(
            runs, config.spread_s, config.generic_tokens, config.verb_synonyms,
            config.negation_words, config.qualifiers, config.part_synonyms,
            config.transition_verbs, config.span_mode == "contiguous",
            config.verb_less_overlap_fraction,
        )
        sort_keys = [_representative_identity(comp) for comp in components]
    keyed = [
        (sort_key, comp)
        for sort_key, comp in zip(sort_keys, components)
        if len({run for run, _ in comp}) >= majority
    ]
    if not keyed:
        raise StepsDisagreement(
            f"no step has samples from {majority} runs across {len(components)} group(s)"
        )
    if len(keyed) > MAX_STEPS:
        raise StepsDisagreement(
            f"{len(keyed)} step candidates exceed the {MAX_STEPS}-step limit"
        )
    keyed.sort(
        key=lambda item: (
            median_time([step.start_s for _, step in item[1]]),
            median_time([step.end_s for _, step in item[1]]),
            item[0],
        )
    )
    drafts = [
        _consensus_step(position, comp, config.spread_s)
        for position, (_, comp) in enumerate(keyed, start=1)
    ]
    flagged: set[int] = set()
    for i in range(len(drafts)):
        for j in range(i + 1, len(drafts)):
            if keyed[i][0] != keyed[j][0] and step_alignment.windows_disagree(
                drafts[i].start_s, drafts[i].end_s, drafts[j].start_s, drafts[j].end_s
            ):
                flagged.add(i)
                flagged.add(j)
    steps = [
        replace(step, time_needs_check=True) if index in flagged else step
        for index, step in enumerate(drafts)
    ]
    return StepBundle(
        device_guess=device_guess,
        confidence=confidence,
        steps=tuple(steps),
        source_runs=config.runs,
        timestamps_approximate=True,
    )


def sam_calls_authorized(outcome: StepBundle | StepsRefusal) -> int:
    """SAM calls the outcome authorizes downstream: 0 for any refusal."""
    if isinstance(outcome, StepsRefusal):
        return 0
    return SAM_MAX_CALLS_PER_JOB


def _failed_actual(result: object) -> Decimal:
    cost = getattr(result, "cost_usd", None)
    if isinstance(cost, Decimal) and cost.is_finite() and cost >= 0:
        return cost
    return Decimal("0")


def _require_valid_run_result(result: object) -> SparkRunResult:
    if not isinstance(result, SparkRunResult):
        raise SparkClientError(f"client must return SparkRunResult, got {type(result).__name__}")
    if not isinstance(result.payload, Mapping):
        raise SparkClientError("run result payload must be a mapping")
    if (
        not isinstance(result.cost_usd, Decimal)
        or not result.cost_usd.is_finite()
        or result.cost_usd < 0
    ):
        raise SparkClientError(f"run result cost must be finite Decimal >= 0, got {result.cost_usd!r}")
    if (
        not isinstance(result.latency_ms, int)
        or isinstance(result.latency_ms, bool)
        or result.latency_ms < 0
    ):
        raise SparkClientError(f"run result latency must be int >= 0, got {result.latency_ms!r}")
    return result


def _require_video_file_id(video_file_id: str) -> str:
    if not isinstance(video_file_id, str) or not video_file_id.strip():
        raise ValueError("video_file_id must be a non-empty string")
    return video_file_id


def _require_scene_cuts(
    scene_cuts: Sequence[float] | None,
) -> tuple[float, ...] | None:
    """Validate the cuts listed in the prompt (None when unlisted)."""
    if scene_cuts is None:
        return None
    if not isinstance(scene_cuts, (tuple, list)):
        raise ValueError(
            "scene_cuts must be None or a tuple/list of times,"
            f" got {type(scene_cuts).__name__}"
        )
    checked: list[float] = []
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
        checked.append(float(cut))
    return tuple(checked)


def _extract_with_raw(
    ctx: JobContext,
    video_file_id: str,
    client: SparkClient,
    config: SparkStageConfig,
    *,
    chunk_extent: ChunkExtent | None = None,
    scene_cuts: Sequence[float] | None = None,
    clocked: bool = False,
    on_raw: Callable[[int, Mapping[str, JSONValue]], None] | None = None,
    retry_call: Callable[[str, float | None, tuple[float, ...] | None, bool], SparkRunResult] | None = None,
    retryable_failures: frozenset[str] = frozenset(),
) -> tuple[StepBundle, list[Mapping[str, JSONValue]]]:
    _require_video_file_id(video_file_id)
    prompt_cuts = _require_scene_cuts(scene_cuts)
    if not isinstance(clocked, bool):
        raise ValueError(f"clocked must be a bool, got {clocked!r}")
    # Resolve the readback-module method before any reservation: a client without
    # it fails before a cent is held.
    run = client.run_steps  # type: ignore[attr-defined]
    if chunk_extent is not None and not _uses_chunk_extent(config):
        chunk_extent = None
    clip_duration_s = chunk_extent.duration_s if chunk_extent is not None else None
    raws: list[Mapping[str, JSONValue]] = []
    runs: list[_ValidRun] = []
    retried = False
    for share in config.run_shares:
        hold = ctx.reserve(RESERVE_KIND, share)  # refusal propagates: no invoke
        try:
            if (
                clip_duration_s is None
                and prompt_cuts is None
                and not clocked
            ):
                # Legacy/sparse and exact modes keep the old call: the
                # duration-less base prompt, byte for byte.
                result = run(video_file_id)
            else:
                result = run(
                    video_file_id,
                    clip_duration_s=clip_duration_s,
                    scene_cuts=prompt_cuts,
                    clocked=clocked,
                )
        except Exception:
            hold.settle(Decimal("0"), "failed")
            raise
        except BaseException:
            # A cancel may land after the request bytes flew: settle the
            # full share as failed, then re-raise the interrupt untouched.
            try:
                hold.settle(share, "failed")
            except BaseException:
                pass
            raise
        try:
            failure = getattr(result, "failure_kind", None)
            if failure is not None:
                raise SparkServiceError(failure) if failure in ("timeout", "transport", "http_5xx", "empty") else SparkClientError(failure)
            checked = _require_valid_run_result(result)
        except SparkServiceError as error:
            hold.settle(_failed_actual(result), "failed")
            if retry_call is None or retried or error.kind not in retryable_failures:
                raise
            retried = True
            retry_result = retry_call(
                video_file_id, clip_duration_s, prompt_cuts, clocked
            )
            retry_failure = getattr(retry_result, "failure_kind", None)
            if retry_failure is not None:
                if retry_failure in ("timeout", "transport", "http_5xx", "empty"):
                    raise SparkServiceError(retry_failure)
                raise SparkClientError(retry_failure)
            checked = _require_valid_run_result(retry_result)
        except SparkClientError:
            hold.settle(_failed_actual(result), "failed")
            raise
        else:
            hold.settle(checked.cost_usd, "succeeded")
        if on_raw is not None:
            # Before validation and consensus: a paid raw survives even
            # when consensus raises StepsDisagreement downstream.
            on_raw(len(raws), checked.payload)
        raws.append(checked.payload)
        runs.append(validate_spark_payload(checked.payload))
    return consensus_bundle(runs, config, chunk_extent=chunk_extent), raws


def extract_steps(
    ctx: JobContext,
    video_file_id: str,
    *,
    client: SparkClient,
    chunk_extent: ChunkExtent | None = None,
) -> StepBundle:
    """Run Spark once per configured run live and return the median consensus bundle.

    With ``chunk_extent`` in time-consensus mode each run's prompt
    states the clip's exact duration and consensus tiles the known
    extent (runs missing it beyond tolerance excluded); otherwise
    the legacy duration-less prompt and the observed span apply.
    """
    return _extract_with_raw(
        ctx, video_file_id, client, spark_config_of(client),
        chunk_extent=chunk_extent,
    )[0]


def _digest_cuts(scene_cuts: Sequence[float] | None) -> str:
    """Digest part for the cuts listed in the prompt (``none`` when unlisted)."""
    if scene_cuts is None:
        return "none"
    if not isinstance(scene_cuts, (tuple, list)):
        raise ValueError(
            "scene_cuts must be None or a tuple/list of times,"
            f" got {type(scene_cuts).__name__}"
        )
    parts: list[str] = []
    for cut in scene_cuts:
        if (
            isinstance(cut, bool)
            or not isinstance(cut, (int, float))
            or not math.isfinite(cut)
            or not cut >= 0
        ):
            raise ValueError(f"scene cut times must be finite >= 0, got {cut!r}")
        parts.append(repr(float(cut)))
    return ",".join(parts)


def _digest_clocked(clocked: bool) -> str:
    """Digest part for the burned-in-clock prompt variant."""
    if not isinstance(clocked, bool):
        raise ValueError(f"clocked must be a bool, got {clocked!r}")
    return "clocked" if clocked else "plain"


def consensus_digest(
    job_id: str,
    video_file_id: str,
    config: SparkStageConfig = DEFAULT_SPARK_CONFIG,
    chunk_extent: ChunkExtent | None = None,
    scene_cuts: Sequence[float] | None = None,
    clocked: bool = False,
) -> str:
    """Cache digest for consensus (job/input/prompt/schema/model/runs/spread/span/mode/tokens/verbs/transitions/verb-less-fraction/negation/qualifiers/parts/boundary-tolerance/chunk-extent/cuts/clock).

    The extent binds the digest only in time-consensus mode; without
    an extent in effect the digest keeps the pre-extent component
    shape exactly (no extent components), so legacy/sparse and exact
    modes hash the same path with or without an extent passed. The
    listed cuts bind only when listed and the clock only when burned,
    so an unlisted, unclocked call keeps the same shape.
    """
    if chunk_extent is not None and not _uses_chunk_extent(config):
        chunk_extent = None
    prompt_cuts = _require_scene_cuts(scene_cuts)
    if not isinstance(clocked, bool):
        raise ValueError(f"clocked must be a bool, got {clocked!r}")
    parts = [
        stage_cache.STAGE_SPARK_CONSENSUS,
        job_id,
        video_file_id,
        SPARK_PROMPT_VERSION,
        SPARK_SCHEMA_VERSION,
        config.model_id,
        str(config.runs),
        repr(config.spread_s),
        config.span_mode,
        config.alignment_mode,
        ",".join(sorted(config.generic_tokens)),
        ";".join(",".join(sorted(group)) for group in config.verb_synonyms),
        ",".join(sorted(config.transition_verbs)),
        repr(config.verb_less_overlap_fraction),
        ",".join(sorted(config.negation_words)),
        ",".join(sorted(config.qualifiers)),
        ";".join(",".join(sorted(group)) for group in config.part_synonyms),
        repr(config.boundary_tolerance_s),
    ]
    if chunk_extent is not None:
        parts.append(repr(chunk_extent.duration_s))
        parts.append(repr(chunk_extent.tolerance_s))
    if prompt_cuts is not None:
        parts.append(_digest_cuts(prompt_cuts))
    if clocked:
        parts.append(_digest_clocked(clocked))
    return stage_cache.digest_for(*parts)


def run_digest(
    job_id: str,
    video_file_id: str,
    index: int,
    config: SparkStageConfig = DEFAULT_SPARK_CONFIG,
    chunk_extent: ChunkExtent | None = None,
    scene_cuts: Sequence[float] | None = None,
    clocked: bool = False,
) -> str:
    """Cache digest for one raw Spark run (index 0..runs-1, prompt duration/cuts/clock bound).

    Like :func:`consensus_digest`, the duration binds only in
    time-consensus mode; otherwise the pre-extent shape applies. The
    listed cuts bind only when listed and the clock only when burned.
    """
    if chunk_extent is not None and not _uses_chunk_extent(config):
        chunk_extent = None
    prompt_cuts = _require_scene_cuts(scene_cuts)
    if not isinstance(clocked, bool):
        raise ValueError(f"clocked must be a bool, got {clocked!r}")
    parts = [
        stage_cache.STAGE_SPARK_RUN,
        job_id,
        video_file_id,
        SPARK_PROMPT_VERSION,
        SPARK_SCHEMA_VERSION,
        config.model_id,
        str(config.runs),
        repr(config.spread_s),
        str(index),
    ]
    if chunk_extent is not None:
        parts.append(repr(chunk_extent.duration_s))
    if prompt_cuts is not None:
        parts.append(_digest_cuts(prompt_cuts))
    if clocked:
        parts.append(_digest_clocked(clocked))
    return stage_cache.digest_for(*parts)


def _boundary_to_json(record: BoundaryInfo) -> JSONValue:
    """Serialize one boundary record to cacheable JSON."""
    return {
        "time_s": record.time_s,
        "support_runs": record.support_runs,
        "total_runs": record.total_runs,
        "nearest_cut_s": record.nearest_cut_s,
        "cut_distance_s": record.cut_distance_s,
        "snapped": record.snapped,
        "snap_delta_s": record.snap_delta_s,
        "needs_check": record.needs_check,
    }


def _boundary_from_json(raw: object, where: str) -> BoundaryInfo:
    """Rebuild one boundary record (strict; anything misshaped raises)."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected mapping, got {raw!r}")
    _require_keys(
        raw,
        {
            "time_s",
            "support_runs",
            "total_runs",
            "nearest_cut_s",
            "cut_distance_s",
            "snapped",
            "snap_delta_s",
            "needs_check",
        },
        where,
    )
    time_s = _require_number(raw["time_s"], f"{where}.time_s")
    if time_s < 0:
        raise SparkSchemaError(f"{where}.time_s: expected >= 0, got {time_s}")
    support = raw["support_runs"]
    total = raw["total_runs"]
    if (
        not isinstance(support, int)
        or isinstance(support, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
    ):
        raise SparkSchemaError(
            f"{where}: support_runs/total_runs must be ints,"
            f" got {support!r}/{total!r}"
        )
    if not 1 <= support <= total:
        raise SparkSchemaError(
            f"{where}: expected 1 <= support_runs <= total_runs,"
            f" got {support}/{total}"
        )
    nearest = raw["nearest_cut_s"]
    distance = raw["cut_distance_s"]
    if nearest is None or distance is None:
        if nearest is not None or distance is not None:
            raise SparkSchemaError(
                f"{where}: nearest_cut_s and cut_distance_s must both"
                " be present or both be null"
            )
    else:
        nearest = _require_number(nearest, f"{where}.nearest_cut_s")
        distance = _require_number(distance, f"{where}.cut_distance_s")
        if nearest < 0 or distance < 0:
            raise SparkSchemaError(
                f"{where}: cut times and distances must be >= 0"
            )
    snapped = raw["snapped"]
    if not isinstance(snapped, bool):
        raise SparkSchemaError(
            f"{where}.snapped: expected bool, got {snapped!r}"
        )
    delta = _require_number(raw["snap_delta_s"], f"{where}.snap_delta_s")
    if snapped == (delta == 0.0):
        raise SparkSchemaError(
            f"{where}: snapped must be true exactly when snap_delta_s"
            f" is nonzero, got {snapped}/{delta}"
        )
    flag = raw["needs_check"]
    if not isinstance(flag, bool):
        raise SparkSchemaError(
            f"{where}.needs_check: expected bool, got {flag!r}"
        )
    return BoundaryInfo(
        time_s=time_s,
        support_runs=support,
        total_runs=total,
        nearest_cut_s=nearest,
        cut_distance_s=distance,
        snapped=snapped,
        snap_delta_s=delta,
        needs_check=flag,
    )


def _excluded_span_to_json(span: ExcludedSpan) -> JSONValue:
    """Serialize one excluded span to cacheable JSON."""
    return {"start_s": span.start_s, "end_s": span.end_s, "kind": span.kind}


def _excluded_span_from_json(raw: object, where: str) -> ExcludedSpan:
    """Rebuild one excluded span (strict; anything misshaped raises)."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected mapping, got {raw!r}")
    _require_keys(raw, {"start_s", "end_s", "kind"}, where)
    start_s = _require_number(raw["start_s"], f"{where}.start_s")
    end_s = _require_number(raw["end_s"], f"{where}.end_s")
    if not (start_s >= 0 and start_s < end_s):
        raise SparkSchemaError(
            f"{where}: expected 0 <= start_s < end_s, got {start_s}/{end_s}"
        )
    kind = _validate_kind(raw["kind"], f"{where}.kind")
    return ExcludedSpan(start_s=start_s, end_s=end_s, kind=kind)


def bundle_to_json(bundle: StepBundle) -> JSONValue:
    """Serialize a consensus bundle to cacheable JSON."""
    return {
        "device_guess": bundle.device_guess,
        "confidence": bundle.confidence,
        "source_runs": bundle.source_runs,
        "timestamps_approximate": bundle.timestamps_approximate,
        "boundaries": [
            _boundary_to_json(record) for record in bundle.boundaries
        ],
        "excluded_spans": [
            _excluded_span_to_json(span) for span in bundle.excluded_spans
        ],
        "steps": [
            {
                "number": step.number,
                "start_s": step.start_s,
                "end_s": step.end_s,
                "action": step.action,
                "tools": list(step.tools),
                "parts": [
                    {"phrase": part.phrase, "position_hint": part.position_hint}
                    for part in step.parts
                ],
                "hazards": list(step.hazards),
                "time_spread_s": step.time_spread_s,
                "time_needs_check": step.time_needs_check,
                "kind": step.kind,
            }
            for step in bundle.steps
        ],
    }


def bundle_from_json(
    value: JSONValue, config: SparkStageConfig = DEFAULT_SPARK_CONFIG
) -> StepBundle:
    """Rebuild a consensus bundle from cache (strict; garbage raises).

    Same bounds as the strict schema: exact keys, finite confidence in
    0..1, 1..12 steps numbered exactly 1..n, 0 <= start < end, non-empty
    action, at most 3 parts, finite spread >= 0, a real bool flag that
    is never false when the spread exceeds the active flag line (true
    also marks honest window disagreement), no two identical steps, a
    closed-set kind per step, and one boundary record per internal
    boundary (wording-grouped bundles carry none), plus the recorded
    excluded spans.
    """
    if not isinstance(value, Mapping):
        raise ValueError("cached consensus must be a mapping")
    _require_keys(
        value,
        {
            "device_guess",
            "confidence",
            "source_runs",
            "timestamps_approximate",
            "boundaries",
            "excluded_spans",
            "steps",
        },
        "cached consensus",
    )
    device_guess = _require_text(value["device_guess"], "cached consensus.device_guess")
    confidence = _require_number(value["confidence"], "cached consensus.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise SparkSchemaError(f"cached consensus.confidence: expected 0..1, got {confidence}")
    source_runs = value["source_runs"]
    if (
        not isinstance(source_runs, int)
        or isinstance(source_runs, bool)
        or source_runs != config.runs
    ):
        raise ValueError(
            f"cached consensus must carry {config.runs} runs and approximate times"
        )
    if value["timestamps_approximate"] is not True:
        raise ValueError(
            f"cached consensus must carry {config.runs} runs and approximate times"
        )
    steps_raw = value["steps"]
    if not isinstance(steps_raw, list) or not 1 <= len(steps_raw) <= MAX_STEPS:
        raise ValueError(f"cached consensus.steps: expected 1..{MAX_STEPS} steps")
    seen: set[tuple[float, float, str, tuple[str, ...], tuple[PartRequest, ...], tuple[str, ...]]] = set()
    steps: list[Step] = []
    for index, raw in enumerate(steps_raw):
        where = f"cached step {index}"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: expected mapping, got {raw!r}")
        _require_keys(
            raw,
            {
                "number", "start_s", "end_s", "action", "tools", "parts",
                "hazards", "time_spread_s", "time_needs_check", "kind",
            },
            where,
        )
        number = raw["number"]
        if not isinstance(number, int) or isinstance(number, bool):
            raise SparkSchemaError(f"{where}.number: expected int, got {number!r}")
        start_s = _require_number(raw["start_s"], f"{where}.start_s")
        end_s = _require_number(raw["end_s"], f"{where}.end_s")
        if not (start_s >= 0 and start_s < end_s):
            raise SparkSchemaError(
                f"{where}: expected 0 <= start_s < end_s, got {start_s}/{end_s}"
            )
        action = _require_text(raw["action"], f"{where}.action")
        tools = _require_str_list(raw["tools"], f"{where}.tools")
        parts_raw = raw["parts"]
        if not isinstance(parts_raw, list) or len(parts_raw) > MAX_PARTS_PER_STEP:
            raise SparkSchemaError(
                f"{where}.parts: expected list of at most {MAX_PARTS_PER_STEP}, got {parts_raw!r}"
            )
        parts = tuple(
            PartRequest(phrase=part.phrase, position_hint=part.position_hint)
            for part in (
                _validate_part(item, f"{where}.parts[{i}]")
                for i, item in enumerate(parts_raw)
            )
        )
        hazards = _require_str_list(raw["hazards"], f"{where}.hazards")
        spread = _require_number(raw["time_spread_s"], f"{where}.time_spread_s")
        if spread < 0:
            raise SparkSchemaError(f"{where}.time_spread_s: expected >= 0, got {spread}")
        flag = raw["time_needs_check"]
        if not isinstance(flag, bool):
            raise SparkSchemaError(f"{where}.time_needs_check: expected bool, got {flag!r}")
        if spread > config.spread_s and not flag:
            raise SparkSchemaError(
                f"{where}: time_needs_check=False contradicts time_spread_s={spread}"
            )
        signature = (start_s, end_s, action, tools, parts, hazards)
        if signature in seen:
            raise SparkSchemaError(f"{where}: duplicate step")
        seen.add(signature)
        kind = _validate_kind(raw["kind"], f"{where}.kind")
        steps.append(
            Step(
                number=number,
                start_s=start_s,
                end_s=end_s,
                action=action,
                tools=tools,
                parts=parts,
                hazards=hazards,
                time_spread_s=spread,
                time_needs_check=flag,
                kind=kind,
            )
        )
    numbers = [step.number for step in steps]
    if sorted(numbers) != list(range(1, len(steps) + 1)):
        raise SparkSchemaError(
            f"cached consensus.steps: numbers must be exactly 1..{len(steps)}, got {numbers}"
        )
    boundaries_raw = value["boundaries"]
    if not isinstance(boundaries_raw, list):
        raise ValueError("cached consensus.boundaries: expected a list")
    boundaries = tuple(
        _boundary_from_json(raw, f"cached boundary {index}")
        for index, raw in enumerate(boundaries_raw)
    )
    if boundaries and len(boundaries) != len(steps) - 1:
        raise SparkSchemaError(
            "cached consensus.boundaries: expected one record per"
            f" internal boundary, got {len(boundaries)} for"
            f" {len(steps)} steps"
        )
    excluded_raw = value["excluded_spans"]
    if not isinstance(excluded_raw, list):
        raise ValueError("cached consensus.excluded_spans: expected a list")
    excluded = tuple(
        _excluded_span_from_json(raw, f"cached excluded span {index}")
        for index, raw in enumerate(excluded_raw)
    )
    return StepBundle(
        device_guess=device_guess,
        confidence=confidence,
        steps=tuple(steps),
        source_runs=config.runs,
        timestamps_approximate=True,
        boundaries=boundaries,
        excluded_spans=excluded,
    )


def extract_steps_cached(
    ctx: JobContext,
    video_file_id: str,
    *,
    client: SparkClient,
    chunk_extent: ChunkExtent | None = None,
    scene_cuts: Sequence[float] | None = None,
    clocked: bool = False,
    retry_call: Callable[[str, float | None, tuple[float, ...] | None, bool], SparkRunResult] | None = None,
    retryable_failures: frozenset[str] = frozenset(),
) -> CachedExtraction:
    """Cached extraction: a hit replays with zero context/client calls.

    Each paid raw is persisted immediately after its call succeeds, so
    a StepsDisagreement still leaves all three raws for offline
    diagnosis; only successful consensus is cached as a bundle
    (refusals re-raise and cache no bundle). The caller labels
    replay=True with SAMPLE_REPLAY_LABEL and performs no live waiting
    on that path. In time-consensus mode the extent binds the
    digests, so a different chunk duration or tolerance never
    replays another chunk's cache; other modes ignore the extent.
    The listed cuts and the clock variant bind when in effect, so
    a different cut list or prompt wording never replays either.
    """
    _require_video_file_id(video_file_id)
    prompt_cuts = _require_scene_cuts(scene_cuts)
    if not isinstance(clocked, bool):
        raise ValueError(f"clocked must be a bool, got {clocked!r}")
    config = spark_config_of(client)
    job_id = str(ctx.job_id)
    hit = stage_cache.cache_get(
        stage_cache.STAGE_SPARK_CONSENSUS,
        consensus_digest(
            job_id, video_file_id, config, chunk_extent, prompt_cuts, clocked
        ),
    )
    if hit is not None:
        return CachedExtraction(bundle=bundle_from_json(hit, config), replay=True)

    def _persist_raw(index: int, raw: Mapping[str, JSONValue]) -> None:
        stage_cache.cache_put(
            stage_cache.STAGE_SPARK_RUN,
            run_digest(
                job_id, video_file_id, index, config, chunk_extent,
                prompt_cuts, clocked,
            ),
            dict(raw),
        )

    bundle, _ = _extract_with_raw(
        ctx, video_file_id, client, config,
        chunk_extent=chunk_extent, scene_cuts=prompt_cuts, clocked=clocked,
        on_raw=_persist_raw,
        retry_call=retry_call, retryable_failures=retryable_failures,
    )
    stage_cache.cache_put(
        stage_cache.STAGE_SPARK_CONSENSUS,
        consensus_digest(
            job_id, video_file_id, config, chunk_extent, prompt_cuts, clocked
        ),
        bundle_to_json(bundle),
    )
    return CachedExtraction(bundle=bundle, replay=False)


# Format 2 binds the cut's decoded facts: records saved before them
# (format 1, nominal duration, no cut facts) fail the format check in
# _get_chunk_record and are never reused.
CHUNK_BUNDLE_FORMAT = "partfidelity-chunk-bundle/2"


def _checked_cut_facts(
    cut_start_s: float, cut_duration_s: float
) -> tuple[float, float]:
    """Validate caller-supplied cut facts, raising on anything unusable.

    Unusable includes facts whose float sum does not advance the
    window end past the start (``1e308 + 1.0``) or overflows
    (``1e308 + 1e308``): such facts can never be persisted.
    """
    if (
        isinstance(cut_start_s, bool)
        or not isinstance(cut_start_s, (int, float))
        or not math.isfinite(cut_start_s)
        or not cut_start_s >= 0
    ):
        raise ValueError(
            f"cut_start_s must be a finite number >= 0, got {cut_start_s!r}"
        )
    if (
        isinstance(cut_duration_s, bool)
        or not isinstance(cut_duration_s, (int, float))
        or not math.isfinite(cut_duration_s)
        or not cut_duration_s > 0
    ):
        raise ValueError(
            "cut_duration_s must be a finite number > 0,"
            f" got {cut_duration_s!r}"
        )
    start, duration = float(cut_start_s), float(cut_duration_s)
    end = start + duration
    if not math.isfinite(end) or not end > start:
        raise ValueError(
            "cut facts must advance the window end past the start,"
            f" got {cut_start_s!r} + {cut_duration_s!r}"
        )
    return start, duration


def chunk_bundle_digest(
    parent_job_id: str,
    source_hash: str,
    index: int,
    window_start_s: float,
    window_end_s: float,
    config: SparkStageConfig = DEFAULT_SPARK_CONFIG,
    chunk_tolerance_s: float | None = None,
    scene_cut_threshold: float | None = None,
    scene_cuts_in_prompt: bool | None = None,
    snap_window_s: float | None = None,
    min_phase_length_s: float | None = None,
    clock_burn_in: bool | None = None,
    merge_same_action: bool | None = None,
    *,
    cut_start_s: float,
    cut_duration_s: float,
) -> str:
    """Resume digest for one finished chunk, scoped to the parent job.

    The digest binds the parent job, the source video hash, the chunk
    index and window, the exclusion tolerance, the
    prompt/schema/model/span/consensus config, the cut's real start
    and decoded duration, and the extraction tune (cut threshold and
    prompt listing, snap window, minimum phase length, clock burn-in,
    same-action merge) — never an ephemeral Meta file id or child job
    id — so a re-entry after interruption reuses the chunk while any
    config, prompt, tolerance, decoded-extent, or tune change
    invalidates it. The caller probes the cut first: the digest is only
    computable after the free local cut. The outro drop needs no
    binding: it runs after stitching, so it never changes a saved
    chunk bundle.
    """
    if not isinstance(parent_job_id, str) or not parent_job_id.strip():
        raise ValueError("parent_job_id must be a non-empty string")
    if not isinstance(source_hash, str) or not source_hash.strip():
        raise ValueError("source_hash must be a non-empty string")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"chunk index must be an int >= 0, got {index!r}")
    if chunk_tolerance_s is not None and (
        isinstance(chunk_tolerance_s, bool)
        or not isinstance(chunk_tolerance_s, (int, float))
        or not math.isfinite(chunk_tolerance_s)
        or not chunk_tolerance_s >= 0
    ):
        raise ValueError(
            "chunk_tolerance_s must be None or a finite number >= 0,"
            f" got {chunk_tolerance_s!r}"
        )
    if scene_cut_threshold is not None and (
        isinstance(scene_cut_threshold, bool)
        or not isinstance(scene_cut_threshold, (int, float))
        or not math.isfinite(scene_cut_threshold)
        or not 0 < scene_cut_threshold <= 1
    ):
        raise ValueError(
            "scene_cut_threshold must be None or a finite number in"
            f" (0, 1], got {scene_cut_threshold!r}"
        )
    if scene_cuts_in_prompt is not None and not isinstance(
        scene_cuts_in_prompt, bool
    ):
        raise ValueError(
            "scene_cuts_in_prompt must be None or a bool,"
            f" got {scene_cuts_in_prompt!r}"
        )
    if snap_window_s is not None and (
        isinstance(snap_window_s, bool)
        or not isinstance(snap_window_s, (int, float))
        or not math.isfinite(snap_window_s)
        or not snap_window_s >= 0
    ):
        raise ValueError(
            "snap_window_s must be None or a finite number >= 0,"
            f" got {snap_window_s!r}"
        )
    if min_phase_length_s is not None and (
        isinstance(min_phase_length_s, bool)
        or not isinstance(min_phase_length_s, (int, float))
        or not math.isfinite(min_phase_length_s)
        or not min_phase_length_s > 0
    ):
        raise ValueError(
            "min_phase_length_s must be None or a finite number > 0,"
            f" got {min_phase_length_s!r}"
        )
    if clock_burn_in is not None and not isinstance(clock_burn_in, bool):
        raise ValueError(
            f"clock_burn_in must be None or a bool, got {clock_burn_in!r}"
        )
    if merge_same_action is not None and not isinstance(
        merge_same_action, bool
    ):
        raise ValueError(
            "merge_same_action must be None or a bool,"
            f" got {merge_same_action!r}"
        )
    cut_start, cut_duration = _checked_cut_facts(cut_start_s, cut_duration_s)
    return stage_cache.digest_for(
        stage_cache.STAGE_CHUNK_BUNDLE,
        parent_job_id,
        source_hash,
        str(index),
        repr(float(window_start_s)),
        repr(float(window_end_s)),
        SPARK_PROMPT_VERSION,
        SPARK_SCHEMA_VERSION,
        config.model_id,
        config.span_mode,
        str(config.runs),
        repr(config.spread_s),
        config.alignment_mode,
        ",".join(sorted(config.generic_tokens)),
        ";".join(",".join(sorted(group)) for group in config.verb_synonyms),
        ",".join(sorted(config.transition_verbs)),
        repr(config.verb_less_overlap_fraction),
        ",".join(sorted(config.negation_words)),
        ",".join(sorted(config.qualifiers)),
        ";".join(",".join(sorted(group)) for group in config.part_synonyms),
        repr(config.boundary_tolerance_s),
        repr(chunk_tolerance_s) if chunk_tolerance_s is not None else "none",
        repr(cut_start),
        repr(cut_duration),
        repr(scene_cut_threshold) if scene_cut_threshold is not None else "none",
        repr(scene_cuts_in_prompt)
        if scene_cuts_in_prompt is not None
        else "none",
        repr(snap_window_s) if snap_window_s is not None else "none",
        repr(min_phase_length_s) if min_phase_length_s is not None else "none",
        repr(clock_burn_in) if clock_burn_in is not None else "none",
        repr(merge_same_action) if merge_same_action is not None else "none",
    )


def save_chunk_bundle(
    digest: str,
    bundle: StepBundle,
    *,
    now: float | None = None,
    cut_start_s: float,
    cut_duration_s: float,
) -> None:
    """Persist one finished chunk bundle atomically for resume.

    The record holds the consensus bundle, its digest echo, the store
    time, and the cut's real start and decoded duration — no footage,
    frames, file ids, or credentials — so a resumed chunk stitches at
    the same real window as its fresh run.
    """
    cut_start, cut_duration = _checked_cut_facts(cut_start_s, cut_duration_s)
    record: JSONValue = {
        "format": CHUNK_BUNDLE_FORMAT,
        "digest": digest,
        "stored_at": time.time() if now is None else now,
        "bundle": bundle_to_json(bundle),
        "cut_start_s": cut_start,
        "cut_duration_s": cut_duration,
    }
    stage_cache.cache_put(stage_cache.STAGE_CHUNK_BUNDLE, digest, record)


def _recorded_cut_window(
    start: JSONValue, duration: JSONValue
) -> ChunkWindow | None:
    """Recorded cut window for persisted facts, or None when malformed.

    Missing, non-numeric, non-finite, or impossible facts fail closed
    to None, as does an end that overflows to non-finite (``1e308 +
    1e308``) or that float addition leaves at the start (``1e308 +
    1.0``): malformed cut metadata is a cache miss, never trusted.
    Huge JSON integers convert defensively, like ``_stored_at_s``.
    """
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, float))
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
    ):
        return None
    try:
        begin, span = float(start), float(duration)
    except (TypeError, OverflowError, ValueError):
        return None
    if (
        not math.isfinite(begin)
        or not begin >= 0
        or not math.isfinite(span)
        or not span > 0
    ):
        return None
    end = begin + span
    if not math.isfinite(end) or not end > begin:
        return None
    return ChunkWindow(start_s=begin, end_s=end)


def _get_chunk_record(
    digest: str, ttl_s: float, moment: float
) -> Mapping[str, JSONValue] | None:
    """Return one validated chunk record, or None when unusable.

    A missing, corrupt, mismatched, future-dated, or TTL-expired
    record fails closed to None, as does a record without valid cut
    facts (saved before them, or malformed); an expired record's file
    is removed best-effort, while a corrupt one is left for diagnosis.
    """
    try:
        raw = stage_cache.cache_get(stage_cache.STAGE_CHUNK_BUNDLE, digest)
    except (ValueError, RecursionError):
        # A deeply nested corrupt entry exhausts the JSON parser's
        # stack: a cache miss for this chunk, never a pipeline failure.
        return None
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return None
    required = {
        "format", "digest", "stored_at", "bundle",
        "cut_start_s", "cut_duration_s",
    }
    if set(raw) - required:
        return None
    if required - set(raw):
        return None
    if raw["format"] != CHUNK_BUNDLE_FORMAT or raw["digest"] != digest:
        return None
    stored_at = _stored_at_s(raw["stored_at"])
    if (
        stored_at is None
        or not 0 <= stored_at <= moment
        or moment - stored_at > ttl_s
    ):
        if stored_at is not None and 0 <= stored_at <= moment:
            stage_cache.cache_delete(stage_cache.STAGE_CHUNK_BUNDLE, digest)
        return None
    if (
        _recorded_cut_window(raw["cut_start_s"], raw["cut_duration_s"])
        is None
    ):
        return None
    return raw


def load_chunk_bundle(
    digest: str,
    config: SparkStageConfig = DEFAULT_SPARK_CONFIG,
    *,
    ttl_s: float,
    now: float | None = None,
) -> StepBundle | None:
    """Return the persisted chunk bundle, or None when unusable.

    A missing, corrupt, mismatched, future-dated, or TTL-expired
    record fails closed to None so only that chunk reruns, as does a
    record without valid cut facts; an expired record's file is
    removed best-effort, while a corrupt one is left for diagnosis
    (the rerun overwrites it atomically on success).
    """
    moment = time.time() if now is None else now
    raw = _get_chunk_record(digest, ttl_s, moment)
    if raw is None:
        return None
    try:
        return bundle_from_json(raw["bundle"], config)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


def load_chunk_cut_window(
    digest: str, *, ttl_s: float, now: float | None = None
) -> ChunkWindow | None:
    """Return one finished chunk's real cut window, or None.

    None covers every unusable case — a missing, corrupt, mismatched,
    or expired record, like :func:`load_chunk_bundle` — and a record
    saved without valid cut facts: the caller treats it as a cache
    miss and reruns the chunk, never trusting nominal or malformed
    metadata.
    """
    moment = time.time() if now is None else now
    raw = _get_chunk_record(digest, ttl_s, moment)
    if raw is None:
        return None
    return _recorded_cut_window(
        raw["cut_start_s"], raw["cut_duration_s"]
    )
