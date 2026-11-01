"""Canonical step-timing scorer: align by time, then check boundaries.

Each truth step pairs one-to-one with the guide step that maximises total
time overlap (an exact Hungarian assignment, never greedy; only pairs with
positive overlap count). Wording never vetoes a pair: it is reported per
pair instead (the shared same-action predicate plus a token similarity),
so the score measures boundaries, not phrasing. A paired truth step passes
when both |d_start| and |d_end| are within tolerance; the run passes at
the 7-of-9 ratio scaled to the truth length by exact integer arithmetic.

The tolerance, the strict report-only lines, and the pass line are
evaluation law from app.contracts.TIMING_SCORE_LAW,
read through a loader that fails loudly on a missing key -- never
literals here, never tunable knobs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.contracts import TIMING_SCORE_LAW
from app.step_alignment import (
    GENERIC_TOKENS,
    NEGATION_WORDS,
    PART_SYNONYM_GROUPS,
    QUALIFIERS,
    TRANSITION_VERBS,
    VERB_SYNONYM_GROUPS,
    content_tokens,
    normalize_generic_tokens,
    normalize_negation_words,
    normalize_part_synonyms,
    normalize_qualifiers,
    normalize_transition_verbs,
    normalize_verb_synonyms,
    same_action,
)


def load_scorer_law(key: str) -> float:
    """Read one timing scoring-law value, failing loudly on a missing key."""
    try:
        value = TIMING_SCORE_LAW[key]
    except KeyError:
        raise KeyError(
            f"timing scoring law has no key {key!r}; fix app.contracts.TIMING_SCORE_LAW"
        ) from None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"timing scoring law key {key!r} must be a finite number, got {value!r}"
        )
    return value


def _load_pass_line(key: str) -> int:
    value = load_scorer_law(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"timing scoring law key {key!r} must be an int, got {value!r}"
        )
    return value


START_END_TOLERANCE_S = load_scorer_law("tolerance_s")
STRICT_TOLERANCE_2_S = load_scorer_law("strict_2_s")
STRICT_TOLERANCE_1_S = load_scorer_law("strict_1_s")
PASS_COUNT = _load_pass_line("pass_count")
PASS_OF = _load_pass_line("pass_of")
if not START_END_TOLERANCE_S > 0:
    raise ValueError("timing scoring law tolerance_s must be > 0")
if not STRICT_TOLERANCE_2_S > 0 or not STRICT_TOLERANCE_1_S > 0:
    raise ValueError("timing scoring law strict lines must be > 0")
if not 0 < PASS_COUNT <= PASS_OF:
    raise ValueError("timing scoring law needs 0 < pass_count <= pass_of")


@dataclass(frozen=True)
class ScoredStep:
    """One guide or GT row reduced to its number, window, and action."""

    number: int
    start_s: float
    end_s: float
    action: str


@dataclass(frozen=True)
class TimeMatch:
    """One aligned guide/GT pair with its boundary verdict."""

    gt_number: int
    guide_number: int
    d_start_s: float
    d_end_s: float
    passed: bool
    wording_same_action: bool
    wording_similarity: float


@dataclass(frozen=True)
class TimeReport:
    """Coverage, boundary accuracy, and the strict overall pass, separately."""

    matches: tuple[TimeMatch, ...]
    missing_gt: tuple[int, ...]
    extra_guide: tuple[int, ...]
    coverage: float  # matched GT rows / GT rows
    boundary_accuracy: float  # passed matches / matches (0.0 when none)
    strict_passes: int  # matched timing passes out of the GT rows
    strict_passes_2s: int  # report-only: matches within ±2 s
    strict_passes_1s: int  # report-only: matches within ±1 s
    passed: bool  # strict_passes / GT rows >= PASS_COUNT / PASS_OF


def as_scored_step(row: Mapping[str, Any] | Any) -> ScoredStep:
    """Adapt a GT dict, a fixture mapping, or a step object to a row."""
    if isinstance(row, Mapping):
        return ScoredStep(
            number=int(row["number"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            action=str(row["action"]),
        )
    return ScoredStep(
        number=int(getattr(row, "number")),
        start_s=float(getattr(row, "start_s")),
        end_s=float(getattr(row, "end_s")),
        action=str(getattr(row, "action")),
    )


def wording_similarity(first: str, second: str) -> float:
    """Jaccard similarity over content tokens (stopwords out), 0..1.

    Identical wordings score 1.0 (two empty token sets included);
    disjoint wordings score 0.0. Report-only: never changes the pass count.
    """
    left = content_tokens(first)
    right = content_tokens(second)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _overlap_s(first: ScoredStep, second: ScoredStep) -> float:
    return max(
        0.0, min(first.end_s, second.end_s) - max(first.start_s, second.start_s)
    )


def _checked_excluded_spans(
    excluded_spans: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Validate excluded truth spans: [start, end] pairs, 0 <= start < end."""
    checked: list[tuple[float, float]] = []
    for span in excluded_spans:
        if (
            isinstance(span, (str, bytes))
            or not isinstance(span, Sequence)
            or len(span) != 2
        ):
            raise ValueError(
                "excluded_spans must be [start_s, end_s] pairs,"
                f" got {span!r}"
            )
        start, end = span
        for value in (start, end):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "excluded_spans must be [start_s, end_s] pairs,"
                    f" got {span!r}"
                )
        if not 0 <= float(start) < float(end):
            raise ValueError(
                "excluded_spans must satisfy 0 <= start_s < end_s,"
                f" got {span!r}"
            )
        checked.append((float(start), float(end)))
    return checked


def _coalesce_spans(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge touching or overlapping excluded spans into their union.

    Adjacent trim records (one outro phase per span) share an edge;
    without coalescing, a guide row spanning the shared edge is inside
    no single span and wrongly counts as extra.
    """
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside_excluded(step: ScoredStep, spans: Sequence[tuple[float, float]]) -> bool:
    """True when the guide row falls entirely inside one excluded span."""
    return any(
        start <= step.start_s and step.end_s <= end for start, end in spans
    )


def _assign_max_overlap(weights: Sequence[Sequence[float]]) -> list[int]:
    """Exact maximum-weight one-to-one assignment (Hungarian, O(n^3)).

    ``weights`` is a square n x n matrix; returns the assigned column per
    row. Deterministic: ties resolve by construction order, so the lowest
    candidate wins. Zero-weight pairs are assigned here and dropped by the
    caller, so only positive-overlap pairs count.
    """
    size = len(weights)
    row_dual = [0.0] * (size + 1)
    col_dual = [0.0] * (size + 1)
    col_match = [0] * (size + 1)  # col_match[j] = row matched to column j
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        col_match[0] = row
        col0 = 0
        min_slack = [math.inf] * (size + 1)
        seen = [False] * (size + 1)
        while True:
            seen[col0] = True
            row0 = col_match[col0]
            delta = math.inf
            col1 = 0
            for col in range(1, size + 1):
                if not seen[col]:
                    # Negated weight: the classic min-cost loop maximises.
                    slack = (
                        -weights[row0 - 1][col - 1]
                        - row_dual[row0]
                        - col_dual[col]
                    )
                    if slack < min_slack[col]:
                        min_slack[col] = slack
                        predecessor[col] = col0
                    if min_slack[col] < delta:
                        delta = min_slack[col]
                        col1 = col
            for col in range(size + 1):
                if seen[col]:
                    row_dual[col_match[col]] += delta
                    col_dual[col] -= delta
                else:
                    min_slack[col] -= delta
            col0 = col1
            if col_match[col0] == 0:
                break
        while col0:
            col1 = predecessor[col0]
            col_match[col0] = col_match[col1]
            col0 = col1
    assigned = [-1] * size
    for col in range(1, size + 1):
        if col_match[col]:
            assigned[col_match[col] - 1] = col - 1
    return assigned


def align_guide_to_gt(
    guide_rows: Sequence[Mapping[str, Any] | Any],
    gt_rows: Sequence[Mapping[str, Any] | Any],
    *,
    generic_tokens: frozenset[str] = GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = VERB_SYNONYM_GROUPS,
    negation_words: frozenset[str] = NEGATION_WORDS,
    qualifiers: frozenset[str] = QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = PART_SYNONYM_GROUPS,
    transition_verbs: frozenset[str] = TRANSITION_VERBS,
    excluded_spans: Sequence[Sequence[float]] = (),
) -> tuple[list[tuple[ScoredStep, ScoredStep]], list[ScoredStep], list[ScoredStep]]:
    """Pair guide rows to GT rows one-to-one by maximum time overlap.

    Returns ``(pairs, missing_gt, extra_guide)`` in GT order. An exact
    assignment (Hungarian), never greedy; only positive-overlap pairs
    count, so a guide row left unmatched is extra and a GT row never
    taken is missing. Wording never vetoes a pair: the wording knobs are
    accepted for signature compatibility and ignored here (``score_guide``
    uses them for its reported wording metric). Guide rows falling
    entirely inside the union of the excluded truth spans (touching
    spans coalesced first) are ignored first (neither matched nor
    extra): they cover truth territory the guide leaves out
    by design, not a wrong step.
    """
    _ = (
        generic_tokens,
        verb_synonyms,
        negation_words,
        qualifiers,
        part_synonyms,
        transition_verbs,
    )
    excluded = _coalesce_spans(_checked_excluded_spans(excluded_spans))
    guide = [
        step for step in (as_scored_step(row) for row in guide_rows)
        if not _inside_excluded(step, excluded)
    ]
    gt = [as_scored_step(row) for row in gt_rows]
    size = max(len(guide), len(gt))
    if size == 0:
        return [], [], []
    weights: list[list[float]] = [[0.0] * size for _ in range(size)]
    for i, step in enumerate(guide):
        for j, truth in enumerate(gt):
            weights[i][j] = _overlap_s(step, truth)
    assigned = _assign_max_overlap(weights)
    index_pairs = sorted(
        (
            (i, assigned[i])
            for i in range(len(guide))
            if 0 <= assigned[i] < len(gt) and weights[i][assigned[i]] > 0.0
        ),
        key=lambda item: (item[1], item[0]),
    )
    matched_guide = {i for i, _ in index_pairs}
    matched_gt = {j for _, j in index_pairs}
    pairs = [(guide[i], gt[j]) for i, j in index_pairs]
    missing = [step for index, step in enumerate(gt) if index not in matched_gt]
    extra = [step for index, step in enumerate(guide) if index not in matched_guide]
    return pairs, missing, extra


def score_guide(
    guide_rows: Sequence[Mapping[str, Any] | Any],
    gt_rows: Sequence[Mapping[str, Any] | Any],
    *,
    generic_tokens: frozenset[str] = GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = VERB_SYNONYM_GROUPS,
    negation_words: frozenset[str] = NEGATION_WORDS,
    qualifiers: frozenset[str] = QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = PART_SYNONYM_GROUPS,
    transition_verbs: frozenset[str] = TRANSITION_VERBS,
    excluded_spans: Sequence[Sequence[float]] = (),
) -> TimeReport:
    """Align guide rows to GT rows by time, then score matched boundaries."""
    generic = normalize_generic_tokens(generic_tokens)
    synonyms = normalize_verb_synonyms(verb_synonyms)
    negs = normalize_negation_words(negation_words)
    quals = normalize_qualifiers(qualifiers)
    psyns = normalize_part_synonyms(part_synonyms)
    trans = normalize_transition_verbs(transition_verbs)
    pairs, missing, extra = align_guide_to_gt(
        guide_rows, gt_rows,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        transition_verbs=transition_verbs,
        excluded_spans=excluded_spans,
    )
    matches: list[TimeMatch] = []
    for step, gt_step in pairs:
        d_start = abs(step.start_s - gt_step.start_s)
        d_end = abs(step.end_s - gt_step.end_s)
        matches.append(
            TimeMatch(
                gt_number=gt_step.number,
                guide_number=step.number,
                d_start_s=d_start,
                d_end_s=d_end,
                passed=bool(
                    d_start <= START_END_TOLERANCE_S
                    and d_end <= START_END_TOLERANCE_S
                ),
                wording_same_action=bool(
                    same_action(
                        step.action, gt_step.action,
                        generic_tokens=generic,
                        verb_synonyms=synonyms,
                        negation_words=negs,
                        qualifiers=quals,
                        part_synonyms=psyns,
                        transition_verbs=trans,
                        phases_mode=True,
                    )
                ),
                wording_similarity=wording_similarity(step.action, gt_step.action),
            )
        )
    gt_count = len(list(gt_rows))
    strict = sum(1 for match in matches if match.passed)
    strict_2s = sum(
        1
        for match in matches
        if match.d_start_s <= STRICT_TOLERANCE_2_S
        and match.d_end_s <= STRICT_TOLERANCE_2_S
    )
    strict_1s = sum(
        1
        for match in matches
        if match.d_start_s <= STRICT_TOLERANCE_1_S
        and match.d_end_s <= STRICT_TOLERANCE_1_S
    )
    return TimeReport(
        matches=tuple(matches),
        missing_gt=tuple(step.number for step in missing),
        extra_guide=tuple(step.number for step in extra),
        coverage=(len(matches) / gt_count) if gt_count else 0.0,
        boundary_accuracy=(strict / len(matches)) if matches else 0.0,
        strict_passes=strict,
        strict_passes_2s=strict_2s,
        strict_passes_1s=strict_1s,
        # The 7-of-9 bar as exact integer math (strict/GT >= count/of):
        # identical to strict >= 7 for 9-row truth, defined for any length.
        passed=bool(gt_count and strict * PASS_OF >= PASS_COUNT * gt_count),
    )


# Former names, kept so existing imports keep working.
D1bMatch = TimeMatch
D1bReport = TimeReport
load_d1b_law = load_scorer_law


__all__ = (
    "PASS_COUNT",
    "PASS_OF",
    "START_END_TOLERANCE_S",
    "STRICT_TOLERANCE_1_S",
    "STRICT_TOLERANCE_2_S",
    "TimeMatch",
    "TimeReport",
    "D1bMatch",
    "D1bReport",
    "ScoredStep",
    "align_guide_to_gt",
    "as_scored_step",
    "load_scorer_law",
    "load_d1b_law",
    "score_guide",
    "wording_similarity",
)
