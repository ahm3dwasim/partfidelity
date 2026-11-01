"""Post-consensus boundary refinement: snap, fold, merge same-action neighbours.

Time consensus tiles the chunk with no gaps; this pass sharpens it.
Each internal boundary snaps to the nearest scene cut within a
configured window — sequentially in ascending order, so a snapped
edge never crosses its neighbour — each phase shorter than a
configured minimum folds into a neighbour (the one it shares label
agreement with, else the shorter one), and adjacent phases with the
same action after normalisation fuse into one. Every boundary keeps
its run support, its cut evidence, and its snap record for later
review, and steps beside a boundary that needs a check carry the
same flag. Edge outro phases drop after stitching (see
:func:`drop_edge_outro_phases`), never here: only whole-video edges
may go, never a middle chunk's edge.

Runs only in contiguous phases mode, after consensus and before the
contiguity gate (see :mod:`app.pipeline`): wording-grouped bundles
carry no per-boundary records and are refused here rather than
refined blind. Inputs tile exactly — a gap or overlap raises instead
of being papered over.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from app import step_alignment
from app.contracts import (
    MAX_PARTS_PER_STEP,
    BoundaryInfo,
    ExcludedSpan,
    Step,
    StepBundle,
)
from app.spark import DEFAULT_PHASE_KIND, OUTRO_PHASE_KIND


def _require_tiling_bundle(bundle: StepBundle) -> tuple[Step, ...]:
    """Return the bundle's steps, or raise when they cannot refine.

    Steps must tile exactly (each end equals the next start) and the
    bundle must carry one record per internal boundary; anything else
    raises instead of refining a shape snap and fold cannot keep.
    """
    if not isinstance(bundle, StepBundle):
        raise ValueError(
            f"bundle must be a StepBundle, got {type(bundle).__name__}"
        )
    steps = tuple(bundle.steps)
    if not steps:
        raise ValueError("refine needs at least one step")
    for left, right in zip(steps, steps[1:]):
        if right.start_s != left.end_s:
            raise ValueError(
                "refine needs exactly tiling steps: step"
                f" {left.number} ends at {left.end_s:g}s but step"
                f" {right.number} starts at {right.start_s:g}s"
            )
    records = tuple(bundle.boundaries)
    if len(records) != len(steps) - 1:
        raise ValueError(
            "refine needs one boundary record per internal boundary,"
            f" got {len(records)} records for {len(steps)} steps"
        )
    return steps


def _require_cuts(cuts: Sequence[float]) -> tuple[float, ...]:
    """Validate scene-cut times (window-relative, any order)."""
    if not isinstance(cuts, (tuple, list)):
        raise ValueError(
            f"cuts must be a tuple/list of times, got {type(cuts).__name__}"
        )
    checked: list[float] = []
    for cut in cuts:
        if (
            isinstance(cut, bool)
            or not isinstance(cut, (int, float))
            or not math.isfinite(cut)
            or not cut >= 0
        ):
            raise ValueError(f"scene cut times must be finite >= 0, got {cut!r}")
        checked.append(float(cut))
    return tuple(checked)


def _require_window(window_s: float) -> float:
    """Validate the snap window (0 snaps only onto exact cuts)."""
    if (
        isinstance(window_s, bool)
        or not isinstance(window_s, (int, float))
        or not math.isfinite(window_s)
        or not window_s >= 0
    ):
        raise ValueError(
            f"snap window must be a finite number >= 0, got {window_s!r}"
        )
    return float(window_s)


def _require_min_length(min_phase_length_s: float) -> float:
    """Validate the minimum phase length."""
    if (
        isinstance(min_phase_length_s, bool)
        or not isinstance(min_phase_length_s, (int, float))
        or not math.isfinite(min_phase_length_s)
        or not min_phase_length_s > 0
    ):
        raise ValueError(
            "minimum phase length must be a finite number > 0,"
            f" got {min_phase_length_s!r}"
        )
    return float(min_phase_length_s)


def _nearest_cut(
    position: float, cuts: Sequence[float]
) -> tuple[float | None, float | None]:
    """Closest cut to ``position`` and its distance (ties: earlier cut)."""
    if not cuts:
        return None, None
    nearest = min(cuts, key=lambda cut: (abs(cut - position), cut))
    return nearest, abs(nearest - position)


def _check_alignment_sets(
    generic_tokens: frozenset[str],
    verb_synonyms: tuple[frozenset[str], ...],
    negation_words: frozenset[str],
    qualifiers: frozenset[str],
    part_synonyms: tuple[frozenset[str], ...],
    transition_verbs: frozenset[str],
) -> None:
    """Validate the label-agreement sets (same shape as the stitcher)."""
    if not isinstance(generic_tokens, frozenset):
        raise ValueError(
            "generic_tokens must be a frozenset,"
            f" got {type(generic_tokens).__name__}"
        )
    step_alignment.normalize_generic_tokens(generic_tokens)
    if not isinstance(verb_synonyms, tuple) or any(
        not isinstance(group, frozenset) for group in verb_synonyms
    ):
        raise ValueError(
            "verb_synonyms must be a tuple of frozensets,"
            f" got {type(verb_synonyms).__name__}"
        )
    step_alignment.normalize_verb_synonyms(verb_synonyms)
    if not isinstance(negation_words, frozenset):
        raise ValueError(
            "negation_words must be a frozenset,"
            f" got {type(negation_words).__name__}"
        )
    step_alignment.normalize_negation_words(negation_words)
    if not isinstance(qualifiers, frozenset):
        raise ValueError(
            "qualifiers must be a frozenset,"
            f" got {type(qualifiers).__name__}"
        )
    step_alignment.normalize_qualifiers(qualifiers)
    if not isinstance(part_synonyms, tuple) or any(
        not isinstance(group, frozenset) for group in part_synonyms
    ):
        raise ValueError(
            "part_synonyms must be a tuple of frozensets,"
            f" got {type(part_synonyms).__name__}"
        )
    step_alignment.normalize_part_synonyms(part_synonyms)
    if not isinstance(transition_verbs, frozenset):
        raise ValueError(
            "transition_verbs must be a frozenset,"
            f" got {type(transition_verbs).__name__}"
        )
    step_alignment.normalize_transition_verbs(transition_verbs)


def snap_bundle_boundaries(
    bundle: StepBundle, cuts: Sequence[float], window_s: float
) -> StepBundle:
    """Snap each internal boundary to its nearest cut within ``window_s``.

    Edges move in ascending order and each snapped edge must stay
    strictly between its neighbours (the already-snapped previous
    edge and the still-original next one), so two boundaries can
    never cross or collide; a cut that would cross vetoes the snap
    and the edge stays. Records gain the nearest cut to the final
    position, whether the edge moved and by how much, and the check
    flag (kept support flag, or no cut within the window); steps
    beside a flagged boundary are flagged too. Times otherwise
    untouched, tiling preserved.
    """
    steps = _require_tiling_bundle(bundle)
    cuts = _require_cuts(cuts)
    window = _require_window(window_s)
    if len(steps) == 1:
        return bundle
    edges = [step.start_s for step in steps] + [steps[-1].end_s]
    records: list[BoundaryInfo] = []
    for position, record in zip(edges[1:-1], bundle.boundaries):
        index = len(records) + 1
        nearest, distance = _nearest_cut(position, cuts)
        target = position
        if nearest is not None and distance is not None and distance <= window:
            if edges[index - 1] < nearest < edges[index + 1]:
                target = nearest
        snapped = target != position
        edges[index] = target
        post_cut, post_distance = _nearest_cut(target, cuts)
        records.append(
            replace(
                record,
                time_s=target,
                nearest_cut_s=post_cut,
                cut_distance_s=post_distance,
                snapped=snapped,
                snap_delta_s=target - position,
                needs_check=(
                    record.needs_check
                    or post_cut is None
                    or (post_distance is not None and post_distance > window)
                ),
            )
        )
    flagged = [record.needs_check for record in records]
    snapped_steps = tuple(
        replace(
            step,
            start_s=edges[number],
            end_s=edges[number + 1],
            time_needs_check=(
                step.time_needs_check
                or (number > 0 and flagged[number - 1])
                or (number < len(flagged) and flagged[number])
            ),
        )
        for number, step in enumerate(steps)
    )
    return replace(bundle, steps=snapped_steps, boundaries=tuple(records))


def fold_short_phases(
    bundle: StepBundle,
    min_phase_length_s: float,
    *,
    generic_tokens: frozenset[str] = step_alignment.GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = (
        step_alignment.VERB_SYNONYM_GROUPS
    ),
    negation_words: frozenset[str] = step_alignment.NEGATION_WORDS,
    qualifiers: frozenset[str] = step_alignment.QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = (
        step_alignment.PART_SYNONYM_GROUPS
    ),
    transition_verbs: frozenset[str] = step_alignment.TRANSITION_VERBS,
) -> StepBundle:
    """Merge every phase shorter than ``min_phase_length_s`` away.

    The short phase merges into the neighbour it shares label
    agreement with (:func:`step_alignment.same_action` on actions
    plus part tokens, phases mode); when both or neither neighbour
    agrees it merges into the shorter one (ties: the earlier). A
    lone phase has no neighbour and survives. Merging absorbs the
    span into the surviving neighbour — never a gap — drops the edge
    between them and its record, keeps the widest spread and any
    flag, and renumbers from 1. Repeats to a fixpoint.
    """
    steps = list(_require_tiling_bundle(bundle))
    records = list(bundle.boundaries)
    minimum = _require_min_length(min_phase_length_s)
    _check_alignment_sets(
        generic_tokens,
        verb_synonyms,
        negation_words,
        qualifiers,
        part_synonyms,
        transition_verbs,
    )
    while len(steps) > 1:
        short = next(
            (
                index
                for index, step in enumerate(steps)
                if step.end_s - step.start_s < minimum
            ),
            None,
        )
        if short is None:
            break
        current = steps[short]
        current_parts = step_alignment.part_tokens(
            part.phrase for part in current.parts
        )
        if short == 0:
            merge_into = short + 1
        elif short == len(steps) - 1:
            merge_into = short - 1
        else:

            def _agrees(other: Step) -> bool:
                return step_alignment.same_action(
                    current.action,
                    other.action,
                    current_parts,
                    step_alignment.part_tokens(
                        part.phrase for part in other.parts
                    ),
                    generic_tokens=generic_tokens,
                    verb_synonyms=verb_synonyms,
                    negation_words=negation_words,
                    qualifiers=qualifiers,
                    part_synonyms=part_synonyms,
                    transition_verbs=transition_verbs,
                    phases_mode=True,
                )

            agree_prev = _agrees(steps[short - 1])
            agree_next = _agrees(steps[short + 1])
            if agree_prev != agree_next:
                merge_into = short - 1 if agree_prev else short + 1
            else:
                prev_length = steps[short - 1].end_s - steps[short - 1].start_s
                next_length = steps[short + 1].end_s - steps[short + 1].start_s
                merge_into = (
                    short - 1 if prev_length <= next_length else short + 1
                )
        survivor = steps[merge_into]
        if merge_into < short:
            merged = replace(
                survivor,
                end_s=current.end_s,
                time_spread_s=max(survivor.time_spread_s, current.time_spread_s),
                time_needs_check=(
                    survivor.time_needs_check or current.time_needs_check
                ),
            )
            steps[merge_into] = merged
            del steps[short]
            del records[short - 1]
        else:
            merged = replace(
                survivor,
                start_s=current.start_s,
                time_spread_s=max(survivor.time_spread_s, current.time_spread_s),
                time_needs_check=(
                    survivor.time_needs_check or current.time_needs_check
                ),
            )
            steps[merge_into] = merged
            del steps[short]
            del records[short]
    renumbered = tuple(
        replace(step, number=number) for number, step in enumerate(steps, start=1)
    )
    return replace(bundle, steps=renumbered, boundaries=tuple(records))


def _merge_key(action: str) -> str:
    """Action text normalised for neighbour merging: lowercased,
    punctuation to spaces, whitespace collapsed."""
    lowered = " ".join(action.strip().lower().split())
    return " ".join(
        "".join(char if char.isalnum() else " " for char in lowered).split()
    )


def _union_in_order(first: tuple, second: tuple) -> tuple:
    """Union of two field tuples, first-phase order, duplicates dropped."""
    return tuple(dict.fromkeys(first + second))


def merge_same_action_phases(bundle: StepBundle) -> StepBundle:
    """Fuse adjacent phases with the same normalised action.

    Two neighbours fuse when their actions match after case,
    whitespace, and punctuation normalisation, and only when that
    normalised key is non-empty and their parts fit the contract limit.
    The fused phase spans both, keeps the first
    phase's action, the widest spread and any flag; tools, parts, and
    hazards union in order with duplicates dropped. Pairs whose parts
    exceed the contract limit stay separate; its kind is the
    shared kind when both agree, else repair (a mixed pair never fuses
    into a droppable phase). The edge between them and its record
    drop; surviving boundaries keep their records, and steps renumber
    from 1. Tiling preserved.
    """
    steps = list(_require_tiling_bundle(bundle))
    records = list(bundle.boundaries)
    index = 0
    while index < len(steps) - 1:
        current = steps[index]
        nxt = steps[index + 1]
        key = _merge_key(current.action)
        if not key or key != _merge_key(nxt.action):
            index += 1
            continue
        parts = _union_in_order(current.parts, nxt.parts)
        if len(parts) > MAX_PARTS_PER_STEP:
            index += 1
            continue
        kind = (
            current.kind if current.kind == nxt.kind else DEFAULT_PHASE_KIND
        )
        steps[index] = replace(
            current,
            end_s=nxt.end_s,
            tools=_union_in_order(current.tools, nxt.tools),
            parts=parts,
            hazards=_union_in_order(current.hazards, nxt.hazards),
            time_spread_s=max(current.time_spread_s, nxt.time_spread_s),
            time_needs_check=(
                current.time_needs_check or nxt.time_needs_check
            ),
            kind=kind,
        )
        del steps[index + 1]
        del records[index]
    renumbered = tuple(
        replace(step, number=number) for number, step in enumerate(steps, start=1)
    )
    return replace(bundle, steps=renumbered, boundaries=tuple(records))


def drop_edge_outro_phases(
    steps: Sequence[Step],
) -> tuple[tuple[Step, ...], tuple[ExcludedSpan, ...]]:
    """Drop leading/trailing outro runs; return kept steps and excluded spans.

    Only whole-video edges drop here (the pipeline calls this after
    stitching and deriving ends): a leading run and a trailing run of
    ``outro_or_unrelated`` phases go, a middle outro stays, the intro
    stays, and a bundle that is outro throughout keeps everything
    (never empties). Kept steps stay contiguous and renumber from 1;
    excluded spans carry exact times for the "skipped, not part of
    the repair" record.
    """
    ordered = tuple(steps)
    if not ordered:
        raise ValueError("drop needs at least one step")
    leading = 0
    while (
        leading < len(ordered)
        and ordered[leading].kind == OUTRO_PHASE_KIND
    ):
        leading += 1
    trailing = 0
    while (
        trailing < len(ordered) - leading
        and ordered[len(ordered) - 1 - trailing].kind == OUTRO_PHASE_KIND
    ):
        trailing += 1
    if leading + trailing >= len(ordered):
        keep_all = tuple(
            replace(step, number=number)
            for number, step in enumerate(ordered, start=1)
        )
        return keep_all, ()
    end = len(ordered) - trailing
    kept = tuple(
        replace(step, number=number)
        for number, step in enumerate(ordered[leading:end], start=1)
    )
    excluded = tuple(
        ExcludedSpan(start_s=step.start_s, end_s=step.end_s, kind=step.kind)
        for step in (*ordered[:leading], *ordered[end:])
    )
    return kept, excluded


def refine_consensus_bundle(
    bundle: StepBundle,
    *,
    cuts: Sequence[float],
    snap_window_s: float,
    min_phase_length_s: float,
    generic_tokens: frozenset[str] = step_alignment.GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = (
        step_alignment.VERB_SYNONYM_GROUPS
    ),
    negation_words: frozenset[str] = step_alignment.NEGATION_WORDS,
    qualifiers: frozenset[str] = step_alignment.QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = (
        step_alignment.PART_SYNONYM_GROUPS
    ),
    transition_verbs: frozenset[str] = step_alignment.TRANSITION_VERBS,
    merge_same_action: bool = True,
) -> StepBundle:
    """Snap, fold, then merge one time-consensus bundle (contiguous only)."""
    if not isinstance(merge_same_action, bool):
        raise ValueError(
            f"merge_same_action must be a bool, got {merge_same_action!r}"
        )
    snapped = snap_bundle_boundaries(bundle, cuts, snap_window_s)
    folded = fold_short_phases(
        snapped,
        min_phase_length_s,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        transition_verbs=transition_verbs,
    )
    if not merge_same_action:
        return folded
    return merge_same_action_phases(folded)


__all__ = (
    "drop_edge_outro_phases",
    "fold_short_phases",
    "merge_same_action_phases",
    "refine_consensus_bundle",
    "snap_bundle_boundaries",
)
