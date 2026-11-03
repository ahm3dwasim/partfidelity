"""Long-footage chunking: plan windows, cut with ffmpeg, stitch guides.

Videos over the chunk length are cut into ``chunk_s`` windows with a small
overlap, each window runs its own Spark extraction under its own ledger
share (see :mod:`app.pipeline`), and the per-chunk bundles stitch into one
guide on the original timeline here: times offset by window start, a step
seen in two windows merged into its union, overlap sightings deduped, and
rival windows kept and flagged. Merging uses the same shared alignment
as the readback module's consensus (:mod:`app.step_alignment`; honest disagreement kept
and flagged), but the inputs are consensus steps on nominal windows, not
raw runs. The old same-semantic-key merge stays available only as the
configured `exact` rollback mode.

Guide cards are contiguous repair phases covering the whole video
(fix 1): after stitching, ends derive from the next start
(``start_1 = 0``, ``end_i = start_(i+1)``, ``end_last = duration``,
intervals ``[start, end)``). Coverage and monotonicity are enforced
*before* deriving ends — a gap beyond tolerance is a missing phase and
fails closed instead of stretching a neighbour — and the old span
behaviour stays only as the configured ``legacy`` span mode.

Cuts are ffmpeg stream copies (fast, no re-encode), so a cut starts on a
keyframe near the nominal start; step times stay approximate and the
overlap absorbs the slop. Each cut is probed after the fact: prompts,
consensus, upload metadata, and validation use the cut's decoded
duration, and stitched times map back with the cut's real start.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app import step_alignment
from app.contracts import ExcludedSpan, Step, StepBundle

# Internal working names under the job directory, never served. Registered
# as provisional artifacts, so the finish gate tombstones them and the
# sweeper deletes the bytes on success, failure, or crash expiry.
CHUNK_FILENAME_PATTERN = "chunk_{index:03d}.mp4"
CHUNK_ARTIFACT_KIND = "provisional"

# Step-span modes. ``contiguous`` derives ends from the next start after
# enforcing coverage; ``legacy`` keeps model spans untouched (rollback).
SPAN_MODES = ("contiguous", "legacy")
DEFAULT_SPAN_MODE = "contiguous"


class ContiguityError(ValueError):
    """Steps do not tile their extent: a phase is missing or misplaced."""


class _Timed(Protocol):
    """Anything with a step window (Step, validated run steps)."""

    @property
    def start_s(self) -> float: ...
    @property
    def end_s(self) -> float: ...


@dataclass(frozen=True)
class ChunkWindow:
    """One nominal cut window on the original timeline, in seconds."""

    start_s: float
    end_s: float


@dataclass(frozen=True)
class StitchedSteps:
    """Chunk bundles stitched onto the original timeline.

    Steps are sorted and renumbered from 1; the count may exceed one
    bundle's 12-step bound, so this is deliberately not a StepBundle.
    Downstream stages read ``steps`` and ``device_guess`` only.
    ``excluded_spans`` records whole-video edge outros dropped after
    deriving ends (empty until the pipeline trims). ``origins`` holds,
    per step, the set of chunk indices whose sightings it merged.
    """

    steps: tuple[Step, ...]
    device_guess: str
    confidence: float
    excluded_spans: tuple[ExcludedSpan, ...] = ()
    origins: tuple[frozenset[int], ...] = ()
    # Fewest runs any chunk's consensus rested on (None: not recorded).
    fewest_runs: int | None = None


class ChunkCutError(RuntimeError):
    """A chunk cut failed (ffmpeg missing, failed, or too slow)."""


def plan_chunks(
    duration_s: float,
    chunk_s: float,
    overlap_s: float,
    *,
    single_window_s: float | None = None,
) -> tuple[ChunkWindow, ...]:
    """Cover ``[0, duration_s]`` with chunk windows and the overlap stride.

    A video within ``single_window_s`` (default: ``chunk_s``) yields
    the single full window; longer videos yield windows of ``chunk_s``
    starting every ``chunk_s - overlap_s`` seconds, the last clamped
    to the duration. Whole-video extraction passes the configured
    single-window ceiling here once the reservation fits.
    """
    for name, value in (("duration_s", duration_s), ("chunk_s", chunk_s)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not value > 0
        ):
            raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    if single_window_s is None:
        single_window_s = chunk_s
    if (
        isinstance(single_window_s, bool)
        or not isinstance(single_window_s, (int, float))
        or not math.isfinite(single_window_s)
        or not single_window_s > 0
    ):
        raise ValueError(
            "single_window_s must be a finite number > 0,"
            f" got {single_window_s!r}"
        )
    if (
        isinstance(overlap_s, bool)
        or not isinstance(overlap_s, (int, float))
        or not math.isfinite(overlap_s)
        or not overlap_s >= 0
    ):
        raise ValueError(
            f"overlap_s must be a finite number >= 0, got {overlap_s!r}"
        )
    if overlap_s >= chunk_s:
        raise ValueError(
            f"overlap_s ({overlap_s!r}) must be below chunk_s ({chunk_s!r})"
        )
    duration = float(duration_s)
    length = float(chunk_s)
    if duration <= float(single_window_s):
        return (ChunkWindow(start_s=0.0, end_s=duration),)
    stride = length - float(overlap_s)
    windows: list[ChunkWindow] = []
    start = 0.0
    while start < duration:
        windows.append(ChunkWindow(start_s=start, end_s=min(start + length, duration)))
        start += stride
    return tuple(windows)


def chunk_filename(index: int) -> str:
    """Internal working name of one cut chunk (never served)."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"chunk index must be an int >= 0, got {index!r}")
    return CHUNK_FILENAME_PATTERN.format(index=index)


def cut_chunk(
    src: Path, window: ChunkWindow, dst: Path, *, timeout_s: float
) -> Path:
    """Cut one nominal window with a stream copy; raise :class:`ChunkCutError`.

    A partial output is unlinked best-effort, so a failed cut never leaves
    bytes the caller mistakes for a chunk.
    """
    duration = window.end_s - window.start_s
    if duration <= 0:
        raise ValueError(f"chunk window has no duration: {window!r}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(window.start_s),
                "-i",
                str(src),
                "-t",
                str(duration),
                "-c",
                "copy",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ChunkCutError(f"could not cut chunk at {window.start_s:g}s") from error
    if completed.returncode != 0 or not dst.is_file():
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise ChunkCutError(f"could not cut chunk at {window.start_s:g}s")
    return dst


@dataclass(frozen=True)
class CutInfo:
    """ffprobe facts about one cut file, on the nominal cut timeline.

    ``start_s`` is the cut's presentation start (0 when the cut opens
    exactly at its nominal start, the usual mp4 case: the muxer shifts
    the keyframe pre-roll out of the presentation) and ``duration_s``
    its decoded duration — what a decoder plays, which a frame
    straddling the cut end or container rounding can push past the
    nominal window length.
    """

    start_s: float
    duration_s: float

    def __post_init__(self) -> None:
        start = self.start_s
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(start)
        ):
            raise ValueError(
                f"cut start must be a finite number, got {start!r}"
            )
        duration = self.duration_s
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not duration > 0
        ):
            raise ValueError(
                f"cut duration must be a finite number > 0, got {duration!r}"
            )
        object.__setattr__(self, "start_s", float(start))
        object.__setattr__(self, "duration_s", float(duration))


def probe_cut(path: Path, *, timeout_s: float) -> CutInfo:
    """Probe one cut file's presentation start and decoded duration.

    The duration is what a decoder plays (ffprobe format duration),
    not the nominal window the cut was asked for; the prompt, the
    consensus extent, upload metadata, validation, and digests all
    take this value. Anything unreadable raises :class:`ChunkCutError`,
    failing the chunk closed like a bad cut.
    """
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "format=duration,start_time",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ChunkCutError(f"could not probe cut {path}") from error
    if completed.returncode != 0:
        raise ChunkCutError(f"could not probe cut {path}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except ValueError as error:
        raise ChunkCutError(f"could not probe cut {path}") from error
    values = payload.get("format") or {}
    try:
        start = float(values["start_time"])
        duration = float(values["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ChunkCutError(f"could not probe cut {path}") from error
    try:
        return CutInfo(start_s=start, duration_s=duration)
    except ValueError as error:
        raise ChunkCutError(f"could not probe cut {path}") from error


def real_cut_window(window: ChunkWindow, probe: CutInfo) -> ChunkWindow:
    """Map one cut back onto the source timeline with its real start.

    Stream-copy cuts snap to keyframes, so the cut's first presented
    frame can sit before its nominal start; the probe's start carries
    that shift (0 when the cut opens exactly where planned). Stitched
    times add this window's start and the consensus extent spans the
    decoded duration, so a short last chunk tiles what the bytes
    actually play. A start before the source is impossible for a real
    cut, so it fails the chunk closed like a bad cut.
    """
    if not isinstance(window, ChunkWindow):
        raise ValueError(
            "real_cut_window needs a ChunkWindow,"
            f" got {type(window).__name__}"
        )
    if not isinstance(probe, CutInfo):
        raise ValueError(
            f"real_cut_window needs a CutInfo, got {type(probe).__name__}"
        )
    start = window.start_s + probe.start_s
    if not start >= 0:
        raise ChunkCutError(
            f"cut {window!r} probed a start before the source:"
            f" {probe.start_s:g}s"
        )
    return ChunkWindow(start_s=start, end_s=start + probe.duration_s)


def _normalize_action(action: str) -> str:
    return " ".join(action.strip().lower().split())


def _parts_signature(step: Step) -> tuple[str, ...]:
    return tuple(
        sorted(" ".join(part.phrase.strip().lower().split()) for part in step.parts)
    )


def _semantic_key(step: Step) -> tuple[str, tuple[str, ...]]:
    """Merge key for stitched steps; mirrors the readback module's raw-run key."""
    return (_normalize_action(step.action), _parts_signature(step))


def _linked(left: Step, right: Step, overlap_s: float) -> bool:
    """Same-key link: windows overlap, touch, or gap within the overlap."""
    gap = max(left.start_s, right.start_s) - min(left.end_s, right.end_s)
    return gap <= overlap_s


def stitch_bundles(
    chunk_bundles: Sequence[tuple[ChunkWindow, StepBundle]],
    *,
    overlap_s: float,
    alignment_mode: str = step_alignment.DEFAULT_ALIGNMENT_MODE,
    generic_tokens: frozenset[str] = step_alignment.GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = step_alignment.VERB_SYNONYM_GROUPS,
    negation_words: frozenset[str] = step_alignment.NEGATION_WORDS,
    qualifiers: frozenset[str] = step_alignment.QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = step_alignment.PART_SYNONYM_GROUPS,
    transition_verbs: frozenset[str] = step_alignment.TRANSITION_VERBS,
    span_mode: str = DEFAULT_SPAN_MODE,
    verb_less_overlap_fraction: float = (
        step_alignment.DEFAULT_VERB_LESS_OVERLAP_FRACTION
    ),
) -> StitchedSteps:
    """Stitch per-chunk consensus bundles onto the original timeline.

    Each step offsets by its window start. Default ``align`` mode merges
    steps the shared monotonic alignment phases together
    (:mod:`app.step_alignment`) into their union (fields follow the
    earlier sighting); ``exact`` mode merges only same-semantic-key
    steps as a configured rollback. Steps that honestly overlap are both
    kept and flagged. Output sorts by window and renumbers from 1.
    """
    if alignment_mode not in step_alignment.ALIGNMENT_MODES:
        raise ValueError(
            "stitch alignment_mode must be one of "
            f"{', '.join(step_alignment.ALIGNMENT_MODES)}, got {alignment_mode!r}"
        )
    if span_mode not in SPAN_MODES:
        raise ValueError(
            "stitch span_mode must be one of "
            f"{', '.join(SPAN_MODES)}, got {span_mode!r}"
        )
    if not isinstance(generic_tokens, frozenset):
        raise ValueError(
            "stitch generic_tokens must be a frozenset, "
            f"got {type(generic_tokens).__name__}"
        )
    step_alignment.normalize_generic_tokens(generic_tokens)
    if not isinstance(verb_synonyms, tuple) or any(
        not isinstance(group, frozenset) for group in verb_synonyms
    ):
        raise ValueError(
            "stitch verb_synonyms must be a tuple of frozensets, "
            f"got {type(verb_synonyms).__name__}"
        )
    step_alignment.normalize_verb_synonyms(verb_synonyms)
    if not isinstance(negation_words, frozenset):
        raise ValueError(
            "stitch negation_words must be a frozenset, "
            f"got {type(negation_words).__name__}"
        )
    step_alignment.normalize_negation_words(negation_words)
    if not isinstance(qualifiers, frozenset):
        raise ValueError(
            "stitch qualifiers must be a frozenset, "
            f"got {type(qualifiers).__name__}"
        )
    step_alignment.normalize_qualifiers(qualifiers)
    if not isinstance(part_synonyms, tuple) or any(
        not isinstance(group, frozenset) for group in part_synonyms
    ):
        raise ValueError(
            "stitch part_synonyms must be a tuple of frozensets, "
            f"got {type(part_synonyms).__name__}"
        )
    step_alignment.normalize_part_synonyms(part_synonyms)
    if not isinstance(transition_verbs, frozenset):
        raise ValueError(
            "stitch transition_verbs must be a frozenset, "
            f"got {type(transition_verbs).__name__}"
        )
    step_alignment.normalize_transition_verbs(transition_verbs)
    step_alignment.normalize_verb_less_overlap_fraction(
        verb_less_overlap_fraction
    )
    if not chunk_bundles:
        raise ValueError("stitch_bundles needs at least one chunk bundle")
    shifted: list[tuple[int, Step]] = []
    for chunk_index, (window, bundle) in enumerate(chunk_bundles):
        if not bundle.steps:
            raise ValueError(f"chunk {chunk_index} bundle has no steps")
        for step in bundle.steps:
            shifted.append(
                (
                    chunk_index,
                    Step(
                        number=step.number,
                        start_s=step.start_s + window.start_s,
                        end_s=step.end_s + window.start_s,
                        action=step.action,
                        tools=step.tools,
                        parts=step.parts,
                        hazards=step.hazards,
                        time_spread_s=step.time_spread_s,
                        time_needs_check=step.time_needs_check,
                        kind=step.kind,
                    ),
                )
            )
    if alignment_mode == "exact":
        keys = [_semantic_key(step) for _, step in shifted]
        parent = list(range(len(shifted)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(shifted)):
            for j in range(i + 1, len(shifted)):
                if keys[i] == keys[j] and _linked(shifted[i][1], shifted[j][1], overlap_s):
                    parent[find(i)] = find(j)
        grouped: dict[int, list[int]] = {}
        for i in range(len(shifted)):
            grouped.setdefault(find(i), []).append(i)
        groups = list(grouped.values())
    else:
        keys = None
        groups = step_alignment.cluster_windows(
            [step_alignment.window_of(step) for _, step in shifted],
            tolerance_s=overlap_s,
            generic_tokens=generic_tokens,
            verb_synonyms=verb_synonyms,
            negation_words=negation_words,
            qualifiers=qualifiers,
            part_synonyms=part_synonyms,
            transition_verbs=transition_verbs,
            phases_mode=span_mode == "contiguous",
            verb_less_overlap_fraction=verb_less_overlap_fraction,
            run_of=[chunk for chunk, _ in shifted],
        )
    merged: list[Step] = []
    merged_keys: list[object] = []
    merged_origins: list[frozenset[int]] = []
    for members in groups:
        members.sort(key=lambda i: (shifted[i][1].start_s, shifted[i][0]))
        first = shifted[members[0]][1]
        start = min(shifted[i][1].start_s for i in members)
        end = max(shifted[i][1].end_s for i in members)
        merged_origins.append(frozenset(shifted[i][0] for i in members))
        merged.append(
            Step(
                number=0,
                start_s=start,
                end_s=end,
                action=first.action,
                tools=first.tools,
                parts=first.parts,
                hazards=first.hazards,
                time_spread_s=max(shifted[i][1].time_spread_s for i in members),
                time_needs_check=any(
                    shifted[i][1].time_needs_check for i in members
                ),
                kind=first.kind,
            )
        )
        if keys is None:
            merged_keys.append(
                (
                    step_alignment.normalize_action(first.action),
                    tuple(
                        sorted(
                            step_alignment.part_tokens(
                                part.phrase for part in first.parts
                            )
                        )
                    ),
                )
            )
        else:
            merged_keys.append(keys[members[0]])
    flagged: set[int] = set()
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            if merged_keys[i] != merged_keys[j] and step_alignment.windows_disagree(
                merged[i].start_s, merged[i].end_s, merged[j].start_s, merged[j].end_s
            ):
                flagged.add(i)
                flagged.add(j)
    order = sorted(
        range(len(merged)),
        key=lambda i: (merged[i].start_s, merged[i].end_s, merged_keys[i]),
    )
    steps = tuple(
        Step(
            number=position,
            start_s=merged[i].start_s,
            end_s=merged[i].end_s,
            action=merged[i].action,
            tools=merged[i].tools,
            parts=merged[i].parts,
            hazards=merged[i].hazards,
            time_spread_s=merged[i].time_spread_s,
            time_needs_check=True if i in flagged else merged[i].time_needs_check,
            kind=merged[i].kind,
        )
        for position, i in enumerate(order, start=1)
    )
    guesses = [bundle.device_guess for _, bundle in chunk_bundles]
    counts = Counter(guesses)
    top = max(counts.values())
    device_guess = next(guess for guess in guesses if counts[guess] == top)
    confidence = float(
        statistics.median([bundle.confidence for _, bundle in chunk_bundles])
    )
    return StitchedSteps(
        steps=steps,
        device_guess=device_guess,
        confidence=confidence,
        origins=tuple(merged_origins[i] for i in order),
        fewest_runs=min(bundle.source_runs for _, bundle in chunk_bundles),
    )


def require_contiguous_cover(
    steps: Sequence[_Timed], extent_s: float, *, tolerance_s: float
) -> None:
    """Enforce that ``steps`` tile ``[0, extent_s)`` within ``tolerance_s``.

    Every start must lie inside the extent, starts must strictly
    increase, consecutive windows must meet within the tolerance (a
    wider gap is a missing phase, a wider overlap a misplaced one),
    and the first start and last end must meet the extent edges
    within the tolerance. Anything else raises :class:`ContiguityError`
    (fail closed); a difference of exactly the tolerance passes.
    """
    if (
        isinstance(extent_s, bool)
        or not isinstance(extent_s, (int, float))
        or not math.isfinite(extent_s)
        or not extent_s > 0
    ):
        raise ValueError(f"extent_s must be a finite number > 0, got {extent_s!r}")
    if (
        isinstance(tolerance_s, bool)
        or not isinstance(tolerance_s, (int, float))
        or not math.isfinite(tolerance_s)
        or not tolerance_s >= 0
    ):
        raise ValueError(
            f"tolerance_s must be a finite number >= 0, got {tolerance_s!r}"
        )
    ordered = tuple(steps)
    if not ordered:
        raise ContiguityError("contiguous cover needs at least one step")
    extent = float(extent_s)
    tolerance = float(tolerance_s)
    for position, step in enumerate(ordered, start=1):
        if step.start_s < -tolerance or step.start_s > extent + tolerance:
            raise ContiguityError(
                f"step {position} starts at {step.start_s:g}s, outside"
                f" [0, {extent:g}] beyond the {tolerance:g}s tolerance"
            )
        if not step.start_s < step.end_s:
            raise ContiguityError(
                f"step {position} has no duration:"
                f" {step.start_s:g} to {step.end_s:g}"
            )
    if ordered[0].start_s > tolerance:
        raise ContiguityError(
            f"first step starts at {ordered[0].start_s:g}s, leaving the"
            f" opening before it uncovered beyond the {tolerance:g}s tolerance"
        )
    for left, right, position in zip(
        ordered, ordered[1:], range(2, len(ordered) + 1)
    ):
        if not right.start_s > left.start_s:
            raise ContiguityError(
                f"step {position} starts at {right.start_s:g}s, not after"
                f" step {position - 1} at {left.start_s:g}s"
            )
        seam = right.start_s - left.end_s
        if seam > tolerance:
            raise ContiguityError(
                f"gap of {seam:g}s between step {position - 1} end"
                f" ({left.end_s:g}s) and step {position} start"
                f" ({right.start_s:g}s): a phase is missing"
            )
        if seam < -tolerance:
            raise ContiguityError(
                f"overlap of {-seam:g}s between step {position - 1} end"
                f" ({left.end_s:g}s) and step {position} start"
                f" ({right.start_s:g}s): a phase is misplaced"
            )
    tail = extent - ordered[-1].end_s
    if tail > tolerance:
        raise ContiguityError(
            f"last step ends at {ordered[-1].end_s:g}s, leaving"
            f" {tail:g}s of the {extent:g}s extent uncovered"
        )
    if tail < -tolerance:
        raise ContiguityError(
            f"last step ends at {ordered[-1].end_s:g}s, past the"
            f" {extent:g}s extent beyond the {tolerance:g}s tolerance"
        )


def trim_seam_overlaps(
    steps: Sequence[Step],
    seams: Sequence[tuple[float, float]],
    origins: Sequence[frozenset[int]],
) -> tuple[Step, ...]:
    """End a step where the next starts when the pair crosses a chunk seam.

    ``seams[k]`` is the overlap zone ``(start_s, end_s)`` between chunk
    ``k`` and ``k + 1`` on the original timeline (next window start,
    previous window end); ``origins`` holds each step's source chunk
    indices. Adjacent chunk cuts overlap by design (plus cut slop), so a
    pair is trimmed only when the left step comes from chunk ``k``, the
    right from ``k + 1`` and both edges lie in seam ``k``'s zone; any
    other overlap is left for the cover check to refuse.
    """
    ordered = tuple(steps)
    eps = 1e-6

    def at_seam(left: int, right: int) -> bool:
        return any(
            not origins[left] & origins[right]
            and k in origins[left]
            and k + 1 in origins[right]
            and zone_start - eps <= ordered[right].start_s
            and ordered[left].end_s <= zone_end + eps
            for k, (zone_start, zone_end) in enumerate(seams)
        )

    return tuple(
        replace(ordered[i], end_s=ordered[i + 1].start_s)
        if ordered[i].end_s > ordered[i + 1].start_s > ordered[i].start_s
        and at_seam(i, i + 1)
        else ordered[i]
        for i in range(len(ordered) - 1)
    ) + ordered[-1:]


def derive_contiguous_ends(
    steps: Sequence[Step], duration_s: float, *, tolerance_s: float
) -> tuple[Step, ...]:
    """Derive contiguous ``[start, end)`` phases over ``[0, duration_s)``.

    Coverage and monotonicity are enforced first
    (:func:`require_contiguous_cover`): a missing or misplaced phase
    raises :class:`ContiguityError` instead of stretching a neighbour.
    Then ``start_1 = 0``, ``end_i = start_(i+1)`` and
    ``end_last = duration_s``; model starts (except the first) and every
    non-time field are kept verbatim, and numbering is unchanged.
    """
    require_contiguous_cover(steps, duration_s, tolerance_s=tolerance_s)
    ordered = tuple(steps)
    duration = float(duration_s)
    starts = [0.0] + [step.start_s for step in ordered[1:]]
    return tuple(
        replace(step, start_s=start, end_s=end)
        for step, start, end in zip(
            ordered, starts, [step.start_s for step in ordered[1:]] + [duration]
        )
    )


__all__ = (
    "CHUNK_ARTIFACT_KIND",
    "CHUNK_FILENAME_PATTERN",
    "SPAN_MODES",
    "DEFAULT_SPAN_MODE",
    "ChunkCutError",
    "ChunkWindow",
    "ContiguityError",
    "CutInfo",
    "StitchedSteps",
    "chunk_filename",
    "cut_chunk",
    "derive_contiguous_ends",
    "plan_chunks",
    "probe_cut",
    "real_cut_window",
    "require_contiguous_cover",
    "stitch_bundles",
    "trim_seam_overlaps",
)
