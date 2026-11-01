"""Settled reader render for guides (the reader module).

Second render of the same ``Guide`` a maker guide shows: same steps, same
real frames, same label decisions. No new pipeline, no model call, no
paraphrase. Short display text drops only safe trailing detail; anything
risky stays intact behind a ``long step`` mark with a one-tap full maker
sentence. Difficulty and verdict are surfaced from the ``Guide`` unchanged:
The hazard module's ordered rule owns them, and this module never recomputes them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import quote

from app.guide_text import TEXT
from app.hazards import stop_reasons
from app.contracts import (
    JSONValue,
    Detection,
    Guide,
    JobId,
    LabelDecision,
    LabelSlot,
    PartRequest,
    Step,
)

MAX_SENTENCES_PER_STEP = 2
MAX_WORDS_PER_SENTENCE = 20

# Provisional hard-stop tag rule (the hazard module settles it): reader-local
# copy of the hazard module's settled rule (battery or glued-screen step) over the step
# hazard tags the guide module assembles into the Guide.

# A dropped trailing sentence must contain none of these (case-insensitive)
# and no digit: verb, object, quantity, hazard, and stop wording is never
# truncated, so shortening with any of them present is refused.
_HAZARD_MARKERS = (
    "batter",
    "adhesive",
    "glue",
    "heat",
    "hot",
    "display",
    "screen",
    "esd",
    "static",
    "sharp",
    "glass",
    "crack",
    "swollen",
    "fire",
    "smoke",
    "volt",
    "mains",
    "shock",
    "acid",
    "chemical",
)
_STOP_MARKERS = (
    "stop",
    "do not",
    "don't",
    "dont",
    "never",
    "must not",
    "cannot",
    "can't",
    "cant",
    "warning",
    "careful",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DIGIT = re.compile(r"\d")

# Sample provenance: the sample page says it is a real guide made from a real
# repair video, and links /built.
SAMPLE_REPLAY_LABEL = "real guide"
SAMPLE_RUN_NOTE = "Made from a real repair video: swapping the internal battery in a Lenovo laptop"
BUILT_URL = "/built"
CALLS_SAMPLE_URL = "/calls/sample"
BATTERY_ORDER_URL = "/calls/sample#battery-order"
CATCH_URL = "/catch"
GUIDE_SAMPLE_URL = "/guide/sample"
READER_SAMPLE_URL = "/reader/sample"
READER_HERO_URL = "/reader/hero"

STATUS_TEXT = {
    "part_pixels_unchanged": TEXT["status_unchanged"],
    "not_checked": "Not checked",
}

READBACK_TEXT = {"yes": "yes", "no": "no", "not_run": "not run"}

# The reader module fixture frames are inlined, not files: there is no static/frames/
# directory. Each placeholder renders the same card the old SVGs showed
# (kind + step number, the guide module note); real Guide frame paths pass through as /...
# below. Drop this map when the fixtures carry the guide module's real frames.
_DATA_URI_SAFE = ":/?@!$&'()*+,;="


def _placeholder_frame_src(kind: str, number: int) -> str:
    label = f"{kind} frame {number}"
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360' "
        f"viewBox='0 0 640 360' role='img' aria-label='{label} placeholder'>"
        "<rect width='640' height='360' fill='#e8e1d1'/>"
        "<text x='320' y='170' text-anchor='middle' "
        f"font-family='Georgia, serif' font-size='44' fill='#211c15'>{label}</text>"
        "<text x='320' y='215' text-anchor='middle' "
        "font-family='Georgia, serif' font-size='20' fill='#5f5747'>"
        "illustration \u2014 not a real photo</text>"
        "</svg>"
    )
    return "data:image/svg+xml," + quote(svg, safe=_DATA_URI_SAFE)


_PLACEHOLDER_FRAME_SRC = {
    f"static/frames/{slug}-step-{n}.svg": _placeholder_frame_src(kind, n)
    for slug, kind, count in (("sample", "Sample", 3), ("hero", "Hero", 4))
    for n in range(1, count + 1)
}


def split_sentences(action: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in _SENTENCE_SPLIT.split(action.strip()))
    return tuple(p for p in parts if p)


def _word_count(sentence: str) -> int:
    return len(sentence.split())


def _dropped_text_is_safe(text: str) -> bool:
    lowered = text.lower()
    if _DIGIT.search(text):
        return False
    return not any(m in lowered for m in _HAZARD_MARKERS + _STOP_MARKERS)


@dataclass(frozen=True)
class ShortenedAction:
    display: str
    full: str
    long_step: bool


def shorten_action(action: str, *, is_stop: bool) -> ShortenedAction:
    """Short display text for one step action.

    Fits two sentences of at most 20 words by dropping only safe trailing
    sentences. The hard-stop step is never shortened. When shortening is
    unsafe the intact action is kept and marked ``long step``.
    """
    full = " ".join(action.split())
    sentences = split_sentences(full)
    if is_stop:
        fits = (
            len(sentences) <= MAX_SENTENCES_PER_STEP
            and all(_word_count(s) <= MAX_WORDS_PER_SENTENCE for s in sentences)
        )
        return ShortenedAction(display=full, full=full, long_step=not fits)
    if len(sentences) <= MAX_SENTENCES_PER_STEP and all(
        _word_count(s) <= MAX_WORDS_PER_SENTENCE for s in sentences
    ):
        return ShortenedAction(display=full, full=full, long_step=False)
    kept = sentences[:MAX_SENTENCES_PER_STEP]
    if any(_word_count(s) > MAX_WORDS_PER_SENTENCE for s in kept):
        # A kept sentence already overflows: cutting mid-phrase is refused.
        return ShortenedAction(display=full, full=full, long_step=True)
    dropped = " ".join(sentences[MAX_SENTENCES_PER_STEP:])
    if dropped and not _dropped_text_is_safe(dropped):
        return ShortenedAction(display=full, full=full, long_step=True)
    return ShortenedAction(display=" ".join(kept), full=full, long_step=True)


def stop_kinds(step: Step) -> tuple[str, ...]:
    """The distinct stop hazards on a step: the canonical ``app.hazards`` rule."""
    return stop_reasons(step)


def is_hard_stop(step: Step) -> bool:
    """True for a battery or glued-screen step (the hazard module stop rule over tags)."""
    return bool(stop_kinds(step))


def label_text(decision: LabelDecision) -> Mapping[str, JSONValue]:
    detection = decision.detection
    return {
        "phrase": detection.phrase if detection is not None else "Part",
        "status": STATUS_TEXT[decision.status],
        "readback": READBACK_TEXT[decision.readback],
    }


def _ordered_tools(steps: tuple[Step, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for step in steps:
        for tool in step.tools:
            if tool not in seen:
                seen.append(tool)
    return tuple(seen)


def _step_labels(
    step: Step, labels: tuple[LabelDecision, ...]
) -> tuple[dict[str, JSONValue], ...]:
    phrases = {part.phrase.lower() for part in step.parts}
    matched = []
    for decision in labels:
        detection = decision.detection
        if detection is not None and detection.phrase.lower() in phrases:
            matched.append(dict(label_text(decision)))
    return tuple(matched)


def _unmatched_labels(
    steps: tuple[Step, ...], labels: tuple[LabelDecision, ...]
) -> tuple[dict[str, JSONValue], ...]:
    phrases = {part.phrase.lower() for step in steps for part in step.parts}
    return tuple(
        dict(label_text(decision))
        for decision in labels
        if decision.detection is None
        or decision.detection.phrase.lower() not in phrases
    )


def reader_context(
    guide: Guide, acknowledged_steps: set[int]
) -> Mapping[str, JSONValue]:
    """JSON-safe reader context for one Guide (reader module).

    Difficulty and verdict are the Guide's own settled values. Steps after
    the first unacknowledged hard stop are marked hidden; the stop step
    itself is always shown intact.
    """
    acknowledged = {int(n) for n in acknowledged_steps}
    stop_numbers = [step.number for step in guide.steps if is_hard_stop(step)]
    first_open_stop = next(
        (n for n in sorted(stop_numbers) if n not in acknowledged), None
    )
    steps: list[dict[str, JSONValue]] = []
    for index, step in enumerate(guide.steps):
        if step.is_intro:  # context above step 1; the reader lists numbered steps only
            continue
        stop = is_hard_stop(step)
        short = shorten_action(step.action, is_stop=stop)
        frame_src: JSONValue = None
        if index < len(guide.frame_paths):
            posix = guide.frame_paths[index].as_posix()
            frame_src = _PLACEHOLDER_FRAME_SRC.get(posix, "/" + posix)
        steps.append(
            {
                "number": step.number,
                "display": short.display,
                "full": short.full,
                "long_step": short.long_step,
                "start_s": step.start_s,
                "end_s": step.end_s,
                "time_approximate": True,
                "tools": list(step.tools),
                "hazards": list(step.hazards),
                "frame_src": frame_src,
                "labels": list(_step_labels(step, guide.labels)),
                "stop": stop,
                "acknowledged": step.number in acknowledged,
                "visible": first_open_stop is None or step.number <= first_open_stop,
            }
        )
    hidden = sum(1 for s in steps if not s["visible"])
    return {
        "job_id": str(guide.job_id),
        "difficulty": guide.difficulty,
        "verdict": guide.verdict,
        "draft": bool(guide.draft_reasons),
        "draft_reasons": list(guide.draft_reasons),
        "tools": list(_ordered_tools(guide.steps)),
        "steps": steps,
        "stop_step": first_open_stop,
        "hidden_after_stop": hidden,
        "labels_unmatched": list(_unmatched_labels(guide.steps, guide.labels)),
    }


def sample_guide() -> Guide:
    """Safe sample contract: exactly three steps, no hazard/battery/adhesive/
    display tag, resolves Easy. Frames are the reader's placeholder cards until a guide
    with real frames is passed in; the reader renders whatever the Guide carries.
    """
    steps = (
        Step(
            number=1,
            start_s=2.0,
            end_s=14.0,
            action="Remove the bottom cover screws with a Phillips screwdriver.",
            tools=("Phillips screwdriver",),
            parts=(PartRequest("bottom cover", "underside"),),
            hazards=(),
            time_spread_s=1.5,
            time_needs_check=False,
        ),
        Step(
            number=2,
            start_s=14.0,
            end_s=26.0,
            action="Ease the bottom cover off with a plastic spudger and set it aside.",
            tools=("Plastic spudger",),
            parts=(PartRequest("cooling fan", "left of the heat pipe"),),
            hazards=(),
            time_spread_s=2.0,
            time_needs_check=False,
        ),
        Step(
            number=3,
            start_s=26.0,
            end_s=44.0,
            action=(
                "Ease the fan cable back into its connector until it clicks. "
                "Route the cable flat along its channel. "
                "The channel keeps the cable clear of the fan blades."
            ),
            tools=("Phillips screwdriver",),
            parts=(PartRequest("fan cable", "above the fan"),),
            hazards=(),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
    )
    labels = (
        LabelDecision(
            slot=LabelSlot(
                step_index=2, part_index=0, part=steps[2].parts[0]
            ),
            detection=None,
            provisional=None,
            checked=False,
            status="not_checked",
            readback="not_run",
        ),
    )
    return Guide(
        job_id=JobId("sample"),
        steps=steps,
        frame_paths=tuple(
            Path(f"static/frames/sample-step-{n}.svg") for n in (1, 2, 3)
        ),
        labels=labels,
        difficulty="Easy",
        verdict="Try it",
        draft_reasons=(),
    )


def hero_guide() -> Guide:
    """Hero fixture: battery hard stop at step 2 blocks steps 3-4 until the
    reader acknowledges it. Synthetic demo data, labelled as such on the
    page; the real hero guide ships with the guide module."""
    battery_detection = Detection(
        phrase="battery connector",
        mask_path=Path("static/frames/hero-mask-2.svg"),
        score=0.82,
        box_xyxy=(8, 8, 40, 24),
    )
    steps = (
        Step(
            number=1,
            start_s=2.0,
            end_s=12.0,
            action="Power down the laptop and unplug the charger.",
            tools=(),
            parts=(),
            hazards=(),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
        Step(
            number=2,
            start_s=12.0,
            end_s=40.0,
            action=(
                "Disconnect the battery connector from the system board before "
                "touching any other part. Lift the connector straight up with a "
                "plastic spudger and keep metal tools away from the board. "
                "A live board can short and destroy the drive."
            ),
            tools=("Plastic spudger",),
            parts=(PartRequest("battery connector", "beside the drive bay"),),
            hazards=("battery",),
            time_spread_s=2.5,
            time_needs_check=False,
        ),
        Step(
            number=3,
            start_s=40.0,
            end_s=58.0,
            action="Remove the fan screws and lift the fan out of its recess.",
            tools=("Phillips screwdriver",),
            parts=(PartRequest("fan screws", "around the fan"),),
            hazards=(),
            time_spread_s=1.5,
            time_needs_check=False,
        ),
        Step(
            number=4,
            start_s=58.0,
            end_s=80.0,
            action=(
                "Seat the new fan in its recess and refit the screws. "
                "Press it gently until it sits flat. "
                "The recess is shaped so it only fits one way."
            ),
            tools=("Phillips screwdriver",),
            parts=(),
            hazards=(),
            time_spread_s=3.0,
            time_needs_check=False,
        ),
    )
    labels = (
        LabelDecision(
            slot=LabelSlot(
                step_index=1, part_index=0, part=steps[1].parts[0]
            ),
            detection=battery_detection,
            provisional=None,
            checked=True,
            status="part_pixels_unchanged",
            readback="not_run",
        ),
        LabelDecision(
            slot=LabelSlot(
                step_index=2, part_index=0, part=steps[2].parts[0]
            ),
            detection=None,
            provisional=None,
            checked=False,
            status="not_checked",
            readback="not_run",
        ),
    )
    return Guide(
        job_id=JobId("hero"),
        steps=steps,
        frame_paths=tuple(Path(f"static/frames/hero-step-{n}.svg") for n in (1, 2, 3, 4)),
        labels=labels,
        difficulty="Hard",
        verdict="See a professional",
        draft_reasons=(),
    )


__all__ = (
    "BATTERY_ORDER_URL",
    "BUILT_URL",
    "CALLS_SAMPLE_URL",
    "CATCH_URL",
    "GUIDE_SAMPLE_URL",
    "MAX_SENTENCES_PER_STEP",
    "MAX_WORDS_PER_SENTENCE",
    "READER_HERO_URL",
    "READER_SAMPLE_URL",
    "SAMPLE_REPLAY_LABEL",
    "SAMPLE_RUN_NOTE",
    "ShortenedAction",
    "hero_guide",
    "is_hard_stop",
    "label_text",
    "reader_context",
    "sample_guide",
    "shorten_action",
    "split_sentences",
)
