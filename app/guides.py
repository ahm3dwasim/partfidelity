"""Maker-guide assembly from shared steps, frames, and label decisions, part of the guide module.

``assemble_guide`` builds the guide only from contracts other modules own
(``Step``, real frame paths, the readback module ``LabelDecision``); it invents no label and
runs no second content pipeline. A label reaches the final guide only when
its status is ``part_pixels_unchanged`` and its read-back passed or was not
required. Everything else renders as ``Not checked`` with no label drawn.
Cards claim decisions by contract slot, and keeps ride one atomic finish
set; the hazard module's difficulty, verdict, and draft reasons arrive verbatim in their
frozen contract shape.

Status text is exactly two strings: the configured visual-change line
(config/guide_text.toml, "No visual changes to this part") or ``Not checked``. No "verified", "safe", "accurate", or "fidelity" claim
appears anywhere in this module's output.
"""

from __future__ import annotations

import html
import math
import struct
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.guide_text import DRAFT_ORDER_REASON, STOP_KINDS, TEXT
from app.reader import stop_kinds
from app.step_notes import StepNote
from app.contracts import (
    ExcludedSpan,
    Guide,
    GuideHazardAssessment,
    JobContext,
    LabelDecision,
    PartRequest,
    Step,
)

STATUS_UNCHANGED = TEXT["status_unchanged"]
STATUS_UNCHANGED_HELP = TEXT["status_unchanged_help"]
INTRO_TITLE = TEXT["intro_title"]
INTRO_NOTE = TEXT["intro_note"]
STATUS_NOT_CHECKED = "Not checked"

#: Read-back outcomes that keep a passed label: the read-back said yes, or no read-back
#: was required (high-confidence label).
PASSED_READBACKS = ("yes", "not_run")

#: The difficulty level type, derived from the frozen contract rather than
#: restated: the guide module never writes a level value of its own.
_DIFFICULTY_TYPE = typing.get_type_hints(Guide)["difficulty"]

__all__ = (
    "STATUS_UNCHANGED",
    "STATUS_UNCHANGED_HELP",
    "INTRO_TITLE",
    "step_title",
    "STATUS_NOT_CHECKED",
    "PASSED_READBACKS",
    "CardPart",
    "StepCard",
    "is_included",
    "status_text",
    "label_phrase",
    "assemble_guide",
    "decisions_by_slot",
    "step_cards",
    "format_approx_time",
    "render_cards_html",
    "gated_cards_html",
    "stop_text",
    "folded_text",
    "verdict_text",
    "difficulty_line_html",
    "image_pixel_size",
    "pin_center_percent",
)


@dataclass(frozen=True)
class CardPart:
    """One requested part on a step card with its display status."""

    phrase: str
    status: str  # STATUS_UNCHANGED or STATUS_NOT_CHECKED
    labelled_path: Path | None  # provisional render; None when not checked
    pin: tuple[float, float] | None = None  # (x %, y %) of box center; None hides the pin


@dataclass(frozen=True)
class StepCard:
    """View model for one step card. Stable API for reader rendering."""

    step: Step
    frame_path: Path | None  # selected real frame, aligned to steps by order
    labelled_path: Path | None  # first passed label image for this step
    parts: tuple[CardPart, ...]
    frame_size: tuple[int, int] | None = None  # frame pixels; None sizes the figure by CSS


def is_included(decision: LabelDecision) -> bool:
    """True only for a passed label whose read-back passed or was not required.

    Trust boundary: the decision arrives from another module, so its status
    and read-back must be exactly ``str`` before any Literal comparison —
    an object with a lying ``__eq__`` (even a ``str`` subclass) never
    counts as passed.
    """
    return (
        type(decision.status) is str
        and decision.status == "part_pixels_unchanged"
        and type(decision.readback) is str
        and decision.readback in PASSED_READBACKS
    )


def status_text(decision: LabelDecision | None) -> str:
    """Display status for a decision; a missing decision is not checked."""
    if decision is None:
        return STATUS_NOT_CHECKED
    return STATUS_UNCHANGED if is_included(decision) else STATUS_NOT_CHECKED


def label_phrase(decision: LabelDecision) -> str | None:
    """Phrase a decision labels, from either its detection or its provisional."""
    if decision.detection is not None:
        return decision.detection.phrase
    if decision.provisional is not None:
        return decision.provisional.detection.phrase
    return None


def _exact_part(part: PartRequest) -> bool:
    """True only for an exact ``PartRequest`` with exact-``str`` fields.

    ``PartRequest.__eq__`` compares ``phrase`` and ``position_hint`` with
    ``==``, and a ``str`` subclass could override ``__eq__`` on either
    side (reflected equality tries a subclass first), so both operands
    must be exact before any equality check.
    """
    return (
        type(part) is PartRequest
        and type(part.phrase) is str
        and type(part.position_hint) is str
    )


def _slot_key_for(
    decision: LabelDecision, steps: Sequence[Step]
) -> tuple[int, int] | None:
    """Return the (step, part) key when the decision's slot is exactly valid.

    Trust boundary: the slot arrives from another module, so anything but a
    unique in-range slot whose part EQUALS the requested part (the full
    ``PartRequest``, including ``position_hint``) is rejected — the caller
    drops it from the guide and the keep-set and the row renders
    ``Not checked``. Indices must be exact ``int`` (``bool`` is not an
    index) and both compared parts must be exact ``PartRequest`` values
    with exact-``str`` fields before any equality check, so an object
    with a lying ``__eq__`` never matches.
    """
    slot = decision.slot
    try:
        step_index = slot.step_index
        part_index = slot.part_index
        part = slot.part
    except AttributeError:
        return None
    if (
        type(step_index) is not int
        or type(part_index) is not int
        or not 0 <= step_index < len(steps)
        or not 0 <= part_index < len(steps[step_index].parts)
        or not _exact_part(part)
    ):
        return None
    requested = steps[step_index].parts[part_index]
    if not _exact_part(requested):
        return None
    try:
        matches = part == requested
    except Exception:
        return None
    if not matches:
        return None
    return (step_index, part_index)


def _effective_hazards(hazards: GuideHazardAssessment | None) -> GuideHazardAssessment:
    """Fail-closed hazard projection: only an exact, well-formed assessment passes.

    Trust boundary: the assessment arrives from another module, so anything
    but an exact ``GuideHazardAssessment`` with an exact-``str`` difficulty,
    verdict, and reasons tuple falls back to the contract default — a
    ``str`` or ``tuple`` subclass with a lying dunder never passes. A bad
    value never raises and never corrupts the guide.
    """
    if (
        type(hazards) is GuideHazardAssessment
        and type(hazards.difficulty) is str
        and hazards.difficulty in typing.get_args(_DIFFICULTY_TYPE)
        and type(hazards.verdict) is str
        and hazards.verdict != ""
        and type(hazards.draft_reasons) is tuple
        and len(hazards.draft_reasons) > 0
        and all(
            type(reason) is str and reason != ""
            for reason in hazards.draft_reasons
        )
    ):
        return hazards
    return GuideHazardAssessment()


def assemble_guide(
    ctx: JobContext,
    steps: Sequence[Step],
    frames: Sequence[Path],
    labels: Sequence[LabelDecision],
    *,
    hazards: GuideHazardAssessment | None = None,
    excluded_spans: Sequence[ExcludedSpan] = (),
) -> Guide:
    """Assemble the final guide, finish done with one atomic keep-set.

    ``hazards`` is the hazard module's assessment in its frozen contract shape, received
    verbatim when well-formed; anything absent or malformed fails closed on
    the contract defaults. ``excluded_spans`` rides through verbatim onto
    the guide (the trimmed ranges the UI reports as skipped). Only passed
    labels with exactly valid slots
    reach the guide: duplicates, out-of-range indices, and slots naming
    another part are dropped from the labels and the keep-set, so they
    render ``Not checked`` with nothing kept. The keep-set rides
    ``finish("done", keep_artifact_ids=[...])`` as one atomic set: no
    per-artifact keep runs, so no partial keep set can survive. Any
    assembly exception — including a failed ``done`` transition — attempts
    ``finish("failed")`` so nothing stays kept, while the original error
    still propagates.
    """
    try:
        included = tuple(d for d in labels if is_included(d))
        seen: set[tuple[int, int]] = set()
        valid: list[LabelDecision] = []
        for decision in included:
            key = _slot_key_for(decision, steps)
            if key is None or key in seen:
                continue
            seen.add(key)
            valid.append(decision)
        included = tuple(valid)
        effective = _effective_hazards(hazards)
        guide = Guide(
            job_id=ctx.job_id,
            steps=tuple(steps),
            frame_paths=tuple(frames),
            labels=included,
            difficulty=effective.difficulty,
            verdict=effective.verdict,
            draft_reasons=effective.draft_reasons,
            excluded_spans=tuple(excluded_spans),
        )
        keep_ids = [
            decision.provisional.artifact_id
            for decision in included
            if decision.provisional is not None
        ]
        ctx.finish("done", keep_artifact_ids=keep_ids)
    except Exception:
        try:
            ctx.finish("failed")
        except Exception:
            pass  # terminal-transition failure has no further lane recovery
        raise
    return guide


def image_pixel_size(path: Path) -> tuple[int, int] | None:
    """Pixel ``(width, height)`` of a frame, or ``None`` when unreadable.

    Stdlib PNG IHDR read: no image dependency and no exception escapes.
    Only PNG is attempted; live frames are PNG (see the guide image
    route's media type). ``None`` drops pins and the exact aspect ratio;
    the card still renders.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(24)
    except (OSError, TypeError, ValueError):
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def pin_center_percent(
    box_xyxy: object, width: object, height: object
) -> tuple[float, float] | None:
    """Pin position as frame-size percentages, or ``None`` when unusable.

    The pin sits at the detection-box center. Anything unexpected (wrong
    shape, non-numeric or non-finite coordinates, non-positive frame)
    yields ``None``: the legend row still renders, only the pin is
    dropped. A center past the frame edge clamps so the pin stays visible.
    """
    if (
        type(width) not in (int, float)
        or type(height) not in (int, float)
        or not width > 0  # NaN must also fail
        or not height > 0  # NaN must also fail
    ):
        return None
    try:
        x0, y0, x1, y1 = box_xyxy  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    if any(type(v) not in (int, float) for v in (x0, y0, x1, y1)):
        return None
    if any(not math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None  # NaN/inf never becomes a clamped pin
    x = round((x0 + x1) / 2 / width * 100, 2)
    y = round((y0 + y1) / 2 / height * 100, 2)
    return (min(100.0, max(0.0, x)), min(100.0, max(0.0, y)))


def _pin_for(decision: LabelDecision | None, size: tuple[int, int] | None) -> tuple[float, float] | None:
    """Pin for one claimed decision, or ``None`` without a box or frame size."""
    if decision is None or size is None:
        return None
    detection = decision.detection
    if detection is None:
        return None
    return pin_center_percent(detection.box_xyxy, size[0], size[1])


def decisions_by_slot(
    labels: Sequence[LabelDecision], steps: Sequence[Step]
) -> dict[tuple[int, int], LabelDecision]:
    """Included decisions keyed by validated contract slot; first wins.

    The single claim rule for step cards: a decision is claimed only
    when included (passed label whose read-back passed or was not
    required) and its slot is exactly valid for these steps. The guide
    renderer and the reader pins claim through this helper, so delivery
    order never matters and repeated phrases across steps never share
    one render.
    """
    by_slot: dict[tuple[int, int], LabelDecision] = {}
    for decision in labels:
        if is_included(decision):
            key = _slot_key_for(decision, steps)
            if key is not None:
                by_slot.setdefault(key, decision)
    return by_slot


def step_cards(guide: Guide) -> tuple[StepCard, ...]:
    """Build one card per step from real frames and passed labels only.

    Frames align to steps by order; a step past the end of ``frame_paths``
    gets no image. Each (step, part) position claims the included decision
    carrying its own contract slot (``step_index``, ``part_index``), so
    delivery order never matters and repeated phrases across steps never
    share one render. An uncovered slot, a slot outside the guide, or a
    slot naming another part is ``Not checked``. A repeated slot keeps the
    first decision. Each claimed part also carries its pin position (box
    center as frame-size %) when the frame size is readable.
    """
    by_slot = decisions_by_slot(guide.labels, guide.steps)
    cards: list[StepCard] = []
    for index, step in enumerate(guide.steps):
        frame = guide.frame_paths[index] if index < len(guide.frame_paths) else None
        size = image_pixel_size(frame) if frame is not None else None
        parts: list[CardPart] = []
        for part_index, request in enumerate(step.parts):
            decision = by_slot.get((index, part_index))
            labelled = (
                decision.provisional.rendered_image_path
                if decision is not None and decision.provisional is not None
                else None
            )
            parts.append(
                CardPart(
                    phrase=request.phrase,
                    status=status_text(decision),
                    labelled_path=labelled,
                    pin=_pin_for(decision, size),
                )
            )
        labelled_first = next(
            (part.labelled_path for part in parts if part.labelled_path is not None),
            None,
        )
        cards.append(
            StepCard(
                step=step,
                frame_path=frame,
                labelled_path=labelled_first,
                parts=tuple(parts),
                frame_size=size,
            )
        )
    return tuple(cards)


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def step_title(step: Step) -> str:
    """The step's heading line: the intro has a title, every other step has its number."""
    return INTRO_TITLE if step.is_intro else f"Step {step.number}"


def format_approx_time(step: Step) -> str:
    """Approximate step time range; every step time is approximate."""
    return f"~{_clock(step.start_s)}-{_clock(step.end_s)} (approximate)"


def _status_badge(status: str) -> str:
    cls = "status-ok" if status == STATUS_UNCHANGED else "status-unchecked"
    return f'<span class="{cls}">{html.escape(status)}</span>'


def _display_attrs(disp, data_name: str, *, only_data: bool = False) -> str:
    """Reserved width/height for a display copy, plus its srcset under ``data_name`` for the toggle script."""
    if disp is None:
        return ""
    data = f' {data_name}="{html.escape(disp.srcset)}"'
    if only_data:
        return data
    return f' width="{disp.width}" height="{disp.height}"{data}'


def _picture(disp, img: str) -> str:
    """``img`` (whose src is the stored PNG) inside a <picture> that offers the resized copies first."""
    if disp is None:
        return img
    return (
        f'<picture><source type="{html.escape(disp.type)}" srcset="{html.escape(disp.srcset)}"'
        f' sizes="{html.escape(disp.sizes)}">{img}</picture>'
    )


def render_cards_html(
    cards: Sequence[StepCard],
    image_url: Callable[[Path], str],
    hazard_blocks: Callable[[Step], Sequence[tuple[str, str]]] | None = None,
    time_controls: Callable[[StepCard], str] | None = None,
    part_controls: Callable[[Step, int, CardPart], str] | None = None,
    part_flags: Callable[[Step, int, CardPart], tuple[str, str] | None] | None = None,
    step_notes: Mapping[int, StepNote] | None = None,
    explained_once: list[bool] | None = None,
    display: Callable[[Path], object | None] | None = None,
) -> str:
    """Render step cards as HTML. Every image URL comes from ``image_url``.

    ``display`` maps one still to its resized display copy (``app.display_images.Display``: src, srcset, sizes,
    width, height) or None; without it, or for None, the image is the stored file itself.

    The main figure is always the original frame; every passed part
    exposes its own labelled image with an original/labelled toggle.
    Cards reference only the real frame and provisional render paths they
    were built from; this function adds no image of its own. Pins sit on
    the frame at each claimed part's box center; the parts list doubles
    as the legend and, like the label checks, hides in fixing view.
    ``hazard_blocks`` maps one step to its ``(tag, text)`` warnings; the
    caller owns the safety text, so this module invents none.
    ``time_controls`` renders one card's set-start controls; the caller
    owns the partial, and the block is filmed-only like the label checks.
    ``part_controls`` renders the correction control inside one part's row
    (step, 0-based position, part); the row is filmed-only, so the control
    never shows in fixing view. ``part_flags`` returns an optional quiet
    note for one part as (element id, html), or None; the row is described by that
    element, and the note never changes the label.
    ``step_notes`` maps a step number to what its footage does not show;
    the note sits under the step's action and the stage repeats its label.
    ``explained_once`` is a one-item flag a caller shares across calls so the
    visual-change line appears once in the whole document.
    """
    out: list[str] = []
    explained = explained_once if explained_once is not None else [False]  # said once, where it first appears
    for card in cards:
        step = card.step
        intro = step.is_intro
        out.append(
            f'<article class="card gd-intro" id="intro">' if intro
            else f'<article class="card" id="step-{step.number}">'
        )
        number_badge = "" if intro else f'<span class="gd-no" aria-hidden="true">{step.number}</span>'
        out.append(
            f"<header>{number_badge}<h2>{html.escape(step_title(step))}</h2>"
            f'<p class="time">{html.escape(format_approx_time(step))}</p>'
        )
        if step.time_needs_check and not intro:
            out.append('<p class="check-time">check time</p>')
        out.append("</header>")
        if intro:
            out.append(f'<p class="gd-intro-note">{html.escape(INTRO_NOTE)}</p>')
        out.append(f'<p class="action">{html.escape(step.action)}</p>')
        note = (step_notes or {}).get(step.number)
        if note is not None:
            out.append(
                f'<div class="gd-step-note" role="note" data-note-stage="{html.escape(note.stage)}">'
                f'<p class="gd-step-note-title">{html.escape(note.label)}</p>'
                f"<p>{html.escape(note.text)}</p></div>"
            )
        blocks = hazard_blocks(step) if hazard_blocks is not None else ()
        for tag, text in blocks:
            out.append('<div class="pf-hazard" role="note">')
            out.append(
                f'<p class="pf-hazard-title" data-copy="hazard.title">'
                f"Warning: {html.escape(tag)}</p>"
            )
            out.append(
                f'<p data-copy="hazard.text">{html.escape(text)}</p>'
            )
            out.append("</div>")
        if card.frame_path is not None:
            alt = f"{step_title(step)} frame: {step.action}"
            if card.parts and not intro:
                alt += "; parts: " + ", ".join(p.phrase for p in card.parts)
            ratio = ""
            if card.frame_size is not None:
                width, height = card.frame_size
                ratio = f' style="aspect-ratio: {width} / {height}"'
            frame_disp = display(card.frame_path) if display is not None else None
            out.append(f'<figure class="shot pf-frame is-loading"{ratio}>')
            out.append(
                _picture(
                    frame_disp,
                    f'<img src="{html.escape(image_url(card.frame_path))}"'
                    f' data-original="{html.escape(frame_disp.src if frame_disp else image_url(card.frame_path))}"'
                    f' data-original-png="{html.escape(image_url(card.frame_path))}"'
                    f'{_display_attrs(frame_disp, "data-original-srcset")}'
                    f' alt="{html.escape(alt)}" loading="lazy" decoding="async">',
                )
            )
            for pos, part in enumerate(card.parts):
                if part.pin is None:
                    continue
                x, y = part.pin
                out.append(
                    f'<span class="pf-pin" id="pin-{step.number}-{pos + 1}"'
                    f' style="--x:{x:g}%;--y:{y:g}%;" aria-hidden="true">'
                    f"{pos + 1}</span>"
                )
            out.append("</figure>")
        if step.tools and not intro:
            out.append('<ul class="tools">')
            out.extend(f"<li>{html.escape(tool)}</li>" for tool in step.tools)
            out.append("</ul>")
        if card.parts and not intro:
            if not explained[0] and any(part.status == STATUS_UNCHANGED for part in card.parts):
                explained[0] = True
                out.append(f'<p class="parts-note pf-filmed-only">{html.escape(STATUS_UNCHANGED_HELP)}</p>')
            out.append('<ul class="parts pf-legend pf-filmed-only">')
            for pos, part in enumerate(card.parts):
                flag = part_flags(step, pos, part) if part_flags is not None else None
                ref = (
                    f' data-pin="pin-{step.number}-{pos + 1}" tabindex="0"'
                    if part.pin is not None
                    else ""
                )
                if flag is not None:
                    ref += f' aria-describedby="{html.escape(flag[0], quote=True)}"'
                item = (
                    f"<li{ref}>"
                    f'<span class="pf-n" aria-hidden="true">{pos + 1}</span> '
                    f"{html.escape(part.phrase)} — {_status_badge(part.status)}"
                )
                if flag is not None:
                    item += flag[1]
                if part_controls is not None:
                    item += part_controls(step, pos, part)
                if part.labelled_path is not None:
                    shot_id = f"shot-{step.number}-{pos}"
                    lab_disp = display(part.labelled_path) if display is not None else None
                    lab_src = image_url(part.labelled_path)
                    item += '<figure class="part-shot">'
                    img = f'<img id="{shot_id}" src="{html.escape(lab_src)}"'
                    if card.frame_path is not None:
                        img += f' data-original="{html.escape(frame_disp.src if frame_disp else image_url(card.frame_path))}"'
                        img += f' data-original-png="{html.escape(image_url(card.frame_path))}"'
                        img += _display_attrs(frame_disp, "data-original-srcset", only_data=True)
                    img += (
                        f' data-labelled="{html.escape(lab_disp.src if lab_disp else lab_src)}"'
                        f' data-labelled-png="{html.escape(lab_src)}"'
                        f'{_display_attrs(lab_disp, "data-labelled-srcset")}'
                        f' alt="{html.escape(f"Step {step.number} {part.phrase}: labelled frame")}"'
                        ' loading="lazy" decoding="async">'
                    )
                    item += _picture(lab_disp, img)
                    if card.frame_path is not None:
                        item += (
                            "<figcaption>"
                            f'<button type="button" class="toggle" data-shot="{shot_id}"'
                            ' aria-pressed="false">Show original</button>'
                            "</figcaption>"
                        )
                    item += "</figure>"
                item += "</li>"
                out.append(item)
            out.append("</ul>")
        if time_controls is not None:
            out.append('<div class="pf-filmed-only">')
            out.append(time_controls(card))
            out.append("</div>")
        out.append("</article>")
    return "\n".join(out)


def stop_text(kinds: Sequence[str]) -> tuple[str, str]:
    """(title, risk sentences) for a stop that names these hazards."""
    names = " and ".join(TEXT[f"stop_name_{k}"] for k in kinds)
    return TEXT["stop_title"].replace("{hazard}", names), " ".join(TEXT[f"stop_risk_{k}"] for k in kinds)


def folded_text(count: int) -> str:
    return TEXT["stop_folded_one"] if count == 1 else TEXT["stop_folded_many"].replace("{n}", str(count))


def gated_cards_html(
    cards: Sequence[StepCard],
    render_card: Callable[[StepCard], str],
    stop_html: Callable[[int, Sequence[str], int], str],
) -> str:
    """Cards in order. A stop follows the first step where each hard-stop hazard appears:
    ``stop_html(step number, new hazards, later step count)`` and one ``after-stop`` wrapper
    holding every later card. Later steps with a hazard already stopped on keep their
    warning chips but add no gate. Shared by the live guide and the offline export."""
    out: list[str] = []
    seen: set[str] = set()
    wrappers = 0
    for index, card in enumerate(cards):
        out.append(render_card(card))
        new = [k for k in stop_kinds(card.step) if k not in seen]
        if new:
            seen.update(new)
            later = sum(not c.step.is_intro for c in cards[index + 1:])
            out.append(stop_html(card.step.number, new, later) + '<div class="after-stop">')
            wrappers += 1
    out.append("</div>" * wrappers)
    return "".join(out)


def verdict_text(guide: Guide) -> str:
    """The stored verdict when it is a real one; otherwise plain wording from difficulty and stop hazards."""
    stored = guide.verdict.strip()
    if stored and stored.upper() != "DRAFT":
        return stored
    kinds = {k for step in guide.steps for k in stop_kinds(step)}
    key = guide.difficulty.lower()
    if key == "hard" and kinds:
        key += "".join(f"_{k}" for k in STOP_KINDS if k in kinds)  # exact hazard set, canonical order
    return TEXT[f"verdict_{key}"]


def difficulty_line_html(guide: Guide) -> str:
    """The hazard-derived difficulty and verdict line (live head and export head)."""
    level = html.escape(guide.difficulty)
    return (
        f'<p class="badges"><span class="badge-{level.lower()}">{level}</span> '
        f'<span class="verdict-line">{html.escape(TEXT["difficulty_prefix"])} {html.escape(verdict_text(guide))}</span></p>'
    )


def draft_notes_html(guide: Guide) -> str:
    """Draft notes: the maker sees the DRAFT text; the reader (fixing view) sees only a reader-facing
    note where the draft carries safety meaning. Other draft notes are maker-only."""
    items = []
    for reason in guide.draft_reasons:
        items.append(f'<li class="maker-only">{html.escape(reason)}</li>')
        if reason == DRAFT_ORDER_REASON:
            items.append(f'<li class="reader-only">{html.escape(TEXT["draft_note_battery_reader"])}</li>')
    return f"<ul>{''.join(items)}</ul>" if items else ""
