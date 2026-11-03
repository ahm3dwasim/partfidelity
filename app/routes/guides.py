"""Maker-guide page route, part of the guide module.

Serves the one-scroll guide page plus its real-frame images. Images are
addressed by integer index into the guide's own path list, so a request
can never traverse to an arbitrary file. The page inlines the module-owned
``static/guide.css`` so the route needs no static mount.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import parse_qs, quote

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from app.contracts import Guide, Step
from app import db, display_images, export_links, guide_text, proof_links, service
from app.brand_config import fill
from app import reader as R
from app.corrections import (
    CONFIRM,
    UNCONFIRM,
    CorrectionError,
    CorrectionStore,
    apply_corrections,
    confirmations,
    generations,
    validated_op,
)
from app.exports import export_zip_bytes, image_zip_bytes
from app.guide_text import TEXT
from app.guides import CardPart, StepCard, difficulty_line_html, draft_notes_html, folded_text, gated_cards_html, image_pixel_size, render_cards_html, step_cards, step_title, stop_text
from app.reader import is_hard_stop
from app.hazards import step_hazard_blocks
from app.evidence import list_calls
from app.providers import open_readback
from app.routes.calls import calls_link_html, resolve_job_context
from app.routes.steps import placeholder_frame, render_time_controls
from app.settings import Settings
from app.step_notes import load_step_notes
from app import film_config
from app.step_times import format_approx_range

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_PATH = _REPO_ROOT / "templates" / "guide.html"
_CSS_PATH = _REPO_ROOT / "static" / "guide.css"

GuideLoader = Callable[[str], Guide | None]
ClipLoader = Callable[[str], Path | None]


def set_guide_loader(app: FastAPI, loader: GuideLoader | None) -> None:
    """Inject the guide lookup on one application instance.

    Production wires the ledger's storage; tests wire a stub per app. State lives on
    the app, so two instances never share a loader.
    """
    app.state.guide_loader = loader


def set_guide_clip_loader(app: FastAPI, loader: ClipLoader | None) -> None:
    """Inject the guide-clip lookup on one application instance.

    Production resolves the retained clip file by guide id; tests wire a
    stub per app. Pages without a loader render the sourceless player.
    """
    app.state.guide_clip_loader = loader


def set_corrections_store(app: FastAPI, store: CorrectionStore | None) -> None:
    """Inject the correction store on one application instance.

    Without a store the guide page renders no correction control.
    """
    app.state.corrections_store = store


def _cfg(request: Request, name: str):
    """A Settings value, or its dataclass default when the app has no settings."""
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        return getattr(settings, name)
    return Settings.__dataclass_fields__[name].default


def _store(request: Request) -> CorrectionStore | None:
    return getattr(request.app.state, "corrections_store", None)


def _load_guide_or_raise(request: Request, job_id: str, *, fail_corrections: bool = False) -> Guide:
    loader = getattr(request.app.state, "guide_loader", None)  # snapshot
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        with db.connect(settings.db_path) as conn:
            cancelled = conn.execute(
                "SELECT cancel_requested_at FROM uploads WHERE job_id = ?", (job_id,)
            ).fetchone()
        if cancelled is not None and cancelled["cancel_requested_at"] is not None:
            raise HTTPException(status_code=404, detail="guide not found")
    if loader is None:  # concurrent set_guide_loader must not interleave
        raise HTTPException(status_code=503, detail="guide storage not wired")
    guide = loader(job_id)
    if guide is None:
        raise HTTPException(status_code=404, detail="guide not found")
    store = _store(request)
    try:
        return apply_corrections(guide, store.read(job_id), max_chars=store.max_chars) if store else guide
    except CorrectionError as error:
        if fail_corrections:
            raise
        raise HTTPException(status_code=503, detail="corrections unavailable") from error


def _load_clip_or_raise(request: Request, job_id: str) -> Path:
    """Return one guide's playable clip, or raise its honest status.

    The guide loads first, so the clip keeps exactly the guide page's
    visibility: unknown or deleted guides 404 here too, and only a
    guide that renders can serve a clip. The loader resolves the file
    from its DB row; a missing row or file 404s.
    """
    _load_guide_or_raise(request, job_id)
    loader = getattr(request.app.state, "guide_clip_loader", None)  # snapshot
    if loader is None:
        raise HTTPException(status_code=503, detail="clip storage not wired")
    clip = loader(job_id)
    if clip is None or not clip.is_file():
        raise HTTPException(status_code=404, detail="clip not found")
    return clip


def _video_attrs(job_id: str, request: Request) -> str:
    """Return the player attributes for one guide: its clip src, or nothing.

    Guides without a retained clip (pre-change
    guides, deleted clips) keep the bare sourceless player.
    """
    loader = getattr(request.app.state, "guide_clip_loader", None)
    if loader is None:
        return ""
    clip = loader(job_id)
    if clip is None or not clip.is_file():
        return ""
    quoted = quote(job_id, safe="")
    return f' src="/guides/{quoted}/clip" preload="metadata"'


def _is_sample(request: Request, job_id: str) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return settings is not None and service.sample_job_id(settings) == job_id


def _sample_note_html(database=None, job_id: str = "") -> str:
    """The sample's plain provenance line: what it was made from."""
    return (
        '<p class="gd-sample-note" role="note">'
        f'<span class="replay-badge">{html.escape(R.SAMPLE_REPLAY_LABEL)}</span> '
        f"{html.escape(R.SAMPLE_RUN_NOTE)}. Your own guide uses your footage.</p>"
        # What the old reader page offered, kept on the one sample: the catch,
        # the call evidence and the hard-stop example. Sharing is the guide's
        # own downloads below; the sample has no share link.
        '<nav class="gd-sample-links" aria-label="More from the sample">'
        f'<a href="/catch">{html.escape(proof_links.load()["catch_cta"])}</a> '
        f'<a href="{html.escape(proof_links.load()["sample_calls"], quote=True)}">Calls</a> '
        f'<a href="{html.escape(proof_links.load()["hard_stop"], quote=True)}">Battery warning in the sample</a></nav>'
        + proof_links.open_weights_note_html(database, job_id)
    )


def _image_paths(guide: Guide) -> tuple[Path, ...]:
    ordered: list[Path] = []
    for card in step_cards(guide):
        candidates = [card.frame_path]
        candidates.extend(part.labelled_path for part in card.parts)
        candidates.append(card.labelled_path)
        for candidate in candidates:
            if candidate is not None and candidate not in ordered:
                ordered.append(candidate)
    return tuple(ordered)


_TOKEN_RE = re.compile(r"\{\{(JOB_ID|STEP_COUNT|DRAFT_NOTES|CARDS|GUIDE_CSS|CALLS_LINK|NOW_PLAYING|GUIDE_VIDEO_ATTRS|FIX_NOTE|DOWNLOADS|SAFETY|DIFFICULTY|KICKER|SAMPLE_NOTE|TOAST_MS|OBSERVER_TOP|OBSERVER_BOTTOM|OBSERVER_HOLD)\}\}")

def _fix_code(request: Request) -> str:
    query = request.scope.get("query_string", b"").decode("utf-8", "replace")
    return parse_qs(query).get("fix", [""])[0]


def _fix_note_html(code: str, request: Request) -> str:
    """The result of the one correction control, worded in config/guide_text.toml:
    successes render as a status, everything else as an alert."""
    if code not in guide_text.FIX_CODES:
        return ""
    text = html.escape(guide_text.TEXT[f"fix_{code}"].replace(
        "{max}", str(_cfg(request, "correction_phrase_max_chars"))))
    if code in guide_text.FIX_OK:
        return f'<p class="fix-note" id="fix-note" role="status">{text}</p>'
    return f'<p class="fix-note is-error" id="fix-note" role="alert">{text}</p>'


def _part_control_html(
    job_id: str, step_index: int, part_index: int, phrase: str, max_chars: int
) -> str:
    """The correction control for one part: rename it or delete its label."""
    quoted = quote(job_id, safe="")
    safe = html.escape(phrase, quote=True)
    return (
        '<details class="fix-label"><summary>Fix label</summary>'
        f'<form class="fix-form" method="post" action="/guides/{quoted}/labels">'
        f'<input type="hidden" name="step" value="{step_index}">'
        f'<input type="hidden" name="part" value="{part_index}">'
        f'<input type="hidden" name="expect" value="{safe}">'
        '<label>Label name <input type="text" name="phrase"'
        f' value="{safe}" maxlength="{max_chars}" autocomplete="off"></label>'
        '<button type="submit" name="action" value="rename">Rename</button>'
        '<button type="submit" name="action" value="delete" class="fix-delete"'
        ' formnovalidate>Delete label</button>'
        "</form></details>"
    )


def _flag_state(request: Request, job_id: str, store: CorrectionStore | None):
    """(config, {label: second-check Outcome}, {label: (generation, when) the person confirmed},
    {label: current generation}).

    Any failure reading receipts or config means no flag: the guide must never
    break because of a background second opinion.
    """
    try:
        cfg = open_readback.load_config()
        found = open_readback.outcomes(list_calls(resolve_job_context(job_id, request))) if cfg.enabled else {}
        if not found:
            return cfg, {}, {}, {}
        ops = store.read(job_id) if store is not None else []
        gens = generations(request.app.state.guide_loader(job_id), ops,
                           max_chars=_cfg(request, "correction_phrase_max_chars"))
        return cfg, found, confirmations(ops), gens
    except Exception:
        return None, {}, {}, {}


def _flag_id(label: str) -> str:
    return "flag-" + re.sub(r"[^A-Za-z0-9_-]", "-", label)


def _flag_html(
    job_id: str, step_index: int, part_index: int, label: str, phrase: str, gen: int, cfg,
    editable: bool, confirmed_at: str | None,
) -> tuple[str, str]:
    """(id, html): the neutral review note under one flagged label.

    Unconfirmed: the note plus "I checked" button. Confirmed: a plain statement of the
    person's own confirmation plus Undo. Neither claims the label is wrong or right.
    The note is in the Tab order; the form names the label generation it was shown for.
    """
    quoted = quote(job_id, safe="")
    safe = html.escape(phrase, quote=True)
    flag_id = _flag_id(label)
    if confirmed_at is None:
        text, action, button, state = html.escape(cfg.flag_text), "confirm", cfg.flag_confirm_label, ""
    else:
        when = html.escape(confirmed_at, quote=True)
        try:
            shown = datetime.fromisoformat(confirmed_at).astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        except ValueError:
            shown = confirmed_at
        text = f'{html.escape(cfg.flag_confirmed_text)} <time datetime="{when}">{html.escape(shown)}</time>'
        action, button, state = "unconfirm", cfg.flag_undo_label, " is-confirmed"
    out = (
        f'<div class="second-check-flag{state}" id="{flag_id}" role="status" tabindex="0"'
        f' aria-label="{html.escape(cfg.flag_aria_label, quote=True)}">'
        f"<p>{text}</p>"
    )
    if editable:
        out += (
            f'<form method="post" action="/guides/{quoted}/labels">'
            f'<input type="hidden" name="step" value="{step_index}">'
            f'<input type="hidden" name="part" value="{part_index}">'
            f'<input type="hidden" name="expect" value="{safe}">'
            f'<input type="hidden" name="gen" value="{gen}">'
            f'<button type="submit" name="action" value="{action}">{html.escape(button)}</button>'
            "</form>"
        )
    return flag_id, out + "</div>"


def _unavailable_html(label: str, cfg) -> tuple[str, str]:
    """(id, html): the recheck after this label's edit could not run."""
    flag_id = _flag_id(label)
    return flag_id, (
        f'<div class="second-check-flag is-unavailable" id="{flag_id}" role="status" tabindex="0"'
        f' aria-label="{html.escape(cfg.flag_aria_label, quote=True)}">'
        f"<p>{html.escape(cfg.flag_unavailable_text)}</p></div>"
    )


def _now_playing_html(cards: Sequence[StepCard]) -> str:
    """Now-playing block for the first step; empty when there is no step.

    The reader script rewrites both lines whenever the active card
    moves, reading the same title and ``data-window`` text the cards
    render, so the initial lines must match the first card exactly.
    """
    if not cards:
        return ""
    step = cards[0].step
    title = f"{step_title(step)}: {step.action}"
    window = format_approx_range(step.start_s, step.end_s)
    return (
        '<div class="guide-now" aria-live="polite">'
        f'<p class="guide-now-title" id="guide-now-title">{html.escape(title)}</p>'
        f'<p class="guide-now-time" id="guide-now-time">{html.escape(window)}</p>'
        "</div>"
    )


def _safety_html(guide: Guide) -> str:
    """"Before you start": one line per step that carries a warning.

    Only names the steps and their hazard tags; the full fixed warning
    text stays on its step card, never paraphrased here.
    """
    rows = []
    for step in guide.steps:
        tags = [tag for tag, _text in step_hazard_blocks(step)]
        if tags:
            rows.append(
                f'<li><a href="#step-{step.number}">Step {step.number}</a>: '
                f"{html.escape(', '.join(tags))}</li>"
            )
    if not rows:
        return ""
    return (
        '<section class="gd-safety" aria-label="Before you start">'
        '<p class="gd-safety-title">Before you start</p>'
        "<p>Read each warning on its step before you work inside the device.</p>"
        f'<ul>{"".join(rows)}</ul>'
        + (f'<p class="gd-safety-note">{html.escape(TEXT["safety_stop_note"])}</p>' if any(is_hard_stop(s) for s in guide.steps) else "")
        +
        "</section>"
    )


def _stop_html(number: int, kinds: Sequence[str], later: int) -> str:
    """The live stop: a real button that ``/static/reader_stop.js`` wires; with no script the
    later steps simply stay shown beside every warning."""
    title, risk = stop_text(kinds)
    return (
        f'<section class="stop" role="alert" aria-label="Hard stop" data-stop="{number}">'
        f'<p><strong>{html.escape(title)}</strong> {html.escape(risk)} {html.escape(TEXT["stop_line"])}</p>'
        + (f'<p class="stop-folded" data-stop-folded>{html.escape(folded_text(later))}</p>' if later else "")
        + f'<button type="button" class="ack" data-stop-ack>{html.escape(TEXT["stop_cta"])}</button>'
        f'<p class="stop-done" role="status" data-stop-done hidden>{html.escape(TEXT["stop_done"])}</p></section>'
    )


def _render_page(
    job_id: str,
    guide: Guide,
    *,
    video_attrs: str = "",
    request: Request | None = None,
    fix: str = "",
    sample: bool = False,
) -> str:
    paths = _image_paths(guide)
    position = {path: index for index, path in enumerate(paths)}
    quoted = quote(job_id, safe="")
    settings = getattr(request.app.state, "settings", None) if request is not None else None
    token_query = ""
    if settings is not None:
        try:
            token = export_links.get_or_create(
                job_id, hours=settings.export_link_retention_hours, path=settings.db_path
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail="guide not found") from error
        token_query = f"?token={quote(token, safe='')}"

    def image_url(path: Path) -> str:
        return f"/guides/{quoted}/image/{position[path]}"

    def display(path: Path):
        size = image_pixel_size(path)
        if size is None:
            return None  # not a readable PNG: the page keeps the stored file
        return display_images.describe(lambda w: f"{image_url(path)}?w={w}", size)

    def time_controls(card: StepCard) -> str:
        if card.frame_path is not None:
            preview = f'{image_url(card.frame_path)}?w={display_images.cached()["widths"][0]}'  # a thumbnail
        else:
            preview = placeholder_frame(step_title(card.step).lower())
        return render_time_controls(card.step, preview_frame_url=preview)

    store = _store(request) if request is not None else None
    editable = store is not None and store.editable(job_id)
    max_chars = _cfg(request, "correction_phrase_max_chars") if request else 0

    step_index = {step.number: i for i, step in enumerate(guide.steps)}

    def part_controls(step: Step, pos: int, part: CardPart) -> str:
        return _part_control_html(
            job_id, step_index[step.number], pos, part.phrase, max_chars
        )

    flag_cfg, found, confirmed, gens = (
        _flag_state(request, job_id, store) if request is not None else (None, {}, {}, {})
    )

    def part_flags(step: Step, pos: int, part: CardPart) -> tuple[str, str] | None:
        label = part.labelled_path.name if part.labelled_path is not None else None
        outcome = found.get(label) if label is not None else None
        gen = gens.get(label, 0)
        if outcome is None or outcome.gen != gen or outcome.phrase_id != open_readback.phrase_id(part.phrase):
            return None  # nothing about this label as it stands (an edited label starts clean)
        if outcome.state == "unavailable":
            # The first check being unavailable adds nothing; a recheck after an edit says so.
            return _unavailable_html(label, flag_cfg) if gen > 0 else None
        if outcome.state != "disagree":
            return None
        seen = confirmed.get(label)
        when = seen[1] if seen is not None and seen[0] == gen else None
        return _flag_html(
            job_id, step_index[step.number], pos, label, part.phrase, gen, flag_cfg, editable, when
        )

    cards = step_cards(guide)
    step_notes = (
        load_step_notes(service.sample_notes_path(settings), required={5, 8})
        if sample and settings is not None
        else None
    )
    explained = [False, False]  # the visual-change line is said once across the whole page
    cards_html = gated_cards_html(
        cards,
        lambda card: render_cards_html(
            [card],
            image_url,
            display=display,
            hazard_blocks=step_hazard_blocks,
            time_controls=time_controls,
            part_controls=part_controls if editable else None,
            part_flags=part_flags if found else None,
            step_notes=step_notes,
            explained_once=explained,
        ),
        _stop_html,
    )
    first_still = next((c.frame_path for c in cards if c.frame_path is not None), None)
    if video_attrs and first_still is not None:
        # the poster is the guide's own first still (already redacted), as a display copy
        video_attrs += f' poster="{html.escape(image_url(first_still))}?w={display_images.cached()["poster_width"]}&poster=1"'
    template = fill(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    css = _CSS_PATH.read_text(encoding="utf-8")
    # Single pass: inserted values are never rescanned, so untrusted text
    # containing a placeholder keeps it literally.
    values = {
        "JOB_ID": html.escape(job_id),
        "TOAST_MS": str(film_config.cached()["film"]["toast_ms"]),
        "OBSERVER_TOP": str(film_config.cached()["film"]["observer_top_pct"]),
        "OBSERVER_BOTTOM": str(film_config.cached()["film"]["observer_bottom_pct"]),
        "OBSERVER_HOLD": str(film_config.cached()["film"]["observer_hold_ms"]),
        "KICKER": "Sample guide" if sample else "Maker guide",
        "SAMPLE_NOTE": _sample_note_html(getattr(settings, "db_path", None), job_id) if sample else "",
        "STEP_COUNT": str(sum(not step.is_intro for step in guide.steps)),
        "DRAFT_NOTES": proof_links.sample_wording(draft_notes_html(guide)) if sample else draft_notes_html(guide),
        "CARDS": cards_html,
        "GUIDE_CSS": css,
        "CALLS_LINK": calls_link_html(job_id),
        "NOW_PLAYING": _now_playing_html(cards),
        "GUIDE_VIDEO_ATTRS": video_attrs,
        "FIX_NOTE": _fix_note_html(fix, request) if request is not None else "",
        "SAFETY": _safety_html(guide),
        "DIFFICULTY": difficulty_line_html(guide),
        "DOWNLOADS": (
            f'<a class="dl-link" href="/guides/{quoted}/images.zip{token_query}" download>'
            "Download numbered images (.zip)</a> "
            f'<a class="dl-link" href="/guides/{quoted}/export.zip{token_query}" download>'
            "Download offline guide (.zip)</a> "
            '<span class="dl-note">Pictures on this page are resized copies made for faster loading; '
            "the numbered images download holds the full-size originals.</span>"
        ),
    }
    return _TOKEN_RE.sub(lambda match: values[match.group(1)], template)


@router.get("/guides/{job_id}", response_class=HTMLResponse)
def guide_page(job_id: str, request: Request) -> HTMLResponse:
    guide = _load_guide_or_raise(request, job_id)
    return HTMLResponse(
        _render_page(
            job_id,
            guide,
            video_attrs=_video_attrs(job_id, request),
            request=request,
            fix=_fix_code(request),
            sample=_is_sample(request, job_id),
        )
    )


@router.get("/guide/{job_id}", response_class=HTMLResponse)
def guide_page_singular(job_id: str, request: Request) -> HTMLResponse:
    """Singular alias of the stored-guide page (``/guide/sample``'s sibling).

    The app mounts the site router first, so ``/guide/sample`` still
    resolves to the sample flow; every other id renders here.
    """
    guide = _load_guide_or_raise(request, job_id)
    return HTMLResponse(
        _render_page(
            job_id,
            guide,
            video_attrs=_video_attrs(job_id, request),
            request=request,
            fix=_fix_code(request),
            sample=_is_sample(request, job_id),
        )
    )


def _form_int(value: str | None) -> int | None:
    return int(value) if value is not None and value.isdigit() else None


@router.post("/guides/{job_id}/labels")
async def correct_label(job_id: str, request: Request) -> RedirectResponse:
    """The one correction control: rename or delete one part label, or confirm/undo a flag.

    The owner is whoever holds this private guide link. A plain form post
    answers with a redirect back to the guide page; ``fix=<code>`` names the
    result so success and every refusal show in plain words.
    """
    loader = getattr(request.app.state, "guide_loader", None)
    if loader is None:
        raise HTTPException(status_code=503, detail="guide storage not wired")
    if loader(job_id) is None:
        raise HTTPException(status_code=404, detail="guide not found")
    raw = parse_qs(
        (await request.body()).decode("utf-8", "replace"), keep_blank_values=True
    )
    fields = {key: values[0] for key, values in raw.items()}
    code = {"rename": "renamed", "confirm": "confirmed", "unconfirm": "unconfirmed"}.get(
        fields.get("action"), "deleted"
    )
    anchor = "fix-note"
    try:
        store = _store(request)
        if store is None or not store.editable(job_id):
            raise CorrectionError("locked")
        with store.locked(job_id):
            # Reload under the lock so the stale check sees the latest edits; a rename
            # queues its recheck before the lock is released, from what was just written.
            op, base, ops = _apply_edit(request, store, job_id, fields)
            if code == "renamed":
                code = _queue_recheck(request, job_id, store, base, ops)
        if code in ("confirmed", "unconfirmed"):
            anchor = _flag_id(op["label"])  # keyboard focus lands on the note it just changed
    except CorrectionError as error:
        code = error.code
    # The result note sits at the top of the page; land on it, not the step.
    return RedirectResponse(
        f"/guides/{quote(job_id, safe='')}?fix={code}#{anchor}", status_code=303
    )


def _apply_edit(
    request: Request, store: CorrectionStore, job_id: str, fields: dict
) -> tuple[dict, Guide, list[dict]]:
    """Validate one form edit against the guide as it stands, then log it.

    Returns (the logged op, the stored guide before corrections, the whole new log).
    Call it under the correction lock.
    """
    ops = store.read(job_id)
    guide = _load_guide_or_raise(request, job_id, fail_corrections=True)
    base = request.app.state.guide_loader(job_id)
    if base is None:
        raise CorrectionError("gone")
    if len(ops) >= store.max_ops:
        raise CorrectionError("full")
    max_chars = _cfg(request, "correction_phrase_max_chars")
    op = validated_op(
        guide,
        {
            "op": fields.get("action"),
            "step": _form_int(fields.get("step")),
            "part": _form_int(fields.get("part")),
            "expect": fields.get("expect"),
            "phrase": fields.get("phrase"),
            "gen": _form_int(fields.get("gen")),
        },
        max_chars=max_chars,
        gens=generations(base, ops, max_chars=max_chars),
    )
    if op["op"] in (CONFIRM, UNCONFIRM):
        _require_current(request, job_id, op, ops)
    store.write(job_id, [*ops, op])
    return op, base, [*ops, op]


def _require_current(request: Request, job_id: str, op: dict, ops: list[dict]) -> None:
    """Confirm needs a current disagreement not yet confirmed; undo needs a current
    confirmation. Anything else came from an out-of-date page: ``stale``."""
    try:
        cfg = open_readback.load_config()
        outcome = open_readback.outcomes(list_calls(resolve_job_context(job_id, request))).get(op["label"])
    except Exception as error:
        raise CorrectionError("failed") from error
    disagrees = (
        cfg.enabled and outcome is not None and outcome.gen == op["gen"] and outcome.state == "disagree"
        and outcome.phrase_id == open_readback.phrase_id(op["phrase"])
    )
    held = confirmations(ops).get(op["label"])
    confirmed = held is not None and held[0] == op["gen"]
    if not disagrees or confirmed != (op["op"] == UNCONFIRM):
        raise CorrectionError("stale")


def _live_second_check(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return settings is not None and settings.providers.mode == "live"


def _queue_recheck(
    request: Request, job_id: str, store: CorrectionStore, base: Guide, ops: list[dict]
) -> str:
    """After a rename, queue the background second check for that one label only, at its
    new generation; returns the result code.

    Same conditions as the pipeline's (live providers; the check enabled in config).
    Never raises: a rename must not fail because a background opinion could not start.
    A check that could not be queued is recorded as "not available" for that generation
    and reported as ``renamed_unchecked``, never as a plain success.
    """
    log = logging.getLogger(__name__)
    if not _live_second_check(request):
        return "renamed"
    op = ops[-1]
    max_chars = _cfg(request, "correction_phrase_max_chars")
    guide = apply_corrections(base, ops, max_chars=max_chars)
    decision = next(
        (d for d in guide.labels if (d.slot.step_index, d.slot.part_index) == (op["step"], op["part"])),
        None,
    )
    if decision is None or not open_readback.eligible(decision):
        return "renamed"  # no label image: nothing the second check could look at
    label = decision.provisional.rendered_image_path.name
    gen = generations(base, ops, max_chars=max_chars)[label]
    try:
        if not open_readback.load_config().enabled:
            return "renamed"
        ctx = resolve_job_context(job_id, request)
        if open_readback.start(ctx, [decision], gens={label: gen}, budget=store) is not None:
            return "renamed"
    except Exception as error:
        log.info("second-check recheck not queued: %s", type(error).__name__)
    try:
        open_readback.record_unavailable(resolve_job_context(job_id, request), decision, gen)
    except Exception as error:
        log.warning("second-check unavailable state not recorded: %s", type(error).__name__)
    return "renamed_unchecked"


def _zip_response(data: bytes, name: str) -> Response:
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/guides/{job_id}/images.zip")
def guide_images_zip(job_id: str, request: Request, token: str = "") -> Response:
    """The user's own numbered copy of the step images, in step order."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not export_links.resolve(token, job_id, path=settings.db_path):
        raise HTTPException(status_code=404, detail="export link not found")
    guide = _load_guide_or_raise(request, job_id)
    data = image_zip_bytes(
        guide, entry_pattern=_cfg(request, "image_zip_entry_pattern")
    )
    return _zip_response(data, _cfg(request, "image_zip_name"))


@router.get("/guides/{job_id}/export.zip")
def guide_export_zip(job_id: str, request: Request, token: str = "") -> Response:
    """The follower export: one offline ``guide.html`` plus ``images/``."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not export_links.resolve(token, job_id, path=settings.db_path):
        raise HTTPException(status_code=404, detail="export link not found")
    guide = _load_guide_or_raise(request, job_id)
    data = export_zip_bytes(
        guide,
        html_name=_cfg(request, "export_html_name"),
        images_dir=_cfg(request, "export_images_dir"),
        frame_pattern=_cfg(request, "export_frame_pattern"),
        label_pattern=_cfg(request, "export_label_pattern"),
        retention_hours=_cfg(request, "export_link_retention_hours"),
    )
    return _zip_response(data, _cfg(request, "export_zip_name"))


@router.get("/guides/{job_id}/clip")
def guide_clip(job_id: str, request: Request) -> FileResponse:
    """Serve one guide's retained clip, with Range seeks.

    Starlette answers ``Range`` with 206 partial bytes (416 past the
    end), so Chrome and Safari scrubbing just works. Inline, never an
    attachment: the guide player is the consumer.
    """
    clip = _load_clip_or_raise(request, job_id)
    return FileResponse(clip, media_type="video/mp4", headers={"Cache-Control": "no-cache"})


@router.get("/guide/{job_id}/clip")
def guide_clip_singular(job_id: str, request: Request) -> FileResponse:
    """Singular alias of the guide-clip route."""
    clip = _load_clip_or_raise(request, job_id)
    return FileResponse(clip, media_type="video/mp4", headers={"Cache-Control": "no-cache"})


@router.get("/guides/{job_id}/image/{index}")
def guide_image(job_id: str, index: int, request: Request, w: int | None = None, poster: bool = False) -> Response:
    """The stored image, or with ``?w=`` a resized display copy of that same file.

    Only the configured widths are served, so the cache cannot be filled with arbitrary sizes.
    ``poster=1`` asks for the JPEG form the clip poster uses (any browser can draw it).
    """
    guide = _load_guide_or_raise(request, job_id)
    paths = _image_paths(guide)
    if index < 0 or index >= len(paths):
        raise HTTPException(status_code=404, detail="image not found")
    path = paths[index]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    # no-cache: a redaction rebuild changes the bytes behind the same URL
    headers = {"Cache-Control": "no-cache"}
    if w is not None:
        if w not in display_images.cached()["widths"]:
            raise HTTPException(status_code=404, detail="image width not found")
        made = display_images.display_bytes(path, w, display_images.poster_cfg() if poster else None)
        if made is not None:
            body, media = made
            if isinstance(body, Path):
                return FileResponse(body, media_type=media, headers=headers)
            return Response(body, media_type=media, headers=headers)
    return FileResponse(path, media_type="image/png", headers=headers)
