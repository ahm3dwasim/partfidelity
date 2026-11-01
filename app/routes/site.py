"""First screen and reader routes (the reader module).

Serves ``/``, the reader views ``/reader/sample`` and ``/reader/hero``
from ``app.reader`` contexts, and a ``/guide/sample`` redirect to the
stored sample guide the pipeline built in replay mode (built on first
hit, then reused; the standalone site app without settings keeps the
reader fallback) for the primary CTA. Pages are server-rendered files under
``templates/`` with ``string.Template`` placeholders (stdlib only: this module
adds no dependency); every interpolated value is HTML-escaped. ``router``
is for integration into the main app; ``create_site_app`` builds a
standalone app with ``/static`` mounted for tests and screenshots.
"""

from __future__ import annotations

from html import escape
import hashlib
import json
import logging
import math
from pathlib import Path
from string import Template
from typing import Mapping

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from app import built_config, disagreement_replay, film_config, proof_links
from app.brand_config import fill
from app import reader as R
from app import service
from app.settings import DEFAULT_MAX_VIDEO_S
from app.contracts import MAX_STEPS, Guide, JSONValue, LabelDecision
from app.guides import decisions_by_slot, image_pixel_size, pin_center_percent
from app.routes.steps import placeholder_frame, render_time_controls

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"

#: Pixel size of the reader's placeholder frames (app.reader inlines 640x360
#: SVG data URIs until real frames ship). Only data-URI frame_src values use
#: it; real frames are measured from their files.
_PLACEHOLDER_FRAME_SIZE = (640, 360)

HEADLINE = "Take it apart. Pass it on."
H1 = HEADLINE  # former name, kept for existing imports
SUB = "Turn a phone video of a repair into a visual guide, one clear step at a time."
# One wording each, used by every template: the sample button and the make button.
SAMPLE_CTA = film_config.cached()["film"]["sample_cta"]
MAKE_CTA = film_config.cached()["film"]["make_cta"]
SECONDARY_CTA = "Try the catch yourself"
# The upload module owns the upload route and its URL is unsettled; /upload is the reader module's
# provisional target, flagged for the maintainer to confirm.
UPLOAD_URL = "/upload"

SAMPLE_TITLE = "Sample guide: fan cable reseat"
HERO_TITLE = "Hero guide: fan replacement"

router = APIRouter()


def duration_text(seconds: float) -> str:
    """'5 minutes' / '90 seconds': the upload limit as people read it."""
    if seconds >= 60 and seconds % 60 == 0:
        n = int(seconds // 60)
        return f"{n} minute" + ("" if n == 1 else "s")
    return f"{int(seconds)} seconds"


def etag_matches(header: str, etag: str) -> bool:
    """RFC 9110 If-None-Match for GET/HEAD: weak comparison over a comma list; ``*`` matches any current representation."""
    tags = [t.strip() for t in header.split(",")]
    if "*" in tags:
        return True
    strip = lambda t: t[2:] if t.startswith("W/") else t
    return strip(etag) in [strip(t) for t in tags if t]


#: path -> (size, mtime_ns, ctime_ns, etag): hashed once, again only when one of them changes (ctime catches a same-size rewrite under a pinned mtime).
_STATIC_ETAGS: dict[str, tuple[int, int, int, str]] = {}


def _content_etag(path: str, stat_result) -> str:
    key = (stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ctime_ns)
    hit = _STATIC_ETAGS.get(path)
    if hit and hit[:3] == key:
        return hit[3]
    etag = '"' + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32] + '"'
    _STATIC_ETAGS[path] = (*key, etag)
    return etag


class CachedStaticFiles(StaticFiles):
    """StaticFiles that sends a long cache lifetime for the home film's frames.

    The film URLs carry ?v=<film.version>, so a new film ships under a new URL.
    """

    def file_response(self, full_path, stat_result, scope, status_code=200):
        if Path(full_path).suffix == ".html":
            # gallery pages are served filled from config/brand.toml, like every template
            return HTMLResponse(fill(Path(full_path).read_text(encoding="utf-8")), status_code=status_code)
        film = film_config.cached()["film"]
        if "/media/teardown/" in Path(full_path).as_posix():
            response = super().file_response(full_path, stat_result, scope, status_code)
            response.headers["Cache-Control"] = f"public, max-age={film['cache_max_age_s']}, immutable"
            return response
        # the image's files all carry a 1970 mtime, so a size+mtime ETag cannot see a same-size change: hash the content
        etag = _content_etag(str(full_path), stat_result)
        headers = {"Cache-Control": film["static_cache_control"], "ETag": etag}
        if scope["method"] in ("GET", "HEAD") and etag_matches(Request(scope).headers.get("if-none-match", ""), etag):
            return Response(status_code=304, headers=headers)
        response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        response.headers.update(headers)
        return response


def add_html_cache_control(app: FastAPI) -> None:
    """Give every HTML response a stable ETag and the configured revalidate header; honour If-None-Match."""

    @app.middleware("http")
    async def html_cache_control(request: Request, call_next):
        response = await call_next(request)
        if not response.headers.get("content-type", "").startswith("text/html"):
            return response
        if "cache-control" not in response.headers:
            response.headers["Cache-Control"] = film_config.cached()["film"]["static_cache_control"]
        if request.method not in ("GET", "HEAD") or response.status_code != 200 or "etag" in response.headers:
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        headers["ETag"] = etag
        if etag_matches(request.headers.get("if-none-match", ""), etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": headers["cache-control"]})
        return Response(body, status_code=200, headers=headers)


def _chapters_html(chapters: list[dict]) -> str:
    """The film's chapter blocks from config/film.toml; non-ASCII text goes out as character references."""

    def esc(text: str) -> str:
        return escape(text).encode("ascii", "xmlcharrefreplace").decode()

    blocks = []
    for i, c in enumerate(chapters, 1):
        lines = [
            '<div class="film-chap" data-chap>',
            f'<p class="film-n">{i:02d} / {esc(c["name"])}</p>',
            f"<h2>{esc(c['title'])}</h2>",
            f"<p>{esc(c['body'])}</p>",
        ]
        if c["cta_text"]:
            lines.append(
                f'<p class="cta-row"><a class="cta-primary" href="{escape(c["cta_url"], quote=True)}">{esc(c["cta_text"])}</a></p>'
            )
        blocks.append("\n".join(lines + ["</div>"]))
    return "\n".join(blocks)


def _parts_html(parts: list[dict]) -> str:
    """The film's part names as a list for screen readers and no-script readers (the moving labels are decoration)."""
    items = "\n".join(f"<li>{escape(p['name'])}</li>" for p in parts)
    return f'<ul class="film-parts" id="film-parts" aria-label="Parts of the laptop, top to bottom">\n{items}\n</ul>'


def _template(name: str) -> Template:
    return Template(fill((TEMPLATES / name).read_text(encoding="utf-8")))


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _home_sample_guide(request: Request) -> Guide | None:
    """The guide the sample buttons open (built on first use), else none.

    The standalone site app (no settings) opens the reader fixture, so it
    describes that one.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return R.sample_guide()
    loader = getattr(request.app.state, "guide_loader", None)
    if loader is None:
        return None
    try:
        # Built from the shipped bundle on first use (no model call), so the
        # teaser is never empty just because nobody opened the sample yet.
        return loader(service.ensure_sample_job(settings))
    except Exception:  # the page still renders; the section just drops its facts
        return None


def _teaser_steps(guide: Guide) -> list:
    """The first step, the one with most warnings (earliest on a tie) and the last, in order."""
    steps = [step for step in guide.steps if not step.is_intro]
    limit = film_config.cached()["film"]["teaser_steps"]
    if len(steps) <= limit:
        return steps
    if limit == 1:
        return steps[:1]
    if limit == 2:
        return [steps[0], steps[-1]]
    riskiest = max(steps[1:-1], key=lambda step: len(step.hazards))
    chosen = {0, steps.index(riskiest), len(steps) - 1}
    for i in range(1, len(steps) - 1):
        if len(chosen) >= limit:
            break
        chosen.add(i)
    return [steps[i] for i in sorted(chosen)]


def _sample_facts_html(guide: Guide | None) -> str:
    """Badges, warnings and three representative steps of the sample, taken from the guide itself."""
    if guide is None or not guide.steps:
        return ""
    count = sum(not step.is_intro for step in guide.steps)
    badges = [f"{count} steps", f"Difficulty: {guide.difficulty}"]
    if guide.draft_reasons:
        badges.append("Draft")
    badge_html = "".join(f'<li class="sample-badge">{escape(b)}</li>' for b in badges)
    hazards = dict.fromkeys(h for step in guide.steps for h in step.hazards)
    warnings = (
        '<p class="sample-warn-head">Warnings in this guide</p><ul class="sample-warns">'
        + "".join(
            f'<li class="sample-warn">{escape(h.replace("_", " ").upper() if len(h) <= 3 else h.replace("_", " ").capitalize())}</li>'
            for h in hazards
        )
        + "</ul>"
        if hazards
        else ""
    )
    why = "".join(
        f'<p class="sample-draft">{escape(r.removeprefix("DRAFT:").strip().capitalize())}.</p>'
        for r in guide.draft_reasons
    )
    steps = "\n".join(
        f'<li value="{step.number}">{escape(step.action.rstrip("."))}</li>' for step in _teaser_steps(guide)
    )
    return (
        f'<ul class="sample-badges" aria-label="About this guide">{badge_html}</ul>\n'
        f"{why}\n{warnings}\n"
        f'<ol class="sample-steps">\n{steps}\n</ol>\n'
        f'<p class="sample-more"><a href="/guide/sample">See all {count} steps in the full guide</a></p>'
    )


def _ack_href(base: str, acknowledged: set[int], number: int) -> str:
    merged = sorted(acknowledged | {number})
    query = "&".join(f"ack={n}" for n in merged)
    return f"{base}?{query}"


def _label_badges(
    labels: list[JSONValue], pin_ids: dict[int, tuple[str, int]] | None = None
) -> str:
    items = []
    for pos, label in enumerate(labels):
        assert isinstance(label, dict)
        status = escape(str(label["status"]))
        phrase = escape(str(label["phrase"]))
        readback = escape(str(label["readback"]))
        pin_attr = ""
        number = ""
        if pin_ids is not None and pos in pin_ids:
            pin_id, pin_number = pin_ids[pos]
            pin_attr = f' data-pin="{pin_id}" tabindex="0"'
            number = f'<span class="pf-n" aria-hidden="true">{pin_number}</span> '
        items.append(
            f"<li{pin_attr}>{number}"
            f'<span class="badge">{status}</span> '
            f"{phrase} <span class=\"hide-simple\">&#183; read-back {readback}</span></li>"
        )
    return "\n".join(items)


def _frame_size_for(
    guide: Guide, index: int, frame_src: JSONValue
) -> tuple[int, int] | None:
    """Pixel size of one reader frame, or ``None`` when unknown.

    Placeholder frames are data URIs at the fixed SVG size; real frames
    are measured from their files. ``None`` drops pins and the exact
    aspect ratio; the card still renders.
    """
    if isinstance(frame_src, str) and frame_src.startswith("data:image/svg+xml"):
        return _PLACEHOLDER_FRAME_SIZE
    if 0 <= index < len(guide.frame_paths):
        path = guide.frame_paths[index]
        if not path.is_absolute():
            path = ROOT / path
        return image_pixel_size(path)
    return None


def _matched_decisions(guide: Guide, index: int) -> list[LabelDecision]:
    """Decisions behind one step's badge rows, in row order.

    Same predicate as ``reader_context`` (detection phrase in the
    step's part phrases), so entry ``k`` here is row ``k`` of the
    badges list. Pins use it only to find their legend row; the claim
    itself is by slot, never by phrase.
    """
    if not 0 <= index < len(guide.steps):
        return []
    phrases = {part.phrase.lower() for part in guide.steps[index].parts}
    return [
        decision
        for decision in guide.labels
        if decision.detection is not None
        and decision.detection.phrase.lower() in phrases
    ]


def _step_pins(
    guide: Guide, index: int, size: tuple[int, int] | None
) -> list[tuple[int | None, int, float, float]]:
    """``(badge row, part position, x %, y %)`` pins for one reader step.

    Decisions are claimed by validated contract slot through the same
    ``decisions_by_slot`` claim the guide renderer uses — same
    inclusion predicate, first wins — never by phrase: repeated
    phrases across steps never share a pin, and failed or not-checked
    decisions get none. The pin number is the 1-based part position,
    like the guide renderer. The badge row is where the claimed
    decision sits in this step's phrase-matched badge rows (``None``
    when it has no row); it keeps legend rows pointing at their pins.
    """
    if size is None or not 0 <= index < len(guide.steps):
        return []
    claimed = decisions_by_slot(guide.labels, guide.steps)
    matched = _matched_decisions(guide, index)
    width, height = size
    pins: list[tuple[int | None, int, float, float]] = []
    for part_index in range(len(guide.steps[index].parts)):
        decision = claimed.get((index, part_index))
        if decision is None or decision.detection is None:
            continue
        pin = pin_center_percent(
            getattr(decision.detection, "box_xyxy", None), width, height
        )
        if pin is None:
            continue
        row = next(
            (
                pos
                for pos, candidate in enumerate(matched)
                if candidate is decision
            ),
            None,
        )
        pins.append((row, part_index + 1, pin[0], pin[1]))
    return pins


def _step_card(
    step: Mapping[str, JSONValue],
    base: str,
    acknowledged: set[int],
    *,
    guide: Guide | None = None,
    guide_index: int = 0,
) -> str:
    number = int(str(step["number"]))
    display = escape(str(step["display"]))
    full = escape(str(step["full"]))
    long_step = step["long_step"] is True
    stop = step["stop"] is True
    acked = step["acknowledged"] is True
    tools = step["tools"]
    assert isinstance(tools, list)
    chips = "".join(f"<li>{escape(str(t))}</li>" for t in tools)
    chips_html = f'<ul class="chips hide-simple" aria-label="Tools for step {number}">{chips}</ul>' if chips else ""
    labels = step["labels"]
    assert isinstance(labels, list)
    frame_src = step["frame_src"]
    size = _frame_size_for(guide, guide_index, frame_src) if guide is not None else None
    pins = _step_pins(guide, guide_index, size) if guide is not None else []
    pin_ids = {
        row: (f"pin-{number}-{pos}", pos)
        for row, pos, _, _ in pins
        if row is not None
    }
    badges = _label_badges(labels, pin_ids or None)
    badges_html = (
        f'<ul class="badges-list pf-legend pf-filmed-only">\n{badges}\n</ul>'
        if badges
        else ""
    )
    frame_html = ""
    if isinstance(frame_src, str):
        alt = escape(f"Step {number}: {step['display']}")
        ratio = ""
        if size is not None:
            ratio = f' style="aspect-ratio: {size[0]} / {size[1]}"'
        pin_spans = "".join(
            f'<span class="pf-pin" id="pin-{number}-{pos}"'
            f' style="--x:{x:g}%;--y:{y:g}%;" aria-hidden="true">{pos}</span>'
            for _row, pos, x, y in pins
        )
        frame_html = (
            f"<figure>"
            f'<div class="pf-frame is-loading"{ratio}>'
            f'<img src="{escape(frame_src)}" alt="{alt}" loading="lazy">'
            f"{pin_spans}</div>"
            f"<figcaption>Step {number} frame</figcaption></figure>"
        )
    reveal = ""
    if long_step:
        reveal = (
            '<details class="full"><summary>Full maker sentence</summary>'
            f"<p>{full}</p></details>\n"
        )
        display = f'{display} <span class="long-mark">long step</span>'
    time_html = (
        f'<p class="time hide-simple">&#8776; {_clock(float(str(step["start_s"])))}'
        f"&#8211;{_clock(float(str(step['end_s'])))} &#183; times approximate</p>"
    )
    time_controls_html = ""
    if guide is not None and 0 <= guide_index < len(guide.steps):
        guide_step = guide.steps[guide_index]
        if isinstance(frame_src, str):
            preview = frame_src
        else:  # flagged steps still need a preview target; real frames pass through
            preview = placeholder_frame(f"step {number}")
        time_controls_html = (
            '<div class="pf-filmed-only">'
            f"{render_time_controls(guide_step, preview_frame_url=preview)}"
            "</div>"
        )
    if stop and not acked:
        href = _ack_href(base, acknowledged, number)
        stop_html = (
            '<section class="stop pf-hazard" aria-label="Hard stop">'
            "<p><strong>Hard stop. See a professional before you go on.</strong></p>"
            f"<p>{display}</p>"
            f'<p><a class="ack" href="{escape(href)}">'
            "I understand &#8212; show the next steps</a></p></section>"
        )
        body = (
            f"{time_html}\n{chips_html}\n{frame_html}\n{badges_html}\n"
            f"{time_controls_html}\n{reveal}"
        )
        return (
            f'<article class="step" id="step-{number}">\n<h3>Step {number}</h3>\n'
            f"{stop_html}\n{body}\n</article>"
        )
    ack_note = ""
    if stop and acked:
        ack_note = '<p class="acked">Stop acknowledged.</p>'
    return (
        f'<article class="step" id="step-{number}">\n<h3>Step {number}</h3>\n'
        f"<p>{display}</p>\n{ack_note}\n{time_html}\n{chips_html}\n"
        f"{frame_html}\n{badges_html}\n{time_controls_html}\n{reveal}</article>"
    )


def guide_summary(ctx: Mapping[str, JSONValue]) -> str:
    """'N steps · about M minutes · Difficulty' for the title block.

    Mirrors static/ui/guide-end.js summaryText: the count covers every
    step (including ones hidden behind an unacknowledged stop) and the
    span runs from the first to the last rendered step. Half-up rounding
    matches JavaScript Math.round at .5 minutes (Python round() would
    round to even: 150 s must read 3 minutes on both sides).
    """
    steps = ctx["steps"]
    assert isinstance(steps, list)
    total = len(steps)
    visible = [
        s for s in steps if isinstance(s, dict) and s["visible"] is True
    ]
    span = ""
    if visible:
        first = visible[0]
        last = visible[-1]
        assert isinstance(first, dict) and isinstance(last, dict)
        minutes = max(
            1,
            math.floor(
                (float(str(last["end_s"])) - float(str(first["start_s"])))
                / 60
                + 0.5
            ),
        )
        span = f"about {minutes} minute" + ("" if minutes == 1 else "s")
    parts = [f"{total} step" + ("" if total == 1 else "s")]
    if span:
        parts.append(span)
    difficulty = str(ctx["difficulty"])
    if difficulty:
        parts.append(difficulty)
    return " · ".join(parts)


def _plate_html(ctx: Mapping[str, JSONValue]) -> str:
    """Side plate: the guide's own first rendered frame, server-side.

    The figure is a decorative duplicate of a frame already on the page,
    so it stays aria-hidden; its alt mirrors that frame's alt so static
    alt-text checks still see a step-named image.
    """
    steps = ctx["steps"]
    assert isinstance(steps, list)
    for step in steps:
        if not isinstance(step, dict) or step["visible"] is not True:
            continue
        src = step["frame_src"]
        if isinstance(src, str):
            number = int(str(step["number"]))
            alt = f"Step {number}: {step['display']}"
            return (
                '<figure class="ge-plate" aria-hidden="true">'
                f'<img src="{escape(src)}" alt="{escape(str(alt))}">'
                '<figcaption data-copy="guide.plate">'
                "First frame of this guide.</figcaption></figure>"
            )
    return ""


def _norm_share_url(share_url: str | None) -> str:
    """Server-issued share URL, normalized — "" when absent.

    None, empty, and whitespace-only all fail closed; a padded URL is
    stripped so the no-JS markup matches what the script reads (the
    script trims the attribute the same way).
    """
    return (share_url or "").strip()


def share_html(share_url: str | None) -> str:
    """Guide-end share section, server-rendered for the no-JS baseline.

    With a server-issued expiring URL the consent-gated copy row is
    shown and the 24-hour note is a true promise about that URL. With
    no URL — None, empty, or whitespace-only (no share-link backend
    ships yet) — the section fails closed: no copy control, no
    consent, no 24-hour promise — the image download and its note only.
    """
    if not _norm_share_url(share_url):
        return (
            '<section class="ge-share" aria-label="Share this guide">\n'
            '<h2 data-copy="guide.share.title">Share this guide</h2>\n'
            '<div class="ge-share-row">\n'
            '<button class="ge-btn ghost" type="button" data-ge-download '
            'data-copy="guide.share.download">Download images</button>\n'
            "</div>\n"
            '<p class="ge-status" role="status" data-ge-status></p>\n'
            '<p class="ge-fine" data-copy="guide.share.note">'
            "There is no share link for this guide yet. "
            "A downloaded set is yours to keep.</p>\n"
            "</section>"
        )
    url = escape(_norm_share_url(share_url), quote=True)
    return (
        '<section class="ge-share" aria-label="Share this guide">\n'
        '<h2 data-copy="guide.share.title">Share this guide</h2>\n'
        '<label class="ge-consent" data-copy="guide.share.consent">'
        '<input type="checkbox" data-ge-consent> '
        "<span>This is my own repair. I agree to share it for 24 hours.</span>"
        "</label>\n"
        '<div class="ge-share-row">\n'
        f'<input class="ge-share-link" data-ge-link readonly value="{url}" '
        'aria-label="24-hour link to this guide" '
        'data-copy="guide.share.link">\n'
        '<button class="ge-btn primary" type="button" data-ge-copy '
        'data-copy="guide.share.copy" disabled>Copy 24-h link</button>\n'
        '<button class="ge-btn ghost" type="button" data-ge-download '
        'data-copy="guide.share.download" disabled>Download images</button>\n'
        "</div>\n"
        '<p class="ge-status" role="status" data-ge-status></p>\n'
        '<p class="ge-fine" data-copy="guide.share.note">'
        "The link and our copies die after 24 hours. "
        "A downloaded set is yours to keep.</p>\n"
        "</section>"
    )


def _render_reader(
    title: str,
    heading: str,
    base: str,
    acknowledged: set[int],
    notice_html: str,
    cross_link_html: str,
    share_url: str | None = None,
) -> str:
    guide = R.sample_guide() if base == R.READER_SAMPLE_URL else R.hero_guide()
    ctx = R.reader_context(guide, acknowledged)
    steps = ctx["steps"]
    assert isinstance(steps, list)
    cards = []
    for index, step in enumerate(steps):
        assert isinstance(step, dict)
        if step["visible"] is True:
            cards.append(
                _step_card(step, base, acknowledged, guide=guide, guide_index=index)
            )
    hidden = int(str(ctx["hidden_after_stop"]))
    locked_html = ""
    if hidden:
        first_hidden = next(
            int(str(s["number"]))
            for s in steps
            if isinstance(s, dict) and s["visible"] is not True
        )
        last = int(str(steps[-1]["number"])) if steps else first_hidden
        locked_html = (
            f'<p class="locked" role="note">Steps {first_hidden}&#8211;{last} '
            "stay hidden until you acknowledge the stop above.</p>"
        )
    tools = ctx["tools"]
    assert isinstance(tools, list)
    # Checkboxes are the no-JS baseline; guide-end.js only adds persistence.
    tools_html = "\n".join(
        '<li><label class="ge-check" data-copy="guide.tools.item">'
        f'<input type="checkbox" value="{escape(str(t))}">'
        f"<span>{escape(str(t))}</span></label></li>"
        for t in tools
    )
    unmatched = ctx["labels_unmatched"]
    assert isinstance(unmatched, list)
    unmatched_html = ""
    if unmatched:
        unmatched_html = (
            '<section class="unmatched pf-filmed-only" aria-label="Other labels">'
            "<h2>Other labels</h2>"
            f'<ul class="badges-list pf-legend">\n{_label_badges(unmatched)}\n</ul></section>'
        )
    draft_html = ""
    if ctx["draft"] is True:
        reasons = ctx["draft_reasons"]
        assert isinstance(reasons, list)
        items = "".join(f"<li>{escape(str(r))}</li>" for r in reasons)
        draft_html = (
            '<section class="draft" aria-label="Draft">'
            "<p><strong>DRAFT.</strong> This guide is not finished.</p>"
            f"<ul>{items}</ul></section>"
        )
    difficulty = str(ctx["difficulty"])
    summary = guide_summary(ctx)
    ge_meta_html = (
        f'<p class="ge-meta" data-copy="guide.meta">{escape(summary)}</p>'
        if summary
        else ""
    )
    issued = _norm_share_url(share_url)
    main_attr = (
        f' data-share-url="{escape(issued, quote=True)}"' if issued else ""
    )
    return _template("reader.html").substitute(
        title=escape(title),
        heading=escape(heading),
        notice_html=notice_html,
        main_attr=main_attr,
        ge_meta_html=ge_meta_html,
        ge_plate_html=_plate_html(ctx),
        difficulty=escape(difficulty),
        difficulty_class=escape(difficulty.lower()),
        verdict=escape(str(ctx["verdict"])),
        draft_html=draft_html,
        tools_html=tools_html,
        steps_html="\n".join(cards),
        locked_html=locked_html,
        labels_unmatched_html=unmatched_html,
        share_html=share_html(share_url),
        cross_link_html=cross_link_html,
    )


def _sample_notice() -> str:
    return (
        f'<p class="provenance"><span class="replay-badge">{escape(R.SAMPLE_REPLAY_LABEL)}</span> '
        f"{escape(R.SAMPLE_RUN_NOTE)} &#183; "
        f'<a href="{R.BUILT_URL}">How this was built</a></p>'
    )


@router.get("/built", response_class=HTMLResponse)
def built() -> str:
    cfg = built_config.load()
    page = {key: escape(value) for key, value in cfg["page"].items() if key != "current_features"}
    url = cfg["page"]["repository_url"]
    page["repository"] = f'<a href="{escape(url, quote=True)}">{escape(url)}</a>' if url else page["repository_pending"]
    label = f'<span class="built-label">{page["history_label"]}</span> '
    order = cfg["page"]["current_features"]
    rows = {
        "summary": cfg["summary"], "tools": cfg["tools"], "licences": cfg["licences"],
        "features": sorted((r for r in cfg["features"] if r.get("current")), key=lambda r: order.index(r["name"])),
        "history": [r for r in cfg["features"] if not r.get("current")],
    }
    for group, group_rows in rows.items():
        page[group] = "".join(
            f'<div class="{"built-history" if row.get("history") else ""}"><dt>{escape(row["name"])}</dt><dd>{label if row.get("history") else ""}{escape(row["text"])}</dd></div>'
            for row in group_rows
        )
    page["history_count"] = len(rows["history"])
    page["header"] = fill((TEMPLATES / "partials/ui/shell-header.html").read_text(encoding="utf-8"))
    return _template("built.html").substitute(page)


@router.get("/disagreement-replay", response_class=HTMLResponse)
def disagreement_replay_page() -> str:
    page = disagreement_replay.render_body()
    page["header"] = fill((TEMPLATES / "partials/ui/shell-header.html").read_text(encoding="utf-8"))
    return _template("replay.html").substitute(page)


@router.get("/disagreement-replay/probe.json", include_in_schema=False)
@router.get("/disagreement-replay/probe-run-1.json", include_in_schema=False)
def disagreement_probe_receipt(request: Request):
    path = disagreement_replay.probe_receipt_path(
        "earlier_receipt_path" if request.url.path.endswith("run-1.json") else "receipt_path")
    if path is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return Response(path.read_bytes(), media_type="application/json", headers={"Cache-Control": "no-store"})


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> str:
    cfg = film_config.cached()
    public = film_config.public(cfg)
    settings = getattr(request.app.state, "settings", None)
    max_video_s = settings.max_video_s if settings is not None else DEFAULT_MAX_VIDEO_S
    links = proof_links.load()
    return _template("index.html").substitute(
        sample_facts_html=_sample_facts_html(_home_sample_guide(request)),
        sample_run_note=escape(R.SAMPLE_RUN_NOTE),
        poster_url=escape(public["poster_url"], quote=True),
        max_video_text=escape(duration_text(max_video_s)),
        sample_cta=SAMPLE_CTA,
        make_cta=MAKE_CTA,
        meta_roles_html=(f'{escape(links["meta_roles"])} <a href="{escape(links["sample_calls"], quote=True)}">'
                         f'{escape(links["meta_roles_link"])}</a> &#183; <a href="/catch">{escape(links["meta_roles_catch_link"])}</a>'
                         f' &#183; <a href="/built">{escape(links["meta_roles_built_link"])}</a>'),
        film_json=json.dumps(public).replace("</", "<\\/"),
        chapters_html=_chapters_html(cfg["chapters"]),
        parts_html=_parts_html(cfg["parts"]),
    )


@router.get(R.GUIDE_SAMPLE_URL, response_class=RedirectResponse, response_model=None)
def guide_sample(request: Request) -> RedirectResponse | HTMLResponse:
    # The primary CTA resolves to the stored sample guide the pipeline
    # built in replay mode (built on first hit, then reused). Without
    # app settings (standalone site app) it keeps the reader fallback.
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return RedirectResponse(url=R.READER_SAMPLE_URL, status_code=303)
    try:
        job_id = service.ensure_sample_job(settings)
    except service.SampleUnavailable as error:
        logging.getLogger(__name__).warning("sample guide unavailable: %s", error)
        return HTMLResponse(_render_reader(
            SAMPLE_TITLE, SAMPLE_TITLE, R.READER_SAMPLE_URL, _parse_ack(request),
            '<p class="fixture-note" role="note">The real sample is unavailable. This illustrated example is not a filmed repair.</p>',
            f'<a href="{R.READER_HERO_URL}">Hard-stop example</a>',
        ))
    return RedirectResponse(url=f"/guides/{job_id}", status_code=303)


@router.get("/reader/sample", response_class=HTMLResponse, response_model=None)
def reader_sample(request: Request) -> str | RedirectResponse:
    # One sample everywhere: the full app sends the old reader link to the
    # stored sample guide (its "I'm fixing this" view is the beginner read).
    notice = _sample_notice()
    if getattr(request.app.state, "settings", None) is not None:
        try:
            service.ensure_sample_job(request.app.state.settings)
        except service.SampleUnavailable as error:
            logging.getLogger(__name__).warning("sample guide unavailable: %s", error)
            notice = '<p class="fixture-note" role="note">The real sample is unavailable. This illustrated example is not a filmed repair.</p>'
        else:
            return RedirectResponse(url=R.GUIDE_SAMPLE_URL, status_code=303)
    acknowledged = _parse_ack(request)
    return _render_reader(
        SAMPLE_TITLE,
        SAMPLE_TITLE,
        R.READER_SAMPLE_URL,
        acknowledged,
        notice,
        f'<a href="{R.READER_HERO_URL}">Hard-stop example</a>',
    )


@router.get("/reader/hero", response_class=HTMLResponse)
def reader_hero(request: Request) -> str:
    acknowledged = _parse_ack(request)
    notice = (
        '<p class="fixture-note" role="note">Example only: these steps and '
        "pictures are illustrations, not a filmed repair. "
        "The battery stop below works the way it does in a real guide.</p>"
    )
    return _render_reader(
        HERO_TITLE,
        HERO_TITLE,
        R.READER_HERO_URL,
        acknowledged,
        notice,
        f'<a href="{R.READER_SAMPLE_URL}">Back to the sample guide</a>',
    )


def _parse_ack(request: Request) -> set[int]:
    found = set()
    for raw in request.query_params.getlist("ack"):
        try:
            number = int(raw)
        except ValueError:
            continue
        # Only real step numbers can acknowledge a stop; anything else is
        # ignored, and re-emitted ack links are built from these ints.
        if 1 <= number <= MAX_STEPS:
            found.add(number)
    return found


def create_site_app() -> FastAPI:
    """Standalone app for tests and screenshots: router + /static."""
    app = FastAPI(title="PartFidelity")
    app.mount("/static", CachedStaticFiles(directory=ROOT / "static", html=False))
    app.include_router(router)
    add_html_cache_control(app)
    return app


__all__ = (
    "HEADLINE",
    "H1",
    "CachedStaticFiles",
    "add_html_cache_control",
    "SAMPLE_CTA",
    "MAKE_CTA",
    "SECONDARY_CTA",
    "SUB",
    "UPLOAD_URL",
    "create_site_app",
    "guide_summary",
    "router",
    "share_html",
)
