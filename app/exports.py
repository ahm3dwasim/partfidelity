"""Numbered image zip and the offline follower export.

Both are the user's own copies: built on request from the stored guide,
never kept by the service. Their download links expire after 24 hours. The export
is one ``guide.html`` plus a local ``images/`` folder, opens in reader view
(the maker detail sits behind the view toggle), and references nothing
outside itself: ``assert_self_contained`` fails the build on any external
reference, form, script source, or network call, so the file makes no
request, carries no tracking and calls no model.
"""

from __future__ import annotations

import html
import io
import re
import tomllib
import zipfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from app.contracts import Guide, UPLOAD_PREVIEW_TTL_HOURS
from app.guide_text import TEXT
from app.guides import STATUS_UNCHANGED, StepCard, difficulty_line_html, draft_notes_html, folded_text, gated_cards_html, render_cards_html, step_cards, stop_text
from app.guides import decisions_by_slot
from app.hazards import step_hazard_blocks
from app.render import badge_extent, badge_radius, choose_badge_center, draw_badge, draw_outline, png_rgb_bytes, read_mask, read_png_rgb, restore_part_pixels

CONFIG = Path(__file__).resolve().parents[1] / "config" / "exports.toml"


class ExportsConfigError(RuntimeError):
    """config/exports.toml is missing a key or holds a value a zip cannot store."""


def load_zip_date_time(path: Path | str = CONFIG) -> tuple[int, int, int, int, int, int]:
    """Read the fixed zip member time; a missing or invalid value fails loudly."""
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, ValueError, RecursionError) as exc:  # ValueError covers UnicodeDecodeError, TOMLDecodeError, null bytes
        raise ExportsConfigError(f"exports config cannot be read: {exc}") from exc
    section = raw.get("zip", {})
    if not isinstance(section, dict):
        raise ExportsConfigError("exports config [zip] must be a table")
    value = section.get("date_time")
    if not (isinstance(value, list) and len(value) == 6 and all(type(v) is int for v in value)):
        raise ExportsConfigError("exports config needs [zip] date_time = [year, month, day, hour, minute, second] as integers")
    year, month, day, hour, minute, second = value
    if not 1980 <= year <= 2107:
        raise ExportsConfigError(f"exports config zip year {year} is outside 1980-2107")
    if second % 2:
        raise ExportsConfigError(f"exports config zip second {second} must be even (zip stores 2-second steps)")
    try:
        datetime(year, month, day, hour, minute, second)
    except (ValueError, OverflowError) as exc:
        raise ExportsConfigError(f"exports config zip date_time is not a real time: {exc}") from exc
    return (year, month, day, hour, minute, second)


def _member(name: str) -> zipfile.ZipInfo:
    """Zip entry with the configured time and no dependence on file mtimes."""
    info = zipfile.ZipInfo(name, date_time=load_zip_date_time())
    info.compress_type = zipfile.ZIP_STORED
    return info


__all__ = (
    "ExportsConfigError",
    "assert_self_contained",
    "export_zip_bytes",
    "follower_html",
    "image_zip_bytes",
)

_ROOT = Path(__file__).resolve().parents[1]
_CSS_FILES = (
    "static/ui/tokens.css",
    "static/guide.css",
    "static/ui/guide_step.css",
    "static/ui/guide-end.css",
)
_FONT_FACE = re.compile(r"@font-face\s*\{[^}]*\}")

#: Export-only rules: the checkbox-hack hard stop (works with no script) and
#: the per-step status line, which stays visible in reader view.
_EXPORT_CSS = """
.ack-box { position: absolute; opacity: 0; pointer-events: none; }
.ack-box:not(:checked) ~ .after-stop .card > :not(header):not(.pf-hazard):not(.action),
.ack-box:not(:checked) ~ .after-stop .card > header .time,
.ack-box:not(:checked) ~ .after-stop .card > header .check-time { display: none; }
.ack-box:focus-visible ~ .stop .ack { outline: 3px solid var(--focus, #1b1d1a); outline-offset: 2px; }
.stop .ack { display: inline-block; cursor: pointer; font-weight: 600; text-decoration: underline; }
.ack-box:checked ~ .stop .ack, .ack-box:checked ~ .stop .stop-folded { display: none; }
body.simple .draft-notes .maker-only { display: none; }
body:not(.simple) .draft-notes .reader-only { display: none; }
.export-note { color: var(--muted, #5d5f58); font-size: 0.9375rem; }
.step-status { list-style: none; padding: 0; margin: 0.75rem 1.25rem 1rem; font-size: 0.9375rem; }
.step-status li { margin: 0.25rem 0; }
"""

#: The only script: view toggle and original/labelled swap. No network API.
_EXPORT_JS = """
var box = document.getElementById("simple-view");
if (box) box.addEventListener("change", function () {
  document.body.classList.toggle("simple", box.checked);
});
document.querySelectorAll("button.toggle").forEach(function (button) {
  button.addEventListener("click", function () {
    var img = document.getElementById(button.dataset.shot);
    if (!img || !img.dataset.labelled) return;
    var on = img.getAttribute("src") === img.dataset.labelled;
    img.setAttribute("src", on ? img.dataset.original : img.dataset.labelled);
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.textContent = on ? "Show labelled" : "Show original";
  });
});
"""


def _css() -> str:
    parts = [
        _FONT_FACE.sub("", (_ROOT / name).read_text(encoding="utf-8"))
        for name in _CSS_FILES
    ]
    return "\n".join(parts) + _EXPORT_CSS


def image_zip_bytes(guide: Guide, *, entry_pattern: str) -> bytes:
    """Zip of one image per pictured step, numbered in step order.

    All passed labels for the step are drawn on one frame. A pictured
    step without passed labels keeps its original frame.
    """
    buffer = io.BytesIO()
    passed = decisions_by_slot(guide.labels, guide.steps)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for index, card in enumerate(step_cards(guide)):
            if card.frame_path is None or card.step.is_intro:  # the intro is context, not a numbered step image
                continue
            labels = [passed[index, part] for part in range(len(card.parts)) if (index, part) in passed]
            if not labels:
                archive.writestr(
                    _member(entry_pattern.format(number=card.step.number)),
                    Path(card.frame_path).read_bytes(),
                )
                continue
            width, height, original = read_png_rgb(card.frame_path)
            rendered = bytearray(original)
            detections = [decision.provisional.detection for decision in labels]
            masks = [read_mask(detection.mask_path, width, height) for detection in detections]
            for detection, mask in zip(detections, masks):
                draw_outline(rendered, width, height, detection.box_xyxy, mask)
            union = bytes(any(mask[i] for mask in masks) for i in range(width * height))
            for mask in masks:
                restore_part_pixels(rendered, original, mask)
            occupied = bytearray(width * height)
            radius = badge_radius(width, height)
            extent = badge_extent(radius)
            centers = []
            for detection in detections:
                blocked = bytes(a or b for a, b in zip(union, occupied))
                center = choose_badge_center(width, height, detection.box_xyxy, blocked, fallback=False)
                if center is None:
                    break
                centers.append(center)
                for y in range(max(0, center[1] - extent), min(height, center[1] + extent + 1)):
                    for x in range(max(0, center[0] - extent), min(width, center[0] + extent + 1)):
                        occupied[y * width + x] = 1
            if len(centers) != len(labels):
                # A source-wide mask leaves no safe pixel: put every number below the frame.
                output_width = max(width, 2 * extent + 1)
                output_height = height + len(labels) * (2 * extent + 1)
                background = original[:3]
                rendered = bytearray(background * (output_width * output_height))
                for y in range(height):
                    rendered[y * output_width * 3 : y * output_width * 3 + width * 3] = original[y * width * 3 : (y + 1) * width * 3]
                centers = [(extent, height + extent + number * (2 * extent + 1)) for number in range(len(labels))]
                width, height = output_width, output_height
            for number, center in enumerate(centers, 1):
                draw_badge(rendered, width, height, center, radius, number)
            archive.writestr(
                _member(entry_pattern.format(number=card.step.number)),
                png_rgb_bytes(width, height, bytes(rendered)),
            )
    return buffer.getvalue()


INTRO_IMAGE_NAME = "intro-original.png"  # unnumbered: the intro is not a step


def _image_names(
    cards: tuple[StepCard, ...], frame_pattern: str, label_pattern: str
) -> dict[Path, str]:
    names: dict[Path, str] = {}
    for card in cards:
        number = card.step.number
        if card.frame_path is not None:
            names.setdefault(
                card.frame_path,
                INTRO_IMAGE_NAME if card.step.is_intro else frame_pattern.format(number=number),
            )
        for pos, part in enumerate(card.parts, start=1):
            if part.labelled_path is not None:
                names.setdefault(
                    part.labelled_path,
                    label_pattern.format(number=number, part=pos),
                )
    return names


def _status_list(card: StepCard) -> str:
    if not card.parts or card.step.is_intro:  # the intro is context: no label checks, no step 0
        return ""
    rows = "".join(
        f"<li>{html.escape(part.phrase)} — "
        f'<span class="{"status-ok" if part.status == STATUS_UNCHANGED else "status-unchecked"}">'
        f"{html.escape(part.status)}</span></li>"
        for part in card.parts
    )
    return f'<ul class="step-status" aria-label="Label checks, step {card.step.number}">{rows}</ul>'


def follower_html(
    guide: Guide, *, images_dir: str, frame_pattern: str, label_pattern: str,
    retention_hours: int = UPLOAD_PREVIEW_TTL_HOURS,
) -> tuple[str, dict[Path, str]]:
    """Return the offline page and ``{source path: images/<name>}``."""
    cards = step_cards(guide)
    names = _image_names(cards, frame_pattern, label_pattern)

    def image_url(path: Path) -> str:
        return f"{images_dir}/{names[path]}"

    explained = [False, False]  # one visual-change explanation for the whole export

    def render_card(card: StepCard) -> str:
        chunk = render_cards_html([card], image_url, hazard_blocks=step_hazard_blocks, explained_once=explained)
        chunk = chunk.replace(' loading="lazy"', "")  # local files: render at once
        head, _, _ = chunk.rpartition("</article>")
        return head + _status_list(card) + "</article>"

    def stop_html(number: int, kinds: Sequence[str], later: int) -> str:
        title, risk = stop_text(kinds)
        return (
            f'<input type="checkbox" class="ack-box" id="ack-{number}">'
            f'<section class="stop pf-hazard" role="alert" aria-label="Hard stop">'
            f'<p><strong>{html.escape(title)}</strong> {html.escape(risk)} {html.escape(TEXT["stop_line"])}</p>'
            + (f'<p class="stop-folded">{html.escape(folded_text(later))}</p>' if later else "")
            + f'<label class="ack" for="ack-{number}">{html.escape(TEXT["stop_cta"])}</label></section>'
        )

    body = gated_cards_html(cards, render_card, stop_html)

    tools = "".join(
        f"<li>{html.escape(tool)}</li>"
        for tool in dict.fromkeys(t for s in guide.steps for t in s.tools)
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fix it step by step — offline guide</title>
<style>
{_css()}
</style>
</head>
<body class="pf-theme-guide simple">
<main id="main" class="guide">
<header class="guide-head">
<p class="kicker">Offline guide</p>
<div class="ge-title"><div class="ge-title-text">
<h1>Fix it step by step</h1>
{difficulty_line_html(guide)}
<p class="meta">{sum(not s.is_intro for s in guide.steps)} steps · times are approximate · photos are the original frames</p>
</div></div>
<div class="pf-view-row"><span class="pf-view" role="group" aria-label="Guide view"><label class="pf-view-label"><input type="checkbox" id="simple-view" class="pf-view-box" checked><span class="pf-view-opt" data-filmed>I filmed this</span><span class="pf-view-opt" data-fixing>I&#8217;m fixing this</span></label></span></div>
<div class="draft-notes">{draft_notes_html(guide)}</div>
</header>
<section class="tools" aria-label="Tools for this guide"><h2>Tools</h2><ul class="tool-list">{tools}</ul></section>
<div class="guide-steps">
{body}
</div>
<footer class="guide-foot">
<p>Labels are drawn on copies of your frames. The part itself is never redrawn.</p>
<p class="export-note">This copy works offline. The download link expires after {retention_hours} hours.</p>
</footer>
</main>
<script>{_EXPORT_JS}</script>
</body>
</html>
"""
    assert_self_contained(page, images_dir=images_dir)
    return page, names


_ATTR = re.compile(
    r"""\b(src|href|action|formaction|poster|data|srcset|xlink:href)\s*=\s*(["'])(.*?)\2""",
    re.I | re.S,
)
_URL_FN = re.compile(r"url\(\s*([^)]*)\)", re.I)
_FORBIDDEN = re.compile(
    r"@import|<meta[^>]*http-equiv|<form|<link|<iframe|<video|<audio|<source|<object|<embed|<base"
    r"|fetch\(|XMLHttpRequest|sendBeacon|WebSocket|EventSource|importScripts",
    re.I,
)


def assert_self_contained(page: str, *, images_dir: str) -> None:
    """Raise ``ValueError`` unless the page references only itself and ``images/``."""
    allowed = images_dir.rstrip("/") + "/"
    for _attr, _quote, value in _ATTR.findall(page):
        target = value.strip()
        if not (target.startswith("#") or target.startswith(allowed)):
            raise ValueError(f"external reference in export: {target[:80]!r}")
    for value in _URL_FN.findall(page):
        target = value.strip().strip("'\"")
        if not target.startswith("#"):
            raise ValueError(f"external url() in export: {target[:80]!r}")
    found = _FORBIDDEN.search(page)
    if found:
        raise ValueError(f"forbidden construct in export: {found.group(0)!r}")


def export_zip_bytes(
    guide: Guide,
    *,
    html_name: str,
    images_dir: str,
    frame_pattern: str,
    label_pattern: str,
    retention_hours: int = UPLOAD_PREVIEW_TTL_HOURS,
) -> bytes:
    """``guide.html`` plus ``images/``: the whole offline export in one zip."""
    page, names = follower_html(
        guide,
        images_dir=images_dir,
        frame_pattern=frame_pattern,
        label_pattern=label_pattern,
        retention_hours=retention_hours,
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(_member(html_name), page)
        for source, name in names.items():
            if source.is_file():
                archive.writestr(_member(f"{images_dir}/{name}"), source.read_bytes())
    return buffer.getvalue()
