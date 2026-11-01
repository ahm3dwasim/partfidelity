"""Steps route: time-control cards over a StepBundle.

The product reader (the reader module) will include `templates/partials/time_controls.html`
per step card; this module renders it today for tests and screenshots via
`GET /steps/preview`, an explicitly labelled harness over illustrative
steps (no model calls). `?replay=1` shows the permanent cached-sample
banner exactly as the replay path renders it.

`refusal_payload` is the non-teardown answer: verdict `not_a_teardown`,
the sample CTA, and zero authorized SAM calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.brand_templates import make_env
from app.brand_config import fill

from app.contracts import PartRequest, Step, StepBundle
from app.spark import StepsRefusal, sam_calls_authorized
from app.stage_cache import SAMPLE_REPLAY_LABEL
from app.step_times import format_approx_range

router = APIRouter(prefix="/steps")

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
PARTIAL = "partials/time_controls.html"
VIDEO_ELEMENT_ID = "guide-video"

_env = make_env(TEMPLATES_DIR)


def placeholder_frame(label: str) -> str:
    """Self-contained SVG placeholder (data URI); real frames come from the SAM module/the guide module."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180">'
        '<rect width="320" height="180" fill="#2b2620"/>'
        '<rect x="14" y="14" width="292" height="152" fill="none" '
        'stroke="#8a7f6a" stroke-width="2"/>'
        f'<text x="160" y="88" fill="#e8e0d0" font-size="22" text-anchor="middle" '
        'font-family="sans-serif">preview frame</text>'
        f'<text x="160" y="116" fill="#b3a68d" font-size="16" text-anchor="middle" '
        'font-family="sans-serif">' + label + "</text>"
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe=":/")  # '#' must stay encoded


def render_time_controls(
    step: Step, *, preview_frame_url: str, video_element_id: str = VIDEO_ELEMENT_ID
) -> str:
    """Render one step's time controls through the owned partial."""
    template = _env.get_template(PARTIAL)
    return template.render(
        number=step.number,
        start=step.start_s,
        end=step.end_s,
        window=format_approx_range(step.start_s, step.end_s),
        flagged=step.time_needs_check,
        preview_frame_url=preview_frame_url,
        video_element_id=video_element_id,
    )


def demo_bundle() -> StepBundle:
    """Illustrative steps for the preview harness (one flagged, one not).

    Contiguous phases tiling ``[0, 150)``: the harness demonstrates the
    product contract, and the time controls keep it (editing step N's
    start also moves step N-1's end).
    """
    return StepBundle(
        device_guess="Cordless hand vacuum",
        confidence=0.82,
        steps=(
            Step(
                number=1,
                start_s=0.0,
                end_s=34.0,
                action="Remove the rear screws",
                tools=("Phillips #0",),
                parts=(PartRequest(phrase="battery", position_hint="rear compartment"),),
                hazards=(),
                time_spread_s=1.5,
                time_needs_check=False,
            ),
            Step(
                number=2,
                start_s=34.0,
                end_s=150.0,
                action="Lift out the motor assembly",
                tools=(),
                parts=(PartRequest(phrase="fan", position_hint="motor housing"),),
                hazards=("Sharp edges",),
                time_spread_s=8.0,
                time_needs_check=True,
            ),
        ),
        source_runs=3,
        timestamps_approximate=True,
    )


def require_preview_contiguous(bundle: StepBundle) -> None:
    """Refuse to preview a bundle whose phases are not contiguous.

    The harness demonstrates the product contract, and the client-side
    controls maintain it by exact attribute equality (step N-1's
    ``data-end`` is always step N's ``data-start``), so the render
    requires exact seams: ``start_1 == 0`` and ``end_i == start_(i+1)``.
    A sparse bundle fails loudly here instead of previewing a guide
    the controls cannot keep gap-free.
    """
    steps = bundle.steps
    if not steps:
        raise ValueError("preview needs at least one step")
    if steps[0].start_s != 0:
        raise ValueError(
            f"preview step 1 must start at 0, got {steps[0].start_s:g}"
        )
    for left, right in zip(steps, steps[1:]):
        if left.end_s != right.start_s:
            raise ValueError(
                f"preview steps {left.number} and {right.number} are not"
                f" contiguous: {left.end_s:g} != {right.start_s:g}"
            )


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
@@brand_icons@@
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step times preview</title>
<link rel="stylesheet" href="/static/guide.css">
<style>
  :root { --ground: #f4efe4; --ink: #241f18; --card: #fffdf7; --accent: #8a5a1d; }
  body { margin: 0; background: var(--ground); color: var(--ink);
         font-family: system-ui, sans-serif; font-size: 17px; line-height: 1.5; }
  main { max-width: 560px; margin: 0 auto; padding: 20px 16px 48px; }
  h1 { font-size: 28px; margin: 0 0 4px; }
  .sub { margin: 0 0 16px; color: #5c5344; }
  .replay-banner { background: #241f18; color: #f4efe4; padding: 10px 14px;
                   margin: 0 0 16px; font-size: 15px; }
  video { width: 100%; background: #171410; display: block; }
  .video-note { font-size: 14px; color: #5c5344; margin: 6px 0 20px; }
  .step-card { background: var(--card); padding: 16px; margin: 0 0 16px; }
  .step-card h2 { font-size: 19px; margin: 0 0 4px; }
  .step-meta { font-size: 14px; color: #5c5344; margin: 0 0 8px; }
  .step-window { font-size: 20px; margin: 8px 0; }
  .approx-note { font-size: 14px; color: #5c5344; }
  .check-time { display: inline-block; background: var(--accent); color: #fff;
                font-size: 15px; padding: 2px 10px; margin: 4px 0 8px; }
  .time-preview { width: 100%; display: block; margin: 4px 0 10px; }
  .nudge-row { display: flex; gap: 10px; margin: 0 0 10px; }
  button { font-size: 16px; padding: 10px 16px; border: none;
           background: #241f18; color: #f4efe4; }
  button:active { background: #000; }
</style>
</head>
<body>
<main class="steps-preview">
  <h1>Step times</h1>
  <p class="sub">Preview harness over illustrative steps. Times are approximate.</p>
  {% if replay %}<p class="replay-banner">{{ replay_label }}</p>{% endif %}
  <div class="guide-cols">
  <div class="guide-video">
  <video id="guide-video" controls playsinline></video>
  <div class="guide-now" aria-live="polite">
  <p class="guide-now-title" id="guide-now-title">{{ now_title }}</p>
  <p class="guide-now-time" id="guide-now-time">{{ now_window }}</p>
  </div>
  <p class="video-note">Pause the video, then &ldquo;Set start here&rdquo; takes that time.</p>
  </div>
  <div class="guide-steps">
  {% for card in cards %}
  <section class="step-card" aria-label="Step {{ card.number }}">
    <h2>Step {{ card.number }}: {{ card.action }}</h2>
    <p class="step-meta">{{ card.meta }}</p>
    {{ card.controls | safe }}
  </section>
  {% endfor %}
  </div>
  </div>
</main>
<script src="/static/time_controls.js"></script>
<script src="/static/reader.js" defer></script>
</body>
</html>
"""


def render_preview_page(bundle: StepBundle, *, replay: bool = False) -> str:
    """Full preview page: one card per step through the real partial + JS."""
    require_preview_contiguous(bundle)
    cards: list[dict[str, Any]] = []
    for step in bundle.steps:
        tools = f"Tools: {', '.join(step.tools)}" if step.tools else "No tools"
        hazards = f" · Hazards: {', '.join(step.hazards)}" if step.hazards else ""
        cards.append(
            {
                "number": step.number,
                "action": step.action,
                "meta": f"{tools}{hazards} · run spread {step.time_spread_s:g}s",
                "controls": render_time_controls(
                    step,
                    preview_frame_url=placeholder_frame(f"step {step.number}"),
                ),
            }
        )
    template = _env.from_string(fill(_PAGE))
    first = bundle.steps[0]
    # _PAGE autoescapes (select_autoescape defaults string templates on),
    # so the now-playing lines go in raw like the card fields above.
    return template.render(
        cards=cards,
        replay=replay,
        replay_label=SAMPLE_REPLAY_LABEL,
        now_title=f"Step {first.number}: {first.action}",
        now_window=format_approx_range(first.start_s, first.end_s),
    )


def refusal_payload(refusal: StepsRefusal) -> dict[str, object]:
    """Non-teardown answer: verdict, sample CTA, zero authorized SAM calls."""
    return {
        "verdict": refusal.verdict,
        "reason": refusal.reason,
        "sample_cta": refusal.sample_cta,
        "sam_calls_authorized": sam_calls_authorized(refusal),
    }


@router.get("/preview", response_class=HTMLResponse)
def preview(replay: bool = False) -> str:
    """Preview harness for tests and screenshots (illustrative steps only)."""
    return render_preview_page(demo_bundle(), replay=replay)
