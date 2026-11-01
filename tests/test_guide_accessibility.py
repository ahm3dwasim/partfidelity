"""Guide accessibility tests: alt text, keyboard focus, AA color tokens."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.contracts import JobId, LabelDecision, PartRequest, Step
from app.guides import assemble_guide
from app.render import render_label, write_png_rgb
from app.routes.guides import router, set_guide_loader
from test_guide import (
    RecordingContext,
    _decision,
    _detection,
    _frame,
    _mask,
    _slot,
)


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]

    def linear(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(c) for c in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: str, second: str) -> float:
    dark, light = sorted((_relative_luminance(first), _relative_luminance(second)))
    return (light + 0.05) / (dark + 0.05)


def _tokens(css: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", css))


def test_step_cards_have_alt_text_focus_and_aa_tokens(tmp_path):
    ctx = RecordingContext()
    frame = _frame(tmp_path / "frame.png")
    mask_path = _mask(tmp_path / "mask.png", (20, 30, 70, 80))
    provisional = render_label(
        ctx, frame, _detection(mask_path, "fan"), tmp_path / "out.png"
    )
    step = Step(
        number=1, start_s=10.0, end_s=18.0, action="Lift out the fan",
        tools=("screwdriver",),
        parts=(PartRequest(phrase="fan", position_hint="center"),),
        hazards=(), time_spread_s=1.0, time_needs_check=True,
    )
    decision: LabelDecision = _decision(provisional, slot=_slot(0, 0, "fan"))
    guide = assemble_guide(ctx, [step], [frame], [decision])

    app = FastAPI()
    app.include_router(router)
    set_guide_loader(app, lambda job_id: guide if job_id == "job-guide" else None)
    body = TestClient(app).get("/guides/job-guide").text

    alts = re.findall(r'<img[^>]*alt="([^"]*)"', body)
    assert alts, "step cards must render images"
    assert all(alt.strip() for alt in alts)
    assert any("fan" in alt for alt in alts)
    assert 'class="toggle"' in body  # native button: keyboard-focusable toggle

    css = (Path(__file__).resolve().parents[1] / "static" / "guide.css").read_text()
    assert ":focus-visible" in css
    tokens = _tokens(css)
    for pair in (
        ("ink", "paper"),
        ("muted", "paper"),
        ("accent", "card"),
        ("ok-ink", "ok-bg"),
        ("unchecked-ink", "unchecked-bg"),
    ):
        ink, background = (tokens[name] for name in pair)
        ratio = _contrast(ink, background)
        assert ratio >= 4.5, f"{pair}: {ratio:.2f}"
