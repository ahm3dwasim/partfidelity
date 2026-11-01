"""UI guide-step tests: card, pins, switch, badges, zero-network, contrast.

Focused tests for this feature's own files plus the include lines
it adds to guide.html/reader.html. Footage under static/ui/media/local/ is
gitignored by design; these tests never read it, only the committed files.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader

from app.guide_text import TEXT
from app.routes.site import create_site_app

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "ui" / "guide_step.css"
JS = ROOT / "static" / "ui" / "guide_step.js"
PARTIAL = ROOT / "templates" / "partials" / "ui" / "guide_step.html"
SWITCH_PARTIAL = ROOT / "templates" / "partials" / "ui" / "guide_view_switch.html"
GALLERY = ROOT / "static" / "ui" / "gallery" / "guide_step.html"
FONTS = (
    ROOT / "static" / "ui" / "fonts" / "Fraunces-600-normal.woff2",
    ROOT / "static" / "ui" / "fonts" / "Inter-400-normal.woff2",
    ROOT / "static" / "ui" / "fonts" / "Inter-600-normal.woff2",
)

CSS_TEXT = CSS.read_text(encoding="utf-8")
JS_TEXT = JS.read_text(encoding="utf-8")
GALLERY_TEXT = GALLERY.read_text(encoding="utf-8")


def _luminance(hex_color: str) -> float:
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def channel(value: float) -> float:
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(v) for v in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _ratio(first: str, second: str) -> float:
    light = _luminance(first)
    dark = _luminance(second)
    if dark > light:
        light, dark = dark, light
    return (light + 0.05) / (dark + 0.05)


def test_component_files_present():
    assert CSS.is_file() and JS.is_file()
    assert PARTIAL.is_file() and GALLERY.is_file()
    assert SWITCH_PARTIAL.is_file()
    for font in FONTS:
        assert font.is_file(), font
        assert font.read_bytes()[:4] == b"wOF2", font  # real woff2, served offline


def test_partial_carries_card_pins_and_proof_slots():
    text = PARTIAL.read_text(encoding="utf-8")
    for slot in (
        "pf-card",
        "pf-frame",
        "pf-pin",
        "pf-legend",
        "pf-proof",
        "pf-hazard",
        "is-loading",
        "data-copy=",
        "box_xyxy",
    ):
        assert slot in text, slot
    assert 'id="simple-view"' not in text  # the switch is page-level, never per card


def _render_partial(**overrides):
    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates" / "partials" / "ui")),
        autoescape=True,
    )
    context = dict(
        number=1,
        window="1:01-1:25",
        frame_src="/guides/job-guide/image/0",
        action="Lift the fan out.",
        phrases="fan",
        size=(200, 150),
        pins=[{"x": 22.5, "y": 36.67, "part": 1}],
        parts=[{"phrase": "fan", "ok": True, "pin": 1}],
        text=TEXT,
    )
    context.update(overrides)
    return env.get_template("guide_step.html").render(**context)


def test_partial_legend_hides_in_fixing_view():
    rendered = _render_partial()
    assert '<ol class="pf-legend pf-filmed-only"' in rendered


def test_partial_legend_rows_reference_only_detected_parts():
    rendered = _render_partial(
        pins=[{"x": 75.0, "y": 50.0, "part": 2}],
        parts=[
            {"phrase": "fan", "ok": False},  # no detection: no pin reference
            {"phrase": "cable", "ok": True, "pin": 2},
        ],
    )
    pins = set(re.findall(r'class="pf-pin" id="([^"]+)"', rendered))
    assert pins == {"pin-1-2"}
    rows = re.findall(r"<li([^>]*)>", rendered)
    assert len(rows) == 2
    assert "data-pin" not in rows[0]
    assert 'data-pin="pin-1-2"' in rows[1]
    assert re.findall(r'<span class="pf-n" aria-hidden="true">(\d+)</span>', rendered) == [
        "1",
        "2",
    ]
    refs = set(re.findall(r'data-pin="([^"]+)"', rendered))
    assert refs and refs <= pins


def test_switch_partial_is_a_single_page_level_control():
    text = SWITCH_PARTIAL.read_text(encoding="utf-8")
    assert text.count('id="simple-view"') == 1
    assert "data-filmed" in text and "data-fixing" in text
    assert "pf-card" not in text


def test_each_page_template_carries_one_switch():
    for name in ("templates/guide.html", "templates/reader.html"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert text.count('id="simple-view"') == 1, name
    assert GALLERY_TEXT.count('id="simple-view"') == 1


def test_gallery_pins_match_legend_and_stay_on_frame():
    cards = re.findall(
        r'<article class="pf-card" id="(step-\d+)">(.*?)</article>',
        GALLERY_TEXT,
        re.DOTALL,
    )
    assert len(cards) == 3
    for card_id, body in cards:
        pins = re.findall(r'class="pf-pin" id="(pin-[\d-]+)"', body)
        refs = re.findall(r'data-pin="(pin-[\d-]+)"', body)
        assert pins, card_id
        assert sorted(pins) == sorted(refs), card_id  # every pin has a legend row
        for x, y in re.findall(r"--x:([\d.]+)%;--y:([\d.]+)%", body):
            assert 0 <= float(x) <= 100 and 0 <= float(y) <= 100, (card_id, x, y)


def test_gallery_uses_local_media_and_copy_slots():
    assert "../media/local/" in GALLERY_TEXT
    assert GALLERY_TEXT.count("data-copy=") >= 10
    assert "http://" not in GALLERY_TEXT and "https://" not in GALLERY_TEXT


def test_css_zero_network_focus_motion_and_tokens():
    for banned in ("http://", "https://", "@import", "fetch(", "XMLHttpRequest"):
        assert banned not in CSS_TEXT, banned
    assert ":focus-visible" in CSS_TEXT or ":focus-within" in CSS_TEXT
    assert "@media (prefers-reduced-motion" in CSS_TEXT
    for token in ("--pf-paper", "--pf-amber", "--pf-on-paper", "--pf-danger"):
        assert token in CSS_TEXT, token


def test_css_text_contrast_aa():
    pairs = [
        ("#1b1d1a", "#faf9f5"),  # body on card
        ("#5d5f58", "#faf9f5"),  # muted / pending badge on card
        ("#1a1300", "#f0a400"),  # pin number / active switch on amber
        ("#1e6b3a", "#e9f2ec"),  # proof badge ok
        ("#c2331f", "#f8e9e4"),  # hazard title on wash
        ("#1a1300", "#f9eed6"),  # check-time chip
        ("#1b1d1a", "#f0eee8"),  # focus ring on page
    ]
    for fg, bg in pairs:
        assert _ratio(fg, bg) >= 4.5, (fg, bg)


def test_css_time_preview_is_responsive():
    match = re.search(r"\.step-time \.time-preview \{(.*?)\}", CSS_TEXT, re.DOTALL)
    assert match, ".step-time .time-preview rule missing"
    body = match.group(1)
    for rule in ("max-width: 100%", "height: auto", "display: block"):
        assert rule in body, rule


def test_css_reduced_motion_kills_loading_pulse():
    start = CSS_TEXT.index("@media (prefers-reduced-motion")
    block = CSS_TEXT[start:]
    depth = 0
    end = None
    for pos, char in enumerate(block):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = pos
                break
    assert end is not None
    query = block[: end + 1]
    match = re.search(r"\.pf-frame\.is-loading \{(.*?)\}", query, re.DOTALL)
    assert match, ".pf-frame.is-loading rule missing inside the reduced-motion query"
    assert "animation: none" in match.group(1)


def test_gallery_set_start_controls_have_a_video_target():
    refs = set(re.findall(r'data-video="([^"]+)"', GALLERY_TEXT))
    assert refs == {"guide-video"}
    assert '<video id="guide-video" controls playsinline></video>' in GALLERY_TEXT


def test_gallery_legends_hide_in_fixing_view():
    assert GALLERY_TEXT.count('class="pf-legend pf-filmed-only"') == 3


def test_css_pulse_runs_only_while_loading():
    base = re.search(r"\.pf-frame \{(.*?)\}", CSS_TEXT, re.DOTALL)
    assert base, ".pf-frame rule missing"
    assert "animation" not in base.group(1)
    loading = re.search(r"\.pf-frame\.is-loading \{(.*?)\}", CSS_TEXT, re.DOTALL)
    assert loading, ".pf-frame.is-loading rule missing"
    assert "animation" in loading.group(1)


def test_css_fixing_view_hides_filmed_detail_and_pins():
    assert "body.simple .pf-filmed-only" in CSS_TEXT
    assert "body.simple .pf-pin" in CSS_TEXT


def test_js_zero_network_and_no_motion():
    for banned in (
        "http://",
        "https://",
        "fetch(",
        "XMLHttpRequest",
        "setTimeout",
        "setInterval",
        "requestAnimationFrame",
        "animate(",
        "transition",
    ):
        assert banned not in JS_TEXT, banned


def test_reader_page_includes_component_and_switch():
    body = TestClient(create_site_app()).get("/reader/sample").text
    assert '<link rel="stylesheet" href="/static/ui/guide_step.css">' in body
    assert '<script src="/static/ui/guide_step.js" defer></script>' in body
    assert 'id="simple-view"' in body
    assert "I filmed this" in body and "fixing this" in body
    assert "Simple view" not in body


def test_guide_template_includes_component_and_switch():
    template = (ROOT / "templates" / "guide.html").read_text(encoding="utf-8")
    assert '<link rel="stylesheet" href="/static/ui/guide_step.css">' in template
    assert '<script src="/static/ui/guide_step.js" defer></script>' in template
    assert '<script src="/static/reader.js" defer></script>' in template
    assert 'id="simple-view"' in template
    assert "I filmed this" in template and "fixing this" in template


def test_component_assets_served():
    client = TestClient(create_site_app())
    for path in (
        "/static/ui/guide_step.css",
        "/static/ui/guide_step.js",
        "/static/ui/gallery/guide_step.html",
        "/static/ui/fonts/Fraunces-600-normal.woff2",
        "/static/ui/fonts/Inter-400-normal.woff2",
    ):
        response = client.get(path)
        assert response.status_code == 200, path


def test_guide_page_renders_with_switch():
    from fastapi import FastAPI

    from app.reader import sample_guide
    from app.routes.guides import router, set_guide_loader

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=ROOT / "static", html=False))
    app.include_router(router)
    set_guide_loader(app, lambda job_id: sample_guide())
    client = TestClient(app)
    body = client.get("/guides/sample").text
    assert 'id="simple-view"' in body
    assert client.get("/static/ui/guide_step.css").status_code == 200


def test_local_media_stays_ignored():
    probe = ROOT / "static" / "ui" / "media" / "local" / "probe.webp"
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", str(probe)],
        cwd=ROOT,
        capture_output=True,
    )
    assert completed.returncode == 0, "media/local must be gitignored"


def test_browser_gallery_switch_hides_filmed_detail():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = None
        for kwargs in ({"headless": True}, {"headless": True, "channel": "chrome"}):
            try:
                browser = playwright.chromium.launch(**kwargs)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium or Chrome binary")
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda exc: errors.append(str(exc)))

            def on_console(message):
                if message.type != "error":
                    return
                # Gallery frames live under gitignored media/local (consented
                # footage, never committed): a clean clone 404s them, and the
                # switch/pin mechanics under test do not need the pixels.
                if message.text.startswith("Failed to load resource") and (
                    "media/local" in str(message.location.get("url", ""))
                ):
                    return
                errors.append(message.text)

            page.on("console", on_console)
            page.goto(GALLERY.as_uri())
            page.wait_for_load_state("load")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_function(
                "document.querySelectorAll('.pf-frame.is-loading').length === 0"
            )
            page.evaluate("window.scrollTo(0, 0)")
            assert page.locator(".pf-pin").count() == 7
            assert page.locator(".pf-pin:visible").count() == 7
            # 3 legends + 3 set-start feet + the page video target + 3 status help lines.
            assert page.locator(".pf-filmed-only:visible").count() == 10
            page.locator("[data-fixing]").click()
            assert page.locator(".pf-filmed-only:visible").count() == 0
            assert page.locator(".pf-pin:visible").count() == 0
            page.locator("[data-filmed]").click()
            assert page.locator(".pf-filmed-only:visible").count() == 10
            assert page.locator(".pf-pin:visible").count() == 7
            page.locator('.pf-legend li[data-pin="pin-2-1"]').hover()
            assert "on" in (page.locator("#pin-2-1").get_attribute("class") or "")
        finally:
            browser.close()
    assert errors == []


# --- Live-card data: pins, sizes, hazard text -------------------------------


def test_pin_center_percent_positions_and_edges():
    from app.guides import pin_center_percent

    assert pin_center_percent((20, 30, 70, 80), 200, 150) == (22.5, 36.67)
    assert pin_center_percent((0, 0, 100, 100), 100, 100) == (50.0, 50.0)
    assert pin_center_percent((-50, -50, -40, -40), 100, 100) == (0.0, 0.0)
    assert pin_center_percent((200, 200, 300, 300), 100, 100) == (100.0, 100.0)
    for bad in (
        ((20, 30, 70, 80), 0, 150),
        ((20, 30, 70, 80), 200, -1),
        ("nope", 200, 150),
        ((20, 30, 70), 200, 150),
        ((True, 0, 1, 1), 10, 10),  # bool is not a coordinate
        ((0, 0, 1, 1), float("nan"), 10),
        ((float("nan"), 0, 1, 1), 10, 10),
        ((0, float("nan"), 1, 1), 10, 10),
        ((0, 0, float("inf"), 1), 10, 10),
        ((0, 0, 1, float("-inf")), 10, 10),
    ):
        assert pin_center_percent(*bad) is None, bad


def test_image_pixel_size_roundtrip_and_garbage(tmp_path):
    from app.guides import image_pixel_size
    from app.render import write_png_rgb

    png = tmp_path / "frame.png"
    write_png_rgb(png, 200, 150, bytes(3) * (200 * 150))
    assert image_pixel_size(png) == (200, 150)
    assert image_pixel_size(tmp_path / "missing.png") is None
    text = tmp_path / "note.txt"
    text.write_text("not a png", encoding="utf-8")
    assert image_pixel_size(text) is None


def test_step_hazard_blocks_use_canonical_text():
    from app.contracts import Step
    from app.hazards import HAZARD_TEMPLATES, UNKNOWN_HAZARD_TEXT, step_hazard_blocks

    def step_with(*hazards):
        return Step(
            number=1,
            start_s=0.0,
            end_s=1.0,
            action="Do it.",
            tools=(),
            parts=(),
            hazards=hazards,
            time_spread_s=0.0,
            time_needs_check=False,
        )

    assert step_hazard_blocks(step_with("battery")) == (
        ("battery", HAZARD_TEMPLATES["battery"]),
    )
    assert step_hazard_blocks(step_with("adhesive", "heat")) == (
        ("adhesive/heat", HAZARD_TEMPLATES["adhesive/heat"]),
    )
    assert step_hazard_blocks(step_with("explosive")) == (
        ("unknown", UNKNOWN_HAZARD_TEXT),
    )
    assert step_hazard_blocks(step_with("  ")) == ()
    assert [tag for tag, _ in step_hazard_blocks(step_with("data", "battery"))] == [
        "data",
        "battery",
    ]


def _synthetic_guide(tmp_path):
    """Two-step guide over synthetic PNGs: one boxed part, one battery step."""
    from app.contracts import (
        ArtifactId,
        Detection,
        Guide,
        JobId,
        LabelDecision,
        LabelSlot,
        PartRequest,
        ProvisionalRender,
        Step,
    )
    from app.render import write_png_rgb

    frames = []
    for name, color in (("frame-1.png", (10, 20, 30)), ("frame-2.png", (30, 20, 10))):
        path = tmp_path / name
        write_png_rgb(path, 200, 150, bytes(color) * (200 * 150))
        frames.append(path)
    labelled = tmp_path / "labelled.png"
    write_png_rgb(labelled, 200, 150, bytes((10, 20, 30)) * (200 * 150))
    steps = (
        Step(
            number=1,
            start_s=10.0,
            end_s=18.0,
            action="Lift the fan out.",
            tools=("spudger",),
            parts=(PartRequest("fan", "center"), PartRequest("cable", "left")),
            hazards=("battery",),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
        Step(
            number=2,
            start_s=18.0,
            end_s=30.0,
            action="Seat the new fan.",
            tools=(),
            parts=(PartRequest("fan screws", "around"),),
            hazards=(),
            time_spread_s=9.0,
            time_needs_check=True,
        ),
    )
    detection = Detection(
        phrase="fan", mask_path=tmp_path / "mask.png", score=0.9,
        box_xyxy=(20, 30, 70, 80),
    )
    provisional = ProvisionalRender(
        detection=detection,
        artifact_id=ArtifactId("ui-test-1"),
        rendered_image_path=labelled,
        position_hint="center",
    )
    labels = (
        LabelDecision(
            slot=LabelSlot(step_index=0, part_index=0, part=steps[0].parts[0]),
            detection=detection,
            provisional=provisional,
            checked=True,
            status="part_pixels_unchanged",
            readback="yes",
        ),
    )
    return Guide(
        job_id=JobId("job-guide"),
        steps=steps,
        frame_paths=tuple(frames),
        labels=labels,
        difficulty="Easy",
        verdict="Try it",
        draft_reasons=(),
    )


def _guide_client(guide):
    from fastapi import FastAPI

    from app.routes.guides import router, set_guide_loader

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=ROOT / "static", html=False))
    app.include_router(router)
    set_guide_loader(app, lambda job_id: guide if job_id == "job-guide" else None)
    return TestClient(app)


def test_guide_route_renders_live_step_cards(tmp_path):
    from app.hazards import HAZARD_TEMPLATES

    client = _guide_client(_synthetic_guide(tmp_path))
    body = client.get("/guides/job-guide").text
    assert body.count('id="simple-view"') == 1
    # Pin at the box center: (20+70)/2/200, (30+80)/2/150.
    assert (
        '<span class="pf-pin" id="pin-1-1"'
        ' style="--x:22.5%;--y:36.67%;" aria-hidden="true">1</span>'
    ) in body
    assert '<ul class="parts pf-legend pf-filmed-only">' in body
    assert 'data-pin="pin-1-1"' in body
    assert body.count('class="pf-hazard"') == 1  # only the battery step
    assert "Warning: battery" in body
    assert HAZARD_TEMPLATES["battery"] in body
    assert 'class="shot pf-frame is-loading" style="aspect-ratio: 200 / 150"' in body
    assert body.count('class="toggle"') == 1  # labelled toggle preserved
    assert "No visual changes to this part" in body and "Not checked" in body
    pins = set(re.findall(r'class="pf-pin" id="([^"]+)"', body))
    refs = set(re.findall(r'data-pin="([^"]+)"', body))
    assert refs and refs <= pins  # every legend ref points at a real pin
    assert client.get("/guides/job-guide/image/0").status_code == 200


def test_guide_route_renders_time_controls_and_video_target(tmp_path):
    client = _guide_client(_synthetic_guide(tmp_path))
    body = client.get("/guides/job-guide").text
    assert '<video id="guide-video" controls playsinline></video>' in body
    assert '<script src="/static/time_controls.js" defer></script>' in body
    assert body.count("<button type=\"button\" data-set-start>") == 2  # one per card
    assert body.count('data-video="guide-video"') == 2
    assert client.get("/static/time_controls.js").status_code == 200


def _slotted_decision(step, step_index, part_index, box, *, status="part_pixels_unchanged", readback="yes"):
    from app.contracts import Detection, LabelDecision, LabelSlot

    detection = None
    if box is not None:
        detection = Detection(
            phrase=step.parts[part_index].phrase,
            mask_path=Path("mask.png"),
            score=0.9,
            box_xyxy=box,
        )
    return LabelDecision(
        slot=LabelSlot(
            step_index=step_index,
            part_index=part_index,
            part=step.parts[part_index],
        ),
        detection=detection,
        provisional=None,
        checked=status == "part_pixels_unchanged",
        status=status,
        readback=readback,
    )


def _two_fan_guide(tmp_path):
    """Two steps both asking for 'fan', each with its own boxed decision."""
    from app.contracts import Guide, JobId, PartRequest, Step
    from app.render import write_png_rgb

    frames = []
    for name in ("fan-1.png", "fan-2.png"):
        path = tmp_path / name
        write_png_rgb(path, 200, 100, bytes((40, 40, 40)) * (200 * 100))
        frames.append(path)
    steps = (
        Step(
            number=1,
            start_s=0.0,
            end_s=10.0,
            action="Lift the fan.",
            tools=(),
            parts=(PartRequest("fan", "left"),),
            hazards=(),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
        Step(
            number=2,
            start_s=10.0,
            end_s=20.0,
            action="Seat the fan.",
            tools=(),
            parts=(PartRequest("fan", "right"),),
            hazards=(),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
    )
    labels = (
        _slotted_decision(steps[0], 0, 0, (0, 0, 100, 50)),
        _slotted_decision(steps[1], 1, 0, (100, 50, 200, 100)),
    )
    return Guide(
        job_id=JobId("two-fan"),
        steps=steps,
        frame_paths=tuple(frames),
        labels=labels,
        difficulty="Easy",
        verdict="Try it",
        draft_reasons=(),
    )


def _mixed_status_guide(tmp_path):
    """One step, three boxed parts: failed read-back, not checked, passed."""
    from app.contracts import Guide, JobId, PartRequest, Step
    from app.render import write_png_rgb

    frame = tmp_path / "mixed.png"
    write_png_rgb(frame, 200, 100, bytes((40, 40, 40)) * (200 * 100))
    steps = (
        Step(
            number=1,
            start_s=0.0,
            end_s=10.0,
            action="Check the parts.",
            tools=(),
            parts=(
                PartRequest("fan", "left"),
                PartRequest("cable", "middle"),
                PartRequest("screws", "right"),
            ),
            hazards=(),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
    )
    labels = (
        _slotted_decision(steps[0], 0, 0, (0, 0, 50, 50), readback="no"),
        _slotted_decision(steps[0], 0, 1, (75, 0, 125, 50), status="not_checked"),
        _slotted_decision(steps[0], 0, 2, (150, 50, 200, 100)),
    )
    return Guide(
        job_id=JobId("mixed"),
        steps=steps,
        frame_paths=(frame,),
        labels=labels,
        difficulty="Easy",
        verdict="Try it",
        draft_reasons=(),
    )


def test_reader_pins_claim_by_slot_not_phrase(tmp_path):
    from app import reader as R
    from app.guides import step_cards
    from app.routes.site import _step_card

    guide = _two_fan_guide(tmp_path)
    rendered = {card.step.number: card for card in step_cards(guide)}
    assert rendered[1].parts[0].pin == (25.0, 25.0)
    assert rendered[2].parts[0].pin == (75.0, 75.0)
    steps = R.reader_context(guide, set())["steps"]
    assert isinstance(steps, list)
    card1 = _step_card(
        steps[0], R.READER_SAMPLE_URL, set(), guide=guide, guide_index=0
    )
    card2 = _step_card(
        steps[1], R.READER_SAMPLE_URL, set(), guide=guide, guide_index=1
    )
    assert 'id="pin-1-1" style="--x:25%;--y:25%;"' in card1
    assert "pin-1-2" not in card1  # step 2's fan never leaks into step 1
    assert 'id="pin-2-1" style="--x:75%;--y:75%;"' in card2
    assert "pin-2-2" not in card2
    for card in (card1, card2):
        pins = set(re.findall(r'class="pf-pin" id="([^"]+)"', card))
        refs = set(re.findall(r'data-pin="([^"]+)"', card))
        assert refs and refs <= pins


def test_reader_pins_skip_failed_and_unchecked(tmp_path):
    from app import reader as R
    from app.routes.site import _step_card

    guide = _mixed_status_guide(tmp_path)
    steps = R.reader_context(guide, set())["steps"]
    assert isinstance(steps, list)
    card = _step_card(
        steps[0], R.READER_SAMPLE_URL, set(), guide=guide, guide_index=0
    )
    assert 'id="pin-1-3" style="--x:87.5%;--y:75%;"' in card
    assert "pin-1-1" not in card  # failed read-back gets no pin
    assert "pin-1-2" not in card  # not-checked gets no pin
    pins = set(re.findall(r'class="pf-pin" id="([^"]+)"', card))
    refs = set(re.findall(r'data-pin="([^"]+)"', card))
    assert refs and refs <= pins


def test_reader_pins_keep_part_numbers_when_labels_arrive_reordered(tmp_path):
    from app import reader as R
    from app.contracts import Guide, JobId, PartRequest, Step
    from app.render import write_png_rgb
    from app.routes.site import _step_card

    frame = tmp_path / "reorder.png"
    write_png_rgb(frame, 200, 100, bytes((40, 40, 40)) * (200 * 100))
    steps = (
        Step(
            number=1,
            start_s=0.0,
            end_s=10.0,
            action="Check the fan and cable.",
            tools=(),
            parts=(PartRequest("fan", "left"), PartRequest("cable", "right")),
            hazards=(),
            time_spread_s=1.0,
            time_needs_check=False,
        ),
    )
    guide = Guide(
        job_id=JobId("reorder"),
        steps=steps,
        frame_paths=(frame,),
        labels=(
            _slotted_decision(steps[0], 0, 1, (100, 0, 200, 100)),  # cable first
            _slotted_decision(steps[0], 0, 0, (0, 0, 100, 100)),  # fan second
        ),
        difficulty="Easy",
        verdict="Try it",
        draft_reasons=(),
    )
    rendered = R.reader_context(guide, set())["steps"]
    assert isinstance(rendered, list)
    card = _step_card(
        rendered[0], R.READER_SAMPLE_URL, set(), guide=guide, guide_index=0
    )
    # Pins stay numbered by part position: fan is part 1, cable part 2.
    assert 'id="pin-1-1" style="--x:25%;--y:50%;"' in card
    assert 'id="pin-1-2" style="--x:75%;--y:50%;"' in card
    # Badge rows follow delivery order (cable first), but each badge shows
    # its pin's part number, never its row number.
    rows = re.findall(
        r'<li data-pin="([^"]+)" tabindex="0">'
        r'<span class="pf-n" aria-hidden="true">(\d+)</span> ',
        card,
    )
    assert ("pin-1-2", "2") in rows
    assert ("pin-1-1", "1") in rows


def test_reader_hero_route_renders_live_step_cards():
    body = TestClient(create_site_app()).get("/reader/hero").text
    assert body.count('id="simple-view"') == 1
    # Battery box (8,8,40,24) center over the 640x360 placeholder.
    assert 'id="pin-2-1" style="--x:3.75%;--y:4.44%;"' in body
    assert '<ul class="badges-list pf-legend pf-filmed-only">' in body
    assert '<section class="stop pf-hazard" aria-label="Hard stop">' in body
    assert body.count("<button type=\"button\" data-set-start>") == 2  # one per visible card
    assert '<script src="/static/time_controls.js" defer></script>' in body
    assert '<video id="guide-video" controls playsinline></video>' in body
    assert 'class="pf-frame is-loading" style="aspect-ratio: 640 / 360"' in body
    pins = set(re.findall(r'class="pf-pin" id="([^"]+)"', body))
    refs = set(re.findall(r'data-pin="([^"]+)"', body))
    assert refs and refs <= pins


def test_reader_sample_route_without_boxes():
    body = TestClient(create_site_app()).get("/reader/sample").text
    assert 'class="pf-pin"' not in body  # no boxes, so no pins
    assert "pf-legend" in body  # the unmatched-labels legend still lists
    assert "pf-filmed-only" in body
    assert body.count("<button type=\"button\" data-set-start>") == 3
    assert '<video id="guide-video"' in body
    assert TestClient(create_site_app()).get("/static/time_controls.js").status_code == 200


# --- Live-server browser tests: toggle + set-start on real routes -----------


@pytest.fixture(scope="module")
def live_base_url(tmp_path_factory):
    """Combined site + guides app on 127.0.0.1; yields the base URL."""
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn
    from fastapi import FastAPI

    from app.routes.guides import router as guides_router
    from app.routes.guides import set_guide_loader
    from app.routes.site import router as site_router
    from app.routes.steps import router as steps_router

    guide = _synthetic_guide(tmp_path_factory.mktemp("live"))
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=ROOT / "static", html=False))
    app.include_router(site_router)
    app.include_router(guides_router)
    app.include_router(steps_router)
    set_guide_loader(
        app, lambda job_id: guide if job_id in ("job-guide", "job-clip") else None
    )
    # D's stage shows the player only for a guide with its clip: job-clip carries a real one.
    from app.routes.guides import set_guide_clip_loader
    from test_pipeline import make_clip

    clip = make_clip(tmp_path_factory.mktemp("clip") / "clip.mp4", seconds=4.0)
    set_guide_clip_loader(app, lambda job_id: clip if job_id == "job-clip" else None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=1
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise AssertionError("uvicorn never served /")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def live_browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
    with sync_playwright() as playwright:
        browser = None
        for kwargs in ({"headless": True}, {"headless": True, "channel": "chrome"}):
            try:
                browser = playwright.chromium.launch(**kwargs)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium or Chrome binary")
        try:
            yield browser
        finally:
            browser.close()


def _watched_page(browser, errors):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on(
        "console",
        lambda message: errors.append(message.text)
        if message.type == "error"
        else None,
    )
    return page


def test_browser_guide_cards_toggle_hides_label_checks(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors)
    try:
        page.goto(f"{live_base_url}/guides/job-guide")
        page.wait_for_load_state("load")
        page.wait_for_function(
            "document.querySelectorAll('.pf-frame.is-loading').length === 0"
        )
        assert page.locator(".pf-pin").count() == 1
        # The outlined copy carries its own drawn labels; the live pins sit
        # on the original frame, so look at that one.
        if page.locator("#gd-outlines").is_visible():
            page.locator("#gd-outlines").click()
        page.locator("[data-stop-ack]").first.click()  # a hard stop folds the later steps until acknowledged
        page.locator("#step-1 .action").click()  # acknowledging moved the stage to step 2; back to step 1
        assert page.locator(".parts:visible").count() == 2
        assert page.locator(".pf-hazard:visible").count() == 1
        page.locator("[data-fixing]").click()
        assert page.locator(".parts:visible").count() == 0
        assert page.locator(".pf-pin:visible").count() == 0
        assert page.locator(".pf-hazard:visible").count() == 1  # followers keep warnings
        page.locator("[data-filmed]").click()
        assert page.locator(".parts:visible").count() == 2
        assert page.locator(".pf-pin:visible").count() == 1
        page.locator('.parts li[data-pin="pin-1-1"]').hover()
        assert "on" in (page.locator("#pin-1-1").get_attribute("class") or "")
    finally:
        page.close()
    assert errors == []


def test_browser_reader_cards_toggle_and_set_start(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors)
    try:
        page.goto(f"{live_base_url}/reader/hero?ack=2")
        page.wait_for_load_state("load")
        # Lazy frames below a short viewport never load on their own; the test is about the toggles.
        page.evaluate("document.querySelectorAll('img[loading=lazy]').forEach(i => { i.loading = 'eager'; })")
        page.wait_for_function(
            "document.querySelectorAll('.pf-frame.is-loading').length === 0"
        )
        assert page.locator("[data-set-start]").count() == 4
        assert page.locator(".badges-list:visible").count() == 2
        assert page.locator("#guide-video").count() == 1
        page.locator("[data-fixing]").click()
        assert page.locator(".badges-list:visible").count() == 0
        assert page.locator("#guide-video:visible").count() == 0
        page.locator("[data-filmed]").click()
        assert page.locator(".badges-list:visible").count() == 2
        page.evaluate(
            "Object.defineProperty(document.getElementById('guide-video'),"
            " 'currentTime', {get: () => 7.5, configurable: true})"
        )
        first = page.locator(".step-time").first
        assert first.get_attribute("data-start") == "2.0"
        first.locator("[data-set-start]").click()
        assert first.get_attribute("data-start") == "7.5"
    finally:
        page.close()
    assert errors == []


def test_browser_edit_start_moves_previous_end(live_browser, live_base_url):
    # Contiguous phases: editing step 2's start also moves step 1's
    # end to the same instant, with no gap or overlap left behind.
    errors: list[str] = []
    page = _watched_page(live_browser, errors)
    try:
        page.goto(f"{live_base_url}/steps/preview")
        page.wait_for_load_state("load")
        cards = page.locator(".step-time")
        assert cards.count() == 2
        first, second = cards.nth(0), cards.nth(1)
        assert first.get_attribute("data-start") == "0.0"
        assert first.get_attribute("data-end") == "34.0"
        assert second.get_attribute("data-start") == "34.0"
        second.locator('[data-nudge="5"]').click()
        assert second.get_attribute("data-start") == "39"
        assert first.get_attribute("data-end") == "39"
        assert "0:00–0:39" in (first.locator("[data-window]").inner_text())
        assert "0:39–2:30" in (second.locator("[data-window]").inner_text())
    finally:
        page.close()
    assert errors == []
