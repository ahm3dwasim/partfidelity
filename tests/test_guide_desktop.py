"""Desktop two-column guide tests.

The guide page and the /steps/preview harness share one layout: at
>= 64rem the video pins sticky on the left while the step cards scroll
on the right; below that they stack as before. Clicking a card seeks
the video to that step's start.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from test_guide import RecordingContext, _decision, _frame, _render, _slot, _step
from test_ui_guide_step import live_base_url, live_browser  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]
CSS_TEXT = (ROOT / "static" / "guide.css").read_text(encoding="utf-8")
JS_TEXT = (ROOT / "static" / "reader.js").read_text(encoding="utf-8")


def _guide(tmp_path, *, action: str | None = None):
    from app.guides import assemble_guide

    ctx = RecordingContext()
    provisional = _render(ctx, tmp_path, "d", "fan")
    first = _step(1, ("fan",), action=action) if action is not None else _step(1, ("fan",))
    return assemble_guide(
        ctx,
        [first, _step(2, ("battery",))],
        [tmp_path / "d-frame.png"],
        [_decision(provisional, slot=_slot(0, 0, "fan"))],
    )


def _guide_body(tmp_path, *, action: str | None = None) -> str:
    from app.routes.guides import router, set_guide_loader

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=ROOT / "static", html=False))
    app.include_router(router)
    guide = _guide(tmp_path, action=action)
    set_guide_loader(app, lambda job_id: guide if job_id == "job-guide" else None)
    page = TestClient(app).get("/guides/job-guide")
    assert page.status_code == 200
    return page.text


def _preview_body() -> str:
    from app.routes.steps import router

    app = FastAPI()
    app.include_router(router)
    page = TestClient(app).get("/steps/preview")
    assert page.status_code == 200
    return page.text


def _assert_two_column_regions(body: str, card_marker: str) -> None:
    for region in ('class="guide-cols"', 'class="guide-video"', 'class="guide-steps"'):
        assert region in body, region
    video_at = body.index('id="guide-video"')
    card_at = body.index(card_marker)
    assert body.index('class="guide-video"') < video_at < body.index('class="guide-steps"')
    assert body.index('class="guide-steps"') < card_at


def test_guide_page_renders_two_column_wrapper_and_regions(tmp_path):
    body = _guide_body(tmp_path)
    _assert_two_column_regions(body, 'id="step-1"')
    assert '<script src="/static/reader.js" defer></script>' in body


def test_preview_page_renders_two_column_wrapper_and_regions():
    body = _preview_body()
    _assert_two_column_regions(body, 'aria-label="Step 1"')
    assert '<link rel="stylesheet" href="/static/guide.css">' in body
    assert '<script src="/static/reader.js" defer></script>' in body


def test_layout_values_live_in_guide_css_custom_properties():
    root = CSS_TEXT.split("* { box-sizing: border-box; }")[0]
    for name in (
        "--guide-gutter",
        "--guide-max",
        "--guide-video-col",
        "--guide-steps-col",
        "--guide-col-gap",
        "--guide-sticky-top",
    ):
        assert f"{name}:" in root, name
    assert CSS_TEXT.count("64rem") == 1  # one breakpoint, in one media query
    assert "@media (min-width: 64rem)" in CSS_TEXT
    assert "position: sticky" in CSS_TEXT
    assert "grid-template-columns: var(--guide-video-col) var(--guide-steps-col)" in CSS_TEXT
    assert "max-width: var(--guide-max)" in CSS_TEXT
    assert "padding-inline: var(--guide-gutter)" in CSS_TEXT
    assert "calc(100vh - 2 * var(--guide-sticky-top))" in CSS_TEXT


def test_active_card_style_cannot_shift_layout():
    match = re.search(
        r"\.card\.is-active, \.step-card\.is-active, #main \.step\.is-active \{(.*?)\}",
        CSS_TEXT,
        re.DOTALL,
    )
    assert match, "active-card rule missing"
    body = match.group(1)
    assert "box-shadow: inset" in body
    assert "var(--accent)" in body
    for shifting in ("border", "margin", "padding", "width", "height"):
        assert shifting not in body, shifting


def test_seek_script_seeks_marks_and_ignores_controls():
    for snippet in (
        ".step-time",
        "data-start",
        "currentTime",
        "timeupdate",
        "is-active",
        '"button, a, input, label, summary"',
    ):
        assert snippet in JS_TEXT, snippet


def test_player_keeps_fixed_16_9_frame_at_every_width():
    base, _, desktop = CSS_TEXT.partition("@media (min-width: 64rem)")
    assert desktop, "desktop query missing"
    root = base.split("* { box-sizing: border-box; }")[0]
    assert "--guide-video-bg:" in root
    frame = re.search(r"\.guide-video video \{(.*?)\}", base, re.DOTALL)
    assert frame, "base player-frame rule missing (phone needs it too)"
    body = frame.group(1)
    for rule in (
        "width: 100%",
        "aspect-ratio: 16 / 9",
        "object-fit: contain",
        "background: var(--guide-video-bg)",
    ):
        assert rule in body, rule
    capped = re.search(r"\.guide-video video \{(.*?)\}", desktop, re.DOTALL)
    assert capped, "desktop player-cap rule missing"
    cap_body = capped.group(1)
    assert "calc(100vh - 2 * var(--guide-sticky-top))" in cap_body
    # Short window: the width shrinks so the 16:9 frame still fits.
    assert "calc((100vh - 2 * var(--guide-sticky-top)) * 16 / 9)" in cap_body


def test_now_playing_block_styles_use_guide_tokens():
    block = re.search(r"\.guide-now \{(.*?)\}", CSS_TEXT, re.DOTALL)
    assert block, ".guide-now rule missing"
    body = block.group(1)
    for rule in (
        "background: var(--card)",
        "border: 1px solid var(--line)",
    ):
        assert rule in body, rule
    assert ".guide-now-title" in CSS_TEXT
    assert ".guide-now-time" in CSS_TEXT
    assert "color: var(--muted)" in CSS_TEXT


def _assert_now_playing(body: str, title: str, window: str) -> None:
    assert 'class="guide-now"' in body
    assert 'id="guide-now-title"' in body
    assert 'id="guide-now-time"' in body
    assert "aria-live" in body
    assert title in body
    assert window in body
    video_region = body.index('class="guide-video"')
    steps_region = body.index('class="guide-steps"')
    assert video_region < body.index('class="guide-now"') < steps_region
    assert body.index('id="guide-video"') < body.index('class="guide-now"')


def test_guide_page_renders_now_playing_first_step(tmp_path):
    body = _guide_body(tmp_path)
    _assert_now_playing(body, "Step 1: Do step 1", "~0:10–0:18")


def test_preview_page_renders_now_playing_first_step():
    body = _preview_body()
    _assert_now_playing(body, "Step 1: Remove the rear screws", "~0:00–0:34")


def test_seek_script_mirrors_active_card_in_now_playing():
    for snippet in (
        "guide-now-title",
        "guide-now-time",
        "[data-window]",
        "textContent",
        ".action",
    ):
        assert snippet in JS_TEXT, snippet


def test_now_playing_escapes_step_action_exactly_once(tmp_path):
    from app.contracts import Step, StepBundle
    from app.routes.steps import render_preview_page

    hostile = '<img src=x onerror="alert(1)">&"quoted"'
    body = _guide_body(tmp_path, action=hostile)
    assert hostile not in body
    assert "&lt;img" in body
    assert "&amp;lt;" not in body
    bundle = StepBundle(
        device_guess="Cordless hand vacuum",
        confidence=0.82,
        steps=(
            Step(
                number=1,
                start_s=0.0,
                end_s=5.0,
                action=hostile,
                tools=(),
                parts=(),
                hazards=(),
                time_spread_s=1.0,
                time_needs_check=False,
            ),
        ),
        source_runs=1,
        timestamps_approximate=True,
    )
    preview = render_preview_page(bundle)
    assert hostile not in preview
    assert "&lt;img" in preview
    assert "&amp;lt;" not in preview


def _watched_page(browser, errors, width, height):
    page = browser.new_page(viewport={"width": width, "height": height})
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on(
        "console",
        lambda message: errors.append(message.text)
        if message.type == "error"
        else None,
    )
    return page


def test_browser_desktop_columns_click_seeks_and_follows(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors, 1440, 900)
    try:
        page.goto(f"{live_base_url}/guides/job-guide")
        page.wait_for_load_state("load")
        page.wait_for_function(
            "document.querySelectorAll('.pf-frame.is-loading').length === 0"
        )
        assert (
            page.evaluate(
                "getComputedStyle(document.querySelector('.guide-cols')).display"
            )
            == "grid"
        )
        assert (
            page.evaluate(
                "getComputedStyle(document.querySelector('.guide-video')).position"
            )
            == "sticky"
        )
        video_box = page.locator(".guide-video").bounding_box()
        steps_box = page.locator(".guide-steps").bounding_box()
        assert video_box and steps_box
        assert video_box["x"] + video_box["width"] <= steps_box["x"]
        # The harness video has no media, so record seeks through a spy.
        page.evaluate(
            "window.__sought = null; window.__at = 0;"
            "Object.defineProperty(document.getElementById('guide-video'),"
            " 'currentTime', {get: () => window.__at,"
            " set: (v) => { window.__sought = v; }, configurable: true});"
        )
        page.locator("[data-stop-ack]").first.click()  # step 1 is a hard stop: step 2 stays folded until then
        page.locator("#step-2 .action").click()
        assert page.evaluate("window.__sought") == 18.0  # step 2 starts at 18s
        assert "is-active" in (page.locator("#step-2").get_attribute("class") or "")
        assert "is-active" not in (page.locator("#step-1").get_attribute("class") or "")
        # Playback moves the active card: 12s sits inside step 1 (10-18s).
        page.evaluate(
            "window.__at = 12;"
            "document.getElementById('guide-video')"
            ".dispatchEvent(new Event('timeupdate'))"
        )
        assert "is-active" in (page.locator("#step-1").get_attribute("class") or "")
        assert "is-active" not in (page.locator("#step-2").get_attribute("class") or "")
        # Buttons keep their own job: set-start takes the paused-video
        # time (the spy still reports 12s), never seeks.
        page.evaluate("window.__sought = null")
        page.locator("#step-2 [data-set-start]").click()
        assert page.evaluate("window.__sought") is None
        assert page.locator("#step-2 .step-time").get_attribute("data-start") == "12"
    finally:
        page.close()
    assert errors == []


def test_browser_play_cancels_a_held_observer_step(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors, 1440, 900)
    try:
        page.goto(f"{live_base_url}/guides/job-guide")
        page.wait_for_load_state("load")
        page.evaluate(
            "window.__playing = false;"
            "Object.defineProperty(document.getElementById('guide-video'),"
            " 'paused', {get: () => !window.__playing, configurable: true});"
        )
        page.evaluate(
            "document.dispatchEvent(new CustomEvent('pf:active-step',"
            " {detail: {card: document.getElementById('step-2')}}))"
        )
        # An old card scrolls into the band during the hold, then Play lands before it ends.
        page.evaluate(
            "document.getElementById('step-1').scrollIntoView({block: 'center'});"
            "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            ".then(() => { window.__playing = true;"
            " document.getElementById('guide-video').dispatchEvent(new Event('play')); })"
        )
        page.wait_for_function("window.__playing")
        stage = "document.querySelector('.guide-stage, .stage, [data-stage]')"
        before = page.evaluate(f"({stage} || document.body).innerText")
        page.wait_for_timeout(700)  # past the hold
        assert page.evaluate(f"({stage} || document.body).innerText") == before
    finally:
        page.close()
    assert errors == []


def test_browser_phone_stacks_video_over_cards(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors, 390, 844)
    try:
        page.goto(f"{live_base_url}/guides/job-clip")
        page.wait_for_load_state("load")
        assert (
            page.evaluate(
                "getComputedStyle(document.querySelector('.guide-cols')).display"
            )
            == "block"
        )
        video_box = page.locator(".guide-video").bounding_box()
        steps_box = page.locator(".guide-steps").bounding_box()
        assert video_box and steps_box
        assert video_box["y"] + video_box["height"] <= steps_box["y"]
    finally:
        page.close()
    assert errors == []


def _assert_16_9_frame(page, locator: str) -> dict:
    box = page.locator(locator).bounding_box()
    assert box and box["width"] > 200 and box["height"] > 100
    ratio = box["width"] / box["height"]
    assert abs(ratio - 16 / 9) < 0.03, (locator, box)
    return box


def test_browser_player_frame_and_now_playing_follow(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors, 1440, 900)
    try:
        page.goto(f"{live_base_url}/guides/job-clip")
        page.wait_for_load_state("load")
        page.wait_for_function(
            "document.querySelectorAll('.pf-frame.is-loading').length === 0"
        )
        # D's stage shows the step's frame first; Play swaps in the player.
        page.locator("#gd-play").click()
        page.evaluate("document.getElementById('guide-video').pause()")
        video_box = _assert_16_9_frame(page, "#guide-video")
        now_box = page.locator(".guide-now").bounding_box()
        assert now_box
        assert now_box["y"] >= video_box["y"] + video_box["height"]
        assert page.locator("#guide-now-title").inner_text() == (
            "Step 1: Lift the fan out."
        )
        assert page.locator("#guide-now-time").inner_text() == "~0:10–0:18"
        page.evaluate(
            "window.__at = 0;"
            "Object.defineProperty(document.getElementById('guide-video'),"
            " 'currentTime', {get: () => window.__at,"
            " set: (v) => { window.__at = v; }, configurable: true});"
        )
        page.locator("[data-stop-ack]").first.click()  # step 1 is a hard stop: step 2 stays folded until then
        page.locator("#step-2 .action").click()
        assert page.locator("#guide-now-title").inner_text() == (
            "Step 2: Seat the new fan."
        )
        assert page.locator("#guide-now-time").inner_text() == "~0:18–0:30"
        # Playback moves now-playing with the active card: 12s is step 1.
        page.evaluate(
            "window.__at = 12;"
            "document.getElementById('guide-video')"
            ".dispatchEvent(new Event('timeupdate'))"
        )
        assert page.locator("#guide-now-title").inner_text() == (
            "Step 1: Lift the fan out."
        )
        assert page.locator("#guide-now-time").inner_text() == "~0:10–0:18"
    finally:
        page.close()
    assert errors == []


def test_browser_short_window_shrinks_player_width(live_browser, live_base_url):
    errors: list[str] = []
    page = _watched_page(live_browser, errors, 1440, 400)
    try:
        page.goto(f"{live_base_url}/guides/job-clip")
        page.wait_for_load_state("load")
        page.locator("#gd-play").click()
        page.evaluate("document.getElementById('guide-video').pause()")
        video_box = _assert_16_9_frame(page, "#guide-video")
        column_box = page.locator(".guide-video").bounding_box()
        assert column_box
        assert video_box["width"] < column_box["width"] - 50
    finally:
        page.close()
    assert errors == []


def test_browser_phone_player_keeps_frame_and_now_playing(
    live_browser, live_base_url
):
    errors: list[str] = []
    page = _watched_page(live_browser, errors, 390, 844)
    try:
        # Phones show the player straight away when the guide has its clip.
        page.goto(f"{live_base_url}/guides/job-clip")
        page.wait_for_load_state("load")
        video_box = _assert_16_9_frame(page, "#guide-video")
        now_box = page.locator(".guide-now").bounding_box()
        steps_box = page.locator(".guide-steps").bounding_box()
        assert now_box and steps_box
        assert now_box["y"] >= video_box["y"] + video_box["height"]
        assert now_box["y"] + now_box["height"] <= steps_box["y"]
        assert page.locator("#guide-now-title").inner_text() == (
            "Step 1: Lift the fan out."
        )
    finally:
        page.close()
    assert errors == []
