"""the live /guides/<job> page carries the same per-hazard stop as the
offline export: difficulty and verdict up top, later steps folded behind the stop until the
button is pressed, every warning visible throughout, and everything shown with no script."""

from __future__ import annotations

import contextlib
import re
import socket
import threading
import time

import pytest
import uvicorn

from app.exports import follower_html
from app.guides import assemble_guide
from test_export import JOB, guide  # noqa: F401 (fixture: one battery stop at step 2 of 3)
from test_export_browser import server  # noqa: F401 (fixture)
from test_guide import RecordingContext, _client_for, _frame, _step
from test_reader_browser import browser  # noqa: F401 (fixture)

STOP_LINE = "See a professional before you go on."
STOP_CTA = "I understand. Show the next steps."


@pytest.fixture
def two_stop_guide(tmp_path):
    """Shape two: a glued-screen stop at step 2 and a battery stop at step 4, five steps."""
    hazards = {2: ("display", "adhesive"), 4: ("battery",)}
    steps = []
    for n in range(1, 6):
        step = _step(n, ("lid",))
        steps.append(type(step)(**{**step.__dict__, "hazards": hazards.get(n, ())}))
    frames = [_frame(tmp_path / f"f{i}.png", color=(10 * i, 60, 90)) for i in range(1, 6)]
    return assemble_guide(RecordingContext(), steps, frames, [])


@pytest.fixture
def one_hazard_guide(tmp_path):
    """The sample's shape: the same battery hazard on steps 2 to 5 of six."""
    steps = []
    for n in range(1, 7):
        step = _step(n, ("lid",))
        steps.append(type(step)(**{**step.__dict__, "hazards": ("battery",) if 2 <= n <= 5 else ()}))
    frames = [_frame(tmp_path / f"f{i}.png", color=(10 * i, 60, 90)) for i in range(1, 7)]
    return assemble_guide(RecordingContext(), steps, frames, [])


def _page(guide) -> str:
    return _client_for(guide).get("/guides/job-guide").text


@pytest.mark.parametrize("shape", ["guide", "two_stop_guide"])
def test_live_page_has_difficulty_verdict_and_stops(shape, request):
    g = request.getfixturevalue(shape)
    body = _page(g)
    stops = {"guide": 1, "two_stop_guide": 2}[shape]
    assert body.count('data-stop="') == stops
    assert body.count('<div class="after-stop">') == stops
    assert body.count(STOP_LINE) == stops
    assert f'<span class="badge-{g.difficulty.lower()}">{g.difficulty}</span>' in body
    assert "Should you try this?" in body
    assert "DRAFT" not in re.search(r'class="verdict-line">([^<]*)', body).group(1)
    assert "I have read the warnings" not in body  # the unconnected checkbox is gone
    # no script needed to see everything: the fold is a script-added class, not markup
    assert "hidden" not in re.search(r'<div class="after-stop">', body).group(0)
    for step in g.steps:  # warnings are never folded away: the stop sits before its wrapper
        assert f'id="step-{step.number}"' in body


def test_stop_follows_its_step_and_wraps_only_later_steps(guide):
    body = _page(guide)
    assert body.index('id="step-2"') < body.index('data-stop="2"') < body.index('<div class="after-stop">') < body.index('id="step-3"')
    assert body.index('id="step-1"') < body.index('data-stop="2"')


def test_export_still_uses_the_same_stop_and_difficulty_line(guide, tmp_path):
    page, _ = follower_html(guide, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}.png")
    assert page.count(STOP_LINE) == 1 and page.count('class="after-stop"') == 1
    assert 'class="ack-box"' in page and "data-stop" not in page


def _visible(page, step):
    """The step's instructions are shown (a folded step shows only its row and warnings)."""
    return page.locator(f"#step-{step} .action").is_visible()


def _warnings_visible(page, step):
    """Every warning block on the step is really visible (computed, not just in the markup)."""
    blocks = page.locator(f"#step-{step} .pf-hazard")
    return blocks.count() > 0 and all(blocks.nth(i).is_visible() for i in range(blocks.count())) \
        and page.locator(f"#step-{step} h2").is_visible() and page.locator(f"#step-{step} .gd-no").is_visible()


@contextlib.contextmanager
def _serve(guide):
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from app.routes.guides import router, set_guide_loader

    app = FastAPI()
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="static"))
    set_guide_loader(app, lambda j: guide if j == "job-guide" else None)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/guides/job-guide"
    finally:
        srv.should_exit = True
        thread.join(timeout=10)


@pytest.mark.parametrize("size", [(390, 800), (1440, 900)])
def test_one_stop_gated_then_open_by_keyboard(browser, server, size):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    try:
        page.goto(f"{server}/guides/{JOB}")
        assert _visible(page, 1) and _visible(page, 2)
        assert not _visible(page, 3)
        assert page.locator("[data-stop] strong").is_visible()  # the warning stays
        assert page.locator("[data-stop-ack]").get_attribute("aria-expanded") == "false"
        page.locator("[data-stop-ack]").focus()
        page.keyboard.press("Enter")
        assert _visible(page, 3)
        assert page.locator("[data-stop-done]").is_visible()
        assert page.locator("[data-stop] strong").is_visible()  # read-on keeps the warning
        assert page.evaluate("document.activeElement.id") == "step-3"
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    finally:
        page.close()


def test_two_stops_each_need_their_own_acknowledgement(browser, two_stop_guide):
    with _serve(two_stop_guide) as url:
        page = browser.new_page(viewport={"width": 390, "height": 800})
        try:
            page.goto(url)
            assert [_visible(page, n) for n in range(1, 6)] == [True, True, False, False, False]
            page.locator('[data-stop="2"] [data-stop-ack]').click()
            assert [_visible(page, n) for n in range(1, 6)] == [True, True, True, True, False]
            assert page.locator('[data-stop="4"] [data-stop-ack]').is_visible()
            page.locator('[data-stop="4"] [data-stop-ack]').click()
            assert _visible(page, 5)
        finally:
            page.close()


@pytest.mark.parametrize("size", [(390, 800), (1440, 900)])
def test_later_warnings_stay_visible_before_acknowledgement(browser, one_hazard_guide, size):
    with _serve(one_hazard_guide) as url:
        page = browser.new_page(viewport={"width": size[0], "height": size[1]})
        try:
            page.goto(url)
            for n in range(3, 6):  # battery warnings on the folded steps 3-5
                assert not _visible(page, n)
                assert _warnings_visible(page, n)
                assert "Disconnect the battery before touching" in page.locator(f"#step-{n} .pf-hazard").first.inner_text()
            page.locator("[data-stop-ack]").click()
            assert all(_visible(page, n) and _warnings_visible(page, n) for n in range(3, 6))
        finally:
            page.close()


def test_no_script_hides_the_inert_button(browser, server):
    context = browser.new_context(java_script_enabled=False, viewport={"width": 390, "height": 800})
    try:
        page = context.new_page()
        page.goto(f"{server}/guides/{JOB}")
        assert not page.locator("[data-stop-ack]").is_visible()
    finally:
        context.close()


def test_without_script_everything_and_every_warning_shows(browser, server):
    context = browser.new_context(java_script_enabled=False, viewport={"width": 390, "height": 800})
    try:
        page = context.new_page()
        page.goto(f"{server}/guides/{JOB}")
        assert _visible(page, 3)
        assert page.locator("[data-stop] strong").is_visible()
    finally:
        context.close()


def test_one_gate_per_hazard_chips_stay_on_later_steps(one_hazard_guide):
    body = _page(one_hazard_guide)
    assert body.count('data-stop="') == 1 and body.count('<div class="after-stop">') == 1
    assert 'data-stop="2"' in body
    for n in range(2, 6):  # every battery step keeps its own visible warning text
        card = body[body.index(f'id="step-{n}"'):body.index(f'id="step-{n + 1}"')]
        assert "Disconnect the battery before touching" in card


def test_stop_names_the_hazard_and_its_risk(one_hazard_guide, two_stop_guide):
    one = _page(one_hazard_guide)
    assert "Hard stop: battery." in one and "Disconnect the battery first." in one
    assert "glued screen" not in re.search(r'<section class="stop".*?</section>', one, re.S).group(0)
    two = _page(two_stop_guide)
    assert "Hard stop: glued screen." in two and "cracking" not in two and "crack the glass" in two
    assert "Hard stop: battery." in two


def test_folded_marker_counts_later_steps(one_hazard_guide, two_stop_guide):
    assert "4 more steps are below, folded to their warnings." in _page(one_hazard_guide)
    two = _page(two_stop_guide)
    assert "3 more steps are below, folded to their warnings." in two and "1 more step is below, folded to its warnings." in two


def test_reader_verdict_is_plain_never_draft(one_hazard_guide, two_stop_guide, guide):
    assert one_hazard_guide.verdict == "DRAFT"
    head = lambda g: re.search(r'class="verdict-line">([^<]*)', _page(g)).group(1)
    assert "DRAFT" not in head(one_hazard_guide)
    assert "safely disconnect a laptop battery" in head(one_hazard_guide)
    assert "glued screen and disconnect the battery" in head(two_stop_guide)
    from dataclasses import replace
    real = replace(one_hazard_guide, verdict="Try it with help")
    assert "Try it with help" in head(real)


def test_end_of_guide_waits_for_the_stop(browser, server):
    page = browser.new_page(viewport={"width": 390, "height": 800})
    try:
        page.goto(f"{server}/guides/{JOB}")
        assert page.locator("[data-stop-folded]").is_visible()
        assert not page.locator(".guide-foot").is_visible()
        page.locator("[data-stop-ack]").click()
        assert page.locator(".guide-foot").is_visible()
        assert not page.locator("[data-stop-folded]").is_visible()
    finally:
        page.close()


def _glued(tags, action):
    step = _step(1, ("lid",))
    return type(step)(**{**step.__dict__, "hazards": tags, "action": action})


@pytest.mark.parametrize("tags,action", [
    (("display", "adhesive"), "Lift the panel."),  # separate tags
    (("sharp",), "Pry the glued screen"),  # action-text fallback
])
def test_canonical_glued_screen_gates_live_and_export(tags, action, tmp_path):
    from app.reader import stop_kinds
    first = _glued((), "Open the lid.")
    steps = [first, type(first)(**{**_glued(tags, action).__dict__, "number": 2}),
             type(first)(**{**first.__dict__, "number": 3})]
    assert stop_kinds(steps[1]) == ("glued_screen",) or "glued_screen" in stop_kinds(steps[1])
    frames = [_frame(tmp_path / f"f{i}.png", color=(10 * i, 60, 90)) for i in range(1, 4)]
    g = assemble_guide(RecordingContext(), steps, frames, [])
    assert 'data-stop="2"' in _page(g) and "Hard stop: glued screen" in _page(g)
    page, _ = follower_html(g, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}.png")
    assert 'class="ack-box"' in page and "Hard stop: glued screen" in page


def test_shoot_stop_evidence(browser, one_hazard_guide):  # writes the 390 shots when PF_SHOTS is set
    import os
    out = os.environ.get("PF_SHOTS")
    if not out:
        pytest.skip("PF_SHOTS not set")
    with _serve(one_hazard_guide) as url:
        page = browser.new_page(viewport={"width": 390, "height": 800})
        try:
            page.goto(url)
            page.locator("[data-stop]").scroll_into_view_if_needed()
            page.screenshot(path=f"{out}/case3-390-before-ack.png", full_page=True)
            page.locator("[data-stop-ack]").click()
            page.screenshot(path=f"{out}/case3-390-after-ack.png", full_page=True)
        finally:
            page.close()


def _hazard_guide(tmp_path, tags_by_step):
    steps = []
    for n in range(1, 5):
        step = _step(n, ("lid",))
        steps.append(type(step)(**{**step.__dict__, "hazards": tags_by_step.get(n, ())}))
    frames = [_frame(tmp_path / f"f{i}.png", color=(10 * i, 60, 90)) for i in range(1, 5)]
    return assemble_guide(RecordingContext(), steps, frames, [])


_B, _G, _U = ("battery",), ("display", "adhesive"), ("zzz_unlisted_hazard",)


@pytest.mark.parametrize("sets,names,absent", [
    ({2: _U}, ["unrecognized hazard"], ["battery", "glued screen"]),
    ({2: _B, 3: _U}, ["battery", "unrecognized hazard"], ["glued screen"]),
    ({2: _G, 3: _U}, ["glued screen", "unrecognized hazard"], ["battery"]),
    ({2: _B, 3: _G, 4: _U}, ["battery", "glued screen", "unrecognized hazard"], []),
])
def test_unknown_and_mixed_unknown_stop_and_verdict_are_honest(tmp_path, sets, names, absent):
    from app.guides import verdict_text
    g = _hazard_guide(tmp_path, sets)
    verdict = verdict_text(g).lower()
    page, _ = follower_html(g, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}.png")
    for body in (_page(g), page):
        assert "Hard stop:" in body and "could not be identified" in verdict
        assert "unrecognized hazard" in body.lower()
        for name in absent:  # a hazard that is not present is never claimed, in a stop or the verdict
            assert name not in verdict
            assert f"Hard stop: {name}" not in body and f"and {name}." not in body
    for name in names:
        key = {"battery": "battery", "glued screen": "glued", "unrecognized hazard": "identified"}[name]
        assert key in verdict


def test_draft_note_is_maker_only_and_reader_gets_a_safety_version(tmp_path):
    from app.hazards import DRAFT_ORDER_REASON
    g = _hazard_guide(tmp_path, {2: _B})
    g = type(g)(**{**g.__dict__, "draft_reasons": (DRAFT_ORDER_REASON, "DRAFT: other maker detail")})
    reader = "The video does not show the battery being disconnected. Disconnect it before you go on."
    page, _ = follower_html(g, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}.png")
    for body in (_page(g), page):
        assert f'<li class="maker-only">{DRAFT_ORDER_REASON}</li>' in body
        assert f'<li class="maker-only">DRAFT: other maker detail</li>' in body
        assert f'<li class="reader-only">{reader}</li>' in body
        assert "DRAFT" not in re.search(r'<li class="reader-only">([^<]*)</li>', body).group(1)
    assert "body.simple .draft-notes .maker-only { display: none; }" in page
    assert "body:not(.simple) .draft-notes .reader-only { display: none; }" in page


def test_export_draft_note_views_in_browser(browser, tmp_path):
    from app.hazards import DRAFT_ORDER_REASON
    g = _hazard_guide(tmp_path, {2: _B})
    g = type(g)(**{**g.__dict__, "draft_reasons": (DRAFT_ORDER_REASON,)})
    page_html, _ = follower_html(g, images_dir="images", frame_pattern="f{number}.png", label_pattern="l{number}.png")
    page = browser.new_page(viewport={"width": 390, "height": 800})
    try:
        page.set_content(page_html)
        assert page.evaluate("document.body.classList.contains('simple')")
        assert not page.locator(".maker-only").is_visible() and page.locator(".reader-only").is_visible()
        assert "DRAFT" not in page.locator(".draft-notes").inner_text()
        page.evaluate("document.body.classList.remove('simple')")
        assert page.locator(".maker-only").is_visible() and not page.locator(".reader-only").is_visible()
    finally:
        page.close()


def test_draft_note_views_in_browser(browser, tmp_path):
    from app.hazards import DRAFT_ORDER_REASON
    g = _hazard_guide(tmp_path, {2: _B})
    g = type(g)(**{**g.__dict__, "draft_reasons": (DRAFT_ORDER_REASON,)})
    with _serve(g) as url:
        page = browser.new_page(viewport={"width": 390, "height": 800})
        try:
            page.goto(url)
            page.evaluate("document.body.classList.add('simple')")
            assert not page.locator(".maker-only").is_visible() and page.locator(".reader-only").is_visible()
            page.evaluate("document.body.classList.remove('simple')")
            assert page.locator(".maker-only").is_visible() and not page.locator(".reader-only").is_visible()
        finally:
            page.close()
