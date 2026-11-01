"""Rendered-page checks for the D refine fixes."""
from __future__ import annotations

import pytest

from test_ui_guide_step import live_base_url, live_browser  # noqa: F401 (fixtures)

# Worst text/background contrast among the visible text elements under `sel`,
# measured on the rendered page (ancestor opacity composited in).
CONTRAST_JS = """
(sel) => {
  const parse = (c) => { const m = c.match(/[\\d.]+/g).map(Number); return {r: m[0], g: m[1], b: m[2], a: m.length > 3 ? m[3] : 1}; };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const mix = (top, bot) => ({r: top.r * top.a + bot.r * (1 - top.a), g: top.g * top.a + bot.g * (1 - top.a), b: top.b * top.a + bot.b * (1 - top.a), a: 1});
  const bgOf = (el) => { const stack = []; for (let n = el; n; n = n.parentElement) { const c = parse(getComputedStyle(n).backgroundColor); if (c.a > 0) { stack.push(c); if (c.a >= 1) break; } } let base = {r: 15, g: 15, b: 15, a: 1}; for (let i = stack.length - 1; i >= 0; i--) base = mix(stack[i], base); return base; };
  const opacityOf = (el) => { let o = 1; for (let n = el; n; n = n.parentElement) o *= parseFloat(getComputedStyle(n).opacity); return o; };
  let worst = 99, who = "";
  for (const root of document.querySelectorAll(sel)) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let t = walker.nextNode(); t; t = walker.nextNode()) {
      if (!t.textContent.trim()) continue;
      const el = t.parentElement;
      if (!el.offsetParent && getComputedStyle(el).position !== "fixed") continue;
      const bg = bgOf(el);
      let fg = parse(getComputedStyle(el).color);
      fg = mix({...fg, a: fg.a * opacityOf(el)}, bg);
      const [a, b] = [lum(fg), lum(bg)].sort((x, y) => y - x);
      const ratio = (a + 0.05) / (b + 0.05);
      if (ratio < worst) { worst = ratio; who = el.className + ":" + t.textContent.trim().slice(0, 20); }
    }
  }
  return [worst, who];
}
"""


@pytest.fixture(scope="module")
def multi_base_url(tmp_path_factory):
    """Site + guides app whose first step has two labelled parts."""
    import dataclasses
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    from app.contracts import Detection, LabelDecision, LabelSlot, ProvisionalRender, ArtifactId
    from app.routes.guides import router, set_guide_loader
    from test_ui_guide_step import ROOT, _synthetic_guide

    guide = _synthetic_guide(tmp_path_factory.mktemp("multi"))
    first = guide.labels[0]
    detection = Detection(
        phrase="cable", mask_path=first.detection.mask_path, score=0.9, box_xyxy=(100, 30, 150, 80)
    )
    second = LabelDecision(
        slot=LabelSlot(step_index=0, part_index=1, part=guide.steps[0].parts[1]),
        detection=detection,
        provisional=ProvisionalRender(
            detection=detection, artifact_id=ArtifactId("ui-test-2"),
            rendered_image_path=first.provisional.rendered_image_path, position_hint="left",
        ),
        checked=True, status="part_pixels_unchanged", readback="yes",
    )
    guide = dataclasses.replace(guide, labels=guide.labels + (second,))
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=ROOT / "static", html=False))
    app.include_router(router)
    set_guide_loader(app, lambda job_id: guide if job_id == "job-guide" else None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/guides/job-guide", timeout=1)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _page(browser, base, path, **kw):
    context = browser.new_context(**{"viewport": {"width": 1280, "height": 800}, **kw})
    page = context.new_page()
    page.goto(base + path)
    return context, page


def test_desktop_stage_offers_every_labelled_part(live_browser, multi_base_url):
    context, page = _page(live_browser, multi_base_url, "/guides/job-guide")
    try:
        labelled = page.eval_on_selector_all(
            ".card:first-of-type .part-shot img[data-labelled]",
            "els => els.map(e => e.getAttribute('data-labelled'))",
        )
        assert len(labelled) >= 2, "fixture must have a step with several labelled parts"
        buttons = page.locator("#gd-stage .gd-parts button")
        assert buttons.count() == len(labelled)
        buttons.nth(1).click()
        assert page.get_attribute("#gd-frame", "src") == labelled[1]
        assert buttons.nth(1).get_attribute("aria-pressed") == "true"
    finally:
        context.close()


def test_playback_across_steps_moves_the_stage(live_browser, live_base_url):
    context, page = _page(live_browser, live_base_url, "/guides/job-clip")
    try:
        assert page.inner_text("#gd-count") == "Step 1 / 2"
        # reader.js follows the clip's time; the stage must follow its card.
        page.evaluate(
            """() => { const v = document.getElementById('guide-video');
              Object.defineProperty(v, 'currentTime', {get: () => 18.5, configurable: true});
              v.dispatchEvent(new Event('timeupdate')); }"""
        )
        assert page.inner_text("#gd-count") == "Step 2 / 2"
        classes = page.eval_on_selector_all(".guide .card", "els => els.map(e => e.classList.contains('is-active'))")
        assert classes == [False, True]
    finally:
        context.close()


def test_switching_to_fixing_view_while_playing_restores_the_frame(live_browser, live_base_url):
    context, page = _page(live_browser, live_base_url, "/guides/job-clip")
    try:
        page.click("#gd-play")
        assert "is-playing" in page.get_attribute("#gd-stage", "class")
        page.click(".pf-view-label")
        assert "is-playing" not in page.get_attribute("#gd-stage", "class")
        box = page.locator("#gd-frame").bounding_box()
        assert box and box["width"] > 50 and box["height"] > 30
        assert not page.locator("#gd-play").is_visible()
    finally:
        context.close()


def test_without_script_cards_keep_their_photos_and_the_player(live_browser, live_base_url):
    context, page = _page(live_browser, live_base_url, "/guides/job-clip", java_script_enabled=False)
    try:
        assert page.locator(".card .shot img").first.is_visible()
        assert page.locator(".card .part-shot img").first.is_visible()
        assert page.locator("#guide-video").is_visible()
    finally:
        context.close()


def test_reader_stop_panel_text_is_readable(live_browser, live_base_url):
    context, page = _page(live_browser, live_base_url, "/reader/hero")
    try:
        worst, who = page.evaluate(CONTRAST_JS, "#main .stop")
        assert worst >= 4.5, (worst, who)
    finally:
        context.close()


def test_inactive_guide_cards_meet_text_contrast(live_browser, live_base_url):
    context, page = _page(live_browser, live_base_url, "/guides/job-guide")
    try:
        page.evaluate("document.querySelectorAll('.guide .card').forEach((c, i) => c.classList.toggle('is-active', i === 0))")
        page.mouse.move(0, 0)
        worst, who = page.evaluate(CONTRAST_JS, ".guide .card:not(.is-active)")
        assert worst >= 4.5, (worst, who)
    finally:
        context.close()


def test_phone_reduced_motion_shows_the_full_name_clear_of_the_nav(live_browser, live_base_url):
    context, page = _page(
        live_browser, live_base_url, "/guides/job-guide",
        viewport={"width": 390, "height": 800}, reduced_motion="reduce",
    )
    try:
        name = page.locator(".pf-wordmark")
        assert name.is_visible() and name.inner_text() == "PartFidelity"
        n = name.bounding_box()
        for link in page.locator(".pf-nav a").all():
            b = link.bounding_box()
            assert n["y"] + n["height"] <= b["y"] + 1 or b["x"] + b["width"] <= n["x"] or n["x"] + n["width"] <= b["x"]
            assert b["x"] >= 0 and b["x"] + b["width"] <= 390
        assert page.evaluate("document.documentElement.scrollWidth") <= 390
    finally:
        context.close()


def test_sample_guide_keeps_the_readers_catch_calls_and_hard_stop_links():
    from app.routes.guides import _sample_note_html

    note = _sample_note_html()
    for href in ('href="/catch"', 'href="/calls/b3f79c933e2043afc3906a159f473a2b"', 'href="/guide/sample#step-5"'):
        assert href in note
