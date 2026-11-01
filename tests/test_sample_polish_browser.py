"""Real-browser checks on the shipped sample guide: caption follows the step, nothing under the bar."""
from __future__ import annotations

import socket
import threading
import time
import urllib.request

import pytest
import uvicorn

from tools.sample_media import ROOT, FILES

pytestmark = pytest.mark.skipif(any(not (ROOT / "app/sample_real" / name).is_file() for name in FILES), reason="real sample media absent; run python tools/sample_media.py unpack")

from app import db
from app.main import create_app
from app.settings import Settings
from test_ui_guide_step import live_browser  # noqa: F401 (fixture)


@pytest.fixture(scope="module")
def sample_url(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample-polish")
    db.configure(root / "db.sqlite3", root / "data")
    db.init_db(root / "db.sqlite3")
    settings = Settings(data_root=root / "data", db_path=root / "db.sqlite3", production=False, pipeline_mode="replay")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/guide/sample", timeout=2)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/guide/sample"
    server.should_exit = True
    thread.join(timeout=10)


def _open(browser, url, width, height):
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    page.add_init_script("""(() => {
      const Original = window.IntersectionObserver;
      window.IntersectionObserver = class extends Original {
        observe(target) {
          if (target.classList.contains('card')) window.__cardObserverMargin = this.rootMargin;
          return super.observe(target);
        }
      };
    })()""")
    page.goto(url)
    page.wait_for_selector("#gd-stage.has-stage")
    return context, page


def _stage_state(page):
    return page.evaluate(
        """() => ({count: document.getElementById('gd-count').textContent,
          start: document.getElementById('gd-start').textContent, end: document.getElementById('gd-end').textContent,
          title: document.getElementById('guide-now-title').textContent, now: document.getElementById('guide-now-time').textContent})"""
    )


EXPECT = {  # step: (counter, start, end, now-playing time, start second)
    5: ("Step 5 / 8", "1:27", "1:52", "~1:27–1:52", 86.8333),
    6: ("Step 6 / 8", "1:52", "2:15", "~1:52–2:15", 111.533),
    8: ("Step 8 / 8", "2:50", "3:09", "~2:50–3:09", 170.433),
}


@pytest.mark.parametrize("width,height", [(1440, 900), (390, 844)])
def test_caption_counter_time_and_play_target_follow_the_step_on_scroll(live_browser, sample_url, width, height):
    context, page = _open(live_browser, sample_url, width, height)
    try:
        margins = page.evaluate("window.__cardObserverMargin.split(' ')")
        assert margins[0].endswith('px') and margins[2].endswith('px')
        assert height + float(margins[0][:-2]) + float(margins[2][:-2]) > 0
        assert _stage_state(page)["count"] == "Intro"
        for n, (count, start, end, now, seek) in EXPECT.items():
            page.evaluate(f"document.getElementById('step-{n}').scrollIntoView({{block: 'center'}})")
            page.wait_for_function(f"document.getElementById('gd-count').textContent === {count!r}")
            state = _stage_state(page)
            assert (state["start"], state["end"], state["now"]) == (start, end, now), (n, state)
            assert state["title"].startswith(f"Step {n}:"), (n, state)
    finally:
        context.close()


def test_play_targets_the_active_steps_start(live_browser, sample_url):
    context, page = _open(live_browser, sample_url, 1440, 900)
    try:
        for n, (count, start, *_rest) in EXPECT.items():
            page.click(f"#step-{n} h2")
            assert page.inner_text("#gd-count") == count
            page.click("#gd-play")
            # The toast names the second Play seeks to; the clip itself may not decode in this browser.
            assert page.inner_text("#gd-toast") == f"Playing from {start}", n
            page.evaluate("document.getElementById('guide-video').pause()")
    finally:
        context.close()


def test_clicking_a_step_and_playing_keep_the_same_caption(live_browser, sample_url):
    context, page = _open(live_browser, sample_url, 1440, 900)
    try:
        page.click("#step-8 h2")
        assert _stage_state(page)["title"].startswith("Step 8:")
        page.click("#gd-play")
        page.wait_for_function("!document.getElementById('guide-video').paused")
        page.evaluate(
            """() => { const v = document.getElementById('guide-video');
              Object.defineProperty(v, 'currentTime', {get: () => 100, configurable: true});
              v.dispatchEvent(new Event('timeupdate')); }"""
        )
        state = _stage_state(page)
        assert state["count"] == "Step 5 / 8" and state["now"] == "~1:27–1:52" and state["start"] == "1:27"
    finally:
        context.close()


def test_step_5_says_the_disconnect_is_not_filmed_and_step_8_claims_no_more_than_the_frame(live_browser, sample_url):
    context, page = _open(live_browser, sample_url, 1440, 900)
    try:
        page.evaluate("document.getElementById('step-5').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 5 / 8'")
        assert page.is_visible("#step-5 .gd-step-note")
        assert "not shown" in page.inner_text("#step-5 .gd-step-note").lower() or "does not show" in page.inner_text("#step-5 .gd-step-note")
        assert "does not show the battery being disconnected" in page.inner_text("#gd-stage-note")
        page.evaluate("document.getElementById('step-8').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 8 / 8'")
        assert "does not show the new battery listed" in page.inner_text("#gd-stage-note")
        page.evaluate("document.getElementById('step-1').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 1 / 8'")
        assert page.is_hidden("#gd-stage-note")
    finally:
        context.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (1440, 720), (390, 800)])
def test_nothing_sits_under_the_bottom_bar(live_browser, sample_url, width, height):
    context, page = _open(live_browser, sample_url, width, height)
    try:
        bar_top = lambda: page.evaluate("document.querySelector('.pf-header').getBoundingClientRect().top")  # noqa: E731
        # Last step's controls and the Done box: focusing them scrolls them clear of the bar.
        for selector in ("#step-8 .gd-done input", "#step-8 [data-set-start]", ".guide-downloads a"):
            page.evaluate(f"document.querySelector({selector!r}).focus()")
            page.wait_for_timeout(150)
            bottom = page.evaluate(f"document.querySelector({selector!r}).getBoundingClientRect().bottom")
            assert bottom <= bar_top() + 0.5, (selector, bottom, bar_top())
        # End of the page: the last text ends above the bar.
        page.evaluate("scrollTo(0, document.documentElement.scrollHeight)")
        page.wait_for_timeout(150)
        assert page.evaluate("document.querySelector('.pf-footer').getBoundingClientRect().bottom") <= bar_top() + 0.5
        if width >= 1024:
            # The pinned stage (frame, controls, caption, note, now-playing) ends above the bar at every step.
            for n in (1, 5, 8):
                page.evaluate(f"document.getElementById('step-{n}').scrollIntoView({{block: 'center'}})")
                page.wait_for_function(f"document.getElementById('gd-count').textContent === 'Step {n} / 8'")
                assert page.evaluate("document.getElementById('gd-stage').getBoundingClientRect().bottom") <= bar_top() + 0.5, n
    finally:
        context.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (390, 844)])
def test_pause_restores_scroll_tracking_and_time_changes_refresh_stage(live_browser, sample_url, width, height):
    context, page = _open(live_browser, sample_url, width, height)
    try:
        page.set_default_timeout(5000)
        page.click('#step-5 h2')
        page.click('#gd-play')
        page.wait_for_function("!document.getElementById('guide-video').paused")
        page.wait_for_function("!document.getElementById('guide-video').seeking")
        page.locator('#guide-video').focus()
        page.keyboard.press('Space')
        page.wait_for_function("document.getElementById('guide-video').paused")
        page.evaluate("document.getElementById('step-6').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 6 / 8'")
        assert page.locator("#gd-stage").evaluate("el => !el.classList.contains('is-playing')")
        assert page.locator("#gd-stage img").first.is_visible()
        page.locator('#step-6 [data-nudge="5"]').evaluate('(button) => button.click()')
        assert page.inner_text('#gd-start') == '1:57'
        page.click('#step-5 h2')
        assert page.inner_text('#gd-end') == '1:57'
    finally:
        context.close()


def test_selecting_part_keeps_keyboard_focus(live_browser, sample_url):
    context, page = _open(live_browser, sample_url, 1440, 900)
    try:
        page.evaluate("document.getElementById('step-5').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 5 / 8'")
        buttons = page.locator('.gd-parts button')
        assert buttons.count() > 1
        buttons.nth(1).focus()
        page.keyboard.press('Enter')
        assert buttons.nth(1).evaluate('(el) => document.activeElement === el')
        assert buttons.nth(1).get_attribute('aria-pressed') == 'true'
    finally:
        context.close()


def test_scroll_band_survives_resize(live_browser, sample_url):
    context, page = _open(live_browser, sample_url, 1440, 900)
    try:
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("document.getElementById('step-5').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 5 / 8'")
        page.set_viewport_size({"width": 1440, "height": 900})
        page.evaluate("document.getElementById('step-8').scrollIntoView({block: 'center'})")
        page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 8 / 8'")
    finally:
        context.close()


@pytest.mark.parametrize("width,height,player_first", [(390, 844, True), (1440, 900, False)])
def test_initial_presentation_phone_player_desktop_frame(live_browser, sample_url, width, height, player_first):
    context, page = _open(live_browser, sample_url, width, height)
    try:
        assert page.locator("#guide-video").is_visible() is player_first
        assert page.locator("#gd-stage img").first.is_visible() is not player_first
        if player_first:  # the paused-scroll state brings the still frame back
            page.evaluate("document.getElementById('step-3').scrollIntoView({block: 'center'})")
            page.wait_for_function("document.getElementById('gd-count').textContent === 'Step 3 / 8'")
            assert page.locator("#gd-stage").evaluate("el => el.classList.contains('is-scrubbed')")
            assert page.locator("#gd-stage img").first.is_visible()
            assert not page.locator("#guide-video").is_visible()
    finally:
        context.close()
