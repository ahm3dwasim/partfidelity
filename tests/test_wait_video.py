"""Waiting screen (own video, parts outlined live): static + browser tests.

Static checks cover the partial, CSS tokens/contrast/motion, JS syntax and
contract keys, and the gallery's zero-network rule. Browser tests drive the
gallery's fake status script plus a stub cancel endpoint in system Chrome,
and a mocked-fetch/image/timer harness for the race paths (stale polls,
load failure/retry, pending-frame repeats, clock cleanup/replay, and the
cancel-vs-done race).

The gallery and every test run on committed drawn stand-ins under
static/ui/media/wait-video/ (see tools/make_wait_video_standins.py), so a
clean checkout runs with no skips. Consented footage under
static/ui/media/local/ stays git-ignored and is never committed; nothing
references it.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app import brand_config

ROOT = Path(__file__).resolve().parents[1]
PARTIAL = ROOT / "templates" / "partials" / "ui" / "wait-video.html"
CSS = ROOT / "static" / "ui" / "wait-video.css"
JS = ROOT / "static" / "ui" / "wait-video.js"
GALLERY = ROOT / "static" / "ui" / "gallery" / "wait-video.html"
GSAP = ROOT / "static" / "ui" / "vendor" / "gsap.min.js"
SYN = ROOT / "static" / "ui" / "media" / "wait-video"
GENERATOR = ROOT / "tools" / "make_wait_video_standins.py"
HARNESS = ROOT / "tests" / "fixtures" / "wait_video_harness.html"
MOCK = ROOT / "tests" / "fixtures" / "wait_video_mock.html"


def test_partial_has_slots_and_overlay():
    body = PARTIAL.read_text(encoding="utf-8")
    assert 'id="wait-video"' in body
    assert 'viewBox="0 0 1280 720"' in body
    assert 'id="wv-overlay"' in body
    assert 'id="wv-clock"' in body
    assert 'id="wv-cancel"' in body
    assert 'id="wv-steps"' in body
    assert 'id="wv-stages"' in body
    for slot in ("wait.loading", "wait.elapsed", "wait.cancel", "wait.open", "wait.note"):
        assert f'data-copy="{slot}"' in body, slot
    # The integration contract is documented where the page author sees it.
    assert "job-status JSON" in body or "Status JSON contract" in body
    assert "polygon" in body
    # The monotonic status revision plus its backward-compatible default.
    assert '"revision"' in body
    assert "Missing revision means 0" in body
    # The cancel contract names the terminal response the deletion copy needs.
    assert "POST job_cancel_url" in body
    assert '"cancelled"' in body


def test_css_tokens_grid_motion_and_contrast():
    css = CSS.read_text(encoding="utf-8")
    for token, value in (("pf-ink", "#131313"), ("pf-amber", "#f0a400"),
                         ("pf-on-ink", "#f2f0eb"), ("pf-danger", "#c2331f")):
        assert f"var(--{token}, {value})" in css, token
    assert "repeat(12, 1fr)" in css
    assert "repeat(4, 1fr)" in css
    assert ":focus-visible" in css
    assert "@media (prefers-reduced-motion" in css
    assert ".wv-cancel.is-hidden" in css

    def luminance(hex_color: str) -> float:
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))

        def channel(v: float) -> float:
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

        r, g, b = (channel(v) for v in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def ratio(fg: str, bg: str) -> float:
        light, dark = sorted((luminance(fg), luminance(bg)), reverse=True)
        return (light + 0.05) / (dark + 0.05)

    assert ratio("#f2f0eb", "#131313") >= 4.5
    assert ratio("#a9a59c", "#131313") >= 4.5
    assert ratio("#f0a400", "#131313") >= 4.5
    assert ratio("#1a1300", "#f0a400") >= 4.5


def test_js_parses_and_uses_contract():
    node = shutil.which("node")
    assert node, "node is required to syntax-check wait-video.js"
    proc = subprocess.run([node, "--check", str(JS)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    js = JS.read_text(encoding="utf-8")
    # The SVG namespace names an XML namespace; it is never fetched.
    scrubbed = js.replace('"http://www.w3.org/2000/svg"', '""')
    assert "http://" not in scrubbed and "https://" not in scrubbed
    for key in ("statusUrl", "cancelUrl", "outlines", "polygon", "fakeScript",
                "prefers-reduced-motion", "strokeDashoffset", "WaitVideo",
                "AbortController", "pollSeq", "pendingFrameId", "startClock",
                "revision", "lastRevision"):
        assert key in js, key
    assert 'method: "POST"' in js
    # The guide link only accepts same-app targets (relative path or anchor),
    # so a hostile status payload cannot plant a javascript: href.
    assert '/(?!\\/)|#)/.test(status.guide_url' in js
    assert ".innerHTML" not in js
    # The cancel control hides without an endpoint, and the deleted-footage
    # copy appears only after a terminal cancellation response: the two
    # remaining mentions are the polled-cancelled state and the POST reply.
    assert "cancelUrl) this.cancelBtn.classList.add" in js
    assert 'body.state !== "cancelled"' in js
    assert js.count("footage was deleted") == 2
    # A repeat poll for the frame already loading must not restart it; the
    # pending id releases on load or error so a retry stays possible.
    assert "status.frame.id !== this.pendingFrameId" in js
    assert "frame.id === self.pendingFrameId) self.pendingFrameId = null" in js
    # A payload older than the last applied one is rejected, and a failed
    # cancel that lands after a terminal state changes nothing.
    assert "rev < this.lastRevision" in js
    assert "if (self.terminal) return;" in js


def test_gallery_is_zero_network_with_fake_script():
    body = GALLERY.read_text(encoding="utf-8")
    assert "http://" not in body and "https://" not in body
    for match in re.findall(r'(?:src|href)="([^"]+)"', body):
        assert not match.startswith("//"), match
        # The empty data: favicon suppresses the /favicon.ico fetch; it loads nothing.
        assert not match.startswith("data:") or match == "data:,", match
    assert "../vendor/gsap.min.js" in body
    assert "../wait-video.js" in body
    assert "../wait-video.css" in body
    # The gallery runs on committed stand-ins only; ignored footage is never
    # referenced, so a clean checkout renders it.
    assert "../media/wait-video/" in body
    assert "media/local" not in body
    assert GSAP.exists() and GSAP.stat().st_size > 10_000
    for stage in ("reading", "naming", "finding", "drawing", "checking"):
        assert f'"{stage}"' in body, stage
    assert 'state: "done"' in body
    assert "wv-replay" in body


def test_standin_frames_within_budget():
    expected = [SYN / f"synth-{tag}.jpg" for tag in ("s09", "s10", "s12", "s16")]
    for frame in expected:
        assert frame.exists(), (
            f"{frame.name} missing: regenerate with "
            "python3 tools/make_wait_video_standins.py")
    # Exactly the drawn set: real footage must never land in this directory.
    assert sorted(p.name for p in SYN.iterdir()) == sorted(f.name for f in expected)
    total = sum(f.stat().st_size for f in expected)
    assert total <= 3 * 1024 * 1024, total
    for frame in expected:
        assert frame.stat().st_size <= 1024 * 1024, frame.name
    assert GENERATOR.exists()


def test_cancel_harness_is_test_only_and_wired():
    body = HARNESS.read_text(encoding="utf-8")
    assert "test only" in body
    assert "/static/ui/wait-video.js" in body
    assert "/static/ui/wait-video.css" in body
    assert 'id="wv-cancel"' in body
    assert "data-cancel-url" in body
    assert "URLSearchParams" in body


def test_mock_harness_is_test_only_and_wired():
    body = MOCK.read_text(encoding="utf-8")
    assert "test only" in body
    assert "http://" not in body and "https://" not in body
    assert "/static/ui/wait-video.js" in body
    assert "/static/ui/wait-video.css" in body
    assert "gsap.min.js" not in body
    assert "__WV_FAKE__" in body
    assert "window.__wv" in body
    assert '"mode"' in body
    assert 'id="wv-img-a"' in body and 'id="wv-img-b"' in body
    assert 'id="wv-replay"' in body


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _BrandFileHandler(SimpleHTTPRequestHandler):
    """Plain file server, except .html is served with its brand tokens filled, as the app does."""

    def do_GET(self):
        path = Path(self.translate_path(self.path.split("?", 1)[0]))
        if path.suffix != ".html" or not path.is_file():
            return super().do_GET()
        body = brand_config.fill(path.read_text(encoding="utf-8")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def file_server():
    port = _free_port()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(_BrandFileHandler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=10)


class _CancelStubHandler(_BrandFileHandler):
    """file_server plus POST /__wv/cancel with a per-test (body, status) reply."""

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/__wv/cancel":
            self.send_error(404, "no stub for this path")
            return
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        payload, status = self.server.cancel_reply
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def wv_stub_server():
    port = _free_port()
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(_CancelStubHandler, directory=str(ROOT)))
    server.cancel_reply = ('{"state": "cancelled"}', 200)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=10)


@contextmanager
def _chrome():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.fail("playwright is not installed; run `pip install -e '.[test]'`",
                    pytrace=False)
    with sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(headless=True, channel="chrome")
        except Exception as exc:  # noqa: BLE001 - missing binary is an env failure
            pytest.fail(f"system Chrome is unavailable: {exc}", pytrace=False)
        try:
            yield launched
        finally:
            launched.close()


def test_browser_gallery_reaches_done_with_outlines(file_server):
    errors, failed = [], []
    with _chrome() as launched:
        page = launched.new_page(viewport={"width": 1440, "height": 900})
        try:
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("response", lambda r: failed.append(f"{r.status} {r.url}")
                    if r.status >= 400 else None)
            page.goto(f"{file_server}/static/ui/gallery/wait-video.html")
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("#wv-guide:not(.is-hidden)", timeout=25000)
            assert "Your guide is ready." in page.content()
            assert page.locator("#wv-overlay polygon").count() >= 2
            assert page.locator(".wv-step").count() == 4
            assert "BATTERY" in page.content().upper()
            clock = page.locator("#wv-clock").inner_text()
            assert re.fullmatch(r"\d+:\d\d", clock), clock
            final = page.locator(".wv-img.is-on").get_attribute("src")
            assert final.endswith("synth-s12.jpg"), final
        finally:
            page.close()
    assert errors == []
    assert failed == []


def test_browser_gallery_hides_cancel_without_endpoint(file_server):
    with _chrome() as launched:
        page = launched.new_page(viewport={"width": 390, "height": 844})
        try:
            page.goto(f"{file_server}/static/ui/gallery/wait-video.html")
            page.wait_for_load_state("networkidle")
            assert page.locator("#wv-cancel").is_hidden()
        finally:
            page.close()


def test_browser_cancel_success_shows_deletion_copy(wv_stub_server):
    server, base = wv_stub_server
    server.cancel_reply = ('{"state": "cancelled"}', 200)
    errors = []
    with _chrome() as launched:
        page = launched.new_page(viewport={"width": 390, "height": 844})
        try:
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/tests/fixtures/wait_video_harness.html?cancel=/__wv/cancel")
            page.wait_for_load_state("networkidle")
            assert page.locator("#wv-cancel").is_visible()
            page.get_by_role("button", name="Cancel and delete").click()
            page.get_by_text("footage was deleted").wait_for()
            page.wait_for_selector("#wv-cancel:disabled")
            assert "footage was deleted" in page.content()
        finally:
            page.close()
    assert errors == []


def test_browser_cancel_nonterminal_shows_no_deletion_copy(wv_stub_server):
    server, base = wv_stub_server
    server.cancel_reply = ('{"state": "active"}', 200)
    with _chrome() as launched:
        page = launched.new_page(viewport={"width": 390, "height": 844})
        try:
            page.goto(f"{base}/tests/fixtures/wait_video_harness.html?cancel=/__wv/cancel")
            page.wait_for_load_state("networkidle")
            page.get_by_role("button", name="Cancel and delete").click()
            page.get_by_text("Cancel failed").wait_for()
            assert "footage was deleted" not in page.content()
            assert page.locator("#wv-cancel:disabled").count() == 0
        finally:
            page.close()


def test_browser_reduced_motion_keeps_one_frame(file_server):
    with _chrome() as launched:
        context = launched.new_context(viewport={"width": 1440, "height": 900},
                                       reduced_motion="reduce")
        try:
            page = context.new_page()
            page.goto(f"{file_server}/static/ui/gallery/wait-video.html")
            page.wait_for_load_state("networkidle")
            page.wait_for_selector("#wv-guide:not(.is-hidden)", timeout=25000)
            # The viewer keeps the first still frame while text updates land.
            still = page.locator(".wv-img.is-on").get_attribute("src")
            assert still.endswith("synth-s09.jpg"), still
            assert page.locator(".wv-step").count() == 4
            assert "Your guide is ready." in page.content()
        finally:
            context.close()


@contextmanager
def _mock_page(file_server, launched, mode):
    """Mock-harness page: fetches, images and timers all park on __wv."""
    errors = []
    page = launched.new_page(viewport={"width": 1280, "height": 800})
    try:
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{file_server}/tests/fixtures/wait_video_mock.html?mode={mode}")
        page.wait_for_load_state("networkidle")
        yield page, errors
    finally:
        page.close()


def _respond(page, index, payload, ok=True):
    assert page.evaluate(
        "([i, ok, body]) => __wv.respond(i, ok, body)", [index, ok, payload])


def _frame_payload(revision, frame_id, **extra):
    payload = {"state": "active", "revision": revision,
               "frame": {"id": frame_id,
                         "src": f"/static/ui/media/wait-video/synth-{frame_id}.jpg"}}
    payload.update(extra)
    return payload


def test_browser_mock_stale_revision_ignored(file_server):
    with _chrome() as launched, _mock_page(file_server, launched, "live") as (page, errors):
        assert page.evaluate("__wv.state()")["pollSeq"] == 1
        fresh = _frame_payload(5, "s12", stages=[
            {"id": "reading", "label": "Reading", "state": "done"},
            {"id": "naming", "label": "Naming", "state": "active"}])
        _respond(page, 0, fresh)
        assert page.evaluate("__wv.state()")["lastRevision"] == 5
        assert page.evaluate("__wv.imgLoad('wv-img-b')") is True
        assert page.evaluate("__wv.state()")["frameId"] == "s12"
        # A newer request carrying an older snapshot must not rewind the UI.
        page.evaluate("ms => __wv.tick(ms)", 1500)
        assert page.evaluate("__wv.state()")["pollSeq"] == 2
        stale = _frame_payload(3, "s10", stages=[
            {"id": "reading", "label": "Reading", "state": "active"},
            {"id": "naming", "label": "Naming", "state": "pending"}])
        _respond(page, 1, stale)
        state = page.evaluate("__wv.state()")
        assert state["lastRevision"] == 5
        assert state["frameId"] == "s12"
        assert len(page.evaluate("__wv.srcSets()")) == 1
        classes = page.evaluate(
            "[...document.querySelectorAll('#wv-stages li')].map(e => e.className)")
        assert classes == ["is-done", "is-active"]
        # The boundary is lower-only: an equal revision still applies.
        page.evaluate("ms => __wv.tick(ms)", 1500)
        _respond(page, 2, _frame_payload(5, "s10"))
        state = page.evaluate("__wv.state()")
        assert state["lastRevision"] == 5
        assert state["pendingFrameId"] == "s10"
        assert len(page.evaluate("__wv.srcSets()")) == 2
    assert errors == []


def test_browser_mock_frame_error_then_retry(file_server):
    with _chrome() as launched, _mock_page(file_server, launched, "live") as (page, errors):
        _respond(page, 0, _frame_payload(1, "s09"))
        assert page.evaluate("__wv.imgLoad('wv-img-b')") is True
        assert page.evaluate("__wv.state()")["frameId"] == "s09"
        # The next frame fails: the old frame stays, pending releases.
        page.evaluate("ms => __wv.tick(ms)", 1500)
        _respond(page, 1, _frame_payload(2, "s12"))
        assert page.evaluate("__wv.state()")["pendingFrameId"] == "s12"
        failing = page.evaluate("__wv.srcSets()[__wv.srcSets().length - 1].id")
        assert page.evaluate("id => __wv.imgError(id)", failing) is True
        state = page.evaluate("__wv.state()")
        assert state["frameId"] == "s09"
        assert state["pendingFrameId"] is None
        # A later poll for the same frame retries the download.
        page.evaluate("ms => __wv.tick(ms)", 1500)
        _respond(page, 2, _frame_payload(3, "s12"))
        assert len(page.evaluate("__wv.srcSets()")) == 3
        retry = page.evaluate("__wv.srcSets()[__wv.srcSets().length - 1].id")
        assert page.evaluate("id => __wv.imgLoad(id)", retry) is True
        assert page.evaluate("__wv.state()")["frameId"] == "s12"
    assert errors == []


def test_browser_mock_pending_frame_not_restarted(file_server):
    with _chrome() as launched, _mock_page(file_server, launched, "live") as (page, errors):
        _respond(page, 0, _frame_payload(1, "s12"))
        assert page.evaluate("__wv.state()")["pendingFrameId"] == "s12"
        assert len(page.evaluate("__wv.srcSets()")) == 1
        # Repeat polls for the frame already loading set no new src.
        for index, revision in ((1, 2), (2, 3)):
            page.evaluate("ms => __wv.tick(ms)", 1500)
            _respond(page, index, _frame_payload(revision, "s12"))
            assert page.evaluate("__wv.state()")["pendingFrameId"] == "s12"
            assert len(page.evaluate("__wv.srcSets()")) == 1
        loading = page.evaluate("__wv.srcSets()[0].id")
        assert page.evaluate("id => __wv.imgLoad(id)", loading) is True
        state = page.evaluate("__wv.state()")
        assert state["frameId"] == "s12"
        assert state["pendingFrameId"] is None
        # A new frame id still loads after the promotion.
        page.evaluate("ms => __wv.tick(ms)", 1500)
        _respond(page, 3, _frame_payload(4, "s10"))
        assert page.evaluate("__wv.state()")["pendingFrameId"] == "s10"
        assert len(page.evaluate("__wv.srcSets()")) == 2
    assert errors == []


def test_browser_mock_clock_cleanup_and_restart(file_server):
    with _chrome() as launched, _mock_page(file_server, launched, "script") as (page, errors):
        assert page.evaluate("__wv.calls()") == []
        page.evaluate("ms => __wv.tick(ms)", 0)
        assert page.evaluate("__wv.state()")["steps"] == 1
        first = page.evaluate("__wv.srcSets()[0].id")
        assert page.evaluate("id => __wv.imgLoad(id)", first) is True
        page.evaluate("ms => __wv.tick(ms)", 10000)
        state = page.evaluate("__wv.state()")
        assert state["terminal"] is True
        assert state["tickOn"] is False
        assert state["clock"] == "0:10"
        assert state["guideHidden"] is False
        assert state["steps"] == 2
        # The clock stays frozen once the terminal state clears it.
        page.evaluate("ms => __wv.tick(ms)", 60000)
        assert page.evaluate("__wv.state()")["clock"] == "0:10"
        # Replay restarts the clock and the script from zero.
        page.get_by_role("button", name="Replay demo").click()
        state = page.evaluate("__wv.state()")
        assert state["terminal"] is False
        assert state["steps"] == 0
        assert state["tickOn"] is True
        assert state["frameId"] is None
        assert state["pendingFrameId"] is None
        page.evaluate("ms => __wv.tick(ms)", 250)
        state = page.evaluate("__wv.state()")
        assert state["steps"] == 1
        assert state["clock"] == "0:00"
    assert errors == []


def test_browser_mock_replay_clears_pending_script_timers(file_server):
    with _chrome() as launched, _mock_page(file_server, launched, "script") as (page, errors):
        page.evaluate("ms => __wv.tick(ms)", 1000)
        assert page.evaluate("__wv.state()")["steps"] == 1
        assert len(page.evaluate("__wv.srcSets()")) == 1
        page.get_by_role("button", name="Replay demo").click()
        assert page.evaluate("__wv.state()")["steps"] == 0
        assert page.evaluate("__wv.state()")["pendingFrameId"] is None
        # Only the fresh run's first entry has fired: the old timers died.
        page.evaluate("ms => __wv.tick(ms)", 4000)
        state = page.evaluate("__wv.state()")
        assert state["steps"] == 1
        assert state["pendingFrameId"] == "s09"
        assert len(page.evaluate("__wv.srcSets()")) == 2
        page.evaluate("ms => __wv.tick(ms)", 20000)
        state = page.evaluate("__wv.state()")
        assert state["terminal"] is True
        assert state["steps"] == 2
        assert state["guideHidden"] is False
        assert len(page.evaluate("__wv.srcSets()")) == 3
    assert errors == []


def test_browser_mock_cancel_race_with_done_keeps_terminal(file_server):
    with _chrome() as launched, _mock_page(file_server, launched, "live") as (page, errors):
        _respond(page, 0, {"state": "active", "revision": 1})
        page.get_by_role("button", name="Cancel and delete").click()
        state = page.evaluate("__wv.state()")
        assert state["note"] == "Cancelling…"
        assert state["cancelDisabled"] is True
        parked = page.evaluate("__wv.parked()")
        assert [call["method"] for call in page.evaluate("__wv.calls()")] == ["GET", "POST"]
        assert parked[1]["method"] == "POST"
        assert parked[1]["settled"] is False
        # Done lands while the cancel POST is still in flight.
        page.evaluate("ms => __wv.tick(ms)", 1500)
        _respond(page, 2, {"state": "done", "revision": 2, "guide_url": "#guide-ready"})
        state = page.evaluate("__wv.state()")
        assert state["terminal"] is True
        assert state["note"] == "Your guide is ready."
        assert state["cancelDisabled"] is True
        assert state["guideHidden"] is False
        # The late cancel failure must not touch the terminal state.
        assert page.evaluate("i => __wv.fail(i, 'Error')", 1) is True
        state = page.evaluate("__wv.state()")
        assert state["note"] == "Your guide is ready."
        assert state["cancelDisabled"] is True
        assert state["guideHidden"] is False
    assert errors == []
