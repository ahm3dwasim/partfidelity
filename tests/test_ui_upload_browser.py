"""Upload page browser tests: live Chromium against uvicorn (a11y + perf).

Follows tests/test_reader_browser.py: uvicorn serves create_uploads_app
(mount-first: /static from disk, then the router) on 127.0.0.1. Local
synthetic fixtures only, zero network.
"""

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

VIEWPORTS = [(1440, 900), (390, 844)]

_MISSING_BROWSER_MARKERS = (
    "Executable doesn't exist",
    "has not been found",
    "is not found",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Uvicorn serving the uploads app on 127.0.0.1; yields the base URL."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("upload-server") / "uvicorn.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.routes.uploads:create_uploads_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                if proc.poll() is not None:
                    tail = log_path.read_bytes()[-2000:].decode("utf-8", "replace")
                    raise AssertionError(f"uvicorn exited early: {tail}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/upload", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise AssertionError("uvicorn never served /upload")
                time.sleep(0.05)
            yield f"http://127.0.0.1:{port}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.fail(
            "playwright is not installed; run `pip install -e '.[test]'` then `make browser`",
            pytrace=False,
        )
    with sync_playwright() as playwright:
        launched = None
        for kwargs in ({"headless": True}, {"headless": True, "channel": "chrome"}):
            try:
                launched = playwright.chromium.launch(**kwargs)
                break
            except Exception as exc:
                if not any(m in str(exc) for m in _MISSING_BROWSER_MARKERS):
                    raise
        if launched is None:
            pytest.fail(
                "no Chromium or Chrome binary found; run `make browser` to install Chromium",
                pytrace=False,
            )
        yield launched
        launched.close()


def _watched_page(context, errors, failed):
    page = context.new_page()

    def on_console(message):
        if message.type == "error":
            errors.append(message.text)

    def on_pageerror(exc):
        errors.append(str(exc))

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.on(
        "response",
        lambda response: failed.append(f"{response.status} {response.url}")
        if response.status >= 400
        else None,
    )
    return page


def test_browser_client_error_is_announced_and_focused(
    browser, live_server, tmp_path
):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 1024)
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = _watched_page(context, errors, failed)
        page.goto(f"{live_server}/upload")
        page.wait_for_load_state("networkidle")
        # Shrink the limit so a 1 KiB fixture trips the size refusal.
        page.evaluate("window.PARTFIDELITY_LIMITS.maxBytes = 10")
        page.set_input_files("#clip", str(clip))
        page.click('button[type="submit"]')
        alert = page.get_by_role("alert")
        alert.wait_for(state="visible")
        assert "300 seconds or less" in (alert.text_content() or "")
        assert page.evaluate("document.activeElement.id") == "client-error"
    finally:
        context.close()
    assert errors == []
    assert failed == []


@pytest.mark.parametrize("width,height", VIEWPORTS)
def test_browser_first_paint_under_budget(browser, live_server, width, height):
    context = browser.new_context(viewport={"width": width, "height": height})
    try:
        page = context.new_page()
        page.goto(f"{live_server}/upload")
        page.wait_for_load_state("networkidle")
        paint = page.evaluate(
            "performance.getEntriesByName('first-contentful-paint')[0].startTime"
        )
        print(f"\n[perf] FCP {width}x{height}: {paint:.0f} ms")
        assert paint < 2000
    finally:
        context.close()


@pytest.mark.parametrize("width,height", VIEWPORTS)
@pytest.mark.parametrize("throttle", [1, 4])
def test_browser_sending_sweep_holds_fps(
    browser, live_server, width, height, throttle
):
    context = browser.new_context(viewport={"width": width, "height": height})
    try:
        page = context.new_page()
        if throttle > 1:
            cdp = context.new_cdp_session(page)
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": throttle})
        page.goto(f"{live_server}/upload")
        page.wait_for_load_state("networkidle")
        page.evaluate(
            "document.querySelector('.pf-panel').classList.add('is-sending')"
        )
        fps = page.evaluate(
            """(ms) => new Promise((resolve) => {
              let frames = 0;
              const start = performance.now();
              function tick() {
                frames++;
                if (performance.now() - start < ms) requestAnimationFrame(tick);
                else resolve((frames * 1000) / (performance.now() - start));
              }
              requestAnimationFrame(tick);
            })""",
            2000,
        )
        print(f"\n[perf] sweep {width}x{height} {throttle}x CPU: {fps:.1f} fps")
        assert fps >= 50
    finally:
        context.close()


@pytest.mark.parametrize("width,height", VIEWPORTS)
def test_browser_fixed_bar_covers_no_control(browser, live_server, width, height):
    """At the page bottom, and for a focused control, nothing sits under the bottom bar."""
    context = browser.new_context(viewport={"width": width, "height": height})
    try:
        page = context.new_page()
        page.goto(f"{live_server}/upload")
        page.wait_for_load_state("networkidle")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        covered = page.evaluate(
            """() => {
              const bar = document.querySelector('.pf-header').getBoundingClientRect();
              return [...document.querySelectorAll('a, button, input, label, p, li, h1, h2, h3')]
                .filter((el) => !el.closest('.pf-header'))
                .filter((el) => {
                  const r = el.getBoundingClientRect();
                  return r.width && r.height && r.bottom > bar.top && r.top < bar.bottom;
                })
                .map((el) => el.tagName + ':' + el.textContent.trim().slice(0, 30));
            }"""
        )
        assert covered == []
        # A keyboard-focused control near the bottom edge scrolls clear of the bar.
        page.evaluate("window.scrollTo(0, 0)")
        page.focus('a[href="/delete-request"]')
        gap = page.evaluate(
            """() => document.querySelector('.pf-header').getBoundingClientRect().top
              - document.activeElement.getBoundingClientRect().bottom"""
        )
        assert gap >= 0
        # Anchors and tab-scrolling keep clear of the bar by CSS, not by luck.
        pad = page.evaluate(
            "parseFloat(getComputedStyle(document.documentElement).scrollPaddingBottom)"
        )
        assert pad >= 64
    finally:
        context.close()
