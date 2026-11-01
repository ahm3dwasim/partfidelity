"""Computed-cascade browser tests for the UI shell.

Drives headless Chromium (bundled Playwright build, or system Chrome) over
all six shell pages and asserts what string tests cannot: real heading
fonts, the accent remap, dual-ring focus, skip-link behaviour, and measured
contrast ratios.
Follows the tests/test_reader_browser.py harness (uvicorn + `make browser`).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment

from app.brand_templates import make_env

from app import evidence
from app.contracts import Guide, JobId
from app.routes.calls import render_calls_page
from app.routes.guides import _render_page
from app.routes.site import router as site_router
from test_calls_page import FakeCtx

ROOT = Path(__file__).resolve().parents[1]

# One orange accent on ink for both themes; the old
# amber names are aliases of it, and on-paper is the fog text on ink.
ACCENT_RGB = "rgb(232, 112, 50)"
AMBER_RGB = ACCENT_RGB
DEEP_AMBER_RGB = ACCENT_RGB
ON_PAPER_RGB = "rgb(236, 236, 233)"  # --pf-on-paper: guide-end pill/links
DISPLAY_FACE = "David Libre"
BODY_FACE = "Roboto"

# Per-theme computed expectations: legacy --accent, focus outline, halo.
# D's focus is one accent outline set off the element, with no halo.
THEME_STYLE = {
    "pf-theme-start": (ACCENT_RGB, ACCENT_RGB, "none"),
    "pf-theme-guide": (ACCENT_RGB, ACCENT_RGB, "none"),
}


def _luminance(rgb: tuple[float, float, float]) -> float:
    def channel(value: float) -> float:
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(v) for v in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: str, second: str) -> float:
    """WCAG ratio between two computed `rgb(r, g, b)` strings."""

    def parse(value: str) -> tuple[float, float, float]:
        parts = value.removeprefix("rgb(").removesuffix(")").split(",")
        return tuple(int(p.strip()) / 255 for p in parts)  # type: ignore[return-value]

    light = _luminance(parse(first))
    dark = _luminance(parse(second))
    if dark > light:
        light, dark = dark, light
    return (light + 0.05) / (dark + 0.05)

# (path, theme class, display face). Six pages, one contract each.
PAGES = (
    ("/", "pf-theme-start", DISPLAY_FACE),
    ("/upload", "pf-theme-start", DISPLAY_FACE),
    ("/preview/shell-test", "pf-theme-start", DISPLAY_FACE),
    ("/reader/sample", "pf-theme-guide", DISPLAY_FACE),
    ("/guide/shell-test", "pf-theme-guide", DISPLAY_FACE),
    ("/calls/shell-test", "pf-theme-guide", DISPLAY_FACE),
)

_MISSING_BROWSER_MARKERS = (
    "Executable doesn't exist",
    "has not been found",
    "is not found",
)


def _jinja() -> Environment:
    return make_env(ROOT / "templates")


def create_shell_test_app() -> FastAPI:
    """Site app plus test-only renders of upload/preview/guide/calls pages."""
    app = FastAPI(title="PartFidelity shell test")
    app.mount("/static", StaticFiles(directory=ROOT / "static", html=False))
    app.include_router(site_router)
    from app.routes.uploads import _copy

    env = _jinja()

    @app.get("/upload", response_class=HTMLResponse)
    def upload() -> str:
        return env.get_template("upload.html").render(
            c=_copy(300.0), error="", blocked=False, blocked_copy="", limits_json="{}"
        )

    @app.get("/preview/shell-test", response_class=HTMLResponse)
    def preview() -> str:
        return env.get_template("preview.html").render(
            c=_copy(300.0),
            job_id="shell-test",
            duration="1.0",
            size="8",
            video_url="/preview/shell-test/video",
            show_video=False,
            state="consent",
            notice="",
            cancelled_copy="",
            hidden_copy="",
            meta_recipient="Meta",
            meta_hours="24",
            local_hours="24",
            verified_copy="",
            acknowledged_copy="",
        )

    @app.get("/guide/shell-test", response_class=HTMLResponse)
    def guide() -> str:
        return _render_page(
            "shell-test",
            Guide(
                steps=(),
                difficulty="Easy",
                verdict="Try it",
                job_id=JobId("shell-test"),
                frame_paths=(),
                labels=(),
                draft_reasons=(),
            ),
        )

    @app.get("/calls/shell-test", response_class=HTMLResponse)
    def calls() -> str:
        evidence.reset_stores()
        try:
            return render_calls_page("shell-test", FakeCtx(JobId("shell-test")))
        finally:
            evidence.reset_stores()

    return app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Uvicorn serving the shell test app on 127.0.0.1; yields base URL."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("shell-server") / "uvicorn.log"
    python_path = os.pathsep.join((str(ROOT), str(ROOT / "tests")))
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "test_ui_shell_browser:create_shell_test_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": python_path},
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


ACCENT_PROBE = (
    "() => {"
    " const probe = document.createElement('div');"
    " probe.style.color = 'var(--accent)';"
    " document.body.appendChild(probe);"
    " const color = getComputedStyle(probe).color;"
    " probe.remove();"
    " return color; }"
)


@pytest.mark.parametrize("path,theme,face", PAGES)
def test_browser_shell_contract(browser, live_server, path, theme, face):
    """Computed cascade per page: heading face, accent, focus, skip link."""
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = _watched_page(context, errors, failed)
        page.goto(f"{live_server}{path}")
        page.wait_for_load_state("networkidle")
        assert theme in page.content(), path

        # Real headings use the theme display face; body is Roboto.
        title_font = page.evaluate(
            "() => getComputedStyle(document.querySelector('main h1')).fontFamily"
        )
        assert face in title_font, (path, title_font)
        body_font = page.evaluate(
            "() => getComputedStyle(document.body).fontFamily"
        )
        assert BODY_FACE in body_font, (path, body_font)

        # The legacy --accent var resolves per theme (bright on ink,
        # deep on paper) and clears AA against the page background.
        want_accent, want_outline, want_halo = THEME_STYLE[theme]
        accent = page.evaluate(ACCENT_PROBE)
        assert accent == want_accent, (path, accent)
        body_bg = page.evaluate(
            "() => getComputedStyle(document.body).backgroundColor"
        )
        ratio = contrast_ratio(accent, body_bg)
        assert ratio >= 4.5, (path, accent, body_bg, ratio)

        # Shared skip link: one, labelled, targeting the page main.
        skips = page.locator("a.skip")
        assert skips.count() == 1, path
        assert skips.first.inner_text() == "Skip to content", path
        assert skips.first.get_attribute("href") == "#main", path
        assert page.locator("main#main").count() == 1, path

        # First Tab lands on the skip link with the dual-ring focus ...
        page.keyboard.press("Tab")
        focused = page.evaluate(
            "() => { const el = document.activeElement;"
            " const style = getComputedStyle(el);"
            " return [el.tagName, el.className,"
            " style.outlineColor, style.boxShadow]; }"
        )
        assert focused[0] == "A" and "skip" in focused[1], (path, focused)
        assert focused[2] == want_outline, (path, focused)
        assert want_halo in focused[3], (path, focused)
        focus_ratio = contrast_ratio(focused[2], body_bg)
        assert focus_ratio >= 3, (path, focused, body_bg, focus_ratio)

        # ... and two more Tabs reach header links, same dual ring (this is
        # the guide.css a:focus-visible override on guide pages).
        for _ in range(2):
            page.keyboard.press("Tab")
            ring = page.evaluate(
                "() => { const style = getComputedStyle(document.activeElement);"
                " return [style.outlineColor, style.boxShadow]; }"
            )
            assert ring[0] == want_outline, (path, ring)
            assert want_halo in ring[1], (path, ring)
            assert contrast_ratio(ring[0], body_bg) >= 3, (path, ring)

        # Activating the skip link jumps to the page main.
        page.locator("a.skip").first.focus()
        page.keyboard.press("Enter")
        page.wait_for_function("() => location.hash === '#main'")
        box = page.locator("main#main").first.bounding_box()
        assert box is not None and box["height"] > 0, path
    finally:
        context.close()
    assert errors == [], (path, errors)
    assert failed == [], (path, failed)


ACCENT_FILLS = (
    ("/", ".cta-primary", "backgroundColor", AMBER_RGB),
    # Merged with guide-end: the title-block pill is ink-filled by
    # guide-end.css (the difficulty hue survives as its ::before chip),
    # so the shell pins the exact ink, never green, instead of amber.
    ("/reader/sample", ".badge-easy", "backgroundColor", ON_PAPER_RGB),
)


@pytest.mark.parametrize("path,selector,prop,expected", ACCENT_FILLS)
def test_browser_accent_fills_stay_bright(browser, live_server, path, selector, prop, expected):
    """Accent fills render their exact merged color, not green/brown."""
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = _watched_page(context, errors, failed)
        page.goto(f"{live_server}{path}")
        page.wait_for_load_state("networkidle")
        assert page.locator(selector).count() >= 1, (path, selector)
        value = page.evaluate(
            "(args) => getComputedStyle(document.querySelector(args[0]))[args[1]]",
            [selector, prop],
        )
        assert value == expected, (path, selector, value)
    finally:
        context.close()
    assert errors == [], (path, errors)
    assert failed == [], (path, failed)


# Guide-page link text: deep amber where the shell remap governs, ink
# where guide-end.css owns the region (guide footer); both clear 4.5:1.
ACCENT_TEXT = (
    # Merged with guide-end: the guide footer links are ink by
    # guide-end.css, so the shell pins the exact ink, never green.
    ("/reader/sample", ".guide-foot a", ON_PAPER_RGB),
    ("/calls/shell-test", ".crumb a", DEEP_AMBER_RGB),
)


@pytest.mark.parametrize("path,selector,expected", ACCENT_TEXT)
def test_browser_accent_text_is_deep_and_aa(browser, live_server, path, selector, expected):
    """Guide link text renders its exact merged color and clears 4.5:1."""
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = _watched_page(context, errors, failed)
        page.goto(f"{live_server}{path}")
        page.wait_for_load_state("networkidle")
        assert page.locator(selector).count() >= 1, (path, selector)
        fg, bg = page.evaluate(
            "(args) => [getComputedStyle(document.querySelector(args[0])).color,"
            " getComputedStyle(document.body).backgroundColor]",
            [selector],
        )
        assert fg == expected, (path, selector, fg)
        ratio = contrast_ratio(fg, bg)
        assert ratio >= 4.5, (path, selector, fg, bg, ratio)
    finally:
        context.close()
    assert errors == [], (path, errors)
    assert failed == [], (path, failed)
