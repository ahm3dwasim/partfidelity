"""Reader page-level tests: real browser plus static zero-network/a11y checks.

The browser tests drive headless Chromium (bundled Playwright build, or
system Chrome) against uvicorn serving the site app on 127.0.0.1. The
static tests keep the zero-network and alt/focus/contrast/motion checks
that need no browser. Local synthetic fixtures only, zero network.
"""

import html as html_lib
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.routes.site import create_site_app

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "site.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "reader.js").read_text(encoding="utf-8")

PAGES = ("/", "/guide/sample", "/reader/sample", "/reader/hero", "/reader/hero?ack=2")
ASSETS = ("/static/site.css", "/static/reader.js")

VIEWPORTS = [(390, 844), (1280, 800)]

_MISSING_BROWSER_MARKERS = (
    "Executable doesn't exist",
    "has not been found",
    "is not found",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_site_app())


@pytest.fixture
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("real network is refused in reader tests")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    return True


def test_reader_pages_use_zero_network(client, no_network):
    for path in PAGES + ASSETS:
        response = client.get(path)
        assert response.status_code == 200, path
    bodies = [client.get(path).text for path in PAGES] + [CSS, JS]
    # xmlns identifiers name an XML namespace; they are never fetched.
    scrubbed = [
        re.sub(r"xmlns='[^']+'", "", html_lib.unescape(body)) for body in bodies
    ]
    for body in scrubbed:
        assert "http://" not in body
        assert "https://" not in body
        assert "fetch(" not in body
        assert "XMLHttpRequest" not in body
        assert "@import" not in body
        assert "@@brand_" not in body
    for body in bodies[: len(PAGES)]:
        for match in re.findall(r'(?:src|href)="([^"]+)"', body):
            assert not match.startswith("//"), match


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


def _token(name: str) -> str:
    match = re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", CSS)
    assert match, name
    return match.group(1)


def test_alt_focus_contrast_reduced_motion(client):
    pages = {path: client.get(path).text for path in PAGES}
    # Alt text: every step image names its step and content. The home film's
    # poster is decorative (alt="") inside a wrapper that carries the label.
    images = re.findall(r"<img ([^>]+)>", "".join(pages.values()))
    steps = [tag for tag in images if 'class="film-poster"' not in tag]
    assert steps and len(images) - len(steps) == 1
    for tag in steps:
        alt = re.search(r'alt="([^"]+)"', tag)
        assert alt and alt.group(1).startswith("Step "), tag
    film_art = re.search(
        r'<div class="film-art" (?:id="film-art" )?role="img" aria-label="([^"]+)" aria-describedby="film-parts">\s*<img ([^>]+)>',
        pages["/"],
    )
    assert film_art and len(film_art.group(1)) > 20
    assert 'class="film-poster"' in film_art.group(2) and 'alt=""' in film_art.group(2)
    # The part names are a real list outside the aria-hidden labels, from config, in film order.
    from app import film_config
    listed = re.search(r'<ul class="film-parts" id="film-parts"[^>]*>(.*?)</ul>', pages["/"], re.S)
    assert listed and re.findall(r"<li>([^<]+)</li>", listed.group(1)) == [p["name"] for p in film_config.cached()["parts"]]
    # Keyboard: skip links, native controls, visible focus.
    for path, body in pages.items():
        assert 'class="skip"' in body, path
    # The only scripts on the home page are the film's config and its player.
    assert pages["/"].count("<script") == 2
    assert 'id="simple-view"' in pages["/reader/sample"]
    assert "<summary>Full maker sentence</summary>" in pages["/reader/sample"]
    assert "onclick" not in "".join(pages.values())
    assert ":focus-visible" in CSS
    # AA contrast on every text token pair the pages use.
    pairs = [
        ("ink", "paper"),
        ("muted", "paper"),
        ("accent-ink", "accent"),
        ("accent-ink", "danger"),
        ("danger", "danger-bg"),
        ("accent-ink", "medium-bg"),
        ("ink", "replay-bg"),
    ]
    for fg, bg in pairs:
        assert _ratio(_token(fg), _token(bg)) >= 4.5, (fg, bg)
    # Reduced motion: one media query kills all travel; the script moves nothing.
    assert "@media (prefers-reduced-motion" in CSS
    for banned in (
        "requestAnimationFrame",
        "setInterval",
        "setTimeout",
        "animate(",
        "transition",
    ):
        assert banned not in JS, banned


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Uvicorn serving the site app on 127.0.0.1; yields the base URL."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("l9-server") / "uvicorn.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.routes.site:create_site_app",
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


@pytest.mark.parametrize("width,height", VIEWPORTS)
def test_browser_hard_stop_blocks_until_acknowledged(
    browser, live_server, width, height
):
    errors, failed = [], []
    context = browser.new_context(viewport={"width": width, "height": height})
    try:
        page = _watched_page(context, errors, failed)
        page.goto(f"{live_server}/reader/hero")
        page.wait_for_load_state("networkidle")
        assert page.locator("#step-3").count() == 0
        assert "stay hidden until you acknowledge" in page.content()
        page.get_by_role("link", name="I understand").click()
        page.wait_for_selector("#step-3")
        assert "Remove the fan screws" in page.content()
        assert "Stop acknowledged." in page.content()
    finally:
        context.close()
    assert errors == []
    assert failed == []


def test_browser_primary_cta_opens_sample_guide(browser, live_server):
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = _watched_page(context, errors, failed)
        page.goto(f"{live_server}/")
        page.wait_for_load_state("networkidle")
        page.locator("#film-open").get_by_role("link", name="Explore the sample").click()
        page.wait_for_url("**/reader/sample")
        assert "Sample guide: fan cable reseat" in page.content()
    finally:
        context.close()
    assert errors == []
    assert failed == []


# Every home film call to action that is on screen sits above the fixed bottom bar,
# and no point inside its pill (corners inset by the radius, plus the centre) hits the bar, so the "See the sample guide" link is never partly covered by the fixed navigation.
_FILM_CTA_HITS = """() => {
  const out = [];
  for (const a of document.querySelectorAll('.film a.cta-primary, .film a.cta-ghost')) {
    const box = a.closest('.film-chap, .film-open'), cs = getComputedStyle(box);
    const r = a.getBoundingClientRect();
    if (r.width === 0 || r.top < 0 || r.top > innerHeight || +cs.opacity < 0.5 || cs.visibility === 'hidden') continue;
    const i = r.height / 2;
    const pts = [[r.left + i, r.top + 3], [r.right - i, r.top + 3], [r.left + i, r.bottom - 3],
                 [r.right - i, r.bottom - 3], [(r.left + r.right) / 2, (r.top + r.bottom) / 2]];
    const bar = document.querySelector('.pf-header');
    // A chapter fading in can briefly miss its own click; only the bar is the defect here.
    const ok = r.bottom <= bar.getBoundingClientRect().top
      && pts.every(([x, y]) => { const e = document.elementFromPoint(x, y); return !!e && !bar.contains(e); });
    out.push([a.textContent.trim(), ok]);
  }
  return out;
}"""


@pytest.mark.parametrize(
    "width,height", [(1280, 800), (1440, 900), (1920, 1080), (390, 844)]
)
def test_browser_bottom_bar_never_covers_a_film_button(
    browser, live_server, width, height
):
    context = browser.new_context(viewport={"width": width, "height": height})
    try:
        page = context.new_page()
        page.goto(f"{live_server}/")
        page.wait_for_load_state("networkidle")
        total = page.evaluate("document.documentElement.scrollHeight")
        seen, covered = set(), []
        for y in range(0, total, max(40, height // 12)):
            page.evaluate(f"scrollTo(0, {y})")
            # Two frames: film.js applies the scroll on the next animation frame.
            page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            for text, ok in page.evaluate(_FILM_CTA_HITS):
                seen.add(text)
                if not ok:
                    covered.append((y, text))
        assert "Explore the sample" in seen
        assert covered == []
    finally:
        context.close()
