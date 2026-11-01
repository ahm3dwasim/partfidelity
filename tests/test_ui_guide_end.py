"""Guide-end component tests: title block, tools
checklist, share section, guide footer. Static checks run on every pass;
the browser tests drive headless Chromium against the site app.

Contract under test: the server renders the no-JS baseline (summary,
side plate, checklist, share section) and the script only enhances it
(persisted ticks, copy button). The copy control only ever uses a
server-issued expiring URL; without one the share section fails closed
(download + note only, no 24-hour promise)."""

import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.routes.site import create_site_app, guide_summary, share_html
from test_reader_browser import browser, live_server  # noqa: F401  (shared fixtures)

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "ui" / "guide-end.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "ui" / "guide-end.js").read_text(encoding="utf-8")
PARTIAL = (
    ROOT / "templates" / "partials" / "ui" / "guide-end.html"
).read_text(encoding="utf-8")
GALLERY = (
    ROOT / "static" / "ui" / "gallery" / "guide-end.html"
).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_site_app())


def test_reader_pages_link_guide_end_assets(client):
    for path in ("/reader/sample", "/reader/hero"):
        body = client.get(path).text
        assert 'href="/static/ui/guide-end.css"' in body, path
        assert 'src="/static/ui/guide-end.js"' in body, path
    for asset in ("/static/ui/guide-end.css", "/static/ui/guide-end.js"):
        response = client.get(asset)
        assert response.status_code == 200, asset


def test_guide_end_zero_network():
    for body in (CSS, JS):
        assert "http://" not in body
        assert "https://" not in body
        assert "fetch(" not in body
        assert "XMLHttpRequest" not in body
        assert "@import" not in body
    for match in re.findall(r'(?:src|href)="([^"]+)"', GALLERY):
        assert not match.startswith("//"), match
        assert "://" not in match, match


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


def _fallback(name: str) -> str:
    match = re.search(rf"--pf-{name},\s*(#[0-9a-fA-F]{{6}})", CSS)
    assert match, name
    return match.group(1)


def test_guide_end_text_contrast_is_aa():
    pairs = [
        ("on-paper", "paper"),
        ("on-paper-muted", "paper"),
        ("amber-ink", "amber"),
        ("paper-2", "on-paper"),
        ("amber", "on-paper"),
    ]
    for fg, bg in pairs:
        assert _ratio(_fallback(fg), _fallback(bg)) >= 4.5, (fg, bg)


def test_difficulty_markers_are_amber_or_neutral():
    assert "--pf-ok" not in CSS
    assert "--pf-danger" not in CSS
    assert "#2f7d4f" not in CSS
    assert "#c2331f" not in CSS


def test_share_keys_match_server_and_gallery():
    partial_keys = set(re.findall(r'data-copy="([^"]+)"', PARTIAL))
    assert partial_keys, "partial must carry copy keys"
    expiring = share_html("https://share.example.invalid/g/demo-1")
    expiring_keys = set(re.findall(r'data-copy="([^"]+)"', expiring))
    assert partial_keys == expiring_keys, partial_keys ^ expiring_keys
    closed = share_html(None)
    closed_keys = set(re.findall(r'data-copy="([^"]+)"', closed))
    gallery_keys = set(re.findall(r'data-copy="([^"]+)"', GALLERY))
    assert closed_keys <= gallery_keys, closed_keys - gallery_keys
    # The gallery shows the no-link state only: no share-link backend
    # ships yet, so no consent tick, copy row, expiry promise, or
    # fixture URL may appear anywhere on the page.
    for hook in ("data-ge-consent", "data-ge-link", "data-ge-copy"):
        assert hook not in GALLERY, hook
    assert "data-share-url" not in GALLERY
    assert "24-h" not in GALLERY
    assert "24 hours" not in GALLERY
    assert "example.invalid" not in GALLERY
    for hook in ("data-ge-download", "data-ge-status"):
        assert hook in PARTIAL, hook
        assert hook in expiring, hook
        assert hook in GALLERY, hook


def test_gallery_uses_committed_fixture():
    assert "../media/local/" not in GALLERY, "gallery must not use ignored media/"
    refs = re.findall(r"\.\./media/guide-end/([A-Za-z0-9_.-]+)", GALLERY)
    assert refs, "gallery must load frames from media/guide-end/"
    total = 0
    for ref in refs:
        assert ref.endswith(".webp"), ref
        rel = Path("static/ui/media/guide-end") / ref
        path = ROOT / rel
        assert path.is_file(), ref
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--", str(rel)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert tracked, f"{rel} is not tracked; a clean clone would 404"
        ignored = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", str(rel)],
        ).returncode
        assert ignored != 0, f"{rel} is git-ignored; a clean clone would 404"
        total += path.stat().st_size
    assert total <= 3 * 1024 * 1024, total


def test_step_cards_left_to_sibling_lane():
    for selector in (".step", "article.step", ".chips", ".fig-toggle"):
        assert selector not in CSS, selector


def test_no_js_baseline_renders_server_side(client):
    """TestClient runs no JS: title, summary, plate, checklist and the
    fail-closed share section must all be in the raw HTML."""
    expected = {"/reader/sample": "3 steps", "/reader/hero": "4 steps"}
    for path, count in expected.items():
        body = client.get(path).text
        assert '<div class="ge-title">' in body, path
        meta = re.search(r'<p class="ge-meta"[^>]*>(.*?)</p>', body)
        assert meta, path
        assert count in meta.group(1), (path, meta.group(1))
        assert "about 1 minute" in meta.group(1), (path, meta.group(1))
        assert 'class="ge-plate"' in body, path
        assert body.count('class="ge-check"') == 2, path
        assert 'class="ge-share"' in body, path
        assert "data-ge-download" in body, path
        assert "data-ge-status" in body, path
        # Fail closed: no server-issued URL, so no copy control, no
        # consent, no 24-hour promise anywhere on the page.
        assert "data-share-url" not in body, path
        assert "data-ge-copy" not in body, path
        assert "data-ge-link" not in body, path
        assert "data-ge-consent" not in body, path
        assert "24-h" not in body, path
        assert "24 hours" not in body, path


def test_share_script_never_falls_back_to_page_url():
    assert "location.href" not in JS
    assert "data-share-url" in JS


def test_share_builder_fails_closed_without_url():
    closed = share_html(None)
    # None, empty, and whitespace-only render the identical closed
    # section, so JS (which trims the attribute) and no-JS agree.
    for blank in ("", "   ", "\t \n"):
        assert share_html(blank) == closed, repr(blank)
    assert "data-ge-download" in closed
    assert "data-ge-status" in closed
    for banned in (
        "data-ge-copy",
        "data-ge-link",
        "data-ge-consent",
        "data-share-url",
        "24-h",
        "24 hours",
    ):
        assert banned not in closed, banned
    url = "https://share.example.invalid/g/demo-1"
    open_ = share_html(url)
    assert f'value="{url}"' in open_
    assert "data-ge-copy" in open_
    assert "24 hours" in open_
    # A padded real URL renders trimmed, matching the script's read.
    assert f'value="{url}"' in share_html(f"  {url}\t")


def test_guide_summary_rounds_minutes_half_up_like_js():
    def summary(span_s: float) -> str:
        return guide_summary(
            {
                "steps": [
                    {"visible": True, "start_s": 0.0, "end_s": span_s}
                ],
                "difficulty": "Easy",
            }
        )

    # .5 minutes round up exactly like Math.round (banker's round()
    # would give 2 and 4 here, diverging from the script).
    assert "about 3 minutes" in summary(150.0)
    assert "about 5 minutes" in summary(270.0)
    # Non-boundary spans are unchanged.
    assert "about 1 minute" in summary(44.0)
    assert "about 2 minutes" in summary(149.0)
    assert "about 3 minutes" in summary(151.0)


def test_gallery_carries_summary_data():
    meta = re.search(r'<p class="ge-meta"[^>]*>(.*?)</p>', GALLERY)
    assert meta, "gallery must render the summary explicitly"
    assert "3 steps" in meta.group(1)
    assert "Medium" in meta.group(1)
    steps = re.findall(r'<article class="step"[^>]*>', GALLERY)
    assert len(steps) == 3, "gallery needs representative steps"
    times = re.findall(r'<p class="time">(.*?)</p>', GALLERY)
    assert len(times) == 3, "each representative step needs a time"


def test_gallery_share_blocks_match_server_builder():
    """The gallery embeds the no-link share_html() output verbatim,
    exactly once: no share-link backend ships yet, so no URL on the
    page may look like a live expiring link."""
    assert '<main id="main">' in GALLERY, "gallery main carries no share URL"
    assert "partfidelity.link" not in GALLERY
    assert "data-share-url" not in GALLERY
    urls = re.findall(r'https?://[^"\s<>]+', GALLERY)
    assert urls == [], urls
    blocks = re.findall(
        r'<section class="ge-share" aria-label="Share this guide">.*?</section>',
        GALLERY,
        re.DOTALL,
    )
    assert len(blocks) == 1, "no-link section only"
    assert blocks[0] == share_html(None)


def _collect(page):
    errors, failed = [], []
    page.on(
        "console",
        lambda message: errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on(
        "response",
        lambda response: failed.append(f"{response.status} {response.url}")
        if response.status >= 400
        else None,
    )
    return errors, failed


def test_browser_reader_fail_closed_and_persistent(browser, live_server):
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        page_errors, page_failed = _collect(page)
        page.goto(f"{live_server}/reader/sample")
        page.wait_for_load_state("networkidle")
        assert "3 steps" in page.locator(".ge-meta").inner_text()
        assert page.locator(".ge-plate").count() == 1
        assert page.locator(".ge-check input").count() == 2
        # Fail closed in the live DOM too: the script must not re-add a
        # copy control when the server issued no URL.
        assert page.locator("[data-ge-copy]").count() == 0
        assert page.locator("[data-ge-link]").count() == 0
        assert page.locator("[data-ge-consent]").count() == 0
        assert page.locator("[data-ge-download]").is_enabled()
        page.locator(".ge-check input").first.check()
        page.get_by_role("button", name="Download images").click()
        page.wait_for_selector("[data-ge-status]:has-text('Downloading 3 images')")
        page.reload()
        page.wait_for_load_state("networkidle")
        assert page.locator(".ge-check input").first.is_checked()
        page.goto(f"{live_server}/reader/hero")
        page.wait_for_load_state("networkidle")
        assert "4 steps" in page.locator(".ge-meta").inner_text()
        errors, failed = page_errors, page_failed
    finally:
        context.close()
    assert errors == []
    assert failed == []


def test_browser_gallery_fail_closed_and_viewports(browser, live_server):
    errors, failed = [], []
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        page_errors, page_failed = _collect(page)
        page.goto(f"{live_server}/static/ui/gallery/guide-end.html")
        page.wait_for_load_state("networkidle")
        for width, height in ((390, 844), (1440, 900)):
            page.set_viewport_size({"width": width, "height": height})
            assert page.locator(".ge-meta").is_visible(), (width, height)
            assert "3 steps" in page.locator(".ge-meta").inner_text()
        # Gallery shows the no-link state only: one share section, no
        # copy control anywhere on the page, download enabled at once.
        assert page.locator(".ge-share").count() == 1
        assert page.locator("[data-ge-copy]").count() == 0
        assert page.locator("[data-ge-link]").count() == 0
        assert page.locator("[data-ge-consent]").count() == 0
        main_share = page.locator("main > section.ge-share")
        assert main_share.locator("[data-ge-download]").is_enabled()
        main_share.get_by_role("button", name="Download images").click()
        page.wait_for_selector(
            "main > section.ge-share "
            "[data-ge-status]:has-text('Downloading 1 image')"
        )
        errors, failed = page_errors, page_failed
    finally:
        context.close()
    assert errors == []
    assert failed == []


def test_browser_issued_url_copy_flow(browser):
    """With a server-issued URL in data-share-url, the consent-gated
    copy flow works: consent enables copy, clicking it copies the
    issued URL, the status confirms. Needs no server: the page is set
    directly with the real script inline."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        permissions=["clipboard-write"],
    )
    try:
        page = context.new_page()
        errors, failed = _collect(page)
        url = "https://share.example.invalid/g/demo-1"
        page.set_content(
            "<!DOCTYPE html><html><body>"
            f'<main id="main" data-share-url="{url}">'
            + share_html(url)
            + "</main>"
            f"<script>{JS}</script>"
            "</body></html>"
        )
        assert page.locator("[data-ge-link]").input_value() == url
        assert page.locator("[data-ge-copy]").is_disabled()
        page.locator("[data-ge-consent]").check()
        assert page.locator("[data-ge-copy]").is_enabled()
        page.get_by_role("button", name="Copy 24-h link").click()
        page.wait_for_selector("[data-ge-status]:has-text('Link copied.')")
        assert errors == []
        assert failed == []
    finally:
        context.close()


def test_browser_whitespace_share_url_fails_closed(browser):
    """A whitespace-only data-share-url fails closed in the live DOM
    exactly like the server: copy markup present in the HTML is
    stripped and the download stays enabled. Needs no server: the page
    is set directly with the real script inline."""
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        errors, failed = _collect(page)
        page.set_content(
            "<!DOCTYPE html><html><body>"
            '<main id="main" data-share-url="  \t ">'
            + share_html("https://share.example.invalid/g/demo-1")
            + "</main>"
            f"<script>{JS}</script>"
            "</body></html>"
        )
        assert page.locator("[data-ge-copy]").count() == 0
        assert page.locator("[data-ge-link]").count() == 0
        assert page.locator("[data-ge-consent]").count() == 0
        assert page.locator("[data-ge-download]").is_enabled()
        assert errors == []
        assert failed == []
    finally:
        context.close()


def test_browser_no_js_baseline(browser, live_server):
    context = browser.new_context(
        viewport={"width": 390, "height": 844}, java_script_enabled=False
    )
    try:
        page = context.new_page()
        for path in ("/reader/sample", "/reader/hero"):
            page.goto(f"{live_server}{path}")
            assert page.locator(".ge-meta").is_visible(), path
            assert page.locator(".ge-share").is_visible(), path
            assert page.locator(".ge-check input").count() == 2, path
        page.goto(f"{live_server}/static/ui/gallery/guide-end.html")
        assert page.locator(".ge-meta").is_visible()
        assert page.locator(".ge-share").count() == 1
        assert page.locator("main > section.ge-share").is_visible()
    finally:
        context.close()
