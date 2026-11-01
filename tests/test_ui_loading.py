"""the shared loading set is complete, themed, and wired.

Covers static/ui/loading.css + loading.js, templates/partials/ui/loading.html,
the gallery page, and the upload.html / calls.html integrations.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader

from app import evidence
from app.contracts import JobId
from app.routes.calls import router as calls_router
from app.routes.site import create_site_app

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "static" / "ui" / "loading.css"
JS = ROOT / "static" / "ui" / "loading.js"
UPLOAD_JS = ROOT / "static" / "upload.js"
GSAP = ROOT / "static" / "ui" / "vendor" / "gsap.min.js"
PARTIAL = ROOT / "templates" / "partials" / "ui" / "loading.html"
GALLERY = ROOT / "static" / "ui" / "gallery" / "loading.html"

PLAIN_LABELS = {
    "upload": "Uploading your video",
    "extract": "Finding steps",
    "locate": "Finding parts",
    "segment": "Outlining parts",
    "readback": "Checking labels",
}


def _js_stages() -> list[str]:
    match = re.search(r"var STAGES = \[(.*?)\];", JS.read_text(encoding="utf-8"))
    assert match, "loading.js must declare var STAGES"
    return re.findall(r'"([a-z]+)"', match.group(1))


def test_js_stages_match_evidence_vocabulary() -> None:
    """The progress bar runs on the real model-call stages, in order."""
    assert _js_stages() == ["upload", "extract", "locate", "segment", "readback"]
    assert set(_js_stages()) == set(evidence.STAGE_VOCABULARY)
    body = JS.read_text(encoding="utf-8")
    for stage, label in PLAIN_LABELS.items():
        assert f"{stage}: " in body or f"{stage}:" in body
        assert label in body


def test_js_rejects_unknown_stage() -> None:
    body = JS.read_text(encoding="utf-8")
    assert "index === -1" in body
    assert "return false" in body


def test_js_uses_gsap_with_fallback() -> None:
    body = JS.read_text(encoding="utf-8")
    assert "window.gsap" in body
    assert "overwrite" in body
    assert "matchMedia" in body
    assert "prefers-reduced-motion" in body
    # Fallback path sets the final state directly when GSAP is absent.
    assert 'fill.style.transform = "scaleX(" + frac + ")"' in body


def test_css_covers_set_and_both_themes() -> None:
    body = CSS.read_text(encoding="utf-8")
    for selector in (
        ".pf-skel-step",
        ".pf-skel-img",
        ".pf-skel-line",
        ".pf-progress",
        ".pf-progress-fill",
        ".pf-stages",
        ".pf-btn",
        '[data-pf-fade]',
        ".pf-state",
        ".pf-state-title",
        ".pf-state-action",
    ):
        assert selector in body, selector
    assert ".pf-theme-ink" in body
    assert "--pf-load-skel" in body
    assert "prefers-reduced-motion" in body
    assert ":focus-visible" in body
    # shared token names with fallbacks until tokens.css lands.
    assert "var(--pf-amber, #f0a400)" in body
    assert "var(--pf-danger, #c2331f)" in body


def _luminance(hexcode: str) -> float:
    channels = [int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5)]

    def linear(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: str, second: str) -> float:
    high, low = max(_luminance(first), _luminance(second)), min(
        _luminance(first), _luminance(second)
    )
    return (high + 0.05) / (low + 0.05)


def test_focus_token_hits_3_to_1_on_both_papers() -> None:
    """--pf-focus is shared on :root and clears WCAG non-text contrast."""
    body = CSS.read_text(encoding="utf-8")
    paper = re.search(
        r":root\s*\{[^}]*--pf-focus:\s*var\(--pf-ink,\s*(#[0-9a-fA-F]{6})\)",
        body,
    )
    assert paper, ":root must define the shared --pf-focus token"
    ink = re.search(
        r"\.pf-theme-ink\s*\{[^}]*--pf-focus:\s*var\(--pf-amber,\s*(#[0-9a-fA-F]{6})\)",
        body,
    )
    assert ink, "ink pages keep the amber focus ring"
    assert "var(--pf-focus" in body
    for background in ("#faf9f5", "#f0eee8"):
        ratio = _contrast(paper.group(1), background)
        assert ratio >= 3.0, (paper.group(1), background, ratio)
    dark = _contrast(ink.group(1), "#1d1d1d")
    assert dark >= 3.0, (ink.group(1), dark)


def _macros():
    env = Environment(loader=FileSystemLoader(ROOT / "templates"))
    return env.get_template("partials/ui/loading.html").module


def test_partial_skeleton_macros() -> None:
    macros = _macros()
    step = macros.skeleton_step()
    assert 'data-pf-skeleton="step"' in step
    assert "pf-skel-img" in step
    assert "pf-skel-line" in step
    image = macros.skeleton_image()
    assert 'data-pf-skeleton="image"' in image


def test_partial_progress_renders_every_real_stage() -> None:
    macros = _macros()
    stages = _js_stages()
    for position, stage in enumerate(stages, start=1):
        html = macros.progress(stage=stage)
        assert 'role="progressbar"' in html
        assert f'data-pf-stage="{stage}"' in html
        assert f"scaleX({position / len(stages)})" in html
        assert f"step {position} of {len(stages)}" in html
        assert PLAIN_LABELS[stage] in html
        for other in stages:
            assert f'data-stage="{other}"' in html


def test_partial_progress_falls_back_to_first_stage() -> None:
    macros = _macros()
    html = macros.progress(stage="nope")
    assert f'data-pf-stage="{_js_stages()[0]}"' in html


def test_partial_button_fade_and_states() -> None:
    macros = _macros()
    button = macros.busy_button("Continue to preview")
    assert 'class="pf-btn"' in button
    assert 'aria-busy="false"' in button
    assert "Continue to preview" in button
    assert 'data-copy="loading.button.label"' in button
    fade = macros.fade_image("/static/x.webp", "A part")
    assert "data-pf-fade" in fade
    assert 'alt="A part"' in fade
    expected = {
        "upload-failed": ("Upload failed", "did not reach us", "/upload", "Try again"),
        "job-cancelled": ("This job was cancelled", "Nothing was kept", "/", "Start again"),
        "link-expired": ("This link has expired", "24 hours", "/", "Go home"),
    }
    for kind, bits in expected.items():
        html = macros.state(kind)
        assert f'data-pf-state="{kind}"' in html
        assert 'role="alert"' in html
        for bit in bits:
            assert bit in html, (kind, bit)
        assert "data-copy=" in html


def test_gallery_shows_every_state_in_both_themes() -> None:
    body = GALLERY.read_text(encoding="utf-8")
    assert "../loading.css" in body
    assert "../loading.js" in body
    assert "../vendor/gsap.min.js" in body
    paper, ink = body.split('<section class="gal-theme gal-ink', 1)
    assert "gal-paper" in paper
    assert "pf-theme-ink" in ink
    markers = (
        'data-pf-skeleton="step"',
        'data-pf-skeleton="image"',
        "data-pf-progress",
        'aria-busy="true"',
        "<img data-pf-fade",
        'data-pf-state="upload-failed"',
        'data-pf-state="job-cancelled"',
        'data-pf-state="link-expired"',
    )
    for marker in markers:
        assert marker in paper, ("paper", marker)
        assert marker in ink, ("ink", marker)
    for stage in _js_stages():
        assert f'data-stage="{stage}"' in paper
        assert f'data-stage="{stage}"' in ink


def test_gsap_vendored_locally() -> None:
    import hashlib

    assert GSAP.exists()
    head = GSAP.read_text(encoding="utf-8")[:400]
    assert "GSAP 3.13.0" in head
    # Pinned to the official 3.13.0 bytes (verified identical to the CDN
    # release); any supply-chain swap turns this red.
    assert (
        hashlib.sha256(GSAP.read_bytes()).hexdigest()
        == "96c01b81f44a3290e2b4532f55e2c9534b2adc43273a19f3756b2cb41f0fd0b6"
    )


def test_macro_params_escaped_without_autoescape() -> None:
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"), autoescape=False
    )
    macros = env.get_template("partials/ui/loading.html").module
    button = macros.busy_button("<img src=x onerror=1>")
    assert "<img src=x" not in button
    assert "&lt;img" in button
    fade = macros.fade_image("/x.webp", '" onload="1')
    assert '" onload="' not in fade
    custom = macros.state("upload-failed", body="<img src=x onerror=1>")
    assert "<img src=x" not in custom
    assert "&lt;img" in custom


def test_partial_state_accepts_specific_message() -> None:
    macros = _macros()
    html = macros.state("upload-failed", body="Trim it to 60 seconds or less.")
    assert 'data-pf-state="upload-failed"' in html
    assert 'role="alert"' in html
    assert "Upload failed" in html
    assert "Trim it to 60 seconds or less." in html
    assert "did not reach us" not in html
    assert "Try again" in html
    assert "ignored" not in macros.state("job-cancelled", body="ignored")
    assert "data-copy=" in macros.state("upload-failed")


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_loading_js_passes_node_check() -> None:
    proc = subprocess.run(
        ["node", "--check", str(JS)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_loading_js_fallback_without_gsap() -> None:
    """The real loading.js runs without gsap and sets final state."""
    proc = subprocess.run(
        ["node", str(ROOT / "tests" / "fixtures" / "ui_loading_no_gsap.js"), str(JS)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NO-GSAP-OK" in proc.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_busy_button_releases_on_client_rejection() -> None:
    """Busy never sticks on too-large/too-long clips; locks only if accepted."""
    proc = subprocess.run(
        [
            "node",
            str(ROOT / "tests" / "fixtures" / "ui_loading_busy_order.js"),
            str(JS),
            str(UPLOAD_JS),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BUSY-ORDER-OK" in proc.stdout


class _OpenUploads:
    def public_uploads_blocked(self) -> bool:
        return False


def test_upload_page_wires_loading_set(tmp_path: Path) -> None:
    from app.routes import uploads as uploads_mod

    app = FastAPI()
    app.include_router(uploads_mod.router)
    uploads_mod.set_test_overrides(uploads=_OpenUploads())
    try:
        body = TestClient(app).get("/upload").text
    finally:
        uploads_mod.clear_test_overrides()
    assert "/static/ui/loading.css" in body
    assert "/static/ui/loading.js" in body
    assert "/static/ui/vendor/gsap.min.js" in body
    assert 'src="/static/ui/loading.js" defer' in body
    assert 'src="/static/ui/vendor/gsap.min.js" defer' in body
    assert "data-pf-busy" in body
    assert 'class="pf-btn"' in body
    assert "Continue to preview" in body
    # The probe-timeout ceiling rides with the other client limits.
    from app.uploads import PROBE_TIMEOUT_MS

    assert f'"probeTimeoutMs": {PROBE_TIMEOUT_MS}' in body
    # Client-side rejections fill the shared pf-state, not a bare div.
    assert 'id="client-error"' in body
    assert 'data-pf-state="upload-failed"' in body
    assert "pf-state-body" in body


def test_rejected_upload_renders_shared_error_state(tmp_path: Path) -> None:
    """A refused POST keeps its message but wears the pf-state component."""
    from app.routes import uploads as uploads_mod
    from app.settings import Settings
    from app.uploads import MAX_BYTES, OVER_LIMIT_MESSAGE

    app = FastAPI()
    app.include_router(uploads_mod.router)
    app.state.settings = Settings(
        data_root=tmp_path / "data",
        db_path=tmp_path / "t.sqlite3",
        production=False,
    )
    uploads_mod.set_test_overrides(uploads=_OpenUploads())
    try:
        response = TestClient(app).post(
            "/upload",
            files={
                "clip": ("big.mp4", b"\x00" * (MAX_BYTES + 1), "video/mp4")
            },
        )
    finally:
        uploads_mod.clear_test_overrides()
    assert response.status_code == 400
    assert OVER_LIMIT_MESSAGE in response.text
    assert 'id="server-error"' in response.text
    assert 'class="pf-state"' in response.text
    assert 'data-pf-state="upload-failed"' in response.text
    assert 'role="alert"' in response.text


def test_calls_page_wires_loading_set() -> None:
    evidence.reset_stores()
    try:
        app = FastAPI()
        app.include_router(calls_router)
        response = TestClient(app).get(f"/calls/{JobId('job-load-1')}")
    finally:
        evidence.reset_stores()
    assert response.status_code == 200
    assert "/static/ui/loading.css" in response.text
    assert "/static/ui/loading.js" in response.text
    assert "/static/ui/vendor/gsap.min.js" in response.text
    assert 'src="/static/ui/loading.js" defer' in response.text
    assert "data-pf-fade" in response.text


def test_static_files_served() -> None:
    client = TestClient(create_site_app())
    for url in (
        "/static/ui/loading.css",
        "/static/ui/loading.js",
        "/static/ui/vendor/gsap.min.js",
        "/static/ui/gallery/loading.html",
    ):
        assert client.get(url).status_code == 200, url
