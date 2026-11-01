"""restyled upload page keeps every behaviour hook, assets on /static."""

import hashlib
import posixpath
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.routes import uploads as routes_mod

ROOT = Path(__file__).resolve().parents[1]

# Every asset the upload page pulls, with the content type the /static
# mount must serve it as.
ASSETS = (
    ("/static/ui/upload.css", "text/css"),
    ("/static/upload.js", "javascript"),
    ("/static/ui/media/upload/Oswald-700-normal.woff2", "font/woff2"),
    ("/static/ui/media/upload/Inter-400-normal.woff2", "font/woff2"),
    ("/static/ui/media/upload/Inter-600-normal.woff2", "font/woff2"),
)

# D upload copy.
HEADLINE = "Start with your video."


def _client() -> TestClient:
    # The normal app: /static mounted first, router after (integration order).
    return TestClient(routes_mod.create_uploads_app())


def _gallery() -> str:
    return (ROOT / "static" / "ui" / "gallery" / "upload.html").read_text()


def test_upload_page_keeps_behaviour_hooks() -> None:
    body = routes_mod._upload_page()
    for hook in (
        'id="upload-form"',
        'id="clip"',
        'id="client-error"',
        "PARTFIDELITY_LIMITS",
        "/static/upload.js",
        'action="/upload"',
        'name="clip"',
    ):
        assert hook in body
    assert 'id="server-error"' in routes_mod._upload_page(error="boom")
    assert 'id="blocked"' in routes_mod._upload_page(blocked=True)


def test_upload_page_has_workshop_elements() -> None:
    body = routes_mod._upload_page()
    assert "/static/ui/upload.css" in body
    assert 'id="drop-zone"' in body
    assert "300 seconds or less" in body
    # The length reads from the configured cap, in minutes (D: "up to 5 minutes").
    assert "Up to 5 minutes" in body
    assert "own footage" in body
    assert "What happens next" in body
    assert body.count("data-copy=") >= 10
    # Exactly the three "what happens next" stages.
    for n in ("upload.next_1", "upload.next_2", "upload.next_3"):
        assert body.count(f'data-copy="{n}"') == 1


def test_upload_headline_is_owner_copy() -> None:
    for body in (routes_mod._upload_page(), _gallery()):
        assert HEADLINE in body
        assert "Turn a teardown video into a guide" not in body


def test_sample_cta_is_first_and_primary() -> None:
    for body in (routes_mod._upload_page(), _gallery()):
        sample = body.index("/guide/sample")
        submit = body.index('type="submit"')
        assert sample < submit
        assert (
            '<a class="pf-btn" href="/guide/sample">'
            "See a sample guide</a>"
        ) in body
        assert 'class="pf-btn pf-btn-ghost" type="submit"' in body


def test_client_error_is_announced() -> None:
    for body in (routes_mod._upload_page(), _gallery()):
        assert (
            '<div class="pf-alert client" id="client-error" '
            'role="alert" tabindex="-1" hidden></div>'
        ) in body
    js = (ROOT / "static" / "upload.js").read_text()
    assert "box.focus()" in js


def test_upload_assets_served_from_static_mount() -> None:
    client = _client()
    for url, content_type in ASSETS:
        response = client.get(url)
        assert response.status_code == 200, url
        assert content_type in response.headers["content-type"], url
    css = client.get("/static/ui/upload.css").text
    assert ".pf-drop" in css
    # D type comes from tokens.css (David Libre + Roboto), not upload.css.
    assert "var(--pf-font-display)" in css
    js = client.get("/static/upload.js").text
    assert "PARTFIDELITY_LIMITS" in js
    for name in (
        "Oswald-700-normal.woff2",
        "Inter-400-normal.woff2",
        "Inter-600-normal.woff2",
    ):
        response = client.get(f"/static/ui/media/upload/{name}")
        assert len(response.content) > 10_000, name


def test_gallery_relative_assets_resolve_under_static_mount() -> None:
    """Every relative ref in the gallery demo must 200 under the mount."""
    base = "/static/ui/gallery/"
    refs = re.findall(r'(?:src|href)="(\.\./[^"]+)"', _gallery())
    assert refs, "gallery has no relative asset refs"
    client = _client()
    assert client.get(base + "upload.html").status_code == 200
    for ref in sorted(set(refs)):
        url = posixpath.normpath(posixpath.join(base, ref))
        assert client.get(url).status_code == 200, ref


def test_upload_fonts_are_distinct_weights() -> None:
    """Each advertised weight ships its own bytes."""
    digests = set()
    for name in (
        "Oswald-700-normal.woff2",
        "Inter-400-normal.woff2",
        "Inter-600-normal.woff2",
    ):
        raw = (ROOT / "static" / "ui" / "media" / "upload" / name).read_bytes()
        digests.add(hashlib.sha256(raw).hexdigest())
    assert len(digests) == 3


def test_upload_ui_unknown_name_404s() -> None:
    client = _client()
    assert client.get("/static/ui/nope.css").status_code == 404
    assert client.get("/static/ui/media/upload/nope.woff2").status_code == 404
    # The old allowlist URLs are gone with the route.
    assert client.get("/static/ui/upload/upload.css").status_code == 404


def test_upload_js_keeps_validation_wiring() -> None:
    js = (ROOT / "static" / "upload.js").read_text()
    for hook in (
        "PARTFIDELITY_LIMITS",
        "probeDuration",
        "durationChecked",
        "drop-zone",
        "picked-file",
    ):
        assert hook in js


def test_upload_copy_follows_the_review() -> None:
    """Approved copy: exact replacements, no stale phrases."""
    for body in (routes_mod._upload_page(), _gallery()):
        assert "Audio is removed" in body
        assert "elapsed time for each stage" in body
        assert "numbered steps illustrated with still frames from your clip" in body.lower()
        for stale in ("sound stripped", "real clock", "labelled photos", "Watch it build"):
            assert stale not in body
        lowered = body.lower()
        for claim in ("for free", "free of charge", "no cost", "instant"):
            assert claim not in lowered


def test_retention_is_stated_the_same_on_both_pages() -> None:
    from app.contracts import META_UPLOAD_MAX_HOURS, UPLOAD_PREVIEW_TTL_HOURS

    upload = routes_mod._upload_page()
    preview = routes_mod._preview_page(
        job_id="j", duration="1.0", size="1", video_url="/v", state="consent", show_video=False
    )
    for body in (upload, preview):
        assert f"about {UPLOAD_PREVIEW_TTL_HOURS} hours after you upload it" in body
        assert f"within {META_UPLOAD_MAX_HOURS} hours at the latest" in body
        assert "stays with it until you delete the guide" in body
        assert "no measured typical time" in body
    # The decision step names who receives it and keeps the rights checkbox.
    assert "Meta" in preview and "own footage" in preview
    # The old two-way promise (clip and Meta both "24 hours or less") is gone.
    assert "24 hours or less" not in upload.replace("Deleted at Meta", "")


def test_hourly_cleanup_matches_the_word_hourly() -> None:
    from app.deploy_config import load_deploy_config

    assert load_deploy_config(ROOT / "deploy" / "modal_deploy.toml").host.sweep_interval_s == 3600


def test_upload_limits_come_from_the_enforcing_code() -> None:
    from app.uploads import MAX_BYTES

    body = routes_mod._upload_page(max_duration_s=90)
    assert "Up to <strong>90 seconds</strong>" in body
    assert f"<strong>{MAX_BYTES // (1024 * 1024)} MB</strong> max" in body


def test_upload_config_fails_loudly(tmp_path) -> None:
    import pytest
    import tomllib

    from app import upload_config

    good = tomllib.loads(upload_config.CONFIG.read_text())
    upload_config.validate(good)
    missing = {**good, "time": {"heading": "x"}}
    with pytest.raises(upload_config.UploadConfigError, match="keys differ"):
        upload_config.validate(missing)
    bad = {**good, "time": {"heading": "x", "text": "takes {minutes} minutes"}}
    with pytest.raises(upload_config.UploadConfigError, match="unknown placeholder"):
        upload_config.validate(bad)
    with pytest.raises(upload_config.UploadConfigError, match="needs values"):
        upload_config.render({"duration": "5 minutes"})
