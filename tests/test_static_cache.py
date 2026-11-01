"""Static and HTML responses must revalidate; film frames keep their immutable lifetime."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import film_config
from app.main import create_app
from app.routes.site import create_site_app
from app.settings import Settings

SRC = film_config.CONFIG.read_text(encoding="utf-8")
POSTER = "/static/ui/media/teardown/" + film_config.cached()["film"]["poster"]
NO_CACHE = film_config.cached()["film"].get("static_cache_control")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_site_app())


@pytest.mark.parametrize("path", ["/static/site.css", "/static/ui/shell.css", "/static/ui/film.js"])
def test_static_asset_revalidates(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == NO_CACHE == "no-cache"


def test_film_frame_keeps_immutable_lifetime(client):
    age = film_config.cached()["film"]["cache_max_age_s"]
    cc = client.get(POSTER).headers["cache-control"]
    assert cc == f"public, max-age={age}, immutable"


def test_conditional_get_returns_304(client):
    first = client.get("/static/ui/shell.css")
    etag = first.headers["etag"]
    second = client.get("/static/ui/shell.css", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.headers["cache-control"] == "no-cache"


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prod")
    settings = Settings(data_root=tmp / "data", db_path=tmp / "t.sqlite3", production=False)
    return TestClient(create_app(settings))


@pytest.mark.parametrize("path", ["/", "/how-it-works", "/built", "/upload", "/static/ui/gallery/upload.html"])
def test_html_pages_revalidate_and_answer_304(prod, path):
    first = prod.get(path)
    assert first.status_code == 200, path
    assert first.headers["cache-control"] == NO_CACHE, path
    etag = first.headers["etag"]
    assert prod.get(path).headers["etag"] == etag, path
    again = prod.get(path, headers={"If-None-Match": etag})
    assert again.status_code == 304, path
    assert again.headers["cache-control"] == NO_CACHE
    assert again.headers["etag"] == etag


def test_upload_js_through_production_app(prod):
    first = prod.get("/static/upload.js")
    assert first.status_code == 200
    assert first.headers["cache-control"] == NO_CACHE
    second = prod.get("/static/upload.js", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 304


def test_missing_config_key_fails_loudly(tmp_path: Path):
    assert "static_cache_control" in SRC
    path = tmp_path / "film.toml"
    path.write_text(SRC.replace("static_cache_control", "static_cache_ctl"), encoding="utf-8")
    with pytest.raises(film_config.FilmConfigError):
        film_config.load(path)


# --- RFC 9110 weak comparison, content-hash static ETags, uploads app ---

from fastapi import FastAPI

from app.routes.site import CachedStaticFiles
from app.routes.uploads import create_uploads_app


@pytest.mark.parametrize(
    "header,expected",
    [
        ('"a"', True),
        ('W/"a"', True),
        ('"x", "a"', True),
        ('"x" , W/"a" ,"y"', True),
        ("*", True),
        ('"b"', False),
        ("", False),
    ],
)
def test_etag_matches_weak_comparison(header, expected):
    from app.routes.site import etag_matches

    assert etag_matches(header, '"a"') is expected


@pytest.mark.parametrize("path", ["/", "/static/ui/shell.css"])
@pytest.mark.parametrize("variant", ["weak", "list", "star"])
def test_conditional_variants_answer_304(prod, path, variant):
    etag = prod.get(path).headers["etag"]
    value = {"weak": "W/" + etag, "list": f'"nope", {etag}', "star": "*"}[variant]
    response = prod.get(path, headers={"If-None-Match": value})
    assert response.status_code == 304, (path, value)
    assert response.headers["etag"] == etag


def test_same_size_different_content_gets_different_etag(tmp_path):
    f = tmp_path / "a.css"
    app = FastAPI()
    app.mount("/s", CachedStaticFiles(directory=tmp_path))
    c = TestClient(app)
    f.write_text("body{color:red}")
    import os

    os.utime(f, (0, 0))
    first = c.get("/s/a.css").headers["etag"]
    f.write_text("body{color:tan}")
    os.utime(f, (0, 0))
    assert len("body{color:red}") == len("body{color:tan}")
    second = c.get("/s/a.css")
    assert second.headers["etag"] != first
    assert c.get("/s/a.css", headers={"If-None-Match": first}).status_code == 200
    assert second.text == "body{color:tan}"


@pytest.mark.parametrize("path", ["/upload", "/static/ui/gallery/upload.html"])
def test_uploads_app_html_headers(path):
    c = TestClient(create_uploads_app())
    first = c.get(path)
    assert first.status_code == 200
    assert first.headers["cache-control"] == NO_CACHE
    etag = first.headers["etag"]
    assert c.get(path, headers={"If-None-Match": etag}).status_code == 304
