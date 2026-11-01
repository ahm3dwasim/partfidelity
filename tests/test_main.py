"""Main-app wiring: every router reachable, /static served, guards on."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.body_limit import BodyLimitMiddleware
from app.main import create_app
from app.settings import Settings


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_root=tmp_path / "data", db_path=tmp_path / "t.sqlite3", production=False
    )
    return TestClient(create_app(settings))


def test_healthz_ok(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_site_router_reachable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/").status_code == 200
    assert client.get("/guide/sample", follow_redirects=False).status_code == 303
    reader = client.get("/reader/sample", follow_redirects=False)
    assert reader.status_code == 303 and reader.headers["location"] == "/guide/sample"
    assert client.get("/reader/hero").status_code == 200


def test_guides_router_reachable_with_loader_wired(tmp_path: Path) -> None:
    client = _client(tmp_path)
    # 404 (not 503) proves the pipeline guide loader is wired: no such job.
    assert client.get("/guides/does-not-exist").status_code == 404
    assert client.get("/guides/does-not-exist/image/0").status_code == 404


def test_calls_and_steps_routers_reachable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/calls/job-1").status_code == 200
    assert client.get("/steps/preview").status_code == 200


def test_uploads_router_reachable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/upload").status_code == 200
    assert client.get("/delete-request").status_code == 200
    assert client.get("/preview/does-not-exist").status_code in (404, 410)


def test_jobs_router_reachable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    missing = client.get("/jobs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "job not found"}
    assert client.get("/jobs/!!!").status_code == 404


def test_static_serves(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for asset in (
        "/static/site.css",
        "/static/guide.css",
        "/static/calls.css",
        "/static/time_controls.js",
        "/static/upload.js",
        "/static/reader.js",
        "/static/hard_stop.js",
    ):
        response = client.get(asset)
        assert response.status_code == 200, asset
        assert response.content
    assert client.get("/static/no-such-file.css").status_code == 404


def test_body_limit_middleware_registered(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=tmp_path / "t.sqlite3", production=False
    )
    app = create_app(settings)
    assert BodyLimitMiddleware in [entry.cls for entry in app.user_middleware]
