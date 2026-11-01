"""First-screen tests: settled copy, CTA hierarchy, replay label."""

import html as html_lib
import json
import re
import struct

import pytest
from fastapi.testclient import TestClient

from app import film_config, proof_links
from app.routes.site import (
    HEADLINE,
    ROOT,
    SAMPLE_CTA,
    MAKE_CTA,
    SUB,
    create_site_app,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_site_app())


@pytest.fixture(scope="module")
def landing(client):
    response = client.get("/")
    assert response.status_code == 200
    return response.text


def test_film_opening_copy_and_links(landing):
    assert HEADLINE == "Take it apart. Pass it on."
    assert SUB == "Turn a phone video of a repair into a visual guide, one clear step at a time."
    assert SAMPLE_CTA == "Explore the sample" and MAKE_CTA == "Make a guide"
    assert f'<h1 id="main-heading">{HEADLINE}</h1>' in landing
    assert f'<p class="sub">{SUB}</p>' in landing
    primary = '<a class="cta-primary" href="/guide/sample">Explore the sample</a>'
    own = '<a class="cta-ghost" href="/upload">Make a guide</a>'
    assert primary in landing and own in landing
    assert landing.index(primary) < landing.index(own)
    # The catch and the sample section stay on the page, below the film.
    assert f'<a class="cta-secondary" href="/catch">{proof_links.load()["catch_cta"]}</a>' in landing
    assert landing.index('id="film"') < landing.index('href="/catch"') < landing.index('class="sample"')
    assert 'id="sample-h"' in landing


def test_chapters_keep_d_sentences_with_part_labels(landing):
    for label in ("01 / The screen", "02 / The keyboard", "03 / Inside"):
        assert label in landing
    assert "SAM 3.1" in landing and "Muse Spark" in landing
    assert landing.count("data-chap") == 3


def test_upload_limit_copy_is_the_settings_limit(landing):
    assert "Clips can be up to 5 minutes long." in landing
    assert "Use my own video" not in landing and "Use your video" not in landing
    assert "60 s" not in landing


def test_one_wording_per_call_to_action(landing):
    # Every link to /upload says "Make a guide" and every link to the sample says "Explore the sample".
    import re

    for href, label in (("/upload", "Make a guide"), ("/guide/sample", "Explore the sample")):
        texts = re.findall(rf'<a class="cta[^"]*" href="{href}">([^<]*)</a>', landing)
        assert texts and set(texts) == {label}, (href, texts)
    assert "Open the sample guide" not in landing


def test_film_config_is_inlined_and_poster_preloaded(landing):
    cfg = film_config.public(film_config.load())
    assert f'rel="preload" as="image" href="{cfg["poster_url"]}"' in landing
    assert json.loads(re.search(r'id="film-config">(.*?)</script>', landing, re.S).group(1)) == cfg


def _require_film_assets():
    root = ROOT / "static/ui/media/teardown"
    cfg = film_config.load()
    if not any((root / item["dir"]).exists() for item in cfg["sets"]):
        pytest.skip("film frame sets absent locally; unpack the deployment assets to check them")
    return root, cfg


def test_film_bundles_and_poster_are_served_with_long_cache(client):
    root, cfg = _require_film_assets()
    plan = film_config.bundles(cfg["film"])
    paths = ["/static/ui/media/teardown/" + cfg["film"]["poster"]]
    paths += [f"/static/ui/media/teardown/{item['dir']}/{plan[0][0]}" for item in cfg["sets"]]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        assert f'max-age={cfg["film"]["cache_max_age_s"]}' in response.headers["cache-control"]
        body = response.content
        assert body[:4] == b"RIFF" or body[6:10] == b"RIFF", path
    assert "immutable" not in client.get("/static/site.css").headers.get("cache-control", "")


def _records(data: bytes) -> dict[int, bytes]:
    out, at = {}, 0
    while at < len(data):
        n, size = struct.unpack_from("<HI", data, at)
        assert n not in out
        out[n] = data[at + 6:at + 6 + size]
        at += 6 + size
    assert at == len(data)
    return out


def test_every_frame_is_in_exactly_the_planned_bundle():
    root, cfg = _require_film_assets()
    film = cfg["film"]
    for item in cfg["sets"]:
        name, seen = item["dir"], []
        for bundle, nums in film_config.bundles(film):
            frames = _records((root / name / bundle).read_bytes())
            assert list(frames) == nums
            assert all(f[:4] == b"RIFF" and f[8:12] == b"WEBP" for f in frames.values())
            seen += nums
        assert sorted(seen) == list(range(1, film["frames"] + 1))
        assert sorted(p.name for p in (root / name).iterdir()) == sorted(b for b, _ in film_config.bundles(film))
