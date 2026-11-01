"""Content-level navigation: every internal link on home and the sample guide opens, and the proof pages say what the record says."""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import proof_links
from app.main import create_app
from tools import catch_media
from catch_bundle import make
from test_integration_flow import _clean_service, lane_db, replay_settings  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]
HREF = re.compile(r'href="(/[^"#]*)')


@pytest.fixture
def placeholder_catch_images(tmp_path):
    """Synthetic, manifest-verified catch crops: unpack a generated bundle against a matching manifest and require catch_media.check() before any navigation assertion. Files already installed are restored afterwards."""
    bundle, manifest = make(tmp_path / "private")
    folder = ROOT / catch_media.TARGET
    names = catch_media.filenames(catch_media.load())
    before = {n: (folder / n).read_bytes() for n in names if (folder / n).is_file()}
    catch_media.unpack(ROOT, bundle=bundle, manifest_path=manifest)
    catch_media.check(ROOT, manifest)
    yield
    for name in names:
        if name in before:
            (folder / name).write_bytes(before[name])
        else:
            (folder / name).unlink()


def _links(client, url):
    page = client.get(url)
    assert page.status_code == 200
    return page, sorted(set(HREF.findall(page.text)))


def test_proof_pages_open_and_say_what_the_record_says(replay_settings, placeholder_catch_images):
    client = TestClient(create_app(replay_settings))
    cfg = proof_links.load()
    home, home_links = _links(client, "/")
    assert f'href="/catch">{cfg["catch_cta"]}</a>' in home.text and "Try the catch yourself" not in home.text
    guide = client.get("/guide/sample")  # redirects to the built sample guide
    assert guide.status_code == 200 and guide.history
    guide_links = sorted(set(HREF.findall(guide.text)))
    assert cfg["sample_calls"] in guide_links and "/catch" in guide_links
    assert 'id="step-6"' in guide.text  # the hard-stop target exists in the installed sample
    # images.zip is opened by test_sample_guide_images_download_opens.
    crawl = {u for u in (*home_links, *guide_links) if "/images.zip" not in u}
    for url in sorted({*crawl, "/catch", "/calls/sample", cfg["sample_calls"]}):
        assert client.get(url).status_code == 200, url  # after redirects
    for url in ("/calls/sample", cfg["sample_calls"]):
        calls = client.get(url)
        assert '<span class="num">18</span><span class="cap">observed runs</span>' in calls.text
    assert client.get("/measured-example").url.path == cfg["sample_calls"]

    catch = client.get("/catch").text
    row = re.search(r'<tr id="row-f3_loose">.*?</tr>', catch, re.S).group(0)
    assert "altered" in row and "Rejected" in row
    assert "blind spot" in catch and "paste-back composite result was recorded" in catch
    data = json.loads((ROOT / "config/catch_table.json").read_text())
    assert f'{data["summary"]["heldout_recipes"]["rejected"]} of {data["summary"]["heldout_recipes"]["total"]} rejected' in catch
    assert "What passes" in catch and 'id="row-f1_loose"' in catch
    images = sorted(set(re.findall(r'src="(/static/ui/media/catch/[^"]+)"', catch)))
    assert len(images) == 4
    for src in images:
        response = client.get(src)
        assert response.status_code == 200 and response.headers["content-type"].startswith("image/"), src


def test_sample_guide_images_download_opens(replay_settings):
    client = TestClient(create_app(replay_settings), raise_server_exceptions=False)
    guide = client.get("/guide/sample")
    link = re.search(r'href="(/guides/[^"]+/images\.zip[^"]*)"', guide.text).group(1)
    response = client.get(link.replace("&amp;", "&"))
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert names and all(archive.read(n)[:8] == b"\x89PNG\r\n\x1a\n" for n in names)


@pytest.mark.parametrize("entry", ["/calls/sample", "/measured-example"])
def test_calls_aliases_install_the_sample_on_a_fresh_db(replay_settings, entry):
    client = TestClient(create_app(replay_settings))
    page = client.get(entry, follow_redirects=True)
    assert page.status_code == 200
    assert '<span class="num">18</span><span class="cap">observed runs</span>' in page.text


def test_existing_sample_install_missing_the_mask_is_backfilled(replay_settings):
    client = TestClient(create_app(replay_settings), raise_server_exceptions=False)
    client.get("/guide/sample")
    root = Path(replay_settings.data_root)
    masks = sorted(root.rglob("*.mask.png"))
    assert masks
    for mask in masks:
        mask.unlink()  # an install made before the mask shipped
    guide = client.get("/guide/sample")
    link = re.search(r'href="(/guides/[^"]+/images\.zip[^"]*)"', guide.text).group(1)
    assert client.get(link.replace("&amp;", "&")).status_code == 200
    assert all(m.is_file() for m in masks)


def test_real_catch_media_matches_the_manifest():
    """Media is provisioned at deploy (never committed), so this runs only where the bundle or installed crops exist."""
    installed = ROOT / catch_media.TARGET
    if not (catch_media.BUNDLE.is_file() or any(installed.glob("*.jpg"))):
        pytest.skip("no var/catch-media.zip or installed catch images; provisioned at deploy")
    if catch_media.BUNDLE.is_file():
        catch_media.unpack(ROOT)
    catch_media.check(ROOT)
