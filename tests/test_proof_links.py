import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import proof_links
from app.routes import calls, guides


def test_sample_links_and_redirects_match_recorded_guide():
    config = proof_links.load()
    guide = json.loads((Path(__file__).resolve().parents[1] / 'app/sample_real/guide.json').read_text())
    assert config['sample_calls'] == '/calls/' + guide['job_id']
    assert config['hard_stop'] == '/guide/sample#step-5'
    note = guides._sample_note_html()
    assert f'href="{config["sample_calls"]}"' in note
    assert f'href="{config["hard_stop"]}"' in note
    app = FastAPI()
    app.include_router(calls.router)
    client = TestClient(app)
    for url in ('/calls/sample', '/measured-example'):
        response = client.get(url, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers['location'] == config['sample_calls']


def test_home_meta_roles_line_links_to_the_sample_receipts_and_names_every_role():
    from app.routes.site import create_site_app
    config = proof_links.load()
    home = TestClient(create_site_app()).get('/').text
    line = home[home.index('film-roles'):]
    line = line[:line.index('</p>')]
    assert f'href="{config["sample_calls"]}">{config["meta_roles_link"]}</a>' in line
    for role in ('Muse Spark', 'SAM 3.1', 'SAM 3', 'Muse Spark Open', 'Muse Image'):
        assert role in line, role
    assert 'href="/catch">' in line and 'href="/built">' in line
    assert len(line) < 520
    assert 'verified' not in line.lower()


def test_meta_roles_config_fails_loud_when_missing(tmp_path, monkeypatch):
    import pytest
    text = (Path(__file__).resolve().parents[1] / 'config/proof_links.toml').read_text()
    bad = text.replace('meta_roles_link = "Sample receipts"', 'meta_roles_link = " "')
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/proof_links.toml').write_text(bad)
    monkeypatch.setattr(proof_links, 'Path', lambda _file: tmp_path / 'app' / 'x.py')
    with pytest.raises(ValueError, match='invalid sample proof link'):
        proof_links.load()


def _load_with(tmp_path, monkeypatch, old, new):
    text = (Path(__file__).resolve().parents[1] / 'config/proof_links.toml').read_text()
    assert old in text
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/proof_links.toml').write_text(text.replace(old, new))
    monkeypatch.setattr(proof_links, 'Path', lambda _file: tmp_path / 'app' / 'x.py')
    return proof_links.load()


def test_meta_roles_must_name_every_model_and_never_say_verified(tmp_path, monkeypatch):
    import pytest
    for old, new in (('Muse Image tests', 'Image tests'), ('This sample used', 'Verified on')):
        with pytest.raises(ValueError, match='invalid sample proof link'):
            _load_with(tmp_path, monkeypatch, old, new)
        (tmp_path / 'config/proof_links.toml').unlink()
        (tmp_path / 'config').rmdir()


def test_meta_roles_rejects_substring_stand_ins(tmp_path, monkeypatch):
    import pytest
    for old, new in (("saved SAM 3 contours", "saved contours"), ("Muse Spark steps and", "steps and")):
        with pytest.raises(ValueError, match='meta_roles'):
            _load_with(tmp_path, monkeypatch, old, new)
        (tmp_path / 'config/proof_links.toml').unlink()
        (tmp_path / 'config').rmdir()


def test_home_glosses_muse_spark_and_sam_on_first_use():
    from app.routes.site import create_site_app
    home = TestClient(create_site_app()).get('/').text.replace('&#x27;', "'")
    assert "Muse Spark steps and label checks (Meta's model that reads the video)" in home
    assert "SAM 3.1 part boxes (Meta's model that outlines the part)" in home
