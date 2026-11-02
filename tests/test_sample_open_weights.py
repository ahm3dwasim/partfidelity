"""The sample's open-weights pointers, the locked newer job, folded step titles and first-minute copy."""
from __future__ import annotations

import re

import pytest

from app import proof_links
from app.corrections import CorrectionStore
from app.guide_text import TEXT
from test_export import JOB, MemoryStore, _client, _post, guide  # noqa: F401 (fixtures)
from test_real_sample import client  # noqa: F401 (fixture)

LINKS = proof_links.load()
NEWER = LINKS["open_weights_job"]


def _guide_page(client):
    return client.get("/guide/sample", follow_redirects=True).text


def test_sample_guide_points_to_the_newer_jobs_receipts_and_the_replay(client):
    page = _guide_page(client)
    note = page[page.index('class="gd-open-weights"'):]
    note = note[:note.index("</p>")]
    assert LINKS["open_weights_note"] in note
    assert f'href="/calls/{NEWER}"' in note and 'href="/disagreement-replay"' in note
    assert "verified" not in note.lower() and "no disagreement recorded" not in note.lower()


def test_sample_receipts_say_the_sample_has_no_open_weights_rows(client):
    page = client.get(LINKS["sample_calls"]).text
    assert LINKS["open_weights_note"] in page and f'href="/calls/{NEWER}"' in page


def test_other_receipts_carry_no_sample_note(client):
    assert "gd-open-weights" not in client.get(f"/calls/{NEWER}").text


def test_fresh_data_root_renders_the_newer_jobs_four_second_checks(client):
    page = client.get(LINKS["open_weights_calls"]).text
    block = page[page.index('class="second-check"'):]
    block = block[:block.index("</section>")]
    assert block.count("Second check (Meta open weights, Spark Open)") == 4 and block.count("agrees") == 4
    assert page.count("<td>spark-open</td>") == 4 and 'class="num">24<' in page
    assert f'href="{LINKS["open_weights_replay"]}"' in block


def test_link_destinations_come_from_config():
    import pathlib
    for name in ("app/proof_links.py", "app/routes/calls.py", "templates/built.html"):
        assert "disagreement-replay" not in pathlib.Path(name).read_text(encoding="utf-8")


def test_newer_job_is_locked_against_label_edits(tmp_path):
    store = CorrectionStore(lambda j: tmp_path / j, "c.json", shared_job=lambda: "sample", locked_jobs=(NEWER,), max_ops=3)
    assert not store.editable(NEWER) and not store.editable("sample") and store.editable("other")


def test_a_locked_job_shows_no_fix_controls_and_refuses_label_posts(guide):
    store = MemoryStore(locked_jobs=(JOB,))
    client = _client(guide, store)
    assert "fix-label" not in client.get(f"/guides/{JOB}").text
    assert _post(client, action="delete", step=0, part=0, expect="fan").headers["location"].startswith(f"/guides/{JOB}?fix=locked")
    assert store.ops == {}


def test_app_passes_the_newer_job_to_the_correction_store(tmp_path):
    from fastapi.testclient import TestClient
    from app import db
    from app.main import create_app
    from app.settings import Settings
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    app = create_app(Settings(data_root=tmp_path / "data", db_path=database, production=False, pipeline_mode="replay"))
    with TestClient(app):
        assert not app.state.corrections_store.editable(NEWER)


def test_sample_says_this_video_not_your_video(client):
    page = _guide_page(client)
    notes = page[page.index('class="draft-notes"'):]
    notes = notes[:notes.index("</div>")]
    assert "battery disconnect not shown in this video" in notes and "your video" not in notes
    home = client.get("/").text
    teaser = home[home.index('class="sample-draft"'):]
    assert "not shown in this video." in teaser[:120]


def test_teaser_marks_the_skipped_steps(client):
    teaser = client.get("/").text
    teaser = teaser[teaser.index('class="sample-steps"'):]
    teaser = teaser[:teaser.index("</ol>")]
    numbers = re.findall(r'<li (?:value="(\d+)"|class="sample-gap")', teaser)
    assert numbers[0] == "1" and numbers[-1] == "8"
    assert re.search(r'value="2"[^<]*</li>\s*<li class="sample-gap"[^>]*>&#8230;</li>\s*<li value="8"', teaser)


def test_not_checked_is_explained_once_at_its_first_use(client):
    page = _guide_page(client)
    help_text = TEXT["status_not_checked_help"]
    assert page.count(help_text) == 1
    first = page.index('class="status-unchecked"')
    assert first < page.index(help_text) < page.index('class="status-unchecked"', first + 1)
    assert "verified" not in help_text.lower()


def test_folded_steps_keep_their_action_title():
    css = open("static/ui/guide-page.css", encoding="utf-8").read()
    rule = re.search(r"\.pf-js \.stop:not\(\.is-acked\) \+ \.after-stop \.card > ([^{]*?),\n", css).group(1)
    assert ":not(.action)" in rule
    from app import exports
    assert ":not(.action)" in exports._EXPORT_CSS


def test_replay_page_names_the_model_readably(client):
    page = client.get("/disagreement-replay").text
    assert "Muse Spark Open (Meta open weights)" in page and "Simulated replay: no model was called" in page


def test_proof_job_pages_say_saved_copy_and_show_no_install_time(client):
    line = proof_links.saved_copy_line(NEWER, client.app.state.settings.db_path)
    assert line.startswith("Saved copy of a recorded run")
    calls = client.get(f"/calls/{NEWER}").text
    status = client.get(f"/jobs/{NEWER}", headers={"accept": "text/html"}).text
    assert line in calls and line in status
    assert 'class="pf-wait-clock"' not in status and "Run log" not in status and "Ran for" not in status
    assert "Saved copy" not in client.get(LINKS["sample_calls"]).text


def test_saved_copy_line_uses_the_configured_date(client, monkeypatch):
    database = client.app.state.settings.db_path
    monkeypatch.setitem(LINKS, "open_weights_run_date", "2026-11-01")
    monkeypatch.setattr(proof_links, "load", lambda: LINKS)
    assert proof_links.saved_copy_line(NEWER, database) == "Saved copy of a recorded run from 2026-11-01"
    assert proof_links.saved_copy_line("other", database) == ""


def test_run_date_config_is_validated(tmp_path, monkeypatch):
    import pathlib
    text = (pathlib.Path(__file__).resolve().parents[1] / "config/proof_links.toml").read_text(encoding="utf-8")
    assert 'open_weights_run_date = ""' in text
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "proof_links.toml").write_text(text.replace('open_weights_run_date = ""', 'open_weights_run_date = "yesterday"'), encoding="utf-8")
    monkeypatch.setattr(proof_links, "__file__", str(tmp_path / "app" / "proof_links.py"))
    with pytest.raises(ValueError):
        proof_links.load()


@pytest.mark.parametrize("content", [None, "not json", "[]", '[{"stage": "x"}]'])
def test_startup_fails_closed_without_a_good_receipts_fixture(tmp_path, monkeypatch, content):
    import pathlib
    from app import db, service
    from app.settings import Settings
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    settings = Settings(data_root=tmp_path / "data", db_path=database, production=False, pipeline_mode="replay")
    real = pathlib.Path(service.__file__).resolve().parent / settings.sample_bundle_dirname / "open_weights_receipts.json"
    fake_app = tmp_path / "app"
    (fake_app / settings.sample_bundle_dirname).mkdir(parents=True)
    if content is not None:
        (fake_app / settings.sample_bundle_dirname / real.name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(service, "__file__", str(fake_app / "service.py"))
    with pytest.raises(RuntimeError):
        service.install_open_weights_receipts(settings)


def _install_with_fixture(tmp_path, monkeypatch, mutate, repin=True):
    """Boot the installer against a copy of the shipped fixture changed by `mutate(rows) -> bytes|None`."""
    import json
    import pathlib
    from app import db, service
    from app.settings import Settings
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    settings = Settings(data_root=tmp_path / "data", db_path=database, production=False, pipeline_mode="replay")
    real = pathlib.Path(service.__file__).resolve().parent / settings.sample_bundle_dirname / "open_weights_receipts.json"
    rows = json.loads(real.read_text(encoding="utf-8"))
    original = real.read_bytes()
    raw = mutate(rows) or (json.dumps(rows).encode() if rows != json.loads(original) else original)
    fake_app = tmp_path / "app"
    (fake_app / settings.sample_bundle_dirname).mkdir(parents=True)
    (fake_app / settings.sample_bundle_dirname / real.name).write_bytes(raw)
    monkeypatch.setattr(service, "__file__", str(fake_app / "service.py"))
    if repin:
        import hashlib
        links = dict(proof_links.load(), open_weights_fixture_sha256=hashlib.sha256(raw).hexdigest())
        monkeypatch.setattr(proof_links, "load", lambda: links)
    service.install_open_weights_receipts(settings)


def test_shipped_fixture_matches_its_pin(tmp_path, monkeypatch):
    _install_with_fixture(tmp_path, monkeypatch, lambda rows: None, repin=False)


def test_a_tampered_byte_fails_startup(tmp_path, monkeypatch):
    import json
    with pytest.raises(RuntimeError, match="sha256"):
        _install_with_fixture(tmp_path, monkeypatch, lambda rows: json.dumps(rows).encode() + b" ", repin=False)


def _break(field, value):
    def mutate(rows):
        rows[0][field] = value
    return mutate


@pytest.mark.parametrize("field,value", [
    ("estimated_cost_usd", "not-a-number"), ("estimated_cost_usd", "-0.01"), ("estimated_cost_usd", "NaN"),
    ("latency_ms", -1), ("stage", "mystery"), ("status", "weird"),
])
def test_invalid_row_values_fail_startup(tmp_path, monkeypatch, field, value):
    with pytest.raises(RuntimeError, match="invalid row"):
        _install_with_fixture(tmp_path, monkeypatch, _break(field, value))


def test_wrong_row_count_fails_startup(tmp_path, monkeypatch):
    def mutate(rows):
        del rows[23:]
    with pytest.raises(RuntimeError, match="row count"):
        _install_with_fixture(tmp_path, monkeypatch, mutate)


def test_healthz_is_503_unless_all_receipts_are_installed(tmp_path):
    from fastapi.testclient import TestClient
    from app import db
    from app.main import create_app
    from app.settings import Settings
    settings = Settings(data_root=tmp_path / "data", db_path=tmp_path / "db.sqlite3", production=False, pipeline_mode="replay")
    client = TestClient(create_app(settings))
    assert client.get("/healthz").status_code == 200
    with db.connect(settings.db_path) as conn:
        conn.execute("DELETE FROM model_calls WHERE id = (SELECT id FROM model_calls WHERE job_id = ? LIMIT 1)", (NEWER,))
        conn.commit()
    assert client.get("/healthz").status_code == 503


FIXTURE_PREFIX = f"receipts:{NEWER}:"


def _settings(tmp_path):
    from app import db
    from app.settings import Settings
    database = tmp_path / "db.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return Settings(data_root=tmp_path / "data", db_path=database, production=False, pipeline_mode="replay")


def _add_real_rows(database, n, model="spark-open", with_job=True):
    from app import db
    with db.connect(database) as conn:
        if with_job:
            conn.execute(
                "INSERT OR IGNORE INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at, deadline_at, confirmed_at,"
                " completed_at, cost_cap_usd, readback_fired_n) VALUES (?, 'done', 'a', 'i', 'v', 't', 't', 't', 't', '0', 0)", (NEWER,))
        for i in range(n):
            conn.execute(
                "INSERT INTO model_calls (id, job_id, stage, model, endpoint, latency_ms, units_in, units_out,"
                " estimated_cost_usd, status, evidence_json, created_at) VALUES (?, ?, 'second_check', ?, 'e', 1, 1, 1, '0', 'ok', '{}', 't')",
                (f"real-{model}-{i}", NEWER, model))
        conn.commit()


def _ids(database):
    from app import db
    with db.connect(database) as conn:
        return sorted(r[0] for r in conn.execute("SELECT id FROM model_calls WHERE job_id = ?", (NEWER,)))


def _install(settings):
    from app import service
    service.install_open_weights_receipts(settings)


def test_installer_leaves_a_real_only_job_untouched(tmp_path):
    settings = _settings(tmp_path)
    _add_real_rows(settings.db_path, 2)
    _install(settings)
    assert _ids(settings.db_path) == ["real-spark-open-0", "real-spark-open-1"]


def test_installer_deletes_only_fixture_rows_beside_real_ones(tmp_path):
    settings = _settings(tmp_path)
    _install(settings)
    assert len(_ids(settings.db_path)) == LINKS["open_weights_receipt_count"]
    _add_real_rows(settings.db_path, 2)
    _install(settings)
    assert _ids(settings.db_path) == ["real-spark-open-0", "real-spark-open-1"]
    _install(settings)
    assert _ids(settings.db_path) == ["real-spark-open-0", "real-spark-open-1"]


def test_saved_copy_line_only_for_a_fixture_sourced_job(tmp_path):
    settings = _settings(tmp_path)
    assert proof_links.saved_copy_line(NEWER, settings.db_path) == ""
    _install(settings)
    assert proof_links.saved_copy_line(NEWER, settings.db_path).startswith("Saved copy of a recorded run")
    _add_real_rows(settings.db_path, 1)
    assert proof_links.saved_copy_line(NEWER, settings.db_path) == ""
    from app import db
    with db.connect(settings.db_path) as conn:
        conn.execute("DELETE FROM model_calls WHERE id LIKE 'receipts:%'")
        conn.commit()
    assert proof_links.saved_copy_line(NEWER, settings.db_path) == ""


def test_real_run_pages_carry_no_saved_copy_line(tmp_path):
    from fastapi.testclient import TestClient
    from app import db
    from app.main import create_app
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert "Saved copy" in client.get(f"/calls/{NEWER}").text
    with db.connect(settings.db_path) as conn:
        conn.execute("DELETE FROM model_calls WHERE id LIKE 'receipts:%'")
        conn.commit()
    _add_real_rows(settings.db_path, 1, with_job=False)
    assert "Saved copy" not in client.get(f"/calls/{NEWER}").text
    assert "Saved copy" not in client.get(f"/jobs/{NEWER}", headers={"accept": "text/html"}).text


def _healthz_client(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    settings = _settings(tmp_path)
    return TestClient(create_app(settings)), settings.db_path


def _replace_with_real(database, n, model="spark-open"):
    from app import db
    with db.connect(database) as conn:
        conn.execute("DELETE FROM model_calls WHERE job_id = ?", (NEWER,))
        conn.commit()
    _add_real_rows(database, n, model, with_job=False)


def test_healthz_needs_exactly_the_fixture_count_while_the_receipts_are_the_fixture(tmp_path):
    client, database = _healthz_client(tmp_path)
    assert client.get("/healthz").status_code == 200
    from app import db
    with db.connect(database) as conn:
        conn.execute("INSERT INTO model_calls SELECT 'receipts:' || ? || ':extra', job_id, stage, model, endpoint, latency_ms, units_in,"
                     " units_out, estimated_cost_usd, status, evidence_json, created_at FROM model_calls WHERE job_id = ? LIMIT 1", (NEWER, NEWER))
        conn.commit()
    assert client.get("/healthz").status_code == 503


def test_healthz_real_run_needs_the_configured_second_check_minimum(tmp_path):
    client, database = _healthz_client(tmp_path)
    low = LINKS["open_weights_real_second_check_min"]
    _replace_with_real(database, low - 1)
    assert client.get("/healthz").status_code == 503
    _replace_with_real(database, low)
    assert client.get("/healthz").status_code == 200
    _replace_with_real(database, low + 5, model="spark-1.3")
    assert client.get("/healthz").status_code == 503


def test_healthz_fails_closed_when_the_job_has_no_receipts(tmp_path):
    client, database = _healthz_client(tmp_path)
    from app import db
    with db.connect(database) as conn:
        conn.execute("DELETE FROM model_calls WHERE job_id = ?", (NEWER,))
        conn.commit()
    assert client.get("/healthz").status_code == 503


@pytest.mark.parametrize("edit", ["drop", "zero", "negative", "bool", "text"])
def test_real_second_check_minimum_is_validated(tmp_path, monkeypatch, edit):
    import pathlib
    text = (pathlib.Path(__file__).resolve().parents[1] / "config/proof_links.toml").read_text(encoding="utf-8")
    line = f'open_weights_real_second_check_min = {LINKS["open_weights_real_second_check_min"]}'
    assert line in text
    bad = {"drop": "", "zero": "open_weights_real_second_check_min = 0", "negative": "open_weights_real_second_check_min = -3",
           "bool": "open_weights_real_second_check_min = true", "text": 'open_weights_real_second_check_min = "4"'}[edit]
    (tmp_path / "config").mkdir()
    (tmp_path / "config/proof_links.toml").write_text(text.replace(line, bad), encoding="utf-8")
    monkeypatch.setattr(proof_links, "__file__", str(tmp_path / "app" / "proof_links.py"))
    with pytest.raises(ValueError):
        proof_links.load()
