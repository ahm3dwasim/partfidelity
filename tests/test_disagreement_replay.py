"""Simulated disagreement replay: config, markup labels, no-JS reading, phone layout (W-2816)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import disagreement_replay as dr
from app.routes.site import create_site_app
from test_ui_guide_step import live_base_url, live_browser  # noqa: F401 (fixtures)

BANNER = dr.load()["banner"]


def _edit(tmp_path: Path, old: str, new: str) -> Path:
    text = dr.CONFIG.read_text()
    assert old in text
    out = tmp_path / "replay.toml"
    out.write_text(text.replace(old, new))
    return out


@pytest.mark.parametrize("old,new", [
    ('shown_label = "keyboard"\n', ''),                                     # missing key
    ('banner = "Simulated replay: no model was called"', 'banner = "Replay"'),  # drops the simulated label
    ('[replay]', '[replay]\nextra = "x"'),                                 # extra key
    ('static_steps = [', 'static_steps = [""'),                            # blank step
    ('{label}', 'label'),                                                  # lost placeholder
])
def test_loader_fails_loudly(tmp_path, old, new):
    with pytest.raises(ValueError):
        dr.load(_edit(tmp_path, old, new))


def test_page_is_labelled_simulated_everywhere_and_never_says_verified():
    page = TestClient(create_site_app()).get("/disagreement-replay")
    assert page.status_code == 200
    assert page.text.count(BANNER) >= 3  # intro pill, card badge, footer
    low = page.text.lower()
    assert "verified" not in low and "observed disagreement" not in low
    assert "separate from the edit catch" in page.text and 'href="/catch"' in page.text


def test_no_js_fallback_reads_in_full_and_controls_are_hidden_until_script_runs():
    text = TestClient(create_site_app()).get("/disagreement-replay").text
    cfg = dr.load()
    for step in cfg["static_steps"]:
        assert step.replace('"', "&quot;") in text.replace("&#x27;", "'")
    assert "Another AI check disagrees with this label." in text  # the note is server-rendered
    assert 'id="replay-live" hidden' in text  # controls need script, so they stay hidden without it


def test_built_links_to_the_replay_and_lists_the_feature():
    text = TestClient(create_site_app()).get("/built").text
    assert 'href="/disagreement-replay"' in text and "Disagreement probe and simulated replay" in text


def test_recorded_probe_table_receipts_links_and_limit():
    client = TestClient(create_site_app())
    page = client.get("/disagreement-replay").text
    cfg = dr.load()
    assert page.index("Recorded probe") < page.index(cfg["card_heading"])  # above the simulated replay
    for url, label, answer in (("/disagreement-replay/probe-run-1.json", "battery", "yes"),
                               ("/disagreement-replay/probe.json", "keyboard", "no")):
        assert f'href="{url}"' in page
        res = client.get(url)
        assert res.status_code == 200
        wrong = res.json()["cases"][0]
        assert (wrong["request_label"], wrong["model_answer"]) == (label, answer)
    assert page.count("meta-models/muse-spark-open-1026") == 1 and "Model: meta-models/muse-spark-open-1026, 2026-11-01" in page
    assert "not an accuracy measure" in page and "visible anywhere in the photo" in page
    assert "verified" not in page.lower()


def test_hero_copy_is_true_about_the_real_probe_and_the_scripted_replay():
    cfg = dr.load()
    text = TestClient(create_site_app()).get("/disagreement-replay").text
    hero = text[text.index('<div class="built-intro">'):text.index('<section')].lower()
    assert "simulated" not in hero and "everything below is a scripted screen" not in text.lower()
    assert cfg["title"] == "Second check: recorded probe and replay" and "two recorded runs" in hero and "label question plus a fan control" in hero and "no model is called" in hero
    assert text.index(cfg["replay_heading"]) < text.index(f'<p class="replay-banner">{BANNER}')


def test_replay_label_matches_the_real_absent_label():
    assert dr.load()["shown_label"] == "keyboard"
    text = TestClient(create_site_app()).get("/disagreement-replay").text
    assert '<strong data-replay-label>keyboard</strong>' in text and "reads &quot;keyboard&quot;" in text


def test_missing_receipt_drops_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "ROOT", tmp_path)
    assert "Recorded probe" not in TestClient(create_site_app()).get("/disagreement-replay").text
    assert TestClient(create_site_app()).get("/disagreement-replay/probe.json").status_code == 404


def test_receipts_ship_in_the_deploy_image():
    import tomllib
    files = tomllib.loads((dr.ROOT / "deploy/modal_deploy.toml").read_text())["modal"]["image_files"]
    for name in ("run-1-battery.json", "run-2-keyboard.json"):
        assert f"docs/evidence/disagreement-probe/{name}" in files


def test_confirm_undo_rename_persist_and_phone_layout(live_browser, live_base_url, tmp_path):  # noqa: F811
    flags = dr.open_readback.load_config()
    page = live_browser.new_page(viewport={"width": 390, "height": 900})
    page.goto(live_base_url + "/disagreement-replay")
    page.evaluate("localStorage.clear()")
    page.reload()
    assert page.locator("#replay-live").is_visible()
    assert page.locator("[data-replay-status]").inner_text() == flags.flag_text
    page.screenshot(path=str(tmp_path / "fix3-390-flagged.png"), full_page=True)
    page.click("[data-replay-confirm]")
    assert page.locator("[data-replay-status]").inner_text() == flags.flag_confirmed_text
    page.reload()  # persists
    assert page.locator("[data-replay-status]").inner_text() == flags.flag_confirmed_text
    page.screenshot(path=str(tmp_path / "replay-390-confirmed.png"), full_page=True)
    page.click("[data-replay-undo]")
    assert page.locator("[data-replay-status]").inner_text() == flags.flag_text
    page.reload()
    assert page.locator("[data-replay-status]").inner_text() == flags.flag_text
    page.click("form[data-replay-rename] button")  # empty label is refused
    assert page.locator("[data-replay-error]").inner_text() == dr.load()["rename_empty"]
    page.fill("form[data-replay-rename] input", "cooling fan")
    page.click("form[data-replay-rename] button")
    assert page.locator("[data-replay-label]").inner_text() == "cooling fan"
    assert "none ran" in page.locator("[data-replay-status]").inner_text()
    page.reload()
    assert page.locator("[data-replay-label]").inner_text() == "cooling fan"
    page.screenshot(path=str(tmp_path / "replay-390-renamed.png"), full_page=True)
    page.click("[data-replay-undo]")
    assert page.locator("[data-replay-label]").inner_text() == dr.load()["shown_label"]
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    page.close()
    nojs = live_browser.new_context(viewport={"width": 390, "height": 900}, java_script_enabled=False).new_page()
    nojs.goto(live_base_url + "/disagreement-replay")
    assert not nojs.locator("#replay-live").is_visible()
    assert nojs.locator(".replay-steps li").count() == len(dr.load()["static_steps"])
    assert nojs.locator(".replay-steps").is_visible()
    nojs.screenshot(path=str(tmp_path / "replay-390-nojs.png"), full_page=True)


def test_recorded_section_shots_at_phone_and_desktop(live_browser, live_base_url, tmp_path):  # noqa: F811
    for width in (390, 1440):
        page = live_browser.new_page(viewport={"width": width, "height": 900})
        page.goto(live_base_url + "/disagreement-replay")
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        assert page.evaluate("(() => { const w = document.querySelector('.replay-table-wrap'); return w.scrollWidth <= w.clientWidth; })()")
        assert page.locator(".replay-table a").first.is_visible()
        page.screenshot(path=str(tmp_path / f"fix3-{width}-recorded.png"), full_page=True)
        page.close()


def test_label_tokens_show_literally_and_a_storage_failure_is_stated(live_browser, live_base_url):  # noqa: F811
    cfg = dr.load()
    page = live_browser.new_page(viewport={"width": 390, "height": 900})
    page.add_init_script("Storage.prototype.setItem = function () { throw new Error('quota'); };")
    page.goto(live_base_url + "/disagreement-replay")
    page.fill("form[data-replay-rename] input", "a$&b")
    page.click("form[data-replay-rename] button")
    assert page.locator("[data-replay-label]").inner_text() == "a$&b"
    assert page.locator("[data-replay-status]").inner_text() == cfg["state_renamed"].replace("{label}", "a$&b")
    assert page.locator("[data-replay-persist]").inner_text() == cfg["persist_failed_note"]
    page.close()
