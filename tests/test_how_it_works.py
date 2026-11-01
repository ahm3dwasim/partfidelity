"""The /how-it-works page: renders every stage, reads its numbers from the app, calls no model."""
import pathlib
import re
import socket
import tomllib
from html import escape as html_escape

import pytest
from fastapi.testclient import TestClient

from app import how_config, how_media
from app.main import create_app
from app.routes import how

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = how_config.CONFIG.read_text()


needs_frames = pytest.mark.skipif(
    bool(how_media.problems(REPO)),
    reason="private how-it-works frames not provisioned here: python3 tools/how_media.py unpack",
)


def _load(tmp_path, text):
    path = tmp_path / "how.toml"
    path.write_text(text)
    return how_config.load(path)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the How it works page must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_route_renders_every_stage_without_a_network_call(no_network):
    resp = TestClient(create_app()).get("/how-it-works")
    assert resp.status_code == 200
    html = resp.text
    cfg = how_config.cached()
    ctx = how_config.context(cfg)
    assert len(ctx["stages"]) == 6
    for i, stage in enumerate(ctx["stages"]):
        assert f'data-stage="{i}"' in html
        assert stage["title"] in html
    for tech in ctx["tech"]:
        assert tech["name"] in html
    assert "Muse Spark 1.3" in html and "SAM 3.1" in html
    assert "{" not in html.split("<main", 1)[1].split("<script", 1)[0]


def test_figure_has_one_labelled_layer_per_stage_and_a_card_each():
    svg = how.FIGURE.read_text(encoding="utf-8")
    n = len(how_config.cached()["stages"])
    assert len(set(re.findall(r'data-layer="(\d+)"', svg))) == n
    assert len(set(re.findall(r'data-label="(\d+)"', svg))) == n
    assert 'aria-hidden="true"' in svg.split(">", 1)[0]
    html = TestClient(create_app()).get("/how-it-works").text
    assert "%%L" not in html  # every part label is filled from the config
    for stage in how_config.cached()["stages"]:
        assert f'>{stage["part"]}</text>' in html
    assert sorted(set(re.findall(r'data-card="(\d+)"', html))) == [str(i) for i in range(n)]
    assert len(re.findall(r'data-card="\d+"[^>]*data-on', html)) == 1  # no-JS: only the first stage is marked


def test_only_the_model_stages_are_labelled_ai():
    # stage 2 is the step model, 4 the outline model, 5 the read-back; the rest is ordinary code or fixed rules
    stages = how_config.cached()["stages"]
    assert [s["kind"] for s in stages] == ["code", "ai", "code", "ai", "ai", "rules"]
    html = TestClient(create_app()).get("/how-it-works").text
    kinds = how_config.cached()["kinds"]
    assert len(re.findall(r'class="how-kind" data-kind="ai"', html)) == 3
    assert kinds["code"] in html and kinds["rules"] in html


@needs_frames
def test_example_frames_exist_and_the_kept_ones_come_from_the_app_ranking():
    ex = how_config.example(how_config.cached())
    media = REPO / "static/ui/media/how"
    for f in ex["scored"]:
        assert (media / f"c{f.timestamp_s:g}.webp").is_file()
    for t in ex["kept"]:
        assert (media / f"k{t:g}.webp").is_file()
    assert ex["kept"] == [124.0, 127.5]
    for row in ex["strip"]:
        assert (media / row["file"]).is_file()
    assert (media / ex["sam"]["file"]).is_file()


def test_example_sam_confidence_leaves_the_readback_with_work_to_do():
    from app.contracts import READBACK_CONFIDENCE_CUTOFF, PART_CHECK_THRESHOLDS

    sam = how_config.cached()["example"]["sam"]
    assert PART_CHECK_THRESHOLDS[sam["part"]] <= sam["confidence"] < READBACK_CONFIDENCE_CUTOFF


def test_example_steps_match_the_recorded_run():
    import json

    recorded = json.loads((REPO / "tests/fixtures/run4_replay.json").read_text())["expected_guide"]
    steps = how_config.cached()["example"]["steps"]
    assert [(x["start"], x["end"], x["title"]) for x in steps] == [
        (r["start_s"], r["end_s"], r["action"]) for r in recorded
    ]
    pick = recorded[how_config.cached()["example"]["step_index"]]
    assert set(how_config.cached()["example"]["guide"]["hazards"]) == set(pick["hazards"])


def test_example_is_identified_as_separate_and_illustrative():
    html = TestClient(create_app()).get("/how-it-works").text
    assert "separate illustrative example" in html.lower()
    stage = html.split('id="stage-steps"', 1)[1].split('</li>\n<li class="how-card"', 1)[0]
    assert 'class="how-tag">Illustrative' in stage
    assert "separate illustrative example, not the linked sample guide" in stage.lower()
    assert "recorded results" not in html.lower()


def test_example_does_not_claim_a_readback_answer():
    html = TestClient(create_app()).get("/how-it-works").text
    ex = how_config.cached()["example"]
    assert html_escape(ex["readback_note"]) in html and html_escape(ex["guide_note"]) in html
    # the stage 5 and 6 example outputs carry the illustrative tag, never a pass
    assert html.count(how_config.cached()["page"]["illustrative"]) >= 2
    assert "passed" not in html.lower()


def test_loader_rejects_a_bad_example(tmp_path):
    for old, new in (
        ("confidence = 0.6718", "confidence = 0.9"),
        ("box = [224, 275, 1022, 530]", "box = [224, 275, 1022, 900]"),
        ("step_index = 3", "step_index = 9"),
        ('hazards = ["battery", "esd"]', 'hazards = ["nope"]'),
    ):
        assert old in SRC
        with pytest.raises(how_config.HowConfigError):
            _load(tmp_path, SRC.replace(old, new, 1))


def test_live_readouts_reach_the_script_config():
    html = TestClient(create_app()).get("/how-it-works").text
    block = re.search(r'<script type="application/json" id="how-config">(.*?)</script>', html, re.S).group(1)
    import json

    public = json.loads(block.replace("<\\/", "</"))
    assert [s["readout"] for s in public["stages"]] == [s["readout"] for s in how_config.context(how_config.cached())["stages"]]
    assert public["dwell_s"] > 0 and public["follow_s"] > 0
    assert (REPO / "static/ui/how-figure.js").is_file() and (REPO / "static/ui/how-iso.css").is_file()


def test_numbers_come_from_the_code_not_the_copy():
    from app import frame_scoring
    from app.contracts import SPARK_RUNS_PER_GUIDE

    facts = how_config.facts(how_config.cached())
    assert facts["best_frames"] == frame_scoring.FRAMES_KEPT_PER_STEP
    assert facts["runs"] == SPARK_RUNS_PER_GUIDE
    # the copy quotes them as placeholders, never as literals
    assert f"{{best_frames}}" in SRC and f"{{runs}}" in SRC


def test_model_name_falls_back_to_the_raw_id():
    cfg = how_config.cached()
    assert how_config.facts(cfg, spark_model_id="unlisted-model-9")["spark"] == "unlisted-model-9"
    assert "unlisted-model-9" in how.render(spark_model_id="unlisted-model-9")


def test_models_table_names_all_three_models_from_config():
    html = TestClient(create_app()).get("/how-it-works").text
    assert "Three models do the reading and checking" in html
    assert "Three Meta models do the reading and checking" in html
    assert "Muse Spark 1.3" in html and "SAM 3.1" in html and "Muse Spark Open" in html
    assert "background second opinion on each label" in html
    assert "flags the label for a person to review" in html
    # the table names the open model through a placeholder, never a literal
    assert 'name = "{spark_open}"' in SRC


def test_open_model_name_comes_from_the_open_readback_config():
    from app.providers.open_readback import load_config as load_open

    cfg = how_config.cached()
    model_id = load_open().model_id
    assert how_config.facts(cfg)["spark_open"] == cfg["names"].get(model_id, model_id)
    assert how_config.facts(cfg, open_model_id="unlisted-open-9")["spark_open"] == "unlisted-open-9"


@pytest.mark.parametrize(
    "old,new",
    [
        ('headline = ', 'headlinex = '),
        ("reads the video {runs} times", "reads the video {no_such_fact} times"),
        ("dwell_s = ", "dwell = "),
        ('kind = "rules"', 'kind = "magic"'),
        ('ai = "AI model"', 'model = "AI model"'),
    ],
)
def test_loader_fails_loudly(tmp_path, old, new):
    assert old in SRC
    with pytest.raises(how_config.HowConfigError):
        _load(tmp_path, SRC.replace(old, new, 1))


def test_loader_rejects_a_wrong_stage_count(tmp_path):
    head, _, rest = SRC.partition("[[stages]]")
    with pytest.raises(how_config.HowConfigError):
        _load(tmp_path, head + "[[stages]]" + rest.split("[[stages]]", 1)[0] + "[[tech]]" + rest.split("[[tech]]", 1)[1])


def test_config_is_shipped_in_the_modal_image():
    deploy = tomllib.loads((REPO / "deploy/modal_deploy.toml").read_text())
    shipped = deploy["modal"]["image_files"]
    assert "config/how_it_works.toml" in shipped


def test_example_frames_are_shipped_in_the_modal_image():
    deploy = tomllib.loads((REPO / "deploy/modal_deploy.toml").read_text())
    assert "static" in deploy["modal"]["image_folders"]


def test_footage_is_ignored_untracked_and_listed_in_the_manifest():
    import subprocess

    cfg = how_media.load(REPO)
    assert len(cfg["files"]) == 24
    tracked = subprocess.check_output(["git", "ls-files", "--", cfg["dir"], cfg["bundle"]], cwd=REPO, text=True)
    assert not tracked
    for e in cfg["files"]:
        subprocess.run(["git", "check-ignore", "-q", f"{cfg['dir']}/{e['name']}"], cwd=REPO, check=True)
    subprocess.run(["git", "check-ignore", "-q", cfg["bundle"]], cwd=REPO, check=True)
    deploy = tomllib.loads((REPO / "deploy/modal_deploy.toml").read_text())["modal"]
    assert "config/how_media.json" in deploy["image_files"]


@needs_frames
def test_provisioned_frames_match_the_manifest():
    assert how_media.problems(REPO) == []


def _fixture_root(tmp_path):
    """A tiny synthetic tree with its own manifest: three fake frames, no private footage."""
    import hashlib
    import json

    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    blobs = {f"pic{i}.webp": bytes([i]) * (10 + i) for i in range(3)}
    manifest = {"dir": "media", "bundle": "bundle/frames.tar",
                "files": [{"name": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in blobs.items()]}
    (root / "config/how_media.json").write_text(json.dumps(manifest))
    return root, blobs


def test_how_media_pack_unpack_check_roundtrip_and_fail_loudly(tmp_path):
    root, blobs = _fixture_root(tmp_path)
    (root / "media").mkdir()
    for n, b in blobs.items():
        (root / "media" / n).write_bytes(b)
    how_media.check(root)
    bundle = how_media.pack(root)
    for n in blobs:
        (root / "media" / n).unlink()
    with pytest.raises(how_media.MediaError, match=r"pic0\.webp: missing[\s\S]*how_media.py unpack"):
        how_media.check(root)
    assert how_media.unpack(root, bundle) == 3
    how_media.check(root)
    (root / "media/pic1.webp").write_bytes(b"x" * len(blobs["pic1.webp"]))  # same size, other content
    with pytest.raises(how_media.MediaError, match=r"pic1\.webp: SHA-256 differs"):
        how_media.check(root)
    assert how_media.problems(root, deep=False) == []  # the cheap per-request check sees size only
    (root / "media/pic2.webp").write_bytes(b"short")
    assert "pic2.webp: size 5" in how_media.problems(root, deep=False)[0]


def test_how_media_unpack_refuses_a_bad_or_missing_bundle(tmp_path):
    import io
    import tarfile

    root, blobs = _fixture_root(tmp_path)
    with pytest.raises(how_media.MediaError, match="bundle not found"):
        how_media.unpack(root)
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tar:
        info = tarfile.TarInfo("pic0.webp")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises(how_media.MediaError, match=r"pic0\.webp: size 3"):
        how_media.unpack(root, bad)
    assert not (root / "media").exists()  # nothing placed from a bundle that fails verification
    extra = tmp_path / "extra.tar"
    with tarfile.open(extra, "w") as tar:
        info = tarfile.TarInfo("evil.webp")
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(how_media.MediaError, match="unexpected member"):
        how_media.unpack(root, extra)


def test_page_renders_stage_text_without_frames_and_logs_why(monkeypatch, caplog):
    monkeypatch.setattr(how_media, "problems", lambda root=how_media.ROOT, deep=True: ["c124.webp: missing"])
    with caplog.at_level("ERROR"):
        resp = TestClient(create_app()).get("/how-it-works")
    assert resp.status_code == 200
    html = resp.text
    assert "<img" not in html.split("<main", 1)[1]
    for stage in how_config.context(how_config.cached())["stages"]:
        assert html_escape(stage["title"]) in html and html_escape(stage["plain"]) in html
    assert "c124.webp: missing" in caplog.text and "how_media.py unpack" in caplog.text


@needs_frames
def test_page_shows_frames_when_provisioned():
    assert "/static/ui/media/how/c124.webp" in TestClient(create_app()).get("/how-it-works").text


def test_deploy_refuses_without_frames_and_names_the_command(tmp_path, monkeypatch):
    import runpy
    import sys
    import types
    from unittest.mock import MagicMock

    modal = types.ModuleType("modal")
    modal.is_local = lambda: True
    modal.Image = MagicMock()
    modal.App = MagicMock()
    modal.App.return_value.function = lambda **kwargs: lambda f: f
    modal.Volume, modal.Secret = MagicMock(), MagicMock()
    modal.concurrent = lambda **kwargs: lambda f: f
    modal.asgi_app = lambda: lambda f: f
    monkeypatch.setitem(sys.modules, "modal", modal)
    from tools import sample_media
    from tools import catch_media
    monkeypatch.setattr(catch_media, "unpack", lambda root: None)
    monkeypatch.setattr(sample_media, "unpack", lambda root: None)  # reach the How it works refusal even without private sample media
    monkeypatch.setattr(how_media, "problems", lambda root=how_media.ROOT, deep=True: ["s3.webp: missing"])
    monkeypatch.syspath_prepend(str(REPO))
    with pytest.raises(SystemExit, match=r"s3\.webp: missing[\s\S]*python3 tools/how_media.py unpack"):
        runpy.run_path(str(REPO / "deploy/modal_app.py"))


def test_loader_rejects_bad_numbers_missing_images_and_zero_scores(tmp_path):
    for old, new in (
        ('t = 3\nfile = "s3.webp"', 't = "3"\nfile = "s3.webp"'),
        ('dwell_s = 3.6', 'dwell_s = inf'),
        ('file = "s3.webp"', 'file = "absent.webp"'),
    ):
        assert old in SRC
        with pytest.raises(how_config.HowConfigError):
            _load(tmp_path, SRC.replace(old, new, 1))
    zero = re.sub(r'(?m)^sharpness = .*$', 'sharpness = 0', SRC)
    with pytest.raises(how_config.HowConfigError):
        _load(tmp_path, zero)


def test_running_chunk_settings_reach_page():
    from types import SimpleNamespace

    app = create_app()
    app.state.settings = SimpleNamespace(chunk_s=37, chunk_overlap_s=7)
    html = TestClient(app).get('/how-it-works').text
    assert '37-second windows' in html and '7 seconds' in html


def test_motion_settings_are_exposed_and_validated(tmp_path):
    import json

    block = re.search(r'<script type="application/json" id="how-config">(.*?)</script>', how.render(), re.S).group(1)
    public = json.loads(block)
    for key, value in how_config.cached()['motion'].items():
        assert public[key] == value
    with pytest.raises(how_config.HowConfigError):
        _load(tmp_path, SRC.replace('reading_position = 0.45', 'reading_position = 1.1'))


def test_browser_contrast_and_reduced_motion():
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='chrome')
        page = browser.new_page(reduced_motion='reduce')
        page.set_content(how.render())
        for css in ('tokens.css', 'how-iso.css', 'how-page.css'):
            page.add_style_tag(content=(REPO / 'static/ui' / css).read_text())
        page.add_script_tag(content=(REPO / 'static/ui/how-figure.js').read_text())
        result = page.evaluate('''() => {
          const color = s => s.match(/[\\d.]+/g).slice(0, 3).map(Number).map(x => {
            x /= 255; return x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4;
          });
          const lum = rgb => .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2];
          const ratio = (a, b) => { a = lum(color(a)); b = lum(color(b)); return (Math.max(a,b)+.05)/(Math.min(a,b)+.05); };
          const bg = getComputedStyle(document.documentElement).getPropertyValue('--pf-ink-2').trim();
          const probe = document.createElement('span'); probe.style.color = bg; document.body.append(probe);
          const background = getComputedStyle(probe).color;
          return [...document.querySelectorAll('.how-cands li, .how-lbl-n, .how-lbl-t')].map(el => {
            let n = el, opacity = 1, filters = [];
            while (n) { const s = getComputedStyle(n); opacity *= Number(s.opacity); filters.push(s.filter); n = n.parentElement; }
            const s = getComputedStyle(el); return {ratio: ratio(el instanceof SVGElement ? s.fill : s.color, background), opacity, filters};
          });
        }''')
        assert result and all(x['ratio'] >= 4.5 and x['opacity'] == 1 and all(f == 'none' for f in x['filters']) for x in result)
        page.locator('[data-play]').click()
        page.wait_for_timeout(100)
        assert page.locator('[data-play]').get_attribute('aria-pressed') == 'false'
        assert page.evaluate('''() => { let count = 0; const old = window.setTimeout; window.setTimeout = (...args) => { count++; return old(...args); }; document.querySelector('[data-play]').click(); return count; }''') == 0
        page.emulate_media(reduced_motion='no-preference')
        page.locator('[data-play]').click()
        assert page.locator('[data-play]').get_attribute('aria-pressed') == 'true'
        page.emulate_media(reduced_motion='reduce')
        expect(page.locator('[data-play]')).to_have_attribute('aria-pressed', 'false')
        browser.close()


SHOTS = pathlib.Path("/tmp/hiw-shots/shots")


@pytest.fixture
def server():
    """The real app under uvicorn, so the page loads every production stylesheet and script from /static."""
    import threading
    import time

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server never started"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


def _open(browser, url, **kw):
    page = browser.new_page(**kw)
    page.goto(f"{url}/how-it-works", wait_until="load")
    return page


def _shot(page, name, **kw):
    try:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / name), **kw)
    except OSError:
        pass


def _stage(page):
    return page.evaluate("document.querySelector('.how-stepper [aria-current=step]')?.dataset.stage")


def test_browser_scroll_select_tour_and_keyboard(server):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = _open(browser, server, viewport={"width": 1440, "height": 900})
        count = len(how_config.cached()["stages"])
        # scrolling through the stages changes the active one, in order
        seen = []
        for i in range(count):
            page.evaluate("i => { const c = document.querySelectorAll('[data-card]')[i]; scrollTo(0, scrollY + c.getBoundingClientRect().top - 300); }", i)
            page.wait_for_function("i => document.querySelector('.how-stepper [aria-current=step]')?.dataset.stage === String(i)", arg=i)
            seen.append(_stage(page))
        assert seen == [str(i) for i in range(count)]
        # selecting a stage from the stepper makes it the active one
        page.locator('.how-stepper [data-stage="2"]').click()
        page.wait_for_function("document.querySelector('.how-stepper [aria-current=step]')?.dataset.stage === '2'")
        # normal motion: Play advances on its own, Pause stops it
        play = page.locator("[data-play]")
        play.click()
        assert play.get_attribute("aria-pressed") == "true"
        page.wait_for_function("document.querySelector('.how-stepper [aria-current=step]')?.dataset.stage === '3'", timeout=10000)
        play.evaluate("el => el.click()")  # a real click would scroll the header button into view and move the stage
        assert play.get_attribute("aria-pressed") == "false"
        stopped = _stage(page)
        page.wait_for_timeout(int(how_config.cached()["motion"]["dwell_s"] * 1000) + 800)
        assert _stage(page) == stopped
        # Enter/Space on the focused Play button keep the tour; any scrolling key hands control back
        play.focus()
        page.keyboard.press("Enter")
        assert play.get_attribute("aria-pressed") == "true"
        page.keyboard.press("Enter")  # pauses again (it is the button's own activation)
        assert play.get_attribute("aria-pressed") == "false"
        play.focus()
        page.keyboard.press("Space")
        assert play.get_attribute("aria-pressed") == "true"
        page.keyboard.press("Space")
        for key in ("PageDown", "PageUp", "ArrowDown", "ArrowUp", "End", "Home"):
            play.focus()
            page.keyboard.press("Enter")
            if play.get_attribute("aria-pressed") != "true":
                page.keyboard.press("Enter")
            assert play.get_attribute("aria-pressed") == "true"
            page.keyboard.press(key)
            assert play.get_attribute("aria-pressed") == "false", key
        browser.close()


def test_browser_390_has_no_horizontal_overflow(server):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = _open(browser, server, viewport={"width": 390, "height": 844})
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.evaluate("document.body.scrollWidth <= innerWidth")
        _shot(page, "how-390.png", full_page=True)
        wide = _open(browser, server, viewport={"width": 1440, "height": 900})
        _shot(wide, "how-1440.png")
        browser.close()


def test_browser_390_fallback_without_frames_has_no_overflow(server, monkeypatch):
    from playwright.sync_api import sync_playwright

    monkeypatch.setattr(how_media, "problems", lambda root=how_media.ROOT, deep=True: ["c124.webp: missing"])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = _open(browser, server, viewport={"width": 390, "height": 844})
        assert page.locator(".how-card img").count() == 0 and page.locator(".how-card").count() == 6
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        _shot(page, "how-390-missing-media.png", full_page=True)
        browser.close()


def test_how_media_temp_names_are_git_ignored():
    import subprocess

    for name in (".how-media-1.tmp", ".how-media-bundle-1.tmp", "assets/.how-media-1.tmp", "tmp/.how-media-bundle-1.tmp",
                 "static/ui/media/how/.how-media-1.tmp"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO).returncode == 0, name


def test_how_media_interrupted_unpack_leaves_no_temp(tmp_path, monkeypatch):
    root, blobs = _fixture_root(tmp_path)
    (root / "media").mkdir()
    for n, b in blobs.items():
        (root / "media" / n).write_bytes(b)
    bundle = how_media.pack(root)
    for n in blobs:
        (root / "media" / n).unlink()

    def boom(*a):
        raise OSError("interrupted")

    monkeypatch.setattr(how_media.os, "replace", boom)
    with pytest.raises(OSError):
        how_media.unpack(root, bundle)
    assert list((root / "media").glob(".*")) == []


def test_how_media_failed_pack_keeps_old_bundle(tmp_path, monkeypatch):
    import tarfile

    root, blobs = _fixture_root(tmp_path)
    (root / "media").mkdir()
    for n, b in blobs.items():
        (root / "media" / n).write_bytes(b)
    bundle = how_media.pack(root)
    before = bundle.read_bytes()

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tarfile.TarFile, "add", boom)
    with pytest.raises(OSError):
        how_media.pack(root)
    assert bundle.read_bytes() == before
    assert [p.name for p in bundle.parent.iterdir()] == [bundle.name]


def _manifest_with(tmp_path, **change):
    import json

    root, _ = _fixture_root(tmp_path)
    path = root / "config/how_media.json"
    cfg = json.loads(path.read_text())
    for k, v in change.items():
        if k == "entry":
            cfg["files"][0].update(v)
        else:
            cfg[k] = v
    path.write_text(json.dumps(cfg))
    return root


@pytest.mark.parametrize("change,match", [
    ({"dir": "../outside"}, "dir must be a relative path"),
    ({"dir": "/abs"}, "dir must be a relative path"),
    ({"bundle": "/abs/b.tar"}, "bundle must be a relative path"),
    ({"bundle": "a/../../b.tar"}, "bundle must be a relative path"),
    ({"files": []}, "non-empty"),
    ({"entry": {"name": "a/b.webp"}}, "bare file name"),
    ({"entry": {"size": "5"}}, "positive integer"),
    ({"entry": {"size": True}}, "positive integer"),
    ({"entry": {"size": 0}}, "positive integer"),
    ({"entry": {"sha256": "abc"}}, "64 hex"),
    ({"entry": {"sha256": "z" * 64}}, "64 hex"),
])
def test_how_media_manifest_rejects_bad_entries(tmp_path, change, match):
    with pytest.raises(how_media.MediaError, match=match):
        how_media.load(_manifest_with(tmp_path, **change))


def test_how_media_manifest_rejects_duplicate_names(tmp_path):
    import json

    root = _manifest_with(tmp_path)
    path = root / "config/how_media.json"
    cfg = json.loads(path.read_text())
    cfg["files"].append(dict(cfg["files"][0]))
    path.write_text(json.dumps(cfg))
    with pytest.raises(how_media.MediaError, match="duplicate name 'pic0.webp'"):
        how_media.load(root)


def test_how_media_pack_refuses_an_unignored_in_repo_bundle(tmp_path):
    root, blobs = _fixture_root(tmp_path)
    (root / "media").mkdir()
    for n, b in blobs.items():
        (root / "media" / n).write_bytes(b)
    with pytest.raises(how_media.MediaError, match="not git-ignored"):
        how_media.pack(root, REPO / "docs" / "leak-check.tar")
    assert not (REPO / "docs" / "leak-check.tar").exists()
    assert how_media.pack(root, tmp_path / "outside.tar").is_file()


def test_how_media_uppercase_digest_is_accepted(tmp_path):
    import json

    root, blobs = _fixture_root(tmp_path)
    (root / "media").mkdir()
    for n, b in blobs.items():
        (root / "media" / n).write_bytes(b)
    path = root / "config/how_media.json"
    cfg = json.loads(path.read_text())
    for f in cfg["files"]:
        f["sha256"] = f["sha256"].upper()
    path.write_text(json.dumps(cfg))
    how_media.check(root)


def test_how_media_git_environment_cannot_bypass_ignore_check(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/nonexistent")
    with pytest.raises(how_media.MediaError, match="not git-ignored"):
        how_media._refuse_unignored(REPO / "docs" / "leak-check.tar")
    how_media._refuse_unignored(REPO / "var" / "how-media-ignored.tar")


def test_how_media_missing_git_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(how_media.MediaError, match="git"):
        how_media._refuse_unignored(tmp_path / "bundle.tar")


def test_how_media_outside_repo_allowed(tmp_path):
    how_media._refuse_unignored(tmp_path / "bundle.tar")


def test_how_media_git_errors_refuse(tmp_path, monkeypatch):
    import subprocess

    for result in (subprocess.CompletedProcess([], 128, b"", b"fatal: other failure\n"),
                   subprocess.CompletedProcess([], 0, b"unexpected", b"")):
        monkeypatch.setattr(how_media.subprocess, "run", lambda *a, **k: result)
        with pytest.raises(how_media.MediaError, match="git"):
            how_media._refuse_unignored(tmp_path / "bundle.tar")
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("git", 1)
    monkeypatch.setattr(how_media.subprocess, "run", timeout)
    with pytest.raises(how_media.MediaError, match="git"):
        how_media._refuse_unignored(tmp_path / "bundle.tar")


def test_how_media_concurrent_packs_use_distinct_temps(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import tarfile
    import threading

    root, blobs = _fixture_root(tmp_path)
    (root / "media").mkdir()
    for name, data in blobs.items():
        (root / "media" / name).write_bytes(data)
    bundle = tmp_path / "bundle.tar"
    barrier = threading.Barrier(8, timeout=10)
    original_add = tarfile.TarFile.add

    def synchronized_add(self, name, *args, **kwargs):
        if pathlib.Path(name).name == "pic0.webp":
            barrier.wait()
        return original_add(self, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", synchronized_add)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: how_media.pack(root, bundle), range(8))) == [bundle] * 8
    with tarfile.open(bundle) as tar:
        assert {m.name: tar.extractfile(m).read() for m in tar} == blobs
    assert list(tmp_path.glob(".how-media-*.tmp")) == []


def test_page_h1_key_still_loads(tmp_path):
    """[page] `h1` is the former name of `headline`; an old config still loads."""
    src = how_config.CONFIG.read_text()
    assert "\nheadline = " in src
    old = tmp_path / "how_it_works.toml"
    old.write_text(src.replace("\nheadline = ", "\nh1 = ", 1))
    assert how_config.load(old)["page"]["headline"] == how_config.load()["page"]["headline"]
