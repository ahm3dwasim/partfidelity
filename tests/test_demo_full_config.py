"""config/demo_full.toml loads strictly: a missing or misspelled key, a bad value, a pace break or a banned claim fails loudly."""
import json
import re
import subprocess
import sys

import pytest

from demo.full import config as df

SRC = df.CONFIG.read_text()


def _load(tmp_path, text):
    path = tmp_path / "demo_full.toml"
    path.write_text(text)
    return df.load(path)


def test_shipped_config_loads_and_runs_inside_the_length_window():
    cfg = df.load()
    o = cfg["output"]
    assert o["min_total_s"] <= sum(s["dur"] for s in cfg["scene"]) <= o["max_total_s"]
    assert cfg["scene"][0]["kind"] == "footage" and cfg["scene"][-1]["kind"] == "card"
    assert cfg["source"]["latest_s"] == 189.0


def test_module_prints_the_checked_config_with_scene_starts():
    out = subprocess.run([sys.executable, "-m", "demo.full.config"], capture_output=True, text=True,
                         check=True, cwd=df.CONFIG.parents[1]).stdout
    cfg = json.loads(out)
    assert cfg["output"]["width"] == 1920
    starts = [s["start"] for s in cfg["scene"]]
    assert starts[0] == 0 and starts == sorted(starts)


@pytest.mark.parametrize(
    "old,new",
    [
        ("width = 1920", "widht = 1920"),                                  # misspelled key
        ("fps = 30", 'fps = "30"'),                                        # wrong type
        ("crf = 23", "# crf gone"),                                        # missing key
        ("latest_s = 189.0", "latest_s = 200.0"),                          # footage allowed past 3:09
        ("Repair videos run on and on.", "The repair has been fully completed."),   # adverb between noun and verb
        ("Repair videos run on and on.", "The restoration has been totally completed."),
        ('ffmpeg = "ffmpeg"', 'ffmpeg = ""'),                              # empty path
        ("map = [[0.0, 2.0], [8.0, 9.4]]", "map = [[0.0, 2.0], [8.0, 195.0]]"),  # footage past 3:09
        ("dur = 10.6", "dur = 14.0"),                                      # a stop holds over 3 s
        ("dur = 10.6", "dur = 8.0"),                                       # a stop holds under 2 s
        ('"g1.card:0-0.55"', '"g1.nothing"'),                              # unknown mark
        ('"g1.card:0-0.55"', '"g1.card:0.6-0.5"'),                         # bad fraction
        ("at = 6.0", "at = 3.0"),                                          # caption cut inside a camera move
        ("min_total_s = 135.0", "min_total_s = 300.0"),                    # total outside the window
        ("Muse Spark Open gives a second opinion", "Muse Spark Open gives a verified opinion"),   # banned claim
        ("Repair videos run on and on.", "Repair videos run on and on and on, and on, and on, and on, and on, and on, and on."),   # too long
        ("table_in = [0.3, 0.75]", "table_in = [0.0, 0.6]"),               # table rises while the scene crossfades in
        ("ring_border_px = 2", "# ring gone"),                             # the shared css reads every [panel] key
        ("tag_weight = 600", "tag_weight = 1200"),                         # weight above 900
        ("flight_s = 1.4", "flight_s = 1.6"),                              # card flight still running at the first guide caption
        ("reveal_start_s = 0.5", "reveal_start_s = 0.0"),                  # card text reveal overlaps the scene crossfade
        ("logo_icon_start_s = 0.3", "logo_icon_start_s = 0.1"),            # logo reveal overlaps the end scene's crossfade
        ("[layout]", "[layout]\nextra = 1"),                               # unknown key
        ('track = ""', 'track = "no/such/file.m4a"'),                      # missing audio file
        ("images = [\"g4\"]", "images = [\"nope\"]"),                      # unknown capture
        ('kind = "footage"\ndur = 8.0', 'kind = "card"\ndur = 8.0'),       # from_footage with no footage scene before it
    ],
)
def test_bad_config_fails_loudly(tmp_path, old, new):
    assert old in SRC
    head, _, tail = SRC.rpartition(old)   # the last match: a caption text also appears in [captions] approved, above the scenes
    with pytest.raises(df.DemoFullConfigError):
        _load(tmp_path, head + new + tail)


def test_every_panel_value_reaches_the_page():
    """preview.css reads these variables; full.js must set each from [panel], or the full cut loses the rings, tags and scrim."""
    root = df.CONFIG.parents[1] / "demo"
    css, js = (root / "preview" / "preview.css").read_text(), (root / "full" / "full.js").read_text()
    for var in sorted(set(re.findall(r"var\(--((?:ring|link|panel-tag|dim)-[a-z-]+)\)", css))):
        assert f"'--{var}': cfg.panel." in js, var


def test_composition_values_come_from_config_into_the_page_css(tmp_path):
    """A changed [composition] value changes the loaded config, and full.js hands every full.css variable to the page from it."""
    cfg = _load(tmp_path, SRC.replace("bar_height_px = 6", "bar_height_px = 9").replace("card_gap_px = 30", "card_gap_px = 41"))
    assert (cfg["composition"]["bar_height_px"], cfg["textcard"]["card_gap_px"]) == (9, 41)
    root = df.CONFIG.parents[1] / "demo" / "full"
    css, js = (root / "full.css").read_text(), (root / "full.js").read_text()
    set_by_js = set(re.findall(r"'--([a-z-]+)':", js))
    used = set(re.findall(r"var\(--([a-z-]+)\)", css))
    assert {v for v in used if not v.startswith('pf-')} <= set_by_js
    assert "'--bar-height': C.bar_height_px" in js and "'--card-gap': T.card_gap_px" in js


def test_caption_tag_weight_comes_from_caption_box_not_the_panel(tmp_path):
    cfg = _load(tmp_path, SRC.replace("tag_font_weight = 600", "tag_font_weight = 700"))
    assert cfg["caption_box"]["tag_font_weight"] == 700 and cfg["panel"]["tag_weight"] == 600
    root = df.CONFIG.parents[1] / "demo"
    js = (root / "full" / "full.js").read_text()
    css = (root / "preview" / "preview.css").read_text()
    assert "'--caption-tag-weight': CB.tag_font_weight" in js
    caption_tag = css.split(".caption .tag {")[1].split("}")[0]
    assert "font-weight: var(--caption-tag-weight, var(--panel-tag-weight))" in caption_tag


def test_empty_repo_url_is_read_from_built_toml(tmp_path):
    import tomllib
    built = tomllib.loads(df.BUILT.read_text())["page"]["repository_url"]
    assert df.load()["links"]["repo_url"] == built
    assert _load(tmp_path, SRC)["links"]["repo_url"] == built


@pytest.mark.parametrize("bad", ["", "https://", "https://user:pass@example.com/repo", "http://x"])
def test_bad_built_repository_url_fails_loud(tmp_path, monkeypatch, bad):
    import tomllib
    good = tomllib.loads(df.BUILT.read_text())["page"]["repository_url"]
    copy = tmp_path / "built.toml"
    copy.write_text(df.BUILT.read_text().replace(good, bad, 1))
    monkeypatch.setattr(df, "BUILT", copy)
    with pytest.raises(df.DemoFullConfigError, match="repository_url"):
        df.load()
