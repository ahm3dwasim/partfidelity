"""config/demo_preview.toml loads strictly: a missing or misspelled key, a bad value or a banned claim fails loudly."""
import json
import subprocess
import sys

import pytest

from demo.preview import config as dp

SRC = dp.CONFIG.read_text()


def _load(tmp_path, text):
    path = tmp_path / "demo_preview.toml"
    path.write_text(text)
    return dp.load(path)


def test_shipped_config_loads_and_beats_cover_the_whole_preview():
    cfg = dp.load()
    t = cfg["timing"]
    assert 10 <= t["duration_s"] <= 15
    assert t["footage"][0] == 0 and t["badge"][1] == t["duration_s"]
    assert cfg["footage"]["end_s"] <= cfg["footage"]["latest_s"] == 189.0


def test_module_prints_the_checked_config_as_json():
    out = subprocess.run([sys.executable, "-m", "demo.preview.config"], capture_output=True, text=True,
                         check=True, cwd=dp.CONFIG.parents[1]).stdout
    assert json.loads(out)["output"]["width"] == 1920


@pytest.mark.parametrize("text", ["Verification complete", "Safely removed", "Check the accuracy", "A complete repair", "Verified", "Safe to open",
                                  "Repair complete", "The repair is complete", "Fully repaired", "Repaired and complete",
                                  "The repair has been completed", "The fix is complete", "The restoration is complete",
                                  "Fix completed", "Restoration has been completed", "The fixes were completed",
                                  "The repair has been fully completed.", "The restoration has been totally completed."])
def test_banned_stems_fail(text):
    cfg = dp.load()
    cfg["captions"]["guide"] = text
    with pytest.raises(dp.DemoPreviewConfigError):
        dp.validate(cfg)


def test_ordinary_caption_passes():
    cfg = dp.load()
    cfg["captions"]["guide"] = "Step 4 lists the tools, warnings and parts to check."
    dp.validate(cfg)


def test_500_character_caption_fails_the_two_line_box():
    cfg = dp.load()
    cfg["captions"]["guide"] = " ".join(["word"] * 110)
    assert len(cfg["captions"]["guide"]) >= 500
    with pytest.raises(dp.DemoPreviewConfigError, match="lines"):
        dp.validate(cfg)


def test_no_machine_paths_in_config():
    assert "/Users/" not in SRC and "/opt/" not in SRC and "~/" not in SRC


@pytest.mark.parametrize(
    "old,new",
    [
        ("width = 1920", "widht = 1920"),                          # misspelled key
        ("fps = 30", "fps = \"30\""),                              # wrong type
        ("crf = 20", "# crf gone"),                                # missing key
        ("end_s = 41.1", "end_s = 190.0"),                         # footage past 3:09
        ("to_guide = [4.0, 5.4]", "to_guide = [4.1, 5.4]"),        # gap between beats
        ("duration_s = 14.0", "duration_s = 16.0"),                # outside 10-15 s
        ('site = "https://was7mahmed', 'site = "http://was7mahmed'),
        ("Review this label.", "Verified label."),                 # banned claim
        ("parts to check.", "parts, all safe."),                   # banned claim
        ("push_s = [5.7, 7.1]", "push_s = [5.7, 8.0]"),             # hold shorter than 2 s
        ("push_s = [5.7, 7.1]", "push_s = [5.0, 7.1]"),            # move starts before the guide beat
        ("margin_frac = 0.06", "margin_frac = 0.4"),
        ("text_px = 36", "text_px = 30"),                          # caption text under 34 px
        ("stage_gap_px = 48", "stage_gap_px = 40"),                # gap under 48 px
        ("fade_s = 0.3", "fade_s = 0.5"),                          # crossfade outside 0.25-0.35 s
        ('align = "center"', 'align = "right"'),
        ("max_chars = 60", "max_chars = 80"),
        ("cuts_s = [5.4, 9.56]", "cuts_s = [3.0, 9.65]"),          # footage caption cut before the card has landed
        ("cuts_s = [5.4, 9.56]", "cuts_s = [5.05, 12.0]"),          # match caption holds under 2.5 s
        ('match_tag = "Recorded probe run · Simulated replay"', 'match_tag = "Verified replay"'),   # banned claim
        ("cuts_s = [5.4, 9.56]", "cuts_s = [5.0, 9.56]"),            # cut lands while the card is still flying (ends 5.008)
        ("cuts_s = [5.4, 9.56]", "cuts_s = [5.05, 9.56]"),           # crossfade runs through the dissolve (ends 5.4)
        ("cuts_s = [5.4, 9.56]", "cuts_s = [5.4, 9.65]"),            # crossfade runs into the table entrance (starts 9.865)
        ("table_in_start = 0.95", "table_in_start = 0.6"),          # the table rises while the caption crossfades
        ("cuts_s = [5.4, 9.56]", "cuts_s = [5.05, 9.0]"),          # cut lands while the slide-out or camera is moving
        ("cuts_s = [5.4, 9.56]", "cuts_s = [5.45, 9.65]"),         # crossfade runs into the camera move (starts 5.7)
        ("slide_out = [0.0, 0.5]", "slide_out = [0.0, 0.6]"),        # a longer slide-out moves the endpoint past the cut
        ("connector_gap_px = 64", "# gap gone"),                    # new composition keys are required
        ("card_aspect = [16, 9]", "card_aspect = [16.0, 9]"),
        ("stroke_px = 2.5", "stroke_px = \"2.5\""),
        ("footage_rise = 0.14", "footage_rise = 1.4"),              # a fraction above 1
        ("link = [0.2, 0.36]", "link = [0.4, 0.36]"),               # span out of order
        ("size_px = 17", "sizee_px = 17"),                         # misspelled tag key
        ("mark_px = 52", "# mark gone"),                           # missing key
        ("[layout]", "[layout]\nextra = 1"),                       # unknown key
    ],
)
def test_bad_config_fails_loudly(tmp_path, old, new):
    assert old in SRC
    with pytest.raises(dp.DemoPreviewConfigError):
        _load(tmp_path, SRC.replace(old, new, 1))


def test_composition_values_come_from_config():
    """Each value that used to be a literal in preview.js/css is a required config key the page reads."""
    cfg = dp.load()
    js = (dp.CONFIG.parents[1] / "demo" / "preview" / "preview.js").read_text()
    for table, key in [("layout", "connector_gap_px"), ("layout", "card_aspect"), ("ring", "row_inset_px"),
                       ("ring", "note_pad_px"), ("tag", "size_px"), ("motion", "fly"), ("motion", "note_ring")]:
        assert key in cfg[table]
        assert key in js, key
