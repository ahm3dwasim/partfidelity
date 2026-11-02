"""Every shipped caption equals an approved string in its config's [captions] approved list, and none trips the word filter.
The exact list is the guard for what the video says; the filter is advisory lint on top of it."""
import re
import tomllib

import pytest

from demo.full import config as df
from demo.preview import config as dp

CONFIGS = [dp.CONFIG, df.CONFIG]


def _shipped(path):
    cfg = tomllib.loads(path.read_text())
    if "scene" in cfg:
        return [v for sc in cfg["scene"] for c in sc["caption"] for v in (c["tag"], c["text"])]
    return [cfg["captions"][k] for k in dp.CAPTION_KEYS]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_caption_is_approved_exactly(path):
    approved = set(tomllib.loads(path.read_text())["captions"]["approved"])
    assert {s for s in _shipped(path) if s not in approved} == set()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_approved_list_has_no_filtered_phrasing_and_no_unused_entries(path):
    cfg = tomllib.loads(path.read_text())
    approved = cfg["captions"]["approved"]
    assert not [a for a in approved if dp.BANNED.search(a)]
    assert set(approved) == set(_shipped(path))


def test_an_edited_caption_is_not_in_the_approved_list(tmp_path):
    head, _, tail = df.CONFIG.read_text().rpartition("Repair videos run on and on.")   # the scene's copy, not the approved entry
    path = tmp_path / "c.toml"
    path.write_text(head + "Repair videos run on." + tail)
    approved = set(tomllib.loads(path.read_text())["captions"]["approved"])
    assert [s for s in _shipped(path) if s not in approved] == ["Repair videos run on. You pause, rewind and lose your place."]


def test_no_committed_config_names_a_machine_path():
    bad = re.compile(r"/Users/|/opt/|/home/|~")
    for path in sorted(df.CONFIG.parent.glob("demo_*.toml")):
        assert not bad.search(path.read_text()), path.name   # whole file, comments included
