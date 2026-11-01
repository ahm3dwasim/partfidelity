"""Outline config and wording match.

Colour, stroke width and the smallest traceable mask live in
config/outline.toml, read through the fail-loud loader; no literals in
the renderer. Live copy never says "traced": live outlines follow box
edges, and only the sample guide is traced (config/built.toml names that
sample production step, and the retired waiting demo in docs history keeps
its own true record, so neither is live copy).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from app import outline_config

ROOT = Path(__file__).resolve().parents[1]
SRC = outline_config.CONFIG.read_text(encoding="utf-8")

LIVE_COPY = (
    ROOT / "templates/wait.html",
    ROOT / "config/wait.toml",
    ROOT / "config/how_it_works.toml",
    ROOT / "config/film.toml",
    ROOT / "templates/guide.html",
    ROOT / "static/ui/guide-page.js",
)


def _load(tmp_path: Path, text: str) -> dict:
    path = tmp_path / "outline.toml"
    path.write_text(text)
    return outline_config.load(path)


def test_loader_reads_the_shipped_config() -> None:
    cfg = outline_config.cached()["outline"]
    assert cfg["color"] == [232, 112, 50]
    assert cfg["width_px"] == 4
    assert cfg["min_mask_px"] == 25


@pytest.mark.parametrize(
    "old,new",
    [
        ("color = [232, 112, 50]", "color = [232, 112]"),
        ("color = [232, 112, 50]", "color = [232, 112, 300]"),
        ("width_px = 4", "width_px = 0"),
        ("width_px = 4", "width_px = true"),
        ("min_mask_px = 25", "min_mask_px = -1"),
        ("min_mask_px = 25", "min_mask = 25"),
        ("[outline]", "[outlines]"),
    ],
)
def test_loader_fails_loudly(tmp_path: Path, old: str, new: str) -> None:
    assert old in SRC
    with pytest.raises(outline_config.OutlineConfigError):
        _load(tmp_path, SRC.replace(old, new, 1))


def test_config_is_shipped_in_the_modal_image() -> None:
    deploy = tomllib.loads((ROOT / "deploy/modal_deploy.toml").read_text())
    assert "config/outline.toml" in deploy["modal"]["image_files"]


def test_live_copy_never_says_traced_or_outline_of_the_part() -> None:
    for path in LIVE_COPY:
        text = path.read_text(encoding="utf-8").lower()
        assert "traced" not in text, path.name
        assert "outline of the part" not in text, path.name
