"""Strict loader for config/redactions.toml (stdlib only): what tools/redact_sample.py hides in the public sample.

A missing, misspelled or mistyped key raises RedactionConfigError. A still's region is ``[x, y, w, h]`` or a quad
``[x1, y1, x2, y2, x3, y3, x4, y4]`` (corners in order) in pixels of the file it belongs to; a clip box is a quad that
moves in a straight line between ``[t, x1, y1, ..., x4, y4]`` keyframes.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config" / "redactions.toml"


class RedactionConfigError(ValueError):
    pass


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _rgb(v):
    return isinstance(v, list) and len(v) == 3 and all(_int(c) and 0 <= c <= 255 for c in v)


def _region(v):
    if isinstance(v, list) and len(v) == 8:
        return all(_int(c) and c >= 0 for c in v)
    return isinstance(v, list) and len(v) == 4 and all(_int(c) for c in v) and v[0] >= 0 and v[1] >= 0 and v[2] > 0 and v[3] > 0


def _check(table, spec, where):
    if not isinstance(table, dict):
        raise RedactionConfigError(f"{where}: expected a table")
    if set(table) != set(spec):
        raise RedactionConfigError(
            f"{where}: keys differ: missing {sorted(set(spec) - set(table))}, unexpected {sorted(set(table) - set(spec))}")
    for key, ok in spec.items():
        if not ok(table[key]):
            raise RedactionConfigError(f"{where}.{key}: bad value {table[key]!r}")


def _regions(v):
    return isinstance(v, list) and all(_region(r) for r in v)


def _keyframes(v):
    return (isinstance(v, list) and len(v) >= 2
            and all(isinstance(k, list) and len(k) == 9 and all(_num(c) for c in k) and k[0] >= 0 for k in v)
            and all(a[0] < b[0] for a, b in zip(v, v[1:])))


_STYLE = {
    "blur_sigma": lambda v: _num(v) and v > 0,
    "panel_rgb": _rgb,
    "caption_rgb": _rgb,
    "caption": lambda v: isinstance(v, str) and bool(v.strip()),
    "caption_width_fraction": lambda v: _num(v) and 0 < v <= 1,
    "caption_margin_px": lambda v: _int(v) and v >= 0,
    "panel_noise_tolerance": lambda v: _int(v) and 0 <= v < 255,
    "webp_quality": lambda v: _int(v) and 1 <= v <= 100,
}
def _name(v):
    """A plain relative path under its media dir: no absolute path, ``..``, ``.``, backslash or empty part."""
    if not isinstance(v, str) or not v or "\\" in v or "\0" in v:
        return False
    path = PurePosixPath(v)
    return not path.is_absolute() and str(path) == v and all(part not in ("", ".", "..") for part in path.parts)


_STILL = {"file": _name, "hide_screen": lambda v: isinstance(v, bool), "blur": _regions}
_HOW = {"file": _name, "blur": _regions}
_HIDE = {"start_s": lambda v: _num(v) and v >= 0, "end_s": _num}
_BOX = {"keyframes": _keyframes}
# A window the built clip is checked over, frame by frame: at least ``min_boxes`` boxes are live in every frame.
_CHECK = {"start_s": lambda v: _num(v) and v >= 0, "end_s": _num, "min_boxes": lambda v: _int(v) and v >= 1}


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"style", "still", "how_still", "clip_hide", "clip_box", "clip_check"}:
        raise RedactionConfigError(f"{path}: top-level tables differ: {sorted(raw)}")
    _check(raw["style"], _STYLE, "style")
    for name, spec in (("still", _STILL), ("how_still", _HOW), ("clip_hide", _HIDE), ("clip_box", _BOX), ("clip_check", _CHECK)):
        if not isinstance(raw[name], list) or not raw[name]:
            raise RedactionConfigError(f"{name}: expected a non-empty array of tables")
        for i, row in enumerate(raw[name]):
            _check(row, spec, f"{name}[{i}]")
    for table in ("clip_hide", "clip_check"):
        for i, row in enumerate(raw[table]):
            if row["end_s"] <= row["start_s"]:
                raise RedactionConfigError(f"{table}[{i}]: end_s must be after start_s")
    for name in ("still", "how_still"):
        files = [row["file"] for row in raw[name]]
        if len(set(files)) != len(files):
            raise RedactionConfigError(f"{name}: a file is listed twice")
    return raw
