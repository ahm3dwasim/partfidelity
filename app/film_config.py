"""Strict loader for config/film.toml (stdlib only): the home-page scroll film.

A missing, misspelled or mistyped key raises FilmConfigError. ``public`` is the dict the page script reads.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config" / "film.toml"


class FilmConfigError(ValueError):
    pass


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str(v):
    return isinstance(v, str) and bool(v)


#: The frame-name grammar static/ui/film.js understands: one {n} or zero-padded {n:0Kd}, path-safe text around it.
FRAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]*\{n(?::0[1-9]d)?\}[A-Za-z0-9._-]*$")
#: Bundle file names: one {i} (the bundle index), path-safe text around it.
BUNDLE_NAME = re.compile(r"^[A-Za-z0-9._-]*\{i\}[A-Za-z0-9._-]*$")
_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
#: A bundle record's frame number is a uint16 (tools/film_frames.py, static/ui/film.js).
MAX_FRAMES = 0xFFFF
#: Stage compositions static/site.css knows.
LAYOUTS = ("phone", "wide")
#: Where static/ui/film.js puts the part labels: all right of the stack, or right and left in turn.
LABEL_SIDES = ("right", "alternate")


def _frame_pair(v):
    return isinstance(v, list) and len(v) == 2 and all(_int(i) and i >= 1 for i in v) and v[0] < v[1]


def _check(table, spec, where):
    if not isinstance(table, dict):
        raise FilmConfigError(f"{where}: expected a table")
    if set(table) != set(spec):
        raise FilmConfigError(
            f"{where}: keys differ (missing {sorted(set(spec) - set(table))}, unknown {sorted(set(table) - set(spec))})"
        )
    for key, ok in spec.items():
        if not ok(table[key]):
            raise FilmConfigError(f"{where}.{key}: bad value {table[key]!r}")


def contained(rel: str, base: str) -> bool:
    """A normalized relative path strictly beneath `base` (no absolute path, '..', '.', empty part or double slash)."""
    parts = rel.split("/")
    return (not rel.startswith("/") and parts[0] == base and len(parts) > 1
            and all(p and p not in (".", "..") and "\\" not in p and "\0" not in p for p in parts))


def _rows(cfg, name, spec):
    rows = cfg.get(name)
    if not isinstance(rows, list) or not rows:
        raise FilmConfigError(f"{name}: expected a non-empty list of tables")
    for i, row in enumerate(rows):
        _check(row, spec, f"{name}[{i}]")
    return rows


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    if set(cfg) != {"film", "archives", "sets", "chapters", "parts"}:
        raise FilmConfigError(f"top level keys differ: {sorted(cfg)}")
    _check(
        cfg["film"],
        {
            "frames": lambda v: _int(v) and 1 < v <= MAX_FRAMES,
            "height_vh": lambda v: _int(v) and v > 100,
            "smoothing": lambda v: _num(v) and 0 < v <= 1,
            "media_url": _str,
            "version": _str,
            "poster": _str,
            "frame_pattern": lambda v: _str(v) and bool(FRAME_PATTERN.match(v)),
            "passes": lambda v: isinstance(v, list) and bool(v) and all(_int(i) and i >= 1 for i in v) and v[-1] == 1
            and all(a > b for a, b in zip(v, v[1:])),
            "bundle_frames": lambda v: _int(v) and v >= 1,
            "bundle_name": lambda v: _str(v) and bool(BUNDLE_NAME.match(v)),
            "parallel_loads": lambda v: _int(v) and v >= 1,
            "load_retries": lambda v: _int(v) and 0 <= v <= 10,
            "load_backoff_ms": lambda v: _int(v) and 0 <= v <= 60000,
            "bitmap_window": lambda v: _int(v) and v >= 0,
            "max_pending_decodes": lambda v: _int(v) and v >= 1,
            "max_frame_bytes": lambda v: _int(v) and 12 <= v < 2**32,
            "opening_fade": _frame_pair,
            "caption_fade_ms": lambda v: _int(v) and 0 <= v <= 2000,
            "anchors": lambda v: _str(v) and bool(_SAFE.match(v)),
            "label_land": lambda v: _num(v) and 0 < v <= 1,
            "label_gap_px": lambda v: _num(v) and v >= 0,
            "label_spacing_px": lambda v: _num(v) and v > 0,
            "label_edge_px": lambda v: _num(v) and v >= 0,
            "cache_max_age_s": lambda v: _int(v) and v >= 0,
            "static_cache_control": _str,
            "phone_art_width_vw": lambda v: _num(v) and 50 <= v <= 200,
            "phone_feather_y_pct": lambda v: _num(v) and 0 <= v < 50,
            "part_rise": lambda v: _int(v) and v >= 1,
            "opening_lift_px": lambda v: _num(v) and v >= 0,
            "sample_cta": _str,
            "make_cta": _str,
            "teaser_steps": lambda v: _int(v) and v >= 1,
            "toast_ms": lambda v: _int(v) and v > 0,
            "observer_top_pct": lambda v: _int(v) and 0 <= v < 100,
            "observer_bottom_pct": lambda v: _int(v) and 0 <= v < 100,
            "observer_hold_ms": lambda v: _int(v) and v >= 0,
        },
        "film",
    )
    roots = {"source_root": "assets", "target_root": "static"}      # config-ok: the approved repo roots are a safety limit
    _check(
        cfg["archives"],
        {
            "source": lambda v: _str(v) and "{dir}" in v and contained(v, roots["source_root"]),
            "anchors": lambda v: _str(v) and contained(v, roots["source_root"]),
            "target": lambda v: _str(v) and contained(v, roots["target_root"]),
        },
        "archives",
    )
    sets = _rows(
        cfg,
        "sets",
        {
            "name": _str,
            "dir": lambda v: _str(v) and bool(_SAFE.match(v)),
            "max_css_width": lambda v: _int(v) and v >= 0,
            "max_device_width": lambda v: _int(v) and v >= 0,
            "layout": lambda v: v in LAYOUTS,
            "labels": lambda v: v in LABEL_SIDES,
        },
    )
    chapters = _rows(
        cfg,
        "chapters",
        {
            "start": lambda v: _int(v) and v >= 1,
            "end": lambda v: _int(v) and v >= 1,
            "name": _str,
            "title": _str,
            "body": _str,
            "cta_text": lambda v: isinstance(v, str),
            "cta_url": lambda v: isinstance(v, str) and (not v or v.startswith("/")),
        },
    )
    parts = _rows(
        cfg,
        "parts",
        {
            "frame": lambda v: _int(v) and v >= 1,
            "id": lambda v: _str(v) and bool(_SAFE.match(v)),
            "name": _str,
            "chapter": lambda v: _int(v) and v >= 1,
            "leave": lambda v: _int(v) and v >= 1,
        },
    )
    film = cfg["film"]
    if film["observer_top_pct"] + film["observer_bottom_pct"] >= 100:
        raise FilmConfigError("film: observer margins must leave a reading band")
    frames = film["frames"]
    if film["opening_fade"][1] > frames:
        raise FilmConfigError("film: opening_fade must lie inside 1..frames")
    if film["passes"][0] >= frames:
        raise FilmConfigError("film: the first pass must step below frames")
    if len({p["id"] for p in parts}) != len(parts):
        raise FilmConfigError("parts: duplicate id")
    for i, c in enumerate(chapters):
        if not c["start"] < c["end"] <= frames:
            raise FilmConfigError(f"chapters[{i}]: need start < end <= frames")
        if bool(c["cta_text"]) != bool(c["cta_url"]):
            raise FilmConfigError(f"chapters[{i}]: cta_text and cta_url go together")
    # A caption is on screen from the end of the opening copy to the last frame: no gap, no overlap.
    if 2 * chapters[0]["start"] > sum(film["opening_fade"]):
        raise FilmConfigError("chapters[0]: must start by the middle of opening_fade (no frame without readable copy)")
    if chapters[-1]["end"] != frames or any(a["end"] != b["start"] for a, b in zip(chapters, chapters[1:])):
        raise FilmConfigError("chapters: each must end where the next starts, the last at frames")
    for i, p in enumerate(parts):
        if p["frame"] > frames:
            raise FilmConfigError(f"parts[{i}]: frame beyond frames")
        if not p["frame"] + film["label_land"] * film["part_rise"] < p["leave"] <= frames:
            raise FilmConfigError(f"parts[{i}]: leave must come after the label shows and within frames")
        if p["chapter"] > len(chapters):
            raise FilmConfigError(f"parts[{i}]: no chapter {p['chapter']}")
    if [p["chapter"] for p in parts] != sorted(p["chapter"] for p in parts):
        raise FilmConfigError("parts must be in chapter order")
    if [p["frame"] for p in parts] != sorted(p["frame"] for p in parts):
        raise FilmConfigError("parts must be in frame order")
    # Captions change with the stage: chapter n > 1 starts on the frame its first part starts to rise.
    for i, c in enumerate(chapters):
        first = [p["frame"] for p in parts if p["chapter"] == i + 1]
        if not first:
            raise FilmConfigError(f"chapters[{i}]: no parts")
        if (c["start"] != first[0]) if i else (c["start"] > first[0]):
            raise FilmConfigError(f"chapters[{i}]: must start on the frame its first part rises ({first[0]})")
    if sets[-1]["max_css_width"] or sets[-1]["max_device_width"]:
        raise FilmConfigError("sets: the last set is the fallback (limits 0)")
    limited = sets[:-1]
    if any(not (s["max_css_width"] and s["max_device_width"]) for s in limited):
        raise FilmConfigError("sets: every set but the last needs both limits")
    for key in ("max_css_width", "max_device_width"):
        if any(a[key] >= b[key] for a, b in zip(limited, limited[1:])):
            raise FilmConfigError(f"sets: {key} must rise set by set")
    for key in ("name", "dir"):
        if len({s[key] for s in sets}) != len(sets):
            raise FilmConfigError(f"sets: duplicate {key}")
    return cfg


def bundles(film: dict) -> list[tuple[str, list[int]]]:
    """(file name, frame numbers in load order) per bundle: the passes in order, each pass cut into bundles of at
    most bundle_frames, so the coarse pass is complete before any later frame arrives. The last frame joins the first
    pass. tools/film_frames.py writes these files; film.js fetches them in this order."""
    frames, seen, order = film["frames"], set(), []
    for step in film["passes"]:
        batch = [n for n in range(1, frames + 1, step) if n not in seen]
        if frames not in seen and frames not in batch:
            batch.append(frames)
        seen.update(batch)
        size = film["bundle_frames"]
        order += [batch[k:k + size] for k in range(0, len(batch), size)]
    return [(film["bundle_name"].format(i=i), nums) for i, nums in enumerate(order)]


@lru_cache(maxsize=1)
def cached() -> dict:
    """The shipped config, validated once per process (the page and every frame response read it)."""
    return load()


def public(cfg: dict) -> dict:
    """What the page script reads: URLs resolved, everything else as configured."""
    film = cfg["film"]
    base = film["media_url"].rstrip("/")
    return {
        **film,
        "poster_url": f"{base}/{film['poster']}?v={film['version']}",
        "base_url": base,
        "sets": cfg["sets"],
        "bundles": [name for name, _ in bundles(film)],
        "chapters": [{"start": c["start"], "end": c["end"]} for c in cfg["chapters"]],
        "parts": cfg["parts"],
    }
