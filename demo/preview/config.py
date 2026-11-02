"""Strict loader for config/demo_preview.toml (stdlib only): the demo motion preview under demo/preview/. Render-time tooling, not runtime: it lives outside app/ so it is not shipped.

A missing, misspelled or mistyped key raises DemoPreviewConfigError. ``python3 -m demo.preview.config``
prints the checked config as JSON for demo/preview/render.mjs and capture.mjs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tomllib

CONFIG = Path(__file__).resolve().parents[2] / "config" / "demo_preview.toml"
TOL = 1e-6
CAPTION_KEYS = ("footage", "footage_tag", "guide", "guide_tag", "match", "match_tag", "recorded_tag", "replay_tag")
BEATS = ("footage", "to_guide", "guide", "to_match", "match", "badge")
#: Truth rule (spec "Truth and privacy"). ADVISORY LINT, not a guarantee: it catches common phrasings at render time. The real
#: guard is the exact approved caption list in each config ([captions] approved), pinned by tests/test_demo_captions.py.
#: config-ok: a pinned safety check.
BANNED = re.compile(
    r"\b(?:verif|saf|accur)\w*"                                                  # stems on word starts
    r"|\bcomplete\w*\s+repair"                                                  # "complete repair"
    r"|\b(?:repair|fix|restoration)(?:ed|es|s)?\s+(?:(?:is|are|was|were|has|have|been|now|and|all|just|already|finally|\w+ly)\s+)*complete\w*"  # "repair complete", "the fix has been completed"
    r"|\b(?:fully|completely|totally)\s+(?:repair|fix|restor)\w*", re.I)         # "fully repaired"


class DemoPreviewConfigError(ValueError):
    pass


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _str(v):
    return isinstance(v, str) and bool(v.strip())


def _span(v):
    return isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v) and v[0] < v[1]


def _frac(v):
    return _num(v) and v <= 1


def _fspan(v):
    return _span(v) and v[1] <= 1


def _pair(v):
    return isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v)


SPEC = {
    "output": {"width": _int, "height": _int, "fps": _int, "crf": _int, "preset": _str, "max_mb": _int,
               "contact_tiles": _int, "contact_tile_width": _int},
    "paths": {"scratch": _str, "ffmpeg": _str, "playwright_from": _str, "footage_zip": _str, "footage_member": _str},
    "path_env": {k: (lambda v: _str(v) and re.fullmatch(r"[A-Z][A-Z0-9_]*", v) is not None)
                 for k in ("ffmpeg", "playwright_from", "footage_zip")},
    "source": {"site": lambda v: _str(v) and v.startswith("https://") and not v.endswith("/"),
               "guide_path": lambda v: _str(v) and v.startswith("/"),
               "replay_path": lambda v: _str(v) and v.startswith("/"),
               "guide_step": _int, "capture_width": _int, "capture_height": _int, "capture_scale": _int,
               "settle_ms": _int},
    "footage": {"start_s": _num, "end_s": _num, "latest_s": _num},
    "timing": {"duration_s": _num, **{b: _span for b in BEATS}},
    "layout": {"margin_px": _int, "radius_px": _int, "match_scale": _num, "connector_gap_px": _num,
               "card_aspect": lambda v: isinstance(v, list) and len(v) == 2 and all(_int(x) for x in v)},
    "ring": {"border_px": _num, "radius_px": _num, "tint_pct": lambda v: _num(v) and v <= 100, "row_inset_px": _pair,
             "note_pad_px": _pair, "anchor_inset_px": _num, "stroke_px": _num},
    "tag": {"size_px": _int, "weight": lambda v: _int(v) and v <= 900, "letter_spacing_em": _num, "gap_px": _num},
    "motion": {"footage_rise": _frac, "footage_rise_px": _num, "footage_play": _frac, "caption_in_s": _num,
               "badge_in": _fspan, "fly": _fspan, "guide_in": _fspan, "dissolve": _fspan, "slot_radius_px": _num,
               "slide_out": _fspan, "panel_rise_px": _num, "table_in_start": _frac, "row_ring": _fspan,
               "link": _fspan, "card_in": _fspan, "note_ring": _fspan, "badge_hold": _fspan,
               "badge_hold_alpha": _frac, "badge_hold_scale": _num, "dim_radius_px": _num, "scrim_spread_px": _int},
    "camera": {"target": lambda v: isinstance(v, list) and len(v) == 4 and all(_num(x) for x in v) and v[2] > 0 and v[3] > 0,
               "margin_frac": lambda v: _num(v) and v < 0.25, "push_s": _span, "hold_min_s": _num,
               "dim": lambda v: _num(v) and v <= 1},
    "caption_box": {"align": lambda v: v in ("center", "left"), "tag_px": _int,
                    "tag_pad_px": lambda v: isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v),
                    "text_px": lambda v: _int(v) and v >= 34, "text_lines": lambda v: _int(v) and v <= 2,
                    "line_height": lambda v: _num(v) and 1 <= v <= 2, "max_chars": lambda v: _int(v) and v <= 60,
                    "tag_gap_px": _num, "stage_gap_px": lambda v: _num(v) and v >= 48,
                    "fade_s": lambda v: _num(v) and 0.25 <= v <= 0.35,
                    "cuts_s": lambda v: isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v) and v[0] < v[1],
                    "min_hold_s": _num, "mark_px": _int},
    "captions": {**{k: _str for k in CAPTION_KEYS}, "approved": lambda v: isinstance(v, list) and bool(v) and all(_str(x) for x in v)},
}


def lines(text: str, width: int) -> int:
    """Lines the text takes wrapped at ``width`` characters on word breaks (an over-long word is split)."""
    n, cur = 1, 0
    for word in text.split():
        while len(word) > width:
            n, cur, word = n + (cur > 0), 0, word[width:]
            n += 1
        if cur and cur + 1 + len(word) > width:
            n, cur = n + 1, 0
        cur += len(word) + (1 if cur else 0)
    return n


def validate(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or set(cfg) != set(SPEC):
        got = set(cfg) if isinstance(cfg, dict) else set()
        raise DemoPreviewConfigError(f"top level keys differ (missing {sorted(set(SPEC) - got)}, unknown {sorted(got - set(SPEC))})")
    for name, spec in SPEC.items():
        table = cfg[name]
        if not isinstance(table, dict) or set(table) != set(spec):
            got = set(table) if isinstance(table, dict) else set()
            raise DemoPreviewConfigError(f"[{name}]: keys differ (missing {sorted(set(spec) - got)}, unknown {sorted(got - set(spec))})")
        for key, ok in spec.items():
            if not ok(table[key]):
                raise DemoPreviewConfigError(f"{name}.{key}: bad value {table[key]!r}")
    f, t = cfg["footage"], cfg["timing"]
    if not f["start_s"] < f["end_s"] <= f["latest_s"]:
        raise DemoPreviewConfigError("footage: need start_s < end_s <= latest_s")
    if t[BEATS[0]][0] != 0 or t[BEATS[-1]][1] != t["duration_s"]:
        raise DemoPreviewConfigError("timing: beats must start at 0 and end at duration_s")
    for a, b in zip(BEATS, BEATS[1:]):
        if t[a][1] != t[b][0]:
            raise DemoPreviewConfigError(f"timing: {a} must end where {b} starts")
    if not 10 <= t["duration_s"] <= 15:
        raise DemoPreviewConfigError("timing.duration_s: the preview runs 10-15 s")
    c = cfg["camera"]
    if not (t["guide"][0] <= c["push_s"][0] and c["push_s"][1] <= t["guide"][1]):
        raise DemoPreviewConfigError("camera.push_s: must sit inside the guide beat")
    if t["guide"][1] - c["push_s"][1] < c["hold_min_s"]:
        raise DemoPreviewConfigError("camera: the framed hold after push_s is shorter than hold_min_s")
    b = cfg["caption_box"]
    # Each caption fades in over fade_s at its start and is gone by the next cut: it holds at least min_hold_s.
    starts = [0.0, *b["cuts_s"]]
    ends = [*b["cuts_s"], t["duration_s"]]
    if not all(0 <= s0 and e - s0 >= b["min_hold_s"] for s0, e in zip(starts, ends)):
        raise DemoPreviewConfigError("caption_box.cuts_s: a caption would stay shorter than min_hold_s")
    if b["cuts_s"][0] < t["to_guide"][0] or b["cuts_s"][1] + b["fade_s"] > t["match"][1]:
        raise DemoPreviewConfigError("caption_box.cuts_s: cuts must fall inside the film's moves")
    # A caption never changes mid-move: each cut starts after the move before it ends and its fade ends before the next begins.
    m = cfg["motion"]
    at = lambda beat, f: t[beat][0] + f * (t[beat][1] - t[beat][0])
    c0, c1, fade = *b["cuts_s"], b["fade_s"]
    if c0 < at("to_guide", m["fly"][1]):
        raise DemoPreviewConfigError("caption_box.cuts_s[0]: the cut lands before the card has finished flying into the guide")
    if c0 + fade > c["push_s"][0]:
        raise DemoPreviewConfigError("caption_box.cuts_s[0]: the crossfade runs into the camera move")
    if c1 < max(c["push_s"][1], at("to_match", m["slide_out"][1])):
        raise DemoPreviewConfigError("caption_box.cuts_s[1]: the cut lands before the camera or the slide-out has finished")
    # No crossfade overlaps any active motion: the dissolve, the table's entrance and every other move of the film.
    motions = {
        "card fly": (at("to_guide", m["fly"][0]), at("to_guide", m["fly"][1])),
        "guide fade-in": (at("to_guide", m["guide_in"][0]), at("to_guide", m["guide_in"][1])),
        "dissolve": (at("to_guide", m["dissolve"][0]), at("to_guide", m["dissolve"][1])),
        "camera push": tuple(c["push_s"]),
        "slide-out": (at("to_match", m["slide_out"][0]), at("to_match", m["slide_out"][1])),
        "table entrance": (at("to_match", m["table_in_start"]), at("match", m["row_ring"][0])),
        "row ring": (at("match", m["row_ring"][0]), at("match", m["row_ring"][1])),
        "link": (at("match", m["link"][0]), at("match", m["link"][1])),
        "card entrance": (at("match", m["card_in"][0]), at("match", m["card_in"][1])),
        "note ring": (at("match", m["note_ring"][0]), at("match", m["note_ring"][1])),
    }
    for i, cut in enumerate(b["cuts_s"]):
        for name, (a0, a1) in motions.items():
            if cut < a1 - TOL and cut + fade > a0 + TOL:
                raise DemoPreviewConfigError(f"caption_box.cuts_s[{i}]: the crossfade ({cut}-{cut + fade:.2f} s) overlaps the {name} ({a0:.2f}-{a1:.2f} s)")
    for key in CAPTION_KEYS:
        text = cfg["captions"][key]
        if lines(text, b["max_chars"]) > b["text_lines"]:
            raise DemoPreviewConfigError(f"captions.{key}: needs more than {b['text_lines']} lines of {b['max_chars']} characters")
        if BANNED.search(text):
            raise DemoPreviewConfigError(f"captions.{key}: claims {BANNED.search(text).group(0)!r}")
    return cfg


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        return validate(tomllib.load(fh))


if __name__ == "__main__":
    json.dump(load(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG), sys.stdout, ensure_ascii=False)
