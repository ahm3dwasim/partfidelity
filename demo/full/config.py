"""Strict loader for config/demo_full.toml (stdlib only): the full demo video under demo/full/.

A missing, misspelled or mistyped key raises DemoFullConfigError, as does a cut that breaks the pace rules (holds, caption
lengths, total length) or footage from 3:09 on. The caption word filter and the crossfade-overlap check are advisory lint, not
guarantees; what the video says is pinned by the exact [captions] approved list (tests/test_demo_captions.py). ``python3 -m
demo.full.config`` prints the checked config as JSON, with each scene's start time added, for demo/full/*.mjs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys
import tomllib

from app import built_config
from demo.preview.config import BANNED, _int, _num, _span, _str

CONFIG = Path(__file__).resolve().parents[2] / "config" / "demo_full.toml"
BUILT = CONFIG.with_name("built.toml")   # the one config that holds the public repository link
KINDS = ("footage", "window", "match", "card")
TOL = 1e-6
#: A stop is "capture.mark", optionally cut to a fraction of the mark: ":y0-y1" (top to bottom) and "@x0-x1" (left to right).
STOP = re.compile(r"^(\w+)\.(\w+)(?::([\d.]+)-([\d.]+))?(?:@([\d.]+)-([\d.]+))?$")


class DemoFullConfigError(ValueError):
    pass


def _pair(v):
    return isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v)


def _rect(v):
    return isinstance(v, list) and len(v) == 4 and all(_num(x) for x in v) and v[2] > 0 and v[3] > 0


def _strs(v):
    return isinstance(v, list) and bool(v) and all(_str(x) for x in v)


def _fr(v):
    return _num(v) and v <= 1


def _map(v):
    return (isinstance(v, list) and len(v) >= 2 and all(_pair(p) for p in v)
            and all(a[0] < b[0] for a, b in zip(v, v[1:])) and v[0][0] == 0)


SPEC = {
    "output": {"width": _int, "height": _int, "fps": _int, "crf": _int, "preset": _str, "max_mb": _int,
               "contact_cols": _int, "contact_rows": _int, "contact_tile_width": _int,
               "min_total_s": _num, "max_total_s": _num},
    "paths": {"scratch": _str, "ffmpeg": _str, "playwright_from": _str, "footage_zip": _str, "footage_member": _str},
    "path_env": {k: (lambda v: _str(v) and re.fullmatch(r"[A-Z][A-Z0-9_]*", v) is not None)
                 for k in ("ffmpeg", "playwright_from", "footage_zip")},
    "composition": {"footage_aspect": lambda v: _pair(v) and all(x > 0 for x in v), "open_rise_s": _num, "open_rise_px": _num,
                    "open_caption_s": _num, "badge_in_s": _num, "flight_move_frac": lambda v: _fr(v) and v > 0,
                    "slot_radius_px": _num, "from_footage_fade": _pair, "veil_pad_px": _num, "veil_label": _str,
                    "bar_inset_px": _num, "bar_bottom_px": _num, "bar_height_px": _num, "bar_radius_px": _num,
                    "dot_px": _num, "dot_rise_px": _num, "time_px": _num, "time_weight": _int, "time_bottom_px": _num,
                    "veil_px": _num, "veil_weight": _int, "veil_letter_spacing_em": _num, "veil_radius_px": _num, "panel_rise_px": _num, "row_ring_inset_px": _pair, "note_pad_px": _pair,
                    "link_anchor_inset_px": _num, "table_label": _str, "card_label": _str},
    "textcard": {"logo_icon_start_s": _num, "logo_icon_s": _num, "logo_icon_scale": _num, "logo_word_start_s": _num,
                 "logo_word_s": _num, "logo_wordmark": _str, "line_s": _num, "line_rise_px": _num, "line_px": _int,
                 "line_small_px": _int, "line_url_px": _int, "logo_icon_px": _int, "logo_word_px": _int,
                 "card_gap_px": _num, "logo_gap_px": _num, "url_weight": _int},
    "captions": {"approved": _strs},
    "source": {"site": lambda v: _str(v) and v.startswith("https://") and not v.endswith("/"),
               "desktop": lambda v: _pair(v) and all(x > 0 for x in v), "phone": lambda v: _pair(v) and all(x > 0 for x in v),
               "capture_scale": _int, "settle_ms": _int, "replay_path": lambda v: _str(v) and v.startswith("/"),
               "source_len_s": _num, "latest_s": _num, "footage_fps": _int, "footage_width": _int},
    "links": {"live_label": _str, "repo_url": lambda v: isinstance(v, str) and (v == "" or v.startswith("https://"))},
    "layout": {"margin_px": _int, "radius_px": _int, "match_scale": _num, "match_gap_px": _num, "world_gap_px": _num,
               "camera_margin": lambda v: _num(v) and v < 0.25, "max_zoom": lambda v: _num(v) and v >= 1,
               "dim": lambda v: _num(v) and v <= 1, "push_s": _num, "lead_s": _num, "hold_min_s": _num,
               "hold_max_s": _num, "drift": _num, "flight_s": _num},
    "panel": {"ring_border_px": _num, "ring_radius_px": _num, "ring_tint_pct": lambda v: _num(v) and v <= 100,
              "link_stroke_px": _num, "tag_px": _int, "tag_weight": lambda v: _int(v) and v <= 900,
              "tag_letter_spacing_em": _num, "tag_gap_px": _num, "dim_radius_px": _num},
    "caption_box": {"align": lambda v: v == "center", "tag_px": _int,
                    "tag_pad_px": _pair, "text_px": lambda v: _int(v) and v >= 34,
                    "text_lines": lambda v: _int(v) and v <= 2, "line_height": lambda v: _num(v) and 1 <= v <= 2,
                    "max_chars": lambda v: _int(v) and v <= 60, "tag_gap_px": _num,
                    "stage_gap_px": lambda v: _num(v) and v >= 48, "fade_s": lambda v: _num(v) and 0.25 <= v <= 0.35,
                    "min_hold_s": _num, "tag_font_weight": _int, "mark_px": _int},
    "audio": {"track": lambda v: isinstance(v, str), "volume": lambda v: _num(v) and v <= 2},
}
CAPTURE = {"name": _str, "path": lambda v: _str(v) and v.startswith("/"), "viewport": lambda v: v in ("desktop", "phone"),
           "ack": lambda v: isinstance(v, bool), "scroll": lambda v: isinstance(v, str),
           "marks": lambda v: isinstance(v, dict) and bool(v) and all(_str(k) and _str(s) for k, s in v.items()),
           "veil": lambda v: isinstance(v, list) and all(_str(x) for x in v)}
CAPTION = {"at": _num, "tag": _str, "text": _str}
SCENE = {
    "footage": {"map": _map, "scrub": lambda v: isinstance(v, bool)},
    "window": {"images": _strs, "stops": lambda v: isinstance(v, list) and bool(v) and all(_str(s) or _rect(s) for s in v),
               "from_footage": lambda v: isinstance(v, bool)},
    "match": {k: _span for k in ("table_in", "ring", "link", "card", "note")},
    "card": {"lines": _strs, "logo": lambda v: isinstance(v, bool), "reveal_start_s": _num, "reveal_gap_s": _num},
}
COMMON = {"id": _str, "kind": lambda v: v in KINDS, "dur": lambda v: _num(v) and v > 0, "caption": lambda v: isinstance(v, list) and bool(v)}


def _keys(where, table, spec):
    if not isinstance(table, dict) or set(table) != set(spec):
        got = set(table) if isinstance(table, dict) else set()
        raise DemoFullConfigError(f"{where}: keys differ (missing {sorted(set(spec) - got)}, unknown {sorted(got - set(spec))})")
    for key, ok in spec.items():
        if not ok(table[key]):
            raise DemoFullConfigError(f"{where}.{key}: bad value {table[key]!r}")


def _check_scene(i, sc, cfg, caps, footage_before):
    where = f"scene[{i}]"
    if not isinstance(sc, dict) or sc.get("kind") not in KINDS:
        raise DemoFullConfigError(f"{where}: kind must be one of {KINDS}")
    _keys(where, sc, {**COMMON, **SCENE[sc["kind"]]})
    L, B, kind = cfg["layout"], cfg["caption_box"], sc["kind"]
    ids = [c["id"] for c in cfg["scene"][:i]]
    if sc["id"] in ids:
        raise DemoFullConfigError(f"{where}.id: duplicate {sc['id']!r}")
    for j, c in enumerate(sc["caption"]):
        _keys(f"{where}.caption[{j}]", c, CAPTION)
        if len(c["text"]) > B["text_lines"] * B["max_chars"]:
            raise DemoFullConfigError(f"{where}.caption[{j}]: text over {B['text_lines']} lines of {B['max_chars']} characters")
        for what in ("text", "tag"):
            if BANNED.search(c[what]):
                raise DemoFullConfigError(f"{where}.caption[{j}].{what}: claims {BANNED.search(c[what]).group(0)!r}")
    ats = [c["at"] for c in sc["caption"]]
    if ats != sorted(ats) or len(set(ats)) != len(ats) or any(a >= sc["dur"] for a in ats):
        raise DemoFullConfigError(f"{where}: caption times must rise and fall inside the scene")
    matched = kind == "window" and sc["from_footage"]
    if (i == 0 and (kind != "footage" or ats[0] != 0)) or (ats[0] != 0 and not matched):
        raise DemoFullConfigError(f"{where}: the first caption starts at 0 (a match-cut scene may start it later); the film opens on footage")
    if kind == "footage":
        f = cfg["source"]
        if any(not 0 <= s <= f["latest_s"] or s > f["source_len_s"] for _, s in sc["map"]):
            raise DemoFullConfigError(f"{where}.map: footage must come from before {f['latest_s']} s")
        if sc["map"][-1][0] > sc["dur"]:
            raise DemoFullConfigError(f"{where}.map: runs past the scene")
    elif kind == "window":
        for name in sc["images"]:
            if name not in caps:
                raise DemoFullConfigError(f"{where}.images: no capture {name!r}")
        for s in sc["stops"]:
            if isinstance(s, str):
                m = STOP.match(s)
                if not m or m[1] not in sc["images"] or m[2] not in caps[m[1]]["marks"]:
                    raise DemoFullConfigError(f"{where}.stops: {s!r} is not a mark of this scene's images")
                for a, b in ((m[3], m[4]), (m[5], m[6])):
                    if a is not None and not 0 <= float(a) < float(b) <= 1:
                        raise DemoFullConfigError(f"{where}.stops: {s!r} has a bad fraction")
        n = len(sc["stops"])
        hold = (sc["dur"] - L["lead_s"]) / n - L["push_s"]
        if not L["hold_min_s"] - TOL <= hold <= L["hold_max_s"] + TOL:
            raise DemoFullConfigError(f"{where}.dur: each stop holds {hold:.2f} s, outside {L['hold_min_s']}-{L['hold_max_s']} s")
        step = L["push_s"] + hold
        pushes = [(L["lead_s"] + k * step, L["lead_s"] + k * step + L["push_s"]) for k in range(n)]
        for a in ats[1:]:
            if any(p0 - TOL <= a <= p1 + TOL for p0, p1 in pushes):
                raise DemoFullConfigError(f"{where}: a caption cut lands inside a camera move")
        if sc["from_footage"]:
            if not footage_before or len(sc["images"]) != 1 or "frame" not in caps[sc["images"][0]]["marks"]:
                raise DemoFullConfigError(f"{where}.from_footage: needs a footage scene before it and one image with a 'frame' mark")
            if L["flight_s"] > L["lead_s"]:
                raise DemoFullConfigError(f"{where}.from_footage: flight_s must fit inside lead_s")
    elif kind == "match":
        for k in SCENE["match"]:
            if sc[k][1] > sc["dur"]:
                raise DemoFullConfigError(f"{where}.{k}: runs past the scene")
        if sc["dur"] - max(sc[k][1] for k in SCENE["match"]) > L["hold_max_s"] + TOL:
            raise DemoFullConfigError(f"{where}: the last move leaves an idle hold over hold_max_s")
    else:
        for line in sc["lines"]:
            if BANNED.search(line):
                raise DemoFullConfigError(f"{where}.lines: claims {BANNED.search(line).group(0)!r}")


def card_lines(cfg, sc):
    """The text lines a card scene shows: a line with {repo} is left out only if links.repo_url is empty after load() (it never is: load fills it)."""
    return [s for s in sc["lines"] if not ("{repo}" in s and not cfg["links"]["repo_url"])]


def _motions(cfg):
    """Every timed move of the film as (label, start, end) in absolute seconds: the opening card's rise, camera pushes, the
    footage card's flight and dissolve into the guide, the match scene's moves and the text cards' logo and line reveals.
    Advisory lint for timeline overlaps, not a guarantee: a motion that is not listed here is not checked."""
    L, C, T, t, out = cfg["layout"], cfg["composition"], cfg["textcard"], 0.0, []
    for i, sc in enumerate(cfg["scene"]):
        if sc["kind"] == "footage" and i == 0:
            out.append((f"{sc['id']} opening rise", t, t + C["open_rise_s"]))
        if sc["kind"] == "window":
            hold = (sc["dur"] - L["lead_s"]) / len(sc["stops"]) - L["push_s"]
            out += [(f"{sc['id']} camera push", t + L["lead_s"] + k * (L["push_s"] + hold),
                     t + L["lead_s"] + k * (L["push_s"] + hold) + L["push_s"]) for k in range(len(sc["stops"]))]
            if sc["from_footage"]:
                out.append((f"{sc['id']} card flight and dissolve", t, t + L["flight_s"]))
        elif sc["kind"] == "match":
            out += [(f"{sc['id']} {k}", t + sc[k][0], t + sc[k][1]) for k in SCENE["match"]]
        elif sc["kind"] == "card":
            if sc["logo"]:
                out += [(f"{sc['id']} logo icon reveal", t + T["logo_icon_start_s"], t + T["logo_icon_start_s"] + T["logo_icon_s"]),
                        (f"{sc['id']} logo wordmark reveal", t + T["logo_word_start_s"], t + T["logo_word_start_s"] + T["logo_word_s"])]
            out += [(f"{sc['id']} line {k} reveal", t + sc["reveal_start_s"] + k * sc["reveal_gap_s"],
                     t + sc["reveal_start_s"] + k * sc["reveal_gap_s"] + T["line_s"]) for k in range(len(card_lines(cfg, sc)))]
        t += sc["dur"]
    return out


def _check_cuts(cfg):
    """No caption crossfade (the cut and the scene change with it) overlaps an active move."""
    fade, t, cuts = cfg["caption_box"]["fade_s"], 0.0, []
    for i, sc in enumerate(cfg["scene"]):
        cuts += [(f"{sc['id']} caption {k}", t + c["at"]) for k, c in enumerate(sc["caption"]) if (i, k) != (0, 0)]
        if i and not (sc["kind"] == "window" and sc["from_footage"]):   # a match cut is the flight itself, not a crossfade
            cuts.append((f"{sc['id']} scene start", t))
        t += sc["dur"]
    for name, cut in cuts:
        for label, a0, a1 in _motions(cfg):
            if cut < a1 - TOL and cut + fade > a0 + TOL:
                raise DemoFullConfigError(f"{name}: the crossfade ({cut:.2f}-{cut + fade:.2f} s) overlaps the {label} ({a0:.2f}-{a1:.2f} s)")


def validate(cfg: dict) -> dict:
    top = set(SPEC) | {"capture", "scene"}
    if not isinstance(cfg, dict) or set(cfg) != top:
        got = set(cfg) if isinstance(cfg, dict) else set()
        raise DemoFullConfigError(f"top level keys differ (missing {sorted(top - got)}, unknown {sorted(got - top)})")
    for name, spec in SPEC.items():
        _keys(f"[{name}]", cfg[name], spec)
    if cfg["source"]["latest_s"] > 189.0:  # config-ok: the owner's fixed 3:09 footage cut-off is a safety floor, not a tunable
        raise DemoFullConfigError("source.latest_s: the clip is used only from before 3:09 (189 s)")
    if not isinstance(cfg["capture"], list) or not isinstance(cfg["scene"], list) or not cfg["scene"]:
        raise DemoFullConfigError("capture and scene must be lists of tables")
    caps = {}
    for j, c in enumerate(cfg["capture"]):
        if isinstance(c, dict):
            c.setdefault("veil", [])
        _keys(f"capture[{j}]", c, CAPTURE)
        if not set(c["veil"]) <= set(c["marks"]):
            raise DemoFullConfigError(f"capture[{j}].veil: every veiled name must be a mark")
        if c["name"] in caps:
            raise DemoFullConfigError(f"capture[{j}].name: duplicate {c['name']!r}")
        caps[c["name"]] = c
    for i, sc in enumerate(cfg["scene"]):
        _check_scene(i, sc, cfg, caps, i > 0 and cfg["scene"][i - 1].get("kind") == "footage")
    starts, t = [], 0.0
    for sc in cfg["scene"]:
        starts += [t + c["at"] for c in sc["caption"]]
        t += sc["dur"]
    minh = cfg["caption_box"]["min_hold_s"]
    if any(b - a < minh - TOL for a, b in zip(starts, [*starts[1:], t])):
        raise DemoFullConfigError("a caption stays shorter than min_hold_s across a scene cut")
    _check_cuts(cfg)
    out = cfg["output"]
    total = sum(sc["dur"] for sc in cfg["scene"])
    if not out["min_total_s"] <= total <= out["max_total_s"] or not 135 <= out["min_total_s"] <= out["max_total_s"] <= 165:
        raise DemoFullConfigError(f"scenes total {total:.1f} s: the cut runs 135-165 s")
    if cfg["audio"]["track"] and not Path(cfg["audio"]["track"]).expanduser().is_file():
        raise DemoFullConfigError(f"audio.track: no file {cfg['audio']['track']!r}")
    return cfg


def _built_repo_url() -> str:
    try:   # the canonical validator: HTTPS, a host, no user or password
        url = built_config.load(BUILT)["page"]["repository_url"]
    except (OSError, ValueError, KeyError) as exc:
        raise DemoFullConfigError(f"{BUILT.name}: [page].repository_url is not a valid public https:// link ({exc})") from exc
    if not url:
        raise DemoFullConfigError(f"{BUILT.name}: [page].repository_url is empty (the end card's repo line needs it)")
    return url


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        cfg = validate(tomllib.load(fh))
    if not cfg["links"]["repo_url"]:   # empty means "take it from config/built.toml [page].repository_url"
        cfg["links"]["repo_url"] = _built_repo_url()
    t = 0.0
    for sc in cfg["scene"]:
        sc["start"], t = t, t + sc["dur"]
    cfg["total_s"] = t
    return cfg


if __name__ == "__main__":
    json.dump(load(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG), sys.stdout, ensure_ascii=False)
