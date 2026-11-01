"""Strict loader for config/teardown.toml (stdlib only).

Imported by tools/teardown_render.py and the tests; the Blender scripts receive the validated dict as JSON.
Missing or unknown keys at any level raise ConfigError; list-of-table sections are checked per row.
"""
from __future__ import annotations

import os
import tomllib

CONFIG_REL = os.path.join("config", "teardown.toml")


class ConfigError(ValueError):
    pass


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pos(v):
    return _num(v) and v > 0


def _s(v):
    return isinstance(v, str) and bool(v)


def _slist(v):
    return isinstance(v, list) and all(_s(i) for i in v)


def _vec3(v):
    return isinstance(v, list) and len(v) == 3 and all(_num(i) for i in v)


def _wh(v):
    return isinstance(v, list) and len(v) == 2 and all(isinstance(i, int) and not isinstance(i, bool) and i > 0 for i in v)


def _int1(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _pts(v):
    return isinstance(v, list) and bool(v) and all(isinstance(i, list) and len(i) == 2 and all(_num(j) for j in i) for i in v)


def _vec2(v):
    return isinstance(v, list) and len(v) == 2 and all(_num(i) for i in v)


def _rgb(v):
    return isinstance(v, list) and len(v) == 3 and all(isinstance(i, int) and 0 <= i <= 255 for i in v)


def _vec3p(v):
    return _vec3(v) and all(i > 0 for i in v)


def _unit(v):
    return _num(v) and 0 <= v <= 1


def _nat(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _lin3(v):
    return _vec3(v) and all(0 <= i <= 1 for i in v)


SPEC = {
    "blender": {"download_url": _s, "build": _s, "cuda_image": _s, "python": _s, "apt_packages": _slist, "pip_packages": _slist},
    "modal": {"app": _s, "gpu": _s, "gpu_timeout_s": _int1, "cpu_timeout_s": _int1, "cpu_count": _int1},
    "budget": {"project": _s, "job": _s, "cap_usd": _pos, "gpu_usd_per_s": _pos, "cpu_usd_per_s": _pos, "overhead_s": _num, "ledger": _s, "check_hook_env": _s},
    "render": {"preview_res": _wh, "still_res": _wh, "film_res": _wh, "phone_width": _int1, "preview_samples": _int1,
               "still_samples": _int1, "film_samples": _int1, "seed": _int1, "denoiser": _s, "adaptive_threshold": _pos,
               "fan_deg_per_frame": _num, "exposure": _num, "look": _s, "texture_ppm": _pos, "n_frames": _int1, "fps": _int1,
               "local_out": _s, "stills_dir": _s, "film_chunk": _int1, "webp_quality": _int1, "film_dir": _s, "film_crf": _int1},
    "scene": {"floor_z": _num, "sweep_radius": _pos, "wall_y": _pos, "bevel_mm": _pos, "bevel_segments": _int1,
              "lid_open_deg": _num, "pivot_yz": lambda v: isinstance(v, list) and len(v) == 2 and all(_num(i) for i in v),
              "deck_bevel": lambda v: isinstance(v, list) and len(v) == 2 and _pos(v[0]) and _int1(v[1]), "corner_mm": _pos, "key_jitter_mm": _num, "key_dish_mm": _num, "wear": _num, "mid_lift": _vec3, "glass_rough": _pos, "glass_spec": _pos},
    "camera": {"sensor_mm": _pos, "clip_start_m": _pos, "clip_end_m": _pos},
    "choreo": {"close_start": _int1, "close_frames": _int1, "rise_start": _int1, "rise_stagger": _num, "rise_frames": _int1},
    "reassemble": {"start": _int1, "stagger": _int1, "frames": _int1},
    "screws": {"lead": _nat, "frames": _int1, "lift_mm": _num, "spread": _unit},
    "float": {"amp_mm": lambda v: _num(v) and 0 <= v < 1, "period": _pos, "phase": _num},
    "dust": {"count": _nat, "radius_mm": _pos, "box": lambda v: isinstance(v, list) and len(v) == 3 and all(_vec2(i) and i[0] < i[1] for i in v),
             "drift_mm": _num, "strength": _pos, "color": _lin3, "seed": _int1},
}
BOARD = {"mask": _rgb, "mask_rough": _unit, "soc_pos": _vec2, "soc_size": _pos, "soc_lid": _pos, "ram_per_side": _int1,
         "ram_size": lambda v: _vec2(v) and all(i > 0 for i in v), "ram_gap": _pos, "passive_len": _pos, "passive_h": _pos,
         "scatter_passives": _nat, "inductor_size": _pos, "inductor_h": _pos,
         "tantalum_size": lambda v: _vec2(v) and all(i > 0 for i in v), "tantalum": _pts, "inductors": _pts, "seed": _int1,
         "silk_designators": _nat, "trace_bundles": _nat, "via_count": _nat, "test_points": _nat}
BOARD_ROWS = {  # list-of-table sections under [board]
    "ic": {"x": _num, "y": _num, "w": _pos, "d": _pos, "h": _pos},
    "shield": {"x": _num, "y": _num, "w": _pos, "d": _pos, "h": _pos},
    "connector": {"x": _num, "y": _num, "w": _pos, "d": _pos},
    "cluster": {"x": _num, "y": _num, "w": _pos, "d": _pos, "ring": lambda v: isinstance(v, bool), "pitch_x": _pos, "pitch_y": _pos, "keep": _unit},
}
BOARD_TABLES = {"ssd": {"controller": _vec3p, "nand": _vec3p, "nand_count": _int1, "dram": _vec3p, "sticker": lambda v: _vec2(v) and all(i > 0 for i in v)},
                "wifi": {"shield": _vec3p}}
# upper bounds that keep the board looking like a real one: black IC packages and passives
MAX_ICS, MAX_PASSIVES = 6, 600
def _rgbs(v):
    return isinstance(v, list) and bool(v) and all(_rgb(i) for i in v)


MATERIAL = {"alu": _rgb, "alu_rough": _unit, "key": _rgb, "legend": _rgb, "trackpad": _rgb, "trackpad_rough": _unit,
            "floor": _rgb, "floor_rough": _unit, "floor_spec": _unit, "world": _rgb, "screen_emission": _pos,
            "pcb_edge": _rgb, "cell": _rgb, "cell_label": _rgb, "tray": _rgb, "sink": _rgb, "blade": _rgb, "copper_fin": _rgb,
            "zif": _rgb, "flex": _rgb, "wire_yellow": _rgb}
SCREEN = {"background": _rgb, "ribbons": _rgbs}


DISPLAY = {"sleep_from": lambda v: _num(v) and 0 <= v < 1, "backlight": lambda v: _num(v) and v >= 0, "halo": lambda v: _num(v) and v >= 0,
           "backlight_color": _lin3,
           "glass_mm": _pos, "cell_mm": _pos, "backlight_mm": _pos, "frame_bevel_mm": _pos,
           "glass_opacity": lambda v: _vec2(v) and all(_unit(i) for i in v) and v[0] <= v[1], "glass_fresnel": _unit,
           "cell_glass": _rgb, "glass_edge": _rgb, "panel_steel": _rgb, "panel_steel_rough": _unit, "fpc": _rgb}
LIGHT_GROUPS = ("key", "top", "rim", "wash")
ROWS = {  # list-of-table sections: required keys per row
    "light": {"name": _s, "kind": lambda v: v in ("area", "spot"), "group": lambda v: v in LIGHT_GROUPS, "loc": _vec3, "target": _vec3,
              "size": lambda v: _vec2(v) and all(i > 0 for i in v), "power": _pos, "color": _lin3,
              "spread": lambda v: _num(v) and 1 <= v <= 180, "floor": lambda v: isinstance(v, bool)},
    "pose.camera": {"frame": _int1, "az": _num, "el": _num, "dist": _pos, "target": _vec3, "lens": _pos, "fstop": _pos},
    "angle": {"name": _s, "az": _num, "el": _num, "lens": _pos, "fstop": _pos},
    "hero": {"name": _s, "state": _s, "angle": _s},
}
STATE = {"frame": _int1, "target": _vec3, "dist": _pos}
LOOK = {"exposure": _num, "key": _pos, "top": _pos, "rim": _pos, "wash": _num, "shell": _pos}


def _check_table(value, spec, where):
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a table")
    want, got = set(spec), set(value)
    if want != got:
        raise ConfigError(f"{where}: missing keys {sorted(want - got)}, unknown keys {sorted(got - want)}")
    for k, ok in spec.items():
        if not ok(value[k]):
            raise ConfigError(f"{where}.{k}: bad value {value[k]!r}")


def _dig(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"{dotted}: section missing")
        cur = cur[part]
    return cur


def _check_board(b):
    if not isinstance(b, dict):
        raise ConfigError("board: expected a table")
    keys = set(BOARD) | set(BOARD_ROWS) | set(BOARD_TABLES)
    if set(b) != keys:
        raise ConfigError(f"board: missing keys {sorted(keys - set(b))}, unknown keys {sorted(set(b) - keys)}")
    _check_table({k: b[k] for k in BOARD}, BOARD, "board")
    for name, spec in BOARD_ROWS.items():
        if not isinstance(b[name], list) or not b[name]:
            raise ConfigError(f"board.{name}: expected a non-empty list of tables")
        for i, row in enumerate(b[name]):
            _check_table(row, spec, f"board.{name}[{i}]")
    for name, spec in BOARD_TABLES.items():
        _check_table(b[name], spec, f"board.{name}")
    if len(b["ic"]) > MAX_ICS:
        raise ConfigError(f"board.ic: {len(b['ic'])} small ICs is more than {MAX_ICS}; a real board has a few large parts")


def _legacy_budget_key(cfg: dict) -> dict:
    """[budget] `lane` is the former name of `job`; a config that still uses it loads as before (a copy, never the caller's dict)."""
    b = cfg.get("budget") if isinstance(cfg, dict) else None
    if isinstance(b, dict) and "lane" in b and "job" not in b:
        return {**cfg, "budget": {("job" if k == "lane" else k): v for k, v in b.items()}}
    return cfg


def validate(cfg: dict) -> dict:
    cfg = _legacy_budget_key(cfg)
    top = set(SPEC) | {"board", "pose", "explode", "state", "angle", "hero", "look", "material", "screen", "display", "light"}
    if set(cfg) != top:
        raise ConfigError(f"top level: missing {sorted(top - set(cfg))}, unknown {sorted(set(cfg) - top)}")
    for sect, spec in SPEC.items():
        _check_table(cfg[sect], spec, sect)
    for dotted, spec in ROWS.items():
        rows = _dig(cfg, dotted)
        if not isinstance(rows, list) or not rows:
            raise ConfigError(f"{dotted}: expected a non-empty list of tables")
        for i, row in enumerate(rows):
            _check_table(row, spec, f"{dotted}[{i}]")
    if set(cfg["pose"]) != {"camera"}:
        raise ConfigError("pose: needs exactly camera")
    _check_table(cfg["material"], MATERIAL, "material")
    _check_table(cfg["screen"], SCREEN, "screen")
    _check_table(cfg["display"], DISPLAY, "display")
    if not isinstance(cfg["look"], dict) or set(cfg["look"]) != {"open", "exploded"}:
        raise ConfigError("look: needs exactly open and exploded")
    for name, lk in cfg["look"].items():
        _check_table(lk, LOOK, f"look.{name}")
    if not isinstance(cfg["state"], dict) or not cfg["state"]:
        raise ConfigError("state: expected tables")
    for name, st in cfg["state"].items():
        _check_table(st, STATE, f"state.{name}")
    _check_board(cfg["board"])
    ex = cfg["explode"]
    if not isinstance(ex, dict) or set(ex) != {"order", "offset", "center", "hide_from", "hide_parts", "rise"} or not _unit(ex["hide_from"]) \
            or not _slist(ex["hide_parts"]) or not set(ex["hide_parts"]) <= set(ex["offset"]) or not _slist(ex["order"]) or not _slist(ex["center"]) \
            or not isinstance(ex["offset"], dict) or not all(_num(x) for x in ex["offset"].values()):
        raise ConfigError("explode: needs order (list) and offset (table of numbers)")
    if sorted(ex["order"]) != sorted(ex["offset"]):
        raise ConfigError("explode: order and offset name different parts")
    if not set(ex["center"]) <= set(ex["offset"]):
        raise ConfigError("explode.center names a part that is not in offset")
    if not _slist(ex["rise"]) or len(set(ex["rise"])) != len(ex["rise"]) or not set(ex["rise"]) <= set(ex["offset"]):
        raise ConfigError("explode: rise must list distinct parts in offset")
    still = [n for n, v in ex["offset"].items() if v and n not in ex["rise"]]
    if still:
        raise ConfigError(f"explode: parts {still} have an offset but rise does not move them")
    ch = cfg["choreo"]
    if ch["close_start"] + ch["close_frames"] > ch["rise_start"] - cfg["screws"]["lead"]:
        raise ConfigError("choreo: the lid must be shut before the first screws back out")
    if cfg["screws"]["lead"] > ch["rise_stagger"]:
        raise ConfigError("screws.lead: screws must not back out before the layer above has started to rise")
    re_, last = cfg["reassemble"], len(ex["rise"]) - 1
    rose = ch["rise_start"] + last * ch["rise_stagger"] + ch["rise_frames"]    # same arithmetic as pose.span / pose.reassembled
    if re_["start"] <= rose:
        raise ConfigError(f"reassemble.start {re_['start']} must come after the exploded span ends at frame {rose}")
    if re_["start"] + last * re_["stagger"] + re_["frames"] >= cfg["render"]["n_frames"]:
        raise ConfigError("reassemble: the last part must be back before render.n_frames")
    if any(st["frame"] > cfg["render"]["n_frames"] for st in cfg["state"].values()):
        raise ConfigError("state: a still names a frame past render.n_frames")
    angles = {a["name"] for a in cfg["angle"]}
    for h in cfg["hero"]:
        if h["state"] not in cfg["state"] or h["angle"] not in angles:
            raise ConfigError(f"hero {h['name']}: unknown state or angle")
    return cfg


def load(root: str | None = None) -> dict:
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, CONFIG_REL)
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return validate(cfg)
