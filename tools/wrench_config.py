"""Strict loader for config/wrench.toml (stdlib only): exact keys at every level, numbers where numbers belong."""
from __future__ import annotations

from pathlib import Path
import math
import tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config" / "wrench.toml"


class WrenchConfigError(ValueError):
    pass


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def pos(v):
    return _num(v) and v > 0


def nn(v):
    return _num(v) and v >= 0


def rng(lo, hi):
    return lambda v: _num(v) and lo <= v <= hi


def integer(lo, hi=10**9):
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _s(v):
    return isinstance(v, str) and bool(v)


def _seq(n, ok):
    return lambda v: isinstance(v, list) and len(v) == n and all(ok(x) for x in v)


def _some(ok, least=1):
    return lambda v: isinstance(v, list) and len(v) >= least and all(ok(x) for x in v)


def _d(**fields):
    return lambda v: isinstance(v, dict) and set(v) == set(fields) and all(ok(v[k]) for k, ok in fields.items())


_slist = _some(_s)
_rgb = _seq(3, rng(0, 1))
_ramp = _some(_seq(2, rng(0, 1)), 2)
_SCHEMA = {
    "blender": dict(download_url=_s, build=_s, image=_s, python=_s, apt_packages=_slist, pip_packages=_slist),
    "modal": dict(app=_s, cpu_count=integer(1, 64), timeout_s=integer(1, 86400)),
    "budget": dict(project=_s, job=_s, cap_usd=pos, cpu_usd_per_s=pos, overhead_s=nn, ledger=_s),
    "wrench": dict(length=pos, ring_r=pos, ring_hole_r=_seq(2, pos), jaw_r=pos, slot_w=pos, slot_depth=pos,
                   shaft_half=_seq(2, pos), fillet=nn, thickness=pos,
                   edge_bevel=lambda v: isinstance(v, list) and len(v) == 2 and pos(v[0]) and integer(1, 64)(v[1]),
                   ridge=lambda v: _d(start=nn, end=pos, width=pos, **{"raise": pos})(v) and v["end"] > v["start"],
                   metal=_d(color=_rgb, roughness=rng(0, 1)),
                   accent=_rgb,
                   accent_finish=_d(roughness=rng(0, 1), coat=rng(0, 1)),
                   jaw_soften=nn,
                   ridge_bevel=_d(ratio=rng(0.01, 1), segments=integer(1, 64), angle_deg=rng(1, 180), smooth_deg=rng(1, 180)),
                   body_smooth_deg=rng(1, 180),
                   circle_res=integer(4, 256)),
    "scene": dict(unit_mm=pos, elevation_deg=rng(1, 89), axis_deg=rng(0, 180), focal_mm=pos, distance=pos,
                  exposure=rng(-10, 10), look=_s, world=_ramp,
                  card=_d(distance=pos, size=pos, strength=nn, ramp=_ramp),
                  rim=_d(energy=nn, size_xy=_seq(2, pos), pos=_seq(3, _num)),
                  fill=_d(energy=nn, size=pos, pos=_seq(3, _num))),
    "mark": dict(variant=_s, mark_px=integer(1, 4096), hi_px=integer(1, 8192), box=_seq(3, pos), gap=nn,
                 tone=_d(gamma=pos, gain=pos),
                 letters=_d(scale=pos, p=_seq(2, _num), f=_seq(2, _num)),
                 shadow=_d(keep=rng(0, 1), falloff_px=nn, reach=pos, cutoff=rng(0, 1)),
                 export=_d(webp_quality=integer(1, 100), webp_alpha_quality=integer(1, 100), webp_method=integer(0, 6))),
    "render": dict(res=integer(1, 8192), samples=integer(1, 100000), denoiser=_s, seed=integer(0, 2**31 - 1), variants=_slist),
}


def validate(cfg: dict) -> dict:
    b = cfg.get("budget") if isinstance(cfg, dict) else None
    if isinstance(b, dict) and "lane" in b and "job" not in b:  # `lane` is the former name of `job`
        cfg = {**cfg, "budget": {("job" if k == "lane" else k): v for k, v in b.items()}}
    if not isinstance(cfg, dict) or set(cfg) != set(_SCHEMA):
        got = set(cfg) if isinstance(cfg, dict) else set()
        raise WrenchConfigError(f"sections differ: missing {sorted(set(_SCHEMA) - got)}, unknown {sorted(got - set(_SCHEMA))}")
    for sec, keys in _SCHEMA.items():
        got = cfg[sec]
        if not isinstance(got, dict) or set(got) != set(keys):
            have = set(got) if isinstance(got, dict) else set()
            raise WrenchConfigError(f"[{sec}] keys differ: missing {sorted(set(keys) - have)}, unknown {sorted(have - set(keys))}")
        for k, ok in keys.items():
            if not ok(got[k]):
                raise WrenchConfigError(f"[{sec}] {k}: bad value {got[k]!r}")
    if cfg["mark"]["variant"] not in cfg["render"]["variants"]:
        raise WrenchConfigError(f"[mark] variant: {cfg['mark']['variant']!r} is not in [render] variants")
    return cfg


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as f:
        return validate(tomllib.load(f))
