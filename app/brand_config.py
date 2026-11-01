"""Validate the brand config and emit complete markup for all mark placements."""
from __future__ import annotations

from functools import lru_cache
import html
from pathlib import Path
import re
import tomllib

from app import proof_links

CONFIG = Path(__file__).resolve().parents[1] / "config" / "brand.toml"
_KEYS = {"real_min_px", "flat", "real", "favicon", "touch_icon", "touch_icon_px", "touch_icon_pad", "touch_icon_bg", "header_px", "footer_px", "gallery_sizes", "mark_view_box", "header_class", "footer_class", "inline_margin_px"}
_TOKEN = re.compile(r"@@([a-z_]+)(?::(\d+))?@@")


class BrandConfigError(ValueError):
    pass


def _static(ref, suffix: str) -> bool:
    return isinstance(ref, str) and ref.startswith("/static/") and ref.endswith(suffix) and ".." not in ref


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or set(cfg) != _KEYS:
        got = set(cfg) if isinstance(cfg, dict) else set()
        raise BrandConfigError(f"keys differ (missing {sorted(_KEYS - got)}, unknown {sorted(got - _KEYS)})")
    for key in ("real_min_px", "touch_icon_px", "header_px", "footer_px"):
        if not _int(cfg[key]) or not 1 <= cfg[key] <= 4096:
            raise BrandConfigError(f"{key}: bad value {cfg[key]!r}")
    if not isinstance(cfg["gallery_sizes"], list) or len(cfg["gallery_sizes"]) != 4 or any(not _int(n) or not 1 <= n <= 4096 for n in cfg["gallery_sizes"]):
        raise BrandConfigError("gallery_sizes: expected four positive pixel sizes")
    for key in ("header_class", "footer_class"):
        if not isinstance(cfg[key], str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", cfg[key]):
            raise BrandConfigError(f"{key}: invalid class")
    if not isinstance(cfg["mark_view_box"], str) or not re.fullmatch(r"\d+ \d+ \d+ \d+", cfg["mark_view_box"]):
        raise BrandConfigError("mark_view_box: invalid viewBox")
    if not _int(cfg["inline_margin_px"]) or not 0 <= cfg["inline_margin_px"] <= 4096:
        raise BrandConfigError("inline_margin_px: invalid size")
    pad = cfg["touch_icon_pad"]
    if isinstance(pad, bool) or not isinstance(pad, (int, float)) or not 0 <= pad < 0.5:
        raise BrandConfigError(f"touch_icon_pad: bad value {pad!r}")
    if not isinstance(cfg["touch_icon_bg"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", cfg["touch_icon_bg"]):
        raise BrandConfigError(f"touch_icon_bg: bad value {cfg['touch_icon_bg']!r}")
    for key in ("flat", "real"):
        ref = cfg[key]
        if not (isinstance(ref, str) and ref.startswith("/static/") and re.fullmatch(r"[\w./-]+\.svg#[\w-]+", ref) and ".." not in ref):
            raise BrandConfigError(f"{key}: bad value {ref!r}")
    if not _static(cfg["favicon"], ".svg") or not re.fullmatch(r"[\w./-]+", cfg["favicon"]):
        raise BrandConfigError(f"favicon: bad value {cfg['favicon']!r}")
    if not _static(cfg["touch_icon"], ".png") or not re.fullmatch(r"[\w./-]+", cfg["touch_icon"]):
        raise BrandConfigError(f"touch_icon: bad value {cfg['touch_icon']!r}")
    return cfg


@lru_cache(maxsize=1)
def load() -> dict:
    with CONFIG.open("rb") as f:
        return validate(tomllib.load(f))


def mark(px: int, cfg: dict | None = None) -> str:
    """The mark reference (path#fragment) for a placement drawn px wide: flat below real_min_px, real from it up."""
    cfg = cfg or load()
    return cfg["real"] if px >= cfg["real_min_px"] else cfg["flat"]


def tokens(cfg: dict | None = None) -> dict:
    """The size-free tokens, and every mark value a page may legitimately carry."""
    cfg = cfg or load()
    return {"favicon": cfg["favicon"], "touch_icon": cfg["touch_icon"]}


def mark_badge(cfg: dict | None = None) -> str:
    cfg = cfg or load()
    px = cfg["header_px"]
    return f'<span class="{cfg["header_class"]}" aria-hidden="true"><svg viewBox="{cfg["mark_view_box"]}" width="{px}" height="{px}"><use href="{mark(px, cfg)}"></use></svg></span>'


def footer_mark(cfg: dict | None = None) -> str:
    cfg = cfg or load()
    px = cfg["footer_px"]
    return f'<svg class="{cfg["footer_class"]}" viewBox="{cfg["mark_view_box"]}" width="{px}" height="{px}" aria-hidden="true"><use href="{mark(px, cfg)}"></use></svg>'


def gallery_mark(index: int, cfg: dict | None = None) -> str:
    cfg = cfg or load()
    px = cfg["gallery_sizes"][index]
    return f'<img src="{mark(px, cfg).split("#")[0]}" width="{px}" height="{px}" alt="PartFidelity mark at {px} pixels">'


def inline_mark(cfg: dict | None = None) -> str:
    cfg = cfg or load()
    px = cfg["header_px"]
    return f'<img src="{cfg["favicon"]}" width="{px}" height="{px}" alt="" style="vertical-align:middle;margin-right:{cfg["inline_margin_px"]}px">'


def icon_links(cfg: dict | None = None) -> str:
    cfg = cfg or load()
    return f'<link rel="icon" href="{cfg["favicon"]}" type="image/svg+xml">\n<link rel="apple-touch-icon" href="{cfg["touch_icon"]}">'


def fill(text: str, cfg: dict | None = None) -> str:
    """Replace every @@token@@; an unknown or malformed one raises, so a typo cannot ship."""
    cfg = cfg or load()
    table = {"brand_badge": mark_badge(cfg), "brand_footer": footer_mark(cfg),
             "brand_inline": inline_mark(cfg), "brand_icons": icon_links(cfg),
             "catch_cta": html.escape(proof_links.load()["catch_cta"])}

    def sub(m):
        name, px = m.group(1), m.group(2)
        if name in ("brand_gallery", "brand_size") and px is not None and int(px) < len(cfg["gallery_sizes"]):
            return gallery_mark(int(px), cfg) if name == "brand_gallery" else str(cfg["gallery_sizes"][int(px)])
        if name in ("mark", "mark_file") and px:
            if int(px) < 1:
                raise BrandConfigError(f"brand token {m.group(0)}: size must be a positive pixel count")
            ref = mark(int(px), cfg)
            return ref if name == "mark" else ref.split("#")[0]
        if name in table and not px:
            return table[name]
        raise BrandConfigError(f"unknown brand token {m.group(0)}")

    out = _TOKEN.sub(sub, text)
    if "@@" in out:
        raise BrandConfigError("malformed brand token: unresolved @@ delimiter")
    return out
