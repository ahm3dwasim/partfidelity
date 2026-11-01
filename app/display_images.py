"""Display copies of guide stills: strict config, derivation and cache. Stdlib plus Pillow.

A copy is made only from a still the guide already serves (the redacted file), never from anything else, and is
keyed by the source bytes' hash, so a redaction rebuild can never leave an older copy behind. The authoritative
PNG is untouched; the guide route serves it when no width is asked for.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

CONFIG = Path(__file__).resolve().parents[1] / "config" / "display_images.toml"
FORMATS = {"webp": ("WEBP", "image/webp", "webp"), "jpeg": ("JPEG", "image/jpeg", "jpg")}
_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


class DisplayImagesError(ValueError):
    pass


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def load(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    spec = {
        "widths": lambda v: isinstance(v, list) and bool(v) and all(_int(i) and i >= 1 for i in v)
        and all(a < b for a, b in zip(v, v[1:])),
        "format": lambda v: v in FORMATS,
        "quality": lambda v: _int(v) and 1 <= v <= 100,
        "cache_dir": lambda v: isinstance(v, str) and bool(_SAFE.match(v)) and v not in (".", ".."),
        "stage_width": _int,
        "poster_width": _int,
        "poster_quality": lambda v: _int(v) and 1 <= v <= 100,
        "sizes": lambda v: isinstance(v, str) and bool(v),
    }
    if set(cfg) != set(spec):
        raise DisplayImagesError(f"keys differ (missing {sorted(set(spec) - set(cfg))}, unknown {sorted(set(cfg) - set(spec))})")
    for key, ok in spec.items():
        if not ok(cfg[key]):
            raise DisplayImagesError(f"{key}: bad value {cfg[key]!r}")
    for key in ("stage_width", "poster_width"):
        if cfg[key] not in cfg["widths"]:
            raise DisplayImagesError(f"{key}: must be one of widths")
    return cfg


@lru_cache(maxsize=1)
def cached() -> dict:
    return load()


@dataclass(frozen=True)
class Display:
    src: str            # the copy the stage and the toggle start from
    srcset: str
    sizes: str
    width: int          # the copy's own pixel size at src, reserved so the page does not jump
    height: int
    type: str = "image/webp"   # media type of the copies, for the <source> of a <picture>


def clamp(width: int, source_width: int) -> int:
    """The width a copy actually has: never past its source."""
    return min(width, source_width)


def srcset_entries(source_width: int, widths: list[int]) -> list[tuple[int, int]]:
    """(requested width, real width) pairs, one per distinct real width."""
    seen: dict[int, int] = {}
    for w in widths:
        seen.setdefault(clamp(w, source_width), w)
    return [(w, real) for real, w in seen.items()]


def describe(url_for: "callable", source_size: tuple[int, int], cfg: dict | None = None) -> Display:
    """The attributes for one image. ``url_for(w)`` is the route URL of the copy at requested width ``w``."""
    cfg = cfg or cached()
    sw, sh = source_size
    srcset = ", ".join(f"{url_for(w)} {real}w" for w, real in srcset_entries(sw, cfg["widths"]))
    real = clamp(cfg["stage_width"], sw)
    return Display(url_for(cfg["stage_width"]), srcset, cfg["sizes"], real, round(sh * real / sw), FORMATS[cfg["format"]][1])


def poster_cfg(cfg: dict | None = None) -> dict:
    """The config for the clip poster: a JPEG, which every browser can draw, at the poster quality."""
    cfg = cfg or cached()
    return {**cfg, "format": "jpeg", "quality": cfg["poster_quality"]}


def display_bytes(source: Path, width: int, cfg: dict | None = None) -> tuple[bytes | Path, str] | None:
    """The copy of ``source`` at ``width`` (a cached file path, or bytes when the folder is not writable) and its
    media type; None when the source cannot be decoded (the caller then serves the original)."""
    cfg = cfg or cached()
    fmt, media, ext = FORMATS[cfg["format"]]
    try:
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:20]
        target = source.parent / cfg["cache_dir"] / f"{source.stem}-{digest}-{width}-{cfg['quality']}.{ext}"
        if target.is_file():
            return target, media
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            mode = "RGBA" if im.mode in ("RGBA", "LA", "PA") and fmt != "JPEG" else "RGB"
            im = im.convert(mode)
            if im.width > width:
                im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, fmt, quality=cfg["quality"])
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    data = buf.getvalue()
    try:
        target.parent.mkdir(exist_ok=True)
        mine = re.compile(rf"^{re.escape(source.stem)}-[0-9a-f]{{20}}-\d+-\d+\.\w+$")
        for old in target.parent.iterdir():   # a rebuilt (re-redacted) still must not leave its old copies
            if mine.match(old.name) and f"-{digest}-" not in old.name:
                old.unlink(missing_ok=True)
        fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")  # unique per request
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(name, target)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
        return target, media
    except OSError:
        return data, media
