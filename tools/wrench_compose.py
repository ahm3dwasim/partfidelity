"""Lay the rendered wrench into the PF mark. Local, Pillow + numpy only.

  python3 tools/wrench_compose.py /tmp/logo-wrench/render

Reads wrench_<variant>.png (floor, carries the shadow) and wrench_<variant>_cut.png (no floor, the clean silhouette) from
the render folder; writes static/ui/wrench-real/wrench-{1x,2x,3x}.{png,webp} and static/ui/icon-real.svg.
All layout values come from config/wrench.toml [mark].
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from wrench_config import load  # noqa: E402

OUT_DIR = ROOT / "static" / "ui" / "wrench-real"
SVG = ROOT / "static" / "ui" / "icon-real.svg"


def cutout(src: Path, variant: str, shadow: dict, tone: dict) -> Image.Image:
    """The wrench from the no-floor render, plus the floor render's shadow as black, faded to nothing away from the wrench."""
    full = np.asarray(Image.open(src / f"wrench_{variant}.png").convert("RGBA")).astype(np.float32) / 255
    cut = Image.open(src / f"wrench_{variant}_cut.png").convert("RGBA")
    c = np.asarray(cut).astype(np.float32) / 255
    sil = Image.fromarray((c[..., 3] > 0.5).astype(np.uint8) * 255)
    near = np.asarray(sil.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(shadow["falloff_px"]))).astype(np.float32) / 255
    sh = np.clip(full[..., 3] - c[..., 3], 0, 1) * np.clip(near * shadow["reach"], 0, 1) * shadow["keep"]
    sh = np.where(sh < shadow["cutoff"], 0, sh)                      # no floor haze
    c[..., :3] = np.clip(c[..., :3] ** tone["gamma"] * tone["gain"], 0, 1)
    a = c[..., 3] + sh * (1 - c[..., 3])
    rgb = np.where(a[..., None] > 0, (c[..., :3] * c[..., 3:4]) / np.maximum(a[..., None], 1e-6), 0)   # shadow pixels are black
    out = Image.fromarray((np.dstack([np.clip(rgb, 0, 1), a]) * 255 + 0.5).astype(np.uint8), "RGBA")
    return out.crop(out.getchannel("A").point(lambda v: 255 if v > 3 else 0).getbbox())


def webp(im: Image.Image, ex: dict) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=ex["webp_quality"], alpha_quality=ex["webp_alpha_quality"], method=ex["webp_method"])
    return buf.getvalue()


def png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def build(src: Path, cfg: dict) -> None:
    m = cfg["mark"]
    im = cutout(src, m["variant"], m["shadow"], m["tone"])
    left, top, h = m["box"]
    w = h * im.width / im.height
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k in (1, 2, 3):
        ph = round(h / 32 * m["mark_px"] * k)
        small = im.resize((round(ph * im.width / im.height), ph), Image.LANCZOS)
        (OUT_DIR / f"wrench-{k}x.png").write_bytes(png(small))
        (OUT_DIR / f"wrench-{k}x.webp").write_bytes(webp(small, m["export"]))
    hi = im.resize((round(m["hi_px"] * im.width / im.height), round(m["hi_px"])), Image.LANCZOS)
    data = base64.b64encode(webp(hi, m["export"])).decode()
    s, (pdx, pdy), (fdx, fdy) = m["letters"]["scale"], m["letters"]["p"], m["letters"]["f"]
    P, F = "M5.5 26V7h5.75a4.75 4.75 0 0 1 0 9.5H5.5", "M20.5 26V7h7.5M20.5 16h6.25"
    SVG.write_text(f"""<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' viewBox="0 0 32 32" width="96" height="96" role="img" aria-label="PartFidelity mark: the letters P and F with a rendered steel wrench passing between them">
  <defs>
    <linearGradient id="pf-real-letter" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#f7f7f4"/><stop offset="1" stop-color="#c4c4bf"/>
    </linearGradient>
    <filter id="pf-real-shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0.3" dy="0.5" stdDeviation="0.35" flood-color="#000" flood-opacity=".35"/></filter>
    <!-- the dark gap round the wrench: its silhouette (alpha above one half, so the soft shadow is left out), grown -->
    <filter id="pf-real-gap" x="-15%" y="-15%" width="130%" height="130%" color-interpolation-filters="sRGB">
      <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 30 -14" result="sil"/>
      <feMorphology in="sil" operator="dilate" radius="{m['gap']}" result="grow"/>
      <feFlood style="flood-color: var(--pf-mark-gap, #0f0f0f)"/><feComposite in2="grow" operator="in" result="gap"/>
      <feMerge><feMergeNode in="gap"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <g id="pf-mark-real">
    <g transform="translate({pdx} {pdy}) scale({s})">
      <path d="{P}" fill="none" stroke-width="4.2" stroke-linejoin="round" stroke="url(#pf-real-letter)" filter="url(#pf-real-shadow)"/>
    </g>
    <image x="{left}" y="{top}" width="{w:.3f}" height="{h}" filter="url(#pf-real-gap)" href="data:image/webp;base64,{data}" xlink:href="data:image/webp;base64,{data}"/>
    <g transform="translate({fdx} {fdy}) scale({s})">
      <path d="{F}" fill="none" stroke-linejoin="round" style="stroke: var(--pf-mark-gap, #0f0f0f)" stroke-width="7"/>
      <path d="{F}" fill="none" stroke-width="4.2" stroke-linejoin="round" stroke="url(#pf-real-letter)" filter="url(#pf-real-shadow)"/>
    </g>
  </g>
</svg>
""")
    print(f"wrench {w:.2f}x{h} units; svg {SVG.stat().st_size // 1024} KB")


if __name__ == "__main__":
    build(Path(sys.argv[1]), load())
