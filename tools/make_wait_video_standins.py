"""Drawn stand-in frames for the wait-video gallery (no real footage).

Generates four deterministic 1280x720 JPEGs under static/ui/media/wait-video/
so the gallery, browser tests and byte budget run in a clean checkout without
the git-ignored consented stills in static/ui/media/local/ (which are never
committed and never used for training).

The drawings echo the demo overlay coordinates in
static/ui/gallery/wait-video.html (battery polygon, fan circle) so the traced
outlines land on drawn shapes. Regenerate with:
    python3 tools/make_wait_video_standins.py
Requires Pillow. Output bytes are deterministic for a given Pillow version
(fixed palette, coordinates, and bitmap font; no randomness, no EXIF).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "ui" / "media" / "wait-video"

W, H = 1280, 720
BG = (35, 35, 35)
PANEL = (29, 29, 29)
PANEL_LINE = (44, 44, 44)
SHAPE_FILL = (58, 60, 54)
SHAPE_LINE = (138, 141, 132)
FAN_LINE = (120, 123, 115)
TEXT_FG = (242, 240, 235)
TEXT_DIM = (169, 165, 156)

# Demo battery polygon, copied from the gallery's fake script.
BATTERY = [(390, 215), (700, 190), (905, 178), (1125, 195),
           (1125, 595), (990, 645), (470, 628), (388, 555)]
FAN_CENTER = (940, 120)
FAN_R = 65

FRAMES = ("synth-s09", "synth-s10", "synth-s12", "synth-s16")


def draw_frame(name: str, index: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # Interior panel.
    d.rectangle([40, 40, W - 40, H - 40], fill=PANEL, outline=PANEL_LINE, width=2)
    # Battery block with cell dividers.
    d.polygon(BATTERY, fill=SHAPE_FILL, outline=SHAPE_LINE)
    for x in (560, 730, 900):
        d.line([(x, 205), (x - 25, 615)], fill=SHAPE_LINE, width=2)
    d.text((430, 400), "BATTERY", font=font, fill=TEXT_DIM)
    # Fan: ring, blades (angle varies per frame), hub.
    cx, cy = FAN_CENTER
    d.ellipse([cx - FAN_R, cy - FAN_R, cx + FAN_R, cy + FAN_R],
              outline=FAN_LINE, width=3)
    base = math.radians(index * 15)
    for blade in range(7):
        angle0 = base + blade * 2 * math.pi / 7
        angle1 = angle0 + 0.5
        d.line([(cx + 18 * math.cos(angle0), cy + 18 * math.sin(angle0)),
                (cx + 52 * math.cos(angle1), cy + 52 * math.sin(angle1))],
               fill=FAN_LINE, width=6)
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=SHAPE_LINE)
    d.text((cx - 12, cy + 74), "FAN", font=font, fill=TEXT_DIM)
    # Corner screws.
    for sx, sy in ((90, 90), (W - 90, 90), (90, H - 90), (W - 90, H - 90)):
        d.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], outline=SHAPE_LINE, width=2)
        d.line([(sx - 9, sy), (sx + 9, sy)], fill=SHAPE_LINE, width=2)
    # Stand-in banner so the fixture is never mistaken for footage.
    d.rectangle([40, 40, 560, 96], fill=(19, 19, 19))
    d.text((56, 52), "SYNTHETIC STAND-IN", font=font, fill=TEXT_FG)
    d.text((56, 68), name + " / drawn fixture, not footage",
           font=font, fill=TEXT_DIM)
    d.text((W - 160, H - 60), "1280x720", font=font, fill=TEXT_DIM)
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, name in enumerate(FRAMES):
        target = OUT / (name + ".jpg")
        draw_frame(name, index).save(target, "JPEG", quality=75)
        size = target.stat().st_size
        total += size
        print(f"{target.relative_to(ROOT)} {size} bytes")
    print(f"total {total} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
