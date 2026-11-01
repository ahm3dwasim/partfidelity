"""Procedural textures for the teardown laptop (Pillow + numpy, no Blender).

usage: python3 textures.py <out_dir> <px_per_mm>
Footprints on the board come from layout.py, so silkscreen and pads sit under the real chips.
"""
import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import layout as L

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, max(6, int(size)))
    return ImageFont.load_default(max(6, int(size)))


def _lift(rgb, k):
    return tuple(min(255, int(c + k)) for c in rgb)


def pcb(out, name, x0, y0, x1, y1, ppm, comps, holes, fingers=None, seed=3, bc=None, joins=None):
    """Matte solder-mask board: bundles only where parts connect, a few vias, footprints and gold pads. Writes <name>.png and <name>_metal.png.

    bc is config [board] (mask colour, trace and via counts); joins is the point every part's trace bundle runs to (the SoC), or None."""
    rnd = random.Random(seed)
    w, h = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    px = lambda x, y: ((x - x0) * ppm, (y1 - y) * ppm)
    mask = tuple(bc["mask"])
    img = Image.new("RGB", (w, h), mask)
    met = Image.new("L", (w, h), 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(met)
    # very faint mask mottling so the fill is not perfectly flat
    noise = (np.random.RandomState(seed).rand(h // 8 + 1, w // 8 + 1) * 14).astype("uint8")
    n = Image.fromarray(noise).resize((w, h), Image.BICUBIC)
    img = Image.composite(Image.new("RGB", (w, h), _lift(mask, 2)), img, n.point(lambda v: 255 if v > 9 else 0).filter(ImageFilter.GaussianBlur(ppm * 0.6)))
    d = ImageDraw.Draw(img)
    tw = max(2, int(0.2 * ppm))
    tcol = _lift(mask, 7)                                                          # copper under mask: barely lighter than the mask
    if joins:                                                                      # parallel L-shaped runs from each part to the SoC
        for cx, cy, cw, cd in comps:
            if (cx, cy) == tuple(joins):
                continue
            for k in range(-2, 3):
                off = k * 0.45
                pts = [px(cx + off, cy), px(cx + off, joins[1] + off), px(joins[0], joins[1] + off)] if abs(cx - joins[0]) > abs(cy - joins[1]) else \
                      [px(cx, cy + off), px(joins[0] + off, cy + off), px(joins[0] + off, joins[1])]
                d.line(pts, fill=tcol, width=tw, joint="curve")
    for _ in range(bc["trace_bundles"]):
        x, y = rnd.uniform(x0 + 5, x1 - 5), rnd.uniform(y0 + 5, y1 - 5)
        pts = [(x, y)]
        for _ in range(rnd.randint(2, 4)):
            if rnd.random() < 0.5:
                x += rnd.uniform(-10, 10)
            else:
                y += rnd.uniform(-10, 10)
            pts.append((x, y))
        for off in (-0.4, 0.0, 0.4):
            d.line([px(a + off, c) for a, c in pts], fill=tcol, width=tw)
    # footprints: pad rows (gold) plus white silkscreen outline and a reference designator
    for i, (cx, cy, cw, cd) in enumerate(comps):
        a, b = px(cx - cw / 2 - 0.9, cy + cd / 2 + 0.9)
        c, e = px(cx + cw / 2 + 0.9, cy - cd / 2 - 0.9)
        if cw > 3 and cd > 3:
            d.rectangle([a, b, c, e], outline=(150, 154, 148), width=max(1, int(0.12 * ppm)))
            d.ellipse([a + 0.4 * ppm, e + 0.4 * ppm, a + 1.0 * ppm, e + 1.0 * ppm], fill=(150, 154, 148))   # pin-1 dot, no designator text
            for k in range(int(cw / 0.8)):                       # perimeter pads peeking out
                for yy in (cy + cd / 2 + 0.55, cy - cd / 2 - 0.55):
                    p = px(cx - cw / 2 + k * 0.8, yy)
                    dm.rectangle([p[0], p[1] - 0.2 * ppm, p[0] + 0.45 * ppm, p[1] + 0.2 * ppm], fill=255)
                    d.rectangle([p[0], p[1] - 0.2 * ppm, p[0] + 0.45 * ppm, p[1] + 0.2 * ppm], fill=(214, 170, 78))
        else:
            for sx in (-1, 1):
                p = px(cx + sx * (cw / 2 + 0.1), cy)
                r = 0.3 * ppm
                dm.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
                d.rectangle([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(214, 170, 78))
    for hx, hy in holes:                                          # screw pads: gold ring, dark hole
        p = px(hx, hy)
        r = 3.2 * ppm
        dm.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(206, 164, 74))
        r = 1.7 * ppm
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(30, 24, 18))
        dm.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=0)
    for _ in range(bc["via_count"]):                                # vias: pale copper ring with a dark drill centre
        p = px(rnd.uniform(x0, x1), rnd.uniform(y0, y1))
        r = 0.34 * ppm
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(176, 150, 96))
        dm.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
        d.ellipse([p[0] - r * 0.4, p[1] - r * 0.4, p[0] + r * 0.4, p[1] + r * 0.4], fill=(20, 28, 22))
    for _ in range(bc["silk_designators"]):                         # blank silkscreen marks: outlines, pin-1 dots, polarity bars (never text)
        p = px(rnd.uniform(x0 + 2, x1 - 10), rnd.uniform(y0 + 2, y1 - 2))
        k = rnd.random()
        col = (150, 154, 148)
        if k < 0.4:
            d.rectangle([p[0], p[1], p[0] + 1.6 * ppm, p[1] + 0.9 * ppm], outline=col, width=max(1, int(0.12 * ppm)))
        elif k < 0.7:
            d.ellipse([p[0], p[1], p[0] + 0.7 * ppm, p[1] + 0.7 * ppm], fill=col)
        else:
            d.line([p, (p[0] + 2.4 * ppm, p[1])], fill=col, width=max(1, int(0.18 * ppm)))
            d.line([p, (p[0], p[1] + 1.4 * ppm)], fill=col, width=max(1, int(0.18 * ppm)))
    for fx, fy in ((x0 + 2.0, y0 + 2.0), (x1 - 2.0, y1 - 2.0)):   # fiducials: bare-copper dots in a clear ring
        p = px(fx, fy)
        r = 1.1 * ppm
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=_lift(mask, 2))
        r = 0.5 * ppm
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(200, 168, 96))
        dm.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    a = px(x0 + 3, y1 - 3)                                         # blank barcode sticker outline (no characters)
    d.rectangle([a[0], a[1], a[0] + 9 * ppm, a[1] + 3.4 * ppm], outline=(210, 214, 206), width=max(1, int(0.15 * ppm)))
    for _ in range(bc["test_points"]):                              # test points
        p = px(rnd.uniform(x0 + 5, x1 - 5), rnd.uniform(y0 + 5, y1 - 5))
        r = 0.55 * ppm
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(210, 172, 84))
        dm.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    if fingers:                                                   # gold edge fingers at the +X short end
        for k in range(int((y1 - y0 - 1.5) / 0.5)):
            a = px(x1 - 3.4, y0 + 0.8 + k * 0.5)
            b = px(x1 - 0.1, y0 + 0.8 + k * 0.5 + 0.3)
            box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
            d.rectangle(box, fill=(222, 182, 92))
            dm.rectangle(box, fill=255)
    img.save(os.path.join(out, name + ".png"))
    met.save(os.path.join(out, name + "_metal.png"))


def chip(out, name, lines, base, ink, size=512, dot=True, border=None):
    img = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(img)
    rnd = np.random.RandomState(abs(hash(name)) % 1000)
    arr = np.asarray(img).astype("int16") + rnd.randint(-5, 6, (size, size, 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    if border:
        d.rectangle([2, 2, size - 3, size - 3], outline=border, width=6)
    f = font(size * 0.09)
    y = size * 0.22
    for ln in lines:
        d.text((size * 0.14, y), ln, fill=ink, font=f)
        y += size * 0.14
    if dot:
        r = size * 0.03
        d.ellipse([size * 0.07, size * 0.82, size * 0.07 + 2 * r, size * 0.82 + 2 * r], outline=ink, width=3)
    img.save(os.path.join(out, name + ".png"))


def keys(out, ppm, mat):
    """Flat black caps with small light legends, as on a modern thin laptop."""
    x0, x1, y0, y1 = L.KB_X0, L.KB_X1, L.KB_Y0, L.KB_Y1
    w, h = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    img = Image.new("RGB", (w, h), tuple(mat["key"]))
    d = ImageDraw.Draw(img)
    ink = tuple(mat["legend"])
    f_big, f_small, f_word = font(4.2 * ppm), font(3.0 * ppm), font(2.6 * ppm)
    for k in L.keys():
        cx, cy = (k["x"] - x0) * ppm, (y1 - k["y"]) * ppm
        main, alt = k["main"], k["alt"]
        if not main:
            continue
        word = len(main) > 1 and main not in ("^", "v", "<", ">")
        if main in ("^", "v", "<", ">"):
            s = {"^": "▲", "v": "▼", "<": "◀", ">": "▶"}[main]
            d.text((cx, cy), s, fill=ink, font=font(2.2 * ppm), anchor="mm")
        elif word:
            d.text((cx + k["w"] / 2 * ppm - 1.8 * ppm, cy + k["h"] / 2 * ppm - 1.8 * ppm), main, fill=ink, font=f_word, anchor="rs")
        elif alt:
            d.text((cx, cy - 2.4 * ppm), alt, fill=ink, font=f_small, anchor="mm")
            d.text((cx, cy + 2.4 * ppm), main, fill=ink, font=f_small, anchor="mm")
        else:
            d.text((cx, cy), main, fill=ink, font=f_big, anchor="mm")
    img.save(os.path.join(out, "keys.png"))


def screen(out, sc):
    """Soft colour ribbons on near-black inside the active area (no text, no brand mark); slim bezels, a slightly deeper chin."""
    W, H = 1920, 1200
    yy, xx = np.mgrid[0:H, 0:W].astype("float32")
    u, v = xx / W, yy / H
    rgb = np.ones((H, W, 3), "float32") * (np.array(sc["background"], "float32") / 255.0)
    for i, col in enumerate(sc["ribbons"]):
        ph = 1.3 + i * 2.1
        centre = 0.30 + 0.20 * i + 0.11 * np.sin(2.4 * u + ph) + 0.05 * np.sin(5.3 * u + 2 * ph)
        glow = np.exp(-((v - centre) ** 2) / (2 * (0.05 + 0.03 * np.sin(1.7 * u + ph)) ** 2))
        rgb += glow[..., None] * (np.array(col, "float32") / 255.0) * 0.75
    rgb *= 1.0 - 0.45 * ((u - 0.5) ** 2 + (v - 0.5) ** 2)[..., None]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(6))
    # glass-slab UV is 0..1 over the whole cover glass; image top = the screen's top edge, bottom = the hinge chin
    gw, gd = L.GLASS[0], L.GLASS[1]
    px = 2048 / gw
    full = Image.new("RGB", (2048, int(2048 * gd / gw)), (0, 0, 0))
    side, top, chin = L.BEZEL
    aw, ah = int((gw - 2 * side) * px), int((gd - top - chin) * px)
    full.paste(img.resize((aw, ah), Image.LANCZOS), (int(side * px), int(top * px)))
    full.save(os.path.join(out, "screen.png"))


def main(out, ppm, data):
    ppm = float(ppm)
    board = data["board"]
    os.makedirs(out, exist_ok=True)
    L.configure(board)
    small = dict(board, trace_bundles=0, via_count=4, silk_designators=0, test_points=0)   # cards carry almost nothing but their parts
    comps = [(L.SOC[0], L.SOC[1], L.SOC[2], L.SOC[2])]
    comps += [(x, y, L.RAM_SIZE[0], L.RAM_SIZE[1]) for x, y in L.RAM]
    comps += [(x, y, w, d) for x, y, w, d, _, _ in L.ICS]
    comps += [(x, y, w, d) for x, y, w, d in L.CONNECTORS]
    pcb(out, "pcb_main", L.BOARD_X0, L.BOARD_Y0, L.BOARD_X1, L.BOARD_Y1, ppm, comps, L.BOARD_SCREWS, bc=board, joins=L.SOC[:2])
    sx, sy, sl, sw = L.SSD
    sb = board["ssd"]
    ssd = [(-10 + k * 18, 0, sb["nand"][0], sb["nand"][1]) for k in range(sb["nand_count"])] + [(27, 0, sb["controller"][0], sb["controller"][1]),
           (27, sb["controller"][1] / 2 + sb["dram"][1] / 2 + 1.2, sb["dram"][0], sb["dram"][1])]
    pcb(out, "pcb_ssd", -sl / 2, -sw / 2, sl / 2, sw / 2, ppm, ssd, [(-sl / 2 + 3.5, 0)], fingers=True, seed=5, bc=small)
    wx, wy, wl, ww = L.WIFI
    pcb(out, "pcb_wifi", -wl / 2, -ww / 2, wl / 2, ww / 2, ppm, [], [(-wl / 2 + 3, 0)], seed=8, bc=small)
    pcb(out, "pcb_io", -8, -30, 8, 30, ppm, [(0, -10, 7, 7)], [(0, 26)], seed=9, bc=small)
    chip(out, "chip_ram", [], (20, 20, 22), (120, 122, 126))
    chip(out, "chip_nand", [], (20, 20, 22), (150, 152, 156))
    chip(out, "chip_ctrl", [], (24, 24, 26), (150, 152, 156))
    chip(out, "chip_soc", [], (150, 152, 156), (60, 62, 66), border=(112, 114, 118))
    keys(out, max(ppm * 1.3, 18.0), data["material"])
    screen(out, data["screen"])


if __name__ == "__main__":
    L.JITTER[0] = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    main(sys.argv[1], sys.argv[2], json.load(open(sys.argv[4])))
