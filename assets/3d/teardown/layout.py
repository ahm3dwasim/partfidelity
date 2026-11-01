"""Pure-Python layout of the teardown laptop (mm, +X right, +Y toward the hinge, +Z up).

Shared by textures.py (system Python + Pillow) and parts.py (Blender), so a
chip's footprint on the board texture always matches the chip that sits on it.
"""
import random

W, D = 304.0, 212.0          # footprint
HINGE_Y, HINGE_Z = 102.0, 6.6  # lid axis; render.py sets it from config scene.pivot_yz
CORNER = [13.0]              # plan corner radius, config scene.corner_mm
DECK_BEVEL = [(0.7, 2)]      # edge chamfer (mm, segments), config scene.deck_bevel
DECK_TOP = 7.4               # top of the keyboard deck
LID_Z0, LID_Z1 = 7.7, 11.3   # closed-pose lid slab: glass side down, 3.6 mm thick
GLASS = (301.6, 209.6, 0.0)  # w, d, centre y of the edge-to-edge cover glass
BEZEL = (4.5, 4.5, 9.0)      # side, top and bottom (chin) black border of the glass round the active area, mm
HINGE_X = 100.0

# ---- keyboard -------------------------------------------------------------
PITCH, FN_PITCH, KEY_GAP = 18.0, 15.0, 2.4
KB_TOP_Y = 92.0
KB_X0, KB_X1 = -135.0, 135.0
KB_Y0, KB_Y1 = KB_TOP_Y - (FN_PITCH + 5 * PITCH) - 3.0, KB_TOP_Y + 3.0   # texture bbox

_NUM = [("`", "~"), ("1", "!"), ("2", "@"), ("3", "#"), ("4", "$"), ("5", "%"), ("6", "^"),
        ("7", "&"), ("8", "*"), ("9", "("), ("0", ")"), ("-", "_"), ("=", "+")]
_ROWS = [
    (FN_PITCH, [("esc", "", 1.5)] + [("F%d" % i, "", 1.0) for i in range(1, 13)] + [("", "", 1.0)]),
    (PITCH, [(a, b, 1.0) for a, b in _NUM] + [("del", "", 1.5)]),
    (PITCH, [("tab", "", 1.5)] + [(c, "", 1.0) for c in "QWERTYUIOP"] + [("[", "{", 1.0), ("]", "}", 1.0)] + [("\\", "|", 1.0)][:0]
     + [("\\", "|", 1.0)]),
    (PITCH, [("caps", "", 1.75)] + [(c, "", 1.0) for c in "ASDFGHJKL"] + [(";", ":", 1.0), ("'", '"', 1.0), ("enter", "", 1.75)]),
    (PITCH, [("shift", "", 2.25)] + [(c, "", 1.0) for c in "ZXCVBNM"] + [(",", "<", 1.0), (".", ">", 1.0), ("/", "?", 1.0), ("shift", "", 2.25)]),
    (PITCH, [("fn", "", 1.0), ("ctrl", "", 1.0), ("alt", "", 1.25), ("super", "", 1.25), ("", "", 4.5),
             ("super", "", 1.25), ("alt", "", 1.25), ("<", "", 1.0), ("UD", "", 1.0), (">", "", 1.0)]),
]


JITTER = [0.0]      # mm, set from config by render.py and textures.py before keys() is used


def keys():
    """[{main, alt, x, y, w, h}] cap centres and cap sizes, mm. Seeded jitter makes the gaps uneven like real stamped caps."""
    rnd = random.Random(5)
    j = JITTER[0]
    out, y_top = [], KB_TOP_Y
    for pitch, row in _ROWS:
        x = -sum(k[2] for k in row) * PITCH / 2
        yc = y_top - pitch / 2
        for main, alt, u in row:
            w = u * PITCH - KEY_GAP
            xc = x + u * PITCH / 2
            if main == "UD":          # up/down share one key slot, half height each
                hh = (pitch - KEY_GAP) / 2 - 0.5
                out.append(dict(main="^", alt="", x=xc, y=yc + pitch / 4, w=w, h=hh))
                out.append(dict(main="v", alt="", x=xc, y=yc - pitch / 4, w=w, h=hh))
            else:
                out.append(dict(main=main, alt=alt, x=xc, y=yc, w=w, h=pitch - KEY_GAP))
            x += u * PITCH
        y_top -= pitch
    return [dict(k, x=k["x"] + rnd.uniform(-j, j), y=k["y"] + rnd.uniform(-j, j) * 0.6,
                 w=k["w"] + rnd.uniform(-j, j) * 0.5, h=k["h"] + rnd.uniform(-j, j) * 0.5) for k in out]


KB_WELL = (0.0, KB_TOP_Y - (FN_PITCH + 5 * PITCH) / 2, 266.0, FN_PITCH + 5 * PITCH + 3.0)  # cx, cy, w, d
TRACKPAD = (0.0, -61.0, 130.0, 74.0)

# ---- logic board ----------------------------------------------------------
# A wide strip across the rear of the case (as in the reference), notched at the right for the blower fan.
BOARD_X0, BOARD_X1, BOARD_Y0, BOARD_Y1 = -116.0, 118.0, 6.0, 98.0
BOARD_Z0, BOARD_T = 1.6, 1.2
BOARD_TOP = BOARD_Z0 + BOARD_T
FAN_NOTCH = (62.0, 66.0)                       # board cut-out: x from here to BOARD_X1, y from BOARD_Y0 to here
BOARD_POLY = [(BOARD_X0 + 4, BOARD_Y1), (BOARD_X1 - 4, BOARD_Y1), (BOARD_X1, BOARD_Y1 - 4), (BOARD_X1, FAN_NOTCH[1]),
              (FAN_NOTCH[0] + 6, FAN_NOTCH[1]), (FAN_NOTCH[0], FAN_NOTCH[1] - 6), (FAN_NOTCH[0], BOARD_Y0),
              (BOARD_X0 + 4, BOARD_Y0), (BOARD_X0, BOARD_Y0 + 4), (BOARD_X0, BOARD_Y1 - 4)]

# set by configure() from config/teardown.toml [board]: SOC, RAM, ICS, SHIELDS, CONNECTORS, POLY_CAPS, INDUCTORS, B
SOC, RAM, RAM_SIZE, ICS, SHIELDS, CONNECTORS, POLY_CAPS, INDUCTORS, B = None, [], (0, 0), [], [], [], [], [], None
BOARD_SCREWS = [(-111.0, 93.0), (113.0, 93.0), (57.0, 11.0), (-111.0, 11.0), (-48.0, 93.0), (30.0, 93.0), (-20.0, 11.0)]
SSD = (-74.0, 22.0, 66.0, 20.0)                # cx, cy, length(X), width(Y): an M.2 card on the board
WIFI = (-98.0, 74.0, 26.0, 22.0)


def configure(b):
    """Place the board's parts from the [board] config: one SoC, memory packages hugging it, a few small ICs, shield cans."""
    global SOC, RAM, RAM_SIZE, ICS, SHIELDS, CONNECTORS, POLY_CAPS, INDUCTORS, B
    B = b
    sx, sy = b["soc_pos"]
    SOC = (sx, sy, b["soc_size"])
    RAM_SIZE = tuple(b["ram_size"])
    rw, rd = RAM_SIZE
    n = b["ram_per_side"]
    dx = b["soc_size"] / 2 + b["ram_gap"] + rw / 2
    RAM = [(sx + side * dx, sy + (k - (n - 1) / 2) * (rd + 1.0)) for side in (-1, 1) for k in range(n)]
    ICS = [(r["x"], r["y"], r["w"], r["d"], r["h"], "ctrl") for r in b["ic"]]
    SHIELDS = [(r["x"], r["y"], r["w"], r["d"], r["h"]) for r in b["shield"]]
    CONNECTORS = [(r["x"], r["y"], r["w"], r["d"]) for r in b["connector"]]
    POLY_CAPS = [tuple(p) for p in b["tantalum"]]
    INDUCTORS = [tuple(p) for p in b["inductors"]]


# battery: a row of pouch cells across the front; speakers at the sides of the board; one blower fan in the board notch
BAT_N, BAT_W, BAT_GAP, BAT_Y, BAT_D = 4, 64.0, 2.4, -55.0, 80.0
BAT_CELLS = [((i - (BAT_N - 1) / 2) * (BAT_W + BAT_GAP), BAT_Y, BAT_W, BAT_D) for i in range(BAT_N)]   # cx, cy, w, d
SPEAKERS = [(-136.0, 52.0, 20.0, 82.0), (136.0, 52.0, 20.0, 82.0)]     # cx, cy, w, d
FAN = (90.0, 36.0, 24.0, 70.0)                 # cx, cy, housing radius, outlet end y
FINS = (90.0, 74.0, 44.0)                      # cx, cy, length along X
SCREW_R = 1.5


def passives():
    """0402-size parts as (cx, cy, w, d, h, kind): dense clusters round the SoC and power stage, a few strays, clear of the big parts."""
    if B is None:
        raise RuntimeError("layout.configure(cfg['board']) was not called")
    rnd = random.Random(B["seed"])
    tw, td = B["tantalum_size"]
    big = [(SOC[0], SOC[1], SOC[2] + 1.6, SOC[2] + 1.6)]
    big += [(x, y, RAM_SIZE[0] + 1.6, RAM_SIZE[1] + 1.6) for x, y in RAM]
    big += [(x, y, w + 1.6, d + 1.6) for x, y, w, d, _, _ in ICS]
    big += [(x, y, w + 1.6, d + 1.6) for x, y, w, d, _ in SHIELDS]
    big += [(x, y, w + 1.6, d + 1.6) for x, y, w, d in CONNECTORS]
    big += [(x, y, tw + 1.6, td + 1.6) for x, y in POLY_CAPS] + [(x, y, B["inductor_size"] + 1.6, B["inductor_size"] + 1.6) for x, y in INDUCTORS]
    big += [(x, y, 5.0, 5.0) for x, y in BOARD_SCREWS]
    big += [(SSD[0], SSD[1], SSD[2] + 2, SSD[3] + 2), (WIFI[0], WIFI[1], WIFI[2] + 2, WIFI[3] + 2)]
    big += [((FAN_NOTCH[0] + BOARD_X1) / 2, (BOARD_Y0 + FAN_NOTCH[1]) / 2, BOARD_X1 - FAN_NOTCH[0] + 3, FAN_NOTCH[1] - BOARD_Y0 + 3)]

    def clear(x, y, w, d):
        return all(abs(x - bx) > (w + bw) / 2 or abs(y - by) > (d + bd) / 2 for bx, by, bw, bd in big)

    pts = []
    for c in B["cluster"]:
        horiz = rnd.random() < 0.5
        nx, ny = int(c["w"] / c["pitch_x"]), int(c["d"] / c["pitch_y"])
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = c["x"] - c["w"] / 2 + i * c["pitch_x"] + rnd.uniform(-0.12, 0.12)
                y = c["y"] - c["d"] / 2 + j * c["pitch_y"] + rnd.uniform(-0.1, 0.1)
                if c["ring"] and abs(x - c["x"]) < SOC[2] / 2 + 1.6 and abs(y - c["y"]) < SOC[2] / 2 + 1.6:
                    continue                                       # keep the ring clear of the package itself
                if rnd.random() < c["keep"]:
                    pts.append((x, y, horiz))
    for _ in range(B["scatter_passives"]):
        pts.append((rnd.uniform(BOARD_X0 + 4, BOARD_X1 - 4), rnd.uniform(BOARD_Y0 + 4, BOARD_Y1 - 4), rnd.random() < 0.5))
    n, h = B["passive_len"], B["passive_h"]
    out = []
    for x, y, horiz in pts:
        w, d = (n, n / 2) if horiz else (n / 2, n)
        if rnd.random() < 0.2:
            w, d = w * 1.5, d * 1.5
        if BOARD_X0 + 1.5 < x < BOARD_X1 - 1.5 and BOARD_Y0 + 1.5 < y < BOARD_Y1 - 1.5 and clear(x, y, w + 0.4, d + 0.4) \
                and all(abs(x - o[0]) > 1.0 or abs(y - o[1]) > 0.8 for o in out[-60:]):
            out.append((x, y, w, d, h if w < n * 1.2 and d < n * 1.2 else h * 1.5, "cap" if rnd.random() < 0.6 else "res"))
    return out
