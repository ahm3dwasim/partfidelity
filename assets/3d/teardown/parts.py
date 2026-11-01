"""Every part of the laptop (mm, closed pose). One builder per teardown layer; see layout.py for positions."""
import math
import random
import traceback

import bmesh
import bpy
from mathutils import Matrix, Vector

import geom as G
import layout as L
import mats
from geom import CUR, Part, bm_box, bm_prism, bm_sweep, box, circle, cutter, cyl, flat_profile, fillet_path, prism, rrect, sub_group, to_obj

ERRORS = []


def M(n):
    return mats.get(n)


def _rot(pts, deg, cx=0.0, cy=0.0):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]


def screws(pts, z, r=1.5, h=0.6, head="steel", lift_group=True):
    """Pan-head screws with a cross slot, in a sub-group that hovers when the layer explodes."""
    if not pts:
        return
    saved = CUR["group"]
    sg = sub_group(CUR["part"] + "_screws", saved)
    CUR["group"] = sg
    bm, bs = bmesh.new(), bmesh.new()
    for x, y in pts:
        bm_prism(bm, circle(r, x, y, 20), z, z + h, taper=0.92)
        for a in (0, 90):
            bm_prism(bs, _rot(rrect(r * 1.5, 0.34, 0.0, x, y), a, x, y), z + h - 0.05, z + h + 0.04)
    to_obj("screw_heads", bm, mat=M(head), width=0.18, seg=2)
    to_obj("screw_slots", bs, mat=M("steel_dark"), bevel=False)
    CUR["group"] = saved


def chip(name, cx, cy, z0, w, d, h, tex, r=0.4):
    return box(name, cx, cy, z0, w, d, h, r, mat=M(tex), uv=(cx - w / 2, cy - d / 2, w, d), width=0.12, seg=2)


def shield(name, cx, cy, z0, w, d, h):
    """Stamped EMI shield can: flat lid with a recessed centre panel, a pick-up hole and two vent holes."""
    box(name, cx, cy, z0, w, d, h, 0.5, mat=M("shield"), width=0.12, seg=2)
    box(name + "_panel", cx, cy, z0 + h, w - 1.4, d - 1.4, 0.06, 0.3, mat=M("shield"), bevel=False)
    for dx in (-w / 4, 0.0, w / 4):
        cyl(name + "_hole", cx + dx, cy + d / 4 * (1 if dx else -1), z0 + h + 0.06, 0.55 if dx else 1.1, 0.04, n=14, mat=M("mesh_dark"), bevel=False)


# ---- base layers --------------------------------------------------------------
def feet():
    with Part("feet", M("rubber")):
        bm = bmesh.new()
        for x, y in ((-122, -84), (122, -84), (-122, 78), (122, 78)):
            bm_prism(bm, circle(5.0, x, y, 28), -0.6, 0.0)
        to_obj("feet", bm, width=0.3)


BOSSES = [(x, y) for x in (-140, -70, 0, 70, 140) for y in (-96, 96)] + [(-144, 0), (144, 0)]


def bottom_cover():
    """Bottom case seen from above: a grey aluminium tray with an inner rim, ribs, screw bosses and a fine vent field."""
    with Part("bottom_cover", M("alu_shell"), bevel=(0.6, 3)):
        R = L.CORNER[0]
        prism("bottom_plate", rrect(L.W, L.D, R), 0.0, 0.9)
        bm = bmesh.new()
        bm_prism(bm, rrect(L.W - 9.4, L.D - 9.4, R - 4.7), 0.5, 4.0)
        prism("inner_rim", rrect(L.W - 6.6, L.D - 6.6, R - 3.3), 0.9, 3.0, cuts=[cutter(bm, "pocket")])
        bm = bmesh.new()
        for x, y, w, d in ((0, -4, 262, 1.4), (-66, -55, 1.2, 74), (66, -55, 1.2, 74), (0, -55, 1.2, 74), (-60, 52, 1.2, 70)):
            bm_box(bm, x, y, 0.9, w, d, 0.7, 0.3)
        to_obj("ribs", bm, width=0.15, seg=2)
        bm = bmesh.new()
        for x, y in BOSSES:
            bm_prism(bm, circle(2.7, x, y, 24), 0.9, 3.0)
        to_obj("bosses", bm, width=0.2, seg=2)
        bm = bmesh.new()
        for x, y in L.BOARD_SCREWS:                                   # standoffs under the board
            bm_prism(bm, circle(2.4, x, y, 20), 0.9, L.BOARD_Z0)
        to_obj("standoffs", bm, mat=M("steel_dark"), width=0.15, seg=2)
        bm = bmesh.new()
        for i in range(84):                                           # perforated vent field under the board
            for j in range(10):
                bm_prism(bm, circle(0.55, -100 + i * 2.4 + (1.2 if j % 2 else 0), 66 + j * 2.2, 8), 0.86, 0.93)
        to_obj("vents", bm, mat=M("mesh_dark"), bevel=False, smooth=False)
        screws(BOSSES, 3.0, r=1.4, h=0.7)


def _wall_cutter(side, y, z, w, h, r, depth=7.0):
    bm = bmesh.new()
    bm_prism(bm, rrect(w, h, r, y, z), 0, depth)                      # profile in (y,z), extruded along +Z then turned
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(-math.pi / 2, 3, "Y"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(side * (L.W / 2 + 1.0), 0, 0))
    return cutter(bm, "port")


def top_case():
    """Unibody top case: grey aluminium with a cut-out per key, the trackpad in its opening, hinge brackets at the rear corners."""
    with Part("top_case", M("alu_grey"), bevel=L.DECK_BEVEL[0]):
        R = L.CORNER[0]
        cuts = []
        bm = bmesh.new()
        bm_prism(bm, rrect(L.W - 4.4, L.D - 4.4, R - 2.2), 0.6, L.DECK_TOP - 1.0)
        cuts.append(cutter(bm, "pocket"))
        bm = bmesh.new()
        for k in L.keys():
            bm_prism(bm, rrect(k["w"] + 0.9, k["h"] + 0.9, 2.0, k["x"], k["y"], 4), 5.5, 8.0)
        cuts.append(cutter(bm, "kb_well"))
        tx, ty, tw, td = L.TRACKPAD
        bm = bmesh.new()
        bm_prism(bm, rrect(tw + 1.0, td + 1.0, 5.4, tx, ty), 5.0, 8.0)
        cuts.append(cutter(bm, "tp_well"))
        for side, ys in ((-1, (-28, -46)), (1, (-28,))):
            for y in ys:
                cuts.append(_wall_cutter(side, y, 3.4, 8.6, 3.0, 1.5))
        prism("deck", rrect(L.W, L.D, R), 0.9, L.DECK_TOP, cuts=cuts)
        prism("trackpad", rrect(tw, td, 5.0, tx, ty), 6.2, L.DECK_TOP - 0.15, mat=M("trackpad_glass"), width=0.3, seg=3)
        bm = bmesh.new()
        for sx in (-1, 1):                                            # hinge brackets: thin dark plates at the rear corners
            bm_prism(bm, rrect(30, 6.5, 1.2, sx * 113, 99.0), L.DECK_TOP, L.DECK_TOP + 0.35)
        to_obj("brackets", bm, mat=M("steel_dark"), width=0.12, seg=2)
        kx, ky, kw, kd = L.KB_WELL
        prism("kb_backer", rrect(kw + 2, kd + 2, 2.5, kx, ky), KB_PLATE_Z - 0.04, KB_PLATE_Z, mat=M("alu_grey_dark"), bevel=False)
        bm = bmesh.new()
        for x in (-125, -60, 0, 60, 125):
            for y in (-18, 96):
                bm_prism(bm, circle(2.2, x, y, 16), 5.4, 6.4)
        to_obj("bosses", bm, width=0.15, seg=2)
        screws([(sx * x, 99.0) for sx in (-1, 1) for x in (102, 113, 124)], L.DECK_TOP + 0.35, r=1.1, h=0.45)


KB_PLATE_Z = 5.6   # keyboard plate underside; everything under the keyboard (pipes, screws) stays below the top case's backer


def keyboard():
    """Black keycaps with real gaps on a dark backing plate; the power key at top right carries a thin steel ring."""
    kx, ky, kw, kd = L.KB_WELL
    with Part("keyboard", M("keycap"), bevel=(0.32, 3)):
        prism("kb_plate", rrect(kw + 4, kd + 4, 2.5, kx, ky), KB_PLATE_Z, KB_PLATE_Z + 0.28, mat=M("alu_grey_dark"), width=0.2, seg=2)
        box("membrane", kx, ky, KB_PLATE_Z + 0.28, kw - 1, kd - 1, 0.1, 2.0, mat=M("kb_membrane"), bevel=False)
        bm = bmesh.new()
        z0, z1 = L.DECK_TOP - 1.4, L.DECK_TOP - 0.3
        for k in L.keys():
            w, h = k["w"], k["h"]
            bm_prism(bm, rrect(w, h, 1.7, k["x"], k["y"], 4), z0, z1, top_pts=rrect(w - 0.4, h - 0.4, 1.5, k["x"], k["y"], 4))
        to_obj("keycaps", bm, uv=(L.KB_X0, L.KB_Y0, L.KB_X1 - L.KB_X0, L.KB_Y1 - L.KB_Y0), angle=50)
        fn_y = max(k["y"] for k in L.keys())
        pk = max((k for k in L.keys() if abs(k["y"] - fn_y) < 2), key=lambda k: k["x"])
        r = min(pk["w"], pk["h"]) / 2 - 1.2
        cyl("power_ring", pk["x"], pk["y"], z1, r, 0.05, n=40, mat=M("steel"), bevel=False)
        cyl("power_face", pk["x"], pk["y"], z1, r - 0.5, 0.08, n=40, mat=M("black_glossy"), bevel=False)


def hinges():
    """Hidden clutch hinge: one long bar and two mounts inside the rear of the deck; nothing shows outside the case."""
    with Part("hinges", M("steel_dark"), bevel=(0.25, 2)):
        cyl("clutch_bar", 0, L.HINGE_Y - 1.0, L.DECK_TOP - 2.6, 1.6, 236, axis="X", n=24, mat=M("alu_grey_dark"))
        for sx in (-1, 1):
            box("mount", sx * 112, L.HINGE_Y - 4.0, 4.2, 14, 6, 1.4, 0.8)


_PIVOT = [None]


# ---- internals ----------------------------------------------------------------
def battery():
    """Matte black pouch cells in a dark tray, thin dividers between them, a flat flex and a small wire lead."""
    tw = L.BAT_N * L.BAT_W + (L.BAT_N - 1) * L.BAT_GAP + 5.0
    td = L.BAT_D + 5.0
    with Part("battery", M("cell_black"), bevel=(0.6, 3)):
        bm = bmesh.new()
        bm_prism(bm, rrect(tw - 2.4, td - 2.4, 2.4, 0, L.BAT_Y), 1.5, 7.0)
        prism("tray", rrect(tw, td, 3.4, 0, L.BAT_Y), 0.9, 4.9, mat=M("tray_dark"), cuts=[cutter(bm, "pocket")], width=0.3, seg=2)
        bm = bmesh.new()
        for i in range(1, L.BAT_N):                                    # dividers between the cells
            x = (i - L.BAT_N / 2) * (L.BAT_W + L.BAT_GAP)
            bm_box(bm, x, L.BAT_Y, 1.5, 1.2, L.BAT_D, 3.2, 0.3)
        for sx in (-1, 1):                                             # screw ears at the tray ends
            for dy in (-28, 28):
                bm_box(bm, sx * (tw / 2 + 3.0), L.BAT_Y + dy, 0.9, 8.0, 9.0, 1.0, 2.0)
        to_obj("dividers", bm, mat=M("tray_dark"), width=0.15, seg=2)
        bm = bmesh.new()
        for cx, cy, w, d in L.BAT_CELLS:
            bm_prism(bm, rrect(w - 1.6, d - 1.6, 2.4, cx, cy), 1.5, 4.3, top_pts=rrect(w - 3.4, d - 3.4, 1.6, cx, cy))
        to_obj("cells", bm, width=0.5, seg=3)
        bm = bmesh.new()
        for cx, cy, w, d in L.BAT_CELLS:                               # faint blank label panel on each cell
            bm_prism(bm, rrect(w - 16, d * 0.42, 1.0, cx, cy + d * 0.18), 4.3, 4.34)
        to_obj("cell_panels", bm, mat=M("cell_label"), bevel=False)
        fy = L.BAT_Y + td / 2
        bm = bmesh.new()
        bm_sweep(bm, fillet_path([(0, fy - 8, 4.4), (0, fy + 2, 4.4), (22, fy + 8, 3.4), (30, L.BOARD_Y0 + 3.5, 3.4)], 4, 6), flat_profile(9, 0.16, 0.06, 1))
        to_obj("flex", bm, mat=M("flex_dark"), bevel=False)
        box("flex_stiffener", 30, L.BOARD_Y0 + 2.5, 3.3, 10, 4, 0.4, 0.4, mat=M("gold"), width=0.08, seg=1)
        box("bms_board", 0, fy - 6, 4.3, 26, 7, 0.6, 0.6, mat=M("pcb_io"), uv=(-13, fy - 9.5, 26, 7), width=0.1, seg=1)
        for k, (m, dx) in enumerate((("wire_red", -1.2), ("wire_black", 0.0), ("wire_yellow", 1.2))):
            bm = bmesh.new()
            x0 = -tw / 2 + 12
            bm_sweep(bm, fillet_path([(x0 + dx, fy - 3, 4.2), (x0 + dx, fy + 3, 4.2), (x0 - 10 + dx, fy + 7, 3.6)], 3, 5), circle(0.42, 0, 0, 8))
            to_obj("lead%d" % k, bm, mat=M(m), bevel=False)
        box("lead_plug", -tw / 2 + 1.0, fy + 7, 2.8, 5, 4.2, 1.6, 0.4, mat=M("zif"), width=0.1, seg=1)
        screws([(sx * (tw / 2 + 3.0), L.BAT_Y + dy) for sx in (-1, 1) for dy in (-28, 28)], 1.9, r=1.3, h=0.6)


def _usb_port(side, x, y):
    box("usbc", x, y, 1.6, 8.0, 8.4, 3.0, 1.2, mat=M("steel"), width=0.15, seg=2)
    box("mouth", x + side * 4.1, y, 2.2, 0.4, 7.0, 1.8, 0.7, mat=M("mesh_dark"), bevel=False)
    box("port_pcb", x - side * 6.0, y, 0.9, 12.0, 9.0, 0.7, 0.5, mat=M("pcb_io"), uv=(x - side * 6 - 6, y - 4.5, 12, 9), width=0.1, seg=1)


def speakers():
    """Two black speaker modules at the sides, with a row of small parts between them: ports, a jack, mics, a transfer board and flex."""
    ry = -5.0                                                         # the small parts' row, between the battery and the board
    with Part("speakers", M("black_plastic"), bevel=(0.5, 3)):
        for cx, cy, w, d in L.SPEAKERS:
            sgn = 1 if cx < 0 else -1
            prism("enclosure", rrect(w, d, 6.0, cx, cy, 8), 0.9, 5.4, top_pts=rrect(w - 2.0, d - 2.0, 5.0, cx, cy, 8))
            box("port_snout", cx + sgn * (w / 2 + 2.0), cy + d / 2 - 14, 1.4, 5.0, 12.0, 3.2, 1.5)
            bm = bmesh.new()
            for i in range(5):
                for j in range(22):
                    bm_prism(bm, circle(0.5, cx - 2.4 + i * 1.2, cy - d / 2 + 14 + j * 2.4 + (0.6 if i % 2 else 0), 8), 5.38, 5.44)
            to_obj("grille", bm, mat=M("mesh_dark"), bevel=False, smooth=False)
            box("foam", cx, cy - d / 2 + 6, 5.4, w - 6, 6, 0.6, 0.8, mat=M("foam"), bevel=False)
            for ey in (cy - d / 2 - 3.0, cy + d / 2 + 3.0):          # steel mounting ears at both ends
                box("ear", cx, ey, 0.9, 9.0, 7.0, 0.5, 2.0, mat=M("steel_dark"), width=0.1, seg=2)
            bm = bmesh.new()
            bm_sweep(bm, fillet_path([(cx, cy - d / 2 + 2, 3.0), (cx, ry + 4, 3.0), (cx + sgn * 14, ry + 1, 1.6)], 4, 5), flat_profile(3.5, 0.14, 0.05, 1))
            to_obj("lead", bm, mat=M("flex_dark"), bevel=False)
        for side in (-1, 1):
            _usb_port(side, side * 143.0, ry)
            box("mic", side * 82.0, ry, 0.9, 6.0, 4.0, 1.6, 0.8, mat=M("rubber"), width=0.15, seg=2)
            box("mic_bracket", side * 92.0, ry, 0.9, 10.0, 5.0, 0.4, 1.0, mat=M("steel"), width=0.08, seg=1)
            bm = bmesh.new()
            bm_sweep(bm, [(side * 24, ry, 1.3), (side * 68, ry, 1.3)], flat_profile(6.0, 0.14, 0.05, 1))
            to_obj("flex", bm, mat=M("polyimide"), bevel=False)
        box("jack", -118.0, ry, 0.9, 9.0, 7.0, 3.4, 0.9)
        cyl("jack_barrel", -123.5, ry, 2.6, 2.2, 3.0, axis="X", n=20, mat=M("steel"), width=0.12, seg=2)
        box("transfer_pcb", 0, ry, 0.9, 46.0, 10.0, 0.8, 1.0, mat=M("pcb_io"), uv=(-23, ry - 5, 46, 10), width=0.1, seg=1)
        for x, w in ((-10.0, 18.0), (12.0, 12.0)):                    # ivory ZIF connectors with fine pins
            box("zif", x, ry + 1.2, 1.7, w, 3.6, 1.0, 0.3, mat=M("zif"), width=0.1, seg=1)
            box("zif_slot", x, ry + 1.2, 2.7, w - 2.4, 1.4, 0.12, 0.2, mat=M("connector"), bevel=False)
            bm = bmesh.new()
            for k in range(int((w - 2.4) / 0.5)):
                bm_prism(bm, rrect(0.2, 0.9, 0.0, x - w / 2 + 1.4 + k * 0.5, ry - 1.0), 1.7, 1.85)
            to_obj("zif_pins", bm, mat=M("gold"), bevel=False, smooth=False)
        bm = bmesh.new()
        for x in (-16, -8, 0, 8, 20):                                  # small SOT chips
            bm_box(bm, x, ry - 3.0, 1.7, 2.6, 1.6, 0.8, 0.2)
        to_obj("sot", bm, mat=M("chip_ctrl"), width=0.05, seg=1)
        screws([(cx, cy + e * (d / 2 + 3.0)) for cx, cy, w, d in L.SPEAKERS for e in (-1, 1)] + [(-20.0, ry), (20.0, ry)], 1.4, r=1.1, h=0.45)


def _zif(name, x, y, w, d, z):
    """Gold FPC connector: ivory body, dark slot, fine gold pins along the long side."""
    long_x = w > d
    box(name, x, y, z, w, d, 1.0, 0.3, mat=M("zif"), width=0.1, seg=1)
    box(name + "_slot", x, y, z + 1.0, w - 1.6 if long_x else d * 0.4, d * 0.4 if long_x else d - 1.6, 0.12, 0.2, mat=M("connector"), bevel=False)
    bm = bmesh.new()
    n = int(((w if long_x else d) - 2.0) / 0.5)
    for k in range(n):
        if long_x:
            bm_prism(bm, rrect(0.22, 0.9, 0.0, x - w / 2 + 1.2 + k * 0.5, y - d / 2 - 0.4), z, z + 0.16)
        else:
            bm_prism(bm, rrect(0.9, 0.22, 0.0, x - w / 2 - 0.4, y - d / 2 + 1.2 + k * 0.5), z, z + 0.16)
    to_obj(name + "_pins", bm, mat=M("gold"), bevel=False, smooth=False)


def logic_board():
    """Dense near-black board: a big SoC under a raised metal frame and lid, memory beside it, shield cans, gold FPC connectors."""
    holes = []
    with Part("logic_board", M("pcb_main"), bevel=(0.12, 2)):
        for x, y in L.BOARD_SCREWS:
            bm = bmesh.new()
            bm_prism(bm, circle(1.7, x, y, 16), 0.5, 4.0)
            holes.append(cutter(bm, "mount_hole"))
        prism("pcb", L.BOARD_POLY, L.BOARD_Z0, L.BOARD_TOP, uv=(L.BOARD_X0, L.BOARD_Y0, L.BOARD_X1 - L.BOARD_X0, L.BOARD_Y1 - L.BOARD_Y0),
              cuts=holes, bevel=True)
        sx, sy, ss = L.SOC
        lid = L.B["soc_lid"]
        box("soc_package", sx, sy, L.BOARD_TOP, ss, ss, 0.6, 0.6, mat=M("pcb_edge"), width=0.15, seg=2)
        bm = bmesh.new()                                               # raised shield frame round the lid
        fw = (ss - lid) / 2 - 0.6
        for ex in (-1, 1):
            bm_box(bm, sx + ex * (lid / 2 + fw / 2 + 0.3), sy, L.BOARD_TOP + 0.6, fw, ss - 1.2, 0.9, 0.3)
            bm_box(bm, sx, sy + ex * (lid / 2 + fw / 2 + 0.3), L.BOARD_TOP + 0.6, lid + 0.6, fw, 0.9, 0.3)
        to_obj("soc_frame", bm, mat=M("steel"), width=0.12, seg=2)
        box("soc_lid", sx, sy, L.BOARD_TOP + 0.6, lid, lid, 0.7, 0.8, mat=M("shield"), width=0.15, seg=2)
        for i, (x, y) in enumerate(L.RAM):
            chip("ram%d" % i, x, y, L.BOARD_TOP, L.RAM_SIZE[0], L.RAM_SIZE[1], 0.9, "chip_ram")
        for i, (x, y, w, d, h, t) in enumerate(L.ICS):
            chip("ic%d" % i, x, y, L.BOARD_TOP, w, d, h, "chip_" + t, r=0.3)
        for i, (x, y, w, d, h) in enumerate(L.SHIELDS):
            shield("shield%d" % i, x, y, L.BOARD_TOP, w, d, h)
        for i, (x, y, w, d) in enumerate(L.CONNECTORS):
            _zif("conn%d" % i, x, y, w, d, L.BOARD_TOP)
        bm = bmesh.new()
        for x, y in L.POLY_CAPS:
            bm_prism(bm, rrect(L.B["tantalum_size"][0], L.B["tantalum_size"][1], 0.7, x, y), L.BOARD_TOP, L.BOARD_TOP + 1.4)
        to_obj("poly_caps", bm, mat=M("polycap"), width=0.2, seg=2)
        bm = bmesh.new()
        for x, y in L.INDUCTORS:
            bm_prism(bm, rrect(L.B["inductor_size"], L.B["inductor_size"], 0.5, x, y), L.BOARD_TOP, L.BOARD_TOP + L.B["inductor_h"])
        to_obj("inductors", bm, mat=M("inductor"), width=0.2, seg=2)
        bc, br, be = bmesh.new(), bmesh.new(), bmesh.new()
        for x, y, w, d, h, kind in L.passives():                       # 0402 bodies with metal end caps
            long_x = w >= d
            cap = min(w, d) * 0.45 if long_x else 0
            body = rrect(w - 2 * cap if long_x else w, d if long_x else d - 2 * min(w, d) * 0.45, 0.0, x, y)
            bm_prism(bc if kind == "cap" else br, body, L.BOARD_TOP, L.BOARD_TOP + h)
            for e in (-1, 1):
                if long_x:
                    bm_prism(be, rrect(cap, d, 0.0, x + e * (w / 2 - cap / 2), y), L.BOARD_TOP, L.BOARD_TOP + h * 1.02)
                else:
                    inset = min(w, d) * 0.45
                    bm_prism(be, rrect(w, inset, 0.0, x, y + e * (d / 2 - inset / 2)), L.BOARD_TOP, L.BOARD_TOP + h * 1.02)
        to_obj("mlcc", bc, mat=M("cap"), width=0.04, seg=1)
        to_obj("resistors", br, mat=M("res"), width=0.04, seg=1)
        to_obj("end_caps", be, mat=M("steel"), bevel=False, smooth=False)
        box("m2_socket", L.SSD[0] + L.SSD[2] / 2 + 2.0, L.SSD[1], L.BOARD_TOP, 4.0, L.SSD[3] + 2, 1.6, 0.4, mat=M("connector"), width=0.1, seg=1)
        screws(L.BOARD_SCREWS, L.BOARD_TOP, r=1.5, h=0.55)
    ssd()
    wifi()


def ssd():
    """M.2 card: matte black board, blank label, a controller under a metal lid, scattered passives, gold edge fingers, a domed screw."""
    cx, cy, ln, wd = L.SSD
    z0 = L.BOARD_TOP + 0.5
    with Part("logic_board", M("pcb_ssd"), bevel=(0.12, 2)):
        box("ssd_pcb", cx, cy, z0, ln, wd, 0.8, 0.5, mat=M("pcb_ssd"), uv=(cx - ln / 2, cy - wd / 2, ln, wd), width=0.1, seg=1)
        sb = L.B["ssd"]
        nw, nd, nh = sb["nand"]
        for k in range(sb["nand_count"]):
            chip("ssd_nand", cx - ln / 2 + 8 + nw / 2 + k * (nw + 3), cy, z0 + 0.8, nw, nd, nh, "chip_nand", r=0.3)
        box("ssd_label", cx - ln / 2 + 6 + sb["sticker"][0] / 2, cy, z0 + 0.8 + nh, sb["sticker"][0], sb["sticker"][1], 0.1, 0.8, mat=M("sticker"), bevel=False)
        cw, cd, ch = sb["controller"]
        box("ssd_ctrl", cx + ln / 2 - 8 - cw / 2, cy, z0 + 0.8, cw, cd, ch, 0.5, mat=M("shield"), width=0.12, seg=2)
        dw, dd, dh = sb["dram"]
        chip("ssd_dram", cx + ln / 2 - 10 - cw - dw / 2, cy - 3, z0 + 0.8, dw, dd, dh, "chip_ram", r=0.3)
        rnd = random.Random(21)
        bc = bmesh.new()
        for _ in range(16):
            x, y = cx + rnd.uniform(-2, ln / 2 - 18), cy + rnd.uniform(-wd / 2 + 2, wd / 2 - 2)
            bm_prism(bc, rrect(1.1, 0.6, 0.0, x, y), z0 + 0.8, z0 + 1.25)
        to_obj("ssd_passives", bc, mat=M("cap"), width=0.04, seg=1)
        screws([(cx - ln / 2 + 3.5, cy)], z0 + 0.8, r=1.6, h=0.7)


def wifi():
    cx, cy, ln, wd = L.WIFI
    z0 = L.BOARD_TOP + 0.5
    with Part("logic_board", M("pcb_wifi"), bevel=(0.12, 2)):
        box("wifi_pcb", cx, cy, z0, ln, wd, 0.8, 0.5, mat=M("pcb_wifi"), uv=(cx - ln / 2, cy - wd / 2, ln, wd), width=0.1, seg=1)
        sw, sd, sh = L.B["wifi"]["shield"]
        shield("wifi_shield", cx + 3, cy, z0 + 0.8, sw, sd, sh)
        for k, (dy, m) in enumerate(((5, "wire_grey"), (-5, "wire_black"))):
            box("ufl", cx - 9, cy + dy, z0 + 0.8, 2.2, 2.2, 0.7, 0.4, mat=M("steel"), bevel=False)
            bm = bmesh.new()
            sgn = 1 if dy > 0 else -1
            bm_sweep(bm, fillet_path([(cx - 9, cy + dy, z0 + 1.2), (cx - 12, cy + dy + sgn * 3, z0 + 1.2), (cx - 12, cy + dy + sgn * 14, z0 + 1.2)], 3, 5), circle(0.5, 0, 0, 8))
            to_obj("coax%d" % k, bm, mat=M(m), bevel=False)
        screws([(cx + ln / 2 - 3, cy)], z0 + 0.8, r=1.3, h=0.5)


def _fan_poly(cx, cy, r, half_w, y_end, n=60):
    a = math.degrees(math.asin(half_w / r))
    pts = []
    for i in range(n + 1):
        t = math.radians(90 + a + (360 - 2 * a) * i / n)
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    pts += [(cx + half_w, y_end), (cx - half_w, y_end)]
    return pts


def cooling():
    """Heat sink and cooling: a dark plate over the SoC, flat copper heat pipes to a copper fin stack, one blower fan with an open impeller."""
    fcx, fcy, r, y_end = L.FAN
    fx, fy, fl = L.FINS
    sx, sy, _ = L.SOC
    z = 4.6                                                           # plate underside, on the SoC lid
    with Part("cooling", M("sink_dark"), bevel=(0.3, 3)):
        plate = [(-112, 92), (-34, 92), (-24, 76), (sx + 20, 76), (sx + 20, 36), (-24, 36), (-34, 44), (-112, 44)]
        prism("plate", plate, z, z + 0.5, width=0.3, seg=3)
        box("spreader", sx, sy, z - 0.3, 26, 26, 0.9, 1.5, mat=M("copper"), width=0.2, seg=2)
        bm = bmesh.new()
        for x, y in ((-112, 92), (-112, 44), (-34, 92)):                # stamped tabs with screw holes
            bm_box(bm, x + (6 if x < -60 else 0), y + (4 if y > 60 else -4), z, 10, 8, 0.5, 2.0)
        to_obj("tabs", bm, width=0.2, seg=2)
        bm = bmesh.new()
        for a in (45, -45):
            bm_prism(bm, _rot(rrect(52, 5.0, 1.6, sx, sy), a, sx, sy), z + 0.5, z + 0.9)
        to_obj("retainer", bm, mat=M("steel_dark"), width=0.12, seg=2)
        for k, dy in enumerate((-5.0, 5.0)):                            # two flat heat pipes, plate to fin stack
            zp = z + 0.25                                              # pipe centre: its top stays under the keyboard backer
            path = fillet_path([(-100, 68 + dy, zp), (-30, 68 + dy, zp), (sx + 4, sy + dy, zp), (30, sy + dy, zp),
                                (52, fy + dy * 0.5, zp), (fx + fl / 2, fy + dy * 0.5, zp)], 14, 10)
            bm = bmesh.new()
            bm_sweep(bm, path, flat_profile(6.4, 1.3, 0.6, 3))
            to_obj("heat_pipe%d" % k, bm, mat=M("copper"), width=0.1, seg=2)
        bm = bmesh.new()
        n = int(fl / 0.9)
        for i in range(n):
            bm_prism(bm, rrect(0.2, 9.0, 0.0, fx - fl / 2 + i * 0.9, fy), L.BOARD_TOP + 0.2, z + 0.6)
        to_obj("fins", bm, mat=M("copper_fin"), smooth=False, width=0.05, seg=1)
        box("fin_rail", fx, fy, z + 0.6, fl + 1, 9.6, 0.3, 0.4, mat=M("copper_fin"), bevel=False)
        bm = bmesh.new()
        bm_prism(bm, _fan_poly(fcx, fcy, r - 1.2, 11.0, y_end + 2, 56), 1.4, z + 0.2)
        cav = cutter(bm, "fan_cavity")
        bm = bmesh.new()
        bm_prism(bm, circle(15.5, fcx, fcy, 48), z - 0.6, z + 1.2)
        hole = cutter(bm, "fan_intake")
        prism("housing", _fan_poly(fcx, fcy, r, 12.4, y_end, 64), 1.0, z + 0.6, cuts=[cav, hole])
        for ang in (200, 340):
            box("ear", fcx + (r + 1.5) * math.cos(math.radians(ang)), fcy + (r + 1.5) * math.sin(math.radians(ang)), 1.0, 8, 8, 0.8, 3.0)
        cyl("hub", fcx, fcy, 1.4, 7.0, z - 1.4, n=40, mat=M("fan_blade"))
        cyl("hub_cap", fcx, fcy, z - 0.05, 4.4, 0.2, n=32, mat=M("steel_dark"), bevel=False)
        bm = bmesh.new()
        nb = 43
        for i in range(nb):                                            # fine forward-curved impeller blades
            phi = 2 * math.pi * i / nb
            left, right = [], []
            for k in range(9):
                t = k / 8
                rr = 7.4 + (r - 3.8 - 7.4) * t
                th = phi + math.radians(34 * t * t + 6 * t)
                px, py = fcx + rr * math.cos(th), fcy + rr * math.sin(th)
                nx, ny = math.cos(th + math.pi / 2 - math.radians(36 * t)), math.sin(th + math.pi / 2 - math.radians(36 * t))
                left.append((px + nx * 0.12, py + ny * 0.12))
                right.append((px - nx * 0.12, py - ny * 0.12))
            bm_prism(bm, left + right[::-1], 1.6, z - 0.5)
        to_obj("blades", bm, mat=M("fan_blade"), width=0.04, seg=1, angle=60)
        screws([(-106, 96), (-106, 40), (-34, 96), (fcx + (r + 1.5) * math.cos(math.radians(200)), fcy + (r + 1.5) * math.sin(math.radians(200))),
                (fcx + (r + 1.5) * math.cos(math.radians(340)), fcy + (r + 1.5) * math.sin(math.radians(340)))]
               + [(sx + 23 * math.cos(math.radians(a)), sy + 23 * math.sin(math.radians(a))) for a in (45, 135, 225, 315)], z + 0.5, r=1.3, h=0.45)


def lid_back():
    """3.6 mm aluminium lid: a shallow tray, open on the screen side, with a 0.8 mm rim round the glass."""
    R = L.CORNER[0]
    with Part("lid_back", M("alu_shell"), lid_pivot=_PIVOT[0], bevel=L.DECK_BEVEL[0]):
        bm = bmesh.new()
        bm_prism(bm, rrect(L.W - 1.6, L.D - 1.6, R - 0.8), L.LID_Z0 - 0.1, L.LID_Z1 - 0.7)
        prism("lid_shell", rrect(L.W, L.D, R), L.LID_Z0, L.LID_Z1, cuts=[cutter(bm, "lid_pocket")])


def _active():
    """(cx, cy, w, d) of the screen's active area: the glass minus its side, top and chin borders (chin at +Y, the hinge)."""
    gw, gd, gcy = L.GLASS
    sd, tp, ch = L.BEZEL
    return 0.0, gcy + (tp - ch) / 2, gw - 2 * sd, gd - tp - ch


def display_panel():
    """The display module, lid side up, as one sealed part (cover glass bonded on the front):
    the dark LCD cell glass, and on it the backlight unit (films inside) in a stamped steel tray with bevelled walls.

    The cell's driver flex folds off the chin edge onto a slim T-con board on the steel back."""
    gw, gd, gcy = L.GLASS
    ax, ay, aw, ad = _active()
    D = mats.DISPLAY
    z = L.LID_Z0 + D["glass_mm"] + 0.1                                # just behind the cover glass
    zt = z + D["cell_mm"]                                             # back face of the cell
    zb = zt + D["backlight_mm"]                                       # steel back of the backlight unit
    with Part("display_panel", M("cell_glass"), lid_pivot=_PIVOT[0], bevel=(0.2, 2)):
        prism("lcd_image", rrect(296, 204, 6.0), z - 0.02, z, mat=M("screen"), uv=(-gw / 2, gcy - gd / 2, gw, gd, False, True), bevel=False, smooth=False)
        prism("lcd_cell", rrect(296, 204, 6.0), z, zt, mat=M("cell_glass"))
        bm = bmesh.new()
        for i in range(6):                                            # source drivers on the chin ledge, left bare by the tray
            bm_box(bm, -110 + i * 44, 98.6, zt, 14, 1.4, 0.25, 0.1)
        to_obj("drivers", bm, mat=M("chip_ctrl"), bevel=False)
        fy, fd = -3.0, 198.0                                          # the tray stops short of the chin, which holds the drivers
        prism("backlight_unit", rrect(294, fd, 5.0, 0, fy), zt, zb, mat=M("panel_steel"), width=D["frame_bevel_mm"], seg=3)
        box("back_rib", 0, fy - 20, zb - 0.02, 220, 90, 0.3, 6.0, mat=M("panel_steel"), width=0.15, seg=2)  # stamped stiffening panel
        box("back_label", -90, -60, zb + 0.28, 40, 18, 0.05, 1.0, mat=M("sticker"), bevel=False, smooth=False)
        zf = zb + 0.02
        bm = bmesh.new()                                              # main flex: off the chin edge, folded back over onto the T-con
        bm_sweep(bm, fillet_path([(18, 97.5, zt + 0.06), (18, 103.2, zt + 0.06), (18, 104.6, zt + 0.35), (18, 103.2, zf), (18, 94, zf)], 1.2, 6),
                 flat_profile(30, 0.12, 0.05, 1))
        to_obj("flex", bm, mat=M("fpc"), bevel=False)
        box("tcon_pcb", 18, 92.0, zf + 0.06, 50, 7, 0.4, 0.4, mat=M("pcb_io"), uv=(-7, 88.5, 50, 7), width=0.08, seg=1)
        box("tcon_chip", 8, 92, zf + 0.46, 6, 5, 0.25, 0.3, mat=M("chip_ctrl"), uv=(5, 89.5, 6, 5), width=0.08, seg=1)
        box("tcon_conn", 34, 92, zf + 0.46, 12, 3.5, 0.3, 0.3, mat=M("connector"), width=0.08, seg=1)

        # cover glass bonded on the front: one solid module, polished edge, black ink bezel
        z0, z1 = L.LID_Z0 + 0.05, L.LID_Z0 + 0.05 + D["glass_mm"]
        R = L.CORNER[0] - 1.2
        prism("cover_glass", rrect(gw - 1.2, gd - 1.2, R - 0.6, 0, gcy), z0, z1, mat=M("cover_glass"), width=0.3, seg=3)
        bm = bmesh.new()
        bm_prism(bm, rrect(gw - 1.2, gd - 1.2, R - 0.6, 0, gcy), z0 - 0.1, z1 + 0.1)
        prism("glass_edge", rrect(gw, gd, R, 0, gcy), z0, z1, mat=M("glass_edge"), cuts=[cutter(bm, "cutter")], width=0.3, seg=3)
        bm = bmesh.new()
        bm_prism(bm, rrect(aw, ad, 0.6, ax, ay), z1 - 0.5, z1 + 0.5)
        prism("glass_ink", rrect(gw - 0.4, gd - 0.4, L.CORNER[0] - 1.4, 0, gcy), z1, z1 + 0.015, mat=M("glass_ink"),
              cuts=[cutter(bm, "cutter")], bevel=False, smooth=False)
        cyl("camera_lens", 0, gcy - gd / 2 + L.BEZEL[1] / 2, z1 + 0.015, 0.9, 0.06, n=16, mat=M("black_glossy"), bevel=False)


BUILD = [feet, bottom_cover, top_case, keyboard, battery, speakers, logic_board, cooling, hinges, lid_back, display_panel]


def build_all():
    for fn in BUILD:
        try:
            fn()
        except Exception:
            ERRORS.append((fn.__name__, traceback.format_exc()))
            print("PART_ERROR", fn.__name__)
            print(traceback.format_exc())
    return ERRORS
