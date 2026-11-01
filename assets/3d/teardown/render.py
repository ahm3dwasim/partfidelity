"""Blender entry point (run inside Modal): blender -b --factory-startup -P render.py -- <job.json> <out_dir>

job.json = {"cfg": <validated teardown.toml>, "gpu": bool, "tex_dir": str, "jobs": [{"name", "pose", "res", "samples"}], "anchors": bool}
Builds the scene once, then poses and renders every job. Writes <name>.png plus report.json. With "anchors" it renders
nothing and writes anchors.json instead: per frame, where each part's outer corners land in the picture (film labels).
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy
from mathutils import Vector

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = sys.argv[sys.argv.index("--") + 2]
os.makedirs(OUT, exist_ok=True)
CFG = job["cfg"]

import geom as G
import mats
import parts
import env
import pose as P

import layout as layout_cfg
layout_cfg.JITTER[0] = CFG["scene"]["key_jitter_mm"]
layout_cfg.configure(CFG["board"])
layout_cfg.HINGE_Y, layout_cfg.HINGE_Z = CFG["scene"]["pivot_yz"]
layout_cfg.CORNER[0] = CFG["scene"]["corner_mm"]
layout_cfg.DECK_BEVEL[0] = tuple(CFG["scene"]["deck_bevel"])
mats.MAT.update(CFG["material"])
G.CUR["dish"] = CFG["scene"]["key_dish_mm"]
mats.TEX_DIR = job["tex_dir"]
mats.WEAR[0] = CFG["scene"]["wear"]
mats.BOARD.update(CFG["board"])
mats.GLASS[:] = [CFG["scene"]["glass_rough"], CFG["scene"]["glass_spec"]]
mats.DISPLAY.update(CFG["display"])
G.CUR["bevel"] = (CFG["scene"]["bevel_mm"], CFG["scene"]["bevel_segments"])
report = {"errors": [], "renders": []}
t0 = time.time()

bpy.ops.wm.read_factory_settings(use_empty=True)
env.world(CFG)
env.cycles(CFG, gpu=job["gpu"])
env.sweep(CFG)
env.lights(CFG)
cam = env.camera(CFG)
import layout as L

pivot = bpy.data.objects.new("lid_pivot", None)
pivot.location = (0, L.HINGE_Y * G.MM, L.HINGE_Z * G.MM)
bpy.context.scene.collection.objects.link(pivot)
parts._PIVOT[0] = pivot
errs = parts.build_all()
report["errors"] = [e[0] for e in errs]
report["build_s"] = round(time.time() - t0, 1)

groups = {}
for o in bpy.data.objects:
    if o.type == "EMPTY" and o.name.startswith("g_"):
        groups.setdefault(o["part"], []).append(o)
screw_groups = {}
for o in bpy.data.objects:
    if o.type == "EMPTY" and o.name.endswith("_screws"):
        screw_groups.setdefault(o["part"], []).append(o)
report["parts"] = sorted(groups)
report["objects"] = len(bpy.data.objects)
report["tris"] = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")

def boolean_report():
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        if o.type == "MESH" and any(m.type == "BOOLEAN" for m in o.modifiers):
            ev = o.evaluated_get(dg)
            tgt = [m.object.name for m in o.modifiers if m.type == "BOOLEAN" and m.object]
            print("BOOL_CHECK", o.name, len(o.data.polygons), len(ev.data.polygons), tgt[:2])


CUT_NAMES = ("cutter", "pocket", "kb_well", "tp_well", "speaker_slots", "port", "jack", "hinge_notch", "mount_hole",
             "fan_cavity", "fan_intake", "ring_hole", "lid_pocket", "lid_notch", "dish_cutter")


def bbox_centres():
    """XY centre (mm) of each part's meshes in the closed, un-exploded pose."""
    out = {}
    for name in CFG["explode"]["offset"]:
        xs, ys = [], []
        for o in bpy.data.objects:
            if o.type == "MESH" and o.get("part") == name and not o.hide_viewport:
                for c in o.bound_box:
                    w = o.matrix_world @ Vector(c)
                    xs.append(w.x / G.MM)
                    ys.append(w.y / G.MM)
        if not xs:
            continue
        out[name] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    return out


_sh = bpy.data.materials.get("alu_shell")
SHELL_BASE = list(_sh.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value) if _sh else [0, 0, 0, 1]
bpy.context.view_layer.update()
CENTRES = bbox_centres()
report["centres"] = {k: [round(v[0], 1), round(v[1], 1)] for k, v in CENTRES.items()}


CENTER = set(CFG["explode"]["center"])
POWERED = [(m, m["power_base"]) for m in bpy.data.materials if "power_base" in m.keys()]
DUST = env.dust(CFG)


def apply_pose(p):
    parts = p["parts"]
    off = P.part_offsets(CFG, parts)
    for name, gs in groups.items():
        cx, cy = CENTRES.get(name, (0.0, 0.0))
        c = parts.get(name, 0.0) if name in CENTER else 0.0
        for g in gs:                       # lid groups live in the pivot's frame, so their z is the lid normal
            g.location = (-cx * c * G.MM, -cy * c * G.MM, (off.get(name, 0.0) + p["bob"].get(name, 0.0)) * G.MM)
    for o in bpy.data.objects:
        if o.type == "LIGHT" and "group" in o.data.keys():
            o.data.energy = o.data["base"] * p[o.data["group"]]
    bpy.context.scene.view_settings.exposure = p["exposure"]
    shell = bpy.data.materials.get("alu_shell")
    if shell is not None:
        node = shell.node_tree.nodes["Principled BSDF"]
        node.inputs["Base Color"].default_value = [v * p["shell"] for v in SHELL_BASE[:3]] + [1.0]
    for m, base in POWERED:                # screen image and key backlight: on while open, asleep once shut
        node = m.node_tree.nodes["Principled BSDF"]
        node.inputs["Emission Strength"].default_value = base * p["power"]
    sw = bpy.data.objects.get("sweep")
    if sw is not None:
        sw.location.z = min(0.0, min(off.values())) * G.MM     # floor follows the lowest part
    sc = CFG["screws"]
    for name, gs in screw_groups.items():  # screws back out and fan outwards from the part's centre, then ride with it
        s_ = p["screws"].get(name, 0.0)
        k = sc["spread"] * s_
        cx, cy = CENTRES.get(name, (0.0, 0.0))
        for g in gs:
            g.scale = (1.0 + k, 1.0 + k, 1.0)
            g.location = (-cx * k * G.MM, -cy * k * G.MM, sc["lift_mm"] * s_ * G.MM)
    for o, base, vel in DUST:              # slow straight drift, no wrap, so no mote ever pops
        o.location = Vector([(b + v * p["frame"]) * G.MM for b, v in zip(base, vel)])
    pivot.rotation_euler = (-math.radians(p["lid_deg"]), 0, 0)
    hide = set(p["hide"])
    if p["explode"] >= CFG["explode"]["hide_from"]:
        hide |= set(CFG["explode"]["hide_parts"])
    for o in bpy.data.objects:
        if o.type == "MESH" and "part" in o.keys():
            is_cut = any(o.name.startswith(c) for c in CUT_NAMES) and o.hide_viewport
            if not is_cut:
                o.hide_render = o["part"] in hide
    c = p["camera"]
    pos = Vector([v * G.MM for v in P.camera_position(c)])
    tgt = Vector([v * G.MM for v in c["target"]])
    cam.location = pos
    cam.rotation_euler = (tgt - pos).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = c["lens"]
    cam.data.dof.aperture_fstop = c["fstop"]
    cam.data.dof.focus_distance = (tgt - pos).length
    bpy.context.view_layer.update()


def anchors():
    """{part: [rx, ry, lx, ly]}: the right-most and left-most projected box corner of the part's visible meshes, as
    fractions of the picture (x from the left, y from the top). static/ui/film.js ends each label's leader line there."""
    from bpy_extras.object_utils import world_to_camera_view
    sc, out = bpy.context.scene, {}
    for name in CFG["explode"]["offset"]:
        pts = []
        for o in bpy.data.objects:
            if o.type == "MESH" and o.get("part") == name and not o.hide_render and not o.hide_viewport:
                for c in o.bound_box:
                    v = world_to_camera_view(sc, cam, o.matrix_world @ Vector(c))
                    pts.append((v.x, 1.0 - v.y))
        if pts:
            r, l = max(pts), min(pts)
            out[name] = [round(r[0], 4), round(r[1], 4), round(l[0], 4), round(l[1], 4)]
    return out


apply_pose(job["jobs"][0]["pose"])
boolean_report()
if job.get("anchors"):
    marks = {}
    for j in job["jobs"]:
        apply_pose(j["pose"])
        s = bpy.context.scene
        s.render.resolution_x, s.render.resolution_y = j["res"]
        marks[str(j["pose"]["frame"])] = anchors()
    json.dump(marks, open(os.path.join(OUT, "anchors.json"), "w"), separators=(",", ":"))
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print("TEARDOWN_DONE", json.dumps({k: report[k] for k in ("errors", "build_s", "objects", "tris")}))
    sys.exit(0)
s = bpy.context.scene
for j in job["jobs"]:
    try:
        t1 = time.time()
        apply_pose(j["pose"])
        s.render.resolution_x, s.render.resolution_y = j["res"]
        s.render.resolution_percentage = 100
        s.cycles.samples = j["samples"]
        s.render.filepath = os.path.join(OUT, j["name"])
        bpy.ops.render.render(write_still=True)
        report["renders"].append({"name": j["name"], "s": round(time.time() - t1, 1)})
    except Exception:
        report["errors"].append("render " + j["name"] + ": " + traceback.format_exc()[-600:])
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
print("TEARDOWN_DONE", json.dumps({k: report[k] for k in ("errors", "build_s", "objects", "tris")}))
