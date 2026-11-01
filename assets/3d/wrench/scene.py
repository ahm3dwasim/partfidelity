"""Blender script (run in the Modal container): builds the spanner from plan.json and renders each variant to a transparent PNG.

  blender -b --factory-startup -P scene.py -- job.json outdir     (job: {"cfg": <config/wrench.toml>, "plan": {...}})
"""
import json
import math
import sys

import bpy
import bmesh
from mathutils import Vector

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = sys.argv[sys.argv.index("--") + 2]
CFG, PLAN = job["cfg"], job["plan"]
W, S, R = CFG["wrench"], CFG["scene"], CFG["render"]
U = 1.0 / S["unit_mm"]
for o in list(bpy.data.objects):   # the factory-startup cube, light and camera
    bpy.data.objects.remove(o)


def srgb(c):
    return [(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) for x in c] + [1.0]


def prism(name, pts, z0, z1, dx=0.0):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [bm.verts.new(((x - dx) * U, y * U, z0 * U)) for x, y in pts]
    face = bm.faces.new(vs)
    if face.normal.z < 0:
        face.normal_flip()
    ext = bmesh.ops.extrude_face_region(bm, geom=[face])
    for v in [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]:
        v.co.z = z1 * U
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def apply(ob, mod):
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.select_set(False)


def smooth(ob, deg):
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(deg))
    ob.select_set(False)


def bevel(ob, width, seg, deg):
    m = ob.modifiers.new("bev", "BEVEL")
    m.width, m.segments, m.limit_method, m.angle_limit = width * U, seg, "ANGLE", math.radians(deg)
    m.harden_normals = False
    apply(ob, m)


T, dx = W["thickness"], W["length"] / 2
body = prism("body", PLAN["outer"], 0, T, dx)
cut = prism("cut", PLAN["hole"], -2, T + 2, dx)
bm_ = body.modifiers.new("hole", "BOOLEAN")
bm_.operation, bm_.object, bm_.solver = "DIFFERENCE", cut, "EXACT"
apply(body, bm_)
bpy.data.objects.remove(cut)
bevel(body, *W["edge_bevel"], W["body_smooth_deg"])
smooth(body, W["body_smooth_deg"])
rg = W["ridge"]
ridge = prism("ridge", PLAN["ridge"], T - 0.3, T + rg["raise"], dx)
rb = W["ridge_bevel"]
bevel(ridge, rg["raise"] * rb["ratio"], rb["segments"], rb["angle_deg"])
smooth(ridge, rb["smooth_deg"])

metal = bpy.data.materials.new("metal")
metal.use_nodes = True
b = metal.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = srgb(W["metal"]["color"])
b.inputs["Metallic"].default_value = 1.0
b.inputs["Roughness"].default_value = W["metal"]["roughness"]
paint = bpy.data.materials.new("accent")
paint.use_nodes = True
p = paint.node_tree.nodes["Principled BSDF"]
p.inputs["Base Color"].default_value = srgb(W["accent"])
p.inputs["Roughness"].default_value = W["accent_finish"]["roughness"]
p.inputs["Coat Weight"].default_value = W["accent_finish"]["coat"]
body.data.materials.clear()   # the boolean leaves an empty slot that the faces would keep using
body.data.materials.append(metal)

# the wrench lies on the floor, ring/jaw axis yawed so it reads at axis_deg on screen
e = math.radians(S["elevation_deg"])
psi = 180.0 + math.degrees(math.atan(math.tan(math.radians(S["axis_deg"])) / math.sin(e)))
for ob in (body, ridge):
    ob.location.z = 0
    ob.parent = None
root = bpy.data.objects.new("root", None)
bpy.context.scene.collection.objects.link(root)
root.rotation_euler.z = math.radians(psi)
root.location.z = 0
for ob in (body, ridge):
    ob.parent = root

floor = bpy.data.objects.new("floor", bpy.data.meshes.new("floor"))
fm = floor.data
fm.from_pydata([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)], [], [(0, 1, 2, 3)])
bpy.context.scene.collection.objects.link(floor)
floor.is_shadow_catcher = True

sc = bpy.context.scene
w = bpy.data.worlds.new("studio")
w.use_nodes = True
nt = w.node_tree
nt.nodes.clear()
tc, sep, mr, ramp, bg, out = (nt.nodes.new(t) for t in (
    "ShaderNodeTexCoord", "ShaderNodeSeparateXYZ", "ShaderNodeMapRange", "ShaderNodeValToRGB", "ShaderNodeBackground", "ShaderNodeOutputWorld"))
nt.links.new(tc.outputs["Generated"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 1.0   # Generated is the view direction scaled to 0..1
nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
els = ramp.color_ramp.elements
for i, (pos, val) in enumerate(S["world"]):
    el = els[i] if i < len(els) else els.new(pos)
    el.position, el.color = pos, [val] * 3 + [1]
nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs[0], out.inputs[0])
sc.world = w

aim = bpy.data.objects.new("aim", None)
sc.collection.objects.link(aim)
aim.location = (0, 0, T * U / 2)


def light(name, spec, shape, size):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.shape = spec["energy"], shape
    if shape == "RECTANGLE":
        d.size, d.size_y = size
    else:
        d.size = size
    ob = bpy.data.objects.new(name, d)
    ob.location = spec["pos"]
    c = ob.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = aim, "TRACK_NEGATIVE_Z", "UP_Y"
    sc.collection.objects.link(ob)


cd = S["card"]
card = bpy.data.objects.new("card", bpy.data.meshes.new("card"))
h = cd["size"] / 2
card.data.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
card.location = (0, cd["distance"] * math.cos(e), cd["distance"] * math.sin(e) + T * U / 2)
ct = card.constraints.new("TRACK_TO")
ct.target, ct.track_axis, ct.up_axis = aim, "TRACK_NEGATIVE_Z", "UP_Y"
sc.collection.objects.link(card)
cm = bpy.data.materials.new("card")
cm.use_nodes = True
cn = cm.node_tree
cn.nodes.clear()
ctc, cg, cr, ce, co = (cn.nodes.new(t) for t in ("ShaderNodeTexCoord", "ShaderNodeTexGradient", "ShaderNodeValToRGB", "ShaderNodeEmission", "ShaderNodeOutputMaterial"))
sp = cn.nodes.new("ShaderNodeSeparateXYZ")
cn.links.new(ctc.outputs["Generated"], sp.inputs[0])
cn.links.new(sp.outputs["Y"], cr.inputs["Fac"])
cels = cr.color_ramp.elements
for i, (pos, val) in enumerate(cd["ramp"]):
    el = cels[i] if i < len(cels) else cels.new(pos)
    el.position, el.color = pos, [val] * 3 + [1]
ce.inputs["Strength"].default_value = cd["strength"]
cn.links.new(cr.outputs["Color"], ce.inputs["Color"])
cn.links.new(ce.outputs[0], co.inputs[0])
card.data.materials.append(cm)
card.visible_camera = False
light("rim", S["rim"], "RECTANGLE", S["rim"]["size_xy"])
light("fill", S["fill"], "SQUARE", S["fill"]["size"])

cam_d = bpy.data.cameras.new("cam")
cam_d.lens = S["focal_mm"]
cam = bpy.data.objects.new("cam", cam_d)
cam.location = (0, -S["distance"] * math.cos(e), S["distance"] * math.sin(e) + T * U / 2)
c = cam.constraints.new("TRACK_TO")
c.target, c.track_axis, c.up_axis = aim, "TRACK_NEGATIVE_Z", "UP_Y"
sc.collection.objects.link(cam)
sc.camera = cam

sc.render.engine = "CYCLES"
cy = sc.cycles
cy.device, cy.samples, cy.use_denoising, cy.denoiser, cy.seed = "CPU", R["samples"], True, R["denoiser"], R["seed"]
sc.render.resolution_x = sc.render.resolution_y = R["res"]
sc.render.resolution_percentage = 100
sc.render.film_transparent = True
sc.render.image_settings.file_format, sc.render.image_settings.color_mode = "PNG", "RGBA"
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - " + S["look"]
sc.view_settings.exposure = S["exposure"]

for v in R["variants"]:
    ridge.data.materials.clear()
    ridge.data.materials.append(paint if v == "orange" else metal)
    for tag, hide in (("", False), ("_cut", True)):   # _cut: no floor, so its alpha is the clean silhouette (the other carries the shadow)
        floor.hide_render = hide
        sc.render.filepath = f"{OUT}/wrench_{v}{tag}.png"
        bpy.ops.render.render(write_still=True)
print("WRENCH_DONE", psi)
