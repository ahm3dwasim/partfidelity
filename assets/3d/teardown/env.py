"""Studio: dark sweep, softbox lights, camera, colour management, Cycles settings."""
import math

import bmesh
import bpy
from mathutils import Vector

import geom as G
import mats

MM = 0.001


def world(cfg):
    w = bpy.data.worlds.new("studio")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = mats._srgb(cfg["material"]["world"])
    bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = w


def sweep(cfg):
    s = cfg["scene"]
    z0, R, wall = s["floor_z"], s["sweep_radius"], s["wall_y"]
    prof = [(-6000.0, z0), (wall - R, z0)]
    for i in range(1, 13):
        a = math.radians(90.0 * i / 12)
        prof.append((wall - R + R * math.sin(a), z0 + R - R * math.cos(a)))
    prof.append((wall, z0 + 6000.0))
    bm = bmesh.new()
    vs = [(bm.verts.new((-6000, y, z)), bm.verts.new((6000, y, z))) for y, z in prof]
    for (pa, pb), (qa, qb) in zip(vs, vs[1:]):
        bm.faces.new((pa, pb, qb, qa))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.scale(bm, vec=(MM, MM, MM), verts=bm.verts)
    me = bpy.data.meshes.new("sweep")
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    obj = bpy.data.objects.new("sweep", me)
    me.materials.append(mats.get("floor"))
    bpy.context.scene.collection.objects.link(obj)
    obj["env"] = True


def lights(cfg):
    """Studio lights from config [[light]]: soft white key and top, coloured rims and kicker, a coloured wash on the sweep."""
    for row in cfg["light"]:
        d = bpy.data.lights.new(row["name"], "AREA" if row["kind"] == "area" else "SPOT")
        if row["kind"] == "area":
            d.shape = "RECTANGLE"
            d.size, d.size_y = row["size"]
            d.spread = math.radians(row["spread"])
        else:
            d.spot_size = math.radians(row["size"][0])
            d.spot_blend = row["size"][1]
        d.energy = row["power"]
        d.use_shadow = True
        d["base"] = row["power"]
        d["group"] = row["group"]
        d.color = row["color"]
        o = bpy.data.objects.new(row["name"], d)
        o.visible_camera = False           # never a bright card in the frame itself; reflections still see it
        o.location = Vector(row["loc"])
        o.rotation_euler = (Vector(row["target"]) - Vector(row["loc"])).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(o)
        o["env"] = True
        if not row["floor"]:
            _skip_floor(o)


def _skip_floor(light):
    """Light linking: this light lights everything except the sweep (colour on the laptop, not on the backdrop)."""
    coll = bpy.data.collections.new(light.name + "_receivers")
    coll.objects.link(bpy.data.objects["sweep"])
    light.light_linking.receiver_collection = coll
    coll.collection_objects[0].light_linking.link_state = "EXCLUDE"


def camera(cfg):
    c = cfg["camera"]
    d = bpy.data.cameras.new("cam")
    d.sensor_width = c["sensor_mm"]
    d.clip_start, d.clip_end = c["clip_start_m"], c["clip_end_m"]
    d.dof.use_dof = True                 # aperture comes per frame from the pose (camera keys and angles carry fstop)
    o = bpy.data.objects.new("cam", d)
    bpy.context.scene.collection.objects.link(o)
    bpy.context.scene.camera = o
    return o


def cycles(cfg, gpu=True):
    r = cfg["render"]
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    cy = s.cycles
    if gpu:
        cp = bpy.context.preferences.addons["cycles"].preferences
        for t in ("OPTIX", "CUDA"):
            try:
                cp.compute_device_type = t
                cp.get_devices()
                break
            except Exception as exc:
                print("DEVTYPE_FAIL", t, exc)
        for d in cp.devices:
            d.use = d.type != "CPU"
        print("DEVICES", cp.compute_device_type, [(d.name, d.type, d.use) for d in cp.devices])
        cy.device = "GPU"
    else:
        cy.device = "CPU"
    cy.seed = r["seed"]
    cy.use_animated_seed = False
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = r["adaptive_threshold"]
    cy.use_denoising = True
    try:
        cy.denoiser = r["denoiser"]
    except Exception as exc:
        print("DENOISER_WARN", exc)
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces, cy.transmission_bounces = 12, 4, 8, 4
    cy.transparent_max_bounces = 16            # see-through glass and backlight films stack up in the exploded view
    cy.sample_clamp_indirect = 8.0
    cy.caustics_reflective = cy.caustics_refractive = False
    s.render.film_transparent = False
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_depth = "8"
    s.view_settings.view_transform = "AgX"
    try:
        s.view_settings.look = r["look"] if r["look"].startswith("AgX") else "AgX - " + r["look"]
    except Exception as exc:
        print("LOOK_WARN", exc)
    s.view_settings.exposure = r["exposure"]
    s.render.filter_size = 1.3


def dust(cfg):
    """A few faint emissive motes in a box round the subject, seen by the camera only (they light nothing).

    Returns [(object, base_mm, velocity_mm_per_frame)] for render.py to move each frame."""
    import random
    dc = cfg["dust"]
    rnd = random.Random(dc["seed"])
    m = bpy.data.materials.new("dust")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = dc["strength"]
    em.inputs["Color"].default_value = (*dc["color"], 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    (x0, x1), (y0, y1), (z0, z1) = dc["box"]
    motes = []
    for i in range(dc["count"]):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=dc["radius_mm"] * rnd.uniform(0.6, 1.4) * MM)
        o = bpy.context.active_object
        o.name = "dust%d" % i
        o.data.materials.append(m)
        for attr in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter", "visible_shadow"):
            setattr(o, attr, False)
        o["env"] = True
        base = (rnd.uniform(x0, x1), rnd.uniform(y0, y1), rnd.uniform(z0, z1))
        vel = tuple(rnd.uniform(-1, 1) * dc["drift_mm"] for _ in range(3))
        motes.append((o, base, vel))
    return motes
