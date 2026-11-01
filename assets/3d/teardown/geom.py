"""Geometry helpers (Blender 5.x bpy + bmesh). All inputs are millimetres; meshes are stored in metres.

Every object created here is parented to the current part group (CUR) and tagged with its part
name so the render script can hide or move whole parts.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

MM = 0.001
CUR = {"part": None, "group": None, "mat": None, "bevel": (0.28, 3), "lid_pivot": None}


# ---- 2D outlines ------------------------------------------------------------
def rrect(w, d, r, cx=0.0, cy=0.0, seg=6):
    if r <= 0.01:
        return [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2), (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]
    pts = []
    for sx, sy, angle0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        kx, ky = cx + sx * (w / 2 - r), cy + sy * (d / 2 - r)
        for i in range(seg + 1):
            a = math.radians(angle0 + 90.0 * i / seg)
            pts.append((kx + r * math.cos(a), ky + r * math.sin(a)))
    return pts


def circle(r, cx=0.0, cy=0.0, n=32):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def fillet_path(pts, r, seg=6):
    """Round the corners of a 2D/3D polyline with quadratic arcs of tangent length ~r."""
    pts = [Vector(p) for p in pts]
    out = [pts[0]]
    for a, p, b in zip(pts, pts[1:], pts[2:]):
        edge_in, edge_out = (p - a), (b - p)
        t = min(r, edge_in.length / 2, edge_out.length / 2)
        p0, p1 = p - edge_in.normalized() * t, p + edge_out.normalized() * t
        for i in range(seg + 1):
            s = i / seg
            out.append((1 - s) ** 2 * p0 + 2 * s * (1 - s) * p + s * s * p1)
    out.append(pts[-1])
    return out


# ---- bmesh building blocks ----------------------------------------------------
def bm_prism(bm, pts, z0, z1, taper=1.0, top_pts=None):
    """Extrude the outline pts between z0 and z1. taper scales the top about the centroid."""
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    bot = [bm.verts.new((x, y, z0)) for x, y in pts]
    tp = top_pts or [((x - cx) * taper + cx, (y - cy) * taper + cy) for x, y in pts]
    top = [bm.verts.new((x, y, z1)) for x, y in tp]
    faces = [bm.faces.new(top), bm.faces.new(bot[::-1])]
    for i in range(n):
        j = (i + 1) % n
        faces.append(bm.faces.new((bot[i], bot[j], top[j], top[i])))
    return faces


def bm_box(bm, cx, cy, z0, w, d, h, r=0.0, taper=1.0):
    return bm_prism(bm, rrect(w, d, r, cx, cy), z0, z0 + h, taper)


def bm_sweep(bm, path, profile, up=(0, 0, 1), cap=True):
    """Sweep a 2D profile (u across, v up; mm) along a 3D polyline path (mm)."""
    path = [Vector(p) for p in path]
    upv = Vector(up)
    rings = []
    for i, p in enumerate(path):
        if i == 0:
            t = path[1] - path[0]
        elif i == len(path) - 1:
            t = path[-1] - path[-2]
        else:
            t = (path[i + 1] - p).normalized() + (p - path[i - 1]).normalized()
        t = t.normalized()
        side = t.cross(upv)
        if side.length < 1e-6:
            side = t.cross(Vector((0, 1, 0)))
        side.normalize()
        nrm = side.cross(t).normalized()
        rings.append([bm.verts.new(p + side * u + nrm * v) for u, v in profile])
    k = len(profile)
    for a, b in zip(rings, rings[1:]):
        for i in range(k):
            j = (i + 1) % k
            bm.faces.new((a[i], a[j], b[j], b[i]))
    if cap:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])


def flat_profile(w, h, r=0.4, seg=3):
    return rrect(w, h, min(r, w / 2 - 0.01, h / 2 - 0.01), 0, 0, seg)


def uv_planar(bm, x0, y0, w, h, flip=False, vflip=False):
    layer = bm.loops.layers.uv.verify()
    for f in bm.faces:
        for lp in f.loops:
            c = lp.vert.co
            u, v = (c.x - x0) / w, (c.y - y0) / h
            if vflip:
                v = 1 - v
            lp[layer].uv = (1 - u, v) if flip else (u, v)


# ---- objects ------------------------------------------------------------------
def _link(obj, parent=True):
    bpy.context.scene.collection.objects.link(obj)
    if parent and CUR["group"] is not None:
        obj.parent = CUR["group"]
        if CUR["lid_pivot"] is not None:      # lid parts: coordinates are the closed-pose world ones
            obj.matrix_parent_inverse = Matrix.Identity(4)
    if CUR["part"]:
        obj["part"] = CUR["part"]
    return obj


def to_obj(name, bm, mat=None, bevel=True, uv=None, smooth=True, cuts=(), width=None, seg=None, angle=35.0):
    if uv:
        uv_planar(bm, *uv)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.scale(bm, vec=(MM, MM, MM), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    obj = _link(bpy.data.objects.new(name, me))
    mat = mat or CUR["mat"]
    if mat is not None:
        me.materials.append(mat)
    for c in cuts:
        m = obj.modifiers.new("cut", "BOOLEAN")
        m.operation = "DIFFERENCE"
        m.object = c
        m.solver = "EXACT"
    if bevel:
        w, s = CUR["bevel"]
        m = obj.modifiers.new("bevel", "BEVEL")
        m.width = (width or w) * MM
        m.segments = seg or s
        m.limit_method = "ANGLE"
        m.angle_limit = math.radians(angle)
        try:
            m.harden_normals = True
        except Exception:
            pass
    return obj


def cutter(bm, name="cutter"):
    """A hidden solid used by to_obj(cuts=[...])."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.scale(bm, vec=(MM, MM, MM), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = _link(bpy.data.objects.new(name, me))       # parented to the part group so the cut follows the part's pose
    obj.hide_render = True
    obj.hide_viewport = True
    return obj


def prism(name, pts, z0, z1, **kw):
    bm = bmesh.new()
    bm_prism(bm, pts, z0, z1, taper=kw.pop("taper", 1.0), top_pts=kw.pop("top_pts", None))
    return to_obj(name, bm, **kw)


def box(name, cx, cy, z0, w, d, h, r=0.0, **kw):
    bm = bmesh.new()
    bm_box(bm, cx, cy, z0, w, d, h, r)
    return to_obj(name, bm, **kw)


def cyl(name, cx, cy, z0, r, h, n=32, axis="Z", **kw):
    """Cylinder; axis X or Y lays it down (rotation about its centre)."""
    bm = bmesh.new()
    bm_prism(bm, circle(r, 0, 0, n), -h / 2, h / 2)
    if axis == "X":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "Y"))
    elif axis == "Y":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(cx, cy, z0 + (h / 2 if axis == "Z" else 0)))
    return to_obj(name, bm, **kw)


def group(name, parent=None, lid=False, pivot=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "PLAIN_AXES"
    e.empty_display_size = 0.01
    bpy.context.scene.collection.objects.link(e)
    if parent is not None:
        e.parent = parent
        if lid:
            e.matrix_parent_inverse = Matrix.Translation(-Vector(pivot) * MM)
    return e


class Part:
    """with Part('battery', mat): ...  -> every object made inside is parented to the part's group."""

    def __init__(self, name, mat=None, lid_pivot=None, bevel=None):
        self.name, self.mat, self.lid_pivot, self.bevel = name, mat, lid_pivot, bevel

    def __enter__(self):
        self.saved = dict(CUR)
        pivot = self.lid_pivot
        g = group("g_" + self.name, parent=pivot, lid=pivot is not None,
                  pivot=tuple(v / MM for v in pivot.location) if pivot is not None else None)
        g["part"] = self.name
        CUR.update(part=self.name, group=g, mat=self.mat, lid_pivot=pivot)
        if self.bevel:
            CUR["bevel"] = self.bevel
        self.g = g
        return g

    def __exit__(self, *exc):
        CUR.update(self.saved)
        return False


def sub_group(name, parent):
    g = group(name, parent=parent)
    g["part"] = CUR["part"]
    return g
