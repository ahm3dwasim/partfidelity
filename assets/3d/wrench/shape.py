"""The combination spanner's plan outline (plain Python + shapely, run before Blender). mm, ring at x=0, jaw at x=length."""
import math

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def _pts(poly):
    return [list(c) for c in poly.exterior.coords[:-1]]


def plan(w: dict) -> dict:
    L, half_start, half_end = w["length"], w["shaft_half"][0], w["shaft_half"][1]
    res = w["circle_res"]
    body = unary_union([
        Point(0, 0).buffer(w["ring_r"], res),
        Point(L, 0).buffer(w["jaw_r"], res),
        Polygon([(0, -half_start), (L, -half_end), (L, half_end), (0, half_start)]),
    ])
    f = w["fillet"]
    body = body.buffer(f, res).buffer(-f, res)           # closing: a fillet where the shaft meets each head
    r = w["slot_w"] / 2
    cx = L - w["slot_depth"] + r
    slot = unary_union([Point(cx, 0).buffer(r, res), Polygon([(cx, -r), (L + w["jaw_r"] + 5, -r), (L + w["jaw_r"] + 5, r), (cx, r)])])
    body = body.difference(slot).buffer(-w["jaw_soften"], res).buffer(w["jaw_soften"], res)   # soften the jaw tips
    tip, root = w["ring_hole_r"]
    star = Polygon([(math.cos(math.radians(7.5 + 15 * i)) * (tip if i % 2 == 0 else root),
                     math.sin(math.radians(7.5 + 15 * i)) * (tip if i % 2 == 0 else root)) for i in range(24)])
    rg = w["ridge"]
    ridge = LineString([(rg["start"], 0), (rg["end"], 0)]).buffer(rg["width"] / 2, res)
    return {"outer": _pts(body), "hole": _pts(star.buffer(-0.4, 4).buffer(0.4, 4)), "ridge": _pts(ridge)}
