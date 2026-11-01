"""Pure-Python pose: lid angle, per-part rise progress, camera orbit and light look for a frame, or for a named still.

The film opens on the open laptop, the lid swings shut ([choreo] close_*), then every part rises one by one out of
the closed laptop (top part first, lid layers included, [choreo] stagger) into one vertical stack. After a hold the
parts return in the opposite order, the lowest first ([reassemble]), and the film ends on the complete laptop.
"""
import math


def smoother(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def _mix(a, b, t):
    if isinstance(a, (list, tuple)):
        return [x + (y - x) * t for x, y in zip(a, b)]
    return a + (b - a) * t


def at(keys, frame, fields):
    """Interpolate rows (dicts with 'frame') at `frame`; returns {field: value}."""
    keys = sorted(keys, key=lambda k: k["frame"])
    if frame <= keys[0]["frame"]:
        return {f: keys[0][f] for f in fields}
    for a, b in zip(keys, keys[1:]):
        if frame <= b["frame"]:
            t = smoother((frame - a["frame"]) / (b["frame"] - a["frame"]))
            return {f: _mix(a[f], b[f], t) for f in fields}
    return {f: keys[-1][f] for f in fields}


LOOK = ("exposure", "key", "top", "rim", "wash", "shell")


def fall_slot(cfg, n):
    """Order a part goes back: the risen layers lowest first (0, 1, ..); parts that never rise (their screws only) go with slot 0."""
    back = list(reversed(cfg["explode"]["rise"]))
    return back.index(n) if n in back else 0


def part_progress(cfg, frame):
    """{part: 0..1}: how far each part has travelled to its exploded place at `frame` (back to 0 once reassembled)."""
    ch, ex, re_ = cfg["choreo"], cfg["explode"], cfg["reassemble"]
    out = {n: 0.0 for n in ex["offset"]}
    for i, n in enumerate(ex["rise"]):
        up = smoother((frame - ch["rise_start"] - i * ch["rise_stagger"]) / ch["rise_frames"])
        k = fall_slot(cfg, n)
        out[n] = up * (1.0 - smoother((frame - re_["start"] - k * re_["stagger"]) / re_["frames"]))
    return out


def screw_progress(cfg, frame):
    """{part: 0..1}: each layer's screws back out ([screws] lead frames) just before that layer moves.

    Parts that never rise (the bottom case) come last, after the final layer has started to rise."""
    ch, ex, sc = cfg["choreo"], cfg["explode"], cfg["screws"]
    order = list(ex["rise"]) + [n for n in ex["offset"] if n not in ex["rise"]]
    re_ = cfg["reassemble"]
    # back in as the layer lands: the last sc["frames"] of its return
    return {n: smoother((frame - (ch["rise_start"] + i * ch["rise_stagger"] - sc["lead"])) / sc["frames"])
            * (1.0 - smoother((frame - (re_["start"] + fall_slot(cfg, n) * re_["stagger"] + re_["frames"] - sc["frames"])) / sc["frames"]))
            for i, n in enumerate(order)}


def bob(cfg, frame, parts):
    """{part: mm}: a sub-millimetre float once a layer has risen; zero while it is seated (scaled by its own progress)."""
    fl = cfg["float"]
    rise = cfg["explode"]["rise"]
    return {n: fl["amp_mm"] * parts.get(n, 0.0) * math.sin(2 * math.pi * frame / fl["period"] + i * fl["phase"]) for i, n in enumerate(rise)}


def span(cfg):
    """(first, last) frame of any part motion."""
    ch, ex = cfg["choreo"], cfg["explode"]
    return ch["rise_start"], ch["rise_start"] + (len(ex["rise"]) - 1) * ch["rise_stagger"] + ch["rise_frames"]


def reassembled(cfg):
    """Frame the last part is back in place."""
    re_ = cfg["reassemble"]
    return re_["start"] + (len(cfg["explode"]["rise"]) - 1) * re_["stagger"] + re_["frames"]


def progress(cfg, frame):
    a, b = span(cfg)
    c = cfg["reassemble"]["start"]
    return smoother((frame - a) / (b - a)) * (1.0 - smoother((frame - c) / (reassembled(cfg) - c)))


def frame_look(cfg, explode):
    """Exposure and light multipliers, open -> exploded by explode progress, plus an edge lift mid-way (zero at both ends)."""
    lk = cfg["look"]
    look = {k: _mix(lk["open"][k], lk["exploded"][k], explode) for k in LOOK}
    bump = 4.0 * explode * (1.0 - explode)
    for k, extra in zip(("rim", "key", "top"), cfg["scene"]["mid_lift"]):
        look[k] *= 1.0 + extra * bump
    return look


def frame_pose(cfg, frame):
    cam = at(cfg["pose"]["camera"], frame, ["az", "el", "dist", "target", "lens", "fstop"])
    ex = progress(cfg, frame)
    ch = cfg["choreo"]
    shut = smoother((frame - ch["close_start"]) / ch["close_frames"])   # ease-in-out swing to closed (0 deg)
    parts = part_progress(cfg, frame)
    a = cfg["display"]["sleep_from"]
    power = 1.0 - smoother((shut - a) / (1.0 - a))                     # screen and key backlight sleep as the lid shuts
    return {"frame": frame, "lid_deg": cfg["scene"]["lid_open_deg"] * (1.0 - shut), "explode": ex, "parts": parts, "power": power,
            "screws": screw_progress(cfg, frame), "bob": bob(cfg, frame, parts), "hide": [], "camera": cam, **frame_look(cfg, ex)}


def still_pose(cfg, state, angle):
    """A frame of the film seen from a named angle: same parts and look, different camera."""
    st = cfg["state"][state]
    ang = next(a for a in cfg["angle"] if a["name"] == angle)
    p = frame_pose(cfg, st["frame"])
    p["camera"] = {"az": ang["az"], "el": ang["el"], "dist": st["dist"], "target": st["target"], "lens": ang["lens"], "fstop": ang["fstop"]}
    return p


def camera_position(cam):
    az, el = math.radians(cam["az"]), math.radians(cam["el"])
    t, d = cam["target"], cam["dist"]
    return [t[0] + d * math.sin(az) * math.cos(el), t[1] - d * math.cos(az) * math.cos(el), t[2] + d * math.sin(el)]


def part_offsets(cfg, parts):
    return {k: v * parts.get(k, 0.0) for k, v in cfg["explode"]["offset"].items()}
