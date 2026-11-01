"""Pack the home film's frame archives into the served bundle files (paths and plan from config/film.toml).

Usage: film_frames.py unpack [--root DIR]: reads DIR/config/film.toml. For each frame set it reads the tar of single
frames and writes the bundles app/film_config.bundles plans into <target>/<dir>/. A bundle is a run of records, each
a little-endian uint16 frame number, a uint32 byte length and the frame's webp bytes unchanged; static/ui/film.js
reads it as a stream. Idempotent: a bundle already on disk with the same bytes is left alone; others are written to
a temp sibling and moved into place atomically; one unpack runs at a time per target (a file lock). Every archive and
target path must be a normalized relative path under assets/ or static/ (checked again before each read, write and delete). Any other file in a set's folder (single frames from an older build,
a stale bundle) is removed, so the folder holds exactly what the page fetches. It also writes the part anchors the
labels use (<target>/<film.anchors>): the render's anchors for every frame, kept to the [[parts]] ids, rounded.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import struct
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import film_config  # noqa: E402


def _safe(root: Path, rel: str, base: str, write: bool = False) -> Path:
    """root/rel after checking, now, that rel is a normalized relative path beneath `base` and that the path as it
    resolves on disk (symlinks followed) is still inside root/base. Called right before every read, write and delete.
    For a write the base itself must also sit inside root (a read may follow a symlinked assets/ checkout)."""
    if not film_config.contained(rel, base):
        raise ValueError(f"{rel!r} is not a normalized relative path under {base}/")
    if write:
        _under(root / base, root)
    return _under(root / rel, root / base)


def _under(path: Path, base: Path) -> Path:
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"{path} is outside {base}")
    return path


def read_frames(tar_path: Path, film: dict, set_dir: str) -> dict[int, bytes]:
    """Frame number -> webp bytes; the archive must hold exactly frames 1..N as '<set_dir>/<frame name>'."""
    names = {f"{set_dir}/{film['frame_pattern'].format(n=n)}": n for n in range(1, film["frames"] + 1)}
    frames = {}
    with tarfile.open(tar_path) as tar:
        for m in tar:
            n = names.get(m.name)
            if not m.isreg() or n is None or n in frames:
                raise ValueError(f"unexpected archive member {m.name!r}")
            frames[n] = tar.extractfile(m).read()
    if len(frames) != film["frames"]:
        raise ValueError(f"{tar_path.name}: {film['frames'] - len(frames)} frames missing")
    return frames


def read_anchors(path: Path, film: dict, ids: list[str]) -> dict[str, dict[str, list[float]]]:
    """The render's anchors (tools/teardown_render.py --mode anchors), checked and cut to what the labels read:
    {frame: {part id: [rx, ry, lx, ly]}} for frames 1..N, each of the ids, fractions rounded to 3 places."""
    raw = json.loads(path.read_text())
    want = {str(n) for n in range(1, film["frames"] + 1)}
    if not isinstance(raw, dict) or set(raw) != want:
        raise ValueError(f"{path.name}: needs exactly frames 1..{film['frames']}")
    out = {}
    for n in sorted(want, key=int):
        row = raw[n]
        if not isinstance(row, dict) or not set(ids) <= set(row):
            raise ValueError(f"{path.name}: frame {n} lacks {sorted(set(ids) - set(row or {}))}")
        pts = {}
        for i in ids:
            v = row[i]
            if not (isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) and -1 <= x <= 2 for x in v)):
                raise ValueError(f"{path.name}: frame {n} {i}: bad anchor {v!r}")
            pts[i] = [round(float(x), 3) for x in v]
        out[n] = pts
    return out


def _put(dest: Path, data: bytes) -> int:
    """Atomic write unless the same bytes are already there; 1 if written."""
    if dest.is_file() and dest.read_bytes() == data:
        return 0
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return 1


def unpack(root: Path = ROOT) -> int:
    """Returns the number of files written (bundles and the anchors). One unpack at a time per target (a file lock),
    so another process's temporaries are never taken for stale ones."""
    cfg = film_config.load(root / "config" / "film.toml")
    arc = cfg["archives"]
    target = _safe(root, arc["target"], "static", write=True)
    target.mkdir(parents=True, exist_ok=True)
    with open(_under(target / ".unpack.lock", target), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _unpack(root, cfg, target)


def _unpack(root: Path, cfg: dict, target: Path) -> int:
    film, arc = cfg["film"], cfg["archives"]
    plan = film_config.bundles(film)
    keep = {name for name, _ in plan}
    written = 0
    for s in cfg["sets"]:
        frames = read_frames(_safe(root, arc["source"].format(dir=s["dir"]), "assets"), film, s["dir"])
        out = _under(target / s["dir"], target)
        out.mkdir(parents=True, exist_ok=True)
        for name, nums in plan:
            data = b"".join(struct.pack("<HI", n, len(frames[n])) + frames[n] for n in nums)
            written += _put(_under(out / name, target), data)
        for stale in out.iterdir():
            if stale.is_file() and stale.name not in keep:
                _under(stale, target).unlink(missing_ok=True)
    anchors = read_anchors(_safe(root, arc["anchors"], "assets"), film, [p["id"] for p in cfg["parts"]])
    written += _put(_under(target / film["anchors"], target), json.dumps(anchors, separators=(",", ":")).encode())
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unpack"])
    ap.add_argument("--root", type=Path, default=ROOT)
    print(f"film files: {unpack(ap.parse_args().root)} written")
