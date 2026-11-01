"""Provision the /how-it-works stage frames: private footage that never enters git (config/how_media.json lists them).

Usage (python3 tools/how_media.py, a thin wrapper):
  how_media.py check  [--root DIR]               verify every frame against the manifest (name, size, SHA-256)
  how_media.py pack   [--root DIR] [--bundle F]  tar the verified frames into the local, gitignored bundle
  how_media.py unpack [--root DIR] [--bundle F]  place the bundle's frames (each verified first), then check
Every command exits non-zero and names the file that is missing or differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "config/how_media.json"


class MediaError(RuntimeError):
    pass


def _safe_rel(value, key: str) -> None:
    p = PurePosixPath(value) if isinstance(value, str) and value else None
    if p is None or p.is_absolute() or ".." in p.parts or "\\" in value:
        raise MediaError(f"{MANIFEST}: {key} must be a relative path inside the repo, got {value!r}")


def load(root: Path = ROOT) -> dict:
    cfg = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {"dir", "bundle", "files"}:
        raise MediaError(f"{MANIFEST}: keys must be dir, bundle, files")
    _safe_rel(cfg["dir"], "dir")
    _safe_rel(cfg["bundle"], "bundle")
    if not isinstance(cfg["files"], list) or not cfg["files"]:
        raise MediaError(f"{MANIFEST}: files must be a non-empty list")
    seen = set()
    for f in cfg["files"]:
        if not isinstance(f, dict) or set(f) != {"name", "size", "sha256"}:
            raise MediaError(f"{MANIFEST}: bad entry {f!r}")
        name = f["name"]
        if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name:
            raise MediaError(f"{MANIFEST}: name must be a bare file name, got {name!r}")
        if name in seen:
            raise MediaError(f"{MANIFEST}: duplicate name {name!r}")
        seen.add(name)
        if type(f["size"]) is not int or f["size"] <= 0:
            raise MediaError(f"{MANIFEST}: {name}: size must be a positive integer, got {f['size']!r}")
        if not isinstance(f["sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", f["sha256"]):
            raise MediaError(f"{MANIFEST}: {name}: sha256 must be 64 hex characters, got {f['sha256']!r}")
        f["sha256"] = f["sha256"].lower()
    return cfg


def names(root: Path = ROOT) -> set[str]:
    return {f["name"] for f in load(root)["files"]}


def _bad(entry: dict, data: bytes) -> str | None:
    if len(data) != entry["size"]:
        return f"{entry['name']}: size {len(data)}, manifest says {entry['size']}"
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        return f"{entry['name']}: SHA-256 differs from the manifest"
    return None


def problems(root: Path = ROOT, *, deep: bool = True) -> list[str]:
    """One line per missing or differing frame; deep=False checks presence and size only (cheap, per request)."""
    cfg = load(root)
    out = []
    for e in cfg["files"]:
        p = root / cfg["dir"] / e["name"]
        if not p.is_file():
            out.append(f"{e['name']}: missing")
        elif deep:
            out.extend(filter(None, [_bad(e, p.read_bytes())]))
        elif p.stat().st_size != e["size"]:
            out.append(f"{e['name']}: size {p.stat().st_size}, manifest says {e['size']}")
    return out


def fix_hint(root: Path = ROOT) -> str:
    return f"python3 tools/how_media.py unpack --bundle {load(root)['bundle']}"


def check(root: Path = ROOT) -> None:
    bad = problems(root)
    if bad:
        raise MediaError("how-it-works frames not provisioned:\n  " + "\n  ".join(bad) + f"\nrun: {fix_hint(root)}")


def _refuse_unignored(out: Path) -> None:
    """A bundle written inside a git repo must be git-ignored; outside any repo is fine."""
    out = out.resolve()
    anchor = next(p for p in out.parents if p.is_dir())
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["LC_ALL"] = "C"
    try:
        result = subprocess.run(["git", "-C", str(anchor), "check-ignore", "-q", str(out)],
                                capture_output=True, env=env, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaError(f"{out}: git ignore check failed: {error}") from error
    if result.returncode == 128 and result.stdout == b"" and result.stderr == (
        b"fatal: not a git repository (or any of the parent directories): .git\n"
    ):
        return
    if result.returncode == 1 and result.stdout == result.stderr == b"":
        raise MediaError(f"{out}: bundle destination is inside a git repo and not git-ignored; footage never enters git")
    if result.returncode != 0 or result.stdout or result.stderr:
        raise MediaError(f"{out}: git ignore check failed (exit {result.returncode}): {result.stderr.decode(errors='replace').strip()}")


def pack(root: Path = ROOT, bundle: Path | None = None) -> Path:
    check(root)
    cfg = load(root)
    out = bundle or root / cfg["bundle"]
    _refuse_unignored(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".how-media-bundle-", suffix=".tmp", dir=out.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            with tarfile.open(fileobj=fh, mode="w") as tar:
                for e in cfg["files"]:
                    tar.add(root / cfg["dir"] / e["name"], arcname=e["name"])
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def unpack(root: Path = ROOT, bundle: Path | None = None) -> int:
    cfg = load(root)
    src = bundle or root / cfg["bundle"]
    if not src.is_file():
        raise MediaError(f"bundle not found: {src} (the footage bundle is private and kept outside git; copy it there)")
    wanted = {e["name"]: e for e in cfg["files"]}
    target = root / cfg["dir"]
    seen, written = set(), 0
    with tarfile.open(src) as tar:
        for m in tar:
            if not m.isreg() or m.name not in wanted:
                raise MediaError(f"{src}: unexpected member {m.name!r}")
            data = tar.extractfile(m).read()
            if (why := _bad(wanted[m.name], data)):
                raise MediaError(f"{src}: {why}")
            seen.add(m.name)
            dest = target / m.name
            if dest.is_file() and dest.read_bytes() == data:
                continue
            target.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".how-media-", suffix=".tmp", dir=target)
            tmp = Path(name)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
            written += 1
    if seen != set(wanted):
        raise MediaError(f"{src}: missing {sorted(set(wanted) - seen)}")
    check(root)
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "pack", "unpack"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--bundle", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            check(a.root)
            print(f"how media: {len(load(a.root)['files'])} frames match the manifest")
        elif a.cmd == "pack":
            print(f"how media: packed into {pack(a.root, a.bundle)}")
        else:
            print(f"how media: {unpack(a.root, a.bundle)} written")
    except MediaError as error:
        sys.exit(str(error))
