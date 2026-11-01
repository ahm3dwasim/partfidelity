"""Manage the private sample media bundle; run from the repository root."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ("sample.mp4", *(f"frames/step-{n:02}.png" for n in range(1, 10)),
         "labels/step-06-part-1.png", "labels/step-06-part-2.png",
         "labels/step-07-part-2.png", "labels/step-08-part-2.png",
         "labels/step-06-part-1.mask.png", "labels/step-06-part-2.mask.png",
         "labels/step-07-part-2.mask.png", "labels/step-08-part-2.mask.png")  # true SAM part masks, one per label; generated, not footage
BUNDLE = ROOT / "var" / "sample-media.zip"
COMMAND = "python tools/sample_media.py unpack"
MANIFEST = ROOT / "config" / "sample_media.sha256"


def manifest(path: Path = MANIFEST) -> dict[str, tuple[str, int]]:
    """name -> (sha256, bytes) for every approved artifact; fails loudly if it is off."""
    rows = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        where = f"sample media manifest {path} line {number}"
        if line.startswith("#"):
            continue
        parts = line.split("  ", 2)
        if len(parts) != 3 or line != line.strip():
            raise ValueError(f"{where}: expected '<sha256>  <bytes>  <name>', got {line!r}")
        digest, size, name = parts
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"{where}: digest is not 64 hex characters: {digest!r}")
        if not re.fullmatch(r"[0-9]+", size) or int(size) <= 0:
            raise ValueError(f"{where}: size is not a positive integer: {size!r}")
        if not name.strip() or name != name.strip():
            raise ValueError(f"{where}: blank or padded name: {name!r}")
        if name in rows:
            raise ValueError(f"{where}: duplicate name {name}")
        rows[name] = (digest.lower(), int(size))
    if set(rows) != set(FILES):
        missing = [n for n in FILES if n not in rows]
        unexpected = sorted(set(rows) - set(FILES))
        raise ValueError(f"sample media manifest {path} names differ from the approved set: "
                         f"missing {missing}, unexpected {unexpected}")
    return rows


def verify(name: str, data: bytes, expected: dict[str, tuple[str, int]]) -> None:
    digest, size = expected[name]
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest or len(data) != size:
        raise ValueError(f"sample media {name} does not match the approved manifest: "
                         f"expected sha256 {digest} ({size} bytes), got {actual} ({len(data)} bytes)")


def check(root: Path = ROOT, expected: dict | None = None) -> None:
    missing = [name for name in FILES if not (root / "app/sample_real" / name).is_file()
               or (root / "app/sample_real" / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Sample media missing: {', '.join(missing)}. Provision with: {COMMAND}")
    expected = expected or manifest()
    for name in FILES:
        verify(name, (root / "app/sample_real" / name).read_bytes(), expected)


def pack(root: Path = ROOT, bundle: Path = BUNDLE, expected: dict | None = None) -> None:
    check(root, expected)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / "app/sample_real" / name, name)


def unpack(root: Path = ROOT, bundle: Path = BUNDLE, expected: dict | None = None) -> None:
    expected = expected or manifest()
    if not bundle.is_file():
        check(root, expected)
        return
    with zipfile.ZipFile(bundle) as archive:
        if len(archive.namelist()) != len(FILES) or set(archive.namelist()) != set(FILES):
            raise ValueError(f"Invalid sample media bundle: {bundle}")
        for name in FILES:  # verify everything before writing anything
            verify(name, archive.read(name), expected)
        for name in FILES:
            base = root / "app/sample_real"
            dest = base / name
            if dest.is_symlink() or not dest.resolve().is_relative_to(base.resolve()):
                raise ValueError(f"unsafe sample media path: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    check(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "pack", "unpack"))
    args = parser.parse_args()
    try:
        {"check": check, "pack": pack, "unpack": unpack}[args.command]()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"sample media: {error}\n")
    print(f"sample media {args.command}: OK")
