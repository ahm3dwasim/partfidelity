"""Build, verify and install the catch page's small images; run from the repository root.

The images are crops of private frozen footage: never committed. `pack` cuts them from the
frozen catch evidence folder into var/catch-media.zip and writes the approved manifest;
`unpack` (run at deploy, like tools/sample_media.py) verifies the zip against that manifest
and installs it under static/ui/media/catch/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "catch_media.json"
MANIFEST = ROOT / "config" / "catch_media.sha256"
BUNDLE = ROOT / "var" / "catch-media.zip"
TARGET = Path("static/ui/media/catch")


def load(path: Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not {"width_px", "quality", "images", "cases"} <= set(cfg):
        raise ValueError(f"{path}: width_px, quality, images and cases are required")
    names = [i["name"] for i in cfg["images"]]
    if any(not re.fullmatch(r"[a-z0-9-]+", n) for n in names):
        raise ValueError(f"{path}: image names must be lowercase letters, digits and dashes")
    if len(set(names)) != len(names) or len(names) > 8:
        raise ValueError(f"{path}: image names must be unique and at most 8")
    for image in cfg["images"]:
        x0, y0, x1, y1 = image["crop"]
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"{path}: bad crop for {image['name']}")
    for case in cfg["cases"]:
        if case["original"] not in names or case["edited"] not in names:
            raise ValueError(f"{path}: case {case['key']} names an unknown image")
    return cfg


def filenames(cfg: dict) -> list[str]:
    return [f"{i['name']}.jpg" for i in cfg["images"]]


def manifest(cfg: dict, path: Path = MANIFEST) -> dict[str, tuple[str, int]]:
    rows = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        digest, size, name = line.split("  ", 2)
        rows[name] = (digest, int(size))
    if set(rows) != set(filenames(cfg)):
        raise ValueError(f"{path}: names differ from {CONFIG.name}; run pack against the frozen evidence")
    return rows


def render(source: Path, crop, cfg: dict) -> bytes:
    import cv2  # lazy: `modal deploy` imports this module from a CLI env without opencv
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"cannot read {source}")
    h, w = image.shape[:2]
    x0, y0, x1, y1 = crop
    part = image[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    width = cfg["width_px"]
    part = cv2.resize(part, (width, round(part.shape[0] * width / part.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, cfg["quality"]])
    if not ok:
        raise ValueError(f"cannot encode {source}")
    return data.tobytes()


def pack(evidence: Path, root: Path = ROOT, bundle: Path | None = None) -> None:
    cfg = load(root / "config" / "catch_media.json")
    bundle = bundle or root / "var" / "catch-media.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        for image in cfg["images"]:
            data = render(evidence / image["from"], image["crop"], cfg)
            archive.writestr(f"{image['name']}.jpg", data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data)}  {image['name']}.jpg")
    (root / "config" / "catch_media.sha256").write_text("\n".join(lines) + "\n")


def check(root: Path = ROOT, manifest_path: Path | None = None) -> None:
    cfg = load(root / "config" / "catch_media.json")
    expected = manifest(cfg, manifest_path or root / "config" / "catch_media.sha256")
    for name, (digest, size) in expected.items():
        path = root / TARGET / name
        if not path.is_file():
            raise RuntimeError(f"Catch media missing: {name}. Provision with: python tools/catch_media.py unpack")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest or len(data) != size:
            raise ValueError(f"catch media {name} does not match the approved manifest")


def unpack(root: Path = ROOT, bundle: Path | None = None, manifest_path: Path | None = None) -> None:
    bundle = bundle or root / "var" / "catch-media.zip"
    cfg = load(root / "config" / "catch_media.json")
    expected = manifest(cfg, manifest_path or root / "config" / "catch_media.sha256")
    if not bundle.is_file():
        return check(root, manifest_path)
    with zipfile.ZipFile(bundle) as archive:
        if sorted(archive.namelist()) != sorted(expected):
            raise ValueError(f"Invalid catch media bundle: {bundle}")
        blobs = {name: archive.read(name) for name in expected}
    for name, (digest, size) in expected.items():  # verify everything before writing anything
        if hashlib.sha256(blobs[name]).hexdigest() != digest or len(blobs[name]) != size:
            raise ValueError(f"catch media {name} does not match the approved manifest")
    (root / TARGET).mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        (root / TARGET / name).write_bytes(data)
    check(root, manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "pack", "unpack"))
    parser.add_argument("--evidence", type=Path, help="frozen catch evidence folder (pack only)")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            if not args.evidence:
                parser.error("pack needs --evidence")
            pack(args.evidence)
        else:
            {"check": check, "unpack": unpack}[args.command]()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"catch media: {error}\n")
    print(f"catch media {args.command}: OK")
