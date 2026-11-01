"""Synthetic catch media: small generated JPEGs, a bundle zip and a manifest that matches them (never the private footage)."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tools import catch_media


def make(folder: Path) -> tuple[Path, Path]:
    """Write folder/catch-media.zip and folder/catch_media.sha256 for the configured image names; return (bundle, manifest)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    bundle, manifest = folder / "catch-media.zip", folder / "catch_media.sha256"
    lines = []
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in catch_media.filenames(catch_media.load()):
            ok, data = cv2.imencode(".jpg", rng.integers(0, 255, (12, 16, 3), dtype=np.uint8))
            assert ok
            data = data.tobytes()
            archive.writestr(name, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data)}  {name}")
    manifest.write_text("\n".join(lines) + "\n")
    return bundle, manifest
