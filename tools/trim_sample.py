"""Trim the sample clip to the guide's last step and rebadge part-N thumbnails.

Rule: the final device-manager screens (~3:12-3:20) are never shown.
Never edits an existing file: it writes the trimmed clip, rebadged thumbnails, ``config/sample_media.json``
(clip length), ``config/sample_media.sha256`` and ``var/sample-media.zip`` into a NEW directory laid out like
the repository, and prints the ``cp`` command a human runs after checking them.

Originals are never modified. On an ordinary error no ``--out`` dir and no temp dir are left. A run killed by a
signal may leave a ``.trim-*`` temp dir, or a complete ``--out`` dir, next to the target; delete it by hand.

    python3 tools/trim_sample.py --out /tmp/trimmed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.render import badge_radius, draw_badge, write_png_rgb, read_png_rgb  # noqa: E402
from tools import sample_media  # noqa: E402

SAMPLE = ROOT / "app" / "sample_real"
CLIP_CONFIG = ROOT / "config" / "sample_media.json"
SLACK_S = 0.1  # one frame of rounding over the last step's end


def trim_clip(sample: Path, stage: Path, end_s: float) -> float:
    out = stage / "sample.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(sample / "sample.mp4"), "-t", f"{end_s:.3f}", "-an",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out)], check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=nw=1:nk=1", str(out)], check=True, capture_output=True, text=True)
    duration = round(float(probe.stdout), 3)
    if not 0 < duration <= end_s + SLACK_S:
        raise ValueError(f"trimmed clip is {duration}s; the last step ends at {end_s}s")
    return duration


def rebadge(sample: Path, stage: Path, guide: dict) -> None:
    """Redraw each thumbnail's badge as its part number, at the badge's existing spot."""
    for label in guide["labels"]:
        name = label["provisional"]["rendered_image_path"]
        width, height, frame = read_png_rgb(sample / guide["frame_paths"][label["slot"]["step_index"]])
        shown_w, shown_h, shown = read_png_rgb(sample / name)
        if (shown_w, shown_h) != (width, height):
            raise ValueError(f"{name}: dimensions {shown_w}x{shown_h} differ from its frame {width}x{height}")
        changed = [i // 3 for i in range(0, len(shown), 3) if shown[i:i + 3] != frame[i:i + 3]]
        xs, ys = [i % width for i in changed], [i // width for i in changed]
        pixels = bytearray(frame)
        draw_badge(pixels, width, height, ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2),
                   badge_radius(width, height), label["slot"]["part_index"] + 1)
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        write_png_rgb(stage / name, width, height, bytes(pixels))
        if read_png_rgb(stage / name)[:2] != (width, height):
            raise ValueError(f"{name}: written dimensions differ from its frame")


def write_manifest(sample: Path, media: Path, out: Path) -> Path:
    rows = ["# Approved sample media: sha256, bytes, name. Footage itself is never committed."]
    for name in sample_media.FILES:
        data = (media / name if (media / name).is_file() else sample / name).read_bytes()
        rows.append(f"{hashlib.sha256(data).hexdigest()}  {len(data)}  {name}")
    out.write_text("\n".join(rows) + "\n")
    sample_media.manifest(out)  # fails loudly if the staged manifest is off
    return out


def run(out: Path, sample: Path = SAMPLE) -> float:
    """Stage and validate every output in a temp dir next to ``out``, then rename it to ``out`` in one step.

    Originals are never modified. On an ordinary error no ``out`` and no temp dir are left. A run killed by a
    signal may leave a ``.trim-*`` temp dir or a complete ``out`` dir next to the target; delete it by hand."""
    out = Path(out).absolute()
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new --out directory")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"{out.parent} does not exist")
    guide = json.loads((sample / "guide.json").read_text())
    stage = Path(tempfile.mkdtemp(prefix=".trim-", dir=out.parent))
    try:
        media = stage / "app" / "sample_real"
        media.mkdir(parents=True)
        (stage / "config").mkdir()
        (stage / "var").mkdir()
        duration = trim_clip(sample, media, guide["steps"][-1]["end_s"])
        rebadge(sample, media, guide)
        write_manifest(sample, media, stage / "config" / "sample_media.sha256")
        (stage / "config" / "sample_media.json").write_text(json.dumps({"clip_duration_s": duration}, indent=1) + "\n")
        with zipfile.ZipFile(stage / "var" / "sample-media.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sample_media.FILES:
                archive.write(media / name if (media / name).is_file() else sample / name, name)
        os.rename(stage, out)
        print(f"clip trimmed to {duration}s; outputs are in {out}. Check them, then copy over the originals:")
        print(f"  cp -R {out}/. {ROOT}/")
        return duration
    finally:  # after a successful rename ``stage`` is gone, so this is a no-op
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="new directory to write; must not exist")
    try:
        run(parser.parse_args().out)
    except FileExistsError as error:
        sys.exit(f"error: {error}")
