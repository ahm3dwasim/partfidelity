"""Hide identifiers and screens in the public sample's media.

Reads ``config/redactions.toml`` and writes redacted copies of the sample guide stills, the part-N thumbnails, the
clip and the /how-it-works stills into a NEW directory laid out like the repository, with ``config/sample_media.sha256``,
``config/how_media.json``, ``var/sample-media.zip`` and ``assets/how/how-frames.tar`` rebuilt to match. It prints the
``cp`` command a human runs after checking them. Originals are never modified.

On an ordinary error no ``--out`` dir and no temp dir are left. A run killed by a signal may leave a ``.redact-*``
temp dir, or a complete ``--out`` dir, next to the target; delete it by hand.

    python3 tools/redact_sample.py --out /tmp/redacted
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import how_media  # noqa: E402
from app.render import read_png_rgb, write_png_rgb  # noqa: E402
from tools import redactions, sample_media  # noqa: E402

SLACK_S = 0.1  # the clip keeps its length to within a few frames
FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # archive members carry no source timestamps, so two runs hash alike


def under(base: Path, name: str) -> Path:
    """``base/name``, refused if it leaves ``base`` (absolute, ``..`` or a symlink out); the loader already checks the spelling."""
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"{name}: leaves {base}")
    return base / name


def _tar_member(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
    return info


def add_zip_member(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
    archive.writestr(info, path.read_bytes())


def _caption_layout(width: int, height: int, style: dict):
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, width // 640)
    (text_w, text_h), _ = cv2.getTextSize(style["caption"], font, 1.0, thickness)
    scale = width * style["caption_width_fraction"] / text_w
    (text_w, text_h), baseline = cv2.getTextSize(style["caption"], font, scale, thickness)
    return font, thickness, scale, (width - text_w) // 2, (height + text_h) // 2, text_w, text_h, baseline


def panel(width: int, height: int, style: dict) -> np.ndarray:
    """A plain neutral frame with the config caption centred on it (RGB)."""
    image = np.full((height, width, 3), style["panel_rgb"], np.uint8)
    font, thickness, scale, x, y, _, _, _ = _caption_layout(width, height, style)
    cv2.putText(image, style["caption"], (x, y), font, scale, tuple(style["caption_rgb"]), thickness, cv2.LINE_AA)
    return image


def caption_box(width: int, height: int, style: dict) -> tuple[int, int, int, int]:
    """``x0, y0, x1, y1`` (end exclusive, clipped to the frame) of the caption text plus ``caption_margin_px`` on every side."""
    _, thickness, _, x, y, text_w, text_h, baseline = _caption_layout(width, height, style)
    m = style["caption_margin_px"] + thickness
    return max(0, x - m), max(0, y - text_h - m), min(width, x + text_w + m), min(height, y + baseline + m)


def blur_regions(image: np.ndarray, regions: list, sigma: float, name: str) -> np.ndarray:
    """Blur each rectangle ``[x, y, w, h]`` or quad (8 corner numbers) in place of its own pixels; nothing outside moves."""
    out = image.copy()
    height, width = image.shape[:2]
    for region in regions:
        if len(region) == 4:
            x, y, w, h = region
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            corners = list(zip(region[::2], region[1::2]))
        x0, y0 = min(c[0] for c in corners), min(c[1] for c in corners)
        x1, y1 = max(c[0] for c in corners), max(c[1] for c in corners)
        if x1 > width or y1 > height:
            raise ValueError(f"{name}: region {region} leaves the {width}x{height} image")
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.array([(cx - x0, cy - y0) for cx, cy in corners], np.int32), 1)
        blurred = cv2.GaussianBlur(image[y0:y1, x0:x1], (0, 0), sigma)
        out[y0:y1, x0:x1][mask == 1] = blurred[mask == 1]
    return out


def _still(row: dict, style: dict, image: np.ndarray) -> np.ndarray:
    if row.get("hide_screen"):
        return panel(image.shape[1], image.shape[0], style)
    return blur_regions(image, row["blur"], style["blur_sigma"], row["file"])


def _read(path: Path) -> np.ndarray:
    width, height, data = read_png_rgb(path)
    return np.frombuffer(data, np.uint8).reshape(height, width, 3)


def redact_guide_stills(sample: Path, media: Path, config: dict) -> None:
    """Frames named in the config, then each part-N thumbnail: the redacted frame plus the badge the thumbnail added."""
    guide = json.loads((sample / "guide.json").read_text())
    frames = {}
    for row in config["still"]:
        original = _read(under(sample, row["file"]))
        frames[row["file"]] = (original, _still(row, config["style"], original))
        out = under(media, row["file"])
        out.parent.mkdir(parents=True, exist_ok=True)
        write_png_rgb(out, original.shape[1], original.shape[0], frames[row["file"]][1].tobytes())
    for label in guide["labels"]:
        name = label["provisional"]["rendered_image_path"]
        under(sample, name), under(media, name)  # every guide-derived path is checked before any read or write
        frame_name = guide["frame_paths"][label["slot"]["step_index"]]
        if frame_name not in frames:
            continue
        original_frame, redacted_frame = frames[frame_name]
        shown = _read(under(sample, name))
        if shown.shape != original_frame.shape:
            raise ValueError(f"{name}: dimensions differ from its frame {frame_name}")
        badge = (shown != original_frame).any(axis=2)[..., None]
        out = under(media, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_png_rgb(out, shown.shape[1], shown.shape[0], np.where(badge, shown, redacted_frame).tobytes())


def box_at(box: dict, t: float) -> np.ndarray | None:
    """The box's quad (4x2) at time ``t``, linear between its keyframes; None outside the first and last keyframe."""
    frames = box["keyframes"]
    for (ta, *qa), (tb, *qb) in zip(frames, frames[1:]):
        if ta <= t <= tb:
            return (np.array(qa) + (np.array(qb) - np.array(qa)) * (t - ta) / (tb - ta)).reshape(4, 2)
    return None


def live_boxes(config: dict, t: float) -> list[np.ndarray]:
    """The quads live at time ``t``."""
    return [quad for box in config["clip_box"] if (quad := box_at(box, t)) is not None]


def redact_frame(frame: np.ndarray, t: float, config: dict, hidden: np.ndarray) -> np.ndarray:
    """One RGB frame: the panel inside a hidden span, else each live box filled with the mean colour under it."""
    if any(h["start_s"] <= t <= h["end_s"] for h in config["clip_hide"]):
        return hidden
    for quad in live_boxes(config, t):
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
        if mask.any():
            frame[mask == 1] = np.array(frame[mask == 1].mean(axis=0), np.uint8)
    return frame


def redact_clip(sample: Path, media: Path, config: dict) -> float:
    """Decode, redact and re-encode frame by frame: boxes move and change shape per frame, which ffmpeg filters cannot."""
    source, out = sample / "sample.mp4", media / "sample.mp4"
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(source)],
                           check=True, capture_output=True, text=True)
    width, height, rate = probe.stdout.strip().split(",")
    width, height = int(width), int(height)
    numerator, denominator = map(int, rate.split("/"))
    hidden = panel(width, height, config["style"])
    size, count = width * height * 3, 0
    procs = []
    try:
        procs.append(subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(source), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                      stdout=subprocess.PIPE))
        reader = procs[0]
        procs.append(subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                       "-r", rate, "-i", "-", "-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                                       "-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(out)],
                                      stdin=subprocess.PIPE))
        writer = procs[1]
        while len(raw := reader.stdout.read(size)) == size:
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
            writer.stdin.write(redact_frame(frame, count * denominator / numerator, config, hidden).tobytes())
            count += 1
    except BaseException:  # a launch or a pipe failed: stop whatever started, never leave a decoder behind
        for proc in procs:
            proc.kill()
        raise
    finally:
        for proc in procs:
            for pipe in (proc.stdout, proc.stdin):
                if pipe:
                    with contextlib.suppress(OSError):
                        pipe.close()
            proc.wait()
    if reader.returncode or writer.returncode or not count:
        raise RuntimeError(f"ffmpeg failed (decode {reader.returncode}, encode {writer.returncode}, {count} frames)")
    lengths = []
    for path in (source, out):
        shown = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                                str(path)], check=True, capture_output=True, text=True)
        lengths.append(round(float(shown.stdout), 3))
    if abs(lengths[0] - lengths[1]) > SLACK_S:
        raise ValueError(f"redacted clip is {lengths[1]}s; the source is {lengths[0]}s")
    return lengths[1]


def _frames(path: Path, width: int, height: int):
    """Every RGB frame of ``path`` in order; the decoder is stopped if the caller stops early."""
    size = width * height * 3
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    try:
        while len(raw := proc.stdout.read(size)) == size:
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    finally:
        proc.kill()
        proc.stdout.close()
        proc.wait()


def check_keyframes(config: dict, width: int, height: int) -> None:
    """Every keyframe quad must still cover pixels inside the probed frame after clipping and the 5x5 erosion (a quad may overhang an edge)."""
    for number, box in enumerate(config["clip_box"]):
        for t, *points in box["keyframes"]:
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, np.round(np.array(points, float).reshape(4, 2)).astype(np.int32), 1)
            if not cv2.erode(mask, np.ones((5, 5), np.uint8)).any():
                raise ValueError(f"clip_box {number} at {t}s: the quad is off the {width}x{height} frame or too small to check")


CUT_DIFF = 25.0       # mean abs difference between consecutive frames that counts as a scene cut (a real cut is 60+, motion is under 5)
CUT_LOOKBACK_S = 1.0  # a source cut this long before a hidden span, or inside its first CUT_MARGIN_FRAMES frames, is uncovered
CUT_MARGIN_FRAMES = 2


def verify_clip(source: Path, built: Path, config: dict, tolerance: float = 8.0) -> int:
    """Decode the built clip and prove the masks are really there. Returns the number of frames checked; raises ValueError on the first gap.

    ``clip_check`` windows: at least ``min_boxes`` boxes are live (or the screen panel is), and the pixels inside each box (shrunk 2 px
    for codec bleed) match what the redaction fills there.
    ``clip_hide`` spans: every frame inside is the panel: no pixel outside the caption box is more than ``panel_noise_tolerance`` from the panel colour, and the caption box matches the rendered caption (mean difference under ``tolerance``), every span is
    sampled, and no scene cut falls in the ``CUT_LOOKBACK_S`` before a span starts or in its first ``CUT_MARGIN_FRAMES`` frames,
    because that cut would show the hidden screen before the panel covers it.
    """
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                            "-of", "csv=p=0", str(built)], check=True, capture_output=True, text=True)
    width, height, rate = probe.stdout.strip().split(",")
    width, height = int(width), int(height)
    numerator, denominator = map(int, rate.split("/"))
    frame_s = denominator / numerator
    check_keyframes(config, width, height)
    hidden = panel(width, height, config["style"])
    box = caption_box(width, height, config["style"])
    windows, spans = config["clip_check"], config["clip_hide"]
    last = max(w["end_s"] for w in windows + spans)
    checked, in_span, previous = 0, [0] * len(spans), None
    for count, (raw, done) in enumerate(zip(_frames(source, width, height), _frames(built, width, height))):
        t = count * frame_s
        if t > last:
            break
        if previous is not None and np.abs(raw.astype(int) - previous).mean() > CUT_DIFF:
            for number, span in enumerate(spans):
                if span["start_s"] - CUT_LOOKBACK_S <= t < span["start_s"] + CUT_MARGIN_FRAMES * frame_s:
                    raise ValueError(f"frame {count} ({t:.3f}s): a scene cut is not covered; hidden span {number} starts at {span['start_s']}s "
                                     f"and needs {CUT_MARGIN_FRAMES} frames of margin before the cut")
        previous = raw.astype(int)
        hide = next((n for n, h in enumerate(spans) if h["start_s"] <= t <= h["end_s"]), None)
        window = next((w for w in windows if w["start_s"] <= t <= w["end_s"]), None)
        if hide is None and window is None:
            continue
        checked += 1
        if hide is not None:
            in_span[hide] += 1
            off = np.abs(done.astype(int) - hidden)
            x0, y0, x1, y1 = box
            outside = np.ones((height, width), bool)
            outside[y0:y1, x0:x1] = False
            if (off.max(axis=2)[outside] > config["style"]["panel_noise_tolerance"]).any() or off[y0:y1, x0:x1].mean() > tolerance:
                raise ValueError(f"frame {count} ({t:.3f}s): the screen panel is not applied")
            continue
        quads = live_boxes(config, t)
        if len(quads) < window["min_boxes"]:
            raise ValueError(f"frame {count} ({t:.3f}s): {len(quads)} boxes live, window needs {window['min_boxes']}")
        expected = redact_frame(raw.copy(), t, config, hidden)
        for quad in quads:
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            if not mask.any():
                raise ValueError(f"frame {count} ({t:.3f}s): a live box is empty after clipping and erosion")
            if np.abs(done[mask == 1].astype(int) - expected[mask == 1]).mean() > tolerance:
                raise ValueError(f"frame {count} ({t:.3f}s): a box region is not filled")
    for number, seen in enumerate(in_span):
        if not seen:
            raise ValueError(f"clip_hide {number}: no frame of the built clip falls inside the span")
    if not checked:
        raise ValueError("clip_check windows matched no frame")
    return checked


def redact_how_stills(root: Path, stage: Path, config: dict) -> None:
    """Redact the listed /how-it-works stills, then rebuild the manifest and the private bundle to match."""
    cfg = how_media.load(root)
    how = root / cfg["dir"]
    target = stage / cfg["dir"]
    target.mkdir(parents=True)
    quality = [cv2.IMWRITE_WEBP_QUALITY, config["style"]["webp_quality"]]
    for row in config["how_still"]:
        image = cv2.imread(str(under(how, row["file"])), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"{how / row['file']}: cannot read the still")
        done = blur_regions(image[..., ::-1], row["blur"], config["style"]["blur_sigma"], row["file"])[..., ::-1]
        if not cv2.imwrite(str(under(target, row["file"])), np.ascontiguousarray(done), quality):
            raise OSError(f"{row['file']}: cannot write the redacted still")
    for entry in cfg["files"]:
        data = (target / entry["name"] if (target / entry["name"]).is_file() else how / entry["name"]).read_bytes()
        entry.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    (stage / how_media.MANIFEST).write_text(json.dumps(cfg, indent=1) + "\n")
    bundle = stage / cfg["bundle"]
    bundle.parent.mkdir(parents=True)
    with tarfile.open(bundle, "w") as tar:
        for entry in cfg["files"]:
            tar.add(target / entry["name"] if (target / entry["name"]).is_file() else how / entry["name"],
                    arcname=entry["name"], filter=_tar_member)


def write_manifest(sample: Path, media: Path, out: Path) -> None:
    rows = ["# Approved sample media: sha256, bytes, name. Footage itself is never committed."]
    for name in sample_media.FILES:
        data = (media / name if (media / name).is_file() else sample / name).read_bytes()
        rows.append(f"{hashlib.sha256(data).hexdigest()}  {len(data)}  {name}")
    out.write_text("\n".join(rows) + "\n")
    sample_media.manifest(out)  # fails loudly if the staged manifest is off


def run(out: Path, root: Path = ROOT) -> float:
    """Stage and validate every output in a temp dir next to ``out``, then rename it to ``out`` in one step."""
    out = Path(out).absolute()
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new --out directory")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"{out.parent} does not exist")
    sample = root / "app" / "sample_real"
    config = redactions.load(root / "config" / "redactions.toml")
    stage = Path(tempfile.mkdtemp(prefix=".redact-", dir=out.parent))
    try:
        media = stage / "app" / "sample_real"
        media.mkdir(parents=True)
        (stage / "config").mkdir()
        (stage / "var").mkdir()
        redact_guide_stills(sample, media, config)
        duration = redact_clip(sample, media, config)
        verify_clip(sample / "sample.mp4", media / "sample.mp4", config)
        write_manifest(sample, media, stage / "config" / "sample_media.sha256")
        with zipfile.ZipFile(stage / "var" / "sample-media.zip", "w") as archive:
            for name in sample_media.FILES:
                add_zip_member(archive, media / name if (media / name).is_file() else sample / name, name)
        redact_how_stills(root, stage, config)
        os.rename(stage, out)
        print(f"redacted clip is {duration}s; outputs are in {out}. Check them, then copy over the originals:")
        print(f"  cp -R {out}/. {root}/")
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
