"""The shipped sample clip stops at the guide's last step; part-N thumbnails carry badge N and the mask outline."""
from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import pytest

from app.render import badge_radius, draw_badge, draw_label_pixels, read_mask, read_png_rgb, restore_part_pixels, write_png_rgb
from tools import sample_media, trim_sample

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "app" / "sample_real"
MAX_CLIP_S = 189.5  # the reference ceiling: the last step ends at 189.233 s


def _guide() -> dict:
    return json.loads((SAMPLE / "guide.json").read_text())


def _need_media() -> None:
    if any(not (SAMPLE / n).is_file() for n in sample_media.FILES):
        pytest.skip("real sample media absent; run python tools/sample_media.py unpack")


def test_recorded_clip_duration_stops_at_the_last_step() -> None:
    recorded = json.loads((ROOT / "config" / "sample_media.json").read_text())["clip_duration_s"]
    assert recorded <= MAX_CLIP_S


def test_shipped_clip_has_nothing_after_the_last_step() -> None:
    _need_media()
    video = cv2.VideoCapture(str(SAMPLE / "sample.mp4"))
    try:
        seconds = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()
    recorded = json.loads((ROOT / "config" / "sample_media.json").read_text())["clip_duration_s"]
    assert seconds <= MAX_CLIP_S
    assert abs(seconds - recorded) <= 0.1


def test_each_thumbnail_badge_number_is_its_part_number() -> None:
    _need_media()
    guide = _guide()
    for label in guide["labels"]:
        frame_w, frame_h, frame = read_png_rgb(SAMPLE / guide["frame_paths"][label["slot"]["step_index"]])
        width, height, shown = read_png_rgb(SAMPLE / label["provisional"]["rendered_image_path"])
        assert (width, height) == (frame_w, frame_h)
        det = label["detection"]
        number = label["slot"]["part_index"] + 1
        mask = read_mask(SAMPLE / det["mask_path"], width, height)
        expected = draw_label_pixels(width, height, frame, det["box_xyxy"], mask, number)
        restore_part_pixels(expected, frame, mask)
        assert bytes(expected) == shown, f"{label['provisional']['rendered_image_path']} is not badge {number}"


# --- the tool itself, against a tiny generated fixture (no private media) ---

W, H = 320, 240
FIX_FILES = ("sample.mp4", "frames/step-01.png", "frames/step-02.png",
             "labels/step-02-part-1.png", "labels/step-02-part-2.png")


def _frame(shade: int) -> bytes:
    return bytes([shade, 80, 120]) * (W * H)


def _fixture(tmp: Path, monkeypatch, label_size=(W, H)) -> dict:
    sample = tmp / "sample_real"
    (sample / "frames").mkdir(parents=True)
    (sample / "labels").mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=10:duration=4",
                    "-pix_fmt", "yuv420p", str(sample / "sample.mp4")], check=True)
    for i, shade in enumerate((30, 60), 1):
        write_png_rgb(sample / f"frames/step-0{i}.png", W, H, _frame(shade))
    labels = []
    for part in (0, 1):
        pixels = bytearray(_frame(60))
        draw_badge(pixels, W, H, (100 + 60 * part, 90), badge_radius(W, H), 1)  # old badge: always 1
        name = f"labels/step-02-part-{part + 1}.png"
        write_png_rgb(sample / name, *label_size, bytes(pixels) if label_size == (W, H) else _frame(1)[:label_size[0] * label_size[1] * 3])
        labels.append({"slot": {"step_index": 1, "part_index": part},
                       "provisional": {"rendered_image_path": name}})
    guide = {"frame_paths": ["frames/step-01.png", "frames/step-02.png"], "labels": labels,
             "steps": [{"end_s": 1.0}, {"end_s": 2.0}]}
    (sample / "guide.json").write_text(json.dumps(guide))
    (tmp / "config").mkdir()
    (tmp / "config/sample_media.json").write_text('{"clip_duration_s": 4.0}\n')
    (tmp / "config/sample_media.sha256").write_text("# old\n")
    monkeypatch.setattr(sample_media, "FILES", FIX_FILES)
    return {"sample": sample, "out": tmp / "out"}


def _snapshot(tmp: Path) -> dict:
    return {str(p): p.read_bytes() for p in tmp.rglob("*") if p.is_file()}


def _clean(tmp: Path, before: dict) -> None:
    assert _snapshot(tmp) == before  # every original byte unchanged, nothing new anywhere
    assert not (tmp / "out").exists() and not list(tmp.glob(".trim-*"))


def test_tool_writes_a_fresh_dir_and_leaves_every_original_unchanged(tmp_path, monkeypatch, capsys) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)
    out = paths["out"]
    duration = trim_sample.run(**paths)
    assert 1.9 <= duration <= 2.1  # last step ends at 2.0 s; the source was 4 s
    assert all(before[k] == v.read_bytes() for k, v in ((str(p), p) for p in tmp_path.rglob("*") if p.is_file())
               if k in before)
    assert json.loads((out / "config/sample_media.json").read_text()) == {"clip_duration_s": duration}
    for part in (0, 1):
        _, _, shown = read_png_rgb(out / f"app/sample_real/labels/step-02-part-{part + 1}.png")
        expected = bytearray(_frame(60))
        draw_badge(expected, W, H, (100 + 60 * part, 90), badge_radius(W, H), part + 1)
        assert bytes(expected) == shown
    rows = sample_media.manifest(out / "config/sample_media.sha256")
    for name in FIX_FILES:
        shipped = out / "app/sample_real" / name
        data = (shipped if shipped.is_file() else paths["sample"] / name).read_bytes()
        assert rows[name] == (hashlib.sha256(data).hexdigest(), len(data))
    with zipfile.ZipFile(out / "var/sample-media.zip") as archive:
        assert set(archive.namelist()) == set(FIX_FILES)
    assert not list(tmp_path.glob(".trim-*"))
    assert f"cp -R {out}/. " in capsys.readouterr().out  # the copy command is printed


def test_tool_refuses_an_existing_out_dir(tmp_path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["out"].mkdir()
    (paths["out"] / "keep.txt").write_text("mine")
    before = _snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        trim_sample.run(**paths)
    assert _snapshot(tmp_path) == before and not list(tmp_path.glob(".trim-*"))


@pytest.mark.parametrize("step", ["trim_clip", "rebadge", "write_manifest"])
def test_tool_creates_nothing_when_a_step_fails(tmp_path, monkeypatch, step) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)

    def boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(trim_sample, step, boom)
    with pytest.raises(OSError):
        trim_sample.run(**paths)
    _clean(tmp_path, before)


def test_tool_creates_nothing_when_the_final_rename_fails(tmp_path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)

    def boom(*_a, **_k):
        raise OSError("I/O error")
    monkeypatch.setattr(trim_sample.os, "rename", boom)
    with pytest.raises(OSError):
        trim_sample.run(**paths)
    _clean(tmp_path, before)


_KILLED_RUN = """
import sys, time
from pathlib import Path
from tools import sample_media, trim_sample
sample_media.FILES = {files!r}
def hang(*_a, **_k):
    Path({ready!r}).write_text("up")
    time.sleep(60)
trim_sample.rebadge = hang
trim_sample.run(Path({out!r}), Path({sample!r}))
"""


def test_originals_are_unchanged_when_the_run_is_killed_by_sigterm(tmp_path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)
    ready = tmp_path.parent / f"{tmp_path.name}-ready"
    code = _KILLED_RUN.format(files=FIX_FILES, ready=str(ready), out=str(paths["out"]), sample=str(paths["sample"]))
    child = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():  # the child's own record that it is mid-run
            assert child.poll() is None and time.monotonic() < deadline, "child never reached the hang point"
            time.sleep(0.05)
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=30)
    finally:
        child.kill()
    # a killed run may leave a .trim-* or out dir (documented); no original may change
    assert all(p.read_bytes() == data for p, data in ((Path(k), v) for k, v in before.items()))


def test_tool_fails_closed_when_a_label_size_differs_from_its_frame(tmp_path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch, label_size=(W // 2, H // 2))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        trim_sample.run(**paths)
    _clean(tmp_path, before)
