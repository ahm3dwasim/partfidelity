"""burned-in clock overlay and scene-cut candidates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.clock import (
    ClockBurnError,
    burn_clock_file,
    burn_clock_in_place,
    clock_text_for_time,
    draw_clock,
)
from app.scene_cuts import (
    SceneCutError,
    detect_scene_cuts,
    scene_cuts_filename,
    write_window_cuts,
)
from app.service import write_sample_clip


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory) -> Path:
    return write_sample_clip(
        tmp_path_factory.mktemp("media") / "tiny.mp4", seconds=3.0, fps=5
    )


def _frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def _decoded_duration(path: Path) -> float:
    """Independent oracle: raw ffprobe format duration."""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_entries", "format=duration", str(path),
        ],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


# --- clock text ------------------------------------------------------------


@pytest.mark.parametrize(
    "time_s, text",
    [
        (0.0, "0:00.0"),
        (9.4, "0:09.4"),
        (60.0, "1:00.0"),
        (65.5, "1:05.5"),
        (214.17, "3:34.1"),
    ],
)
def test_clock_text_for_time(time_s: float, text: str) -> None:
    assert clock_text_for_time(time_s) == text


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "65"])
def test_clock_text_rejects_bad_times(bad) -> None:
    with pytest.raises(ValueError):
        clock_text_for_time(bad)


def test_draw_clock_returns_the_text_it_draws() -> None:
    frame = np.zeros((240, 320, 3), np.uint8)
    assert draw_clock(frame, 65.5) == "1:05.5"
    assert frame.sum() > 0  # the white glyphs landed on the black frame
    with pytest.raises(ValueError):
        draw_clock(np.zeros((240, 320), np.uint8), 1.0)
    with pytest.raises(ValueError):
        draw_clock(frame, -1.0)


# --- clock burn ------------------------------------------------------------


def test_burn_keeps_duration_and_frame_count(tiny_clip: Path, tmp_path: Path) -> None:
    dst = tmp_path / "clocked.mp4"
    frames = burn_clock_file(tiny_clip, dst, timeout_s=120)
    assert frames == 15
    assert _frame_count(dst) == _frame_count(tiny_clip)
    assert _fps(dst) == pytest.approx(_fps(tiny_clip))
    # The pipeline probes the stream-copy cut before the burn, so the
    # burn must preserve the decoded duration: exact on whole-frame
    # cuts, container rounding (measured <= 0.4 ms) otherwise — far
    # below the 2 s consensus tolerance either way.
    assert (
        abs(_decoded_duration(dst) - _decoded_duration(tiny_clip)) < 0.05
    )
    assert dst.stat().st_size > 0


def test_burn_draws_the_right_time_per_frame(
    tiny_clip: Path, tmp_path: Path
) -> None:
    # OCR-free: the burned frame N differs from the source exactly in
    # the clock pad, and the text for N/fps is the drawing input.
    dst = tmp_path / "clocked.mp4"
    burn_clock_file(tiny_clip, dst, timeout_s=120)
    fps = _fps(tiny_clip)
    src_cap = cv2.VideoCapture(str(tiny_clip))
    dst_cap = cv2.VideoCapture(str(dst))
    try:
        for index, expected in ((0, "0:00.0"), (7, "0:01.4"), (14, "0:02.8")):
            src_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            dst_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, src_frame = src_cap.read()
            _, dst_frame = dst_cap.read()
            # The drawing input for frame N is exactly N/fps.
            assert clock_text_for_time(index / fps) == expected
            corner = np.abs(
                dst_frame[:110, :220].astype(int) - src_frame[:110, :220].astype(int)
            ).mean()
            assert corner > 1.0  # the clock pad changed the corner
    finally:
        src_cap.release()
        dst_cap.release()


def test_burn_in_place_replaces_atomically(
    tiny_clip: Path, tmp_path: Path
) -> None:
    import shutil

    target = tmp_path / "chunk_000.mp4"
    shutil.copy(tiny_clip, target)
    frames = burn_clock_in_place(target, timeout_s=120)
    assert frames == 15
    assert _frame_count(target) == 15
    assert list(tmp_path.glob("*.clocktmp.mp4")) == []


def test_burn_rejects_bad_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        burn_clock_file(tmp_path / "absent.mp4", tmp_path / "o.mp4", timeout_s=0)
    with pytest.raises(ClockBurnError):
        burn_clock_file(tmp_path / "absent.mp4", tmp_path / "o.mp4", timeout_s=30)
    with pytest.raises(ValueError):
        burn_clock_in_place(tmp_path / "absent.mp4", timeout_s=-1)


def test_burn_failure_leaves_no_output(tiny_clip: Path, tmp_path: Path) -> None:
    dst = tmp_path / "clocked.mp4"
    with pytest.raises(ClockBurnError):
        burn_clock_file(tiny_clip / "not-a-file", dst, timeout_s=30)
    assert not dst.exists()


# --- scene cuts ------------------------------------------------------------


def _hard_cut_clip(path: Path, *, seconds: float = 6.0, fps: int = 5) -> Path:
    """Black then white halves: exactly one hard cut halfway."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    count = int(seconds * fps)
    try:
        for index in range(count):
            if index < count // 2:
                writer.write(np.zeros((240, 320, 3), np.uint8))
            else:
                writer.write(np.full((240, 320, 3), 255, np.uint8))
    finally:
        writer.release()
    return path


def test_detect_scene_cuts_finds_a_hard_cut(tmp_path: Path) -> None:
    clip = _hard_cut_clip(tmp_path / "hard.mp4")
    assert detect_scene_cuts(clip, 0.15, timeout_s=60) == (3.0,)


def test_detect_scene_cuts_empty_on_smooth_motion(tiny_clip: Path) -> None:
    assert detect_scene_cuts(tiny_clip, 0.15, timeout_s=60) == ()


def test_detect_scene_cuts_rejects_bad_inputs(
    tiny_clip: Path, tmp_path: Path
) -> None:
    for bad in (0.0, -0.1, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            detect_scene_cuts(tiny_clip, bad, timeout_s=60)
    with pytest.raises(ValueError):
        detect_scene_cuts(tiny_clip, 0.15, timeout_s=0)
    with pytest.raises(SceneCutError):
        detect_scene_cuts(tmp_path / "absent.mp4", 0.15, timeout_s=30)


def test_scene_cuts_filename() -> None:
    assert scene_cuts_filename(0) == "scene_cuts_000.json"
    assert scene_cuts_filename(12) == "scene_cuts_012.json"
    for bad in (-1, True, "0"):
        with pytest.raises(ValueError):
            scene_cuts_filename(bad)


def test_write_window_cuts_round_trip(tmp_path: Path) -> None:
    path = write_window_cuts(tmp_path, 2, 116.0, 0.15, (9.4, 17.3))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "window_index": 2,
        "window_start_s": 116.0,
        "threshold": 0.15,
        "cuts": [9.4, 17.3],
    }
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(ValueError):
        write_window_cuts(tmp_path, -1, 0.0, 0.15, ())
