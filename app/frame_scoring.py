"""Local frame sampling and sharpness/motion ranking.

Samples 2 fps inside each step window, ranks candidates by Laplacian
sharpness discounted by inter-frame motion, and keeps the best two real
frames per step. Person-occlusion logic stays out: it joins only if a
measured fixture proves it necessary.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.contracts import Step
from app.sam import publish_staged

SAMPLE_FPS = 2.0
FRAMES_KEPT_PER_STEP = 2


def sample_times(
    start_s: float, end_s: float, fps: float = SAMPLE_FPS
) -> tuple[float, ...]:
    """2 fps grid inside ``[start_s, end_s]``; every stamp stays in the window."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if end_s < start_s:
        raise ValueError(f"empty window: start {start_s} > end {end_s}")
    count = int((end_s - start_s) * fps)
    return tuple(min(start_s + k / fps, end_s) for k in range(count + 1))


@dataclass(frozen=True)
class ScoredFrame:
    """One candidate frame with its sharpness and motion measurements."""

    timestamp_s: float
    sharpness: float
    motion: float

    @property
    def score(self) -> float:
        """Sharpness discounted by motion; higher wins."""
        return self.sharpness / (1.0 + self.motion)


def laplacian_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian; higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean absolute pixel difference between consecutive candidates."""
    return float(cv2.mean(cv2.absdiff(prev_gray, curr_gray))[0])


def combined_score(sharpness: float, motion: float) -> float:
    """Rank value for one candidate; higher wins."""
    return ScoredFrame(timestamp_s=0.0, sharpness=sharpness, motion=motion).score


def pick_best_two(frames: Sequence[ScoredFrame]) -> tuple[ScoredFrame, ...]:
    """Top two by score, best first; ties break toward the earlier stamp."""
    ranked = sorted(frames, key=lambda f: (-f.score, f.timestamp_s))
    return tuple(ranked[:FRAMES_KEPT_PER_STEP])


def score_frames(
    video_path: Path, steps: Sequence[Step]
) -> Mapping[int, tuple[Path, ...]]:
    """Best two real frames per step window, keyed by step number.

    Winners are written as PNGs into ``<video-stem>_frames/`` beside the
    video, so they live under the job directory and leave with it. A window
    with no decodable frame maps to an empty tuple, as does a window whose
    frames cannot be written (earlier files stay untouched; only the
    window's temporary files are removed).
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        out_dir = video_path.with_name(video_path.stem + "_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            step.number: _best_two_for_window(capture, out_dir, step)
            for step in steps
        }
    finally:
        capture.release()


def _best_two_for_window(
    capture: cv2.VideoCapture, out_dir: Path, step: Step
) -> tuple[Path, ...]:
    grays: list[tuple[float, np.ndarray]] = []
    for stamp in sample_times(step.start_s, step.end_s):
        capture.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        grays.append((stamp, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    scored: list[ScoredFrame] = []
    prev: np.ndarray | None = None
    for stamp, gray in grays:
        motion = 0.0 if prev is None else frame_motion(prev, gray)
        scored.append(
            ScoredFrame(
                timestamp_s=stamp,
                sharpness=laplacian_sharpness(gray),
                motion=motion,
            )
        )
        prev = gray
    staged: list[tuple[Path, Path]] = []
    complete = False
    try:
        for rank, winner in enumerate(pick_best_two(scored)):
            capture.set(cv2.CAP_PROP_POS_MSEC, winner.timestamp_s * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                return ()
            dest = out_dir / f"step{step.number}_t{winner.timestamp_s:.2f}s_{rank}.png"
            fd, tmp_name = tempfile.mkstemp(
                prefix=dest.name + ".", suffix=".tmp.png", dir=out_dir
            )
            tmp = Path(tmp_name)
            staged.append((tmp, dest))
            os.close(fd)
            if not cv2.imwrite(str(tmp), frame):
                return ()
        publish_staged(staged)
        complete = True
        return tuple(dest for _tmp, dest in staged)
    finally:
        if not complete:
            for staged_tmp, _dest in staged:
                staged_tmp.unlink(missing_ok=True)


__all__ = (
    "FRAMES_KEPT_PER_STEP",
    "SAMPLE_FPS",
    "ScoredFrame",
    "combined_score",
    "frame_motion",
    "laplacian_sharpness",
    "pick_best_two",
    "sample_times",
    "score_frames",
)
