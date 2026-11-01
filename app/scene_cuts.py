"""Scene-cut candidates: one ffmpeg pass per extraction window.

``select='gt(scene,T)'`` with the configured threshold finds the hard
cuts a step boundary very often lands on (measured 7/8 true edges of the sample video
within 3 s at 0.15); the cut times index the uploaded chunk bytes, so
they are window-relative. Each window's cuts are stored beside the
job for diagnosis; consensus snaps to them and scores confidence
from them in memory. A failed pass raises instead of refining
against cuts that were never measured.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

# One file per window (never one shared file): chunks of one job can
# run on different workers, so a shared read-modify-write would race.
# Unregistered job-dir files, kept like the guide itself.
SCENE_CUTS_FILENAME_PATTERN = "scene_cuts_{index:03d}.json"

_PTS_TIME_RE = re.compile(r"pts_time:([0-9.]+)")


class SceneCutError(RuntimeError):
    """The scene-cut pass failed (ffmpeg missing, failed, or too slow)."""


def scene_cuts_filename(index: int) -> str:
    """Storage name of one window's cuts (never served)."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"cuts index must be an int >= 0, got {index!r}")
    return SCENE_CUTS_FILENAME_PATTERN.format(index=index)


def detect_scene_cuts(
    path: Path, threshold: float, *, timeout_s: float
) -> tuple[float, ...]:
    """Return the window-relative scene-cut times, ascending and unique.

    A cut at a threshold of 0 would fire on every frame, so the
    threshold must sit strictly inside (0, 1].
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError(
            "scene-cut threshold must be a finite number in (0, 1],"
            f" got {threshold!r}"
        )
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or not timeout_s > 0
    ):
        raise ValueError(
            f"timeout_s must be a finite number > 0, got {timeout_s!r}"
        )
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(path),
                "-map",
                "0:v",
                "-filter:v",
                f"select='gt(scene,{threshold})',showinfo",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SceneCutError(f"could not detect scene cuts in {path}") from error
    if completed.returncode != 0:
        raise SceneCutError(f"could not detect scene cuts in {path}")
    cuts = {
        float(match.group(1))
        for line in completed.stderr.splitlines()
        if "pts_time:" in line
        for match in [_PTS_TIME_RE.search(line)]
        if match is not None
    }
    return tuple(sorted(cuts))


def write_window_cuts(
    job_dir: Path,
    index: int,
    window_start_s: float,
    threshold: float,
    cuts: tuple[float, ...],
) -> Path:
    """Store one window's cuts atomically; return the file path."""
    if (
        isinstance(window_start_s, bool)
        or not isinstance(window_start_s, (int, float))
        or not math.isfinite(window_start_s)
        or not window_start_s >= 0
    ):
        raise ValueError(
            "window_start_s must be a finite number >= 0,"
            f" got {window_start_s!r}"
        )
    name = scene_cuts_filename(index)  # validates the index
    record = {
        "window_index": index,
        "window_start_s": float(window_start_s),
        "threshold": float(threshold),
        "cuts": [float(cut) for cut in cuts],
    }
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    dst = job_dir / name
    tmp = dst.with_name(name + ".tmp")
    tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


__all__ = (
    "SCENE_CUTS_FILENAME_PATTERN",
    "SceneCutError",
    "detect_scene_cuts",
    "scene_cuts_filename",
    "write_window_cuts",
)
