"""Burned-in extraction clock: overlay m:ss.s on every frame before upload.

The box's ffmpeg has no drawtext filter (no libfreetype), so the clock
is drawn with OpenCV (large white on black, top-left) and piped as
raw frames to ffmpeg/libx264 for encoding — the approach the
whole-video probe used. The overlaid times are window-relative: each
chunk's clock starts at 0:00.0. Output stays audio-free like the
preview it derives from. A failed burn raises instead of uploading
unclocked bytes behind a clock-reading prompt.
"""

from __future__ import annotations

import contextlib
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Clock look (probe-whole's, readable at 720p): large duplex glyphs,
# white on a black pad, top-left with a fixed margin.
CLOCK_FONT = cv2.FONT_HERSHEY_DUPLEX
CLOCK_SCALE = 2.2
CLOCK_THICKNESS = 4
CLOCK_MARGIN_X = 30
CLOCK_MARGIN_Y = 30
CLOCK_PAD = 12
CLOCK_TEXT_COLOR = (255, 255, 255)
CLOCK_BACK_COLOR = (0, 0, 0)
# Encode (probe-whole's): fast, visually lossless, browser-safe pixels.
ENCODER_PRESET = "veryfast"
ENCODER_CRF = "20"


class ClockBurnError(RuntimeError):
    """The clock overlay failed (unreadable input, ffmpeg missing/failed/slow)."""


def clock_text_for_time(time_s: float) -> str:
    """Format ``time_s`` as the overlaid ``m:ss.s`` clock text (truncated)."""
    if (
        isinstance(time_s, bool)
        or not isinstance(time_s, (int, float))
        or not math.isfinite(time_s)
        or not time_s >= 0
    ):
        raise ValueError(
            f"clock time must be a finite number >= 0, got {time_s!r}"
        )
    moment = float(time_s)
    return (
        f"{int(moment // 60)}:{int(moment % 60):02d}"
        f".{int((moment * 10) % 10)}"
    )


def draw_clock(frame: np.ndarray, time_s: float) -> str:
    """Draw the clock for ``time_s`` onto ``frame``; return the text drawn."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError(
            "draw_clock needs a BGR frame (height x width x 3 array),"
            f" got {type(frame).__name__}"
        )
    text = clock_text_for_time(time_s)
    (text_w, text_h), _ = cv2.getTextSize(
        text, CLOCK_FONT, CLOCK_SCALE, CLOCK_THICKNESS
    )
    x, y = CLOCK_MARGIN_X, CLOCK_MARGIN_Y
    cv2.rectangle(
        frame,
        (x - CLOCK_PAD, y - CLOCK_PAD),
        (x + text_w + CLOCK_PAD, y + text_h + CLOCK_PAD),
        CLOCK_BACK_COLOR,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y + text_h),
        CLOCK_FONT,
        CLOCK_SCALE,
        CLOCK_TEXT_COLOR,
        CLOCK_THICKNESS,
        cv2.LINE_AA,
    )
    return text


def _require_timeout(timeout_s: float) -> float:
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or not timeout_s > 0
    ):
        raise ValueError(
            f"timeout_s must be a finite number > 0, got {timeout_s!r}"
        )
    return float(timeout_s)


def burn_clock_file(src: Path, dst: Path, *, timeout_s: float) -> int:
    """Overlay the window-relative clock on ``src``; write ``dst``.

    Every source frame is drawn and encoded in order, so the output
    keeps the input's duration, frame count, size, and rate, minus
    audio. Returns the frames written. A partial output is unlinked
    best-effort, so a failed burn never leaves bytes the caller
    mistakes for a clocked chunk.
    """
    timeout = _require_timeout(timeout_s)
    src = Path(src)
    dst = Path(dst)
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise ClockBurnError(f"could not open video {src}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as error:
        raise ClockBurnError(f"could not read video {src}") from error
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(fps)
        or not fps > 0
        or width <= 0
        or height <= 0
    ):
        raise ClockBurnError(f"video {src} has no usable rate or size")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                ENCODER_PRESET,
                "-crf",
                ENCODER_CRF,
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dst),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise ClockBurnError("could not start ffmpeg for the clock") from error
    assert proc.stdin is not None
    outcome: dict[str, object] = {"frames": 0}

    def _pump() -> None:
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                draw_clock(frame, index / fps)
                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError) as error:
                    outcome["error"] = error
                    break
                index += 1
            outcome["frames"] = index
        except Exception as error:  # never lose the pump's traceback
            outcome["error"] = error
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    pump = threading.Thread(target=_pump, name="clock-pump", daemon=True)
    deadline = time.monotonic() + timeout
    pump.start()
    try:
        pump.join(timeout)
        if pump.is_alive():
            raise ClockBurnError(
                f"clock burn of {src} timed out after {timeout:g}s"
            )
        if "error" in outcome:
            raise ClockBurnError(
                f"clock burn of {src} failed: {outcome['error']!r}"
            ) from outcome["error"]  # type: ignore[misc]
        remaining = deadline - time.monotonic()
        try:
            returncode = proc.wait(timeout=max(remaining, 0.0))
        except subprocess.TimeoutExpired as error:
            raise ClockBurnError(
                f"clock burn of {src} timed out after {timeout:g}s"
            ) from error
    finally:
        capture.release()
        if proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(OSError):
                proc.wait(timeout=10)
    frames = outcome["frames"]
    assert isinstance(frames, int)
    if returncode != 0 or frames <= 0 or not dst.is_file():
        with contextlib.suppress(OSError):
            dst.unlink(missing_ok=True)
        raise ClockBurnError(f"clock burn of {src} produced no output")
    return frames


def burn_clock_in_place(path: Path, *, timeout_s: float) -> int:
    """Overlay the clock on ``path`` atomically; return frames written.

    Encodes to a sibling temp file and renames over the original, so
    a failed burn never leaves a half-clocked chunk behind. A temp
    file stranded by a crash is an unregistered job-dir leftover, the
    same class the sweeper already clears.
    """
    timeout = _require_timeout(timeout_s)
    path = Path(path)
    # The temp keeps a video extension: ffmpeg picks the muxer from it.
    tmp = path.with_name(path.stem + ".clocktmp.mp4")
    try:
        frames = burn_clock_file(path, tmp, timeout_s=timeout)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp, path)
    except OSError as error:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise ClockBurnError(f"could not replace {path} with its clock") from error
    return frames


__all__ = (
    "CLOCK_BACK_COLOR",
    "CLOCK_FONT",
    "CLOCK_MARGIN_X",
    "CLOCK_MARGIN_Y",
    "CLOCK_PAD",
    "CLOCK_SCALE",
    "CLOCK_TEXT_COLOR",
    "CLOCK_THICKNESS",
    "ENCODER_CRF",
    "ENCODER_PRESET",
    "ClockBurnError",
    "burn_clock_file",
    "burn_clock_in_place",
    "clock_text_for_time",
    "draw_clock",
)
