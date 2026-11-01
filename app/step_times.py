"""Step-time consensus and presentation helpers.

Pure functions shared by extraction (median/spread/flag) and the steps
route (approximate-time display). The ±5s nudge itself runs client-side
in static/time_controls.js; the clamp rule (never below zero) is stated
there, next to the code that applies it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from app.contracts import TIME_SPREAD_FLAG_S


def median_time(samples: Sequence[float]) -> float:
    """Median of per-run start/end samples (statistics.median; 2 samples average)."""
    if not samples:
        raise ValueError("median_time needs at least one sample")
    return float(statistics.median(samples))


def time_spread(starts: Sequence[float], ends: Sequence[float]) -> float:
    """Run spread for one step: the wider of the start range and end range."""
    if not starts or not ends:
        raise ValueError("time_spread needs start and end samples")
    return max(max(starts) - min(starts), max(ends) - min(ends))


def needs_time_check(spread_s: float) -> bool:
    """True when the run spread strictly exceeds the 5-second flag line."""
    return spread_s > TIME_SPREAD_FLAG_S


def format_approx(seconds: float) -> str:
    """One honest approximate time, e.g. 83.5 -> '~1:23'."""
    total = max(0, int(round(seconds)))
    return f"~{total // 60}:{total % 60:02d}"


def format_approx_range(start_s: float, end_s: float) -> str:
    """Approximate step window, e.g. '~1:20–1:35'."""
    total_start = max(0, int(round(start_s)))
    total_end = max(0, int(round(end_s)))
    return f"~{total_start // 60}:{total_start % 60:02d}–{total_end // 60}:{total_end % 60:02d}"
