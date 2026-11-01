"""Checked-part configuration: only parts proven in earlier measurement are checked.

Hosted SAM 3.1 checks ``fan`` and ``battery`` only. Thresholds live in the frozen contracts mapping; Modal-proven parts
join later by extending that configuration, never by editing this module.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.contracts import PART_CHECK_THRESHOLDS


def normalize_phrase(phrase: str) -> str:
    """Lowercase and collapse whitespace so ``"  Fan "`` matches ``"fan"``."""
    return " ".join(phrase.strip().lower().split())


def threshold_for(
    phrase: str, *, thresholds: Mapping[str, float] = PART_CHECK_THRESHOLDS
) -> float | None:
    """Return the check threshold for an exact part name, else None.

    Matching is exact on purpose: ``"battery connector"`` must not match
    ``"battery"``, and ``"fan screws"`` must not match ``"fan"``.
    """
    return thresholds.get(normalize_phrase(phrase))


def is_checked_part(
    phrase: str, *, thresholds: Mapping[str, float] = PART_CHECK_THRESHOLDS
) -> bool:
    """True only for parts that carry a configured threshold."""
    return threshold_for(phrase, thresholds=thresholds) is not None


def checked_parts(
    *, thresholds: Mapping[str, float] = PART_CHECK_THRESHOLDS
) -> tuple[str, ...]:
    """Configured part names in mapping order."""
    return tuple(thresholds)


__all__ = (
    "checked_parts",
    "is_checked_part",
    "normalize_phrase",
    "threshold_for",
)
