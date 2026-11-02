"""Strict loader for config/guide_text.toml: the guide's visual-change status and the
result line shown after each label edit (``fix_<code>``)."""
from __future__ import annotations

import tomllib
from pathlib import Path

# DRAFT flag reason when the battery-first order rule fails (verbatim); hazards sets it, guides words it.
DRAFT_ORDER_REASON = "DRAFT: battery disconnect not shown in your video"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "guide_text.toml"
FIX_CODES = ("renamed", "renamed_unchecked", "deleted", "confirmed", "unconfirmed", "stale",
             "blank", "long", "bad", "gone", "locked", "full", "failed")
#: Codes that report a success (a status line); the rest are refusals (an alert).
STOP_KINDS = ("battery", "glued_screen", "unknown_hazard")
VERDICT_KEYS = ("easy", "medium", "hard", "hard_battery", "hard_glued_screen", "hard_unknown_hazard",
                "hard_battery_glued_screen", "hard_battery_unknown_hazard", "hard_glued_screen_unknown_hazard",
                "hard_battery_glued_screen_unknown_hazard")
FIX_OK = ("renamed", "deleted", "confirmed", "unconfirmed")
_KEYS = ("status_unchanged", "status_unchanged_help", "status_not_checked_help", "intro_title", "intro_note", "stop_line", "stop_cta", "stop_done",
         "difficulty_prefix", "safety_stop_note", "draft_note_battery_reader",
         "stop_title", "stop_folded_one", "stop_folded_many",
         *(f"stop_{part}_{kind}" for part in ("name", "risk") for kind in STOP_KINDS),
         *(f"verdict_{key}" for key in VERDICT_KEYS), *(f"fix_{code}" for code in FIX_CODES))


class GuideTextError(ValueError):
    pass


def load_guide_text(path: Path = CONFIG) -> dict[str, str]:
    """Read the wording; a missing, misspelled or blank key raises."""
    with open(path, "rb") as fh:
        top = tomllib.load(fh)
    table = top.get("guide_text")
    if set(top) != {"guide_text"} or not isinstance(table, dict) or set(table) != set(_KEYS):
        raise GuideTextError(f"{path}: expected one [guide_text] table with keys {list(_KEYS)}")
    for key in _KEYS:
        if not isinstance(table[key], str) or not table[key].strip():
            raise GuideTextError(f"guide_text.{key}: expected non-blank text")
    return dict(table)


TEXT = load_guide_text()
