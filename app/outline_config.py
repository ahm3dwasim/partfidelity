"""Strict loader for config/outline.toml (stdlib only): rendered part outlines.

A missing, misspelled or mistyped key raises OutlineConfigError.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config" / "outline.toml"


class OutlineConfigError(ValueError):
    pass


def _color(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in value)
    )


def _pos_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _non_neg_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_SPEC = {"color": _color, "width_px": _pos_int, "min_mask_px": _non_neg_int}


def validate(cfg: dict) -> dict:
    if set(cfg) != {"outline"}:
        raise OutlineConfigError(f"top-level keys must be outline, got {sorted(cfg)}")
    table = cfg["outline"]
    if not isinstance(table, dict):
        raise OutlineConfigError("outline: expected a table")
    if set(table) != set(_SPEC):
        raise OutlineConfigError(
            f"outline: keys differ (missing {sorted(set(_SPEC) - set(table))}, "
            f"unknown {sorted(set(table) - set(_SPEC))})"
        )
    for key, ok in _SPEC.items():
        if not ok(table[key]):
            raise OutlineConfigError(f"outline.{key}: bad value {table[key]!r}")
    return cfg


def load(path: Path = CONFIG) -> dict:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise OutlineConfigError(f"{path}: {error}") from error
    return validate(raw)


@lru_cache(maxsize=1)
def cached() -> dict:
    return load()
