"""Strict loader for config/retention.toml (stdlib only).

A missing file, a misspelled or extra key, or a bad value raises RetentionConfigError.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config" / "retention.toml"


class RetentionConfigError(ValueError):
    pass


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RetentionConfigError(f"cannot load retention config {path}: {error}") from error
    if set(cfg) != {"guide_lifetime_kinds", "failed_rescan_batch"}:
        raise RetentionConfigError(f"top level keys differ: {sorted(cfg)}")
    kinds = cfg["guide_lifetime_kinds"]
    if not isinstance(kinds, list) or not kinds or not all(isinstance(k, str) and k for k in kinds):
        raise RetentionConfigError(f"guide_lifetime_kinds: expected a non-empty list of names, got {kinds!r}")
    batch = cfg["failed_rescan_batch"]
    if type(batch) is not int or batch < 1:
        raise RetentionConfigError(f"failed_rescan_batch: expected a positive integer, got {batch!r}")
    return cfg


def guide_lifetime_kinds(path: Path = CONFIG) -> tuple[str, ...]:
    return tuple(_load(path)["guide_lifetime_kinds"])


def failed_rescan_batch(path: Path = CONFIG) -> int:
    return _load(path)["failed_rescan_batch"]
