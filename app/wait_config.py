"""Strict loader for config/wait.toml (stdlib only): the waiting screen.

A missing, misspelled or mistyped key raises WaitConfigError.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib

from app.contracts import RECEIPT_STAGES

CONFIG = Path(__file__).resolve().parents[1] / "config" / "wait.toml"
#: stage.txt values the pipeline writes (app/pipeline.py _write_stage).
MARKERS = ("", "extract", "segment", "readback")


class WaitConfigError(ValueError):
    pass


def _str(v):
    return isinstance(v, str) and bool(v.strip())


def _pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


_PAGE = {
    "title": _str, "done_title": _str, "stopped_title": _str, "lede": _str,
    "log_lines": _pos_int,
}
_STAGE = {
    "id": _str, "label": _str, "detail": _str, "log_done": _str,
    "receipts": lambda v: isinstance(v, list) and all(r in RECEIPT_STAGES for r in v),
    "marker": lambda v: v in MARKERS,
}


def _check(table, spec, where):
    if not isinstance(table, dict):
        raise WaitConfigError(f"{where}: expected a table")
    if set(table) != set(spec):
        raise WaitConfigError(
            f"{where}: keys differ (missing {sorted(set(spec) - set(table))}, unknown {sorted(set(table) - set(spec))})"
        )
    for key, ok in spec.items():
        if not ok(table[key]):
            raise WaitConfigError(f"{where}.{key}: bad value {table[key]!r}")


def validate(cfg: dict) -> dict:
    if set(cfg) != {"page", "stages"}:
        raise WaitConfigError(f"top-level keys must be page and stages, got {sorted(cfg)}")
    _check(cfg["page"], _PAGE, "page")
    stages = cfg["stages"]
    if not isinstance(stages, list) or len(stages) < 2:
        raise WaitConfigError("stages: expected at least two tables")
    for i, row in enumerate(stages):
        _check(row, _STAGE, f"stages[{i}]")
    ids = [s["id"] for s in stages]
    if len(set(ids)) != len(ids):
        raise WaitConfigError(f"stages: duplicate id in {ids}")
    receipts = [r for s in stages for r in s["receipts"]]
    if len(set(receipts)) != len(receipts):
        raise WaitConfigError(f"stages: a receipt is owned twice in {receipts}")
    markers = [s["marker"] for s in stages if s["marker"]]
    if len(set(markers)) != len(markers):
        raise WaitConfigError(f"stages: a marker is used twice in {markers}")
    return cfg


def load(path: Path = CONFIG) -> dict:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WaitConfigError(f"{path}: {error}") from error
    return validate(raw)


@lru_cache(maxsize=1)
def config() -> dict:
    return load()
