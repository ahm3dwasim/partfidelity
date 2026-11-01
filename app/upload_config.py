"""Strict loader for config/upload.toml (stdlib only): upload and confirmation copy.

A missing, misspelled or mistyped key, or a ``{placeholder}`` outside
PLACEHOLDERS, raises UploadConfigError. :func:`render` fills the placeholders
from the values the enforcing code supplies, so no limit or hour is typed twice.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Formatter
import tomllib

CONFIG = Path(__file__).resolve().parents[1] / "config" / "upload.toml"
PLACEHOLDERS = frozenset({"duration", "size", "recipient", "meta_hours", "local_hours"})


class UploadConfigError(ValueError):
    pass


def _tables(*names):
    return {n: str for n in names}


_SPEC = {
    "page": _tables(
        "title", "kicker", "headline", "lede", "tips_label", "drop_title", "drop_before",
        "drop_link", "consent", "sample_before", "sample_link", "submit", "footer_note",
    ),
    "next": _tables(
        "heading", "step_1_title", "step_1_text", "step_2_title", "step_2_text",
        "step_3_title", "step_3_text",
    ),
    "retention": _tables(
        "heading", "processing", "meta", "published", "delete", "delete_link", "delete_after",
    ),
    "time": _tables("heading", "text"),
    "preview": _tables(
        "title", "kicker", "headline", "note", "recipient_lead", "recipient_text",
        "consent_agree", "consent_rights", "confirm", "cancel", "footer_note",
    ),
}
_LISTS = {
    "tips": ("mark", "lead", "text"),
    "spec": ("pre", "strong", "post"),
}
# Strings that may be empty (a pill with no prefix, no suffix, no bold part).
_MAY_BE_EMPTY = {"pre", "strong", "post", "delete_after"}


def _check_string(value, where, may_be_empty):
    if not isinstance(value, str) or (not value.strip() and not may_be_empty):
        raise UploadConfigError(f"{where}: bad value {value!r}")
    try:
        names = {n for _, n, _, _ in Formatter().parse(value) if n is not None}
    except ValueError as error:
        raise UploadConfigError(f"{where}: {error}") from error
    unknown = names - PLACEHOLDERS
    if unknown:
        raise UploadConfigError(f"{where}: unknown placeholder {sorted(unknown)}")


def _check_keys(table, keys, where):
    if not isinstance(table, dict) or set(table) != set(keys):
        got = sorted(table) if isinstance(table, dict) else table
        raise UploadConfigError(f"{where}: keys differ (expected {sorted(keys)}, got {got})")
    for key in keys:
        _check_string(table[key], f"{where}.{key}", key in _MAY_BE_EMPTY)


def validate(cfg: dict) -> dict:
    if set(cfg) != set(_SPEC) | set(_LISTS):
        raise UploadConfigError(f"top-level keys must be {sorted(set(_SPEC) | set(_LISTS))}, got {sorted(cfg)}")
    for name, keys in _SPEC.items():
        _check_keys(cfg[name], keys, name)
    for name, keys in _LISTS.items():
        rows = cfg[name]
        if not isinstance(rows, list) or not rows:
            raise UploadConfigError(f"{name}: expected a non-empty list of tables")
        for i, row in enumerate(rows):
            _check_keys(row, keys, f"{name}[{i}]")
    return cfg


@lru_cache(maxsize=1)
def load(path: Path = CONFIG) -> dict:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise UploadConfigError(f"{path}: {error}") from error
    return validate(raw)


def _fill(node, values):
    if isinstance(node, str):
        return node.format_map(values)
    if isinstance(node, list):
        return [_fill(item, values) for item in node]
    return {key: _fill(item, values) for key, item in node.items()}


def render(values: dict) -> dict:
    """The whole copy with every placeholder filled; ``values`` must hold all of them."""
    missing = PLACEHOLDERS - set(values)
    if missing:
        raise UploadConfigError(f"render needs values for {sorted(missing)}")
    return _fill(load(), values)
