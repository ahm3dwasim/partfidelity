"""Replayable stage cache.

Caches each raw Spark run and the deterministic consensus by an
input/prompt/schema digest, so the sample replays without model calls.
A cached replay is labelled permanently (SAMPLE_REPLAY_LABEL) and never
simulates live waiting: the cache-hit path performs no waits at all.

Storage is one JSON file per (stage, digest) under a configured root
(atomic write via temp file + rename, safe for two worker processes).
`cache_get` / `cache_put` match the frozen contract exactly; the root is
module configuration because the frozen signatures carry no root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from app.contracts import JSONValue

STAGE_SPARK_RUN = "spark_run"
STAGE_SPARK_CONSENSUS = "spark_consensus"
STAGE_CHUNK_BUNDLE = "chunk_bundle"

SAMPLE_REPLAY_LABEL = "Replay of cached sample, not a live run"

_DEFAULT_ROOT = Path("var/stage_cache")
_CACHE_ROOT: Path = _DEFAULT_ROOT

_STAGE_RE = re.compile(r"[a-z][a-z0-9_]*")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def configure_cache_root(path: Path | None) -> None:
    """Point the cache at `path` (`None` restores the default). Tests use tmp."""
    global _CACHE_ROOT
    _CACHE_ROOT = _DEFAULT_ROOT if path is None else Path(path)


def cache_root() -> Path:
    """Current cache root (tests assert they redirected it to tmp)."""
    return _CACHE_ROOT


def digest_for(*parts: str) -> str:
    """sha256 hex over length-prefixed parts (no concatenation ambiguity)."""
    hasher = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _checked_path(stage: str, input_digest: str) -> Path:
    if not _STAGE_RE.fullmatch(stage):
        raise ValueError(f"invalid cache stage: {stage!r}")
    if not _DIGEST_RE.fullmatch(input_digest):
        raise ValueError(f"invalid cache digest: {input_digest!r}")
    return _CACHE_ROOT / stage / f"{input_digest}.json"


def cache_get(stage: str, input_digest: str) -> JSONValue | None:
    """Return the cached JSON value, or None on a cache miss."""
    path = _checked_path(stage, input_digest)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt stage cache entry: {path}") from exc
    return value


def cache_put(stage: str, input_digest: str, value: JSONValue) -> None:
    """Store a JSON-serializable value atomically (raises TypeError if not)."""
    path = _checked_path(stage, input_digest)
    try:
        text = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"stage cache value is not JSON-serializable: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp_name).replace(path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def cache_delete(stage: str, input_digest: str) -> None:
    """Remove one entry best-effort; a missing file is not an error."""
    path = _checked_path(stage, input_digest)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
