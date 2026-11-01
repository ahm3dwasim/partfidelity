"""Strict loaders for the Glimmer Modal proof (deploy/glimmer_modal.py).

The config and the download keys of the lock fail loudly on a missing,
misspelled or wrongly typed key. No secrets live in either file.
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from tools.glimmer_probe import DEFAULT_PIN, ProofError, load_pin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "glimmer_modal.json"
_EXACT_PIN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.!+_-]*\Z")


@dataclass(frozen=True)
class ModalConfig:
    app_name: str
    function_name: str
    cpu: int
    memory_mb: int
    timeout_s: int
    volume_name: str
    volume_mount: str
    remote_root: str
    python_version: str
    apt_packages: tuple[str, ...]
    hf_base_url: str
    vm_id_prefix: str
    image_px: int
    cmake_args: str
    runtime_deps: tuple[str, ...]
    grade_max_tokens: int
    download_timeout_s: int
    proof_timeout_s: int


@dataclass(frozen=True)
class DownloadSpec:
    repo: str
    revision: str
    files: tuple[tuple[str, int, str], ...]  # (name, bytes, sha256)
    runtime_package: str
    runtime_version: str
    runtime_sdist_sha256: str


def _strict(raw: object, schema: dict[str, type], label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ProofError(f"{label} must be a JSON object")
    missing = [k for k in schema if k not in raw]
    if missing:
        raise ProofError(f"{label} is missing key(s): {', '.join(missing)}")
    extra = [k for k in raw if k not in schema and not k.startswith("_")]
    if extra:
        raise ProofError(f"{label} has unknown key(s): {', '.join(sorted(extra))}")
    for key, kind in schema.items():
        value = raw[key]
        if type(value) is not kind or (kind is str and not value.strip()):
            raise ProofError(f"{label} key {key} must be a non-empty {kind.__name__}")
        if kind is int and value <= 0:
            raise ProofError(f"{label} key {key} must be positive")
    return raw


def _read(path: Path | str, label: str) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProofError(f"cannot read {label} ({type(exc).__name__})") from exc


def load_config(path: Path | str = DEFAULT_CONFIG) -> ModalConfig:
    schema: dict[str, type] = {f: str for f in ModalConfig.__dataclass_fields__}
    schema.update({"cpu": int, "memory_mb": int, "timeout_s": int, "image_px": int, "apt_packages": list, "runtime_deps": list, "grade_max_tokens": int, "download_timeout_s": int, "proof_timeout_s": int})
    raw = _strict(_read(path, "the Modal config"), schema, "Modal config")
    packages = raw["apt_packages"]
    if not packages or not all(isinstance(p, str) and p.strip() for p in packages):
        raise ProofError("Modal config key apt_packages must be a non-empty list of strings")
    deps = raw["runtime_deps"]
    if not deps or not all(isinstance(d, str) and _EXACT_PIN.match(d) for d in deps):
        raise ProofError("Modal config key runtime_deps must be a non-empty list of exact name==version pins")
    # two artifacts download serially, and the proof runs twice (probe, then the live-reproducing verdict)
    if raw["timeout_s"] <= 2 * raw["download_timeout_s"] + 2 * raw["proof_timeout_s"]:  # type: ignore[operator]
        raise ProofError("Modal config key timeout_s must exceed 2 * download_timeout_s + 2 * proof_timeout_s")
    return ModalConfig(**{**raw, "apt_packages": tuple(packages), "runtime_deps": tuple(deps)})  # type: ignore[arg-type]


def load_download_spec(lock: Path | str = DEFAULT_PIN) -> DownloadSpec:
    """The pinned files to fetch, cross-checked against the probe's own pin."""
    pin = load_pin(lock)  # every probe fact must already be a real, non-placeholder value
    schema: dict[str, type] = {
        "weights_repo": str, "weights_file": str, "weights_bytes": int,
        "mmproj_file": str, "mmproj_bytes": int,
        "runtime_package": str, "runtime_sdist_sha256": str,
    }
    raw = _read(lock, "the pin")
    for key, kind in schema.items():  # the pin also holds probe keys, so check ours only
        if not isinstance(raw, dict) or type(raw.get(key)) is not kind:
            raise ProofError(f"pin key {key} is missing or not a {kind.__name__}")
    sdist = raw["runtime_sdist_sha256"]
    if len(sdist) != 64 or any(c not in "0123456789abcdef" for c in sdist):
        raise ProofError("pin key runtime_sdist_sha256 must be a 64-hex sha256")
    return DownloadSpec(
        repo=raw["weights_repo"],
        revision=pin.revision,
        files=(
            (raw["weights_file"], raw["weights_bytes"], pin.weights_sha256),
            (raw["mmproj_file"], raw["mmproj_bytes"], pin.mmproj_sha256),
        ),
        runtime_package=raw["runtime_package"],
        runtime_version=pin.runtime_version,
        runtime_sdist_sha256=sdist,
    )


def synthetic_png(px: int) -> bytes:
    """A generated grey-gradient image with one flat block: no footage, no faces, no customer data."""
    if px < 16:
        raise ProofError("synthetic image must be at least 16 px")
    rows = bytearray()
    for y in range(px):
        rows.append(0)  # PNG filter: none
        for x in range(px):
            inside = px // 4 <= x < 3 * px // 4 and px // 4 <= y < 3 * px // 4
            rows += bytes((200, 60, 40)) if inside else bytes((x * 255 // px,) * 3)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", px, px, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b"")
    )
