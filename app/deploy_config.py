"""Deploy configuration for the Modal host, read from ``deploy/modal_deploy.toml``.

Stdlib only, so the Modal entry module can import it on a machine that has
Modal but not the app's runtime dependencies. A missing, misspelled or
mistyped key fails loudly.
"""

from __future__ import annotations

import tomllib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class HostConfig:
    """Where state lives and how often it is saved and swept."""

    data_root: Path  # Volume: job files, written temp-then-rename
    snapshot_dir: Path  # Volume: db.snapshot / db.snapshot.prev
    local_db: Path  # container-local disk: the live SQLite file
    snapshot_interval_s: float
    snapshot_settle_s: float
    sweep_interval_s: float
    snapshot_max_age_s: float


@dataclass(frozen=True)
class ModalConfig:
    app_name: str
    volume_name: str
    secret_name: str
    python_version: str
    timeout_s: int
    scaledown_window_s: int
    max_inputs: int
    remote_root: Path
    volume_mount: Path
    image_folders: tuple[str, ...]  # repo folders copied into the image
    image_files: tuple[str, ...]  # single repo files copied into the image


@dataclass(frozen=True)
class DeployConfig:
    modal: ModalConfig
    host: HostConfig


def _section(raw: Mapping[str, Any], name: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise RuntimeError(f"deploy config is missing the [{name}] section")
    missing = [key for key in keys if key not in section]
    if missing:
        raise RuntimeError(f"deploy config [{name}] is missing: {', '.join(missing)}")
    unknown = sorted(set(section) - set(keys))
    if unknown:
        raise RuntimeError(f"deploy config [{name}] has unknown keys: {', '.join(unknown)}")
    return section


def _positive(section: Mapping[str, Any], key: str) -> float:
    value = section[key]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise RuntimeError(f"deploy config {key} must be a positive number")
    return float(value)


def _positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section[key]
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"deploy config {key} must be a positive integer")
    return value


def load_deploy_config(path: Path | str) -> DeployConfig:
    """Read the deploy TOML; a missing or misspelled key fails loudly."""
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    modal_keys = (
        "app_name", "volume_name", "secret_name", "python_version",
        "timeout_s", "scaledown_window_s", "max_inputs", "remote_root", "volume_mount",
        "image_folders", "image_files",
    )
    host_keys = (
        "data_root", "snapshot_dir", "local_db",
        "snapshot_interval_s", "snapshot_settle_s", "sweep_interval_s", "snapshot_max_age_s",
    )
    modal = _section(raw, "modal", modal_keys)
    host = _section(raw, "host", host_keys)
    unknown = set(raw) - {"modal", "host"}
    if unknown:
        raise RuntimeError(f"deploy config has unknown sections: {', '.join(sorted(unknown))}")
    for key in ("app_name", "volume_name", "secret_name", "python_version", "remote_root", "volume_mount"):
        if not isinstance(modal[key], str) or not modal[key].strip():
            raise RuntimeError(f"deploy config {key} must be a non-empty string")
    for key in ("data_root", "snapshot_dir", "local_db"):
        if not isinstance(host[key], str) or not host[key].strip():
            raise RuntimeError(f"deploy config {key} must be a non-empty string")
    for key in ("image_folders", "image_files"):
        value = modal[key]
        if (not isinstance(value, list) or (key == "image_folders" and not value)
                or not all(isinstance(i, str) and i.strip() and not Path(i).is_absolute()
                           and ".." not in Path(i).parts for i in value)):
            raise RuntimeError(f"deploy config {key} must be a list of relative paths")
    mount = Path(modal["volume_mount"])
    remote = Path(modal["remote_root"])
    if not mount.is_absolute() or not remote.is_absolute() or not Path(host["local_db"]).is_absolute():
        raise RuntimeError("deploy paths must be absolute")
    if Path(host["local_db"]).resolve().is_relative_to(mount.resolve()):
        raise RuntimeError("local_db must be outside the Volume mount")
    return DeployConfig(
        modal=ModalConfig(
            app_name=modal["app_name"],
            volume_name=modal["volume_name"],
            secret_name=modal["secret_name"],
            python_version=modal["python_version"],
            timeout_s=_positive_int(modal, "timeout_s"),
            scaledown_window_s=_positive_int(modal, "scaledown_window_s"),
            max_inputs=_positive_int(modal, "max_inputs"),
            remote_root=remote,
            volume_mount=mount,
            image_folders=tuple(modal["image_folders"]),
            image_files=tuple(modal["image_files"]),
        ),
        host=HostConfig(
            data_root=Path(host["data_root"]),
            snapshot_dir=Path(host["snapshot_dir"]),
            local_db=Path(host["local_db"]),
            snapshot_interval_s=_positive(host, "snapshot_interval_s"),
            snapshot_settle_s=_positive(host, "snapshot_settle_s"),
            sweep_interval_s=_positive(host, "sweep_interval_s"),
            snapshot_max_age_s=_positive(host, "snapshot_max_age_s"),
        ),
    )
