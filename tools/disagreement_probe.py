"""Probe the background Spark Open opinion with a wrong label and its control (lane mh-disagreement).

Run on Modal by the manager (this file never reaches the network by itself in tests):

    modal run deploy/disagreement_probe_modal.py

Sends the existing Spark Open call (``app.providers.open_readback._ask``, same prompt, image
size, model and prices) for one step image twice: with a deliberately wrong part label, then
with the correct one. Writes both answers with timestamps to a JSON receipt and prints a
short summary. Labels, image and receipt path come from ``config/disagreement_probe.toml``.
The key is read only from BASETEN_API_KEY and never written anywhere.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from app.providers import open_readback as ob
from app.providers._transport import HttpTransport, UrllibTransport

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "disagreement_probe.toml"
SCHEMA = "disagreement-probe/1"
KEY_VAR = "BASETEN_API_KEY"
_PROBE_KEYS = {"image", "correct_label", "wrong_label", "wrong_case", "control_case", "receipt_path", "earlier_receipt_path", "note"}
_MODAL_KEYS = {"app_name", "secret_name", "python_version", "remote_root", "timeout_s", "image_folders", "image_files"}


class ProbeError(ValueError):
    pass


@dataclass(frozen=True)
class ProbeConfig:
    image: str
    correct_label: str
    wrong_label: str
    wrong_case: str
    control_case: str
    receipt_path: str
    earlier_receipt_path: str
    note: str
    modal: Mapping[str, object]


def _relative(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ProbeError(f"{what} must be a relative path inside the repository")
    return value


def load_config(path: Path = CONFIG) -> ProbeConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ProbeError(f"cannot read {path.name}: {error}") from error
    if set(raw) != {"probe", "modal"} or set(raw["probe"]) != _PROBE_KEYS or set(raw["modal"]) != _MODAL_KEYS:
        raise ProbeError(f"{path.name}: unexpected or missing keys")
    probe, modal = raw["probe"], raw["modal"]
    for key in _PROBE_KEYS:
        if not isinstance(probe[key], str) or not probe[key].strip():
            raise ProbeError(f"probe.{key} must be a non-empty string")
    if probe["wrong_label"].strip().lower() == probe["correct_label"].strip().lower():
        raise ProbeError("wrong_label must differ from correct_label")
    if probe["wrong_case"] == probe["control_case"]:
        raise ProbeError("wrong_case and control_case must differ")
    _relative(probe["image"], "probe.image")
    _relative(probe["receipt_path"], "probe.receipt_path")
    _relative(probe["earlier_receipt_path"], "probe.earlier_receipt_path")
    for key in _MODAL_KEYS - {"timeout_s", "image_folders", "image_files"}:
        if not isinstance(modal[key], str) or not modal[key].strip():
            raise ProbeError(f"modal.{key} must be a non-empty string")
    if not isinstance(modal["timeout_s"], int) or isinstance(modal["timeout_s"], bool) or modal["timeout_s"] <= 0:
        raise ProbeError("modal.timeout_s must be a positive integer")
    for key in ("image_folders", "image_files"):
        if not isinstance(modal[key], list) or not all(isinstance(n, str) for n in modal[key]):
            raise ProbeError(f"modal.{key} must be a list of strings")
        for name in modal[key]:
            _relative(name, f"modal.{key}")
    return ProbeConfig(**{k: probe[k] for k in _PROBE_KEYS}, modal=modal)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _agreement(answer: str | None) -> str:
    return {"yes": "agree", "no": "disagree"}.get(answer or "", "unavailable")


def run_probe(
    cfg: ProbeConfig,
    image_path: Path,
    key: str | None,
    transport: HttpTransport | None = None,
    ocfg: ob.OpenReadbackConfig | None = None,
    now: Callable[[], str] = _now,
) -> dict:
    """Two Spark Open calls on one image; returns the receipt (never contains the key)."""
    if not key:
        raise ProbeError(f"{KEY_VAR} is not set; nothing was sent")
    ocfg = ocfg or ob.load_config()
    for label in (cfg.wrong_label, cfg.correct_label):
        if len(label) > ocfg.label_max_chars:
            raise ProbeError("label longer than open_readback.label_max_chars")
    image_bytes = image_path.read_bytes()  # also fails loudly when the image is missing
    url = ob._jpeg_data_url(image_path, ocfg.image_max_px, ocfg.jpeg_quality)
    transport = transport or UrllibTransport()
    cases = []
    for case, label in ((cfg.wrong_case, cfg.wrong_label), (cfg.control_case, cfg.correct_label)):
        started_at, t0 = now(), time.monotonic()
        answer = units_in = units_out = error = None
        try:
            answer, units_in, units_out = ob._ask(ocfg, key, transport, url, label)
        except Exception as exc:  # a failed call is recorded as unavailable, not hidden
            error = type(exc).__name__
        cases.append({
            "case": case,
            "request_label": label,
            "model_answer": answer,
            "agreement": _agreement(answer),
            "started_at": started_at,
            "finished_at": now(),
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "units_in": units_in,
            "units_out": units_out,
            "cost_usd": (str(ob._cost(ocfg, units_in, units_out)) if units_in is not None and units_out is not None else None),
            "error": error,
        })
    return {
        "schema": SCHEMA,
        "model_id": ocfg.model_id,
        "image": {"name": image_path.name, "sha256": hashlib.sha256(image_bytes).hexdigest()},
        "note": cfg.note,
        "cases": cases,
    }


def write_receipt(receipt: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)  # unique per call, same dir for an atomic replace
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt, indent=2) + "\n")
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def summary(receipt: dict) -> str:
    lines = [f"model {receipt['model_id']}, image {receipt['image']['name']}"]
    for c in receipt["cases"]:
        lines.append(f"  {c['case']}: label '{c['request_label']}' -> {c['model_answer'] or 'no answer'} ({c['agreement']}, {c['latency_ms']} ms)")
    lines.append(receipt["note"])
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    receipt = run_probe(cfg, ROOT / cfg.image, os.environ.get(KEY_VAR))
    print(summary(receipt))
    print("receipt:", write_receipt(receipt, ROOT / cfg.receipt_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
