"""Modal entry for the disagreement probe (a separate app from the web app).

Run (from the repo root, by the manager; this file never runs modal itself):

    modal run deploy/disagreement_probe_modal.py

Two Spark Open calls (about $0.002) from one CPU function that holds BASETEN_API_KEY from the named
Secret. The receipt JSON comes back and is written to the path in config/disagreement_probe.toml
(var/ locally). The sample label image is private; unpack it first with `python tools/sample_media.py unpack`.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import modal

LOCAL = modal.is_local()
ROOT_ENV = "PARTFIDELITY_ROOT"
ROOT = Path(__file__).resolve().parents[1] if LOCAL else Path(os.environ[ROOT_ENV])
sys.path.insert(0, str(ROOT))

from tools.disagreement_probe import ROOT as PROBE_ROOT, load_config, run_probe, summary, write_receipt  # noqa: E402

cfg = load_config(ROOT / "config" / "disagreement_probe.toml")
mcfg = cfg.modal


def build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version=mcfg["python_version"])
    if not LOCAL:
        return image
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    image = image.pip_install(*deps).env({ROOT_ENV: mcfg["remote_root"]})
    remote = Path(mcfg["remote_root"])
    for folder in mcfg["image_folders"]:
        # the private sample bundle ships only as the one named label image below
        image = image.add_local_dir(ROOT / folder, str(remote / folder),
                                    ignore=lambda path: "sample_real" in path.parts or "__pycache__" in path.parts)
    for name in mcfg["image_files"]:
        image = image.add_local_file(ROOT / name, str(remote / name))
    return image


app = modal.App(mcfg["app_name"])


@app.function(image=build_image(), secrets=[modal.Secret.from_name(mcfg["secret_name"])], timeout=mcfg["timeout_s"])
def probe() -> dict:
    return run_probe(cfg, PROBE_ROOT / cfg.image, os.environ.get("BASETEN_API_KEY"))


@app.local_entrypoint()
def main() -> None:
    receipt = probe.remote()
    print(summary(receipt))
    print("receipt:", write_receipt(receipt, ROOT / cfg.receipt_path))
