"""Glimmer offline proof on a Modal Linux CPU box (a separate app from the web app).

Run (from the repo root, by the maintainer; this file never runs modal itself):

    modal run deploy/glimmer_modal.py

One CPU function, no GPU: the claim is CPU. It downloads the two pinned files
from public Hugging Face (no token, no Modal Secret) into a Volume, checks
their sha256 against tools/glimmer_requirements.lock, grades one generated
synthetic image with tools/glimmer_probe.py and prints the JSON row and the
verdict. Every changeable value is in config/glimmer_modal.json and the lock.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import modal

LOCAL = modal.is_local()
ROOT_ENV = "PARTFIDELITY_ROOT"
ROOT = Path(__file__).resolve().parents[1] if LOCAL else Path(os.environ[ROOT_ENV])
sys.path.insert(0, str(ROOT))

from tools.glimmer_modal_config import load_config, load_download_spec, synthetic_png

cfg = load_config(ROOT / "config" / "glimmer_modal.json")
spec = load_download_spec(ROOT / "tools" / "glimmer_requirements.lock")
LOCK = str(Path(cfg.remote_root) / "tools" / "glimmer_requirements.lock")


def build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version=cfg.python_version).apt_install(*cfg.apt_packages)
    sdist = f"/tmp/sdist/{spec.runtime_package.replace('-', '_')}-{spec.runtime_version}.tar.gz"
    image = image.run_commands(
        f"pip download {spec.runtime_package}=={spec.runtime_version} --no-deps --no-binary :all: -d /tmp/sdist",
        f"echo '{spec.runtime_sdist_sha256}  {sdist}' | sha256sum -c -",
        f"pip install {' '.join(cfg.runtime_deps)}",  # the sdist's own runtime deps, exact pins
        f"CMAKE_ARGS='{cfg.cmake_args}' pip install --no-deps {sdist}",
        # a broken runtime fails the image build, not the proof
        'python -c "import llama_cpp, llama_cpp.mtmd_cpp; from llama_cpp.llama_chat_format import MTMDChatHandler; '
        'print(llama_cpp.__version__)"',
    ).env({ROOT_ENV: cfg.remote_root})
    if not LOCAL:
        return image
    remote = Path(cfg.remote_root)
    for name in ("app/__init__.py", "app/contracts.py", "tools/glimmer_probe.py",
                 "tools/glimmer_modal_config.py", "tools/glimmer_requirements.lock",
                 "config/glimmer_modal.json"):
        image = image.add_local_file(ROOT / name, str(remote / name))
    return image


app = modal.App(cfg.app_name)
volume = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pinned(dest: Path) -> dict[str, Path]:
    """Download each pinned file once; a cached file is reused only if its digest matches."""
    dest.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, size, want in spec.files:
        target = dest / name
        if not (target.is_file() and target.stat().st_size == size and _sha256(target) == want):
            part = target.with_suffix(target.suffix + ".part")
            url = f"{cfg.hf_base_url}/{spec.repo}/resolve/{spec.revision}/{name}"
            print(f"downloading {name} ({size} bytes)", flush=True)
            with urllib.request.urlopen(url, timeout=cfg.download_timeout_s) as resp, part.open("wb") as out:  # no token sent
                while chunk := resp.read(1 << 22):
                    out.write(chunk)
            if _sha256(part) != want:
                part.unlink()
                raise SystemExit(f"downloaded {name} does not match the pinned sha256")
            part.replace(target)
            volume.commit()
        paths[name] = target
    return paths


@app.function(
    image=build_image(),
    volumes={cfg.volume_mount: volume},
    cpu=float(cfg.cpu),
    memory=cfg.memory_mb,
    timeout=cfg.timeout_s,
    name=cfg.function_name,
)
def glimmer_proof() -> dict[str, str]:
    paths = fetch_pinned(Path(cfg.volume_mount) / spec.revision)
    weights, mmproj = (paths[name] for name, _, _ in spec.files)
    with tempfile.TemporaryDirectory(prefix="glimmer-proof-") as tmp:  # per call: a warm container can run twice
        work = Path(tmp)
        image = work / "synthetic.png"
        image.write_bytes(synthetic_png(cfg.image_px))
        vm_id = f"{cfg.vm_id_prefix}-{datetime.date.today().isoformat()}"
        probe = [sys.executable, str(Path(cfg.remote_root) / "tools" / "glimmer_probe.py")]
        common = ["--pin", LOCK, "--weights", str(weights), "--mmproj", str(mmproj), "--vm-id", vm_id,
                  "--timeout-s", str(cfg.proof_timeout_s)]
        env = {"PATH": os.environ["PATH"], "HOME": "/tmp",
               "GLIMMER_MAX_TOKENS": str(cfg.grade_max_tokens)}  # no secrets reach the probe
        row = work / "row.json"
        first = subprocess.run([*probe, "probe", "--image", str(image), "--out", str(row), *common],
                               check=False, capture_output=True, text=True, env=env)
        out = {"vm_id": vm_id, "probe_exit": str(first.returncode), "row": first.stdout, "probe_stderr": first.stderr[-2000:]}
        if first.returncode == 0:
            second = subprocess.run([*probe, "verdict", "--image", str(image), "--row", str(row), *common],
                                    check=False, capture_output=True, text=True, env=env)
            out |= {"verdict_exit": str(second.returncode), "verdict": second.stdout.strip(), "verdict_stderr": second.stderr[-2000:]}
    return out


@app.local_entrypoint()
def main() -> None:
    for key, value in glimmer_proof.remote().items():
        print(f"--- {key}\n{value}")
