"""Render the PF mark's wrench in Blender on Modal.

Blender never runs on this Mac: the plan outline and the scene are built inside the Modal container.

  modal run tools/wrench_render.py --out /tmp/logo-wrench/render --run r01

Every value (versions, size, samples, geometry, light, cap) comes from config/wrench.toml. Spend: each run reserves its
worst case in the ledger (docs/3d/WRENCH_SPEND.md) under one lock before any remote work and settles it to the measured
seconds; a run that would pass the cap is refused.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/app")  # inside the container the scene scripts are mounted here
from wrench_config import load  # noqa: E402

CFG = load(ROOT / "config" / "wrench.toml") if (ROOT / "config" / "wrench.toml").exists() else load(Path("/config/wrench.toml"))
_BL, _MO, _BU = CFG["blender"], CFG["modal"], CFG["budget"]

try:
    import modal
except ImportError:  # the ledger works without it
    modal = None


class BudgetRefused(RuntimeError):
    pass


_ROW = re.compile(r"^\|\s*\d{4}-\d\d-\d\d[^|]*\|[^|]*\|\s*(?:[0-9.]+|interrupted)\s*\|\s*([0-9.]+)\s*\|\s*$")


def ledger_path() -> Path:
    return ROOT / _BU["ledger"]


def spent(path: Path | None = None) -> float:
    path = path or ledger_path()
    if not path.exists():
        return 0.0
    return round(sum(float(m.group(1)) for m in map(_ROW.match, path.read_text().splitlines()) if m), 4)


def run_cost(seconds: float) -> float:
    return round((seconds + _BU["overhead_s"]) * _BU["cpu_usd_per_s"] * _MO["cpu_count"], 4)


def worst_case() -> float:
    return run_cost(_MO["timeout_s"])


def _append(path: Path, run: str, seconds: str, usd: float) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    head = "" if path.exists() else (f"# Modal render cost, job {_BU['job']} (cap ${_BU['cap_usd']:.2f})\n\n"
                                     "Estimated from measured container seconds x the config rate (+ start-up overhead); "
                                     "Modal billing is the source of truth.\n\n| date (UTC) | run | seconds | usd |\n|---|---|---|---|\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{head}| {now} | {run} | {seconds} | {usd:.4f} |\n")


def reserve_and_run(run: str, remote, path: Path | None = None):
    """Check the cap against spend + this run's worst case under one lock, run, then record the measured cost.
    A run cut off after remote work started is booked at its worst case."""
    if not run or any(c in run for c in "|\r\n"):
        raise ValueError(f"run label {run!r} must be non-empty and hold no '|' or line break (ledger delimiters)")
    path = path or ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        total = spent(path)
        if total + worst_case() > _BU["cap_usd"]:
            raise BudgetRefused(f"spent ${total:.2f} + worst case ${worst_case():.2f} exceeds the render cap ${_BU['cap_usd']:.2f}")
        _append(path, run, "interrupted", worst_case())     # the reservation; replaced below once the run settles
        try:
            res = remote()
        except BaseException:
            raise
        lines = path.read_text().splitlines(keepends=True)
        reservation = lines.pop()
        fields = reservation.split("|")
        fields[3] = f" {res['wall']:.0f} "
        fields[4] = f" {run_cost(res['wall']):.4f} "
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".settle-", delete=False) as staged:
            try:
                staged.write("".join(lines) + "|".join(fields))
                staged.flush()
                os.fsync(staged.fileno())
                os.replace(staged.name, path)
            except BaseException:
                Path(staged.name).unlink(missing_ok=True)   # the reservation in the ledger stays
                raise
    return res


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version=_BL["python"])
        .apt_install(*_BL["apt_packages"])
        .pip_install(*_BL["pip_packages"])
        .run_commands(
            f"wget -q {_BL['download_url']} -O /tmp/b.tar.xz && tar -xf /tmp/b.tar.xz -C /opt"
            f" && ln -s /opt/{_BL['build']}/blender /usr/local/bin/blender && rm /tmp/b.tar.xz && blender --version"
        )
        .add_local_python_source("wrench_config")
        .add_local_file(str(ROOT / "config/wrench.toml"), "/config/wrench.toml")
        .add_local_dir(str(ROOT / "assets/3d/wrench"), "/app", ignore=["__pycache__", "*.pyc"])
    )
    app = modal.App(_MO["app"], image=image)

    @app.function(cpu=_MO["cpu_count"], timeout=int(_MO["timeout_s"]))
    def render(res: int = 0, samples: int = 0) -> dict:
        import glob
        import shutil

        work = "/tmp/work"
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work + "/out")
        t0 = time.time()
        from shape import plan  # noqa: E402

        cfg = json.loads(json.dumps(CFG))
        cfg["render"].update({k: v for k, v in (("res", res), ("samples", samples)) if v})   # a cheap look: --res/--samples
        json.dump({"cfg": cfg, "plan": plan(CFG["wrench"])}, open(work + "/job.json", "w"))
        p = subprocess.run(["blender", "-b", "--factory-startup", "-P", "/app/scene.py", "--", work + "/job.json", work + "/out"],
                           capture_output=True, text=True)
        log = p.stdout + p.stderr
        files = {os.path.basename(f): open(f, "rb").read() for f in sorted(glob.glob(work + "/out/*"))}
        return {"files": files, "wall": time.time() - t0, "rc": p.returncode, "log": log[-6000:]}

    @app.local_entrypoint()
    def main(out: str = "/tmp/logo-wrench/render", run: str = "r", res: int = 0, samples: int = 0):
        res = reserve_and_run(run, lambda: render.remote(res, samples))
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        for name, data in res["files"].items():
            (o / name).write_bytes(data)
        (o / f"{run}.log").write_text(res["log"])
        print(json.dumps({"rc": res["rc"], "files": sorted(res["files"]), "wall_s": round(res["wall"], 1), "spent_usd": spent()}))
        if res["rc"] or not res["files"]:
            print(res["log"])
            raise SystemExit("render failed")
