"""Build and render the photoreal laptop teardown in Blender on Modal.

Blender never runs on this Mac: the scene is built by assets/3d/teardown/render.py inside the Modal
container and every render (previews included) happens there. Locally this tool only plans the jobs,
checks the spend cap, uploads nothing but code, and writes the returned PNGs.

  modal run tools/teardown_render.py --mode check            # CPU: build + tiny render
  modal run tools/teardown_render.py --mode contact --round-name r01
  modal run tools/teardown_render.py --mode heroes           # 4 stills at full samples
  modal run tools/teardown_render.py --mode frame --frame 90
  modal run tools/teardown_render.py --mode anchors          # CPU: no render, where each part sits per film frame

The Modal profile is Modal's own choice: MODAL_PROFILE if set, else your default profile.
All values (versions, GPU, resolutions, samples, poses, budget) come from config/teardown.toml.
Spend: every run reserves its worst case in docs/3d/RENDER_SPEND.md (cap in the config) under one lock before
any remote work, then settles it to the measured seconds, or keeps the worst case if the run is cut off after
remote work started. An optional extra check runs the command in the environment variable named by
[budget] check_hook_env (unset = the ledger alone); a non-zero exit refuses the run.
Any failed texture, render or chunk ends the run with RenderFailed (non-zero exit) after the logs are written.
"""
from __future__ import annotations

import contextlib
from contextlib import contextmanager
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "assets" / "3d" / "teardown"))
sys.path.insert(0, "/app")  # inside the container the scene scripts are mounted here

import pose as P  # noqa: E402  (pure Python, shared with the Blender scripts)
from teardown_config import load  # noqa: E402

CFG = load()
_BL, _MO, _BU, _RN = CFG["blender"], CFG["modal"], CFG["budget"], CFG["render"]

try:
    import modal
except ImportError:  # planning and the ledger work without it
    modal = None


class BudgetRefused(RuntimeError):
    pass


class RenderFailed(RuntimeError):
    pass


# ---- job planning (pure) -------------------------------------------------------
def _range_alias(value, legacy, name, legacy_name):
    """`f0`/`f1` are the former names of `first`/`last`; either spelling works, never both."""
    if legacy is None:
        return value
    if value is not None:
        raise ValueError(f"pass {name} or {legacy_name}, not both")
    return legacy


def _job(name, p, res, samples):
    return {"name": name, "pose": p, "res": list(res), "samples": samples}


def plan_jobs(cfg: dict, mode: str, frame: int | None = None, first: int | None = None, last: int | None = None, *,
              f0: int | None = None, f1: int | None = None) -> list:
    first, last = _range_alias(first, f0, "first", "f0"), _range_alias(last, f1, "last", "f1")
    r = cfg["render"]
    if mode == "check":
        return [_job("check_top.png", P.still_pose(cfg, "top", "a"), (320, 180), 4),
                _job("check_exploded.png", P.still_pose(cfg, "exploded", "a"), (320, 180), 4)]
    if mode == "contact":
        out = []
        for st in cfg["state"]:
            for a in cfg["angle"]:
                out.append(_job(f"{st}_{a['name']}.png", P.still_pose(cfg, st, a["name"]), r["preview_res"], r["preview_samples"]))
        return out
    if mode == "heroes":
        return [_job(h["name"], P.still_pose(cfg, h["state"], h["angle"]), r["still_res"], r["still_samples"]) for h in cfg["hero"]]
    if mode == "frame":
        if frame is None:
            raise ValueError("frame mode needs --frame")
        return [_job(f"frame_{frame:04d}.png", P.frame_pose(cfg, frame), r["preview_res"], r["preview_samples"])]
    if mode == "anchors":
        return [_job(f"{n:04d}", P.frame_pose(cfg, n), r["film_res"], 1) for n in range(1, r["n_frames"] + 1)]
    if mode == "sequence":
        a, b = first or 1, last or r["n_frames"]
        return [_job(f"{n:04d}.png", P.frame_pose(cfg, n), r["film_res"], r["film_samples"]) for n in range(a, b + 1)]
    raise ValueError(f"unknown mode {mode!r}")


# ---- spend ledger (docs/3d/RENDER_SPEND.md) -------------------------------------
_ROW = re.compile(r"^\|\s*\d{4}-\d\d-\d\d[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|\s*$")


def ledger_path() -> Path:
    return HERE.parent / _BU["ledger"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def spent(path: Path | None = None) -> float:
    """Settled spend plus open reservations (a reserved row carries its worst case)."""
    path = path or ledger_path()
    if not path.exists():
        return 0.0
    return round(sum(float(m.group(1)) for m in map(_ROW.match, path.read_text().splitlines()) if m), 4)


_STOPS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
_SHIELD = {"depth": 0, "pending": None, "body": 0}     # main thread only: Python runs signal handlers there


def _stop(signum, frame):
    """spend()'s handler: a stop becomes KeyboardInterrupt, held while the ledger is mid-change (or spend() itself runs)."""
    if _SHIELD["depth"] or not _SHIELD["body"] or (frame is not None and frame.f_code in _HOLD_CODES):
        _SHIELD["pending"] = _SHIELD["pending"] or signum
    else:
        raise KeyboardInterrupt(f"signal {signum}")


def _replay() -> None:
    if _SHIELD["depth"] == 0 and _SHIELD["pending"]:
        signum, _SHIELD["pending"] = _SHIELD["pending"], None
        raise KeyboardInterrupt(f"signal {signum}")


@contextmanager
def _shielded():
    """Stop signals wait until the block ends. The mask covers a single-threaded process; with other threads alive the
    kernel hands a process signal to one of them and Python runs the handler here anyway, so spend()'s _stop holds it."""
    main = threading.current_thread() is threading.main_thread()
    held = signal.pthread_sigmask(signal.SIG_BLOCK, _STOPS)
    _SHIELD["depth"] += main
    try:
        yield
    finally:
        _SHIELD["depth"] -= main
        signal.pthread_sigmask(signal.SIG_SETMASK, held)
        if main:
            _replay()


@contextmanager
def _locked(path: Path):
    """One exclusive lock per ledger (a sibling .lock file) for every read-check-write; stop signals wait until it ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _shielded(), open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _text(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return (f"# Modal render cost, job {_BU['job']} (cap ${_BU['cap_usd']:.2f})\n\n"
            "Estimated from measured container seconds x the config rate (+ start-up overhead); "
            "Modal billing is the source of truth. A `reserved:` row is a run in flight at its worst case.\n\n"
            "| date (UTC) | run | mode | seconds | usd |\n|---|---|---|---|---|\n")


class Reservation:
    """A worst-case row in the ledger; settle() replaces it with the real spend, release() removes it."""

    def __init__(self, path: Path, rid: str, run: str, mode: str, usd: float):
        self.path, self.rid, self.run, self.mode, self.usd = path, rid, run, mode, usd
        self.started = False          # set just before remote work: from then on an abort keeps a charge
        self.abort_usd = usd          # what an abort after start records (callers lower it as parts settle)
        self.closed = False

    def _swap(self, row: str | None) -> None:
        with _locked(self.path):
            lines = self.path.read_text().splitlines(keepends=True)
            mark = f"| reserved:{self.rid} |"
            hit = [i for i, l in enumerate(lines) if mark in l]
            if len(hit) != 1:
                raise BudgetRefused(f"ledger lost reservation {self.rid}")
            lines[hit[0]:hit[0] + 1] = [row] if row else []
            _write(self.path, "".join(lines))
            self.closed = True

    def settle(self, seconds: float | str, usd: float) -> None:
        secs = seconds if isinstance(seconds, str) else f"{seconds:.0f}"
        self._swap(f"| {_now()} | {self.run} | {self.mode} | {secs} | {usd:.4f} |\n")

    def release(self) -> None:
        self._swap(None)


def reserve(run: str, mode: str, usd: float, path: Path | None = None, rid: str | None = None) -> Reservation:
    """Check the cap and record the worst case in one locked step, so two runs cannot both pass one balance."""
    path = path or ledger_path()
    with _locked(path):
        total = spent(path)
        if total + usd > _BU["cap_usd"]:
            raise BudgetRefused(f"spent ${total:.2f} + worst case ${usd:.2f} exceeds the render cap ${_BU['cap_usd']:.2f}")
        rid = rid or uuid.uuid4().hex[:12]
        _write(path, _text(path) + f"| {_now()} | {run} | {mode} | reserved:{rid} | {usd:.4f} |\n")
    return Reservation(path, rid, run, mode, usd)


def _drop(path: Path, rid: str) -> None:
    """Remove the reserved row `rid` if it is there: a stop that lands between the ledger write and the return of the
    Reservation leaves spend() holding only the id it chose up front, so that is what the cleanup works from."""
    with _locked(path):
        mark = f"| reserved:{rid} |"
        lines = path.read_text().splitlines(keepends=True) if path.exists() else []
        if any(mark in l for l in lines):
            _write(path, "".join(l for l in lines if mark not in l))


@contextmanager
def spend(run: str, mode: str, usd: float, path: Path | None = None):
    """Reserve; on any exit not settled by the caller, keep abort_usd if remote work started, else release.

    SIGINT, SIGTERM and SIGHUP become KeyboardInterrupt, held off while the reservation is written, handed over or closed.
    """
    old, r, rid = {}, None, uuid.uuid4().hex[:12]
    if threading.current_thread() is threading.main_thread():
        for sig in _STOPS:
            old[sig] = signal.signal(sig, _stop)
    try:
        with _shielded():  # a stop arriving between the ledger write and `r = ...` waits until r holds the reservation
            r = reserve(run, mode, usd, path, rid)
        _SHIELD["body"] = 1   # only now may a stop raise: before this and after the body, every frame holds it
        try:
            yield r
        finally:
            _SHIELD["body"] = 0
    finally:
        try:
            with _shielded():
                if r is None:
                    _drop(path or ledger_path(), rid)  # reserve() may have written its row before r was assigned
                elif not r.closed:
                    r.settle("interrupted", r.abort_usd) if r.started else r.release()
        finally:
            for sig, h in old.items():
                signal.signal(sig, h)
            if old:
                _replay()          # a stop held in spend()'s own frame after the cleanup shield


# With another thread alive the handler can run at any bytecode, including the shield's own entry and exit (before its
# mask or depth is set) and the contextlib calls around them. A stop caught in any of these frames is held and replayed
# once the shield is down, so cleanup always runs.
_SPEND_CODE = spend.__wrapped__.__code__
_HOLD_CODES = {_SPEND_CODE, _shielded.__wrapped__.__code__, _locked.__wrapped__.__code__,
               contextlib._GeneratorContextManager.__enter__.__code__, contextlib._GeneratorContextManager.__exit__.__code__}


def run_cost(seconds: float, gpu: bool) -> float:
    rate = _BU["gpu_usd_per_s"] if gpu else _BU["cpu_usd_per_s"] * _MO["cpu_count"]
    return round((seconds + _BU["overhead_s"]) * rate, 4)


def worst_case(gpu: bool) -> float:
    return run_cost(_MO["gpu_timeout_s"] if gpu else _MO["cpu_timeout_s"], gpu)


def external_check() -> None:
    """Optional extra spend check: the command in the environment variable named by the config; unset = ledger only."""
    cmd = os.environ.get(_BU["check_hook_env"], "").strip()
    if not cmd:
        return
    res = subprocess.run(shlex.split(cmd), text=True, capture_output=True, timeout=30)
    if res.returncode != 0:
        raise BudgetRefused(f"spend check {_BU['check_hook_env']} refused (exit {res.returncode}): " + res.stdout.strip()[:200])


# ---- running a mode (remote callables are passed in, so the flow is testable without Modal) ----
def _expected(jobs: list, webp: bool, mode: str = "") -> set:
    if mode == "anchors":
        return {"anchors.json"}
    if not webp:
        return {j["name"] for j in jobs}
    widths = {_RN["film_res"][0], _RN["phone_width"]}
    return {f"webp-{w}/{j['name'][:-4]}.webp" for j in jobs for w in widths}


def _problems(res: dict, jobs: list, webp: bool, mode: str = "") -> list:
    out = []
    if res["tex_rc"]:
        out.append(f"textures exit {res['tex_rc']}")
    elif res["rc"]:
        out.append(f"blender exit {res['rc']}")
    missing = _expected(jobs, webp, mode) - set(res["files"])
    if missing:
        out.append(f"{len(missing)} outputs missing, e.g. {sorted(missing)[0]}")
    return out


def _save(out_dir: Path, res: dict, log_name: str) -> None:
    for fname, data in res["files"].items():
        dest = out_dir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data) if isinstance(data, bytes) else dest.write_text(data)
    (out_dir / log_name).write_text(res["log"] + "\n--- tail ---\n" + res["tail"] + "\n--- textures ---\n" + res["tex_err"])


def run_single(mode: str, jobs: list, gpu: bool, label: str, out_dir: Path, remote, ledger: Path | None = None) -> dict:
    """One container for all jobs; raises RenderFailed after saving outputs and logs if anything failed."""
    payload = {"cfg": CFG, "gpu": gpu, "jobs": jobs, "contact": mode == "contact", "anchors": mode == "anchors"}
    with spend(label, mode, worst_case(gpu), ledger) as r:
        r.started = True
        res = remote(payload)
        cost = run_cost(res["wall"], gpu)
        r.settle(res["wall"], cost)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(out_dir, res, "modal.log")
    summary = {"mode": mode, "out": str(out_dir), "files": sorted(res["files"]), "rc": res["rc"], "tex_rc": res["tex_rc"],
               "wall_s": round(res["wall"], 1), "cost_usd": cost, "spent_usd": spent(ledger)}
    print(json.dumps(summary, indent=1))
    print(res["log"])
    bad = _problems(res, jobs, False, mode)
    if bad:
        raise RenderFailed(f"{mode} failed: {'; '.join(bad)} (logs in {out_dir / 'modal.log'})")
    return summary


def run_film(jobs: list, gpu: bool, name: str, out_dir: Path, mapper, ledger: Path | None = None) -> dict:
    """Parallel chunks, WebP sets written locally. The worst case of every chunk is reserved up front; each chunk
    that returns lowers the abort charge to its measured cost. Raises RenderFailed if any chunk failed."""
    n = _RN["film_chunk"]
    chunks = [jobs[i:i + n] for i in range(0, len(jobs), n)]
    per = worst_case(gpu)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = [{"cfg": CFG, "gpu": gpu, "jobs": c, "contact": False, "webp": True} for c in chunks]
    total, secs, bad, left = 0.0, 0.0, [], len(chunks)
    with spend(name, "sequence", per * len(chunks), ledger) as r:
        r.started = True
        for i, res in enumerate(mapper(payloads)):
            left -= 1
            if isinstance(res, BaseException):
                total += per                          # unknown container time: charge the chunk's worst case
                (out_dir / f"modal_{i}.log").write_text(f"remote error: {res!r}\n")
                bad.append(f"chunk {i}: {res!r}"[:200])
            else:
                cost = run_cost(res["wall"], gpu)
                total, secs = total + cost, secs + res["wall"]
                _save(out_dir, res, f"modal_{i}.log")
                bad += [f"chunk {i}: {p}" for p in _problems(res, chunks[i], True)]
            r.abort_usd = round(total + per * left, 4)
        r.settle(secs, round(total, 4))
    summary = {"mode": "sequence", "chunks": len(chunks), "bad_chunks": bad, "cost_usd": round(total, 4), "spent_usd": spent(ledger)}
    print(json.dumps(summary))
    if bad:
        raise RenderFailed(f"sequence failed: {len(bad)} problem(s), first: {bad[0]} (logs in {out_dir})")
    return summary


# ---- Modal app -----------------------------------------------------------------
if modal is not None:
    ROOT = HERE.parent
    image = (
        modal.Image.from_registry(_BL["cuda_image"], add_python=_BL["python"])
        .apt_install(*_BL["apt_packages"])
        .pip_install(*_BL["pip_packages"])
        .run_commands(
            f"wget -q {_BL['download_url']} -O /tmp/b.tar.xz && tar -xf /tmp/b.tar.xz -C /opt"
            f" && ln -s /opt/{_BL['build']}/blender /usr/local/bin/blender && rm /tmp/b.tar.xz && blender --version"
        )
        .add_local_python_source("teardown_config")
        .add_local_file(str(ROOT / "config/teardown.toml"), "/config/teardown.toml")
        .add_local_dir(str(ROOT / "assets/3d/teardown"), "/app", ignore=["__pycache__", "*.pyc"])
    )
    app = modal.App(_MO["app"], image=image)


def _run(job: dict, work: str = "/tmp/work") -> dict:
    """Runs inside the container: textures (Pillow), then Blender builds the scene and renders every job."""
    import glob
    import shutil

    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work + "/out")
    t0 = time.time()
    json.dump({"board": job["cfg"]["board"], "screen": job["cfg"]["screen"], "material": job["cfg"]["material"]}, open(work + "/board.json", "w"))
    tex = subprocess.run([sys.executable, "/app/textures.py", work + "/tex", str(job["cfg"]["render"]["texture_ppm"]),
                                      str(job["cfg"]["scene"]["key_jitter_mm"]), work + "/board.json"],
                         capture_output=True, text=True, cwd="/app")
    if tex.returncode:                                # no Blender run on missing textures
        return {"files": {}, "wall": time.time() - t0, "rc": None, "tex_rc": tex.returncode, "tex_err": tex.stderr[-800:],
                "log": "TEXTURES_FAILED", "tail": tex.stdout[-1500:]}
    job["tex_dir"] = work + "/tex"
    json.dump(job, open(work + "/job.json", "w"))
    p = subprocess.run(["blender", "-b", "--factory-startup", "-P", "/app/render.py", "--", work + "/job.json", work + "/out"],
                       capture_output=True, text=True)
    log = p.stdout + p.stderr
    files = {os.path.basename(f): open(f, "rb").read() for f in sorted(glob.glob(work + "/out/*"))}
    if job.get("webp"):
        files = _to_webp(files, job["cfg"]["render"])
    if job.get("contact"):
        files["contact.png"] = _contact([(n, b) for n, b in files.items() if n.endswith(".png")])
    keep = "\n".join(l for l in log.splitlines() if re.search(r"DEVICES|DEVTYPE|PART_ERROR|MAT_WARN|TEARDOWN_DONE|BOOL_CHECK|Error|Traceback|WARN|^Time:", l))
    return {"files": files, "wall": time.time() - t0, "rc": p.returncode, "tex_rc": tex.returncode, "tex_err": tex.stderr[-800:],
            "log": keep[-4000:], "tail": log[-1500:]}


def _to_webp(files, r):
    """Film frames leave the container as WebP at full and phone width (PNGs are 10x larger and are not kept)."""
    import io

    from PIL import Image

    out = {}
    for n, b in files.items():
        if not n.endswith(".png"):
            out[n] = b
            continue
        im = Image.open(io.BytesIO(b)).convert("RGB")
        for w in (im.width, r["phone_width"]):
            buf = io.BytesIO()
            (im if w == im.width else im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)).save(buf, "WEBP", quality=r["webp_quality"], method=5)
            out[f"webp-{w}/{n[:-4]}.webp"] = buf.getvalue()
    return out


def _contact(pngs):
    import io

    from PIL import Image, ImageDraw

    ims = [(n, Image.open(io.BytesIO(b)).convert("RGB")) for n, b in pngs]
    if not ims:
        return b""
    w, h = ims[0][1].size
    cols = 3
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, (n, im) in enumerate(ims):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        ImageDraw.Draw(sheet).text(((i % cols) * w + 8, (i // cols) * h + 6), n, fill=(255, 255, 255))
    out = io.BytesIO()
    sheet.save(out, "PNG")
    return out.getvalue()


if modal is not None:
    @app.function(gpu=_MO["gpu"].split(",") if "," in _MO["gpu"] else _MO["gpu"], timeout=_MO["gpu_timeout_s"])
    def run_gpu(job: dict) -> dict:
        return _run(job)

    @app.function(cpu=_MO["cpu_count"], timeout=_MO["cpu_timeout_s"])
    def run_cpu(job: dict) -> dict:
        return _run(job)

    @app.local_entrypoint()
    def main(mode: str = "check", round_name: str = "", frame: int = 0, first: int = 0, last: int = 0, name: str = "",
             f0: int = 0, f1: int = 0):
        gpu = mode not in ("check", "anchors")
        external_check()
        # --f0/--f1 are the former flags for --first/--last.
        jobs = plan_jobs(CFG, mode, frame or None, first or None, last or None, f0=f0 or None, f1=f1 or None)
        if mode == "sequence":
            run_film(jobs, gpu, name or "film", Path(_RN["local_out"]) / _RN["film_dir"],
                     lambda p: run_gpu.map(p, order_outputs=True, return_exceptions=True))
            return
        out_dir = Path(_RN["local_out"]) / ("rounds/" + round_name if mode in ("check", "contact", "frame") and round_name else
                                            _RN["stills_dir"] if mode == "heroes" else "anchors" if mode == "anchors" else "frames")
        run_single(mode, jobs, gpu, name or round_name or mode, out_dir, (run_gpu if gpu else run_cpu).remote)
