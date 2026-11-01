"""Focused tests for the teardown config loader, pose maths and spend ledger (no Blender, no Modal)."""
import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "assets/3d/teardown")]

import pose  # noqa: E402
import teardown_config as tc  # noqa: E402
import teardown_render as tr  # noqa: E402


def test_config_loads_and_parts_match():
    cfg = tc.load()
    assert sorted(cfg["explode"]["order"]) == sorted(cfg["explode"]["offset"])
    assert cfg["budget"]["cap_usd"] == 15.0


def test_gpu_list_names_hardware_types():
    # Each entry goes straight to Modal's gpu= argument: a type name such as A10 or A100-80GB, optionally with :count.
    import re
    gpus = tc.load()["modal"]["gpu"].split(",")
    assert gpus and all(re.fullmatch(r"[A-Z][A-Z0-9]*(-[0-9A-Z]+)?(:[0-9]+)?", g) for g in gpus), gpus


@pytest.mark.parametrize("mutate", [
    lambda c: c["render"].pop("seed"),
    lambda c: c["render"].update(sed=1),
    lambda c: c["pose"].pop("camera"),
    lambda c: c["hero"][0].update(state="nope"),
    lambda c: c["explode"]["offset"].pop("cooling"),
    lambda c: c.update(extra={}),
])
def test_config_fails_loudly(mutate):
    cfg = copy.deepcopy(tc.load())
    mutate(cfg)
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


@pytest.mark.parametrize("mutate", [
    lambda c: c["reassemble"].update(start=1),                                      # overlaps the rise
    lambda c: c["reassemble"].update(start=pose.span(c)[1]),                        # exactly at the end of the rise: still no hold
    lambda c: c["reassemble"].update(start=249),                                    # finishes after the film
    lambda c: c["reassemble"].update(start=c["render"]["n_frames"] - pose.reassembled(c) + c["reassemble"]["start"]),  # ends on the last frame
])
def test_reassemble_must_fit_between_the_rise_and_the_end(mutate):
    cfg = copy.deepcopy(tc.load())
    mutate(cfg)
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


def test_film_starts_open_then_the_lid_closes_before_anything_rises():
    cfg = tc.load()
    ch, n = cfg["choreo"], cfg["render"]["n_frames"]
    first = pose.frame_pose(cfg, 1)
    assert all(v == 0 for v in first["parts"].values()) and first["lid_deg"] == cfg["scene"]["lid_open_deg"]
    assert first["camera"]["el"] >= 60                                    # opens on a top view of the deck
    lid = [pose.frame_pose(cfg, f)["lid_deg"] for f in range(1, n + 1)]
    assert all(a >= b for a, b in zip(lid, lid[1:]))                      # the lid only ever closes
    shut = ch["close_start"] + ch["close_frames"]
    assert lid[shut - 1] == 0 and not any(pose.part_progress(cfg, shut).values())   # shut before any part moves
    assert 20 <= ch["close_frames"] <= 45                                 # about 1 s at 25 fps (shorter opening)
    mid = lid[ch["close_start"] + ch["close_frames"] // 2 - 1]
    assert 0.3 < mid / cfg["scene"]["lid_open_deg"] < 0.7                 # eased in and out, half-way at mid-time
    assert pose.part_progress(cfg, ch["rise_start"])["lid_back"] == 0 and shut < ch["rise_start"]   # a beat on the closed laptop
    top = pose.frame_pose(cfg, cfg["reassemble"]["start"])
    assert top["explode"] == 1 and all(top["parts"][p] == 1 for p in cfg["explode"]["rise"])   # the exploded stack, held
    last = pose.frame_pose(cfg, n)                                                              # the film ends reassembled
    assert last["explode"] == 0 and not any(last["parts"].values()) and last["lid_deg"] == 0
    assert pose.part_offsets(cfg, {"lid_back": 0.5})["lid_back"] == cfg["explode"]["offset"]["lid_back"] / 2


def test_lid_sits_on_top_of_the_stack_and_leaves_first():
    cfg = tc.load()
    rise, off = cfg["explode"]["rise"], cfg["explode"]["offset"]
    assert rise[:2] == ["lid_back", "display_panel"]
    assert max(off.values()) == off["lid_back"]


def _project_max(cfg, f):
    """Largest |x| or |y| (1 = frame edge) of every part's box corners through the film camera at frame f."""
    import math
    import layout as L
    lid = {"lid_back", "display_panel"}
    p = pose.frame_pose(cfg, f)
    off, a = pose.part_offsets(cfg, p["parts"]), -math.radians(p["lid_deg"])
    hy, hz = cfg["scene"]["pivot_yz"]
    c = p["camera"]
    cp, t = pose.camera_position(c), c["target"]
    fw = [t[i] - cp[i] for i in range(3)]
    k = math.sqrt(sum(v * v for v in fw))
    fw = [v / k for v in fw]
    k = math.hypot(fw[0], fw[1])
    r = [fw[1] / k, -fw[0] / k, 0.0]
    u = [r[1] * fw[2] - r[2] * fw[1], r[2] * fw[0] - r[0] * fw[2], r[0] * fw[1] - r[1] * fw[0]]
    scale = c["lens"] / (cfg["camera"]["sensor_mm"] / 2)
    aspect = cfg["render"]["film_res"][1] / cfg["render"]["film_res"][0]
    worst = 0.0
    for n, o in off.items():
        for x in (-L.W / 2, L.W / 2):
            for y in (-L.D / 2, L.D / 2):
                for z in ((L.LID_Z0, L.LID_Z1) if n in lid else (0.0, L.DECK_TOP)):
                    if n in lid:      # lid parts live in the pivot frame: offset along the lid normal, then the swing
                        yy, zz = y - hy, z - hz + o
                        pt = (x, hy + yy * math.cos(a) - zz * math.sin(a), hz + yy * math.sin(a) + zz * math.cos(a))
                    else:
                        pt = (x, y, z + o)
                    d = [pt[i] - cp[i] for i in range(3)]
                    depth = sum(d[i] * fw[i] for i in range(3))
                    worst = max(worst, abs(scale * sum(d[i] * r[i] for i in range(3)) / depth),
                                abs(scale * sum(d[i] * u[i] for i in range(3)) / depth / aspect))
    return worst


def test_every_part_stays_inside_the_frame():
    """Every frame keeps every part in frame (the close-up that cut the stack is gone)."""
    cfg = tc.load()
    worst = max(_project_max(cfg, f) for f in range(1, cfg["render"]["n_frames"] + 1))
    assert worst < 0.92, worst


def test_camera_is_one_steady_wide_shot_after_the_opening():
    """The opening is the top view easing to the wide shot while the lid shuts;
    from the end of the opening (frame 40, lid already shut) to the last frame every camera value is identical."""
    cfg = tc.load()
    ch = cfg["choreo"]
    keys = [k for k in cfg["pose"]["camera"] if k["frame"] <= 40]
    assert [(k["frame"], k["az"], k["el"], k["dist"], k["target"]) for k in keys] == [
        (1, 0.0, 74.0, 1750.0, [0.0, 40.0, 80.0]), (10, 2.0, 73.0, 1750.0, [0.0, 40.0, 80.0]),
        (40, 27.0, 16.0, 1950.0, [0.0, 0.0, 135.0])]
    assert ch["close_start"] + ch["close_frames"] <= 40 <= ch["rise_start"]
    cams = {json.dumps(pose.frame_pose(cfg, f)["camera"], sort_keys=True) for f in range(40, cfg["render"]["n_frames"] + 1)}
    assert len(cams) == 1
    assert max(_project_max(cfg, f) for f in range(40, cfg["render"]["n_frames"] + 1)) < 0.9


def test_parts_rise_one_by_one_top_first_and_never_cross():
    cfg = tc.load()
    rise, off = cfg["explode"]["rise"], cfg["explode"]["offset"]
    assert [off[n] for n in rise] == sorted((off[n] for n in rise), reverse=True)   # the top part leaves first
    starts = []
    for n in rise:
        starts.append(next(f for f in range(1, 400) if pose.part_progress(cfg, f)[n] > 0))
    assert starts == sorted(starts) and len(set(starts)) == len(starts)            # staggered, not all at once
    for f in range(1, cfg["render"]["n_frames"] + 1):                              # an upper part always stays above the one below it
        z = [off[n] * pose.part_progress(cfg, f)[n] for n in rise]
        assert all(a >= b for a, b in zip(z, z[1:])), f
    assert pose.span(cfg)[1] < cfg["reassemble"]["start"]                          # a hold on the full stack before the return
    assert pose.reassembled(cfg) < cfg["render"]["n_frames"] - 20                  # and a hold on the whole laptop at the end


def test_parts_return_lowest_first_never_crossing_and_the_camera_holds_still():
    """the reassembly mirrors the rise (lowest part lands first, an upper part is never below one under it);
    the camera is steady (see test_camera_is_one_steady_wide_shot_after_the_opening)."""
    cfg = tc.load()
    rise, off = cfg["explode"]["rise"], cfg["explode"]["offset"]
    ends = [next(f for f in range(cfg["reassemble"]["start"], 400) if pose.part_progress(cfg, f)[n] == 0) for n in reversed(rise)]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)
    for f in range(cfg["reassemble"]["start"], cfg["render"]["n_frames"] + 1):
        z = [off[n] * pose.part_progress(cfg, f)[n] for n in rise]
        assert all(a >= b for a, b in zip(z, z[1:])), f


def test_jobs_cover_each_state_and_angle():
    cfg = tc.load()
    names = [j["name"] for j in tr.plan_jobs(cfg, "contact")]
    assert {"top_top.png", "closing_a.png", "closed_a.png", "exploded_b.png"} <= set(names)
    assert len(tr.plan_jobs(cfg, "heroes")) == 4
    with pytest.raises(ValueError):
        tr.plan_jobs(cfg, "frame")


def test_ledger_cap_counts_open_reservations(tmp_path):
    led = tmp_path / "spend.md"
    r = tr.reserve("a", "contact", 14.5, led)
    assert tr.spent(led) == 14.5                        # an open reservation already counts
    with pytest.raises(tr.BudgetRefused):
        tr.reserve("b", "contact", 1.0, led)
    r.settle(100, 0.25)
    assert tr.spent(led) == 0.25 and "| reserved:" not in led.read_text()
    tr.reserve("c", "contact", 0.5, led).release()
    assert tr.spent(led) == 0.25 and led.read_text().count("| contact |") == 1


def test_threads_cannot_both_pass_one_balance(tmp_path, monkeypatch):
    led = tmp_path / "spend.md"
    real = tr.spent

    def slow(path=None):                                # widen the check-then-write window
        total = real(path)
        time.sleep(0.05)
        return total

    monkeypatch.setattr(tr, "spent", slow)
    wins, refusals = [], []

    def go():
        try:
            wins.append(tr.reserve("t", "heroes", 10.0, led))
        except tr.BudgetRefused:
            refusals.append(1)

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(refusals) == 5
    assert real(led) == 10.0


_CHILD = """
import sys, time
sys.path[:0] = [{tools!r}, {assets!r}]
import teardown_render as tr
from pathlib import Path
led, mode = Path(sys.argv[1]), sys.argv[2]
if mode == "race":
    try:
        tr.reserve("p", "heroes", 10.0, led)
        print("@@won@@", flush=True)
    except tr.BudgetRefused:
        print("@@refused@@", flush=True)
    sys.exit(0)
if mode.startswith("defer"):
    import os, signal, threading
    def kill():
        if mode == "defer-thread":  # sent from a live thread with the stop masked there, so it can only land on the main thread
            def go():
                signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGTERM])
                os.kill(os.getpid(), signal.SIGTERM)
            t = threading.Thread(target=go)
            t.start()
            t.join()
        else:
            os.kill(os.getpid(), signal.SIGTERM)
    with tr.spend("p", "heroes", 1.0, led):
        with tr._locked(led):
            kill()
            deadline = time.monotonic() + 10
            while not (tr._SHIELD["pending"] or signal.SIGTERM in signal.sigpending()) and time.monotonic() < deadline:
                pass
            if not (tr._SHIELD["pending"] or signal.SIGTERM in signal.sigpending()):
                print("SIGTERM was not pending at the wait deadline", file=sys.stderr)
                sys.exit(3)
            led.with_name("inside").write_text("still inside the lock")
with tr.spend("p", "heroes", 2.0, led) as r:
    r.started = mode == "started"
    print("@@ready@@", flush=True)
    time.sleep(60)
"""
_TAGS = ("@@ready@@", "@@won@@", "@@refused@@")   # sentinels: imports (e.g. Modal's banner) may print first


def _child(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(tools=str(ROOT / "tools"), assets=str(ROOT / "assets/3d/teardown")))
    return script


def _env(tmp_path, noisy):
    """noisy: a fake `modal` first on the path that prints on import, as the real one does on some machines."""
    if not noisy:
        return None
    fake = tmp_path / "noisy"
    fake.mkdir(exist_ok=True)
    (fake / "modal.py").write_text("import sys\nfrom unittest import mock\nprint('Modal banner: ready to go')\n"
                                  "print('@@ready')\nsys.modules[__name__] = mock.MagicMock()\n")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(fake), os.environ.get("PYTHONPATH")]))}


def _tags(text):
    return [line.strip() for line in text.splitlines() if line.strip() in _TAGS]


def _await_ready(p):
    for line in p.stdout:
        if line.strip() == "@@ready@@":
            return
    pytest.fail(f"child exited before ready: {p.communicate()[1]}")


@pytest.mark.parametrize("noisy", [False, True])
def test_processes_cannot_both_pass_one_balance(tmp_path, noisy):
    led, script = tmp_path / "spend.md", _child(tmp_path)
    procs = [subprocess.Popen([sys.executable, str(script), str(led), "race"], stdout=subprocess.PIPE, text=True,
                              env=_env(tmp_path, noisy)) for _ in range(4)]
    outs = sorted(t for p in procs for t in _tags(p.communicate(timeout=60)[0]))
    assert outs == ["@@refused@@", "@@refused@@", "@@refused@@", "@@won@@"]
    assert tr.spent(led) == 10.0


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("sig,started,usd", [
    (signal.SIGTERM, True, 2.0), (signal.SIGINT, True, 2.0), (signal.SIGTERM, False, 0.0), (signal.SIGINT, False, 0.0),
])
def test_interrupted_run_keeps_worst_case_or_releases(tmp_path, sig, started, usd, noisy):
    led, script = tmp_path / "spend.md", _child(tmp_path)
    p = subprocess.Popen([sys.executable, str(script), str(led), "started" if started else "idle"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(tmp_path, noisy))
    _await_ready(p)
    p.send_signal(sig)
    p.communicate(timeout=30)
    assert p.returncode != 0
    text = led.read_text()
    assert "| reserved:" not in text
    assert tr.spent(led) == usd
    assert ("| interrupted |" in text) is started


@pytest.fixture(params=[False, True], ids=["alone", "other-thread"])
def bystander(request):
    """True: a live thread that leaves stop signals unmasked, so the kernel hands it a process signal (as Modal's do)."""
    if not request.param:
        yield
        return
    done = threading.Event()
    t = threading.Thread(target=done.wait, daemon=True)
    t.start()
    yield
    done.set()
    t.join()


@pytest.mark.parametrize("mode", ["defer", "defer-thread"])
def test_stop_signal_waits_until_the_ledger_write_is_done(tmp_path, mode):
    """Run in a child so the stop never meets the test runner's own threads or handlers; the marker file proves the body
    got past the signal before the held stop was raised at the lock's exit."""
    led, script = tmp_path / "spend.md", _child(tmp_path)
    p = subprocess.run([sys.executable, str(script), str(led), mode], capture_output=True, text=True, timeout=30)
    assert p.returncode != 0, p.stdout
    assert "KeyboardInterrupt" in p.stderr
    assert (tmp_path / "inside").read_text() == "still inside the lock"
    assert "| reserved:" not in led.read_text()
    assert tr.spent(led) == 0.0                         # not started: released


def test_a_stop_caught_in_the_shield_machinery_is_held_not_raised():
    """With another thread alive the handler can fire inside the shield's own entry/exit or the contextlib calls around it;
    raising there would skip spend()'s cleanup. Those frames hold the stop; any other frame (the caller's body) raises."""
    import contextlib
    import types
    codes = [tr._SPEND_CODE, tr._shielded.__wrapped__.__code__, tr._locked.__wrapped__.__code__,
             contextlib._GeneratorContextManager.__enter__.__code__, contextlib._GeneratorContextManager.__exit__.__code__]
    for code in codes:
        tr._SHIELD["pending"] = None
        tr._stop(signal.SIGTERM, types.SimpleNamespace(f_code=code))           # depth 0, but in machinery: held
        assert tr._SHIELD["pending"] == signal.SIGTERM
    tr._SHIELD["pending"] = None
    other = types.SimpleNamespace(f_code=test_a_stop_caught_in_the_shield_machinery_is_held_not_raised.__code__)
    tr._stop(signal.SIGTERM, other)    # outside the spend body even a frame no list names holds it (a late delivery could land anywhere)
    assert tr._SHIELD["pending"] == signal.SIGTERM
    tr._SHIELD["pending"], tr._SHIELD["body"] = None, 1
    try:
        with pytest.raises(KeyboardInterrupt):
            tr._stop(signal.SIGTERM, other)
    finally:
        tr._SHIELD["body"] = 0
    assert tr._SHIELD == {"depth": 0, "pending": None, "body": 0}


def test_stop_between_ledger_write_and_handoff_still_releases(tmp_path, monkeypatch, bystander):
    """A stop landing after reserve() wrote its row and unlocked, before spend() holds the id, must not orphan the row."""
    led, made = tmp_path / "spend.md", tr.Reservation

    def late_stop(*a):
        assert "| reserved:" in led.read_text()           # the row is written and the lock is gone: the old window
        os.kill(os.getpid(), signal.SIGTERM)
        _until_held(signal.SIGTERM)
        return made(*a)

    monkeypatch.setattr(tr, "Reservation", late_stop)
    with pytest.raises(KeyboardInterrupt):
        with tr.spend("w", "heroes", 1.0, led):
            pass
    assert "| reserved:" not in led.read_text()
    assert tr.spent(led) == 0.0
    assert tr._SHIELD == {"depth": 0, "pending": None, "body": 0} and signal.getsignal(signal.SIGTERM) is not tr._stop


def test_stop_landing_before_the_reservation_is_assigned_still_drops_its_row(tmp_path, monkeypatch):
    """The stop hits after reserve() wrote its row but before spend() holds the Reservation: the id chosen up front cleans up."""
    led, real = tmp_path / "spend.md", tr.reserve

    def reserve_then_stop(*a, **kw):
        real(*a, **kw)
        assert "| reserved:" in led.read_text()
        raise KeyboardInterrupt("signal 15")

    monkeypatch.setattr(tr, "reserve", reserve_then_stop)
    with pytest.raises(KeyboardInterrupt):
        with tr.spend("w", "heroes", 1.0, led):
            pass
    assert "| reserved:" not in led.read_text()
    assert tr.spent(led) == 0.0


def _res(jobs, webp=False, rc=0, tex_rc=0, wall=100.0):
    if webp:
        files = {f"webp-{w}/{j['name'][:-4]}.webp": b"x" for j in jobs for w in (tr._RN["film_res"][0], tr._RN["phone_width"])}
    else:
        files = {j["name"]: b"png" for j in jobs}
    return {"files": files, "wall": wall, "rc": rc, "tex_rc": tex_rc, "tex_err": "", "log": "TEARDOWN_DONE", "tail": ""}


def test_single_run_settles_measured_cost(tmp_path):
    led, jobs = tmp_path / "spend.md", tr.plan_jobs(tc.load(), "check")
    out = tr.run_single("check", jobs, False, "r", tmp_path / "out", lambda p: _res(p["jobs"]), led)
    assert out["cost_usd"] == tr.run_cost(100.0, False) == tr.spent(led)
    assert (tmp_path / "out/check_top.png").is_file()


@pytest.mark.parametrize("kw,match", [({"tex_rc": 1}, "textures exit 1"), ({"rc": 3}, "blender exit 3")])
def test_single_run_failure_raises_after_keeping_logs(tmp_path, kw, match):
    led, jobs = tmp_path / "spend.md", tr.plan_jobs(tc.load(), "check")
    with pytest.raises(tr.RenderFailed, match=match):
        tr.run_single("check", jobs, False, "r", tmp_path / "out", lambda p: _res(p["jobs"], **kw), led)
    assert (tmp_path / "out/modal.log").is_file()
    assert tr.spent(led) == tr.run_cost(100.0, False)


def test_anchors_run_plans_every_film_frame_and_needs_only_its_json(tmp_path):
    """the label anchors come from the scene itself, on CPU, one pose per film frame and no render."""
    cfg, led = tc.load(), tmp_path / "spend.md"
    jobs = tr.plan_jobs(cfg, "anchors")
    assert [j["pose"]["frame"] for j in jobs] == list(range(1, cfg["render"]["n_frames"] + 1))
    assert all(j["res"] == cfg["render"]["film_res"] for j in jobs)
    seen = {}

    def remote(p):
        seen.update(p)
        return {**_res([]), "files": {"anchors.json": b"{}"}}

    tr.run_single("anchors", jobs, False, "a", tmp_path / "out", remote, led)
    assert seen["anchors"] is True and (tmp_path / "out/anchors.json").is_file()
    with pytest.raises(tr.RenderFailed, match="1 outputs missing"):
        tr.run_single("anchors", jobs, False, "a", tmp_path / "out", lambda p: _res([]), led)


def test_single_run_missing_output_fails(tmp_path):
    led, jobs = tmp_path / "spend.md", tr.plan_jobs(tc.load(), "check")
    with pytest.raises(tr.RenderFailed, match="1 outputs missing"):
        tr.run_single("check", jobs, False, "r", tmp_path / "out", lambda p: _res(p["jobs"][:1]), led)


def test_single_run_remote_error_keeps_worst_case(tmp_path):
    led, jobs = tmp_path / "spend.md", tr.plan_jobs(tc.load(), "check")

    def boom(payload):
        raise RuntimeError("container lost")

    with pytest.raises(RuntimeError, match="container lost"):
        tr.run_single("check", jobs, False, "r", tmp_path / "out", boom, led)
    assert tr.spent(led) == tr.worst_case(False) and "| interrupted |" in led.read_text()


def test_film_failed_chunk_fails_the_run(tmp_path):
    led = tmp_path / "spend.md"
    jobs = tr.plan_jobs(tc.load(), "sequence", first=1, last=3 * tr._RN["film_chunk"])

    def mapper(payloads):
        yield _res(payloads[0]["jobs"], webp=True)
        yield RuntimeError("chunk 1 died")
        yield _res(payloads[2]["jobs"], webp=True, rc=1)

    with pytest.raises(tr.RenderFailed, match="2 problem"):
        tr.run_film(jobs, True, "f", tmp_path / "film", mapper, led)
    assert {p.name for p in (tmp_path / "film").glob("modal_*.log")} == {"modal_0.log", "modal_1.log", "modal_2.log"}
    assert tr.spent(led) == round(2 * tr.run_cost(100.0, True) + tr.worst_case(True), 4)


def test_film_interrupted_keeps_returned_cost_plus_outstanding_worst(tmp_path):
    led = tmp_path / "spend.md"
    jobs = tr.plan_jobs(tc.load(), "sequence", first=1, last=3 * tr._RN["film_chunk"])

    def mapper(payloads):
        yield _res(payloads[0]["jobs"], webp=True)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tr.run_film(jobs, True, "f", tmp_path / "film", mapper, led)
    assert tr.spent(led) == round(tr.run_cost(100.0, True) + 2 * tr.worst_case(True), 4)


def test_film_all_good_settles_sum(tmp_path):
    led = tmp_path / "spend.md"
    jobs = tr.plan_jobs(tc.load(), "sequence", first=1, last=2 * tr._RN["film_chunk"])
    out = tr.run_film(jobs, True, "f", tmp_path / "film", lambda ps: (_res(p["jobs"], webp=True) for p in ps), led)
    assert out["bad_chunks"] == [] and tr.spent(led) == round(2 * tr.run_cost(100.0, True), 4)
    assert (tmp_path / "film/webp-900/0001.webp").is_file()


def test_container_stops_before_blender_when_textures_fail(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="PIL missing")

    monkeypatch.setattr(tr.subprocess, "run", fake_run)
    job = {"cfg": tc.load(), "jobs": tr.plan_jobs(tc.load(), "check")}
    res = tr._run(job, str(tmp_path / "work"))
    assert calls == [sys.executable]                    # textures only; Blender never starts
    assert res["tex_rc"] == 2 and res["files"] == {} and "PIL missing" in res["tex_err"]
    assert tr._problems(res, job["jobs"], False)[0] == "textures exit 2"


def test_external_check_is_optional_and_generic(monkeypatch):
    var = tc.load()["budget"]["check_hook_env"]
    monkeypatch.delenv(var, raising=False)
    tr.external_check()                                 # unset: the ledger alone
    monkeypatch.setenv(var, f"{sys.executable} -c 'raise SystemExit(0)'")
    tr.external_check()
    monkeypatch.setenv(var, f"{sys.executable} -c 'print(\"over pace\"); raise SystemExit(4)'")
    with pytest.raises(tr.BudgetRefused, match="exit 4"):
        tr.external_check()


def test_render_tool_carries_no_private_paths_or_profile():
    # The spend ledger is not part of the public copy; check it only where it exists.
    rels = ("tools/teardown_render.py", "config/teardown.toml", "docs/3d/RENDER_SPEND.md")
    for rel in (r for r in rels if (ROOT / r).exists()):
        text = (ROOT / rel).read_text()
        assert "Path.home()" not in text and "~/" not in text and "profile =" not in text, rel
    assert "profile" not in tc.load()["modal"]


def test_explode_layers_are_evenly_spaced():
    cfg = tc.load()
    offs = sorted(cfg["explode"]["offset"][n] for n in cfg["explode"]["rise"])
    gaps = {round(b - a, 3) for a, b in zip([0.0] + offs, offs)}
    assert len(gaps) == 1, gaps


def test_still_pose_is_a_film_frame_from_another_angle():
    cfg = tc.load()
    p = pose.still_pose(cfg, "exploded", "a")
    f = pose.frame_pose(cfg, cfg["state"]["exploded"]["frame"])
    assert p["parts"] == f["parts"] and p["exposure"] == f["exposure"]
    assert p["camera"]["az"] == next(a["az"] for a in cfg["angle"] if a["name"] == "a")


@pytest.mark.parametrize("mutate", [
    lambda c: c["look"]["open"].pop("exposure"),
    lambda c: c["light"][0].update(group="nope"),
    lambda c: c["light"][0].update(color=[2.0, 0.0, 0.0]),
    lambda c: c["light"][0].update(spread=0.0),
    lambda c: c["light"][0].update(floor="no"),
    lambda c: c["light"][0].pop("floor"),
    lambda c: c["material"].pop("alu"),
    lambda c: c["explode"]["rise"].append("glass_bezel"),
    lambda c: c["explode"]["offset"].update(feet=5.0),
    lambda c: c["state"]["closed"].update(frame=999),
    lambda c: c["choreo"].pop("rise_stagger"),
    lambda c: c["choreo"].update(close_frames=c["choreo"]["rise_start"]),
    lambda c: c["explode"]["center"].append("nope"),
    lambda c: c["render"].pop("stills_dir"),
])
def test_polish_keys_fail_loudly(mutate):
    cfg = copy.deepcopy(tc.load())
    mutate(cfg)
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


def test_key_jitter_is_deterministic_and_bounded():
    import layout
    layout.JITTER[0] = 0.0
    base = layout.keys()
    layout.JITTER[0] = 0.22
    a, b = layout.keys(), layout.keys()
    layout.JITTER[0] = 0.0
    assert a == b and len(a) == len(base)
    assert max(abs(x["x"] - y["x"]) for x, y in zip(a, base)) <= 0.22 + 1e-9
    assert len({round(x["w"] - y["w"], 4) for x, y in zip(a, base)}) > 5


def test_film_look_goes_from_open_to_exploded_and_back():
    cfg = tc.load()
    lk = cfg["look"]
    for key in pose.LOOK:
        assert pose.frame_pose(cfg, 1)[key] == pytest.approx(lk["open"][key])
        assert pose.frame_pose(cfg, cfg["reassemble"]["start"])[key] == pytest.approx(lk["exploded"][key])
        assert pose.frame_pose(cfg, cfg["render"]["n_frames"])[key] == pytest.approx(lk["open"][key])


def test_lights_are_coloured_and_laptop_is_not_orange():
    cfg = tc.load()
    rims = [l["color"] for l in cfg["light"] if l["group"] == "rim"]
    assert any(c[2] > 0.8 and c[0] < 0.3 for c in rims) and any(c[0] > 0.8 and c[1] < 0.3 for c in rims)   # a cool and a warm rim
    r, g, b = cfg["material"]["alu"]
    assert all(not l["floor"] for l in cfg["light"] if l["group"] == "rim")    # rims colour the laptop, not the sweep
    assert max(r, g, b) - min(r, g, b) < 16                                       # neutral space grey, colour only from the lights


def test_film_plan_is_chunkable_and_named_by_frame():
    cfg = tc.load()
    jobs = tr.plan_jobs(cfg, "sequence")
    assert len(jobs) == cfg["render"]["n_frames"] and jobs[0]["name"] == "0001.png" and jobs[-1]["name"] == f"{cfg['render']['n_frames']:04d}.png"


def test_wear_and_shell_are_required_keys():
    cfg = copy.deepcopy(tc.load())
    cfg["scene"].pop("wear")
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)
    cfg = copy.deepcopy(tc.load())
    cfg["look"]["exploded"].pop("shell")
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


def test_board_is_few_big_parts_and_clustered_passives():
    import layout
    cfg = tc.load()
    layout.configure(cfg["board"])
    assert len(layout.ICS) <= tc.MAX_ICS and len(layout.RAM) == 4 and len(layout.SHIELDS) >= 2   # was 14 small ICs
    soc = layout.SOC
    assert all(abs(x - soc[0]) < soc[2] / 2 + 12 for x, _ in layout.RAM)                  # memory hugs the SoC
    ps = layout.passives()
    assert 300 < len(ps) <= tc.MAX_PASSIVES     # dense like a real board, but clustered, not 700 evenly scattered
    rects = cfg["board"]["cluster"]
    inside = [p for p in ps if any(abs(p[0] - c["x"]) <= c["w"] / 2 + 1 and abs(p[1] - c["y"]) <= c["d"] / 2 + 1 for c in rects)]
    assert len(ps) - len(inside) <= cfg["board"]["scatter_passives"]                      # clustered, only a few strays


def test_board_config_fails_loudly():
    cfg = copy.deepcopy(tc.load())
    cfg["board"].pop("mask")
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)
    cfg = copy.deepcopy(tc.load())
    cfg["board"]["ic"] = cfg["board"]["ic"] * 3
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)
    cfg = copy.deepcopy(tc.load())
    cfg["board"]["cluster"][0]["keep"] = 2
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


def test_mid_lift_only_between_ends():
    cfg = tc.load()
    n = cfg["render"]["n_frames"]
    assert pose.frame_pose(cfg, 1)["rim"] == pytest.approx(cfg["look"]["open"]["rim"])
    assert pose.frame_pose(cfg, cfg["reassemble"]["start"])["rim"] == pytest.approx(cfg["look"]["exploded"]["rim"])
    assert pose.frame_pose(cfg, n)["rim"] == pytest.approx(cfg["look"]["open"]["rim"])
    mid = next(f for f in range(1, n) if pose.frame_pose(cfg, f)["explode"] >= 0.5)
    assert pose.frame_pose(cfg, mid)["rim"] > max(cfg["look"]["open"]["rim"], cfg["look"]["exploded"]["rim"])
    assert cfg["explode"]["hide_parts"] == ["hinges"] and pose.frame_pose(cfg, 1)["explode"] < cfg["explode"]["hide_from"]


def test_screws_back_out_just_before_their_layer_moves():
    cfg = tc.load()
    lead = cfg["screws"]["lead"]
    assert 0 < lead <= cfg["choreo"]["rise_stagger"]
    for n in cfg["explode"]["rise"]:
        start = next(f for f in range(1, 400) if pose.part_progress(cfg, f)[n] > 0)
        assert pose.screw_progress(cfg, start)[n] > 0                              # screws already out when the layer moves
        assert pose.screw_progress(cfg, start - lead - 1)[n] == 0
    held = pose.frame_pose(cfg, cfg["reassemble"]["start"])
    assert held["screws"]["bottom_cover"] == 1                                     # the bottom case gives up its screws too
    last = pose.frame_pose(cfg, cfg["render"]["n_frames"])
    assert not any(last["screws"].values())                                        # every screw is back in when the laptop is whole


def test_float_bob_is_sub_millimetre_and_zero_while_seated():
    cfg = tc.load()
    assert 0 < cfg["float"]["amp_mm"] < 1
    for f in range(1, cfg["render"]["n_frames"] + 1):
        p = pose.frame_pose(cfg, f)
        assert all(abs(v) < 1 for v in p["bob"].values())
        assert all(p["bob"][n] == 0 for n in p["bob"] if p["parts"][n] == 0)


def test_camera_aperture_comes_from_every_key():
    cfg = tc.load()
    assert pose.frame_pose(cfg, cfg["render"]["n_frames"])["camera"]["fstop"] >= 8  # deeper focus on the wide hold
    assert all("fstop" in pose.still_pose(cfg, h["state"], h["angle"])["camera"] for h in cfg["hero"])


@pytest.mark.parametrize("mutate", [
    lambda c: c["screws"].pop("lead"),
    lambda c: c["screws"].update(lead=99),
    lambda c: c["float"].update(amp_mm=1.5),
    lambda c: c["dust"].update(box=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    lambda c: c["pose"]["camera"][0].pop("fstop"),
    lambda c: c["angle"][0].pop("fstop"),
    lambda c: c["choreo"].update(closeup=[96, 216]),
    lambda c: c["material"].pop("cell"),
])
def test_new_sections_fail_loudly(mutate):
    cfg = copy.deepcopy(tc.load())
    mutate(cfg)
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


def test_screen_and_backlight_sleep_once_the_lid_is_shut():
    cfg = tc.load()
    ch = cfg["choreo"]
    assert pose.frame_pose(cfg, 1)["power"] == pytest.approx(1.0)
    shut = ch["close_start"] + ch["close_frames"]
    for f in range(shut, cfg["render"]["n_frames"] + 1):           # off before any layer rises, so no separated layer shows the image
        assert pose.frame_pose(cfg, f)["power"] == pytest.approx(0.0, abs=1e-9)
    assert all(0.0 <= pose.frame_pose(cfg, f)["power"] <= 1.0 for f in range(1, shut))


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("display"),
    lambda c: c["display"].pop("glass_mm"),
    lambda c: c["display"].update(backlight_mm=0),
    lambda c: c["display"].update(film_gap_mm=7.0),
    lambda c: c["display"].update(sleep_from=1.0),
    lambda c: c["display"].update(glass_opacity=[0.5, 0.1]),
    lambda c: c["display"].update(panel_steel=[1, 2]),
    lambda c: c["display"].update(backlight=-1),
    lambda c: c["display"].pop("cell_glass"),
    lambda c: c["display"].update(panel_steel_rough=1.5),
    lambda c: c["display"].update(backlight_colour=[1, 1, 1]),
])
def test_display_section_fails_loudly(mutate):
    cfg = copy.deepcopy(tc.load())
    mutate(cfg)
    with pytest.raises(tc.ConfigError):
        tc.validate(cfg)


def test_display_panel_thin_sheets_are_flat_shaded():
    # Smooth shading on an unbevelled sheet averages the rim's side normals into the top face, so the panel read as a
    # creased, tented sheet in film v6. Every unbevelled prism or box of the display module stays flat-shaded.
    import ast
    tree = ast.parse((ROOT / "assets/3d/teardown/parts.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "display_panel")
    flat = {}
    for c in ast.walk(fn):
        if isinstance(c, ast.Call) and getattr(c.func, "id", "") in ("prism", "box"):
            kw = {k.arg: k.value for k in c.keywords}
            if isinstance(kw.get("bevel"), ast.Constant) and kw["bevel"].value is False:
                flat[c.args[0].value] = isinstance(kw.get("smooth"), ast.Constant) and kw["smooth"].value is False
    assert flat == {"lcd_image": True, "back_label": True, "glass_ink": True}


def test_display_module_is_one_solid_part_that_fits_the_lid():
    """Physical layers, not floating paper-like ones:
    no backlight film floats apart any more, and glass + cell + backlight unit fit the lid pocket, so the closed and open
    laptop look as before."""
    import layout as L
    cfg = tc.load()
    d = cfg["display"]
    assert not {"film_gap_mm", "lcd_gaps", "diffuser", "light_guide", "reflector"} & set(d)
    top = L.LID_Z0 + d["glass_mm"] + 0.1 + d["cell_mm"] + d["backlight_mm"]
    assert top <= L.LID_Z1 - 0.7 - 0.05, top                       # lid_back's pocket ends 0.7 mm below its outer face
    assert d["glass_mm"] >= 0.7 and d["backlight_mm"] >= 1.0          # thick enough to read as solid at the film's scale
    src = (ROOT / "assets/3d/teardown/render.py").read_text()
    assert "film_gap_mm" not in src and "FILMS" not in src


def test_render_script_binds_every_module_it_reads():
    """render.py only runs inside Blender, so an import alias that no longer matches its uses would surface there first.
    Every name read as ``name.attr`` must be imported, assigned or defined in the script itself."""
    import ast
    import builtins
    tree = ast.parse((ROOT / "assets/3d/teardown/render.py").read_text())
    bound = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
    read = {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    assert read <= bound, sorted(read - bound)
    assert "layout_cfg" in read


def test_plan_jobs_keeps_the_former_f0_f1_keywords():
    """`f0`/`f1` were the names of `first`/`last`; the old call still plans the same frames."""
    cfg, n = tc.load(), 3 * tr._RN["film_chunk"]
    old = tr.plan_jobs(cfg, "sequence", f0=1, f1=n)
    assert [j["name"] for j in old] == [j["name"] for j in tr.plan_jobs(cfg, "sequence", first=1, last=n)]
    assert [j["name"] for j in tr.plan_jobs(cfg, "sequence", f0=2, f1=3)] == ["0002.png", "0003.png"]
    with pytest.raises(ValueError, match="not both"):
        tr.plan_jobs(cfg, "sequence", first=1, f0=1)


def test_cli_keeps_the_former_f0_f1_flags(monkeypatch):
    """`modal run ... --mode sequence --f0 A --f1 B` still reaches the planner as first/last."""
    import importlib.util
    from unittest import mock

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, *a, **k):
            return lambda f: f

        def local_entrypoint(self, *a, **k):
            return lambda f: f

    fake = mock.MagicMock()
    fake.App = App
    monkeypatch.setitem(sys.modules, "modal", fake)
    spec = importlib.util.spec_from_file_location("teardown_render_cli", ROOT / "tools" / "teardown_render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []
    monkeypatch.setattr(mod, "external_check", lambda: None)
    monkeypatch.setattr(mod, "run_film", lambda jobs, *a, **k: seen.append([j["name"] for j in jobs]))
    mod.main(mode="sequence", f0=2, f1=3)
    mod.main(mode="sequence", first=2, last=3)
    assert seen == [["0002.png", "0003.png"]] * 2


def test_budget_lane_key_still_loads():
    """[budget] `lane` is the former name of `job`; an old config still loads."""
    import tomllib
    raw = tomllib.loads((ROOT / tc.CONFIG_REL).read_text())
    raw["budget"]["lane"] = raw["budget"].pop("job")
    assert tc.validate(raw)["budget"]["job"] == tc.load()["budget"]["job"]
    raw["budget"]["job"] = "x"
    with pytest.raises(tc.ConfigError, match="unknown keys"):
        tc.validate(raw)
