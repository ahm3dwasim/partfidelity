"""SIGTERM/SIGINT during a preview run closes what is open, removes partial output and exits non-zero (capture.mjs registry)."""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

HARNESS = """
import { hold, track } from './demo/preview/capture.mjs';
import { writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
const child = spawn('sleep', ['60']);
hold(async () => { child.kill('SIGKILL'); });
writeFileSync(track(process.argv[1]), 'partial');
writeFileSync(process.argv[2], String(child.pid));
setInterval(() => {}, 1000);
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_kills_child_removes_partial_output_and_exits_nonzero(tmp_path, sig):
    partial, pidfile = tmp_path / "preview.mp4", tmp_path / "pid"
    proc = subprocess.Popen(["node", "--input-type=module", "-e", HARNESS, str(partial), str(pidfile)], cwd=ROOT)
    try:
        deadline = time.time() + 20
        while not pidfile.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert pidfile.exists(), "harness never started"
        child = int(pidfile.read_text())
        assert partial.exists() and _alive(child)
        proc.send_signal(sig)
        assert proc.wait(timeout=15) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
    deadline = time.time() + 5
    while _alive(child) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(child), "child process left running"
    assert not partial.exists(), "partial output left behind"


MEMO = """
import { hold } from './demo/preview/capture.mjs';
let closes = 0, finished = false;
const close = hold(async () => { closes++; await new Promise((ok) => setTimeout(ok, 200)); finished = true; });
const first = close(), second = close();
await second;   // a concurrent caller waits for the in-flight close, it does not return early
console.log(JSON.stringify({ same: first === second, closes, finishedWhenSecondResolved: finished }));
"""

RUN = """
import { run } from './demo/preview/capture.mjs';
// The child writes its own pid, then execs sleep (same pid): no pgrep needed, which the Modal image lacks.
await run('sh', ['-c', 'echo $$ > "$0"; exec sleep 60', process.argv[1]]).catch(() => {});
"""


def test_hold_memoizes_the_in_flight_close():
    out = subprocess.run(["node", "--input-type=module", "-e", MEMO], cwd=ROOT, capture_output=True, text=True, timeout=20, check=True).stdout
    assert json.loads(out) == {"same": True, "closes": 1, "finishedWhenSecondResolved": True}


def test_run_child_is_killed_on_sigterm(tmp_path):
    """run() is what the contact-sheet ffmpeg and unzip go through: an async child registered with hold()."""
    pidfile = tmp_path / "pid"
    proc = subprocess.Popen(["node", "--input-type=module", "-e", RUN, str(pidfile)], cwd=ROOT)
    try:
        deadline = time.time() + 20
        while not (pidfile.exists() and pidfile.read_text().strip()) and time.time() < deadline:
            time.sleep(0.05)
        assert pidfile.exists() and pidfile.read_text().strip(), "child never started"
        kids = [pidfile.read_text().strip()]
        assert _alive(int(kids[0])), "no child running"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 143
    finally:
        if proc.poll() is None:
            proc.kill()
    deadline = time.time() + 5
    while any(_alive(int(k)) for k in kids) and time.time() < deadline:
        time.sleep(0.05)
    assert not any(_alive(int(k)) for k in kids), "child left running"


SRC_TRACKED = """
import { commit, hold, track, run } from './demo/preview/capture.mjs';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
// The moved-in sample (src) is tracked from before extraction, so a signal after the move removes it.
hold(async () => {});   // installs the signal handler
const src = track(join(process.argv[1], 'src'));
mkdirSync(src, { recursive: true });
writeFileSync(join(src, 'sample.mp4'), 'x');
writeFileSync(join(process.argv[1], 'ready'), 'up');
setInterval(() => {}, 1000);
"""


@pytest.mark.parametrize("capture", ["preview", "full"])
def test_both_captures_track_src_before_the_move(capture):
    text = (ROOT / "demo" / capture / "capture.mjs").read_text()
    assert text.index("track(src)") < text.index("renameSync(join(part")


def test_tracked_src_is_removed_on_sigterm(tmp_path):
    proc = subprocess.Popen(["node", "--input-type=module", "-e", SRC_TRACKED, str(tmp_path)], cwd=ROOT)
    try:
        deadline = time.time() + 20
        while not (tmp_path / "ready").exists() and time.time() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "src" / "sample.mp4").exists()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 143
    finally:
        if proc.poll() is None:
            proc.kill()
    assert not (tmp_path / "src").exists()
