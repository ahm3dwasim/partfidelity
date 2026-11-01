"""Makefile: `make verify` honors a PY override through browser setup."""

import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_verify_passes_py_override_to_browser_setup():
    proc = subprocess.run(
        ["make", "-n", "PY=/some/python", "verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'PY="/some/python" ./tools/ensure_test_browser.sh' in proc.stdout


def test_browser_setup_uses_py_from_environment(tmp_path):
    fake = tmp_path / "fake-python"
    log = tmp_path / "calls.log"
    fake.write_text(
        "#!/bin/sh\n" f'echo "$@" >> "{log}"\n' "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    proc = subprocess.run(
        ["bash", "tools/ensure_test_browser.sh"],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "PY": str(fake)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert "-c import playwright" in calls
    assert "-m playwright install chromium" in calls
