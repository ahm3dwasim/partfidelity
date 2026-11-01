"""Runs static/ui/film.js in the deterministic node harness (tests/js/film_harness.js) against the shipped config."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app import film_config
from tools import film_frames

HARNESS = Path(__file__).resolve().parent / "js" / "film_harness.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed: the film.js harness needs it")
def test_film_js_harness(tmp_path):
    cfg = tmp_path / "film.json"
    plan = tmp_path / "plan.json"
    anchors = tmp_path / "anchors.json"
    loaded = film_config.load()
    cfg.write_text(json.dumps(film_config.public(loaded)))
    plan.write_text(json.dumps(film_config.bundles(loaded["film"])))
    served = film_frames.read_anchors(film_frames.ROOT / loaded["archives"]["anchors"], loaded["film"], [p["id"] for p in loaded["parts"]])
    anchors.write_text(json.dumps(served))
    res = subprocess.run(
        [shutil.which("node"), str(HARNESS), str(cfg), str(plan), str(anchors)],
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "film.js harness: 24/24 passed" in res.stdout, res.stdout
