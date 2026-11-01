import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import catch_media, catch_page_data

ROOT = Path(__file__).resolve().parents[1]


def test_rows_and_summary_follow_the_gate_reason():
    evaluate = {"caption": "c", "eval_limits": "l", "ssim_cutoff": 0.4, "rows": [
        {"split": "calibration", "clip": "H1", "candidate": "edits/a.jpg", "recipe": "owner-edit", "label": "faithful", "iou": 1.0, "ssim": 0.9, "reason": "faithful edit"},
        {"split": "calibration", "clip": "H1", "candidate": "edits/b.jpg", "recipe": "owner-edit", "label": "altered", "iou": 0.5, "ssim": 0.1, "reason": "SAM IoU below cutoff"},
        {"split": "heldout", "clip": "V3", "candidate": "recipes/c.png", "recipe": "blur", "label": "altered", "iou": None, "ssim": None, "reason": "no unambiguous SAM mask"}]}
    media = {"cases": [{"key": "k", "candidate": "edits/b.jpg", "original": "o", "edited": "e"}]}
    out = catch_page_data.build(evaluate, media)
    assert [r["accepted"] for r in out["rows"]] == [True, False, False]
    assert [r["number"] for r in out["rows"]] == [1, 2, None]
    assert out["summary"]["faithful_edits"] == {"total": 1, "accepted": 1, "rejected": 0}
    assert out["summary"]["heldout_recipes"]["rejected"] == 1 and out["cases"][0]["row"]["name"] == "b"


def test_committed_table_and_media_config_agree():
    data = json.loads((ROOT / "config/catch_table.json").read_text())
    cfg = catch_media.load()
    for case in data["cases"]:
        assert case["original"][:-4] in [i["name"] for i in cfg["images"]]
    assert set(catch_media.manifest(cfg)) == set(catch_media.filenames(cfg))
    assert data["summary"]["altered_edit"]["rejected"] == 1


def test_pack_then_unpack_round_trips_and_rejects_a_tampered_bundle(tmp_path):
    cfg = catch_media.load()
    evidence = tmp_path / "evidence"
    for image in cfg["images"]:
        path = evidence / image["from"]
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.random.default_rng(1).integers(0, 255, (72, 128, 3), dtype=np.uint8))
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    (root / "config/catch_media.json").write_text((ROOT / "config/catch_media.json").read_text())
    catch_media.pack(evidence, root)
    catch_media.unpack(root)
    catch_media.check(root)
    (root / catch_media.TARGET / "f3-edit.jpg").write_bytes(b"x")
    with pytest.raises(ValueError):
        catch_media.check(root)


def test_catch_media_imports_without_opencv():
    """`modal deploy` imports tools.catch_media from a CLI env that has no cv2; only render() needs it."""
    import subprocess
    import sys
    code = "import sys; sys.modules['cv2'] = None; import tools.catch_media as m; assert callable(m.render)"
    done = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_redrawn_recipe_name_and_its_former_name_build_the_same_rows():
    def evaluate(recipe):
        return {"caption": "c", "eval_limits": "l", "ssim_cutoff": 0.4, "rows": [
            {"split": "calibration", "clip": "H1", "candidate": "edits/a.jpg", "recipe": recipe, "label": "faithful", "iou": 1.0, "ssim": 0.9, "reason": "faithful edit"}]}
    media = {"cases": []}
    new, old = catch_page_data.build(evaluate("redrawn-edit"), media), catch_page_data.build(evaluate("owner-edit"), media)
    assert new == old and new["rows"][0]["recipe"] == "redrawn-edit" and new["rows"][0]["number"] == 1


def test_catch_page_names_redrawn_edits_and_still_reads_the_former_recipe(monkeypatch):
    from app.routes import catch

    page = catch.catch()
    assert "The ten redrawn edits" in page and "owner" not in page.lower() and "Oct 2026" not in page
    assert page.count("<td>redrawn-edit</td>") == 10
    legacy = json.loads((ROOT / "config/catch_table.json").read_text())
    for row in legacy["rows"]:
        row["recipe"] = "owner-edit" if row["recipe"] == "redrawn-edit" else row["recipe"]
    monkeypatch.setattr(catch, "load", lambda: legacy)
    assert catch.catch() == page
