from dataclasses import replace
import csv
import json
import sys
import shutil

import tools.catch_table as catch_table

import cv2
import numpy as np
import pytest

from app.corruptions import corrupt
from app.image_match import _ssim, compare
from app.sam import SamMask, SamStillResult
from app.settings import ImageMatchSettings, load_settings
from tools.catch_table import main, split


class FakeSam:
    def __init__(self, mask):
        self.mask = mask

    def segment_still(self, image_path, phrase, *, include_confidence=True):
        if self.mask is None:
            return SamStillResult(())
        ok, data = cv2.imencode(".png", self.mask)
        assert ok
        return SamStillResult((SamMask((0, 0, 1, 1), 0.9, data.tobytes()),))


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(37)
    source = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[35:60, 35:60] = 255
    source_path, mask_path, candidate_path = (tmp_path / n for n in ("source.png", "mask.png", "candidate.png"))
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(mask_path), mask)
    return source, mask, source_path, mask_path, candidate_path


def test_faithful_altered_noop_and_missing_mask(images):
    source, mask, source_path, mask_path, candidate_path = images
    cfg = replace(ImageMatchSettings(), ssim_cutoff=0.95)
    faithful = source.copy()
    faithful[4:15, 4:15] = 255 - faithful[4:15, 4:15]
    assert cv2.imwrite(str(candidate_path), faithful)
    result = compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask), cfg)
    assert result.verdict == "ACCEPTED", result
    assert compare(source_path, mask_path, candidate_path, "fan", FakeSam(None), cfg).verdict == "REJECTED"
    assert cv2.imwrite(str(candidate_path), corrupt(faithful, mask, "recolour", cfg))
    result = compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask), cfg)
    assert result.iou_pass and not result.ssim_pass and result.verdict == "REJECTED"
    assert cv2.imwrite(str(candidate_path), source)
    assert not compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask), cfg).edit_pass
    assert cv2.imwrite(str(candidate_path), faithful)
    assert not compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask), replace(cfg, ssim_cutoff=None)).ssim_pass


def test_recipes_only_touch_mask(images):
    source, mask, *_ = images
    for recipe in ("blur", "shift", "recolour"):
        output = corrupt(source, mask, recipe, ImageMatchSettings())
        assert np.array_equal(output[mask == 0], source[mask == 0])
        assert np.any(output[mask != 0] != source[mask != 0])


def test_ssim_does_not_score_background_in_windows(images):
    source, mask, *_ = images
    candidate = source.copy()
    candidate[mask == 0] = 255 - candidate[mask == 0]
    assert _ssim(source, candidate, mask != 0, ImageMatchSettings()) == pytest.approx(1.0)


def test_registration_moves_image_and_refound_mask_together(images):
    source, mask, source_path, mask_path, candidate_path = images
    matrix = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted = cv2.warpAffine(source, matrix, (96, 96))
    shifted_mask = cv2.warpAffine(mask, matrix, (96, 96), flags=cv2.INTER_NEAREST)
    assert cv2.imwrite(str(candidate_path), shifted)
    result = compare(source_path, mask_path, candidate_path, "fan", FakeSam(shifted_mask),
                     replace(ImageMatchSettings(), ssim_cutoff=0.8))
    assert result.verdict == "ACCEPTED", result
    assert result.iou == pytest.approx(1.0)


def test_split_and_settings():
    calibration, heldout = split([{"clip": "b"}, {"clip": "a"}, {"clip": "b"}])
    assert {row["clip"] for row in calibration} == {"a"}
    assert {row["clip"] for row in heldout} == {"b"}
    assert load_settings({"PARTFIDELITY_MATCH_IOU": "0.9"}).image_match.iou_cutoff == 0.9
    with pytest.raises(ValueError):
        split([{"clip": "a"}])


def test_table_freezes_dataset_and_reports_heldout(images, tmp_path, monkeypatch, capsys):
    source, mask, source_path, mask_path, _ = images
    rows = []
    for clip in ("a", "b"):
        for label in ("altered", "faithful"):
            candidate = source.copy()
            candidate[4:15, 4:15] = 255 - candidate[4:15, 4:15]
            if label == "altered":
                candidate = corrupt(candidate, mask, "recolour", ImageMatchSettings())
            name = f"{clip}-{label}.png"
            assert cv2.imwrite(str(tmp_path / name), candidate)
            rows.append(dict(clip=clip, source=source_path.name, mask=mask_path.name,
                             candidate=name, refound=mask_path.name, phrase="fan", label=label, recipe=label))
    with (tmp_path / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    frozen = tmp_path / "frozen.json"
    monkeypatch.setattr(sys, "argv", ["catch_table", "calibrate", str(frozen), str(tmp_path)])
    main()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["catch_table", "evaluate", str(frozen), str(tmp_path)])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["caption"] == "Controlled demonstration — 1 altered / 1 faithful held out; not accuracy or precision."
    heldout = [row for row in report["rows"] if row["split"] == "heldout"]
    assert {row["label"]: row["verdict"] for row in heldout} == {"altered": "REJECTED", "faithful": "ACCEPTED"}
    with (tmp_path / "cases.csv").open("a") as stream:
        stream.write("\n")
    # A label change, unlike formatting, changes the frozen dataset fingerprint.
    rows[-1]["label"] = "altered"
    with (tmp_path / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="differs"):
        main()


def test_settings_defaults_match():
    assert load_settings({}).image_match == ImageMatchSettings()


@pytest.mark.parametrize("field", ["source", "mask", "candidate"])
@pytest.mark.parametrize("bad", [None, b"not an image"])
def test_bad_input_rejected(images, field, bad):
    source, mask, source_path, mask_path, candidate_path = images
    assert cv2.imwrite(str(candidate_path), source)
    paths = {"source": source_path, "mask": mask_path, "candidate": candidate_path}
    if bad is None:
        paths[field] = candidate_path.parent / "missing.png"
    else:
        paths[field].write_bytes(bad)
    result = compare(paths["source"], paths["mask"], paths["candidate"], "fan", FakeSam(mask), ImageMatchSettings())
    assert result.verdict == "REJECTED" and "cannot decode" in result.reason


def test_invalid_source_mask_and_ecc_failure(images, monkeypatch):
    source, mask, source_path, mask_path, candidate_path = images
    assert cv2.imwrite(str(candidate_path), source)
    assert cv2.imwrite(str(mask_path), np.zeros_like(mask))
    assert "source mask" in compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask), ImageMatchSettings()).reason
    assert cv2.imwrite(str(mask_path), mask)
    def fail(*args, **kwargs):
        raise cv2.error("ECC failed")
    monkeypatch.setattr(cv2, "findTransformECC", fail)
    assert compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask), ImageMatchSettings()).reason == "registration or SAM failed"


@pytest.mark.parametrize("data", [b"broken", b""])
def test_malformed_refound(images, data):
    source, mask, source_path, mask_path, candidate_path = images
    assert cv2.imwrite(str(candidate_path), source)
    class RawSam:
        def segment_still(self, *args, **kwargs):
            return SamStillResult((SamMask((0, 0, 1, 1), None, data),))
    result = compare(source_path, mask_path, candidate_path, "fan", RawSam(), ImageMatchSettings())
    assert result.verdict == "REJECTED"


def test_wrong_size_refound(images):
    source, mask, source_path, mask_path, candidate_path = images
    assert cv2.imwrite(str(candidate_path), source)
    result = compare(source_path, mask_path, candidate_path, "fan", FakeSam(mask[:40, :40]), ImageMatchSettings())
    assert result.verdict == "REJECTED" and result.reason == "invalid SAM mask"


def make_dataset(images, folder):
    source, mask, source_path, mask_path, _ = images
    rows = []
    for clip in ("a", "b"):
        for label in ("altered", "faithful"):
            candidate = source.copy()
            candidate[4:15, 4:15] = 255 - candidate[4:15, 4:15]
            if label == "altered":
                candidate = corrupt(candidate, mask, "recolour", ImageMatchSettings())
            name = f"{clip}-{label}.png"
            assert cv2.imwrite(str(folder / name), candidate)
            rows.append(dict(clip=clip, source=source_path.name, mask=mask_path.name, candidate=name,
                             refound=mask_path.name, phrase="fan", label=label))
    with (folder / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_frozen_tampering_and_relocation(images, tmp_path, monkeypatch, capsys):
    make_dataset(images, tmp_path)
    frozen = tmp_path / "frozen.json"
    monkeypatch.setattr(sys, "argv", ["catch_table", "calibrate", str(frozen), str(tmp_path)])
    main()
    frozen_data = json.loads(frozen.read_text())
    original_cutoff = frozen_data["settings"]["ssim_cutoff"]
    frozen_data["settings"]["ssim_cutoff"] = 0.123
    frozen.write_text(json.dumps(frozen_data))
    monkeypatch.setattr(sys, "argv", ["catch_table", "evaluate", str(frozen), str(tmp_path)])
    with pytest.raises(ValueError, match="settings differ"):
        main()
    frozen_data["settings"]["ssim_cutoff"] = original_cutoff
    frozen.write_text(json.dumps(frozen_data))
    moved = tmp_path / "moved"
    moved.mkdir()
    for path in tmp_path.glob("*.png"):
        shutil.copy(path, moved / path.name)
    shutil.copy(tmp_path / "cases.csv", moved / "cases.csv")
    monkeypatch.setattr(sys, "argv", ["catch_table", "evaluate", str(frozen), str(moved)])
    capsys.readouterr()
    main()
    assert json.loads(capsys.readouterr().out)["rows"]


def test_dataset_changes_during_calibration(images, tmp_path, monkeypatch):
    make_dataset(images, tmp_path)
    frozen = tmp_path / "frozen.json"
    original = catch_table.measure
    def mutate(case, cfg):
        result = original(case, cfg)
        Path(case["_paths"]["candidate"]).write_bytes(b"changed")
        return result
    from pathlib import Path
    monkeypatch.setattr(catch_table, "measure", mutate)
    monkeypatch.setattr(sys, "argv", ["catch_table", "calibrate", str(frozen), str(tmp_path)])
    with pytest.raises(ValueError, match="changed during calibration"):
        main()
    assert not frozen.exists()


@pytest.mark.parametrize("name", ["source", "mask", "candidate", "refound"])
@pytest.mark.parametrize("bad", ["../outside.png", "/tmp/outside.png"])
def test_paths_must_remain_in_folder(images, tmp_path, name, bad):
    rows = make_dataset(images, tmp_path)
    rows[0][name] = bad
    with (tmp_path / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="stay under"):
        catch_table.read_cases([tmp_path], "cases.csv")
