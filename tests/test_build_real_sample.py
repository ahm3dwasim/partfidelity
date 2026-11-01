"""build_real_sample refuses, touching nothing, when a rebuild would lose real masks or meet the intro card."""
import json
from pathlib import Path

import pytest

from tools import build_real_sample as brs


def _bundle(out: Path, mask: str, steps=({"number": 1, "kind": "repair"},)) -> Path:
    (out / "labels").mkdir(parents=True)
    (out / "labels" / "keep.png").write_bytes(b"x")
    (out / "guide.json").write_text(json.dumps({"steps": list(steps), "labels": [{"detection": {"mask_path": mask}}]}))
    return out


def test_refuses_to_overwrite_real_masks(tmp_path):
    out = _bundle(tmp_path / "out", "labels/step-01-part-1-mask.png")
    with pytest.raises(SystemExit, match="real mask"):
        brs._refuse_if_lossy("", out)
    assert (out / "labels" / "keep.png").exists()


def test_refuses_the_intro_card_on_the_page_and_in_the_old_guide(tmp_path):
    with pytest.raises(SystemExit, match="intro"):
        brs._refuse_if_lossy('<article class="card" id="step-0">', tmp_path / "none")
    out = _bundle(tmp_path / "out", "labels/no-mask.png", steps=({"number": 0, "kind": "intro"},))
    with pytest.raises(SystemExit, match="intro"):
        brs._refuse_if_lossy("", out)


def test_placeholder_only_output_may_be_rebuilt(tmp_path):
    brs._refuse_if_lossy("", _bundle(tmp_path / "out", "labels/no-mask.png"))


def test_refuses_unreferenced_real_mask_file(tmp_path):
    out = _bundle(tmp_path / "out", "labels/no-mask.png")
    (out / "labels" / "preserved-real.mask.png").write_bytes(b"m")
    with pytest.raises(SystemExit, match="real mask"):
        brs._refuse_if_lossy("", out)
    assert (out / "labels" / "preserved-real.mask.png").exists()


def test_refuses_nested_real_mask_file(tmp_path):
    out = _bundle(tmp_path / "out", "labels/no-mask.png")
    nested = out / "labels" / "nested" / "real.mask.png"
    nested.parent.mkdir()
    nested.write_bytes(b"m")
    with pytest.raises(SystemExit, match="real mask"):
        brs._refuse_if_lossy("", out)
    assert nested.exists()
