"""The wrench render's config loader and spend cap (no Modal call: the remote is a stub)."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import wrench_config  # noqa: E402
import wrench_render  # noqa: E402


def test_committed_config_loads():
    cfg = wrench_config.load()
    assert cfg["budget"]["cap_usd"] == 1.0 and cfg["mark"]["variant"] in cfg["render"]["variants"]


@pytest.mark.parametrize("edit", [
    lambda c: c["wrench"].pop("fillet"),
    lambda c: c["wrench"].update(fillet="9"),
    lambda c: c["scene"].update(extra=1),
    lambda c: c["mark"]["letters"].pop("f"),
    lambda c: c.pop("budget"),
])
def test_loader_fails_loudly(edit):
    cfg = wrench_config.load()
    edit(cfg)
    with pytest.raises(wrench_config.WrenchConfigError):
        wrench_config.validate(cfg)


def test_ledger_settles_to_measured_seconds_and_refuses_past_the_cap(tmp_path):
    ledger = tmp_path / "spend.md"
    res = wrench_render.reserve_and_run("t1", lambda: {"wall": 60.0}, ledger)
    assert res["wall"] == 60.0
    assert wrench_render.spent(ledger) == wrench_render.run_cost(60.0) < 0.1
    ledger.write_text(ledger.read_text() + "| 2026-10-30 10:00 | big | 9 | 0.9900 |\n")
    ran = []
    with pytest.raises(wrench_render.BudgetRefused):
        wrench_render.reserve_and_run("t2", lambda: ran.append(1) or {"wall": 1.0}, ledger)
    assert not ran, "a refused run must not start remote work"


def test_interrupted_reservations_limit_later_runs(tmp_path):
    ledger = tmp_path / "spend.md"
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            wrench_render.reserve_and_run("interrupted", lambda: (_ for _ in ()).throw(KeyboardInterrupt()), ledger)
    assert wrench_render.spent(ledger) == round(2 * wrench_render.worst_case(), 4)
    called = []
    with pytest.raises(wrench_render.BudgetRefused):
        wrench_render.reserve_and_run("later", lambda: called.append(True), ledger)
    assert not called


def test_settlement_failure_keeps_history_and_reservation(tmp_path, monkeypatch):
    ledger = tmp_path / "spend.md"
    wrench_render.reserve_and_run("previous", lambda: {"wall": 10}, ledger)
    before = ledger.read_text()
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(wrench_render.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        wrench_render.reserve_and_run("next", lambda: {"wall": 60}, ledger)
    assert ledger.read_text().startswith(before)
    assert wrench_render.spent(ledger) == round(wrench_render.run_cost(10) + wrench_render.worst_case(), 4)
    assert not list(tmp_path.glob("spend.md.settle-*")), "the staged file is unlinked when the replace fails"


@pytest.mark.parametrize("label", ["r|01", "r\n01", "r\r01", ""])
def test_run_labels_with_ledger_delimiters_are_rejected_before_any_work(tmp_path, label):
    ledger, ran = tmp_path / "spend.md", []
    with pytest.raises(ValueError, match="run label"):
        wrench_render.reserve_and_run(label, lambda: ran.append(1) or {"wall": 60.0}, ledger)
    assert not ran and not ledger.exists()


@pytest.mark.parametrize("section,key,value", [
    ("budget", "cap_usd", float("nan")),
    ("budget", "cpu_usd_per_s", -1),
    ("modal", "cpu_count", 1.5),
    ("render", "samples", 0),
    ("render", "res", float("inf")),
])
def test_rejects_unsafe_numbers(section, key, value):
    cfg = wrench_config.load()
    cfg[section][key] = value
    with pytest.raises(wrench_config.WrenchConfigError):
        wrench_config.validate(cfg)


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, path + (i,))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path


def _with(cfg, path, value):
    import copy
    out = copy.deepcopy(cfg)
    node = out
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return out


# leaves where a negative number is meaningful (a position, an offset, an exposure)
_SIGNED = {("scene", "exposure"), ("scene", "rim", "pos"), ("scene", "fill", "pos"), ("mark", "letters", "p"), ("mark", "letters", "f")}
# leaves where zero is not a valid value
_ZERO_BAD = {("budget", "cap_usd"), ("budget", "cpu_usd_per_s"), ("modal", "cpu_count"), ("modal", "timeout_s"),
             ("render", "res"), ("render", "samples"), ("mark", "mark_px"), ("mark", "hi_px"), ("wrench", "length"),
             ("wrench", "thickness"), ("wrench", "ring_r"), ("wrench", "jaw_r"), ("wrench", "circle_res"),
             ("scene", "focal_mm"), ("scene", "distance"), ("scene", "unit_mm"), ("mark", "tone", "gamma"),
             ("wrench", "ridge", "width"), ("wrench", "ridge_bevel", "segments"), ("mark", "export", "webp_quality")}


def _key(path):
    return tuple(p for p in path if isinstance(p, str))


def test_every_numeric_field_rejects_nan_inf_wrong_type_negative_and_zero():
    cfg = wrench_config.load()
    leaves = list(_leaves(cfg))
    assert len(leaves) > 60
    for path in leaves:
        for bad in (float("nan"), float("inf"), "1", True, None):
            with pytest.raises(wrench_config.WrenchConfigError, match="bad value|integer"):
                wrench_config.validate(_with(cfg, path, bad))
        if _key(path) not in _SIGNED:
            with pytest.raises(wrench_config.WrenchConfigError):
                wrench_config.validate(_with(cfg, path, -1))
        if _key(path) in _ZERO_BAD:
            with pytest.raises(wrench_config.WrenchConfigError):
                wrench_config.validate(_with(cfg, path, 0))


@pytest.mark.parametrize("path", [("render", "res"), ("render", "samples"), ("modal", "cpu_count"), ("modal", "timeout_s"),
                                  ("mark", "mark_px"), ("mark", "hi_px"), ("render", "seed"), ("wrench", "circle_res"),
                                  ("wrench", "ridge_bevel", "segments"), ("mark", "export", "webp_quality")])
def test_integer_fields_reject_fractions(path):
    with pytest.raises(wrench_config.WrenchConfigError):
        wrench_config.validate(_with(wrench_config.load(), path, 2.5))


def test_new_tunables_are_in_the_schema_and_no_literal_is_left_in_the_render_code():
    root = Path(__file__).resolve().parents[1]
    scene = (root / "assets/3d/wrench/scene.py").read_text(encoding="utf-8")
    for literal in ("default_value = 0.32", "default_value = 0.4", "* 0.45", "smooth(ridge, 50)"):
        assert literal not in scene
    assert "1.4" not in (root / "assets/3d/wrench/shape.py").read_text(encoding="utf-8")
    assert "quality=90" not in (root / "tools/wrench_compose.py").read_text(encoding="utf-8")


def test_shadow_strength_and_cutoff_come_from_the_config(tmp_path):
    import numpy as np
    from PIL import Image
    import wrench_compose

    cfg = wrench_config.load()["mark"]
    assert {"reach", "cutoff"} <= set(cfg["shadow"])
    assert "near * 3" not in (Path(wrench_compose.__file__)).read_text() and "< 0.02" not in Path(wrench_compose.__file__).read_text()
    # a wrench dot with a soft floor shadow beside it: a higher cutoff drops the shadow, a lower one keeps it
    d = tmp_path
    full, cut = np.zeros((40, 40, 4), np.uint8), np.zeros((40, 40, 4), np.uint8)
    cut[18:22, 18:22] = (200, 200, 200, 255)
    full[:] = cut
    full[18:22, 22:26, 3] = 200
    Image.fromarray(full).save(d / "wrench_v.png")
    Image.fromarray(cut).save(d / "wrench_v_cut.png")

    def width(**over):
        sh = dict(cfg["shadow"], keep=1.0, falloff_px=1, **over)
        return wrench_compose.cutout(d, "v", sh, cfg["tone"]).width

    assert width(cutoff=0.0) > width(cutoff=0.9)


def test_budget_lane_key_still_loads():
    """[budget] `lane` is the former name of `job`; an old config still loads."""
    import tomllib
    raw = tomllib.loads(wrench_config.CONFIG.read_text())
    raw["budget"]["lane"] = raw["budget"].pop("job")
    assert wrench_config.validate(raw)["budget"]["job"] == wrench_config.load()["budget"]["job"]
    raw["budget"]["job"] = "x"
    with pytest.raises(wrench_config.WrenchConfigError, match="unknown"):
        wrench_config.validate(raw)
