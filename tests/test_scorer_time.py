"""Time-overlap scorer: GT vs GT is N/N on every reference list.

Rule: align guide rows to truth by time overlap, keep
the bar at 7 of 9 within ±3 s. Wording never vetoes
a pair; it is reported per pair instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.scorer_time import (
    PASS_COUNT,
    PASS_OF,
    START_END_TOLERANCE_S,
    STRICT_TOLERANCE_1_S,
    STRICT_TOLERANCE_2_S,
    _assign_max_overlap,
    align_guide_to_gt,
    load_scorer_law,
    score_guide,
    wording_similarity,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _sample_gt() -> list[dict]:
    data = json.loads((FIXTURES / "sample_steps_proposed.json").read_text(encoding="utf-8"))
    return data["steps"]


def _row(number: int, start: float, end: float, action: str) -> dict:
    return {"number": number, "start_s": start, "end_s": end, "action": action}


# Reference V2/V3/V4 lists (frame audits 2026-10-28, decisions recorded in
# each file): times in seconds, actions verbatim.
V2_GT = [
    _row(1, 0, 63, "Intro and identify the replacement RAM modules"),
    _row(2, 63, 112, "Remove the removable battery and disable the built-in battery in BIOS"),
    _row(3, 112, 125, "Unscrew the base cover"),
    _row(4, 125, 168, "Pry off and remove the base cover"),
    _row(5, 168, 235, "Replace both RAM modules"),
    _row(6, 235, 279, "Reinstall the base cover, reconnect power, and reinstall the removable battery"),
]

V3_GT = [
    _row(1, 0, 33, "Intro and identify the M.2 SSD"),
    _row(2, 33, 60, "Flip the laptop, remove the removable battery, and disable the built-in battery in BIOS"),
    _row(3, 60, 83, "Unscrew the base cover"),
    _row(4, 83, 110, "Pry off and remove the base cover"),
    _row(5, 110, 151, "Install the M.2 SSD in the empty WWAN slot"),
    _row(6, 151, 185, "Reinstall the base cover and removable battery"),
]

V4_GT = [
    _row(1, 0, 9, "Intro and show the replacement RAM"),
    _row(2, 9, 36, "Flip the laptop, remove the removable battery, and disable the built-in battery in BIOS"),
    _row(3, 36, 59, "Unscrew the base cover"),
    _row(4, 59, 88, "Pry off and remove the base cover"),
    _row(5, 88, 124, "Replace both RAM modules"),
    _row(6, 124, 166, "Reinstall the base cover and removable battery; reconnect power"),
]


def test_gt_vs_gt_scores_full_marks_on_all_four_lists() -> None:
    cases = [("sample", _sample_gt()), ("V2", V2_GT), ("V3", V3_GT), ("V4", V4_GT)]
    for name, gt in cases:
        report = score_guide(gt, gt)
        assert report.strict_passes == len(gt), name
        assert report.missing_gt == (), name
        assert report.extra_guide == (), name
        assert report.coverage == pytest.approx(1.0), name
        assert report.strict_passes_2s == len(gt), name
        assert report.strict_passes_1s == len(gt), name
        assert report.passed is True, name


def test_sample_fixture_carries_the_170_4_boundary() -> None:
    # Rule: the step 8/9 boundary is 170.4 (the
    # power-cable sequence belongs to "Power on and verify"), not 179.
    data = json.loads((FIXTURES / "sample_steps_proposed.json").read_text(encoding="utf-8"))
    assert data["status"] == "proposed"
    assert data["boundary_note"].startswith("2026-10-29")
    assert "170.4" in data["boundary_note"]
    steps = data["steps"]
    assert steps[7]["end_s"] == 170.4
    assert steps[8]["start_s"] == 170.4


def test_sample_gt_vs_gt_is_nine_of_nine() -> None:
    gt = _sample_gt()
    assert [s["number"] for s in gt] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    report = score_guide(gt, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(n, n) for n in range(1, 10)]
    assert all(m.passed for m in report.matches)
    assert report.strict_passes == 9
    assert report.passed is True


def test_old_veto_rows_now_pair_with_wording_reported() -> None:
    # Rows 1, 6, 7 failed the old same-action gate (generic "battery",
    # empty part sets) even against identical text; now they pair and the
    # wording verdict is reported without changing the pass count.
    gt = _sample_gt()
    report = score_guide(gt, gt)
    by_gt = {m.gt_number: m for m in report.matches}
    for number in (1, 6, 7):
        assert by_gt[number].passed is True
        assert by_gt[number].wording_same_action is False
        assert by_gt[number].wording_similarity == pytest.approx(1.0)


def test_same_times_with_different_wording_still_pass() -> None:
    gt = _sample_gt()
    guide = [
        dict(step, action=action)
        for step, action in zip(gt, [
            "Opening titles",
            "Turn the machine; take the external battery",
            "Switch off the integrated cell in setup",
            "Loosen the bottom cover",
            "Lever off and detach the bottom cover",
            "Take out the external battery",
            "Fit the external battery",
            "Put back the bottom cover and external battery; plug in",
            "Demo battery check",
        ])
    ]
    report = score_guide(guide, gt)
    assert report.strict_passes == 9
    assert report.passed is True
    assert any(m.wording_same_action is False for m in report.matches)


def test_guide_shifted_3_5s_fails_shifted_2_9s_passes() -> None:
    gt = [_row(1, 100.0, 120.0, "Unscrew the base cover")]
    shifted_out = [_row(1, 103.5, 123.5, "Unscrew the base cover")]
    shifted_in = [_row(1, 102.9, 122.9, "Unscrew the base cover")]
    failing = score_guide(shifted_out, gt)
    assert [(m.guide_number, m.gt_number) for m in failing.matches] == [(1, 1)]
    assert failing.matches[0].passed is False
    assert failing.strict_passes == 0
    assert failing.passed is False
    passing = score_guide(shifted_in, gt)
    assert passing.matches[0].passed is True
    assert passing.strict_passes == 1
    assert passing.passed is True  # 1/1 meets the 7-of-9 ratio


def test_boundary_exactly_at_tolerance_passes() -> None:
    gt = [_row(1, 100.0, 120.0, "Unscrew the base cover")]
    guide = [_row(1, 103.0, 117.0, "Unscrew the base cover")]
    report = score_guide(guide, gt)
    assert report.matches[0].passed is True  # <= 3.0 s, unchanged rule


def test_extra_and_missing_steps_are_reported_not_crashed() -> None:
    gt = [
        _row(1, 0.0, 10.0, "Intro"),
        _row(2, 10.0, 20.0, "Unscrew the base cover"),
        _row(3, 20.0, 30.0, "Pry off the base cover"),
    ]
    guide = [
        _row(1, 0.0, 10.0, "Intro"),
        _row(2, 10.0, 20.0, "Unscrew the base cover"),
        _row(9, 50.0, 60.0, "Unrelated epilogue"),
    ]
    report = score_guide(guide, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 1), (2, 2)]
    assert report.missing_gt == (3,)
    assert report.extra_guide == (9,)
    assert report.strict_passes == 2
    assert report.passed is False  # 2/3 below the 7-of-9 ratio


def test_touching_windows_do_not_pair() -> None:
    gt = [_row(1, 0.0, 10.0, "Intro")]
    guide = [_row(1, 10.0, 20.0, "Intro")]
    report = score_guide(guide, gt)
    assert report.matches == ()
    assert report.missing_gt == (1,)
    assert report.extra_guide == (1,)


def test_over_split_guide_pairs_larger_half_and_fails_honestly() -> None:
    gt = [_row(4, 37.0, 60.0, "Unscrew the base cover")]
    guide = [
        _row(1, 37.0, 49.0, "Unscrew part one"),
        _row(2, 49.0, 60.0, "Unscrew part two"),
    ]
    report = score_guide(guide, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 4)]
    assert report.matches[0].passed is False  # end 11 s off
    assert report.missing_gt == ()
    assert report.extra_guide == (2,)
    assert report.strict_passes == 0


def test_assignment_is_exact_not_greedy() -> None:
    # A largest-edge-first greedy takes (0, 0) = 10 and strands row 1
    # (total 10); the exact assignment takes 9 + 9 = 18.
    assigned = _assign_max_overlap([[10.0, 9.0], [9.0, 0.0]])
    assert assigned == [1, 0]


def test_assignment_prefers_total_overlap_end_to_end() -> None:
    gt = [
        _row(1, 0.0, 10.0, "Unscrew the base cover"),
        _row(2, 10.0, 20.0, "Pry off the base cover"),
    ]
    guide = [
        _row(1, 0.0, 19.0, "Cover work alpha"),
        _row(2, 0.0, 10.0, "Cover work beta"),
    ]
    pairs, missing, extra = align_guide_to_gt(guide, gt)
    assert [(g.number, t.number) for g, t in pairs] == [(2, 1), (1, 2)]
    assert missing == []
    assert extra == []


def test_strict_lines_are_report_only() -> None:
    gt = [_row(1, 100.0, 120.0, "Unscrew the base cover")]
    guide = [_row(1, 102.5, 122.5, "Unscrew the base cover")]
    report = score_guide(guide, gt)
    assert report.strict_passes == 1
    assert report.strict_passes_2s == 0
    assert report.strict_passes_1s == 0
    assert report.passed is True  # the bar is still ±3 s


def test_wording_similarity_bounds() -> None:
    assert wording_similarity("Pry off the base cover", "Pry off the base cover") == pytest.approx(1.0)
    assert wording_similarity("Unscrew the base cover", "Pry off the base cover") == pytest.approx(0.5)
    assert wording_similarity("Intro", "Remove the internal battery") == pytest.approx(0.0)


def test_scoring_law_values_and_loud_missing_key() -> None:
    assert START_END_TOLERANCE_S == pytest.approx(3.0)
    assert STRICT_TOLERANCE_2_S == pytest.approx(2.0)
    assert STRICT_TOLERANCE_1_S == pytest.approx(1.0)
    assert (PASS_COUNT, PASS_OF) == (7, 9)
    with pytest.raises(KeyError, match="no key"):
        load_scorer_law("no_such_key")
