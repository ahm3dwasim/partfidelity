"""snap boundaries to cuts, fold short phases, confidence."""

from __future__ import annotations

import pytest

from app.boundaries import (
    fold_short_phases,
    refine_consensus_bundle,
    snap_bundle_boundaries,
)
from app.contracts import BoundaryInfo, Step, StepBundle


def _record(time_s, *, support=3, total=3, check=False) -> BoundaryInfo:
    return BoundaryInfo(
        time_s=time_s,
        support_runs=support,
        total_runs=total,
        nearest_cut_s=None,
        cut_distance_s=None,
        snapped=False,
        snap_delta_s=0.0,
        needs_check=check,
    )


def _step(number, start, end, action="Remove the rear screws", **over) -> Step:
    fields = {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": (),
        "parts": (),
        "hazards": (),
        "time_spread_s": 0.1,
        "time_needs_check": False,
    }
    fields.update(over)
    return Step(**fields)


def _bundle(spans, actions=None, supports=None) -> StepBundle:
    steps = tuple(
        _step(n, s, e, (actions or {}).get(n, "Remove the rear screws"))
        for n, (s, e) in enumerate(spans, start=1)
    )
    records = tuple(
        _record(e, support=(supports or {}).get(n, 3))
        for n, (s, e) in enumerate(spans[:-1], start=1)
    )
    return StepBundle(
        device_guess="laptop",
        confidence=0.9,
        steps=steps,
        source_runs=3,
        timestamps_approximate=True,
        boundaries=records,
    )


def _spans(bundle):
    return [(s.start_s, s.end_s) for s in bundle.steps]


# --- snap ------------------------------------------------------------------


def test_snap_moves_each_edge_to_its_nearest_cut_in_window() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)])
    snapped = snap_bundle_boundaries(bundle, (9.4, 11.5, 19.0), 2.0)
    assert _spans(snapped) == [(0.0, 9.4), (9.4, 19.0), (19.0, 30.0)]
    first, second = snapped.boundaries
    assert (first.snapped, first.snap_delta_s) == (True, pytest.approx(-0.6))
    assert (first.nearest_cut_s, first.cut_distance_s) == (9.4, pytest.approx(0.0))
    assert (second.snapped, second.snap_delta_s) == (True, pytest.approx(-1.0))
    assert [s.time_needs_check for s in snapped.steps] == [False, False, False]


def test_snap_tie_between_two_cuts_picks_the_earlier() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0)])
    snapped = snap_bundle_boundaries(bundle, (9.0, 11.0), 2.0)
    assert _spans(snapped) == [(0.0, 9.0), (9.0, 20.0)]
    (record,) = snapped.boundaries
    assert record.snapped is True
    assert record.time_s == 9.0


def test_snap_never_crosses_another_boundary() -> None:
    # One cut sits between two close edges: the first snaps onto it
    # and the second stays, so no zero-length phase appears.
    bundle = _bundle([(0.0, 10.0), (10.0, 11.0), (11.0, 20.0)])
    snapped = snap_bundle_boundaries(bundle, (10.6,), 2.0)
    assert _spans(snapped) == [(0.0, 10.6), (10.6, 11.0), (11.0, 20.0)]
    first, second = snapped.boundaries
    assert (first.snapped, first.snap_delta_s) == (True, pytest.approx(0.6))
    assert (second.snapped, second.snap_delta_s) == (False, 0.0)
    assert second.time_s == 11.0


def test_snap_outside_the_window_stays_and_flags() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0)])
    snapped = snap_bundle_boundaries(bundle, (15.0,), 2.0)
    assert _spans(snapped) == [(0.0, 10.0), (10.0, 20.0)]
    (record,) = snapped.boundaries
    assert record.snapped is False
    assert record.nearest_cut_s == 15.0
    assert record.cut_distance_s == pytest.approx(5.0)
    assert record.needs_check is True
    assert [s.time_needs_check for s in snapped.steps] == [True, True]


def test_snap_without_cuts_flags_everything() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0)])
    snapped = snap_bundle_boundaries(bundle, (), 2.0)
    assert _spans(snapped) == [(0.0, 10.0), (10.0, 20.0)]
    (record,) = snapped.boundaries
    assert record.nearest_cut_s is None
    assert record.cut_distance_s is None
    assert record.needs_check is True


def test_snap_on_an_exact_cut_does_not_move() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0)])
    snapped = snap_bundle_boundaries(bundle, (10.0,), 2.0)
    assert _spans(snapped) == [(0.0, 10.0), (10.0, 20.0)]
    (record,) = snapped.boundaries
    assert record.snapped is False  # already exactly on the cut
    assert record.snap_delta_s == 0.0
    assert record.cut_distance_s == pytest.approx(0.0)
    assert record.needs_check is False


def test_snap_flags_only_steps_beside_a_bad_boundary() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)])
    snapped = snap_bundle_boundaries(bundle, (10.0,), 2.0)
    assert [s.time_needs_check for s in snapped.steps] == [False, True, True]


def test_snap_single_step_is_a_noop() -> None:
    bundle = _bundle([(0.0, 3.0)])
    assert snap_bundle_boundaries(bundle, (1.0,), 2.0) == bundle


@pytest.mark.parametrize(
    "bundle,cuts,window",
    [
        ("not-a-bundle", (), 2.0),
        (_bundle([(0.0, 10.0), (10.0, 20.0)]), (float("nan"),), 2.0),
        (_bundle([(0.0, 10.0), (10.0, 20.0)]), (-1.0,), 2.0),
        (_bundle([(0.0, 10.0), (10.0, 20.0)]), (), -1.0),
        (_bundle([(0.0, 10.0), (10.0, 20.0)]), (), float("inf")),
    ],
)
def test_snap_rejects_bad_inputs(bundle, cuts, window) -> None:
    with pytest.raises(ValueError):
        snap_bundle_boundaries(bundle, cuts, window)


def test_snap_rejects_wording_grouped_bundles() -> None:
    steps = (
        _step(1, 0.0, 10.0),
        _step(2, 10.0, 20.0),
    )
    bundle = StepBundle(
        device_guess="laptop",
        confidence=0.9,
        steps=steps,
        source_runs=3,
        timestamps_approximate=True,
        boundaries=(),  # wording-grouped: no per-boundary support
    )
    with pytest.raises(ValueError, match="one boundary record"):
        snap_bundle_boundaries(bundle, (10.0,), 2.0)


def test_snap_rejects_gapped_steps() -> None:
    bundle = _bundle([(0.0, 9.0), (10.0, 20.0)])
    with pytest.raises(ValueError, match="tiling"):
        snap_bundle_boundaries(bundle, (10.0,), 2.0)


# --- fold ------------------------------------------------------------------


def test_fold_merges_short_phase_into_agreeing_neighbour() -> None:
    bundle = _bundle(
        [(0.0, 10.0), (10.0, 12.0), (12.0, 30.0)],
        actions={
            1: "Remove the rear screws",
            2: "Remove the rear screws",
            3: "Close the cover",
        },
    )
    folded = fold_short_phases(bundle, 5.0)
    assert [(s.number, s.start_s, s.end_s, s.action) for s in folded.steps] == [
        (1, 0.0, 12.0, "Remove the rear screws"),
        (2, 12.0, 30.0, "Close the cover"),
    ]
    (record,) = folded.boundaries
    assert record.time_s == 12.0  # the surviving edge keeps its record


def test_fold_without_agreement_merges_into_shorter_neighbour() -> None:
    bundle = _bundle(
        [(0.0, 10.0), (10.0, 12.0), (12.0, 40.0)],
        actions={
            1: "Remove the rear screws",
            2: "Power on the laptop",
            3: "Close the cover",
        },
    )
    folded = fold_short_phases(bundle, 5.0)
    assert [(s.start_s, s.end_s) for s in folded.steps] == [
        (0.0, 12.0),
        (12.0, 40.0),
    ]
    assert folded.steps[0].action == "Remove the rear screws"


def test_fold_first_phase_merges_into_its_only_neighbour() -> None:
    bundle = _bundle([(0.0, 2.0), (2.0, 20.0)])
    folded = fold_short_phases(bundle, 5.0)
    assert [(s.number, s.start_s, s.end_s) for s in folded.steps] == [
        (1, 0.0, 20.0)
    ]
    assert folded.boundaries == ()


def test_fold_last_phase_merges_into_its_only_neighbour() -> None:
    bundle = _bundle([(0.0, 18.0), (18.0, 20.0)])
    folded = fold_short_phases(bundle, 5.0)
    assert [(s.number, s.start_s, s.end_s) for s in folded.steps] == [
        (1, 0.0, 20.0)
    ]
    assert folded.boundaries == ()


def test_fold_repeats_to_a_fixpoint() -> None:
    bundle = _bundle([(0.0, 2.0), (2.0, 4.0), (4.0, 20.0)])
    folded = fold_short_phases(bundle, 5.0)
    assert [(s.start_s, s.end_s) for s in folded.steps] == [(0.0, 20.0)]


def test_fold_keeps_a_lone_short_phase() -> None:
    bundle = _bundle([(0.0, 2.0)])
    assert fold_short_phases(bundle, 5.0) == bundle


def test_fold_merges_spread_and_flags() -> None:
    first = _step(1, 0.0, 10.0, time_spread_s=1.0, time_needs_check=True)
    second = _step(2, 10.0, 12.0, time_spread_s=3.0, time_needs_check=False)
    bundle = StepBundle(
        device_guess="laptop",
        confidence=0.9,
        steps=(first, second),
        source_runs=3,
        timestamps_approximate=True,
        boundaries=(_record(10.0),),
    )
    (merged,) = fold_short_phases(bundle, 5.0).steps
    assert (merged.start_s, merged.end_s) == (0.0, 12.0)
    assert merged.time_spread_s == 3.0
    assert merged.time_needs_check is True


def test_fold_rejects_bad_minimum_and_bad_sets() -> None:
    bundle = _bundle([(0.0, 10.0), (10.0, 20.0)])
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError):
            fold_short_phases(bundle, bad)
    with pytest.raises(ValueError):
        fold_short_phases(bundle, 5.0, generic_tokens={"battery"})  # type: ignore[arg-type]


# --- refine -----------------------------------------------------------------


def test_refine_snaps_then_folds() -> None:
    bundle = _bundle(
        [(0.0, 10.0), (10.0, 12.0), (12.0, 30.0)],
        actions={
            1: "Remove the rear screws",
            2: "Remove the rear screws",
            3: "Close the cover",
        },
    )
    refined = refine_consensus_bundle(
        bundle, cuts=(9.6, 12.4), snap_window_s=2.0, min_phase_length_s=5.0
    )
    assert [(s.start_s, s.end_s) for s in refined.steps] == [
        (0.0, 12.4),
        (12.4, 30.0),
    ]
    (record,) = refined.boundaries
    assert record.snapped is True
    assert record.time_s == 12.4


def test_refine_leaves_long_phases_on_unmoved_edges() -> None:
    # Distinct actions: this pins snap/fold, not the same-action merge
    # (covered in test_trim_merge.py).
    bundle = _bundle(
        [(0.0, 10.0), (10.0, 20.0)],
        actions={1: "Remove the rear screws", 2: "Close the cover"},
    )
    refined = refine_consensus_bundle(
        bundle, cuts=(), snap_window_s=2.0, min_phase_length_s=5.0
    )
    assert _spans(refined) == [(0.0, 10.0), (10.0, 20.0)]
    assert [s.time_needs_check for s in refined.steps] == [True, True]
