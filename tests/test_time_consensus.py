"""phases-mode consensus by time, not wording.

In contiguous phases mode consensus clusters phase-boundary times
across runs (majority support, median time) and tiles the known
chunk extent (else the observed span) with no gaps by construction;
wording only chooses each segment's label (most-agreed `same_action`
group, then largest overlap) and flags disagreement. A run ending
past the known chunk end beyond tolerance is excluded like any
invalid run, so one over-long run cannot stretch the tail.
Legacy/sparse and exact modes keep the wording-grouped consensus
untouched and ignore the extent.

The spark3/spark4 fixture attempts tile [0, 60] identically with and
without the extent; the spark5 attempts (one over-long run each) tile
[0, 60] only with it — without it the tail stretches to 214/125 and
the contiguity gate fails, the live spark5 killer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.chunks import ContiguityError, require_contiguous_cover
from app.spark import (
    ChunkExtent,
    SparkStageConfig,
    StepsDisagreement,
    consensus_bundle,
    validate_legacy_spark_payload,
    validate_spark_payload,
)

from test_spark_schema import make_payload, make_step

FIXTURE = Path(__file__).parent / "fixtures" / "time_consensus.json"
FIXTURE5 = Path(__file__).parent / "fixtures" / "chunk_extent.json"
LIVE = SparkStageConfig(spread_s=2.0)  # the live runs ran tol=2.0
LEGACY = SparkStageConfig(span_mode="legacy")
EXACT = SparkStageConfig(alignment_mode="exact")
EXTENT60 = ChunkExtent(duration_s=60.0, tolerance_s=2.0)

# (start, end, action, spread, flagged) per attempt, in order.
EXPECTED = {
    "spark3/chunk0": [
        (0.0, 10.0, "Introduce laptop on table", 1.0, True),
        (10.0, 17.0, "Flip over and remove external battery", 1.0, False),
        (17.0, 36.0, "Boot to BIOS and disable internal battery", 1.0, False),
        (36.0, 60.0, "Flip over and loosen bottom cover screws", 1.0, True),
    ],
    "spark3/chunk0_retry1": [
        (0.0, 10.0, "Intro show closed laptop", 1.0, True),
        (10.0, 17.0, "Flip over and remove external battery", 1.0, False),
        (17.0, 31.0, "Power on and disable internal battery in BIOS", 0.0, True),
        (31.0, 37.0, "Shut down and flip to bottom side", 1.0, True),
        (37.0, 60.0, "Loosen bottom cover screws", 1.0, True),
    ],
    "spark4/chunk0": [
        (0.0, 10.0, "Intro overview", 1.0, True),
        (10.0, 17.0, "Flip over and remove external battery", 2.0, True),
        (17.0, 36.5, "Power on and disable internal battery in BIOS", 2.0, True),
        (36.5, 60.0, "Unscrew and open bottom cover", 1.0, True),
    ],
    "spark4/chunk0_retry1": [
        (0.0, 9.0, "Show device overview", 1.0, True),
        (9.0, 17.0, "Flip and remove external battery", 1.0, False),
        (17.0, 23.0, "Power on to startup menu", 0.0, True),
        (23.0, 31.0, "Disable built-in battery in BIOS", 0.0, True),
        (31.0, 37.0, "Power off and flip to bottom", 2.0, True),
        (37.0, 60.0, "Loosen bottom cover screws", 2.0, True),
    ],
}


def _load_attempts() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["attempts"]


def _consensus(step_sets, config=LIVE, extent=None):
    runs = [validate_spark_payload(make_payload(steps=steps)) for steps in step_sets]
    return consensus_bundle(runs, config, chunk_extent=extent)


def _row(step):
    return (
        step.start_s,
        step.end_s,
        step.action,
        step.time_spread_s,
        step.time_needs_check,
    )


# --- unit behaviour ----------------------------------------------------------


def test_one_run_only_boundary_drops_without_leaving_a_gap() -> None:
    pair = (
        make_step(start=0.0, end=10.0, action="Introduce gadget on desk",
                  parts=("gadget lid",)),
        make_step(start=10.0, end=20.0, action="Remove base screws",
                  parts=("base screws",)),
    )
    split = (
        make_step(start=0.0, end=10.0, action="Introduce gadget on desk",
                  parts=("gadget lid",)),
        make_step(start=10.0, end=15.0, action="Remove base screws",
                  parts=("base screws",)),
        make_step(start=15.0, end=20.0, action="Lift off base cover",
                  parts=("base cover",)),
    )
    bundle = _consensus([pair, pair, split])
    assert [_row(step) for step in bundle.steps] == [
        (0.0, 10.0, "Introduce gadget on desk", 0.0, False),
        (10.0, 20.0, "Remove base screws", 0.0, False),
    ]  # the lone 15.0 edge drops; the span still tiles


def test_short_phase_survives_wide_tolerance() -> None:
    """one boundary per run per cluster.

    Three identical [0,10),[10,12),[12,20) runs at tolerance 5 keep
    three phases; the 10 and 12 edges never merge into one median.
    """
    triple = (
        make_step(start=0.0, end=10.0),
        make_step(start=10.0, end=12.0),
        make_step(start=12.0, end=20.0),
    )
    bundle = _consensus(
        [triple, triple, triple], SparkStageConfig(spread_s=5.0)
    )
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 10.0),
        (10.0, 12.0),
        (12.0, 20.0),
    ]


def test_reordered_runs_are_excluded_and_fail_closed() -> None:
    """reordered runs are not partitions."""
    bad = (
        make_step(start=10.0, end=20.0),
        make_step(start=0.0, end=10.0),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([bad, bad, bad])


def test_overlapping_runs_are_excluded_and_fail_closed() -> None:
    """overlapping runs are never normalized."""
    bad = (
        make_step(start=0.0, end=12.0),
        make_step(start=10.0, end=20.0),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([bad, bad, bad])


def test_small_gap_runs_are_excluded_and_fail_closed() -> None:
    """a 0.05 s gap is excluded, never normalized."""
    bad = (
        make_step(start=0.0, end=10.0),
        make_step(start=10.05, end=20.0),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([bad, bad, bad])


def test_small_overlap_runs_are_excluded_and_fail_closed() -> None:
    """a 0.05 s overlap is excluded, never normalized."""
    bad = (
        make_step(start=0.0, end=10.05),
        make_step(start=10.0, end=20.0),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([bad, bad, bad])


def test_one_bad_run_is_excluded_consensus_from_other_two() -> None:
    """a single invalid run drops, two good remain."""
    good = (
        make_step(start=0.0, end=10.0),
        make_step(start=10.0, end=20.0),
    )
    bad = (
        make_step(start=0.0, end=12.0),
        make_step(start=10.0, end=20.0),
    )
    bundle = _consensus([good, good, bad])
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 10.0),
        (10.0, 20.0),
    ]


def _good_60():
    return (
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=60.0),
    )


def test_overlong_run_excluded_two_good_runs_give_consensus() -> None:
    """One run past the chunk end drops; the other two tile [0, 60]."""
    long = (
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=60.0),
        make_step(start=60.0, end=214.0),
    )
    bundle = _consensus([_good_60(), _good_60(), long], extent=EXTENT60)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
    ]


def test_majority_overlong_runs_fail_closed() -> None:
    """Two over-long runs of three leave fewer than a majority usable."""
    long = (
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=125.0),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([_good_60(), long, long], extent=EXTENT60)


def test_run_ending_within_tolerance_is_kept_to_chunk_end() -> None:
    """A run ending 1 s past the chunk end stays; consensus ends at 60."""
    near = (
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=61.0),
    )
    bundle = _consensus([_good_60(), _good_60(), near], extent=EXTENT60)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
    ]
    assert bundle.steps[-1].time_spread_s == 1.0  # ends {60, 60, 61}


def test_run_ending_exactly_at_duration_plus_tolerance_is_kept() -> None:
    """The exclusion bound is exclusive: ending AT 62.0 with tol 2 passes."""
    edge = (
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=62.0),
    )
    bundle = _consensus([_good_60(), _good_60(), edge], extent=EXTENT60)
    assert bundle.steps[-1].end_s == 60.0


def test_run_ending_just_past_duration_plus_tolerance_is_excluded() -> None:
    """Ending at 62.01 with tol 2 excludes; two good runs still agree."""
    past = (
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=62.01),
    )
    bundle = _consensus([_good_60(), _good_60(), past], extent=EXTENT60)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
    ]


def test_short_run_is_excluded_not_stretched() -> None:
    """A run covering only [0, 50] drops; tiling gains no coverage."""
    short = (
        make_step(start=0.0, end=25.0),
        make_step(start=25.0, end=50.0),
    )
    bundle = _consensus([_good_60(), _good_60(), short], extent=EXTENT60)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
    ]


def test_late_start_run_is_excluded_not_backfilled() -> None:
    """A run covering only [5, 60] drops; the opening is not invented."""
    late = (
        make_step(start=5.0, end=30.0),
        make_step(start=30.0, end=60.0),
    )
    bundle = _consensus([_good_60(), _good_60(), late], extent=EXTENT60)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
    ]


def test_run_covering_within_tolerance_on_both_sides_is_kept() -> None:
    """[1, 59] with tol 2 stays; the consensus still tiles exactly [0, 60]."""
    near = (
        make_step(start=1.0, end=30.0),
        make_step(start=30.0, end=59.0),
    )
    bundle = _consensus([_good_60(), _good_60(), near], extent=EXTENT60)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
    ]


def test_all_fragment_runs_fail_closed() -> None:
    """Three sparse fragments agree with each other but cover nothing:
    no usable run remains, so consensus raises instead of tiling."""
    fragment = (make_step(start=0.5, end=2.5),)
    with pytest.raises(StepsDisagreement):
        _consensus([fragment, fragment, fragment], extent=EXTENT60)


def test_tolerance_swallowing_the_clip_excludes_disjoint_fragments() -> None:
    """Short-clip regime: tolerance 2 over a 1 s clip demands real
    coverage of both endpoints, so three runs on [1.5, 2.5] fail
    closed instead of tiling a fabricated [0, 1]."""
    extent = ChunkExtent(duration_s=1.0, tolerance_s=2.0)
    fragment = (
        make_step(start=1.5, end=2.0),
        make_step(start=2.0, end=2.5),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([fragment, fragment, fragment], extent=extent)


def test_tolerance_swallowing_the_clip_keeps_full_cover() -> None:
    """Positive control: three runs covering [0, 1] tile it gap-free."""
    extent = ChunkExtent(duration_s=1.0, tolerance_s=2.0)
    full = (
        make_step(start=0.0, end=0.5),
        make_step(start=0.5, end=1.0),
    )
    bundle = _consensus([full, full, full], extent=extent)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]


def test_tolerance_swallowing_the_clip_excludes_a_short_run() -> None:
    """A run covering [0, 0.9] of a 1 s clip misses the far endpoint,
    so it drops even though two full-cover runs still agree."""
    extent = ChunkExtent(duration_s=1.0, tolerance_s=2.0)
    full = (
        make_step(start=0.0, end=0.5),
        make_step(start=0.5, end=1.0),
    )
    short = (
        make_step(start=0.0, end=0.45),
        make_step(start=0.45, end=0.9),
    )
    bundle = _consensus([full, full, short], extent=extent)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]


def test_tolerance_swallowing_the_clip_still_excludes_overshoot() -> None:
    """Endpoint cover without an overshoot bound would admit a [0, 5]
    run on a 1 s clip; past duration plus tolerance still excludes."""
    extent = ChunkExtent(duration_s=1.0, tolerance_s=2.0)
    full = (
        make_step(start=0.0, end=0.5),
        make_step(start=0.5, end=1.0),
    )
    over = (
        make_step(start=0.0, end=0.5),
        make_step(start=0.5, end=5.0),
    )
    bundle = _consensus([full, full, over], extent=extent)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]


def test_tolerance_just_under_duration_keeps_slack() -> None:
    """Boundary pin: with tolerance just under the duration the slack
    rule applies, so a run within tolerance on both sides stays."""
    extent = ChunkExtent(duration_s=2.0, tolerance_s=1.9)
    full = (
        make_step(start=0.0, end=1.0),
        make_step(start=1.0, end=2.0),
    )
    near = (
        make_step(start=1.0, end=2.0),
        make_step(start=2.0, end=2.5),
    )
    bundle = _consensus([full, full, near], extent=extent)
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]


def test_extent_rejects_misshaped_values() -> None:
    for bad_duration in (0, -1.0, True, "60", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ChunkExtent(duration_s=bad_duration, tolerance_s=2.0)  # type: ignore[arg-type]
    for bad_tolerance in (-1.0, True, "2", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ChunkExtent(duration_s=60.0, tolerance_s=bad_tolerance)  # type: ignore[arg-type]
    runs = [validate_spark_payload(make_payload(steps=_good_60()))] * 3
    with pytest.raises(ValueError):
        consensus_bundle(runs, LIVE, chunk_extent="60")  # type: ignore[arg-type]


def test_legacy_and_exact_modes_ignore_the_extent() -> None:
    """Non-contiguous modes keep model spans with or without an extent."""
    step_sets = [_good_60(), _good_60(), _good_60()]
    for config in (LEGACY, EXACT):
        plain = _consensus(step_sets, config)
        with_extent = _consensus(step_sets, config, EXTENT60)
        assert [(s.start_s, s.end_s) for s in with_extent.steps] == [
            (s.start_s, s.end_s) for s in plain.steps
        ]


def test_bridged_label_vote_flags_disagreement() -> None:
    """complete-link, no bridge through B.

    A matches B and B matches C, but A and C explicitly disagree
    (disjoint parts); the segment is kept and flagged.
    """
    runs = [
        (
            make_step(
                start=0.0, end=10.0, action="Remove cover screws",
                parts=("fan",),
            ),
        ),
        (
            make_step(
                start=0.0, end=10.0, action="Remove cover screws",
                parts=("fan", "battery"),
            ),
        ),
        (
            make_step(
                start=0.0, end=10.0, action="Remove cover screws",
                parts=("battery",),
            ),
        ),
    ]
    (step,) = _consensus(runs).steps
    assert (step.start_s, step.end_s) == (0.0, 10.0)
    assert step.time_needs_check is True


def test_disagreeing_run_loses_the_label_vote_and_flags() -> None:
    agree = (
        make_step(start=0.0, end=10.0, action="Remove base screws",
                  parts=("base screws",)),
    )
    dissent = (
        make_step(start=0.0, end=10.0, action="Pry off base cover",
                  parts=("base cover",)),
    )
    bundle = _consensus([agree, agree, dissent])
    (step,) = bundle.steps
    assert _row(step) == (0.0, 10.0, "Remove base screws", 0.0, True)


def test_full_wording_disagreement_keeps_a_deterministic_segment() -> None:
    runs = [
        (
            make_step(start=0.0, end=10.0, action="Remove base screws",
                      parts=("base screws",)),
        ),
        (
            make_step(start=0.0, end=10.0, action="Pry off base cover",
                      parts=("base cover",)),
        ),
        (
            make_step(start=0.0, end=10.0, action="Inspect cooling fan",
                      parts=("cooling fan",)),
        ),
    ]
    first = _consensus(runs)
    second = _consensus(runs)
    assert [_row(step) for step in first.steps] == [
        (0.0, 10.0, "Remove base screws", 0.0, True)
    ]  # three-way tie breaks to the earliest run, kept and flagged
    assert [_row(step) for step in second.steps] == [
        (0.0, 10.0, "Remove base screws", 0.0, True)
    ]


def test_fewer_than_majority_nonempty_runs_fails_closed() -> None:
    full = (make_step(start=0.0, end=10.0),)
    with pytest.raises(StepsDisagreement):
        _consensus([full, (), ()])


def test_majority_nonempty_runs_proceed_without_the_empty_run() -> None:
    full = (
        make_step(start=0.0, end=10.0, action="Remove base screws",
                  parts=("base screws",)),
        make_step(start=10.0, end=20.0, action="Pry off base cover",
                  parts=("base cover",)),
    )
    bundle = _consensus([full, full, ()])
    assert [_row(step) for step in bundle.steps] == [
        (0.0, 10.0, "Remove base screws", 0.0, False),
        (10.0, 20.0, "Pry off base cover", 0.0, False),
    ]


def test_more_than_twelve_segments_fails_closed_not_truncated() -> None:
    # Three valid 12-step runs whose supported edges land on twelve
    # distinct times (1 in runs 1+3, 2-11 in all three, 12 in runs
    # 2+3): thirteen segments tile the span, past the 12-step bundle
    # limit.
    def _partition(cuts):
        bounds = [0.0] + [float(c) for c in cuts] + [15.0]
        return tuple(
            make_step(start=start, end=end)
            for start, end in zip(bounds, bounds[1:])
        )

    runs = [
        _partition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        _partition([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        _partition([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    ]
    with pytest.raises(StepsDisagreement):
        _consensus(runs, SparkStageConfig(spread_s=0.5))


def test_eight_segments_still_pass() -> None:
    runs = [
        tuple(
            make_step(start=float(i), end=float(i + 1))
            for i in range(8)
        )
        for _ in range(3)
    ]
    bundle = _consensus(runs, SparkStageConfig(spread_s=0.5))
    assert len(bundle.steps) == 8
    assert [s.number for s in bundle.steps] == list(range(1, 9))


def test_legacy_mode_still_fails_closed_on_wording_split() -> None:
    """The same wording split time consensus keeps, legacy drops."""
    runs = [
        (
            make_step(start=0.0, end=10.0, action="Remove base screws",
                      parts=("base screws",)),
        ),
        (
            make_step(start=0.0, end=10.0, action="Pry off base cover",
                      parts=("base cover",)),
        ),
        (
            make_step(start=0.0, end=10.0, action="Inspect cooling fan",
                      parts=("cooling fan",)),
        ),
    ]
    with pytest.raises(StepsDisagreement):
        _consensus(runs, LEGACY)
    with pytest.raises(StepsDisagreement):
        _consensus(runs, EXACT)
    (kept,) = _consensus(runs, LIVE).steps
    assert kept.time_needs_check is True


# --- live-raw fixtures -------------------------------------------------------


def test_fixture_is_redacted_step_payloads_only() -> None:
    attempts = _load_attempts()
    assert [attempt["id"] for attempt in attempts] == list(EXPECTED)
    for attempt in attempts:
        assert len(attempt["runs"]) == 3
        for payload in attempt["runs"]:
            assert set(payload) == {"device_guess", "confidence", "steps"}
            for step in payload["steps"]:
                assert set(step) == {
                    "number", "start_s", "end_s", "action",
                    "tools", "parts", "hazards",
                }
                for part in step["parts"]:
                    assert set(part) == {"phrase", "position_hint"}
    text = FIXTURE.read_text(encoding="utf-8")
    for marker in (
        "file-", "/tmp/", "/Users/", "Bearer", "META_API",
        ".png", ".mp4", "job_id", "file_id",
    ):
        assert marker not in text


@pytest.mark.parametrize("attempt_id", list(EXPECTED))
def test_fixture_attempt_matches_recorded_consensus(attempt_id: str) -> None:
    attempt = next(a for a in _load_attempts() if a["id"] == attempt_id)
    runs = [validate_legacy_spark_payload(payload) for payload in attempt["runs"]]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT60)
    assert [_row(step) for step in bundle.steps] == EXPECTED[attempt_id]


@pytest.mark.parametrize("attempt_id", list(EXPECTED))
def test_fixture_attempt_tiles_its_chunk(attempt_id: str) -> None:
    attempt = next(a for a in _load_attempts() if a["id"] == attempt_id)
    runs = [validate_legacy_spark_payload(payload) for payload in attempt["runs"]]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT60)
    require_contiguous_cover(bundle.steps, 60.0, tolerance_s=2.0)


# --- spark5 live-raw fixtures: one over-long run per attempt ------------------
#
# Failing-before cases: without the extent the tail stretches to 214/125
# and the contiguity gate fails (the live spark5 killer); with the known
# extent the over-long run drops and the other two tile [0, 60].
# Boundaries below were checked against the live per-phase diagnosis
# (the saved results) and the reference step table, not just replayed.

EXPECTED5 = {
    "spark5/chunk0": [
        (0.0, 9.5, "Intro overview", 1.0, True),
        (9.5, 17.0, "Flip over and remove external battery", 1.0, False),
        (17.0, 31.0, "Power on and disable built-in battery in BIOS", 0.0, True),
        (31.0, 37.0, "Shut down and flip to service side", 0.0, True),
        (37.0, 60.0, "Loosen screws and open bottom cover", 0.0, True),
    ],
    "spark5/chunk0_retry1": [
        (0.0, 9.5, "Introduce closed laptop", 1.0, False),
        (9.5, 17.0, "Flip and remove external battery", 1.0, False),
        (17.0, 31.0, "Disable internal battery in BIOS", 0.0, False),
        (31.0, 37.5, "Power down and reposition for opening", 1.0, True),
        (37.5, 60.0, "Loosen screws and open bottom cover", 1.0, True),
    ],
}


def _load_attempts5() -> list[dict]:
    return json.loads(FIXTURE5.read_text(encoding="utf-8"))["attempts"]


def test_fixture5_is_redacted_step_payloads_only() -> None:
    attempts = _load_attempts5()
    assert [attempt["id"] for attempt in attempts] == list(EXPECTED5)
    for attempt in attempts:
        assert len(attempt["runs"]) == 3
        for payload in attempt["runs"]:
            assert set(payload) == {"device_guess", "confidence", "steps"}
            for step in payload["steps"]:
                assert set(step) == {
                    "number", "start_s", "end_s", "action",
                    "tools", "parts", "hazards",
                }
                for part in step["parts"]:
                    assert set(part) == {"phrase", "position_hint"}
    text = FIXTURE5.read_text(encoding="utf-8")
    for marker in (
        "file-", "/tmp/", "/Users/", "Bearer", "META_API",
        ".png", ".mp4", "job_id", "file_id",
    ):
        assert marker not in text


@pytest.mark.parametrize("attempt_id", list(EXPECTED5))
def test_fixture5_attempt_matches_recorded_consensus(attempt_id: str) -> None:
    attempt = next(a for a in _load_attempts5() if a["id"] == attempt_id)
    runs = [validate_legacy_spark_payload(payload) for payload in attempt["runs"]]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT60)
    assert [_row(step) for step in bundle.steps] == EXPECTED5[attempt_id]


@pytest.mark.parametrize("attempt_id", list(EXPECTED5))
def test_fixture5_attempt_tiles_its_chunk(attempt_id: str) -> None:
    attempt = next(a for a in _load_attempts5() if a["id"] == attempt_id)
    runs = [validate_legacy_spark_payload(payload) for payload in attempt["runs"]]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT60)
    require_contiguous_cover(bundle.steps, 60.0, tolerance_s=2.0)


@pytest.mark.parametrize("attempt_id", list(EXPECTED5))
def test_fixture5_attempt_without_extent_reproduces_live_tail(attempt_id: str) -> None:
    """Without the extent the killer survives: the tail overhangs the chunk."""
    attempt = next(a for a in _load_attempts5() if a["id"] == attempt_id)
    runs = [validate_legacy_spark_payload(payload) for payload in attempt["runs"]]
    bundle = consensus_bundle(runs, LIVE)
    assert bundle.steps[-1].end_s > 62.0  # 214.0 and 125.0, past 60+2
    with pytest.raises(ContiguityError, match="past the 60"):
        require_contiguous_cover(bundle.steps, 60.0, tolerance_s=2.0)
