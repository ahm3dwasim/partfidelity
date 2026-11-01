"""Step-time tests: median consensus, flags, controls, refusals."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.contracts import Step
from app.routes.steps import refusal_payload, render_preview_page, render_time_controls, router
from app.spark import (
    NotATeardown,
    SparkRunResult,
    SparkStageConfig,
    StepsDisagreement,
    extract_steps,
    sam_calls_authorized,
)
from app.step_times import (
    format_approx,
    format_approx_range,
    median_time,
    needs_time_check,
    time_spread,
)
from app.stage_cache import SAMPLE_REPLAY_LABEL

from test_spark_schema import RecordingContext, ScriptedSparkClient, make_payload, make_step


def _result(payload, cost="0.08"):
    return SparkRunResult(payload=payload, cost_usd=Decimal(cost), latency_ms=900)


def _three_runs(step_sets):
    return [_result(make_payload(steps=steps)) for steps in step_sets]


def _legacy_client(log, runs):
    """A scripted client on the sparse path (wording-grouped consensus).

    Contiguous phases mode matches runs by time and keeps wording
    splits as flagged segments; the fail-closed wording pins below
    live on the unchanged legacy path.
    """
    client = ScriptedSparkClient(log, runs)
    client.spark_stage_config = SparkStageConfig(span_mode="legacy")
    return client


def test_three_runs_use_median_and_flag_spread_over_five():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(start=10.0, end=20.0), make_step(start=20.0, end=150.0)),
            (make_step(start=12.0, end=22.0), make_step(start=22.0, end=160.0)),
            (make_step(start=11.0, end=21.0), make_step(start=21.0, end=190.0)),
        ]
    )
    client = ScriptedSparkClient(log, runs)
    bundle = extract_steps(ctx, "vid-1", client=client)
    tight, wide = bundle.steps
    # Time consensus: the {20, 22, 21} edges agree on 21, and the two
    # segments tile the observed [10, 190] span with no gaps.
    assert (tight.start_s, tight.end_s) == (10.0, 21.0)
    assert tight.time_spread_s == 2.0
    assert tight.time_needs_check is False
    assert (wide.start_s, wide.end_s) == (21.0, 190.0)
    assert wide.time_spread_s == 40.0
    assert wide.time_needs_check is True  # spread strictly over 5 s


def test_timestamps_always_approximate():
    log: list = []
    ctx = RecordingContext(log)
    client = ScriptedSparkClient(log, _three_runs([(make_step(),)] * 3))
    bundle = extract_steps(ctx, "vid-1", client=client)
    assert bundle.timestamps_approximate is True
    assert bundle.source_runs == 3
    html = render_time_controls(bundle.steps[0], preview_frame_url="frame")
    assert "approximate" in html  # every card says the time is approximate


def test_flagged_controls_and_every_card_set_start():
    flagged = Step(
        number=2, start_s=95.0, end_s=150.0, action="Lift the motor",
        tools=(), parts=(), hazards=(), time_spread_s=8.0, time_needs_check=True,
    )
    plain = Step(
        number=1, start_s=12.0, end_s=34.0, action="Remove screws",
        tools=(), parts=(), hazards=(), time_spread_s=1.5, time_needs_check=False,
    )
    flagged_html = render_time_controls(flagged, preview_frame_url="preview.png")
    assert "Check time" in flagged_html
    assert "preview.png" in flagged_html
    assert 'data-nudge="-5"' in flagged_html and 'data-nudge="5"' in flagged_html
    assert "Set start here" in flagged_html
    plain_html = render_time_controls(plain, preview_frame_url="preview.png")
    assert "Check time" not in plain_html
    assert "data-nudge" not in plain_html
    assert "Set start here" in plain_html  # every card gets the control


def test_preview_route_renders_both_cards_and_replay_banner():
    app = FastAPI()
    app.include_router(router)
    page = TestClient(app).get("/steps/preview")
    assert page.status_code == 200
    assert "Check time" in page.text
    assert page.text.count("data-set-start") == 2  # one button per card
    assert SAMPLE_REPLAY_LABEL not in page.text
    replay = TestClient(app).get("/steps/preview", params={"replay": True})
    assert replay.status_code == 200
    assert SAMPLE_REPLAY_LABEL in replay.text  # permanent cached-sample label


def test_no_chunk_second_pass():
    log: list = []
    ctx = RecordingContext(log)
    # Steps sit 15 s apart: same-key windows within 5 s are one step by design.
    steps = tuple(make_step(start=float(i * 20), end=float(i * 20 + 5)) for i in range(4))
    # Sparse gapped runs: the legacy wording path (time consensus
    # excludes non-partitions); the pin is the call count, not time.
    client = _legacy_client(log, _three_runs([steps] * 3))
    bundle = extract_steps(ctx, "vid-1", client=client)
    assert len(bundle.steps) == 4
    assert len(client.calls) == 3  # exactly three 1x runs, no per-chunk pass
    assert len([e for e in log if e[0] == "reserve"]) == 3


def test_non_teardown_authorizes_zero_sam_calls():
    cases = [
        ({"device": "unknown", "confidence": 0.9}, "unknown_device"),
        ({"device": "Vacuum", "confidence": 0.2}, "low_confidence"),
    ]
    for kwargs, reason in cases:
        log: list = []
        ctx = RecordingContext(log)
        payload = make_payload(device=kwargs["device"], confidence=kwargs["confidence"])
        client = ScriptedSparkClient(log, [_result(payload) for _ in range(3)])
        with pytest.raises(NotATeardown) as exc:
            extract_steps(ctx, "vid-1", client=client)
        refusal = exc.value.refusal
        assert refusal.reason == reason
        assert refusal_payload(refusal) == {
            "verdict": "not_a_teardown",
            "reason": reason,
            "sample_cta": "Try the sample guide",
            "sam_calls_authorized": 0,
        }
        assert sam_calls_authorized(refusal) == 0


def test_divergent_actions_fail_closed_not_consensus():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(start=10.0, end=20.0, action="Remove the rear screws", parts=("battery",)),),
            (make_step(start=10.0, end=20.0, action="Lift out the motor", parts=("fan",)),),
            (make_step(start=10.0, end=20.0, action="Detach the ribbon cable", parts=("motor",)),),
        ]
    )
    with pytest.raises(StepsDisagreement):
        extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))


def test_divergent_parts_fail_closed_not_consensus():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(parts=("battery",)),),
            (make_step(parts=("fan",)),),
            (make_step(parts=("motor",)),),
        ]
    )
    with pytest.raises(StepsDisagreement):
        extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))


def test_unmatched_time_groups_fail_closed_not_consensus():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(start=0.0, end=5.0), make_step(start=10.0, end=15.0)),
            (make_step(start=100.0, end=105.0), make_step(start=110.0, end=115.0)),
            (make_step(start=200.0, end=205.0), make_step(start=210.0, end=215.0)),
        ]
    )
    with pytest.raises(StepsDisagreement):
        extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))


def test_equivalence_ignores_ordinal():
    log: list = []
    ctx = RecordingContext(log)
    step_a = make_step(start=10.0, end=20.0, action="Remove the rear screws", parts=("battery",))
    step_b = make_step(start=30.0, end=40.0, action="Lift out the motor", parts=("fan",))
    same = make_payload(steps=(dict(step_a), dict(step_b)))  # #1 A, #2 B
    swapped = make_payload(steps=(dict(step_b), dict(step_a)))  # #1 B, #2 A
    runs = [_result(same), _result(swapped), _result(same)]
    bundle = extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))
    assert [s.action for s in bundle.steps] == ["Remove the rear screws", "Lift out the motor"]
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [(10.0, 20.0), (30.0, 40.0)]
    assert [s.time_spread_s for s in bundle.steps] == [0.0, 0.0]
    assert [s.time_needs_check for s in bundle.steps] == [False, False]


def test_two_run_majority_wins_over_dissent():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(action="Remove the rear screws"),),
            (make_step(action="Remove the rear screws"),),
            (make_step(action="Lift out the motor"),),
        ]
    )
    bundle = extract_steps(ctx, "vid-1", client=ScriptedSparkClient(log, runs))
    assert [s.action for s in bundle.steps] == ["Remove the rear screws"]
    assert bundle.steps[0].time_spread_s == 0.0


def test_chained_overlap_semantic_majority_wins():
    log: list = []
    ctx = RecordingContext(log)
    dissent = make_step(
        start=0.0, end=10.0, action="Lift out the motor", parts=("fan",),
    )
    runs = _three_runs(
        [
            (dissent,),
            (make_step(start=8.0, end=18.0),),
            (make_step(start=16.0, end=26.0),),
        ]
    )
    bundle = extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))
    assert [s.action for s in bundle.steps] == ["Remove the rear screws"]
    step = bundle.steps[0]
    assert (step.start_s, step.end_s) == (12.0, 22.0)  # median of the B pair
    assert [p.phrase for p in step.parts] == ["battery"]
    assert step.time_spread_s == 8.0
    assert step.time_needs_check is True


def test_chained_overlap_true_disagreement_fails_closed():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(start=0.0, end=10.0, action="Remove the rear screws", parts=("battery",)),),
            (make_step(start=8.0, end=18.0, action="Lift out the motor", parts=("fan",)),),
            (make_step(start=16.0, end=26.0, action="Detach the ribbon cable", parts=("motor",)),),
        ]
    )
    with pytest.raises(StepsDisagreement):
        extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))


def test_adjacent_semantic_steps_stay_split():
    log: list = []
    ctx = RecordingContext(log)
    windows = [((0.0, 10.0), (10.0, 20.0)), ((0.0, 11.0), (11.0, 20.0)), ((0.0, 9.0), (9.0, 20.0))]
    runs = _three_runs(
        [
            (
                make_step(start=a[0], end=a[1], action="Remove the rear screws", parts=("battery",)),
                make_step(start=b[0], end=b[1], action="Lift out the motor", parts=("fan",)),
            )
            for a, b in windows
        ]
    )
    bundle = extract_steps(ctx, "vid-1", client=ScriptedSparkClient(log, runs))
    assert [s.action for s in bundle.steps] == ["Remove the rear screws", "Lift out the motor"]
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [(0.0, 10.0), (10.0, 20.0)]
    assert [s.time_needs_check for s in bundle.steps] == [False, False]


def test_touching_chain_within_tolerance_is_one_step():
    log: list = []
    ctx = RecordingContext(log)
    runs = _three_runs(
        [
            (make_step(start=0.0, end=5.0),),
            (make_step(start=5.0, end=10.0),),
            (make_step(start=10.0, end=15.0),),
        ]
    )
    bundle = extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))
    (step,) = bundle.steps
    assert step.action == "Remove the rear screws"
    assert (step.start_s, step.end_s) == (5.0, 10.0)  # median of the chain
    assert step.time_spread_s == 10.0
    assert step.time_needs_check is True


def test_overlapping_rivals_stay_kept_and_flagged():
    log: list = []
    ctx = RecordingContext(log)
    pair = (
        make_step(start=0.0, end=10.0, action="Remove the rear screws", parts=("battery",)),
        make_step(start=2.0, end=8.0, action="Lift out the motor", parts=("fan",)),
    )
    runs = _three_runs([pair, pair, (dict(pair[0]),)])
    bundle = extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))
    assert [s.action for s in bundle.steps] == ["Remove the rear screws", "Lift out the motor"]
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [(0.0, 10.0), (2.0, 8.0)]
    assert [s.time_needs_check for s in bundle.steps] == [True, True]  # kept, never dropped


def test_more_than_twelve_candidates_fail_closed_not_truncated():
    log: list = []
    ctx = RecordingContext(log)
    # Thirteen distinct two-run candidates across three valid runs of
    # at most twelve steps: keys 0-4 live in runs 0+1, keys 5-8 in
    # runs 0+2, keys 9-12 in runs 1+2.
    # verb-carrying placeholders, since
    # verb-less wordings never link -- without them this raises before
    # the cap under test is even reached.
    def keyed(key):
        return make_step(
            start=10.0, end=20.0, action=f"Remove part{key}", parts=(f"part{key}",),
        )

    membership = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        (0, 1, 2, 3, 4, 9, 10, 11, 12),
        (5, 6, 7, 8, 9, 10, 11, 12),
    )
    runs = _three_runs([tuple(keyed(k) for k in keys) for keys in membership])
    with pytest.raises(StepsDisagreement):
        extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))


def test_exactly_eight_candidates_still_return_eight_steps():
    log: list = []
    ctx = RecordingContext(log)
    # verb-carrying placeholders, since
    # verb-less wordings never link under the strict signature; the cap
    # under test is unchanged (8 kept, the 13-step sibling still refuses).
    steps = tuple(
        make_step(
            start=10.0, end=20.0, action=f"Remove part{key}", parts=(f"part{key}",),
        )
        for key in range(8)
    )
    payload = make_payload(steps=steps)
    empty = make_payload(steps=(), confidence=0.9)
    runs = [_result(payload), _result(payload), _result(empty)]
    bundle = extract_steps(ctx, "vid-1", client=_legacy_client(log, runs))
    assert len(bundle.steps) == 8  # the cap fails closed only past 12
    assert [s.number for s in bundle.steps] == list(range(1, 9))


def test_step_time_helpers():
    assert median_time([10.0, 12.0, 11.0]) == 11.0
    assert median_time([10.0, 12.0]) == 11.0  # two samples average
    with pytest.raises(ValueError):
        median_time([])
    assert time_spread([10.0, 12.0], [20.0, 30.0]) == 10.0  # wider of start/end range
    assert needs_time_check(5.0) is False  # strictly exceeds the 5 s line
    assert needs_time_check(5.01) is True
    assert format_approx(83.5) == "~1:24"
    assert format_approx_range(80.0, 95.0) == "~1:20–1:35"
