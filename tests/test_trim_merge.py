"""fuse same-action neighbours, drop non-repair edges.

Rule (frames 3:03-3:34 of the sample video show a RAM/SSD teaser):
the guide leaves out segments that are not part of the repair. The ground truth
is now 0-10 ... 170.4-189.2 with 189.2-214.17 excluded; the live-v2
consensus pair 170.43-179.1 + 179.1-189.23 ("Power on and verify in OS")
fuses into one phase.

Replay fixtures (tests/fixtures/trim_merge.json) copy the live-v2
raw steps only (no footage). The saved raws carry no kind tags, so the
replay validates them through the legacy path, tests the merge alone
on them, and tests the drop with kinds assigned in-test.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app import db, stage_cache
from app.admission import admit_job
from app.boundaries import (
    drop_edge_outro_phases,
    merge_same_action_phases,
    refine_consensus_bundle,
)
from app.chunks import ChunkWindow, StitchedSteps, derive_contiguous_ends, stitch_bundles
from app.contracts import (
    BoundaryInfo,
    ExcludedSpan,
    Guide,
    PartRequest,
    Step,
    StepBundle,
)
from app.scorer_time import score_guide
from app.ledger import LifecycleRepositoryImpl
from app.guides import assemble_guide
from app.pipeline import (
    PipelineClients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    guide_from_json,
    guide_to_json,
    load_guide,
    run_stages,
)
from app.providers.meta_spark import SPARK_PROMPT, SPARK_SCHEMA
from app.service import write_sample_clip
from app.settings import ProviderSettings, Settings, load_settings
from app.spark import (
    PHASE_KINDS,
    ChunkExtent,
    SparkRunResult,
    SparkSchemaError,
    SparkStageConfig,
    bundle_from_json,
    bundle_to_json,
    chunk_bundle_digest,
    consensus_bundle,
    extract_steps,
    validate_legacy_spark_payload,
    validate_spark_payload,
)
from app.uploads_store import UploadStoreImpl
from test_spark_schema import (
    RecordingContext,
    ScriptedSparkClient,
    make_payload,
    make_step,
)

FIXTURE = Path(__file__).parent / "fixtures" / "trim_merge.json"
SAMPLE_PROPOSED = Path(__file__).parent / "fixtures" / "sample_steps_proposed.json"
SAMPLE_V2 = Path(__file__).parent / "fixtures" / "extract_v2.json"

LIVE = SparkStageConfig(spread_s=2.0)
EXTENT214 = ChunkExtent(duration_s=214.17, tolerance_s=2.0)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _record(time_s: float) -> BoundaryInfo:
    return BoundaryInfo(
        time_s=time_s,
        support_runs=3,
        total_runs=3,
        nearest_cut_s=None,
        cut_distance_s=None,
        snapped=False,
        snap_delta_s=0.0,
        needs_check=False,
    )


def _step(number: int, start: float, end: float, action: str, **over) -> Step:
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


def _bundle(steps: list[Step]) -> StepBundle:
    return StepBundle(
        device_guess="laptop",
        confidence=0.9,
        steps=tuple(steps),
        source_runs=3,
        timestamps_approximate=True,
        boundaries=tuple(_record(step.end_s) for step in steps[:-1]),
    )


def _scorer_gt() -> list[dict]:
    # Reference times for the sample video (extract_v2) with the proposed actions
    # (wording is report-only): the same GT test_extract_v2 scores.
    # The proposed fixture's own step times keep a stale 118 split the
    # scorer never reads; only its actions and excluded span are live.
    truth = json.loads(SAMPLE_V2.read_text(encoding="utf-8"))["sample_truth"]
    proposed = json.loads(SAMPLE_PROPOSED.read_text(encoding="utf-8"))["steps"]
    return [
        {
            "number": number,
            "start_s": start,
            "end_s": end,
            "action": step["action"],
        }
        for number, ((start, end), step) in enumerate(
            zip(truth, proposed), start=1
        )
    ]


def _excluded() -> list[tuple[float, float]]:
    data = json.loads(SAMPLE_V2.read_text(encoding="utf-8"))
    return [tuple(span) for span in data["sample_excluded"]]


def _scorer_rows(steps) -> list[dict]:
    return [
        {
            "number": step.number,
            "start_s": step.start_s,
            "end_s": step.end_s,
            "action": step.action,
        }
        for step in steps
    ]


# --- Ground-truth fixtures end at 189.2 with the teaser excluded -----------------


def test_sample_truth_ends_at_189_2_with_excluded_tail() -> None:
    data = json.loads(SAMPLE_PROPOSED.read_text(encoding="utf-8"))
    steps = data["steps"]
    assert [s["number"] for s in steps] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert steps[0]["start_s"] == 0
    assert steps[8] == {
        "number": 9,
        "start_s": 170.4,
        "end_s": 189.2,
        "action": steps[8]["action"],
    }
    assert steps[8]["action"]
    (span,) = data["excluded_spans"]
    assert span["start_s"] == 189.2
    assert span["end_s"] == pytest.approx(214.167)
    assert span["kind"] == "outro_or_unrelated"
    # Contiguous: truth still tiles [0, 189.2), excluded covers the rest.
    for left, right in zip(steps, steps[1:]):
        assert right["start_s"] == left["end_s"]
    assert steps[-1]["end_s"] == span["start_s"]
    assert "189.2" in data["boundary_note"]


def test_extract_v2_truth_matches_proposed_and_excludes_tail() -> None:
    data = json.loads(SAMPLE_V2.read_text(encoding="utf-8"))
    truth = data["sample_truth"]
    assert truth[-1] == [170.4, 189.2]
    assert len(truth) == 9
    proposed = json.loads(SAMPLE_PROPOSED.read_text(encoding="utf-8"))
    # Both fixtures end step 9 at 189.2 with the same excluded tail
    # (the proposed mid-steps keep a stale 118 split the scorer never
    # reads; only the tail changed here).
    assert proposed["steps"][-1]["end_s"] == 189.2
    assert proposed["steps"][-1]["start_s"] == truth[-1][0]
    assert data["sample_excluded"] == [[189.2, 214.17]]
    (span,) = proposed["excluded_spans"]
    assert span["start_s"] == data["sample_excluded"][0][0]
    assert span["end_s"] == pytest.approx(data["sample_excluded"][0][1], abs=0.01)


# --- prompt and schema carry the closed kind set -----------------------------


def test_kind_closed_set_is_intro_repair_verify_outro() -> None:
    assert PHASE_KINDS == ("intro", "repair", "verify", "outro_or_unrelated")


def test_prompt_asks_for_a_kind_per_phase() -> None:
    assert "kind" in SPARK_PROMPT
    for kind in PHASE_KINDS:
        assert kind in SPARK_PROMPT


def test_schema_requires_kind_per_phase() -> None:
    items = SPARK_SCHEMA["properties"]["steps"]["items"]
    assert items["properties"]["kind"] == {
        "type": "string",
        "enum": list(PHASE_KINDS),
    }
    assert "kind" in items["required"]


# --- validation: kind required; legacy path defaults missing to repair -------


def test_validate_accepts_each_kind_and_rejects_a_missing_kind() -> None:
    for kind in PHASE_KINDS:
        run = validate_spark_payload(
            make_payload(steps=(make_step(kind=kind),))
        )
        assert run.steps[0].kind == kind
    with pytest.raises(SparkSchemaError):
        validate_spark_payload(make_payload(steps=(make_step(kind=None),)))


def test_legacy_validate_defaults_missing_kind_to_repair() -> None:
    run = validate_legacy_spark_payload(
        make_payload(steps=(make_step(kind=None),))
    )
    assert run.steps[0].kind == "repair"
    kinded = validate_legacy_spark_payload(
        make_payload(steps=(make_step(kind="intro"),))
    )
    assert kinded.steps[0].kind == "intro"
    with pytest.raises(SparkSchemaError):
        validate_legacy_spark_payload(
            make_payload(steps=(make_step(kind="recap"),))
        )


@pytest.mark.parametrize("bad", ["recap", "REPAIR", "", 3, None, ["repair"]])
def test_validate_rejects_a_kind_outside_the_closed_set(bad) -> None:
    step = make_step()
    step["kind"] = bad
    with pytest.raises(SparkSchemaError):
        validate_spark_payload(make_payload(steps=(step,)))


def test_validate_still_rejects_unknown_step_keys() -> None:
    step = make_step()
    step["chapter"] = "one"
    with pytest.raises(SparkSchemaError):
        validate_spark_payload(make_payload(steps=(step,)))


def test_live_extract_rejects_a_kindless_provider_response() -> None:
    log: list = []
    kindless = make_payload(steps=(make_step(kind=None),))
    results = [
        SparkRunResult(
            payload=dict(kindless), cost_usd=Decimal("0.08"),
            latency_ms=1200,
        )
        for _ in range(3)
    ]
    client = ScriptedSparkClient(log, results)
    with pytest.raises(SparkSchemaError):
        extract_steps(RecordingContext(log), "vid-1", client=client)


# --- consensus votes the segment kind by run majority ------------------------


def _kinded_run(kinds: tuple[str, ...], bounds: tuple[float, ...]):
    steps = tuple(
        {**make_step(number=n, start=s, end=e), "kind": kind}
        for n, (s, e, kind) in enumerate(
            zip(bounds, bounds[1:], kinds), start=1
        )
    )
    return validate_spark_payload(make_payload(steps=steps))


def test_time_consensus_kind_is_the_run_majority() -> None:
    bounds = (0.0, 10.0, 20.0, 30.0)
    runs = [
        _kinded_run(("intro", "repair", "outro_or_unrelated"), bounds),
        _kinded_run(("intro", "verify", "outro_or_unrelated"), bounds),
        _kinded_run(("repair", "repair", "repair"), bounds),
    ]
    bundle = consensus_bundle(
        runs, LIVE, chunk_extent=ChunkExtent(duration_s=30.0, tolerance_s=2.0)
    )
    assert [s.kind for s in bundle.steps] == ["intro", "repair", "outro_or_unrelated"]


def test_time_consensus_outro_needs_a_majority_not_a_plurality() -> None:
    bounds = (0.0, 10.0, 20.0)
    runs = [
        _kinded_run(("outro_or_unrelated", "repair"), bounds),
        _kinded_run(("intro", "repair"), bounds),
        _kinded_run(("repair", "repair"), bounds),
    ]
    bundle = consensus_bundle(
        runs, LIVE, chunk_extent=ChunkExtent(duration_s=20.0, tolerance_s=2.0)
    )
    # 1-1-1 on the first segment: no majority, and the label winner
    # is the lone outro vote, so the fail-keep fallback stands.
    assert bundle.steps[0].kind == "repair"
    assert bundle.steps[1].kind == "repair"


def test_exact_consensus_ignores_kind_for_grouping() -> None:
    config = SparkStageConfig(alignment_mode="exact")
    runs = [
        validate_spark_payload(
            make_payload(
                steps=(
                    {**make_step(number=1, start=0.0, end=10.0), "kind": kind},
                )
            )
        )
        for kind in ("intro", "repair", "verify")
    ]
    bundle = consensus_bundle(runs, config)
    assert len(bundle.steps) == 1
    # The exact key ignores kind (same grouping as kindless runs); the
    # kept step honestly carries its median-start sample's kind.
    assert bundle.steps[0].kind == "intro"


# --- merge: same action after case/whitespace/punctuation --------------------


def test_merge_fuses_adjacent_duplicates_and_keeps_survivors() -> None:
    bundle = _bundle(
        [
            _step(1, 0.0, 170.43, "Flip over and remove external battery"),
            _step(2, 170.43, 179.1, "Power on and verify in OS"),
            _step(3, 179.1, 189.23, "Power on and verify in OS"),
            _step(4, 189.23, 214.17, "Show memory upgrade parts"),
        ]
    )
    merged = merge_same_action_phases(bundle)
    assert [(s.start_s, s.end_s) for s in merged.steps] == [
        (0.0, 170.43),
        (170.43, 189.23),
        (189.23, 214.17),
    ]
    assert [s.number for s in merged.steps] == [1, 2, 3]
    assert merged.steps[1].action == "Power on and verify in OS"
    assert [b.time_s for b in merged.boundaries] == [170.43, 189.23]
    # The fused edge's record is gone; survivors keep their records.
    assert len(merged.boundaries) == len(merged.steps) - 1


def test_merge_normalises_case_whitespace_and_punctuation() -> None:
    bundle = _bundle(
        [
            _step(1, 0.0, 10.0, "Power on, and verify  in OS!"),
            _step(2, 10.0, 20.0, "  power ON and verify in os "),
            _step(3, 20.0, 30.0, "power-on and verify in OS?"),
            _step(4, 30.0, 40.0, "Show memory upgrade parts"),
        ]
    )
    merged = merge_same_action_phases(bundle)
    assert [(s.start_s, s.end_s) for s in merged.steps] == [
        (0.0, 30.0),
        (30.0, 40.0),
    ]
    assert merged.steps[0].action == "Power on, and verify  in OS!"


def test_merge_leaves_distinct_neighbours_and_singletons() -> None:
    bundle = _bundle(
        [
            _step(1, 0.0, 10.0, "Remove the rear screws"),
            _step(2, 10.0, 20.0, "Remove the front screws"),
            _step(3, 20.0, 30.0, "Remove the rear screws"),
        ]
    )
    merged = merge_same_action_phases(bundle)
    assert [(s.start_s, s.end_s) for s in merged.steps] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]
    solo = _bundle([_step(1, 0.0, 10.0, "Intro")])
    assert merge_same_action_phases(solo).steps == solo.steps


def test_merge_does_not_fuse_negations_or_different_objects() -> None:
    bundle = _bundle(
        [
            _step(1, 0.0, 10.0, "Do not remove the battery"),
            _step(2, 10.0, 20.0, "Remove the battery!"),
            _step(3, 20.0, 30.0, "Remove the fan"),
        ]
    )
    merged = merge_same_action_phases(bundle)
    assert [(s.start_s, s.end_s) for s in merged.steps] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]


def test_merge_carries_max_spread_or_flags_and_honest_kind() -> None:
    bundle = _bundle(
        [
            _step(
                1, 0.0, 10.0, "Power on",
                time_spread_s=1.0, time_needs_check=True, kind="verify",
            ),
            _step(
                2, 10.0, 20.0, "power on!",
                time_spread_s=4.0, time_needs_check=False, kind="verify",
            ),
        ]
    )
    (fused,) = merge_same_action_phases(bundle).steps
    assert fused.time_spread_s == 4.0
    assert fused.time_needs_check is True
    assert fused.kind == "verify"
    mixed = _bundle(
        [
            _step(1, 0.0, 10.0, "Power on", kind="verify"),
            _step(2, 10.0, 20.0, "power on", kind="outro_or_unrelated"),
        ]
    )
    (kept,) = merge_same_action_phases(mixed).steps
    # Mixed kinds never fuse into a droppable phase.
    assert kept.kind == "repair"


def test_merge_unions_tools_parts_and_hazards_in_order() -> None:
    bundle = _bundle(
        [
            _step(
                1, 0.0, 10.0, "Disconnect the battery cable",
                tools=("spudger",),
                parts=(PartRequest("battery", "bay"),),
                hazards=(),
            ),
            _step(
                2, 10.0, 20.0, "disconnect the battery cable!",
                tools=("spudger", "tweezers"),
                parts=(
                    PartRequest("battery", "bay"),
                    PartRequest("cable", "right"),
                ),
                hazards=("battery",),
            ),
        ]
    )
    (fused,) = merge_same_action_phases(bundle).steps
    # The hazard lives only on the second phase: the fused step keeps
    # it, and tools/parts union with duplicates dropped, first first.
    assert fused.hazards == ("battery",)
    assert fused.tools == ("spudger", "tweezers")
    assert fused.parts == (
        PartRequest("battery", "bay"),
        PartRequest("cable", "right"),
    )


def test_merge_keeps_phases_when_parts_union_exceeds_limit() -> None:
    bundle = _bundle(
        [
            _step(
                1, 0.0, 10.0, "Remove the screws",
                parts=tuple(
                    PartRequest(f"screw-{n}", "edge") for n in (1, 2, 3)
                ),
            ),
            _step(
                2, 10.0, 20.0, "remove the screws",
                parts=tuple(
                    PartRequest(f"screw-{n}", "edge") for n in (4, 5)
                ),
            ),
        ]
    )
    merged = merge_same_action_phases(bundle)
    assert merged.steps == bundle.steps
    assert merged.boundaries == bundle.boundaries


def test_merge_ignores_an_empty_normalised_key() -> None:
    bundle = _bundle(
        [
            _step(1, 0.0, 10.0, "..."),
            _step(2, 10.0, 20.0, "!!!"),
        ]
    )
    merged = merge_same_action_phases(bundle)
    assert [(s.start_s, s.end_s) for s in merged.steps] == [
        (0.0, 10.0),
        (10.0, 20.0),
    ]


def test_refine_merges_by_default_and_skips_when_off() -> None:
    bundle = _bundle(
        [
            _step(1, 0.0, 10.0, "Power on and verify in OS"),
            _step(2, 10.0, 20.0, "power on and verify in os"),
        ]
    )
    assert len(
        refine_consensus_bundle(
            bundle, cuts=(), snap_window_s=0.0, min_phase_length_s=5.0
        ).steps
    ) == 1
    assert len(
        refine_consensus_bundle(
            bundle, cuts=(), snap_window_s=0.0, min_phase_length_s=5.0,
            merge_same_action=False,
        ).steps
    ) == 2


def test_refine_rejects_a_non_bool_merge_switch() -> None:
    bundle = _bundle([_step(1, 0.0, 10.0, "Intro")])
    with pytest.raises(ValueError):
        refine_consensus_bundle(
            bundle, cuts=(), snap_window_s=0.0, min_phase_length_s=5.0,
            merge_same_action="yes",  # type: ignore[arg-type]
        )


# --- drop: only leading/trailing outro, middle kept, intro kept --------------


def test_drop_removes_edge_outro_and_records_spans() -> None:
    steps = (
        _step(1, 0.0, 8.0, "Teaser montage", kind="outro_or_unrelated"),
        _step(2, 8.0, 10.0, "Intro", kind="intro"),
        _step(3, 10.0, 180.0, "Repair work", kind="repair"),
        _step(4, 180.0, 189.2, "Power on and verify", kind="verify"),
        _step(5, 189.2, 214.17, "Show memory upgrade parts", kind="outro_or_unrelated"),
    )
    kept, excluded = drop_edge_outro_phases(steps)
    assert [(s.start_s, s.end_s) for s in kept] == [
        (8.0, 10.0),
        (10.0, 180.0),
        (180.0, 189.2),
    ]
    assert [s.number for s in kept] == [1, 2, 3]
    assert excluded == (
        ExcludedSpan(start_s=0.0, end_s=8.0, kind="outro_or_unrelated"),
        ExcludedSpan(start_s=189.2, end_s=214.17, kind="outro_or_unrelated"),
    )


def test_drop_keeps_middle_outro_and_never_empties() -> None:
    steps = (
        _step(1, 0.0, 10.0, "Intro", kind="intro"),
        _step(2, 10.0, 20.0, "Sponsor message", kind="outro_or_unrelated"),
        _step(3, 20.0, 30.0, "Repair work", kind="repair"),
    )
    kept, excluded = drop_edge_outro_phases(steps)
    assert [s.number for s in kept] == [1, 2, 3]
    assert excluded == ()
    all_outro = (
        _step(1, 0.0, 10.0, "Teaser", kind="outro_or_unrelated"),
        _step(2, 10.0, 20.0, "Showcase", kind="outro_or_unrelated"),
    )
    kept, excluded = drop_edge_outro_phases(all_outro)
    assert [s.number for s in kept] == [1, 2]
    assert excluded == ()
    (solo_kept, solo_excluded) = drop_edge_outro_phases(
        (_step(1, 0.0, 10.0, "Teaser", kind="outro_or_unrelated"),)
    )
    assert [s.number for s in solo_kept] == [1]
    assert solo_excluded == ()


def test_drop_keeps_a_leading_intro_and_rejects_empty() -> None:
    steps = (
        _step(1, 0.0, 10.0, "Intro", kind="intro"),
        _step(2, 10.0, 20.0, "Repair work", kind="repair"),
    )
    kept, excluded = drop_edge_outro_phases(steps)
    assert [s.number for s in kept] == [1, 2]
    assert excluded == ()
    with pytest.raises(ValueError):
        drop_edge_outro_phases(())


# --- scorer ignores guide rows inside an excluded truth span -----------------


def test_scorer_ignores_guide_rows_inside_excluded_spans() -> None:
    gt = _scorer_gt()
    guide = list(gt) + [
        {
            "number": 10,
            "start_s": 189.23,
            "end_s": 214.17,
            "action": "Show memory upgrade parts",
        }
    ]
    excluded = _excluded()
    report = score_guide(guide, gt, excluded_spans=excluded)
    assert report.strict_passes == 9
    assert report.extra_guide == ()
    assert report.passed is True
    without = score_guide(guide, gt)
    assert without.extra_guide == (10,)
    assert without.strict_passes == 9


def test_scorer_keeps_guide_rows_only_partly_in_excluded_spans() -> None:
    gt = _scorer_gt()
    guide = _scorer_rows(
        [_step(9, 170.4, 214.17, "Power on and verify plus showcase")]
    )
    report = score_guide(guide, gt, excluded_spans=_excluded())
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(9, 9)]
    assert report.matches[0].passed is False


def test_scorer_rejects_misshaped_excluded_spans() -> None:
    gt = _scorer_gt()
    guide = _scorer_rows([_step(1, 0.0, 10.0, "Intro")])
    for bad in ([(10.0, 0.0)], [(0.0, 0.0)], [(-1.0, 5.0)], ["nope"], [(1.0,)]):
        with pytest.raises(ValueError):
            score_guide(guide, gt, excluded_spans=bad)


def test_scorer_coalesces_touching_excluded_spans() -> None:
    gt = _scorer_gt()
    row = {
        "number": 10,
        "start_s": 10.0,
        "end_s": 30.0,
        "action": "Showcase montage",
    }
    # Two trailing outros record adjacent spans (10,20) and (20,30);
    # a guide row spanning the shared edge is still excluded.
    report = score_guide(
        [row], gt, excluded_spans=[(10.0, 20.0), (20.0, 30.0)]
    )
    assert report.extra_guide == ()
    report = score_guide(
        [row], gt, excluded_spans=[(10.0, 25.0), (20.0, 30.0)]
    )
    assert report.extra_guide == ()


# --- settings switches default on --------------------------------------------


def test_merge_and_exclude_switches_default_on_and_parse() -> None:
    settings = load_settings({})
    assert settings.merge_same_action is True
    assert settings.exclude_outro_phases is True
    settings = load_settings(
        {
            "PARTFIDELITY_MERGE_SAME_ACTION": "no",
            "PARTFIDELITY_EXCLUDE_OUTRO_PHASES": "0",
        }
    )
    assert settings.merge_same_action is False
    assert settings.exclude_outro_phases is False
    with pytest.raises(RuntimeError):
        load_settings({"PARTFIDELITY_MERGE_SAME_ACTION": "maybe"})
    with pytest.raises(RuntimeError):
        load_settings({"PARTFIDELITY_EXCLUDE_OUTRO_PHASES": "maybe"})


def test_chunk_bundle_digest_binds_the_merge_switch() -> None:
    base = {
        "parent_job_id": "p",
        "source_hash": "aaa",
        "index": 0,
        "window_start_s": 0.0,
        "window_end_s": 60.0,
        "config": LIVE,
        "chunk_tolerance_s": 2.0,
        "cut_start_s": 0.0,
        "cut_duration_s": 60.0,
        "scene_cut_threshold": 0.15,
        "scene_cuts_in_prompt": False,
        "snap_window_s": 2.0,
        "min_phase_length_s": 5.0,
        "clock_burn_in": True,
        "merge_same_action": True,
    }
    first = chunk_bundle_digest(**base)
    assert chunk_bundle_digest(**base) == first
    assert chunk_bundle_digest(**{**base, "merge_same_action": False}) != first
    with pytest.raises(ValueError):
        chunk_bundle_digest(**{**base, "merge_same_action": "yes"})


# --- bundle JSON round-trips kinds and excluded spans ------------------------


def test_bundle_json_round_trips_kinds_and_excluded_spans() -> None:
    bundle = StepBundle(
        device_guess="laptop",
        confidence=0.9,
        steps=(
            _step(1, 0.0, 10.0, "Intro", kind="intro"),
            _step(2, 10.0, 20.0, "Repair work", kind="repair"),
        ),
        source_runs=3,
        timestamps_approximate=True,
        boundaries=(_record(10.0),),
        excluded_spans=(
            ExcludedSpan(start_s=20.0, end_s=30.0, kind="outro_or_unrelated"),
        ),
    )
    assert bundle_from_json(bundle_to_json(bundle), LIVE) == bundle


def test_bundle_json_rejects_a_kind_outside_the_closed_set() -> None:
    bundle = _bundle([_step(1, 0.0, 10.0, "Intro", kind="intro")])
    raw = bundle_to_json(bundle)
    assert isinstance(raw, dict)
    steps = raw["steps"]
    assert isinstance(steps, list)
    first = steps[0]
    assert isinstance(first, dict)
    first["kind"] = "recap"
    with pytest.raises((ValueError, SparkSchemaError)):
        bundle_from_json(raw, LIVE)


# --- guide assembly and guide.json carry kinds and excluded spans -----------


class _AssembleCtx:
    """Minimal assembly context: an id in, one finish out."""

    def __init__(self) -> None:
        self.job_id = "job-trim"
        self.finishes: list = []

    def finish(self, outcome, *, keep_artifact_ids=(), failed_output_names=None):
        self.finishes.append((outcome, tuple(keep_artifact_ids)))


def _assembled_guide() -> Guide:
    return Guide(
        job_id="job-trim",  # type: ignore[arg-type]
        steps=(
            _step(1, 0.0, 10.0, "Intro", kind="intro"),
            _step(2, 10.0, 20.0, "Power on and verify", kind="verify"),
        ),
        frame_paths=(),
        labels=(),
        difficulty="Easy",
        verdict="DRAFT",
        draft_reasons=(),
        excluded_spans=(
            ExcludedSpan(
                start_s=20.0, end_s=30.0, kind="outro_or_unrelated"
            ),
        ),
    )


def test_assemble_guide_carries_excluded_spans() -> None:
    ctx = _AssembleCtx()
    guide = assemble_guide(
        ctx,  # type: ignore[arg-type]
        list(_assembled_guide().steps), [], [],
        excluded_spans=[
            ExcludedSpan(
                start_s=20.0, end_s=30.0, kind="outro_or_unrelated"
            )
        ],
    )
    assert guide.excluded_spans == (
        ExcludedSpan(start_s=20.0, end_s=30.0, kind="outro_or_unrelated"),
    )
    assert ctx.finishes == [("done", ())]
    assert assemble_guide(
        _AssembleCtx(),  # type: ignore[arg-type]
        list(_assembled_guide().steps), [], [],
    ).excluded_spans == ()


def test_guide_json_round_trips_kinds_and_excluded_spans() -> None:
    guide = _assembled_guide()
    assert guide_from_json(guide_to_json(guide)) == guide


def test_guide_json_validates_kind() -> None:
    raw = guide_to_json(_assembled_guide())
    assert isinstance(raw, dict)
    # Guide files stored before kinds load as repair.
    kindless = json.loads(json.dumps(raw))
    for step in kindless["steps"]:
        del step["kind"]
    compat = guide_from_json(kindless)
    assert [step.kind for step in compat.steps] == ["repair", "repair"]
    # A present kind must be exactly one of the closed set.
    bad_step = json.loads(json.dumps(raw))
    bad_step["steps"][0]["kind"] = "recap"
    with pytest.raises(ValueError):
        guide_from_json(bad_step)
    bad_span = json.loads(json.dumps(raw))
    bad_span["excluded_spans"][0]["kind"] = "recap"
    with pytest.raises(ValueError):
        guide_from_json(bad_span)


def test_load_pre_change_guide_and_reject_malformed_excluded_spans(tmp_path: Path) -> None:
    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "guide_pre_trim.json").read_text()
    )
    guide_file = tmp_path / "jobs" / "job-trim" / "guide.json"
    guide_file.parent.mkdir(parents=True)
    guide_file.write_text(json.dumps(raw))
    guide = load_guide("job-trim", data_root=tmp_path)
    assert guide is not None
    assert guide.excluded_spans == ()
    assert all(step.kind == "repair" for step in guide.steps)
    for malformed in (None, {}, [{"start_s": 1, "end_s": 2, "kind": "recap"}]):
        raw["excluded_spans"] = malformed
        guide_file.write_text(json.dumps(raw))
        assert load_guide("job-trim", data_root=tmp_path) is None


# --- stitch and derive preserve kinds ----------------------------------------


def test_stitch_carries_step_kinds() -> None:
    left = _bundle(
        [
            _step(1, 0.0, 10.0, "Intro", kind="intro"),
            _step(2, 10.0, 60.0, "Repair work", kind="repair"),
        ]
    )
    right = _bundle(
        [
            _step(1, 0.0, 30.0, "Repair work", kind="repair"),
            _step(2, 30.0, 54.17, "Show memory upgrade parts", kind="outro_or_unrelated"),
        ]
    )
    stitched = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0), left),
            (ChunkWindow(160.0, 214.17), right),
        ],
        overlap_s=2.0,
    )
    assert isinstance(stitched, StitchedSteps)
    assert [s.kind for s in stitched.steps] == [
        "intro", "repair", "repair", "outro_or_unrelated",
    ]
    derived = derive_contiguous_ends(stitched.steps, 214.17, tolerance_s=100.0)
    assert [s.kind for s in derived] == [s.kind for s in stitched.steps]


# --- $0 replay on the live-v2 raws --------------------------------------------

KINDS_R1 = {1: "intro", 11: "verify", 12: "outro_or_unrelated"}
KINDS_R2 = {1: "intro", 10: "verify", 11: "verify", 12: "outro_or_unrelated"}
KINDS_R3 = {1: "intro", 9: "repair", 10: "verify", 11: "outro_or_unrelated"}


def _kinded(payload: dict, kinds: dict[int, str]) -> dict:
    steps = [
        {**step, "kind": kinds.get(step["number"], "repair")}
        for step in payload["steps"]
    ]
    return {**payload, "steps": steps}


def test_replay_live_v2_merge_fuses_the_verify_pair_and_scores_8() -> None:
    fixture = _fixture()
    runs = [
        validate_legacy_spark_payload(fixture[name])
        for name in ("raw_R1", "raw_R2", "raw_R3")
    ]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT214)
    assert len(bundle.steps) == 12
    refined = refine_consensus_bundle(
        bundle,
        cuts=tuple(fixture["sample_cuts"]),
        snap_window_s=2.0,
        min_phase_length_s=5.0,
    )
    assert len(refined.steps) == 11
    fused = refined.steps[9]
    assert fused.action == "Power on and verify in OS"
    assert (fused.start_s, fused.end_s) == (
        pytest.approx(170.43, abs=0.05),
        pytest.approx(189.23, abs=0.05),
    )
    assert len(refined.boundaries) == 10
    report = score_guide(
        _scorer_rows(refined.steps), _scorer_gt(), excluded_spans=_excluded()
    )
    assert report.strict_passes == 8
    assert report.passed is True
    assert [m.gt_number for m in report.matches if not m.passed] == [3]


def test_replay_live_v2_single_runs_match_manager() -> None:
    fixture = _fixture()
    for name, expected in (("raw_R1", 9), ("raw_R2", 3), ("raw_R3", 5)):
        run = validate_legacy_spark_payload(fixture[name])
        report = score_guide(
            _scorer_rows(run.steps), _scorer_gt(), excluded_spans=_excluded()
        )
        assert report.strict_passes == expected, name


def test_replay_live_v2_kinded_drop_cuts_the_teaser() -> None:
    fixture = _fixture()
    kind_maps = {"raw_R1": KINDS_R1, "raw_R2": KINDS_R2, "raw_R3": KINDS_R3}
    runs = [
        validate_spark_payload(_kinded(fixture[name], kind_maps[name]))
        for name in ("raw_R1", "raw_R2", "raw_R3")
    ]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT214)
    assert [s.kind for s in bundle.steps] == [
        "intro", "repair", "repair", "repair", "repair", "repair",
        "repair", "repair", "repair", "verify", "verify", "outro_or_unrelated",
    ]
    refined = refine_consensus_bundle(
        bundle,
        cuts=tuple(fixture["sample_cuts"]),
        snap_window_s=2.0,
        min_phase_length_s=5.0,
    )
    assert len(refined.steps) == 11
    kept, excluded = drop_edge_outro_phases(refined.steps)
    assert len(kept) == 10
    assert kept[0].kind == "intro"  # the leading intro is never dropped
    assert kept[-1].end_s == pytest.approx(189.23, abs=0.05)
    (span,) = excluded
    assert span.start_s == pytest.approx(189.23, abs=0.05)
    assert span.end_s == pytest.approx(214.17, abs=0.01)
    assert span.kind == "outro_or_unrelated"
    report = score_guide(_scorer_rows(kept), _scorer_gt(), excluded_spans=_excluded())
    assert report.strict_passes == 8
    # Only the 31.2 BIOS-split fragment is extra (untouched by the trim
    # rule); the teaser row is gone with the drop, not listed as extra.
    assert report.extra_guide == (4,)


# --- pipeline: the guide drops the tail; switches off keep it ----------------


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


@pytest.fixture
def uploads(lane_db: Path) -> UploadStoreImpl:
    return UploadStoreImpl(lane_db)


@pytest.fixture
def lifecycle(lane_db: Path) -> LifecycleRepositoryImpl:
    return LifecycleRepositoryImpl(lane_db)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    root = tmp_path / "stage_cache"
    monkeypatch.setattr(stage_cache, "_CACHE_ROOT", root)
    return root


@pytest.fixture(scope="module")
def clip6(tmp_path_factory) -> Path:
    return write_sample_clip(
        tmp_path_factory.mktemp("media") / "six.mp4", seconds=6.0, fps=5
    )


def _kinded_payload() -> dict:
    steps = [
        {
            "number": 1, "start_s": 0.0, "end_s": 2.0,
            "action": "Introduce laptop on table", "tools": [],
            "parts": [
                {"phrase": "laptop lid", "position_hint": "center"}
            ],
            "hazards": [], "kind": "intro",
        },
        {
            "number": 2, "start_s": 2.0, "end_s": 4.0,
            "action": "Remove the rear screws", "tools": [],
            "parts": [{"phrase": "screw", "position_hint": "rear edge"}],
            "hazards": [], "kind": "repair",
        },
        {
            "number": 3, "start_s": 4.0, "end_s": 6.0,
            "action": "Show memory upgrade parts", "tools": [],
            "parts": [], "hazards": [], "kind": "outro_or_unrelated",
        },
    ]
    return {"device_guess": "laptop", "confidence": 0.9, "steps": steps}


def _kinded_clients() -> PipelineClients:
    runs = [
        SparkRunResult(
            payload=_kinded_payload(),
            cost_usd=Decimal("0.08"),
            latency_ms=900,
        )
        for _ in range(3)
    ]
    return PipelineClients(
        spark=ReplaySparkClient(runs),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )


def _dup_payload() -> dict:
    steps = [
        {
            "number": 1, "start_s": 0.0, "end_s": 2.0,
            "action": "Remove the rear screws", "tools": [],
            "parts": [{"phrase": "screw", "position_hint": "rear edge"}],
            "hazards": [], "kind": "repair",
        },
        {
            "number": 2, "start_s": 2.0, "end_s": 4.0,
            "action": "remove the rear screws!", "tools": [],
            "parts": [{"phrase": "screw", "position_hint": "rear edge"}],
            "hazards": [], "kind": "repair",
        },
        {
            "number": 3, "start_s": 4.0, "end_s": 6.0,
            "action": "Show memory upgrade parts", "tools": [],
            "parts": [], "hazards": [], "kind": "outro_or_unrelated",
        },
    ]
    return {"device_guess": "laptop", "confidence": 0.9, "steps": steps}


def _dup_clients() -> PipelineClients:
    runs = [
        SparkRunResult(
            payload=_dup_payload(),
            cost_usd=Decimal("0.08"),
            latency_ms=900,
        )
        for _ in range(3)
    ]
    return PipelineClients(
        spark=ReplaySparkClient(runs),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )


def test_pipeline_drops_the_outro_tail_from_the_guide(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
    tmp_cache: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False
    )
    assert settings.exclude_outro_phases is True
    ctx = admit_job("trim-drop", "trim-drop", "video-trim-drop")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=_kinded_clients(), lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 2.0),
        (2.0, 4.0),
    ]
    assert [s.number for s in result.guide.steps] == [1, 2]
    # End to end: the trimmed span rides the guide and the stored
    # guide.json round-trips it with the step kinds.
    assert result.guide.excluded_spans == (
        ExcludedSpan(
            start_s=4.0, end_s=6.0, kind="outro_or_unrelated"
        ),
    )
    assert [s.kind for s in result.guide.steps] == ["intro", "repair"]
    stored = load_guide(result.job_id, data_root=settings.data_root)
    assert stored == result.guide


def test_pipeline_keeps_the_tail_when_the_switch_is_off(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
    tmp_cache: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        exclude_outro_phases=False,
    )
    ctx = admit_job("trim-keep", "trim-keep", "video-trim-keep")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=_kinded_clients(), lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
    ]


def test_pipeline_direct_path_merges_same_action_duplicates(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
    tmp_cache: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False
    )
    assert settings.merge_same_action is True
    ctx = admit_job("trim-direct", "trim-direct", "video-trim-direct")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=_dup_clients(), lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    # The 6 s clip takes the direct path: the duplicate pair fuses
    # and the outro tail still trims.
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 4.0),
    ]
    assert result.guide.excluded_spans == (
        ExcludedSpan(start_s=4.0, end_s=6.0, kind="outro_or_unrelated"),
    )


def test_pipeline_direct_path_keeps_duplicates_when_merge_is_off(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
    tmp_cache: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        merge_same_action=False,
    )
    ctx = admit_job("trim-direct-off", "trim-direct-off", "video-trim-direct-off")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=_dup_clients(), lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 2.0),
        (2.0, 4.0),
    ]


def test_pipeline_legacy_span_keeps_every_step_byte_unchanged(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
    tmp_cache: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        step_span_mode="legacy",
    )
    ctx = admit_job("trim-legacy", "trim-legacy", "video-trim-legacy")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=_dup_clients(), lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
    ]
    assert result.guide.excluded_spans == ()


def test_pipeline_exact_alignment_keeps_duplicates(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
    tmp_cache: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        providers=ProviderSettings(step_alignment_mode="exact"),
    )
    ctx = admit_job("trim-exact", "trim-exact", "video-trim-exact")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=_dup_clients(), lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
    ]
    assert result.guide.excluded_spans == ()
