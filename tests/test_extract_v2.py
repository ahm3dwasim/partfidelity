"""whole-video extraction, clock, cuts, snap, fold, 12 steps.

Extraction fixes:
the step cap 8 -> 12 end to end, whole-video mode for videos that fit
the honest reservation, a burned-in m:ss.s clock on every extraction
chunk, scene-cut candidates with an optional prompt listing, snap of
each consensus boundary to its cut, the minimum-phase fold, and
per-boundary confidence records on the bundle.

Replay fixtures (tests/fixtures/extract_v2.json) come from the $0
live evidence: probe-whole's raw_A/raw_B (whole-video single runs),
the raw whole-video runs 1, 2 and 3 (whole-video with the cut list in the prompt),
and spark5's chunk-0 attempt0/retry1 runs, plus the 42 sample-video scene cuts
at threshold 0.15. The probes captured steps only, so titles map to
actions verbatim and the device/confidence envelope is synthesized;
spark5 payloads are verbatim. Scoring is one-to-one by time overlap
against the reference ground truth (0-10 ... 170.4-189.2, 87-112-137
since 2026-10-29, 170.4 since 2026-10-29, tail 189.2-214.17 excluded
since 2026-10-29), both ends within 3 s; the replay pins use
the time-aligned scorer.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import db, stage_cache
from app.admission import admit_job, job_dir
from app.boundaries import refine_consensus_bundle
from app.chunks import ChunkWindow, CutInfo
from app.contracts import MAX_STEPS, BoundaryInfo, Step, StepBundle
from app.scorer_time import score_guide
from app.ledger import LifecycleRepositoryImpl
from app.pipeline import (
    PipelineClients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    chunk_budget_refused,
    chunk_cost_estimate,
    plan_extraction_windows,
    run_stages,
)
from app.providers import select_providers, whole_video_fits
from app.providers.meta_spark import (
    SPARK_MAX_OUTPUT_TOKENS,
    SPARK_PROMPT,
    SPARK_SCHEMA,
    spark_prompt_for_clip,
    spark_worst_case,
    window_input_tokens,
)
from app.readback import ReadbackResult
from app.settings import (
    DEFAULT_SPARK_RESERVATION_USD,
    SPARK_RESERVATION_MARGIN,
    ProviderSettings,
    Settings,
    load_settings,
)
from app.spark import (
    ChunkExtent,
    SparkRunResult,
    SparkSchemaError,
    SparkStageConfig,
    StepsDisagreement,
    bundle_from_json,
    bundle_to_json,
    chunk_bundle_digest,
    consensus_bundle,
    consensus_digest,
    run_digest,
    validate_legacy_spark_payload,
    validate_spark_payload,
)
from app.uploads_store import UploadStoreImpl
from test_providers import FakeClock, FakeCtx, make_spark_client, ok, spark_response
from test_providers import ScriptedTransport
from test_spark_schema import make_payload, make_step

FIXTURE = Path(__file__).parent / "fixtures" / "extract_v2.json"

LIVE = SparkStageConfig(spread_s=2.0)  # the live runs ran tol=2.0
EXTENT60 = ChunkExtent(duration_s=60.0, tolerance_s=2.0)
EXTENT214 = ChunkExtent(duration_s=214.17, tolerance_s=2.0)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _truth() -> list[tuple[float, float]]:
    return [tuple(span) for span in _fixture()["sample_truth"]]


def _score(steps, truth=None, tol: float = 3.0) -> int:
    """One-to-one greedy overlap pairing; both ends within ``tol``."""
    truth = _truth() if truth is None else truth
    cands: list[tuple[float, int, int]] = []
    for ti, (ts, te) in enumerate(truth):
        for pi, step in enumerate(steps):
            overlap = max(0.0, min(te, step.end_s) - max(ts, step.start_s))
            if overlap > 0:
                cands.append((overlap, ti, pi))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_t: set[int] = set()
    used_p: set[int] = set()
    pairing: dict[int, int] = {}
    for _, ti, pi in cands:
        if ti not in used_t and pi not in used_p:
            used_t.add(ti)
            used_p.add(pi)
            pairing[ti] = pi
    passed = 0
    for ti, (ts, te) in enumerate(truth):
        pi = pairing.get(ti)
        if pi is None:
            continue
        step = steps[pi]
        if abs(step.start_s - ts) <= tol and abs(step.end_s - te) <= tol:
            passed += 1
    return passed


# --- cap 12: validate and flow through; 13 refused ---------------------------


def _partition_run(count: int, span: float):
    bounds = [i * span / count for i in range(count + 1)]
    return make_payload(
        steps=tuple(
            make_step(start=start, end=end, action=f"Repair phase {n}")
            for n, (start, end) in enumerate(zip(bounds, bounds[1:]), start=1)
        )
    )


@pytest.mark.parametrize("count", [9, 10, 11, 12])
def test_n_step_guides_validate_and_flow_through(count: int) -> None:
    payload = _partition_run(count, 120.0)
    runs = [validate_spark_payload(payload) for _ in range(3)]
    bundle = consensus_bundle(
        runs, LIVE, chunk_extent=ChunkExtent(duration_s=120.0, tolerance_s=2.0)
    )
    assert len(bundle.steps) == count
    assert [s.number for s in bundle.steps] == list(range(1, count + 1))
    assert len(bundle.boundaries) == count - 1
    assert bundle_from_json(bundle_to_json(bundle), LIVE).steps == bundle.steps


def test_thirteen_steps_refused_by_schema_and_cache() -> None:
    payload = _partition_run(13, 130.0)
    with pytest.raises(SparkSchemaError):
        validate_spark_payload(payload)
    # Consensus refuses past the cap too (thirteen segments from
    # twelve-step runs); the cache round-trip refuses a stored 13.
    assert MAX_STEPS == 12
    cached = json.loads(json.dumps(bundle_to_json(_twelve_step_bundle())))
    cached["steps"].append(dict(cached["steps"][0]))
    with pytest.raises(ValueError):
        bundle_from_json(cached, LIVE)


def _twelve_step_bundle():
    runs = [validate_spark_payload(_partition_run(12, 130.0)) for _ in range(3)]
    return consensus_bundle(
        runs, LIVE, chunk_extent=ChunkExtent(duration_s=130.0, tolerance_s=2.0)
    )


def test_schema_and_prompt_share_the_one_cap() -> None:
    assert SPARK_SCHEMA["properties"]["steps"]["maxItems"] == MAX_STEPS == 12
    assert f"(at most {MAX_STEPS})" in SPARK_PROMPT
    assert SPARK_SCHEMA["properties"]["steps"]["items"]["properties"]["parts"][
        "maxItems"
    ] == 3


# --- planning: whole-video first when it fits --------------------------------


def _planning_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        data_root=tmp_path / "data",
        db_path=tmp_path / "test.sqlite3",
        production=False,
        **overrides,
    )


def _windows(windows) -> list[tuple[float, float]]:
    return [(w.start_s, w.end_s) for w in windows]


def test_short_video_stays_a_single_window(tmp_path: Path) -> None:
    settings = _planning_settings(tmp_path)
    assert _windows(plan_extraction_windows(60.0, settings)) == [(0.0, 60.0)]


def test_whole_video_singles_when_the_reservation_fits(tmp_path: Path) -> None:
    settings = _planning_settings(tmp_path)
    assert _windows(plan_extraction_windows(65.0, settings)) == [(0.0, 65.0)]
    assert _windows(plan_extraction_windows(100.0, settings)) == [(0.0, 100.0)]
    assert _windows(plan_extraction_windows(214.17, settings)) == [(0.0, 214.17)]
    assert _windows(plan_extraction_windows(240.0, settings)) == [(0.0, 240.0)]


def test_whole_video_falls_back_to_chunks_past_the_fit(tmp_path: Path) -> None:
    settings = _planning_settings(tmp_path)
    assert _windows(plan_extraction_windows(241.0, settings)) == [
        (0.0, 60.0),
        (58.0, 118.0),
        (116.0, 176.0),
        (174.0, 234.0),
        (232.0, 241.0),
    ]
    # The fallback path still works when the envelope is the old
    # $0.26: 101 s and the 214 s clip chunk exactly as before.
    old_cap = _planning_settings(tmp_path, spark_reservation_usd=Decimal("0.26"))
    assert _windows(plan_extraction_windows(101.0, old_cap)) == [
        (0.0, 60.0),
        (58.0, 101.0),
    ]
    assert _windows(plan_extraction_windows(214.17, old_cap)) == [
        (0.0, 60.0),
        (58.0, 118.0),
        (116.0, 176.0),
        (174.0, 214.17),
    ]


def test_whole_video_ceiling_and_override(tmp_path: Path) -> None:
    settings = _planning_settings(tmp_path)
    assert len(plan_extraction_windows(300.0, settings)) == 6
    chunked = _planning_settings(tmp_path, whole_video_max_s=60.0)
    assert _windows(plan_extraction_windows(65.0, chunked)) == [
        (0.0, 60.0),
        (58.0, 65.0),
    ]


def test_cheap_prices_single_a_long_video(tmp_path: Path) -> None:
    settings = _planning_settings(
        tmp_path,
        providers=ProviderSettings(
            spark_input_usd_per_token=Decimal("0.0000001"),
            spark_output_usd_per_token=Decimal("0.0000004"),
        ),
    )
    assert _windows(plan_extraction_windows(214.17, settings)) == [(0.0, 214.17)]


def test_plan_extraction_windows_needs_settings_free_defaults(tmp_path: Path) -> None:
    assert _windows(plan_extraction_windows(65.0, None)) == [(0.0, 65.0)]
    assert _windows(plan_extraction_windows(214.17, None)) == [(0.0, 214.17)]


def test_chunk_cost_estimate_uses_effective_planning(tmp_path: Path) -> None:
    settings = _planning_settings(tmp_path)
    count, required, cap = chunk_cost_estimate(65.0, object(), settings)
    assert count == 1
    count, _, _ = chunk_cost_estimate(214.17, object(), settings)
    assert count == 1


def test_whole_video_job_past_the_per_video_cap_still_refused(
    tmp_path: Path,
) -> None:
    # The $1.60 per-video cap still binds: at 1.5x prices the 6-chunk
    # 300 s plan needs 6 x $0.37125 and refuses, while the 214 s
    # single-window job fits at docs prices.
    pricey = _planning_settings(
        tmp_path,
        providers=ProviderSettings(
            spark_input_usd_per_token=Decimal("0.000001875"),
            spark_output_usd_per_token=Decimal("0.000006375"),
        ),
    )
    trio = select_providers(pricey, replay=ReplayTrioForTest())
    refused, count, required, cap = chunk_budget_refused(
        300.0, trio.spark, pricey
    )
    assert refused is True
    assert count == 6
    assert cap == Decimal("1.60")
    assert required > cap
    settings = _planning_settings(tmp_path)
    cheap_trio = select_providers(settings, replay=ReplayTrioForTest())
    refused, count, required, _ = chunk_budget_refused(
        214.17, cheap_trio.spark, settings
    )
    assert refused is False
    assert count == 1
    # The single window prices window-sized (3 runs at 74960 input
    # tokens), not at the parent client's floor shares.
    assert required == Decimal("0.4086")


def test_whole_video_fits_matches_the_factory_comparison() -> None:
    # At the docs prices against the $0.48675 reservation, the honest
    # 273 s window fits and 274 s does not: 3 x (ceil(273 x 350) x
    # $0.00000125 + 10000 x $0.00000425) is $0.4858125, inside;
    # 274 s is $0.487125, past the cap. The factory enforces the same
    # comparison (see the select_providers fit/refusal tests below).
    cap = Decimal("0.48675")
    assert (
        whole_video_fits(
            273.0,
            Decimal("0.00000125"),
            Decimal("0.00000425"),
            3,
            tokens_per_video_s=350.0,
            spark_cap_usd=cap,
        )
        is True
    )
    assert (
        whole_video_fits(
            274.0,
            Decimal("0.00000125"),
            Decimal("0.00000425"),
            3,
            tokens_per_video_s=350.0,
            spark_cap_usd=cap,
        )
        is False
    )
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError):
            whole_video_fits(
                bad, Decimal("0.00000125"), Decimal("0.00000425"), 3,
                tokens_per_video_s=350.0, spark_cap_usd=cap,
            )
    for bad_cap in (Decimal("0"), Decimal("-1"), "0.48675"):
        with pytest.raises(ValueError):
            whole_video_fits(
                100.0, Decimal("0.00000125"), Decimal("0.00000425"), 3,
                tokens_per_video_s=350.0, spark_cap_usd=bad_cap,
            )


def test_window_input_tokens_floors_at_the_chunk_bound() -> None:
    assert window_input_tokens(60.0, 350.0) == 32000
    assert window_input_tokens(214.17, 350.0) == 74960
    with pytest.raises(ValueError):
        window_input_tokens(0.0, 350.0)
    with pytest.raises(ValueError):
        window_input_tokens(60.0, 0.0)
    worst = spark_worst_case(
        Decimal("0.00000125"), Decimal("0.00000425"), input_tokens=74960
    )
    assert worst == Decimal(74960) * Decimal("0.00000125") + Decimal(
        10000
    ) * Decimal("0.00000425")
    with pytest.raises(ValueError):
        spark_worst_case(
            Decimal("0.00000125"), Decimal("0.00000425"), input_tokens=0
        )


# --- prompt: clock variant and cut listing -----------------------------------


def test_clocked_prompt_names_the_burned_in_clock() -> None:
    plain = spark_prompt_for_clip(60.0)
    clocked = spark_prompt_for_clip(60.0, clocked=True)
    assert "burned into the top-left corner" in clocked
    assert "burned into the top-left corner" not in plain
    assert "on-screen clock" in clocked
    assert "m:ss" in plain and "m:ss" in clocked
    assert "exactly 60 seconds" in clocked


def test_cut_listing_rounds_to_01_with_guidance() -> None:
    prompt = spark_prompt_for_clip(214.17, scene_cuts=(9.4, 17.333, 36.766667))
    assert "Visible scene cuts happen at these times (seconds): 9.4, 17.3, 36.8" in prompt
    assert "not every cut is a step boundary" in prompt
    assert "may fall between cuts" in prompt
    assert "Treat each change of what is being done to the laptop" in prompt
    assert "as a new step." in prompt
    unlisted = spark_prompt_for_clip(214.17, scene_cuts=())
    assert "Visible scene cuts" not in unlisted


def test_prompt_builder_rejects_misshaped_tuning() -> None:
    with pytest.raises(ValueError):
        spark_prompt_for_clip(60.0, clocked="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        spark_prompt_for_clip(60.0, scene_cuts=(float("nan"),))
    with pytest.raises(ValueError):
        spark_prompt_for_clip(60.0, scene_cuts=(-1.0,))


def test_live_client_sends_clocked_cut_prompt() -> None:
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport(
        [ok(spark_response({"device_guess": "x", "confidence": 1, "steps": []}))]
    )
    make_spark_client(ctx, transport, clock).run_steps(
        "file-1", clip_duration_s=60.0, scene_cuts=(9.4, 17.3), clocked=True
    )
    assert len(transport.posts) == 1
    body = json.loads(transport.posts[0]["body"].decode("utf-8"))
    sent = body["input"][0]["content"][0]["text"]
    assert sent == spark_prompt_for_clip(
        60.0, clocked=True, scene_cuts=(9.4, 17.3)
    )
    assert body["text"]["format"]["schema"]["properties"]["steps"]["maxItems"] == 12


def test_live_client_tuning_needs_a_duration() -> None:
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport(
        [ok(spark_response({"device_guess": "x", "confidence": 1, "steps": []}))]
    )
    client = make_spark_client(ctx, transport, clock)
    with pytest.raises(ValueError, match="need clip_duration_s"):
        client.run_steps("file-1", scene_cuts=(9.4,))
    with pytest.raises(ValueError, match="need clip_duration_s"):
        client.run_steps("file-1", clocked=True)
    assert transport.posts == []


# --- settings loaders --------------------------------------------------------


def test_extract_v2_loader_defaults() -> None:
    settings = load_settings({})
    assert settings.whole_video_max_s == 240.0
    assert settings.spark_input_tokens_per_video_s == 350.0
    assert settings.clock_burn_in is True
    assert settings.scene_cut_threshold == 0.15
    assert settings.scene_cuts_in_prompt is True  # on by the replay result
    assert settings.snap_window_s == 2.0
    assert settings.min_phase_length_s == 5.0
    assert settings.spark_reservation_usd == Decimal("0.48675")


def test_spark_reservation_default_is_derived_not_guessed() -> None:
    # The default is the honest 3-run worst case for a 240 s window at
    # the default rate and docs prices, plus the 10% margin — recomputed
    # here through the same helpers the factory uses, so a drift in any
    # input fails this test instead of silently under-funding windows.
    # The output cap is one shared constant, not two pinned equal.
    from app import settings as settings_mod

    assert settings_mod.SPARK_MAX_OUTPUT_TOKENS is SPARK_MAX_OUTPUT_TOKENS
    honest = (
        spark_worst_case(
            Decimal("0.00000125"),
            Decimal("0.00000425"),
            input_tokens=window_input_tokens(240.0, 350.0),
        )
        * 3
    )
    assert honest == Decimal("0.4425")
    assert DEFAULT_SPARK_RESERVATION_USD == honest * SPARK_RESERVATION_MARGIN
    assert load_settings({}).spark_reservation_usd == Decimal("0.48675")


def test_extract_v2_loader_overrides() -> None:
    settings = load_settings(
        {
            "PARTFIDELITY_WHOLE_VIDEO_MAX_S": "120",
            "PARTFIDELITY_SPARK_INPUT_TOKENS_PER_VIDEO_S": "400",
            "PARTFIDELITY_CLOCK_BURN_IN": "no",
            "PARTFIDELITY_SCENE_CUT_THRESHOLD": "0.3",
            "PARTFIDELITY_SCENE_CUTS_IN_PROMPT": "no",
            "PARTFIDELITY_SNAP_WINDOW_S": "1.5",
            "PARTFIDELITY_MIN_PHASE_LENGTH_S": "8",
            "PARTFIDELITY_SPARK_RESERVATION_USD": "1.00",
        }
    )
    assert settings.whole_video_max_s == 120.0
    assert settings.spark_input_tokens_per_video_s == 400.0
    assert settings.clock_burn_in is False
    assert settings.scene_cut_threshold == 0.3
    assert settings.scene_cuts_in_prompt is False
    assert settings.snap_window_s == 1.5
    assert settings.min_phase_length_s == 8.0
    assert settings.spark_reservation_usd == Decimal("1.00")


@pytest.mark.parametrize(
    "env",
    [
        {"PARTFIDELITY_WHOLE_VIDEO_MAX_S": "0"},
        {"PARTFIDELITY_WHOLE_VIDEO_MAX_S": "soon"},
        {"PARTFIDELITY_SPARK_INPUT_TOKENS_PER_VIDEO_S": "-1"},
        {"PARTFIDELITY_CLOCK_BURN_IN": "maybe"},
        {"PARTFIDELITY_SCENE_CUT_THRESHOLD": "0"},
        {"PARTFIDELITY_SCENE_CUT_THRESHOLD": "1.5"},
        {"PARTFIDELITY_SCENE_CUTS_IN_PROMPT": "2"},
        {"PARTFIDELITY_SNAP_WINDOW_S": "-1"},
        {"PARTFIDELITY_MIN_PHASE_LENGTH_S": "0"},
        {"PARTFIDELITY_SPARK_RESERVATION_USD": "0"},
        {"PARTFIDELITY_SPARK_RESERVATION_USD": "soon"},
    ],
)
def test_extract_v2_loader_rejects_garbage(env) -> None:
    with pytest.raises(RuntimeError):
        load_settings(env)


def test_extract_v2_settings_post_init_bounds(tmp_path: Path) -> None:
    base = {
        "data_root": tmp_path / "data",
        "db_path": tmp_path / "test.sqlite3",
        "production": False,
    }
    for field, bad in (
        ("whole_video_max_s", 0.0),
        ("spark_input_tokens_per_video_s", -1.0),
        ("clock_burn_in", "yes"),
        ("scene_cut_threshold", 0.0),
        ("scene_cuts_in_prompt", 1),
        ("snap_window_s", -0.5),
        ("min_phase_length_s", 0.0),
        ("spark_reservation_usd", Decimal("0")),
        ("spark_reservation_usd", Decimal("-1")),
    ):
        with pytest.raises(ValueError):
            Settings(**{**base, field: bad})


# --- digests bind cuts, clock, and tune --------------------------------------


def test_consensus_and_run_digests_bind_cuts_and_clock() -> None:
    plain = consensus_digest("j", "v", LIVE, EXTENT60)
    assert consensus_digest("j", "v", LIVE, EXTENT60) == plain
    assert consensus_digest("j", "v", LIVE, EXTENT60, (9.4,), True) != plain
    assert consensus_digest(
        "j", "v", LIVE, EXTENT60, (9.4,), True
    ) == consensus_digest("j", "v", LIVE, EXTENT60, (9.4,), True)
    assert consensus_digest("j", "v", LIVE, EXTENT60, (9.4,)) != plain
    assert run_digest("j", "v", 0, LIVE, EXTENT60) == run_digest(
        "j", "v", 0, LIVE, EXTENT60
    )
    assert run_digest(
        "j", "v", 0, LIVE, EXTENT60, (9.4,), True
    ) != run_digest("j", "v", 0, LIVE, EXTENT60)
    with pytest.raises(ValueError):
        consensus_digest("j", "v", LIVE, EXTENT60, (float("nan"),))
    with pytest.raises(ValueError):
        run_digest("j", "v", 0, LIVE, EXTENT60, None, "yes")  # type: ignore[arg-type]


def test_chunk_bundle_digest_binds_the_tune() -> None:
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
    for knob, value in (
        ("scene_cut_threshold", 0.3),
        ("scene_cuts_in_prompt", True),
        ("snap_window_s", 1.0),
        ("min_phase_length_s", 8.0),
        ("clock_burn_in", False),
        ("merge_same_action", False),
    ):
        assert chunk_bundle_digest(**{**base, knob: value}) != first
    with pytest.raises(ValueError):
        chunk_bundle_digest(**{**base, "scene_cut_threshold": 0.0})
    with pytest.raises(ValueError):
        chunk_bundle_digest(**{**base, "snap_window_s": -1.0})
    with pytest.raises(ValueError):
        chunk_bundle_digest(**{**base, "merge_same_action": "yes"})


# --- factory: window-sized shares --------------------------------------------


def _live_settings(tmp_path: Path, **overrides) -> Settings:
    params = {
        "data_root": tmp_path / "data",
        "db_path": tmp_path / "test.sqlite3",
        "production": False,
        "meta_api_key": "k-test",
        "providers": ProviderSettings(mode="live"),
    }
    params.update(overrides)
    return Settings(**params)


def test_select_providers_sizes_shares_to_the_window(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    chunk = select_providers(settings, FakeCtx(), window_s=60.0)
    assert chunk.spark.spark_stage_config.run_shares == (
        spark_worst_case(Decimal("0.00000125"), Decimal("0.00000425")),
    ) * 3
    whole = select_providers(settings, FakeCtx(), window_s=100.0)
    expected = spark_worst_case(
        Decimal("0.00000125"), Decimal("0.00000425"), input_tokens=35000
    )
    assert whole.spark.spark_stage_config.run_shares == (expected,) * 3


def test_select_providers_builds_a_214s_window(tmp_path: Path) -> None:
    # The 214 s whole-video job fits the $0.48675 reservation: 3 runs
    # at ceil(214.17 x 350) = 74960 input tokens.
    settings = _live_settings(tmp_path)
    trio = select_providers(settings, FakeCtx(), window_s=214.17)
    expected = spark_worst_case(
        Decimal("0.00000125"), Decimal("0.00000425"), input_tokens=74960
    )
    assert trio.spark.spark_stage_config.run_shares == (expected,) * 3


def test_select_providers_refuses_an_unfundable_window(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    with pytest.raises(RuntimeError, match="Spark reservation"):
        select_providers(settings, FakeCtx(), window_s=274.0)
    with pytest.raises(RuntimeError):
        select_providers(settings, FakeCtx(), window_s=0.0)


def test_select_providers_replay_ignores_the_window(tmp_path: Path) -> None:
    settings = _planning_settings(tmp_path)
    trio = select_providers(
        settings,
        replay=ReplayTrioForTest(),
        window_s=274.0,  # would refuse in live mode; replay keeps the floor
    )
    assert trio.spark.spark_stage_config.run_shares == (
        spark_worst_case(Decimal("0.00000125"), Decimal("0.00000425")),
    ) * 3


def test_spark_config_reservation_bounds_the_shares() -> None:
    shares = (Decimal("0.15"), Decimal("0.15"), Decimal("0.15"))
    with pytest.raises(ValueError, match="Spark reservation"):
        SparkStageConfig(run_shares=shares)  # $0.45 past the $0.26 default
    config = SparkStageConfig(
        run_shares=shares, reservation_usd=Decimal("0.48675")
    )
    assert sum(config.run_shares, Decimal("0")) == Decimal("0.45")
    with pytest.raises(ValueError, match="reservation_usd"):
        SparkStageConfig(run_shares=shares, reservation_usd=Decimal("0"))


# --- the child Spark client sizes from decoded duration -----------


class _LiveSparkStub:
    """Placeholder parent Spark client: chunks build real clients, and no
    read-back fires (the scripted steps carry no parts)."""

    def read_back(self, *args, **kwargs):
        raise AssertionError("no read-back should fire without parts")


class _LiveMetaFake:
    """Meta files fake live mode accepts (replay adapters are rejected)."""

    def __init__(self) -> None:
        self._inner = ReplayMetaFilesClient()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def upload_calls(self):
        return self._inner.upload_calls


def _tiling_payload(duration_s: float) -> dict:
    """Two phases tiling [0, duration_s): the scripted Spark answer."""
    half = duration_s / 2.0
    return make_payload(
        device="laptop",
        steps=[
            make_step(
                start=0.0, end=half, action="Remove the rear screws",
                parts=(), tools=(),
            ),
            make_step(
                start=half, end=duration_s, action="Close the cover",
                parts=(), tools=(),
            ),
        ]
    )


def _run_live_sizing(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    monkeypatch,
    settings: Settings,
    decoded_s: float | None,
):
    """Run one 62 s live window; record the sizing input select saw."""
    from app import pipeline as pipeline_mod

    clip = _hard_cut_clip(tmp_path / "sizing.mp4")
    scripted = ScriptedTransport(
        [ok(spark_response(_tiling_payload(decoded_s or 62.0)))] * 3
    )
    real_select = pipeline_mod.select_providers
    recorded: list = []
    trios: list = []

    def _recording_select(settings_arg, ctx=None, **kwargs):
        recorded.append(kwargs.get("window_s"))
        trio = real_select(settings_arg, ctx, transport=scripted, **kwargs)
        trios.append(trio)
        return trio

    monkeypatch.setattr(pipeline_mod, "select_providers", _recording_select)
    if decoded_s is not None:
        monkeypatch.setattr(
            pipeline_mod,
            "probe_cut",
            lambda path, *, timeout_s: CutInfo(
                start_s=0.0, duration_s=decoded_s
            ),
        )
    meta = _LiveMetaFake()
    clients = PipelineClients(
        spark=_LiveSparkStub(), sam=object(), meta=meta
    )
    ctx = admit_job("sizing-parent", "sizing-parent", "video-sizing-parent")
    result = run_stages(
        ctx, "parent-file-id", clip,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=False,
        settings=settings, duration_s=62.0,
    )
    return result, recorded, trios, meta, scripted


def test_chunk_client_sizes_from_decoded_overrun(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # Nominal 62.0 s window, decoded 62.9 s (overrun): the child client
    # is sized from the 62.9 the bytes actually play, not the nominal.
    settings = _live_settings(tmp_path, db_path=lane_db)
    result, recorded, trios, _, scripted = _run_live_sizing(
        tmp_path, lane_db, uploads, lifecycle, monkeypatch, settings, 62.9
    )
    assert result.state == "done", result.error
    assert recorded == [62.9]
    floor = spark_worst_case(Decimal("0.00000125"), Decimal("0.00000425"))
    assert trios[0].spark.spark_stage_config.run_shares == (floor,) * 3
    assert len(scripted.posts) == 3


def test_chunk_client_sizes_from_decoded_just_inside_the_cap(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # At 1.05x prices the honest 62.9 s window costs $0.259875 for 3
    # runs — just inside the $0.26 chunk reservation — and still builds.
    settings = _live_settings(
        tmp_path, db_path=lane_db,
        providers=ProviderSettings(
            mode="live",
            spark_input_usd_per_token=Decimal("0.0000013125"),
            spark_output_usd_per_token=Decimal("0.0000044625"),
        ),
    )
    honest = (
        spark_worst_case(
            Decimal("0.0000013125"),
            Decimal("0.0000044625"),
            input_tokens=window_input_tokens(62.9, 350.0),
        )
        * 3
    )
    assert honest == Decimal("0.259875")  # $0.000125 inside $0.26
    result, recorded, trios, _, scripted = _run_live_sizing(
        tmp_path, lane_db, uploads, lifecycle, monkeypatch, settings, 62.9
    )
    assert result.state == "done", result.error
    assert recorded == [62.9]
    assert trios[0].spark.spark_stage_config.run_shares == (honest / 3,) * 3
    assert len(scripted.posts) == 3


def test_unfundable_window_refuses_before_any_upload(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # At 2x prices even the 60 s chunk floor ($0.495 for 3 runs) cannot
    # fit the reservation: the child client refuses before any upload
    # or Spark call, and the job fails closed.
    settings = _live_settings(
        tmp_path, db_path=lane_db,
        providers=ProviderSettings(
            mode="live",
            spark_input_usd_per_token=Decimal("0.0000025"),
            spark_output_usd_per_token=Decimal("0.0000085"),
        ),
    )
    result, recorded, trios, meta, scripted = _run_live_sizing(
        tmp_path, lane_db, uploads, lifecycle, monkeypatch, settings, None
    )
    assert result.state == "failed"
    assert result.error == "step extraction failed"
    assert len(recorded) == 1  # the first chunk's select refused
    assert recorded[0] == pytest.approx(60.0, abs=1.0)
    assert trios == []
    assert meta.upload_calls == []
    assert scripted.posts == []


class ReplayTrioForTest:
    """Minimal recorded trio: the factory wraps it untouched."""

    def __init__(self) -> None:
        inner = ReplaySparkClient(
            runs=[
                SparkRunResult(
                    payload={"device_guess": "x", "confidence": 1, "steps": []},
                    cost_usd=Decimal("0"),
                    latency_ms=0,
                )
            ]
            * 3
        )
        self.spark = inner
        self.sam = ReplaySamClient({})
        self.readback = None


# --- $0 replay: probe and spark5 raws through the pipeline --------------------


def _replay(attempt, extent, cuts):
    runs = [validate_legacy_spark_payload(payload) for payload in attempt]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=extent)
    refined = refine_consensus_bundle(
        bundle, cuts=cuts, snap_window_s=2.0, min_phase_length_s=5.0
    )
    return bundle, refined


def test_replay_spark5_attempt0() -> None:
    fixture = _fixture()
    cuts = tuple(c for c in fixture["sample_cuts"] if c < 60.0)
    bundle, refined = _replay(fixture["spark5_attempt0"], EXTENT60, cuts)
    # The over-long run (0-214) drops from consensus: every boundary
    # holds 2 of 3 runs.
    assert [b.support_runs for b in bundle.boundaries] == [2, 2, 2, 2]
    assert _score(bundle.steps) == 3
    assert _score(refined.steps) == 3
    assert [b.snapped for b in refined.boundaries] == [True] * 4
    assert [round(b.time_s, 2) for b in refined.boundaries] == [
        9.4,
        17.3,
        31.2,
        36.77,
    ]


def test_replay_spark5_retry1() -> None:
    fixture = _fixture()
    cuts = tuple(c for c in fixture["sample_cuts"] if c < 60.0)
    bundle, refined = _replay(fixture["spark5_retry1"], EXTENT60, cuts)
    assert [b.support_runs for b in bundle.boundaries] == [2, 2, 2, 2]
    assert _score(bundle.steps) == 3
    assert _score(refined.steps) == 3
    assert [b.snapped for b in refined.boundaries] == [True] * 4


def test_replay_probe_pair_as_two_run_consensus() -> None:
    fixture = _fixture()
    config = SparkStageConfig(
        runs=2, spread_s=2.0, run_shares=(Decimal("0.09"), Decimal("0.09"))
    )
    runs = [
        validate_legacy_spark_payload(fixture["probe_a"]),
        validate_legacy_spark_payload(fixture["probe_b"]),
    ]
    bundle = consensus_bundle(runs, config, chunk_extent=EXTENT214)
    refined = refine_consensus_bundle(
        bundle,
        cuts=tuple(fixture["sample_cuts"]),
        snap_window_s=2.0,
        min_phase_length_s=5.0,
    )
    assert len(bundle.steps) == 5
    assert _score(bundle.steps) == 1
    assert _score(refined.steps) == 2
    assert [b.support_runs for b in refined.boundaries] == [2, 2, 2, 2]
    assert [b.snapped for b in refined.boundaries] == [True] * 4


def test_replay_single_runs_validate_and_score() -> None:
    fixture = _fixture()
    # Probe A loses step 9 to the excluded tail (its 170.5-214.17
    # phase vs the truth end 189.2); probe B is unchanged at 5.
    for variant, expected in (("probe_a", 1), ("probe_b", 5)):
        run = validate_legacy_spark_payload(fixture[variant])
        assert _score(run.steps) == expected


def test_replay_records_need_check_from_support_and_cuts() -> None:
    fixture = _fixture()
    cuts = tuple(c for c in fixture["sample_cuts"] if c < 60.0)
    _, refined = _replay(fixture["spark5_attempt0"], EXTENT60, cuts)
    # 2-of-3 support flags every boundary even after clean snaps.
    assert [b.needs_check for b in refined.boundaries] == [True] * 4
    assert [s.time_needs_check for s in refined.steps] == [True] * 5


def test_refined_bundle_round_trips_through_cache_json() -> None:
    fixture = _fixture()
    cuts = tuple(c for c in fixture["sample_cuts"] if c < 60.0)
    _, refined = _replay(fixture["spark5_attempt0"], EXTENT60, cuts)
    assert bundle_from_json(bundle_to_json(refined), LIVE) == refined


# --- chunk path: clock, cuts, snap, fold end to end ---------------------------


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


def _hard_cut_clip(path: Path, *, seconds: float = 62.0, fps: int = 5) -> Path:
    """Black then white halves: exactly one hard cut at half duration."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    count = int(seconds * fps)
    try:
        for index in range(count):
            if index < count // 2:
                writer.write(np.zeros((240, 320, 3), np.uint8))
            else:
                writer.write(np.full((240, 320, 3), 255, np.uint8))
    finally:
        writer.release()
    return path


def _payload_step(number, start, end, action, kind="repair"):
    return {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": [],
        "parts": [],
        "hazards": [],
        "kind": kind,
    }


def test_whole_video_chunk_path_snaps_folds_and_stores_cuts(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
) -> None:
    clip = _hard_cut_clip(tmp_path / "hardcut.mp4")
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False
    )
    # One 62 s window (whole-video): the 31.4 s consensus edge snaps to
    # the 31.0 s hard cut, and the 2 s middle phase folds into the
    # agreeing later phase.
    steps = [
        _payload_step(1, 0.0, 31.4, "Remove the rear screws"),
        _payload_step(2, 31.4, 33.4, "Close the cover"),
        _payload_step(3, 33.4, 62.0, "Close the cover"),
    ]
    spark = ReplaySparkClient(
        [
            SparkRunResult(
                payload={
                    "device_guess": "laptop",
                    "confidence": 0.82,
                    "steps": [dict(step) for step in steps],
                },
                cost_usd=Decimal("0.08"),
                latency_ms=900,
            )
            for _ in range(3)
        ],
        readbacks={
            "battery": ReadbackResult(
                answer="yes", cost_usd=Decimal("0.04"), latency_ms=120
            )
        },
    )
    clients = PipelineClients(
        spark=spark, sam=ReplaySamClient({}), meta=ReplayMetaFilesClient()
    )
    ctx = admit_job("extract-v2-e2e", "extract-v2-e2e", "video-extract-v2-e2e")
    result = run_stages(
        ctx,
        "parent-file-id",
        clip,
        clients=clients,
        lifecycle=lifecycle,
        uploads=uploads,
        replay=True,
        settings=settings,
        duration_s=62.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 31.0),
        (31.0, 62.0),
    ]
    assert [s.action for s in result.guide.steps] == [
        "Remove the rear screws",
        "Close the cover",
    ]
    assert len(spark.run_calls) == 3  # one window, three runs
    cuts_path = job_dir(str(ctx.job_id)) / "scene_cuts_000.json"
    assert json.loads(cuts_path.read_text(encoding="utf-8"))["cuts"] == [31.0]


def test_direct_path_ignores_long_window_planning(tmp_path: Path) -> None:
    # A single short window never takes the chunk path, however the
    # ceiling is set; anything longer always does.
    from app.pipeline import _takes_chunk_path

    settings = _planning_settings(tmp_path)
    assert (
        _takes_chunk_path((ChunkWindow(0.0, 60.0),), settings) is False
    )
    assert _takes_chunk_path((ChunkWindow(0.0, 60.1),), settings) is True
    assert (
        _takes_chunk_path(
            (ChunkWindow(0.0, 60.0), ChunkWindow(58.0, 61.0)), settings
        )
        is True
    )


# --- $0 replay: whole-video runs through the pipeline -----------------
# Scored with the time-aligned scorer (exact overlap assignment,
# wording report-only) against the updated truth (170.4).


def _scorer_gt() -> list[dict]:
    """New-truth rows for the sample video for the timing scorer (wording is report-only)."""
    proposed = json.loads(
        (FIXTURE.parent / "sample_steps_proposed.json").read_text(encoding="utf-8")
    )["steps"]
    return [
        {"number": n, "start_s": ts, "end_s": te, "action": step["action"]}
        for n, ((ts, te), step) in enumerate(zip(_truth(), proposed), start=1)
    ]


def _scorer_score(steps) -> int:
    rows = [
        {
            "number": step.number,
            "start_s": step.start_s,
            "end_s": step.end_s,
            "action": step.action,
        }
        for step in steps
    ]
    excluded = [tuple(span) for span in _fixture()["sample_excluded"]]
    return score_guide(rows, _scorer_gt(), excluded_spans=excluded).strict_passes


def test_replay_probe_c_validate_only_matches_manager() -> None:
    fixture = _fixture()
    # Tail exclusion (2026-10-29): runs 1 and 3 lose step 9 (their
    # 170.4-214.17 tails vs the truth end 189.2); run 2 never passed it.
    for variant, expected in (("cutlist_run_1", 4), ("cutlist_run_2", 6), ("cutlist_run_3", 7)):
        run = validate_legacy_spark_payload(fixture[variant])
        assert _scorer_score(run.steps) == expected


def test_replay_probe_c_consensus_and_refine() -> None:
    fixture = _fixture()
    runs = [
        validate_legacy_spark_payload(fixture[variant])
        for variant in ("cutlist_run_1", "cutlist_run_2", "cutlist_run_3")
    ]
    bundle = consensus_bundle(runs, LIVE, chunk_extent=EXTENT214)
    # The product time consensus lands exactly on run 3's bounds (as does
    # the original clustering consensus); run 1's 7-step partition drops
    # the 11.4/31.2/59.6 and 170.4 edges, hence 2-of-3 there.
    assert [b.time_s for b in bundle.boundaries] == [
        11.4, 17.3, 31.2, 36.8, 59.6, 86.8, 111.5, 134.8, 170.4,
    ]
    assert [(b.support_runs, b.total_runs) for b in bundle.boundaries] == [
        (2, 3), (3, 3), (2, 3), (3, 3), (2, 3), (3, 3), (3, 3), (3, 3), (2, 3),
    ]
    # 7: the consensus tail still ends at 214.17, past the 189.2 truth end.
    assert _scorer_score(bundle.steps) == 7
    refined = refine_consensus_bundle(
        bundle,
        cuts=tuple(fixture["sample_cuts"]),
        snap_window_s=2.0,
        min_phase_length_s=5.0,
    )
    # Every edge already sits on a prompt-listed cut (rounded to 0.1 s),
    # so the snap only nudges six of them onto the exact cut times; the
    # fold finds nothing short; the merge then fuses the two duplicate
    # pairs (17.3-31.2 + 31.2-36.77 "Disable the internal battery in
    # BIOS", 36.77-59.57 + 59.57-86.83 "Open the bottom cover") and
    # drops exactly their two edge records.
    assert [b.snapped for b in refined.boundaries] == [
        True, False, True, True, True, False, True,
    ]
    assert [round(b.time_s, 4) for b in refined.boundaries] == [
        11.4333, 17.3, 36.7667, 86.8333, 111.5333, 134.8, 170.4333,
    ]
    assert len(refined.steps) == 8
    # 6: the BIOS fusion fixes step 3 (17.3-36.77 within ±3 s of
    # 17-37), but the cover fusion spans GT 4 + GT 5 (36.77-86.83:
    # GT 4 goes missing, GT 5 fails on its start), and step 9 still
    # ends at 214.17.
    assert _scorer_score(refined.steps) == 6


def _single_run_bundle(run) -> StepBundle:
    """One validated run as a 1-of-1 bundle, so snap/fold can replay it."""
    return StepBundle(
        device_guess=run.device_guess,
        confidence=run.confidence,
        steps=tuple(
            Step(
                number=step.number,
                start_s=step.start_s,
                end_s=step.end_s,
                action=step.action,
                tools=(),
                parts=(),
                hazards=(),
                time_spread_s=0.0,
                time_needs_check=False,
            )
            for step in run.steps
        ),
        source_runs=1,
        timestamps_approximate=True,
        boundaries=tuple(
            BoundaryInfo(
                time_s=step.start_s,
                support_runs=1,
                total_runs=1,
                nearest_cut_s=None,
                cut_distance_s=None,
                snapped=False,
                snap_delta_s=0.0,
                needs_check=False,
            )
            for step in run.steps[1:]
        ),
    )


# --- one window estimate everywhere -------------------
# The verdict's prices: input=$0.00000375, output=$0.0000034. The 60 s
# floor worst case is 32000 x $0.00000375 + 10000 x $0.0000034 = $0.154
# per run, $0.462 for three — past the $0.26 default Spark reservation.

VERDICT_IN = Decimal("0.00000375")
VERDICT_OUT = Decimal("0.0000034")


def _verdict_live_settings(tmp_path: Path, **overrides) -> Settings:
    params: dict = {
        "providers": ProviderSettings(
            mode="live",
            spark_input_usd_per_token=VERDICT_IN,
            spark_output_usd_per_token=VERDICT_OUT,
        )
    }
    params.update(overrides)
    return _live_settings(tmp_path, **params)


def test_no_window_build_refuses_floor_past_default_cap(
    tmp_path: Path,
) -> None:
    # a direct/no-window job keeps the $0.26 default cap, so
    # the $0.462 floor at the verdict's prices refuses at build time —
    # before any hold — instead of dying on its second Spark hold.
    settings = _verdict_live_settings(tmp_path)
    ctx = FakeCtx()
    with pytest.raises(RuntimeError, match="Spark reservation"):
        select_providers(settings, ctx)
    assert ctx.holds == []


def test_explicit_window_build_gets_raised_window_sized_cap(
    tmp_path: Path,
) -> None:
    # an explicitly windowed build gets the raised,
    # window-sized cap ($0.462 for 60 s at the verdict's prices), not
    # the configured $0.48675 ceiling.
    settings = _verdict_live_settings(tmp_path)
    trio = select_providers(settings, FakeCtx(), window_s=60.0)
    config = trio.spark.spark_stage_config
    assert config.run_shares == (Decimal("0.154"),) * 3
    assert config.reservation_usd == Decimal("0.462")


def test_window_cap_and_client_come_from_the_same_call(
    tmp_path: Path, lane_db: Path
) -> None:
    # the admitted job-context cap and the built client's
    # reservation are the same value — one call feeds both, and three
    # full-share holds fit it exactly.
    from app.providers import spark_window_cap_usd

    settings = _verdict_live_settings(tmp_path, db_path=lane_db)
    cap = spark_window_cap_usd(
        60.0, VERDICT_IN, VERDICT_OUT, 3, tokens_per_video_s=350.0
    )
    assert cap == Decimal("0.462")
    ctx = admit_job(
        "case3-cap", "case3-cap", "video-case3-cap",
        internal=True, spark_cap_usd=cap,
    )
    trio = select_providers(settings, ctx, window_s=60.0)
    assert trio.spark.spark_stage_config.reservation_usd == cap
    holds = [ctx.reserve("spark_steps", Decimal("0.154")) for _ in range(3)]
    for hold in holds:
        hold.settle(Decimal("0.154"), "succeeded")


def test_window_estimate_is_runs_times_window_worst() -> None:
    # The one estimate: run count times the per-run worst case for the
    # window's input-token bound plus the output cap. None sizes the
    # 60 s chunk floor, exactly like a short window.
    from app.providers.meta_spark import spark_window_estimate_usd

    cheap_in = Decimal("0.00000125")
    cheap_out = Decimal("0.00000425")
    assert spark_window_estimate_usd(
        60.0, cheap_in, cheap_out, 3, tokens_per_video_s=350.0
    ) == Decimal("0.2475")
    assert spark_window_estimate_usd(
        None, cheap_in, cheap_out, 3, tokens_per_video_s=350.0
    ) == Decimal("0.2475")
    assert spark_window_estimate_usd(
        214.17, cheap_in, cheap_out, 3, tokens_per_video_s=350.0
    ) == Decimal("0.4086")
    for bad_window in (0.0, -1.0, float("nan"), True, "60"):
        with pytest.raises(ValueError):
            spark_window_estimate_usd(
                bad_window, cheap_in, cheap_out, 3, tokens_per_video_s=350.0
            )
    with pytest.raises(ValueError):
        spark_window_estimate_usd(
            60.0, Decimal("0"), cheap_out, 3, tokens_per_video_s=350.0
        )
    with pytest.raises(ValueError):
        spark_window_estimate_usd(
            60.0, cheap_in, cheap_out, 0, tokens_per_video_s=350.0
        )
    with pytest.raises(ValueError):
        spark_window_estimate_usd(
            60.0, cheap_in, cheap_out, 3, tokens_per_video_s=0.0
        )


def test_preflight_prices_a_214s_window_window_sized(
    tmp_path: Path,
) -> None:
    # with a 214.17 s clip and a $0.30 job cap, preflight
    # must refuse — the window-sized $0.4086, not the $0.2475 floor
    # shares, is priced against the cap.
    settings = _planning_settings(tmp_path, job_cost_cap_usd=Decimal("0.30"))
    trio = select_providers(settings, replay=ReplayTrioForTest())
    refused, count, required, cap = chunk_budget_refused(
        214.17, trio.spark, settings
    )
    assert refused is True
    assert count == 1
    assert required == Decimal("0.4086")
    assert cap == Decimal("0.30")
    # The priced figure is the same call child selection uses.
    from app.providers.meta_spark import spark_window_estimate_usd

    assert required == spark_window_estimate_usd(
        214.17,
        Decimal("0.00000125"),
        Decimal("0.00000425"),
        3,
        tokens_per_video_s=350.0,
    )


def test_retry_prices_the_decoded_window_window_sized(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # a 214.17 s whole-video window whose first
    # attempt disagrees needs a $0.4086 retry; against a $0.60 job cap
    # (preflight passes at $0.4086) the retry refuses up front with the
    # window-sized figure, before any retry upload or Spark call.
    from test_chunk_retry import (
        _agreeing_runs,
        _disagreeing_runs,
        _legacy_spark,
    )
    from app import pipeline as pipeline_mod

    settings = _planning_settings(
        tmp_path,
        step_span_mode="legacy",
        job_cost_cap_usd=Decimal("0.60"),
        chunk_retry_budget_usd=Decimal("1.00"),
        clock_burn_in=False,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "probe_cut",
        lambda path, *, timeout_s: CutInfo(start_s=0.0, duration_s=214.17),
    )
    clip = _hard_cut_clip(tmp_path / "retry214.mp4", seconds=214.0)
    clients = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("case3-retry", "case3-retry", "video-case3-retry")
    result = run_stages(
        ctx,
        "parent-file-id",
        clip,
        clients=clients,
        lifecycle=lifecycle,
        uploads=uploads,
        replay=True,
        settings=settings,
        duration_s=214.17,
    )
    assert result.state == "failed"
    assert result.error == (
        "step extraction failed: chunk 0 disagreed and a $0.4086 retry"
        " would exceed the $0.60 job budget"
    )
    assert len(clients.spark.run_calls) == 3
    assert len(clients.meta.upload_calls) == 1


def test_over_cap_readback_config_refused_before_any_request(
    tmp_path: Path,
) -> None:
    # 8000 x $0.000005 + 2000 x $0.0000001 is a $0.0402
    # read-back worst case, past the $0.04 per-call hold — a config the
    # Spark and envelope checks admit must still refuse loudly, before
    # any job, or read_back would fail on settlement overrun.
    from app.providers.meta_readback import readback_worst_case

    assert readback_worst_case(
        Decimal("0.000005"), Decimal("0.0000001")
    ) == Decimal("0.0402")
    settings = _live_settings(
        tmp_path,
        providers=ProviderSettings(
            mode="live",
            spark_input_usd_per_token=Decimal("0.000005"),
            spark_output_usd_per_token=Decimal("0.0000001"),
        ),
    )
    ctx = FakeCtx()
    with pytest.raises(RuntimeError, match="read-back"):
        select_providers(settings, ctx, window_s=60.0)
    assert ctx.holds == []


def test_replay_probe_b_single_run_snap_and_fold() -> None:
    # One saved run cannot run product consensus (it needs exactly 3
    # runs), so B replays as validate -> snap -> fold on a 1-of-1
    # bundle; the snap moves every edge, the fold finds nothing short.
    fixture = _fixture()
    run = validate_legacy_spark_payload(fixture["probe_b"])
    assert _scorer_score(run.steps) == 5
    refined = refine_consensus_bundle(
        _single_run_bundle(run),
        cuts=tuple(fixture["sample_cuts"]),
        snap_window_s=2.0,
        min_phase_length_s=5.0,
    )
    assert [b.snapped for b in refined.boundaries] == [True] * 7
    assert [round(b.time_s, 4) for b in refined.boundaries] == [
        17.3, 36.7667, 59.5667, 86.8333, 111.5333, 134.8, 164.6333,
    ]
    assert len(refined.steps) == 8
    assert _scorer_score(refined.steps) == 5


# --- decoded-duration preflight -------------------
# Verdict: a 240 s clip at a $0.45 job cap, decoded 273 s. Nominal
# preflight admits $0.4425 while the child can spend $0.4858125. The
# pipeline now checks the running decoded-duration total after each
# probe and before child admission (and prices retries off it),
# refusing before any upload or Spark hold.


def _held_reservation_count(database: Path) -> int:
    conn = db.connect(database)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE state = 'held'"
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _run_decoded_budget_case(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    monkeypatch,
    *,
    nominal_s: float,
    decoded_s: float,
    job_cap_usd: Decimal,
):
    """Run one live whole-video window; record admission/select/upload/call."""
    from app import pipeline as pipeline_mod

    clip = _hard_cut_clip(tmp_path / "decoded240.mp4", seconds=nominal_s)
    scripted = ScriptedTransport(
        [ok(spark_response(_tiling_payload(decoded_s)))] * 3
    )
    real_select = pipeline_mod.select_providers
    recorded: list = []

    def _recording_select(settings_arg, ctx=None, **kwargs):
        recorded.append(kwargs.get("window_s"))
        return real_select(settings_arg, ctx, transport=scripted, **kwargs)

    monkeypatch.setattr(pipeline_mod, "select_providers", _recording_select)
    monkeypatch.setattr(
        pipeline_mod,
        "probe_cut",
        lambda path, *, timeout_s: CutInfo(
            start_s=0.0, duration_s=decoded_s
        ),
    )
    settings = _live_settings(
        tmp_path, db_path=lane_db, job_cost_cap_usd=job_cap_usd
    )
    meta = _LiveMetaFake()
    clients = PipelineClients(
        spark=_LiveSparkStub(), sam=object(), meta=meta
    )
    ctx = admit_job("decoded-parent", "decoded-parent", "video-decoded-parent")
    real_admit = pipeline_mod.admit_job
    admitted: list = []

    def _recording_admit(*args, **kwargs):
        admitted.append(args[0] if args else None)
        return real_admit(*args, **kwargs)

    # Patched after the parent admission, so every recorded call is a
    # chunk child admission.
    monkeypatch.setattr(pipeline_mod, "admit_job", _recording_admit)
    result = run_stages(
        ctx, "parent-file-id", clip,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=False,
        settings=settings, duration_s=nominal_s,
    )
    return result, recorded, admitted, meta, scripted, ctx


def test_decoded_overrun_refuses_before_any_spark_hold(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # The verdict scenario: 240 s nominal admits $0.4425 under the
    # $0.45 cap, but the 273 s decoded total is $0.4858125 — refused
    # after the probe, before any child admission, upload, or hold.
    result, recorded, admitted, meta, scripted, ctx = (
        _run_decoded_budget_case(
            tmp_path, lane_db, uploads, lifecycle, monkeypatch,
            nominal_s=240.0, decoded_s=273.0,
            job_cap_usd=Decimal("0.45"),
        )
    )
    assert result.state == "failed"
    assert result.error == (
        "step extraction failed: chunk 0 decoded to 273.0 s and the"
        " $0.4858125 planned total would exceed the $0.45 job budget"
    )
    assert admitted == []  # no child admitted, so no hold exists
    assert recorded == []  # no child Spark client built
    assert meta.upload_calls == []
    assert scripted.posts == []
    assert uploads.chunk_claim_taken(f"{ctx.job_id}:chunk:0") is False
    assert _held_reservation_count(lane_db) == 0


def test_in_cap_decoded_overrun_is_admitted(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # Same nominal clip and $0.45 cap, decoded 241.5 s ($0.44446875):
    # inside the cap, so the child is admitted and the job completes.
    result, recorded, admitted, meta, scripted, _ = (
        _run_decoded_budget_case(
            tmp_path, lane_db, uploads, lifecycle, monkeypatch,
            nominal_s=240.0, decoded_s=241.5,
            job_cap_usd=Decimal("0.45"),
        )
    )
    assert result.state == "done", result.error
    assert recorded == [241.5]
    assert len(admitted) == 1
    assert len(meta.upload_calls) == 1
    assert len(scripted.posts) == 3
    assert result.guide is not None
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 120.0),
        (120.0, 240.0),
    ]


def test_retry_prices_the_decoded_planned_total(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    tmp_cache: Path,
    monkeypatch,
) -> None:
    # Retry side: nominal 214.17 s ($0.4086) with decoded 220.0 s
    # ($0.41625) at a $0.83 cap. The nominal base plus the retry
    # ($0.82485) would admit; the decoded base plus the retry
    # ($0.8325) refuses up front, before any retry upload or call.
    from test_chunk_retry import (
        _agreeing_runs,
        _disagreeing_runs,
        _legacy_spark,
    )
    from app import pipeline as pipeline_mod

    settings = _planning_settings(
        tmp_path,
        step_span_mode="legacy",
        job_cost_cap_usd=Decimal("0.83"),
        chunk_retry_budget_usd=Decimal("1.00"),
        clock_burn_in=False,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "probe_cut",
        lambda path, *, timeout_s: CutInfo(start_s=0.0, duration_s=220.0),
    )
    clip = _hard_cut_clip(tmp_path / "retry220.mp4", seconds=214.0)
    clients = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("case4-retry", "case4-retry", "video-case4-retry")
    result = run_stages(
        ctx,
        "parent-file-id",
        clip,
        clients=clients,
        lifecycle=lifecycle,
        uploads=uploads,
        replay=True,
        settings=settings,
        duration_s=214.17,
    )
    assert result.state == "failed"
    assert result.error == (
        "step extraction failed: chunk 0 disagreed and a $0.41625 retry"
        " would exceed the $0.83 job budget"
    )
    assert len(clients.spark.run_calls) == 3
    assert len(clients.meta.upload_calls) == 1
