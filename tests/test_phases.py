"""contiguous coarse repair-phase partition.

Proves the Spark prompt requires a gap-free phase partition, chunk and
global coverage are enforced before ends derive (a missing phase fails
closed instead of stretching a neighbour), editing step N's start moves
step N-1's end, downstream stages still reference the intended phase
after renumbering, and the replayed run reports its score by matched phase.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app import db, stage_cache
from app.admission import admit_job
from app.chunks import (
    DEFAULT_SPAN_MODE,
    SPAN_MODES,
    ChunkWindow,
    ContiguityError,
    derive_contiguous_ends,
    require_contiguous_cover,
    stitch_bundles,
)
from app.contracts import Step, StepBundle
from app.scorer_time import score_guide
from app.hazards import assess_hazards
from app.ledger import LifecycleRepositoryImpl
from app.pipeline import (
    PipelineClients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    run_stages,
)
from app.providers.meta_spark import SPARK_PROMPT, spark_prompt_for_clip
from app.readback import ReadbackResult
from app.reader import is_hard_stop, reader_context
from app.routes.steps import demo_bundle, render_preview_page, require_preview_contiguous
from app.service import write_sample_clip
from app.settings import (
    CONTIGUITY_TOLERANCE_ENV_VAR,
    DEFAULT_CONTIGUITY_TOLERANCE_S,
    STEP_SPAN_MODE_ENV_VAR,
    Settings,
    load_settings,
)
from app.spark import (
    ChunkExtent,
    SparkRunResult,
    SparkStageConfig,
    chunk_bundle_digest,
    consensus_bundle,
    extract_steps,
    load_chunk_bundle,
    save_chunk_bundle,
    validate_legacy_spark_payload,
    validate_spark_payload,
)
from app.uploads_store import UploadStoreImpl

from test_providers import FakeCtx, FakeClock, make_spark_client, ok, spark_response
from test_providers import ScriptedTransport
from test_spark_schema import make_payload, make_step

FIXTURES = Path(__file__).parent / "fixtures"
REPLAY = FIXTURES / "run4_replay.json"
GT_PROPOSED = FIXTURES / "sample_steps_proposed.json"

FAN_BOX = (12, 30, 140, 150)
BATTERY_BOX = (200, 90, 300, 210)


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
def tmp_cache(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "stage_cache"
    monkeypatch.setattr(stage_cache, "_CACHE_ROOT", root)
    return root


@pytest.fixture(scope="module")
def clip6(tmp_path_factory) -> Path:
    return write_sample_clip(
        tmp_path_factory.mktemp("media") / "six.mp4", seconds=6.0, fps=5
    )


@pytest.fixture(scope="module")
def clip65(tmp_path_factory) -> Path:
    return write_sample_clip(
        tmp_path_factory.mktemp("media") / "sixtyfive.mp4", seconds=65.0, fps=5
    )


def _settings(tmp_path: Path, lane_db: Path, **overrides) -> Settings:
    overrides.setdefault("pipeline_mode", "replay")
    # These tests pin multi-chunk planning (whole-video would single
    # the 65 s clips); override per test as needed.
    overrides.setdefault("whole_video_max_s", 60.0)
    return Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        **overrides,
    )


def _step(number, start, end, action="Phase", **over) -> Step:
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


def _payload_step(number, start, end, action, parts, tools=(), hazards=(), kind="repair"):
    return {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": list(tools),
        "parts": [
            {"phrase": phrase, "position_hint": hint} for phrase, hint in parts
        ],
        "hazards": list(hazards),
        "kind": kind,
    }


def _run(steps, *, cost="0.08", device="laptop", conf=0.82) -> SparkRunResult:
    return SparkRunResult(
        payload={"device_guess": device, "confidence": conf, "steps": steps},
        cost_usd=Decimal(cost),
        latency_ms=900,
    )


def _tiling(steps, duration: float) -> None:
    """Assert steps tile [0, duration): start_1=0, seams meet, end covers."""
    assert steps[0].start_s == 0.0
    assert steps[-1].end_s == pytest.approx(duration)
    for left, right in zip(steps, steps[1:]):
        assert left.end_s == right.start_s


# --- cover enforcement and end derivation ------------------------------------


def test_derive_enforces_then_derives() -> None:
    steps = (
        _step(1, 0.5, 20.0, "Introduce laptop", tools=("Mallet",)),
        _step(2, 20.0, 59.0, "Remove screws"),
    )
    derived = derive_contiguous_ends(steps, 60.0, tolerance_s=2.0)
    assert [(s.start_s, s.end_s) for s in derived] == [(0.0, 20.0), (20.0, 60.0)]
    assert [s.number for s in derived] == [1, 2]
    assert [s.action for s in derived] == ["Introduce laptop", "Remove screws"]
    assert derived[0].tools == ("Mallet",)  # non-time fields kept verbatim
    assert [s.time_spread_s for s in derived] == [0.1, 0.1]


def test_missing_middle_phase_fails_closed() -> None:
    steps = (
        _step(1, 0.0, 10.0, "Introduce laptop"),
        _step(2, 30.0, 60.0, "Remove screws"),
    )
    with pytest.raises(ContiguityError, match="a phase is missing"):
        derive_contiguous_ends(steps, 60.0, tolerance_s=2.0)


def test_cover_rejects_each_violation() -> None:
    extent, tol = 60.0, 2.0
    cases = [
        ([_step(1, -3.0, 60.0)], "outside"),  # start before the chunk
        ([_step(1, 0.0, 30.0), _step(2, 63.0, 64.0)], "outside"),  # past end
        ([_step(1, 0.0, 30.0), _step(2, 30.0, 30.0)], "no duration"),
        ([_step(1, 10.0, 30.0), _step(2, 30.0, 60.0)], "opening"),  # no intro
        ([_step(1, 0.0, 30.0), _step(2, 30.0, 50.0)], "uncovered"),  # short tail
        ([_step(1, 0.0, 30.0), _step(2, 30.0, 63.0)], "past the"),  # long tail
        ([_step(1, 0.0, 30.0), _step(2, 30.0, 40.0), _step(3, 30.0, 60.0)],
         "not after"),  # non-monotonic starts
        ([_step(1, 0.0, 10.0), _step(2, 30.0, 60.0)], "missing"),  # gap
        ([_step(1, 0.0, 40.0), _step(2, 30.0, 60.0)], "misplaced"),  # overlap
        ([], "at least one step"),
    ]
    for steps, match in cases:
        with pytest.raises(ContiguityError, match=match):
            require_contiguous_cover(steps, extent, tolerance_s=tol)


def test_tolerance_boundary_passes() -> None:
    # A seam miss of exactly the tolerance is absorbed, not failed.
    steps = (
        _step(1, 2.0, 28.0, "Introduce laptop"),
        _step(2, 30.0, 58.0, "Remove screws"),
    )
    require_contiguous_cover(steps, 60.0, tolerance_s=2.0)
    derived = derive_contiguous_ends(steps, 60.0, tolerance_s=2.0)
    _tiling(derived, 60.0)


def test_single_step_derives_full_extent() -> None:
    (only,) = derive_contiguous_ends(
        [_step(1, 0.5, 59.5, "Whole repair")], 60.0, tolerance_s=2.0
    )
    assert (only.start_s, only.end_s) == (0.0, 60.0)


def test_cover_rejects_bad_extent_and_tolerance() -> None:
    steps = [_step(1, 0.0, 60.0)]
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="extent_s"):
            require_contiguous_cover(steps, bad, tolerance_s=2.0)  # type: ignore[arg-type]
    for bad in (-1.0, float("nan"), True):
        with pytest.raises(ValueError, match="tolerance_s"):
            require_contiguous_cover(steps, 60.0, tolerance_s=bad)  # type: ignore[arg-type]


# --- settings: span mode and tolerance ---------------------------------------


def test_span_settings_defaults() -> None:
    assert SPAN_MODES == ("contiguous", "legacy")
    assert DEFAULT_SPAN_MODE == "contiguous"
    assert load_settings({}).step_span_mode == "contiguous"
    assert DEFAULT_CONTIGUITY_TOLERANCE_S == 2.0
    assert load_settings({}).contiguity_tolerance_s == 2.0


def test_span_settings_from_env() -> None:
    tuned = load_settings(
        {STEP_SPAN_MODE_ENV_VAR: "legacy", CONTIGUITY_TOLERANCE_ENV_VAR: "0.5"}
    )
    assert tuned.step_span_mode == "legacy"
    assert tuned.contiguity_tolerance_s == 0.5


def test_misspelled_span_mode_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="PARTFIDELITY_STEP_SPAN_MODE"):
        load_settings({STEP_SPAN_MODE_ENV_VAR: "contigous"})
    with pytest.raises(ValueError, match="step_span_mode"):
        Settings(
            data_root=tmp_path, db_path=tmp_path / "t.sqlite3",
            production=False, step_span_mode="contigous",
        )
    with pytest.raises(ValueError, match="span_mode"):
        SparkStageConfig(span_mode="contigous")
    with pytest.raises(ValueError, match="contiguity_tolerance_s"):
        Settings(
            data_root=tmp_path, db_path=tmp_path / "t.sqlite3",
            production=False, contiguity_tolerance_s=-1.0,
        )
    with pytest.raises(RuntimeError, match="PARTFIDELITY_CONTIGUITY_TOLERANCE_S"):
        load_settings({CONTIGUITY_TOLERANCE_ENV_VAR: "soon"})


def test_span_mode_change_invalidates_chunk_bundle(tmp_cache: Path) -> None:
    kwargs = {
        "parent_job_id": "parent-1",
        "source_hash": "aaa",
        "index": 0,
        "window_start_s": 0.0,
        "window_end_s": 60.0,
        "cut_start_s": 0.0,
        "cut_duration_s": 60.0,
    }
    legacy = chunk_bundle_digest(
        **kwargs, config=SparkStageConfig(span_mode="legacy")
    )
    current = chunk_bundle_digest(**kwargs)
    assert legacy != current
    bundle = StepBundle(
        device_guess="laptop", confidence=0.8, steps=(_step(1, 0.0, 60.0),),
        source_runs=3, timestamps_approximate=True,
    )
    save_chunk_bundle(
        legacy, bundle, cut_start_s=0.0, cut_duration_s=60.0
    )
    assert load_chunk_bundle(legacy, ttl_s=60.0) is not None
    assert load_chunk_bundle(current, ttl_s=60.0) is None  # other mode misses


# --- prompt: one contiguous-partition contract --------------------------------


def test_prompt_requires_contiguous_partition() -> None:
    assert "Partition the whole clip" in SPARK_PROMPT
    assert "coarse repair phases" in SPARK_PROMPT
    assert "transition moments" in SPARK_PROMPT
    assert "never emit those as standalone steps" in SPARK_PROMPT
    assert "no gaps and no overlaps" in SPARK_PROMPT
    assert "the first phase starts at 0" in SPARK_PROMPT
    assert "each phase ends where the next begins" in SPARK_PROMPT
    assert "the last phase ends at the end of the clip" in SPARK_PROMPT
    assert "(at most 12)" in SPARK_PROMPT


def test_live_client_sends_the_one_prompt_text() -> None:
    # The prompt text lives in one place: the wire body carries the
    # constant verbatim, so no second copy can drift.
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport(
        [ok(spark_response({"device_guess": "x", "confidence": 1, "steps": []}))]
    )
    make_spark_client(ctx, transport, clock).run_steps("file-1")
    assert len(transport.posts) == 1
    body = json.loads(transport.posts[0]["body"].decode("utf-8"))
    sent = body["input"][0]["content"][0]["text"]
    assert sent == SPARK_PROMPT


def test_duration_prompt_states_the_exact_clip_length() -> None:
    prompt = spark_prompt_for_clip(60.0)
    assert prompt.startswith(SPARK_PROMPT)  # extends the one text, never copies
    assert "exactly 60 seconds" in prompt
    assert "[0, 60]" in prompt
    assert "the last phase must end at 60" in prompt
    short = spark_prompt_for_clip(34.5)  # the duration comes from the chunk
    assert "exactly 34.5 seconds" in short
    assert "[0, 34.5]" in short
    assert "60" not in short


def test_duration_prompt_rejects_misshaped_durations() -> None:
    for bad in (0, -1.0, True, "60", None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            spark_prompt_for_clip(bad)  # type: ignore[arg-type]


def test_live_client_sends_the_duration_prompt() -> None:
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport(
        [ok(spark_response({"device_guess": "x", "confidence": 1, "steps": []}))]
    )
    make_spark_client(ctx, transport, clock).run_steps(
        "file-1", clip_duration_s=60.0
    )
    assert len(transport.posts) == 1
    body = json.loads(transport.posts[0]["body"].decode("utf-8"))
    sent = body["input"][0]["content"][0]["text"]
    assert sent == spark_prompt_for_clip(60.0)


@pytest.mark.parametrize("mode", ["legacy", "exact"])
def test_legacy_and_exact_modes_send_the_base_prompt_byte_for_byte(
    mode: str,
) -> None:
    """Rollback-path wire identity: with a real extent passed, legacy
    and exact extraction still put exactly SPARK_PROMPT on the wire."""
    config = (
        SparkStageConfig(span_mode="legacy")
        if mode == "legacy"
        else SparkStageConfig(alignment_mode="exact")
    )
    steps = [
        make_step(start=0.0, end=30.0),
        make_step(start=30.0, end=60.0),
    ]
    payload = make_payload(steps=steps)
    script = [ok(spark_response(payload)) for _ in range(config.runs)]
    ctx, clock = FakeCtx(), FakeClock()
    transport = ScriptedTransport(script)
    client = make_spark_client(ctx, transport, clock)
    client.spark_stage_config = config
    extract_steps(
        ctx, "file-1", client=client,
        chunk_extent=ChunkExtent(duration_s=60.0, tolerance_s=2.0),
    )
    assert len(transport.posts) == config.runs
    for post in transport.posts:
        body = json.loads(post["body"].decode("utf-8"))
        assert body["input"][0]["content"][0]["text"] == SPARK_PROMPT


# --- builder check 1: a 60 s chunk becomes a gap-free [0, 60) partition ------


def _covering_runs(windows: list[tuple[float, float]]) -> list:
    """Three varied runs tiling a chunk with distinct phase actions/parts."""
    phases = [
        ("Introduce laptop on table", [("laptop lid", "center")]),
        ("Remove the rear screws", [("screw", "rear edge")]),
        ("Disconnect the battery cable", [("battery cable", "lower right")]),
    ]
    assert len(windows) == len(phases)
    runs = []
    for drift in (0.0, 0.5, 0.25):
        runs.append(
            _run(
                [
                    _payload_step(
                        number, start + drift, end + drift, action, parts
                    )
                    for number, ((start, end), (action, parts)) in enumerate(
                        zip(windows, phases), start=1
                    )
                ]
            )
        )
    return runs


def test_three_run_60s_fixture_partitions_gap_free() -> None:
    runs = [
        validate_spark_payload(result.payload)
        for result in _covering_runs([(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)])
    ]
    bundle = consensus_bundle(runs)
    assert [s.action for s in bundle.steps] == [
        "Introduce laptop on table",
        "Remove the rear screws",
        "Disconnect the battery cable",
    ]
    derived = derive_contiguous_ends(
        bundle.steps, 60.0, tolerance_s=DEFAULT_CONTIGUITY_TOLERANCE_S
    )
    _tiling(derived, 60.0)


def test_single_chunk_pipeline_partitions_gap_free(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip6: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)  # default: contiguous
    spark = ReplaySparkClient(
        _covering_runs([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)])
    )
    # Time consensus resolves phases wider than the boundary
    # tolerance: these 2 s miniature phases need a sub-2 s line
    # (the 5 s default would merge their edges into one boundary).
    spark.spark_stage_config = SparkStageConfig(boundary_tolerance_s=1.0)
    clients = PipelineClients(
        spark=spark,
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("phases-1", "phases-1", "video-phases-1")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=6.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    _tiling(result.guide.steps, 6.0)
    assert [s.number for s in result.guide.steps] == [1, 2, 3]


# --- builder check 2: a missing middle phase fails closed --------------------


def _two_chunk_clients(sparse_second: bool) -> PipelineClients:
    chunk0 = [
        _run(
            [
                _payload_step(1, 0.0, 30.0, "Introduce laptop on table",
                              [("laptop lid", "center")]),
                _payload_step(2, 30.0, 60.0, "Remove the rear screws",
                              [("screw", "rear edge")]),
            ]
        )
        for _ in range(3)
    ]
    if sparse_second:
        chunk1 = [
            _run(
                [_payload_step(1, 0.5, 2.5, "Show screwdriver tool",
                                [("screwdriver", "center")])]
            )
            for _ in range(6)  # initial attempt plus the one bounded retry
        ]
    else:
        chunk1 = [
            _run(
                [_payload_step(1, 0.0, 7.0, "Disconnect the battery cable",
                                [("battery", "lower right")],
                                hazards=("battery",))]
            )
            for _ in range(3)
        ]
    return PipelineClients(
        spark=ReplaySparkClient(
            chunk0 + chunk1,
            readbacks={
                "battery": ReadbackResult(
                    answer="yes", cost_usd=Decimal("0.04"), latency_ms=120
                )
            },
        ),
        sam=ReplaySamClient({"battery": (BATTERY_BOX, [0.61, 0.55])}),
        meta=ReplayMetaFilesClient(),
    )


def test_chunk_missing_phase_retries_once_then_fails_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients(sparse_second=True)
    ctx = admit_job("phases-2", "phases-2", "video-phases-2")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed", result.guide
    assert result.guide is None
    assert result.error == "step extraction failed"
    # Chunk 0 extracted once; chunk 1 ran its initial attempt plus the
    # one bounded retry, then gave up without stretching a neighbour.
    assert len(clients.spark.run_calls) == 9


def test_legacy_span_mode_keeps_sparse_spans(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, step_span_mode="legacy")
    clients = _two_chunk_clients(sparse_second=True)
    # Honest legacy extraction: without a stage config the replay
    # client would run the contiguous time consensus (whose known
    # extent now excludes these sparse fragments outright).
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("phases-legacy", "phases-legacy", "video-phases-legacy")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    # Rollback behaviour: model spans pass through untouched, gaps kept.
    assert [(s.start_s, s.end_s) for s in result.guide.steps] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (58.5, 60.5),
    ]
    assert len(clients.spark.run_calls) == 6  # no retry: nothing enforced


# --- builder check 4: downstream stages track the renumbered phases ----------


def test_chunked_pipeline_downstream_tracks_phases(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients(sparse_second=False)
    ctx = admit_job("phases-4", "phases-4", "video-phases-4")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    guide = result.guide
    steps = guide.steps
    # Stitched then derived: chunk 1's head meets chunk 0's tail within
    # tolerance and the guide tiles the whole video.
    _tiling(steps, 65.0)
    assert [s.number for s in steps] == [1, 2, 3]
    assert steps[2].action == "Disconnect the battery cable"

    # Frame sampling: one kept frame per phase, in step order.
    assert len(guide.frame_paths) == len(steps)

    # SAM slot selection and labels: every kept label names its own
    # phase's part, and its image follows the configured slot pattern.
    assert clients.sam.calls, "battery phase must reach SAM"
    kept = [d for d in guide.labels if d.provisional is not None]
    assert kept, "battery label must be kept"
    for decision in kept:
        phase = steps[decision.slot.step_index]
        assert phase.number == decision.slot.step_index + 1
        assert decision.slot.part in phase.parts
        assert decision.provisional is not None
        assert decision.provisional.rendered_image_path.name == (
            settings.label_image_name(
                step=phase.number, part=decision.slot.part_index
            )
        )

    # Hazards: the battery phase keeps its stop, numbered at its phase.
    assessment = assess_hazards(tuple(steps), "laptop")
    assert not assessment.refused
    stops = [s for s in steps if is_hard_stop(s)]
    assert [s.number for s in stops] == [3]
    assert guide.difficulty in ("Easy", "Medium", "Hard")

    # Reader cards: same phases, same windows, same labels, stop visible.
    context = reader_context(guide, set())
    cards = context["steps"]
    assert [(c["number"], c["start_s"], c["end_s"]) for c in cards] == [
        (s.number, s.start_s, s.end_s) for s in steps
    ]
    assert [c["number"] for c in cards if c["stop"]] == [3]
    assert context["stop_step"] == 3
    assert context["labels_unmatched"] == []


# --- builder check 5: the replayed run reports its score by matched phase --------------


def _replay_fixture() -> dict:
    return json.loads(REPLAY.read_text(encoding="utf-8"))


def _gt_proposed() -> list[dict]:
    data = json.loads(GT_PROPOSED.read_text(encoding="utf-8"))
    assert data["status"] == "proposed"
    return data["steps"]


def _gt_excluded() -> list[tuple[float, float]]:
    data = json.loads(GT_PROPOSED.read_text(encoding="utf-8"))
    return [(span["start_s"], span["end_s"]) for span in data["excluded_spans"]]


def _replay_stitched(mode: str):
    fixture = _replay_fixture()
    config = SparkStageConfig(alignment_mode=mode)
    bundles = []
    for chunk in fixture["chunks"]:
        runs = [validate_legacy_spark_payload(payload) for payload in chunk["runs"]]
        bundles.append(
            (
                ChunkWindow(chunk["window"][0], chunk["window"][1]),
                consensus_bundle(runs, config),
            )
        )
    return fixture, stitch_bundles(bundles, overlap_s=2.0, alignment_mode=mode)


def test_run4_replay_fails_contiguity_and_reports_scorer() -> None:
    # Offline replay cannot show how Spark responds to the new prompt
    # (that needs the paid check): it proves the old sparse output now
    # fails the new gate closed, and reports where that guide stands.
    fixture, stitched = _replay_stitched("exact")
    with pytest.raises(ContiguityError):
        derive_contiguous_ends(
            stitched.steps, fixture["video_duration_s"],
            tolerance_s=DEFAULT_CONTIGUITY_TOLERANCE_S,
        )
    _, aligned = _replay_stitched("align")
    with pytest.raises(ContiguityError):
        derive_contiguous_ends(
            aligned.steps, fixture["video_duration_s"],
            tolerance_s=DEFAULT_CONTIGUITY_TOLERANCE_S,
        )
    report = score_guide(stitched.steps, _gt_proposed(), excluded_spans=_gt_excluded())
    # Time alignment: pairs by overlap, so the
    # one real timing hit scores 1/9 strict; the bar is still 7/9.
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [
        (1, 2),
        (2, 3),
        (3, 5),
        (4, 7),
        (5, 8),
        (6, 9),
    ]
    assert report.missing_gt == (1, 4, 6)
    assert report.extra_guide == ()
    assert report.coverage == pytest.approx(6 / 9)
    assert report.boundary_accuracy == pytest.approx(1 / 6)
    assert report.strict_passes == 1
    assert report.passed is False


# --- builder check 3 (server side): the preview harness stays contiguous ----


def test_demo_bundle_tiles_and_sparse_preview_refused() -> None:
    demo = demo_bundle()
    _tiling(demo.steps, 150.0)
    page = render_preview_page(demo)
    assert page.count("data-set-start") == 2
    sparse = StepBundle(
        device_guess=demo.device_guess,
        confidence=demo.confidence,
        steps=(
            demo.steps[0],
            Step(
                number=2,
                start_s=95.0,
                end_s=150.0,
                action=demo.steps[1].action,
                tools=demo.steps[1].tools,
                parts=demo.steps[1].parts,
                hazards=demo.steps[1].hazards,
                time_spread_s=demo.steps[1].time_spread_s,
                time_needs_check=demo.steps[1].time_needs_check,
            ),
        ),
        source_runs=demo.source_runs,
        timestamps_approximate=True,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        require_preview_contiguous(sparse)
    with pytest.raises(ValueError, match="not contiguous"):
        render_preview_page(sparse)
