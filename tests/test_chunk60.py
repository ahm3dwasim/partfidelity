"""long footage in 60-second chunks."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from app import db
from app.admission import admit_job, job_dir
from app.chunks import (
    CHUNK_ARTIFACT_KIND,
    ChunkCutError,
    ChunkWindow,
    CutInfo,
    chunk_filename,
    cut_chunk,
    plan_chunks,
    probe_cut,
    real_cut_window,
    stitch_bundles,
)
from app.contracts import (
    JOB_DEADLINE_S,
    MetaFilesError,
    PartRequest,
    Step,
    StepBundle,
)
from app.ledger import BudgetExhausted, LifecycleRepositoryImpl
from app.pipeline import (
    PipelineClients,
    _fail,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    REFUSAL_MARKER_FILENAME,
    STEP_EXTRACTION_OVER_BUDGET,
    run_job,
    run_stages,
)
from app.readback import ReadbackResult
from app.service import write_sample_clip
from app.settings import Settings, load_settings
from app.spark import SparkRunResult, SparkStageConfig
from app.uploads import (
    OverLimitError,
    check_upload_bounds,
    over_limit_message,
    prepare_preview,
)
from app.uploads_store import UploadStoreImpl

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


@pytest.fixture(scope="module")
def clip61(tmp_path_factory) -> Path:
    return write_sample_clip(
        tmp_path_factory.mktemp("media") / "sixtyone.mp4", seconds=61.0, fps=5
    )


def _settings(tmp_path: Path, lane_db: Path, **overrides) -> Settings:
    # Sparse scripted runs predate the contiguous-phase contract: these
    # tests pin the legacy rollback mode (override per test as needed).
    overrides.setdefault("step_span_mode", "legacy")
    # These tests pin multi-chunk planning (whole-video would single
    # the 61/65 s clips); override per test as needed.
    overrides.setdefault("whole_video_max_s", 60.0)
    return Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        **overrides,
    )


def _jobs(database: Path) -> list[sqlite3.Row]:
    conn = db.connect(database)
    try:
        return conn.execute(
            "SELECT id, state FROM jobs ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()


def _settled_spark(database: Path, job_id: str) -> list[Decimal]:
    conn = db.connect(database)
    try:
        rows = conn.execute(
            "SELECT actual_usd FROM reservations"
            " WHERE job_id = ? AND call_kind = 'spark_steps'"
            " AND state IN ('settled', 'failed')",
            (job_id,),
        ).fetchall()
        return [Decimal(row["actual_usd"]) for row in rows]
    finally:
        conn.close()


def _artifacts(database: Path, job_id: str) -> list[sqlite3.Row]:
    conn = db.connect(database)
    try:
        return conn.execute(
            "SELECT kind, relative_path, delete_state FROM artifacts"
            " WHERE job_id = ? ORDER BY relative_path",
            (job_id,),
        ).fetchall()
    finally:
        conn.close()


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


def _step(number, start, end, action, parts=(), **over) -> Step:
    fields = {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": (),
        "parts": tuple(
            PartRequest(phrase=phrase, position_hint=hint) for phrase, hint in parts
        ),
        "hazards": (),
        "time_spread_s": 0.1,
        "time_needs_check": False,
    }
    fields.update(over)
    return Step(**fields)


def _bundle(steps, *, device="laptop", conf=0.82) -> StepBundle:
    return StepBundle(
        device_guess=device,
        confidence=conf,
        steps=tuple(steps),
        source_runs=3,
        timestamps_approximate=True,
    )


def _sam_scores(n: int, winner: float = 0.82, runner_up: float = 0.60) -> list[float]:
    """Alternating decisive winner/runner-up: ties choose nothing."""
    return [winner if i % 2 == 0 else runner_up for i in range(n)]


# --- planning: boundaries and overlap -------------------------------------


@pytest.mark.parametrize(
    "duration, expected",
    [
        (6.0, [(0.0, 6.0)]),
        (59.9, [(0.0, 59.9)]),
        (60.0, [(0.0, 60.0)]),
        (60.1, [(0.0, 60.0), (58.0, 60.1)]),
        (61.0, [(0.0, 60.0), (58.0, 61.0)]),
        (65.0, [(0.0, 60.0), (58.0, 65.0)]),
        (
            214.0,
            [(0.0, 60.0), (58.0, 118.0), (116.0, 176.0), (174.0, 214.0)],
        ),
        (
            300.0,
            [
                (0.0, 60.0),
                (58.0, 118.0),
                (116.0, 176.0),
                (174.0, 234.0),
                (232.0, 292.0),
                (290.0, 300.0),
            ],
        ),
    ],
)
def test_plan_chunks_boundaries(duration: float, expected) -> None:
    windows = plan_chunks(duration, 60.0, 2.0)
    assert [(w.start_s, w.end_s) for w in windows] == expected
    assert windows[0].start_s == 0.0
    assert windows[-1].end_s == pytest.approx(duration)
    for first, second in zip(windows, windows[1:]):
        assert first.end_s - second.start_s == pytest.approx(2.0)


@pytest.mark.parametrize(
    "duration, chunk, overlap",
    [
        (0.0, 60.0, 2.0),
        (-1.0, 60.0, 2.0),
        (float("nan"), 60.0, 2.0),
        (65.0, 0.0, 2.0),
        (65.0, 60.0, -0.5),
        (65.0, 60.0, 60.0),
        (65.0, 60.0, 61.0),
    ],
)
def test_plan_chunks_rejects_bad_inputs(duration, chunk, overlap) -> None:
    with pytest.raises(ValueError):
        plan_chunks(duration, chunk, overlap)


def test_chunk_filename() -> None:
    assert chunk_filename(0) == "chunk_000.mp4"
    assert chunk_filename(12) == "chunk_012.mp4"
    with pytest.raises(ValueError):
        chunk_filename(-1)


# --- cutting ---------------------------------------------------------------


def test_cut_chunk_round_trip(tmp_path: Path, clip6: Path) -> None:
    from app.uploads import _probe_media

    dst = tmp_path / "cut.mp4"
    assert cut_chunk(clip6, ChunkWindow(1.0, 3.0), dst, timeout_s=60.0) == dst
    assert dst.is_file()
    assert _probe_media(dst).duration_s == pytest.approx(2.0, abs=0.6)


def test_cut_chunk_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ChunkCutError):
        cut_chunk(
            tmp_path / "absent.mp4",
            ChunkWindow(0.0, 5.0),
            tmp_path / "out.mp4",
            timeout_s=60.0,
        )


def test_probe_cut_reports_decoded_duration_and_start(
    tmp_path: Path, clip6: Path
) -> None:
    dst = tmp_path / "cut.mp4"
    cut_chunk(clip6, ChunkWindow(1.0, 3.0), dst, timeout_s=60.0)
    assert probe_cut(dst, timeout_s=60.0) == CutInfo(
        start_s=0.0, duration_s=2.0
    )


def test_probe_cut_reads_a_nonzero_start(tmp_path: Path) -> None:
    # An offset source proves the probe reads start_time (the real-start
    # input); the pipeline's own mp4 cuts always normalize it to 0.
    src = tmp_path / "off.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=5",
            "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
            "-output_ts_offset", "10", str(src),
        ],
        check=True, timeout=300,
    )
    assert probe_cut(src, timeout_s=60.0) == CutInfo(
        start_s=10.0, duration_s=2.0
    )


def test_probe_cut_refuses_unreadable_bytes(tmp_path: Path) -> None:
    with pytest.raises(ChunkCutError):
        probe_cut(tmp_path / "absent.mp4", timeout_s=60.0)
    garbage = tmp_path / "garbage.mp4"
    garbage.write_text("not a video", encoding="utf-8")
    with pytest.raises(ChunkCutError):
        probe_cut(garbage, timeout_s=60.0)


def test_cut_info_rejects_misshaped_values() -> None:
    for bad in (True, "0", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CutInfo(start_s=bad, duration_s=1.0)  # type: ignore[arg-type]
    for bad in (0, -1.0, True, "1", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CutInfo(start_s=0.0, duration_s=bad)  # type: ignore[arg-type]


def test_real_cut_window_maps_with_the_real_start() -> None:
    window = ChunkWindow(58.0, 65.0)
    assert real_cut_window(window, CutInfo(start_s=0.0, duration_s=7.0)) == window
    assert real_cut_window(
        window, CutInfo(start_s=-3.0, duration_s=7.0)
    ) == ChunkWindow(55.0, 62.0)
    assert real_cut_window(
        window, CutInfo(start_s=0.5, duration_s=7.0)
    ) == ChunkWindow(58.5, 65.5)
    with pytest.raises(ChunkCutError):
        real_cut_window(
            ChunkWindow(1.0, 3.0), CutInfo(start_s=-3.0, duration_s=2.0)
        )
    with pytest.raises(ValueError):
        real_cut_window("58-65", CutInfo(start_s=0.0, duration_s=7.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        real_cut_window(window, (0.0, 7.0))  # type: ignore[arg-type]


# --- stitching: offsets, merge, dedupe -------------------------------------


def test_stitch_offsets_and_renumbers() -> None:
    stitched = stitch_bundles(
        [
            (
                ChunkWindow(0.0, 60.0),
                _bundle([_step(1, 0.5, 2.5, "Remove the rear screws",
                               [("fan", "left")])]),
            ),
            (
                ChunkWindow(58.0, 65.0),
                _bundle([_step(1, 3.0, 5.5, "Disconnect the battery cable",
                               [("battery", "right")], hazards=("battery",))]),
            ),
        ],
        overlap_s=2.0,
    )
    assert [(s.number, s.start_s, s.end_s) for s in stitched.steps] == [
        (1, 0.5, 2.5),
        (2, 61.0, 63.5),
    ]
    assert stitched.steps[1].parts[0].phrase == "battery"
    assert stitched.device_guess == "laptop"


def test_stitch_merges_boundary_spanning_step() -> None:
    stitched = stitch_bundles(
        [
            (
                ChunkWindow(0.0, 60.0),
                _bundle([_step(2, 55.0, 59.5, "Pry the cover",
                               [("spudger", "edge")], time_spread_s=0.2)]),
            ),
            (
                ChunkWindow(58.0, 118.0),
                _bundle([_step(1, 0.5, 4.0, "pry  the COVER",
                               [("spudger", "other edge")],
                               time_spread_s=0.4, time_needs_check=True)]),
            ),
        ],
        overlap_s=2.0,
    )
    (merged,) = stitched.steps
    assert merged.number == 1
    assert (merged.start_s, merged.end_s) == (55.0, 62.0)
    assert merged.action == "Pry the cover"  # earlier sighting wins
    assert merged.time_spread_s == pytest.approx(0.4)
    assert merged.time_needs_check is True


def test_stitch_dedupes_overlap_sighting() -> None:
    step = _step(1, 58.5, 59.5, "Lift the lid", [("fan", "left")])
    stitched = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0), _bundle([step])),
            (
                ChunkWindow(58.0, 118.0),
                _bundle([_step(1, 0.5, 1.5, "Lift the lid",
                               [("fan", "left")])]),
            ),
        ],
        overlap_s=2.0,
    )
    assert len(stitched.steps) == 1
    assert (stitched.steps[0].start_s, stitched.steps[0].end_s) == (58.5, 59.5)


def test_stitch_flags_rival_windows() -> None:
    stitched = stitch_bundles(
        [
            (
                ChunkWindow(0.0, 60.0),
                _bundle([_step(1, 54.0, 59.0, "Unscrew the base",
                               [("fan", "left")])]),
            ),
            (
                ChunkWindow(58.0, 118.0),
                _bundle([_step(1, 0.0, 1.5, "Unplug the display",
                               [("battery", "right")])]),
            ),
        ],
        overlap_s=2.0,
    )
    assert len(stitched.steps) == 2
    assert [s.time_needs_check for s in stitched.steps] == [True, True]


def test_stitch_device_majority_and_median_confidence() -> None:
    stitched = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0),
             _bundle([_step(1, 1.0, 2.0, "First")], device="laptop", conf=0.9)),
            (ChunkWindow(58.0, 118.0),
             _bundle([_step(1, 10.0, 12.0, "Second")], device="tablet", conf=0.7)),
            (ChunkWindow(116.0, 176.0),
             _bundle([_step(1, 20.0, 22.0, "Third")], device="tablet", conf=0.8)),
        ],
        overlap_s=2.0,
    )
    assert stitched.device_guess == "tablet"
    assert stitched.confidence == pytest.approx(0.8)


def test_stitch_rejects_empty() -> None:
    with pytest.raises(ValueError):
        stitch_bundles([], overlap_s=2.0)


def test_stitch_keeps_distant_same_key_steps() -> None:
    stitched = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0),
             _bundle([_step(1, 5.0, 8.0, "Remove a screw", [("fan", "left")])])),
            (ChunkWindow(58.0, 118.0),
             _bundle([_step(1, 50.0, 53.0, "Remove a screw", [("fan", "left")])])),
        ],
        overlap_s=2.0,
    )
    assert [(s.start_s, s.end_s) for s in stitched.steps] == [
        (5.0, 8.0),
        (108.0, 111.0),
    ]


def test_stitch_device_tie_prefers_earliest_chunk() -> None:
    stitched = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0),
             _bundle([_step(1, 1.0, 2.0, "First")], device="laptop")),
            (ChunkWindow(58.0, 118.0),
             _bundle([_step(1, 10.0, 12.0, "Second")], device="tablet")),
        ],
        overlap_s=2.0,
    )
    assert stitched.device_guess == "laptop"


# --- pipeline: per-chunk extraction and stitching ----------------------------


def _two_chunk_runs() -> list[SparkRunResult]:
    """Six scripted runs: a fan step for chunk 0, a battery step for chunk 1."""
    runs = []
    for start, end in ((0.5, 2.5), (0.6, 2.6), (0.4, 2.4)):
        runs.append(
            _run(
                [_payload_step(1, start, end, "Remove the rear screws",
                                [("fan", "left side")], tools=("Phillips #0",))],
                cost="0.08",
            )
        )
    for start, end in ((3.0, 5.5), (3.1, 5.6), (2.9, 5.4)):
        runs.append(
            _run(
                [_payload_step(1, start, end, "Disconnect the battery cable",
                                [("battery", "lower right")],
                                hazards=("battery",))],
                cost="0.07",
            )
        )
    return runs


def _two_chunk_clients() -> PipelineClients:
    return PipelineClients(
        spark=ReplaySparkClient(
            _two_chunk_runs(),
            readbacks={
                "battery": ReadbackResult(
                    answer="yes", cost_usd=Decimal("0.04"), latency_ms=120
                )
            },
        ),
        sam=ReplaySamClient(
            {
                "fan": (FAN_BOX, [0.72, 0.60]),
                "battery": (BATTERY_BOX, [0.61, 0.55]),
            }
        ),
        meta=ReplayMetaFilesClient(),
    )


def test_chunked_pipeline_stitches_replay(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy wording consensus, matching this file's legacy span
    # settings: the drifted medians below pin the sparse path.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60", "chunk60", "video-chunk60")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    steps = result.guide.steps
    assert [s.number for s in steps] == [1, 2]
    assert steps[0].start_s == pytest.approx(0.5)
    assert steps[1].start_s == pytest.approx(61.0)  # 58 s offset
    assert steps[1].end_s == pytest.approx(63.5)
    assert [d.readback for d in result.guide.labels] == ["not_run", "yes"]

    # Each chunk ran its own 3-run extraction against its own Meta file.
    assert len(clients.spark.run_calls) == 6
    assert len(set(clients.spark.run_calls)) == 2
    assert "parent-file-id" not in clients.spark.run_calls
    assert len(clients.meta.upload_calls) == 2
    assert len(clients.meta.delete_calls) == 2
    for file_id in clients.meta.delete_calls:
        assert clients.meta.get_file(file_id) is None

    # One parent plus two settled sub-jobs; Spark money sits on the subs.
    jobs = _jobs(lane_db)
    assert len(jobs) == 3
    assert {row["state"] for row in jobs} == {"done"}
    parent_id = str(ctx.job_id)
    # Each chunk claimed before upload and attached after: the sub-jobs
    # hold confirmed upload rows keyed by their idempotency keys.
    conn = db.connect(lane_db)
    try:
        upload_rows = conn.execute(
            "SELECT state, claim_id, meta_file_id FROM uploads"
            " WHERE job_id != ?",
            (parent_id,),
        ).fetchall()
    finally:
        conn.close()
    assert len(upload_rows) == 2
    assert {row["claim_id"] for row in upload_rows} == {
        f"{parent_id}:chunk:0", f"{parent_id}:chunk:1",
    }
    for upload_row in upload_rows:
        assert upload_row["state"] == "confirmed"
        assert upload_row["meta_file_id"] is not None
    assert _settled_spark(lane_db, parent_id) == []
    sub_ids = [str(row["id"]) for row in jobs if str(row["id"]) != parent_id]
    assert len(sub_ids) == 2
    for sub_id in sub_ids:
        assert len(_settled_spark(lane_db, sub_id)) == 3

    # Chunk working files are tombstoned provisionals under the parent job.
    artifacts = [
        row for row in _artifacts(lane_db, parent_id)
        if row["kind"] == "provisional"
    ]
    assert [row["delete_state"] for row in artifacts] == [
        "pending_delete",
        "pending_delete",
    ]
    names = sorted(Path(row["relative_path"]).name for row in artifacts)
    assert names == ["chunk_000.mp4", "chunk_001.mp4"]


def test_chunked_replay_children_keep_base_deadline(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    """Replay runs never extend child deadlines; live mode owns that."""
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60-replay-dl", "chunk60-replay-dl", "video-replay-dl")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    conn = db.connect(lane_db)
    try:
        children = conn.execute(
            "SELECT created_at, deadline_at FROM jobs WHERE anon_id LIKE ?",
            (f"{ctx.job_id}/chunk/%",),
        ).fetchall()
    finally:
        conn.close()
    assert len(children) == 2
    for child in children:
        envelope = db.parse_dt(child["deadline_at"]) - db.parse_dt(
            child["created_at"]
        )
        assert envelope == timedelta(seconds=JOB_DEADLINE_S)


def test_upfront_budget_refusal_makes_no_side_effect(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    settings = _settings(
        # Below 2 x $0.2475 window estimates, so preflight refuses.
        tmp_path, lane_db, job_cost_cap_usd=Decimal("0.49")
    )
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-cap", "chunk60-cap", "video-chunk60-cap")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "refused"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    assert clients.spark.run_calls == []
    assert clients.meta.upload_calls == []
    assert len(_jobs(lane_db)) == 1  # no sub-job admitted
    job_dir = tmp_path / "data" / "jobs" / str(ctx.job_id)
    assert list(job_dir.glob("chunk_*.mp4")) == []


def test_refused_chunk_fails_job_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path, monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    real_admit = pipeline_mod.admit_job

    def _refuse_subs(anon_id: str, ip_hash: str, video_hash: str, **kwargs):
        if "/chunk/" in anon_id:
            raise BudgetExhausted("test: no room for a chunk")
        return real_admit(anon_id, ip_hash, video_hash, **kwargs)

    monkeypatch.setattr(pipeline_mod, "admit_job", _refuse_subs)
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    ctx = real_admit("chunk60-ref", "chunk60-ref", "video-chunk60-ref")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    assert result.guide is None
    assert clients.spark.run_calls == []  # refused before any invoke
    assert len(_jobs(lane_db)) == 1
    job_dir = tmp_path / "data" / "jobs" / str(ctx.job_id)
    assert not (job_dir / "guide.json").exists()


def test_midway_chunk_refusal_leaves_no_debris(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path, monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    real_admit = pipeline_mod.admit_job
    calls = 0

    def _refuse_second_sub(anon_id: str, ip_hash: str, video_hash: str, **kwargs):
        nonlocal calls
        if "/chunk/" in anon_id:
            calls += 1
            if calls > 1:
                raise BudgetExhausted("test: chunk 1 refused")
        return real_admit(anon_id, ip_hash, video_hash, **kwargs)

    monkeypatch.setattr(pipeline_mod, "admit_job", _refuse_second_sub)
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = real_admit("chunk60-mid", "chunk60-mid", "video-chunk60-mid")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    assert result.guide is None
    # Chunk 0 did real work and settled; chunk 1 never ran.
    assert len(clients.spark.run_calls) == 3
    states = sorted(row["state"] for row in _jobs(lane_db))
    assert states == ["done", "failed"]
    # Every uploaded chunk file was deleted; no held reservation survives.
    assert len(clients.meta.upload_calls) == 1
    assert len(clients.meta.delete_calls) == 1
    conn = db.connect(lane_db)
    try:
        held = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE state = 'held'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert held == 0


def _single_fan_clients(costs=("0.08", "0.08", "0.07")) -> PipelineClients:
    runs = []
    for (start, end), cost in zip(
        ((0.5, 2.5), (0.6, 2.6), (0.4, 2.4)), costs
    ):
        runs.append(
            _run(
                [_payload_step(1, start, end, "Remove the rear screws",
                                [("fan", "left side")])],
                cost=cost,
            )
        )
    return PipelineClients(
        spark=ReplaySparkClient(runs),
        sam=ReplaySamClient({"fan": (FAN_BOX, [0.72, 0.60])}),
        meta=ReplayMetaFilesClient(),
    )


def test_settle_refusal_says_budget_limit(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _single_fan_clients(costs=("0.50", "0.08", "0.07"))
    ctx = admit_job("chunk60-settle", "chunk60-settle", "video-chunk60-settle")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=6.0,
    )
    assert result.state == "failed"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    assert "step extraction failed" not in (result.error or "")


def test_chunk_settle_refusal_says_budget_limit(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    runs = _two_chunk_runs()
    runs[0] = _run(
        [_payload_step(1, 0.5, 2.5, "Remove the rear screws",
                        [("fan", "left side")])],
        cost="0.50",
    )
    clients = PipelineClients(
        spark=ReplaySparkClient(
            runs,
            readbacks={
                "battery": ReadbackResult(
                    answer="yes", cost_usd=Decimal("0.04"), latency_ms=120
                )
            },
        ),
        sam=ReplaySamClient(
            {
                "fan": (FAN_BOX, [0.72, 0.60]),
                "battery": (BATTERY_BOX, [0.61, 0.55]),
            }
        ),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("chunk60-csettle", "chunk60-csettle", "video-chunk60-csettle")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    assert {row["state"] for row in _jobs(lane_db)} == {"failed"}
    marker = job_dir(str(ctx.job_id)) / REFUSAL_MARKER_FILENAME
    assert json.loads(marker.read_text()) == {"kind": "budget"}


def test_fail_with_refusal_kind_writes_marker_and_fails_job(
    tmp_path: Path, lane_db: Path,
) -> None:
    ctx = admit_job("chunk60-fkind", "chunk60-fkind", "video-chunk60-fkind")
    result = _fail(ctx, ctx.job_id, "over the limit", replay=True,
                   settings=_settings(tmp_path, lane_db), refusal_kind="budget")
    assert result.state == "failed"
    marker = job_dir(str(ctx.job_id)) / REFUSAL_MARKER_FILENAME
    assert json.loads(marker.read_text()) == {"kind": "budget"}


def _wide_steps(prefix: str, count: int, start_at: float, width: float):
    # Contiguous: phases meet edge to edge (time consensus excludes
    # gapped runs, so the old 2 s gaps cannot ride along).
    steps = []
    for index in range(count):
        start = start_at + index * width
        steps.append(
            _payload_step(
                index + 1, start, start + width, f"{prefix} {index}",
                [("fan", "left side"), ("battery", "lower right")],
            )
        )
    return steps


def test_sam_budget_degrades_on_stitched_guide(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    runs = []
    for jitter in (0.0, 0.05, -0.05):
        steps = _wide_steps("Open latch", 8, 0.5 + jitter, 5.0)
        runs.append(_run(steps, cost="0.08"))
    for jitter in (0.0, 0.05, -0.05):
        steps = _wide_steps("Check board", 1, 0.5 + jitter, 5.0)
        runs.append(_run(steps, cost="0.07"))
    clients = PipelineClients(
        spark=ReplaySparkClient(runs),
        sam=ReplaySamClient(
            {
                "fan": (FAN_BOX, _sam_scores(40)),
                "battery": (BATTERY_BOX, _sam_scores(40, 0.81, 0.61)),
            }
        ),
        meta=ReplayMetaFilesClient(),
    )
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60-sam", "chunk60-sam", "video-chunk60-sam")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert len(result.guide.steps) == 9
    # 18 slots x 2 frames exceeds the 32-call SAM budget: the finished
    # 16 slots stay, the rest are unchecked, the job still completes.
    assert len(clients.sam.calls) == 32
    assert len(result.guide.labels) == 16
    assert clients.spark.readback_calls == []


def test_replay_single_chunk_takes_no_sub_job(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip6: Path,
) -> None:
    clients = _single_fan_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    result = run_job(
        clip6,
        clients=clients,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="chunk60-replay",
        ip_hash="chunk60-replay",
        replay=True,
        settings=_settings(tmp_path, lane_db),
    )
    assert result.state == "done", result.error
    assert len(_jobs(lane_db)) == 1
    assert result.job_id is not None
    job_dir = tmp_path / "data" / "jobs" / str(result.job_id)
    assert list(job_dir.glob("chunk_*.mp4")) == []
    assert result.guide is not None
    assert len(result.guide.steps) == 1


def test_live_chunked_without_settings_raises(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    clients = PipelineClients(spark=object(), sam=object(), meta=object())
    ctx = admit_job("chunk60-live", "chunk60-live", "video-chunk60-live")
    with pytest.raises(ValueError, match="needs settings"):
        run_stages(
            ctx, "parent-file-id", tmp_path / "absent.mp4",
            clients=clients, lifecycle=lifecycle, uploads=uploads, replay=False,
            settings=None, duration_s=65.0,
        )


# --- config: chunk knobs and the grown job cap -------------------------------


def test_chunk_settings_defaults() -> None:
    settings = load_settings({})
    assert settings.max_video_s == 300.0
    assert settings.chunk_s == 60.0
    assert settings.chunk_overlap_s == 2.0
    assert settings.chunk_cut_timeout_s == 300.0
    assert settings.chunk_deadline_allowance_s == 360.0
    assert settings.deadline_extension_max_s == 2160.0
    assert settings.job_cost_cap_usd == Decimal("1.60")


def test_chunk_settings_from_env() -> None:
    settings = load_settings(
        {
            "PARTFIDELITY_MAX_VIDEO_S": "120",
            "PARTFIDELITY_CHUNK_S": "30",
            "PARTFIDELITY_CHUNK_OVERLAP_S": "0",
            "PARTFIDELITY_CHUNK_CUT_TIMEOUT_S": "45",
            "PARTFIDELITY_CHUNK_DEADLINE_ALLOWANCE_S": "900",
            "PARTFIDELITY_DEADLINE_EXTENSION_MAX_S": "3600",
            "PARTFIDELITY_JOB_COST_CAP_USD": "0.50",
        }
    )
    assert settings.max_video_s == 120.0
    assert settings.chunk_s == 30.0
    assert settings.chunk_overlap_s == 0.0
    assert settings.chunk_cut_timeout_s == 45.0
    assert settings.chunk_deadline_allowance_s == 900.0
    assert settings.deadline_extension_max_s == 3600.0
    assert settings.job_cost_cap_usd == Decimal("0.50")


@pytest.mark.parametrize(
    "env",
    [
        {"PARTFIDELITY_MAX_VIDEO_S": "0"},
        {"PARTFIDELITY_MAX_VIDEO_S": "abc"},
        {"PARTFIDELITY_CHUNK_S": "-1"},
        {"PARTFIDELITY_CHUNK_OVERLAP_S": "-1"},
        {"PARTFIDELITY_CHUNK_OVERLAP_S": "60"},
        {"PARTFIDELITY_CHUNK_CUT_TIMEOUT_S": "0"},
        {"PARTFIDELITY_CHUNK_DEADLINE_ALLOWANCE_S": "0"},
        {"PARTFIDELITY_CHUNK_DEADLINE_ALLOWANCE_S": "abc"},
        {"PARTFIDELITY_DEADLINE_EXTENSION_MAX_S": "0"},
        {"PARTFIDELITY_DEADLINE_EXTENSION_MAX_S": "-1"},
        {"PARTFIDELITY_DEADLINE_EXTENSION_MAX_S": "abc"},
        {"PARTFIDELITY_JOB_COST_CAP_USD": "0"},
        {"PARTFIDELITY_JOB_COST_CAP_USD": "free"},
    ],
)
def test_chunk_settings_reject_bad_values(env) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        load_settings(env)


# --- upload cap from config ------------------------------------------------


def _mocked_probe(monkeypatch, *, duration_s: float):
    from app import uploads as uploads_mod

    mocked = uploads_mod.MediaInfo(
        duration_s=duration_s,
        containers=frozenset({"mp4"}),
        has_video_stream=True,
        video_codec="h264",
        video_frames=610,
    )
    monkeypatch.setattr(uploads_mod, "_probe_media", lambda path: mocked)


def test_upload_cap_from_config(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"probe-is-mocked")
    _mocked_probe(monkeypatch, duration_s=61.0)
    info = check_upload_bounds(clip, size_bytes=clip.stat().st_size)
    assert info.duration_s == 61.0
    with pytest.raises(OverLimitError) as excinfo:
        check_upload_bounds(clip, size_bytes=clip.stat().st_size,
                             max_duration_s=60.0)
    assert excinfo.value.reason == "duration"
    assert excinfo.value.message == over_limit_message(60.0)
    assert "60 seconds" in excinfo.value.message
    assert over_limit_message(120.0).startswith("Trim it to 120 seconds")


def test_real_61s_clip_accepted(clip61: Path) -> None:
    info = check_upload_bounds(clip61, size_bytes=clip61.stat().st_size)
    assert info.duration_s == pytest.approx(61.0, abs=1.0)


def _route_client(tmp_path: Path, lane_db: Path, settings: Settings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routes import uploads as routes_mod

    application = FastAPI()
    application.include_router(routes_mod.router)
    application.state.settings = settings
    return TestClient(application)


def test_route_accepts_61s_clip(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip61: Path,
) -> None:
    from app.routes import uploads as routes_mod

    settings = _settings(tmp_path, lane_db)
    routes_mod.set_test_overrides(uploads=uploads, lifecycle=lifecycle)
    try:
        client = _route_client(tmp_path, lane_db, settings)
        posted = client.post(
            "/upload",
            files={"clip": ("clip.mp4", clip61.read_bytes(), "video/mp4")},
            follow_redirects=False,
        )
    finally:
        routes_mod.clear_test_overrides()
    assert posted.status_code == 303, posted.text[:500]
    job_id = posted.headers["location"].rsplit("/", 1)[-1]
    loaded = uploads.load_job(job_id)
    assert loaded is not None
    assert loaded[1].preview.duration_s == pytest.approx(61.0, abs=1.0)


def test_route_limits_follow_settings(tmp_path: Path, lane_db: Path) -> None:
    settings = _settings(tmp_path, lane_db, max_video_s=120.0)
    client = _route_client(tmp_path, lane_db, settings)
    body = client.get("/upload").text
    assert "120 seconds or less" in body
    assert '"maxDuration": 120' in body


def test_stitch_zero_overlap_links_touching_only() -> None:
    # the placeholder wording carries a
    # known verb, since verb-less wordings never link under the strict
    # signature; the touching/gapped expectations are unchanged.
    touching = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0),
             _bundle([_step(1, 50.0, 60.0, "Remove the case")])),
            (ChunkWindow(60.0, 120.0),
             _bundle([_step(1, 0.0, 5.0, "Remove the case")])),
        ],
        overlap_s=0.0,
    )
    assert len(touching.steps) == 1
    gapped = stitch_bundles(
        [
            (ChunkWindow(0.0, 60.0),
             _bundle([_step(1, 50.0, 59.0, "Remove the case")])),
            (ChunkWindow(60.0, 120.0),
             _bundle([_step(1, 1.0, 5.0, "Remove the case")])),
        ],
        overlap_s=0.0,
    )
    assert len(gapped.steps) == 2


# --- chunk children: admission, recovery and caps ---


def test_internal_child_admits_active_when_slots_full(lane_db: Path) -> None:
    admit_job("slot-a", "slot-a", "video-slot-a")
    admit_job("slot-b", "slot-b", "video-slot-b")
    admit_job("waiter", "waiter", "video-waiter")
    child = admit_job(
        "parent/chunk/0", "parent/chunk/0", "video-chunk", internal=True
    )
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[str(child.job_id)] == "active"
    assert sorted(states.values()) == [
        "active", "active", "active", "waiting",
    ]


def test_chunk_subjobs_stay_active_under_contention(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60-busy", "chunk60-busy", "video-chunk60-busy")
    filler = admit_job("chunk60-filler", "chunk60-filler", "video-chunk60-filler")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[str(ctx.job_id)] == "done"
    assert states[str(filler.job_id)] == "active"
    subs = sorted(
        state for job_id, state in states.items()
        if job_id not in (str(ctx.job_id), str(filler.job_id))
    )
    assert subs == ["done", "done"]


def test_chunk_child_finish_uses_bounded_recovery(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path, monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod
    from app.ledger import ContentionError, JobContextImpl

    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-flaky", "chunk60-flaky", "video-chunk60-flaky")
    parent_id = str(ctx.job_id)
    real_extract = pipeline_mod.extract_steps_cached

    def _blow_chunk_budget(job_ctx, *args, **kwargs):
        if str(job_ctx.job_id) != parent_id:
            raise BudgetExhausted("test: chunk over budget")
        return real_extract(job_ctx, *args, **kwargs)

    real_finish = JobContextImpl.finish
    flaked: list[str] = []

    def _flake_once(self, outcome, *args, **kwargs):
        if (
            outcome == "failed"
            and str(self.job_id) != parent_id
            and not flaked
        ):
            flaked.append(str(self.job_id))
            raise ContentionError("test: one contention")
        return real_finish(self, outcome, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "extract_steps_cached", _blow_chunk_budget)
    monkeypatch.setattr(JobContextImpl, "finish", _flake_once)
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert flaked, "expected one flaked child finish"
    assert result.state == "failed"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[parent_id] == "failed"
    children = [
        state for job_id, state in states.items() if job_id != parent_id
    ]
    assert children == ["failed"]


def test_chunk_meta_files_registered_for_sweeper(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path, monkeypatch,
) -> None:
    from datetime import timedelta

    from app import pipeline as pipeline_mod

    real_admit = pipeline_mod.admit_job
    calls = 0

    def _refuse_second_sub(anon_id: str, ip_hash: str, video_hash: str, **kwargs):
        nonlocal calls
        if "/chunk/" in anon_id:
            calls += 1
            if calls > 1:
                raise BudgetExhausted("test: chunk 1 refused")
        return real_admit(anon_id, ip_hash, video_hash, **kwargs)

    monkeypatch.setattr(pipeline_mod, "admit_job", _refuse_second_sub)
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = real_admit("chunk60-sweep", "chunk60-sweep", "video-chunk60-sweep")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    conn = db.connect(lane_db)
    try:
        rows = conn.execute(
            "SELECT id, job_id, delete_state FROM meta_files"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    # The finished chunk's file was deleted and its outcome recorded, so
    # no row sits live for the sweeper to re-delete at provider expiry.
    assert rows[0]["delete_state"] == "verified_absent"
    jobs = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert jobs[rows[0]["job_id"]] == "done"
    due = lifecycle.list_due_meta_files(db.utcnow() + timedelta(hours=25))
    assert [str(record.id) for record in due] == []


def test_run_stages_refuses_over_cap_duration(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, max_video_s=60.0)
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-cap", "chunk60-cap", "video-chunk60-cap")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "refused"
    assert result.error == over_limit_message(60.0)
    assert clients.spark.run_calls == []
    assert clients.meta.upload_calls == []
    assert len(_jobs(lane_db)) == 1


def test_run_stages_accepts_duration_at_cap(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    clip6: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, max_video_s=60.0)
    clients = _single_fan_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60-edge", "chunk60-edge", "video-chunk60-edge")
    result = run_stages(
        ctx, "parent-file-id", clip6,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=60.0,
    )
    assert result.state == "done", result.error


def test_run_job_refuses_over_cap_clip(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip61: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, max_video_s=60.0)
    clients = _two_chunk_clients()
    result = run_job(
        clip61,
        clients=clients,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="chunk60-jobcap",
        ip_hash="chunk60-jobcap",
        replay=True,
        settings=settings,
    )
    assert result.state == "refused"
    assert result.error == over_limit_message(60.0)


def test_route_advertises_fractional_cap(
    tmp_path: Path, lane_db: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, max_video_s=60.5)
    client = _route_client(tmp_path, lane_db, settings)
    body = client.get("/upload").text
    assert "60.5 seconds or less" in body
    assert '"maxDuration": 60.5' in body


def test_unknown_preview_advertises_configured_cap(
    tmp_path: Path, lane_db: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, max_video_s=120.0)
    client = _route_client(tmp_path, lane_db, settings)
    gone = client.get("/preview/no-such-job-1")
    assert gone.status_code == 404
    assert "120 seconds or less" in gone.text
    assert '"maxDuration": 120.0' in gone.text


def test_failed_chunk_delete_is_due_immediately(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, monkeypatch,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")

    def _boom(file_id) -> None:
        raise MetaFilesError("test: delete failed")

    monkeypatch.setattr(clients.meta, "delete_file", _boom)
    ctx = admit_job("chunk60-del", "chunk60-del", "video-chunk60-del")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, replay=True,
        settings=settings, duration_s=65.0, uploads=uploads,
    )
    assert result.state == "done", result.error
    conn = db.connect(lane_db)
    try:
        rows = conn.execute(
            "SELECT id, delete_state FROM meta_files ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert [row["delete_state"] for row in rows] == ["retry", "retry"]
    due = lifecycle.list_due_meta_files(db.utcnow())
    assert sorted(str(record.id) for record in due) == sorted(
        row["id"] for row in rows
    )


def test_crashed_chunk_upload_is_found_by_sweeper(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    from datetime import timedelta

    from app.deletion import sweep_expired
    from app.uploads import META_UPLOAD_REQUEST_HOURS, hash_file

    parent = admit_job("chunk60-crash", "chunk60-crash", "video-chunk60-crash")
    parent_id = str(parent.job_id)
    chunk_path = job_dir(parent_id) / chunk_filename(0)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(b"x" * 100)
    artifact_id = parent.artifacts.register(CHUNK_ARTIFACT_KIND, chunk_path)
    sub = admit_job(
        f"{parent_id}/chunk/0", f"{parent_id}/chunk/0",
        hash_file(chunk_path), internal=True,
    )
    key = f"{parent_id}:chunk:0"
    uploads.claim_chunk_upload(
        sub.job_id, key, artifact_id, duration_s=60.0,
        size_bytes=chunk_path.stat().st_size, now=db.utcnow(),
    )
    meta = ReplayMetaFilesClient()
    uploaded = meta.upload_file(
        chunk_path, expires_after_hours=META_UPLOAD_REQUEST_HOURS,
        idempotency_key=key,
    )
    # Crash between upload and registration: the claim row still names the
    # key, so the sweeper recovers the provider file and deletes it. The
    # overdue expiry runs first so the sweep itself stamps real times.
    recoverable = uploads.list_recoverable(db.utcnow())
    assert [str(row.preview.job_id) for row in recoverable] == [str(sub.job_id)]
    from app.ledger import expire_overdue_jobs

    expire_overdue_jobs(db.utcnow() + timedelta(hours=1))
    sweep_expired(
        db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=meta
    )
    assert meta.get_file(uploaded.file_id) is None
    conn = db.connect(lane_db)
    try:
        row = conn.execute(
            "SELECT delete_state FROM meta_files WHERE id = ?",
            (str(uploaded.file_id),),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["delete_state"] == "verified_absent"
    assert uploads.list_recoverable(db.utcnow()) == ()


def test_route_budget_preflight_refuses_before_any_meta_upload(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path,
) -> None:
    from app.routes import uploads as routes_mod

    settings = _settings(
        tmp_path, lane_db, pipeline_mode="replay",
        job_cost_cap_usd=Decimal("0.49"),  # below 2 x $0.2475 window estimates
    )
    meta = ReplayMetaFilesClient()
    routes_mod.set_test_overrides(
        uploads=uploads, lifecycle=lifecycle, meta_client=meta
    )
    try:
        client = _route_client(tmp_path, lane_db, settings)
        posted = client.post(
            "/upload",
            files={"clip": ("clip.mp4", clip65.read_bytes(), "video/mp4")},
            follow_redirects=False,
        )
        assert posted.status_code == 303, posted.text[:500]
        job_id = posted.headers["location"].rsplit("/", 1)[-1]
        refused = client.post(
            f"/preview/{job_id}/confirm",
            data={"consent": "on", "ownership": "on"},
            follow_redirects=False,
        )
    finally:
        routes_mod.clear_test_overrides()
    assert refused.status_code == 303, refused.text[:500]
    assert refused.headers["location"] == f"/jobs/{job_id}"
    assert meta.upload_calls == []
    loaded = uploads.load_job(job_id)
    assert loaded is not None
    _, record = loaded
    assert record.state == "failed"
    assert record.meta_file_id is None
    assert record.last_error == STEP_EXTRACTION_OVER_BUDGET
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[job_id] == "failed"


def test_live_chunked_without_settings_settles_parent(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    clients = PipelineClients(spark=object(), sam=object(), meta=object())
    ctx = admit_job("chunk60-noset", "chunk60-noset", "video-chunk60-noset")
    with pytest.raises(ValueError, match="needs settings"):
        run_stages(
            ctx, "parent-file-id", tmp_path / "absent.mp4",
            clients=clients, lifecycle=lifecycle, uploads=uploads, replay=False,
            settings=None, duration_s=65.0,
        )
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states == {str(ctx.job_id): "failed"}
    conn = db.connect(lane_db)
    try:
        open_holds = conn.execute(
            "SELECT COUNT(*) FROM reservations"
            " WHERE state IN ('held', 'job_hold')"
        ).fetchone()[0]
    finally:
        conn.close()
    assert open_holds == 0


# --- preflight and cancellation ---


def test_run_job_budget_preflight_refuses_before_any_upload(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path,
) -> None:
    settings = _settings(
        # Below 2 x $0.2475 window estimates, so preflight refuses.
        tmp_path, lane_db, job_cost_cap_usd=Decimal("0.49")
    )
    clients = _two_chunk_clients()
    result = run_job(
        clip65,
        clients=clients,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="chunk60-jobpre",
        ip_hash="chunk60-jobpre",
        replay=True,
        settings=settings,
    )
    assert result.state == "refused"
    assert result.error == STEP_EXTRACTION_OVER_BUDGET
    assert clients.meta.upload_calls == []
    assert clients.spark.run_calls == []
    assert result.job_id is not None
    loaded = uploads.load_job(result.job_id)
    assert loaded is not None
    assert loaded[1].meta_file_id is None
    assert len(_jobs(lane_db)) == 1  # no sub-job admitted


def test_chunk_child_preview_path_exists(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60-owner", "chunk60-owner", "video-chunk60-owner")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    parent_id = str(ctx.job_id)
    conn = db.connect(lane_db)
    try:
        sub_ids = [
            row["job_id"]
            for row in conn.execute(
                "SELECT job_id FROM uploads WHERE job_id != ?", (parent_id,)
            ).fetchall()
        ]
    finally:
        conn.close()
    assert len(sub_ids) == 2
    for sub_id in sub_ids:
        loaded = uploads.load_job(sub_id)
        assert loaded is not None
        assert loaded[1].preview.local_video.is_file()


def test_parent_cancellation_stops_before_next_child_upload(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    # Legacy extraction to match this file's legacy settings.
    clients.spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    ctx = admit_job("chunk60-cancel", "chunk60-cancel", "video-chunk60-cancel")
    with open(clip65, "rb") as handle:
        preview = prepare_preview(
            ctx, handle, clip65.name, uploads=uploads,
            max_duration_s=settings.max_video_s,
        )
    real_cut = pipeline_mod.cut_chunk
    cuts = 0

    def _cancel_on_second_cut(src, window, dst, **kwargs):
        nonlocal cuts
        cuts += 1
        if cuts == 2:
            # Cancellation races the second cut, after the loop-top
            # check already passed: the pre-upload check must catch it.
            uploads.request_cancellation(ctx.job_id, db.utcnow())
        return real_cut(src, window, dst, **kwargs)

    monkeypatch.setattr(pipeline_mod, "cut_chunk", _cancel_on_second_cut)
    result = run_stages(
        ctx, "parent-file-id", preview.local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=preview.duration_s,
    )
    assert cuts == 2
    assert result.state == "failed"
    assert result.error == "job cancelled"
    assert len(clients.spark.run_calls) == 3  # chunk 0 only
    assert len(clients.meta.upload_calls) == 1
    assert len(clients.meta.delete_calls) == 1
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[str(ctx.job_id)] == "failed"
    children = sorted(
        state for job_id, state in states.items()
        if job_id != str(ctx.job_id)
    )
    assert children == ["done", "failed"]


def test_upfront_parent_cancellation_runs_no_child(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-precancel", "chunk60-precancel", "video-chunk60-pre")
    with open(clip65, "rb") as handle:
        preview = prepare_preview(
            ctx, handle, clip65.name, uploads=uploads,
            max_duration_s=settings.max_video_s,
        )
    uploads.request_cancellation(ctx.job_id, db.utcnow())
    result = run_stages(
        ctx, "parent-file-id", preview.local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=preview.duration_s,
    )
    assert result.state == "failed"
    assert result.error == "job cancelled"
    assert clients.meta.upload_calls == []
    assert clients.spark.run_calls == []
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states == {str(ctx.job_id): "failed"}


def test_cancel_during_upload_skips_extraction(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-uplcancel", "chunk60-uplcancel", "video-chunk60-upl")
    with open(clip65, "rb") as handle:
        preview = prepare_preview(
            ctx, handle, clip65.name, uploads=uploads,
            max_duration_s=settings.max_video_s,
        )
    real_upload = pipeline_mod._upload_chunk_file

    def _cancel_after_upload(*args, **kwargs):
        file_id = real_upload(*args, **kwargs)
        # Cancellation lands during the first chunk upload, after the
        # pre-upload check passed: the post-upload check must catch it.
        uploads.request_cancellation(ctx.job_id, db.utcnow())
        return file_id

    monkeypatch.setattr(pipeline_mod, "_upload_chunk_file", _cancel_after_upload)
    result = run_stages(
        ctx, "parent-file-id", preview.local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=preview.duration_s,
    )
    assert result.state == "failed"
    assert result.error == "job cancelled"
    assert clients.spark.run_calls == []
    assert len(clients.meta.upload_calls) == 1
    assert len(clients.meta.delete_calls) == 1
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[str(ctx.job_id)] == "failed"
    children = sorted(
        state for job_id, state in states.items()
        if job_id != str(ctx.job_id)
    )
    assert children == ["failed"]


def test_cancel_check_fails_closed_on_store_error(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, monkeypatch,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-storeerr", "chunk60-storeerr", "video-chunk60-err")
    with open(clip65, "rb") as handle:
        preview = prepare_preview(
            ctx, handle, clip65.name, uploads=uploads,
            max_duration_s=settings.max_video_s,
        )
    uploads.request_cancellation(ctx.job_id, db.utcnow())

    def _boom(job_id):
        raise sqlite3.OperationalError("test: transient store read failure")

    monkeypatch.setattr(uploads, "load_job", _boom)
    result = run_stages(
        ctx, "parent-file-id", preview.local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=preview.duration_s,
    )
    assert result.state == "failed"
    assert result.error == "job cancelled"
    assert clients.meta.upload_calls == []
    assert clients.spark.run_calls == []
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states == {str(ctx.job_id): "failed"}


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_chunk_extraction_base_exception_settles_parent(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, monkeypatch, signal,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    clients = _two_chunk_clients()
    ctx = admit_job("chunk60-sig", "chunk60-sig", "video-chunk60-sig")
    parent_id = str(ctx.job_id)
    real_extract = pipeline_mod.extract_steps_cached

    def _interrupt(job_ctx, *args, **kwargs):
        if str(job_ctx.job_id) != parent_id:
            raise signal("test: operator interrupt")
        return real_extract(job_ctx, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "extract_steps_cached", _interrupt)
    with pytest.raises(signal):
        run_stages(
            ctx, "parent-file-id", clip65,
            clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
            settings=settings, duration_s=65.0,
        )
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[parent_id] == "failed"
    children = [
        state for job_id, state in states.items() if job_id != parent_id
    ]
    assert children == ["failed"]
    conn = db.connect(lane_db)
    try:
        open_holds = conn.execute(
            "SELECT COUNT(*) FROM reservations"
            " WHERE state IN ('held', 'job_hold')"
        ).fetchone()[0]
    finally:
        conn.close()
    assert open_holds == 0


def test_seam_overlap_from_real_cut_windows_does_not_fail_the_cover():
    # Live V2 (279 s): chunk 0 cut to 60.134 s, chunk 1 starts at 58 s, so the seam
    # steps overlap by 2.134 s, just over the 2 s tolerance; every attempt failed.
    from app.chunks import ContiguityError, derive_contiguous_ends, trim_seam_overlaps

    steps = (
        _step(1, 0.0, 60.134, "Open the laptop"),
        _step(2, 58.0, 66.9, "Remove the cover"),
        _step(3, 66.9, 120.0, "Replace the memory"),
    )
    with pytest.raises(ContiguityError):
        derive_contiguous_ends(steps, 120.0, tolerance_s=2.0)
    adjacent = (frozenset({0}), frozenset({1}), frozenset({1}))
    derived = derive_contiguous_ends(
        trim_seam_overlaps(steps, [(58.0, 60.134)], adjacent), 120.0, tolerance_s=2.0
    )
    assert [(s.start_s, s.end_s) for s in derived] == [
        (0.0, 58.0), (58.0, 66.9), (66.9, 120.0)
    ]
    wide = trim_seam_overlaps(steps, [(59.0, 60.134)], adjacent)
    with pytest.raises(ContiguityError):
        derive_contiguous_ends(wide, 120.0, tolerance_s=2.0)


def test_overlap_away_from_any_seam_still_fails_the_cover():
    from app.chunks import ContiguityError, derive_contiguous_ends, trim_seam_overlaps

    steps = (_step(1, 0.0, 10.0, "Open"), _step(2, 9.0, 20.0, "Remove"))
    adjacent = (frozenset({0}), frozenset({1}))
    for seams in ([], [(55.0, 57.0)]):
        with pytest.raises(ContiguityError):
            derive_contiguous_ends(
                trim_seam_overlaps(steps, seams, adjacent), 20.0, tolerance_s=0.5
            )


def test_same_chunk_overlap_inside_a_seam_zone_still_fails_the_cover():
    from app.chunks import ContiguityError, derive_contiguous_ends, trim_seam_overlaps

    steps = (_step(1, 0.0, 59.5, "Open"), _step(2, 58.5, 65.0, "Remove"))
    # [57,59.5]/[58.5,65] shape: both steps come from chunk 0, so no seam crossing.
    shared = (frozenset({0, 1}), frozenset({1}))  # origins overlap: not a pure seam pair
    for origins in (
        (frozenset({0}), frozenset({0})),
        (frozenset({1}), frozenset({1})),
        shared,
    ):
        with pytest.raises(ContiguityError):
            derive_contiguous_ends(
                trim_seam_overlaps(steps, [(58.0, 60.134)], origins),
                65.0,
                tolerance_s=0.5,
            )
