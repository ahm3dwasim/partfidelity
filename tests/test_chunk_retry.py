"""per-chunk evidence, retry/resume of the failing chunk."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from app import db, stage_cache
from app.admission import admit_job
from app.chunks import (
    ChunkWindow,
    CutInfo,
    cut_chunk,
    plan_chunks,
    require_contiguous_cover,
)
from app.ledger import LifecycleRepositoryImpl
from app.pipeline import (
    PipelineClients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    _ChunkAttemptsExhausted,
    _ChunkClaimHeld,
    _ChunkRetryRefused,
    _ChunkSettlementPending,
    _extract_chunked,
    _load_resumed_bundle,
    chunk_retry_fits,
    run_stages,
)
from app.service import write_sample_clip
from app.settings import Settings, load_settings
from app.spark import (
    CHUNK_BUNDLE_FORMAT,
    DEFAULT_SPARK_CONFIG,
    SparkRunResult,
    SparkStageConfig,
    StepsDisagreement,
    bundle_from_json,
    bundle_to_json,
    chunk_bundle_digest,
    consensus_digest,
    extract_steps,
    extract_steps_cached,
    load_chunk_bundle,
    load_chunk_cut_window,
    run_digest,
    save_chunk_bundle,
)
from app.stage_cache import (
    STAGE_CHUNK_BUNDLE,
    STAGE_SPARK_CONSENSUS,
    STAGE_SPARK_RUN,
)
from app.uploads_store import UploadStoreImpl

from test_spark_schema import (
    RecordingContext,
    ScriptedSparkClient,
    make_payload,
    make_step,
)


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
def clip65(tmp_path_factory) -> Path:
    return write_sample_clip(
        tmp_path_factory.mktemp("media") / "sixtyfive.mp4", seconds=65.0, fps=5
    )


@pytest.fixture(scope="module")
def clip61av(tmp_path_factory) -> Path:
    """61 s video+audio clip: phone-footage-like, so mid-stream cuts
    decode longer than nominal — the short-chunk scenario."""
    path = tmp_path_factory.mktemp("media") / "sixtyone_av.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=61:size=320x240:rate=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=61",
            "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, timeout=300,
    )
    return path


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


_LEGACY_CONFIG = SparkStageConfig(span_mode="legacy")


def _tune_kwargs(settings):
    """The extraction tune the pipeline binds into chunk digests."""
    return {
        "scene_cut_threshold": settings.scene_cut_threshold,
        "scene_cuts_in_prompt": settings.scene_cuts_in_prompt,
        "snap_window_s": settings.snap_window_s,
        "min_phase_length_s": settings.min_phase_length_s,
        "clock_burn_in": settings.clock_burn_in,
        "merge_same_action": settings.merge_same_action,
    }


def _legacy_spark(*args, **kwargs) -> ReplaySparkClient:
    """A replay Spark client on the legacy wording consensus.

    Replay clients carry no stage config of their own, so without
    this they would run the contiguous time consensus (which keeps
    wording splits as flagged segments instead of raising); these
    retry tests pin the sparse fail-closed path.
    """
    client = ReplaySparkClient(*args, **kwargs)
    client.spark_stage_config = _LEGACY_CONFIG
    return client


def _payload_step(number, start, end, action, parts=(), kind="repair"):
    return {
        "number": number,
        "start_s": start,
        "end_s": end,
        "action": action,
        "tools": [],
        "parts": [
            {"phrase": phrase, "position_hint": hint} for phrase, hint in parts
        ],
        "hazards": [],
        "kind": kind,
    }


def _run(steps, *, cost="0.08") -> SparkRunResult:
    return SparkRunResult(
        payload={"device_guess": "laptop", "confidence": 0.82, "steps": steps},
        cost_usd=Decimal(cost),
        latency_ms=900,
    )


def _agreeing_runs(action: str, parts=()) -> list[SparkRunResult]:
    """Three runs that reach consensus on one local step.

    The wording must carry a non-generic part: an empty part set
    never matches, so generic-only wordings cannot agree.
    """
    runs = []
    for start, end in ((0.5, 2.5), (0.6, 2.6), (0.4, 2.4)):
        runs.append(_run([_payload_step(1, start, end, action, parts)]))
    return runs


def _disagreeing_runs() -> list[SparkRunResult]:
    """Three runs with no two-run majority: consensus must raise."""
    return [
        _run([_payload_step(1, 0.5, 2.5, "Alpha lift", [("alpha", "")])]),
        _run([_payload_step(1, 3.0, 5.0, "Beta press", [("beta", "")])]),
        _run([_payload_step(1, 6.0, 8.0, "Gamma turn", [("gamma", "")])]),
    ]


def _jobs(database: Path) -> list[sqlite3.Row]:
    conn = db.connect(database)
    try:
        return conn.execute(
            "SELECT id, state FROM jobs ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()


def _claim_ids(database: Path) -> list[str]:
    conn = db.connect(database)
    try:
        rows = conn.execute("SELECT claim_id FROM uploads").fetchall()
        return sorted(row["claim_id"] for row in rows)
    finally:
        conn.close()


def _upload_row(database: Path, claim_id: str) -> sqlite3.Row:
    conn = db.connect(database)
    try:
        row = conn.execute(
            "SELECT job_id, meta_file_id FROM uploads WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


# --- retry: one disagreement costs exactly one extra chunk attempt --------


def test_disagreement_retried_once_fifteen_calls(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, chunk_s=20.0, chunk_overlap_s=2.0)
    runs = (
        _agreeing_runs("Open the lid")
        + _agreeing_runs("Remove the rear screws")
        + _disagreeing_runs()
        + _agreeing_runs("Disconnect the battery cable")
        + _agreeing_runs("Lift out the battery pack")
    )
    assert len(runs) == 15
    clients = PipelineClients(
        spark=_legacy_spark(runs),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("retry-15", "retry-15", "video-retry-15")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "done", result.error
    assert result.guide is not None
    assert [s.number for s in result.guide.steps] == [1, 2, 3, 4]

    # Exactly 15 Spark calls: 4 chunks x 3 plus one 3-call retry.
    assert len(clients.spark.run_calls) == 15
    from collections import Counter

    per_file = Counter(clients.spark.run_calls)
    assert len(per_file) == 5  # chunk 2 ran under two Meta file ids
    assert set(per_file.values()) == {3}
    assert len(clients.meta.upload_calls) == 5

    parent_id = str(ctx.job_id)
    claims = _claim_ids(lane_db)
    assert claims == [
        f"{parent_id}:chunk:0",
        f"{parent_id}:chunk:1",
        f"{parent_id}:chunk:2",
        f"{parent_id}:chunk:2:retry:1",
        f"{parent_id}:chunk:3",
    ]
    # Chunks 0/1 were never called twice: their file ids saw 3 calls each.
    for claim in (f"{parent_id}:chunk:0", f"{parent_id}:chunk:1"):
        assert per_file[_upload_row(lane_db, claim)["meta_file_id"]] == 3
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert sorted(states.values()) == ["done"] * 5 + ["failed"]
    failed_attempt = _upload_row(lane_db, f"{parent_id}:chunk:2")["job_id"]
    assert states[str(failed_attempt)] == "failed"

    # The failed attempt's three paid raws survived for offline diagnosis.
    failed = _upload_row(lane_db, f"{parent_id}:chunk:2")
    sub_id = str(failed["job_id"])
    file_id = failed["meta_file_id"]
    # Legacy extraction keeps the pre-extent cache path: the run and
    # consensus digests take no extent even though the pipeline knows it.
    for index in range(3):
        assert (
            tmp_cache / STAGE_SPARK_RUN
            / f"{run_digest(sub_id, file_id, index, _LEGACY_CONFIG, None, None, settings.clock_burn_in)}.json"
        ).exists()
    assert (
        tmp_cache / STAGE_SPARK_CONSENSUS
        / f"{consensus_digest(sub_id, file_id, _LEGACY_CONFIG, None, None, settings.clock_burn_in)}.json"
    ).exists() is False


def test_retry_exhausted_fails_closed_without_touching_finished_chunks(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    clients = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid") + _disagreeing_runs() * 2
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("retry-doom", "retry-doom", "video-retry-doom")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    assert result.error == "step extraction failed"
    # Chunk 0 ran once (3 calls); chunk 1 ran twice and still disagreed.
    assert len(clients.spark.run_calls) == 9
    assert len(clients.meta.upload_calls) == 3
    from collections import Counter

    per_file = Counter(clients.spark.run_calls)
    parent_id = str(ctx.job_id)
    first_file = _upload_row(lane_db, f"{parent_id}:chunk:0")["meta_file_id"]
    assert per_file[first_file] == 3
    assert _claim_ids(lane_db) == [
        f"{parent_id}:chunk:0",
        f"{parent_id}:chunk:1",
        f"{parent_id}:chunk:1:retry:1",
    ]


# --- retry budget: refused up front with an honest message -----------------


def test_retry_over_job_cap_refused_upfront(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db, job_cost_cap_usd=Decimal("0.60"))
    clients = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("retry-cap", "retry-cap", "video-retry-cap")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    assert result.error == (
        "step extraction failed: chunk 0 disagreed and a $0.2475 retry"
        " would exceed the $0.60 job budget"
    )
    # Refused before any retry upload or Spark call.
    assert len(clients.spark.run_calls) == 3
    assert len(clients.meta.upload_calls) == 1
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[str(ctx.job_id)] == "failed"
    assert sorted(states.values()) == ["failed", "failed"]


def test_retry_over_per_video_budget_refused_upfront(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(
        tmp_path, lane_db, chunk_retry_budget_usd=Decimal("0")
    )
    clients = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("retry-bud", "retry-bud", "video-retry-bud")
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "failed"
    assert result.error == (
        "step extraction failed: chunk 0 disagreed and the $0"
        " per-video retry budget is spent"
    )
    assert len(clients.spark.run_calls) == 3
    assert len(clients.meta.upload_calls) == 1


def test_chunk_retry_fits_bounds() -> None:
    assert chunk_retry_fits(
        Decimal("0"), Decimal("0.26"), Decimal("1.04"),
        Decimal("1.60"), Decimal("0.26"),
    ) is True
    # Retry budget spent: a second retry never fits the default budget.
    assert chunk_retry_fits(
        Decimal("0.26"), Decimal("0.26"), Decimal("1.04"),
        Decimal("1.60"), Decimal("0.26"),
    ) is False
    # Six chunks under the $1.60 cap: zero retry fits (7 x $0.26).
    assert chunk_retry_fits(
        Decimal("0"), Decimal("0.26"), Decimal("1.56"),
        Decimal("1.60"), Decimal("1.00"),
    ) is False
    # Exact fit on the cap is allowed.
    assert chunk_retry_fits(
        Decimal("0"), Decimal("0.26"), Decimal("1.34"),
        Decimal("1.60"), Decimal("1.00"),
    ) is True


# --- resume: finished chunks cost zero cut/upload/calls --------------------


class _CrashAfterCalls:
    """Spark double that simulates a process crash after N calls."""

    def __init__(
        self, runs: list[SparkRunResult], *, crash_after: int
    ) -> None:
        self._runs = list(runs)
        self._crash_after = crash_after
        self.run_calls: list[str] = []
        # Same legacy consensus as _legacy_spark: the crash run and
        # its resume must key chunk digests identically.
        self.spark_stage_config = _LEGACY_CONFIG

    def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False) -> SparkRunResult:
        self.run_calls.append(video_file_id)
        if len(self.run_calls) > self._crash_after:
            raise RuntimeError("test: simulated crash")
        return self._runs.pop(0)


def test_resume_after_interruption_reuses_finished_chunks(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db, chunk_s=20.0, chunk_overlap_s=2.0)
    windows = plan_chunks(65.0, 20.0, 2.0)
    assert len(windows) == 4
    ctx = admit_job("resume-4", "resume-4", "video-resume-4")
    # One provider across both runs: a crash does not reset Meta ids.
    meta = ReplayMetaFilesClient()
    crashing = PipelineClients(
        spark=_CrashAfterCalls(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
            + _agreeing_runs("Disconnect the battery cable")
            + _agreeing_runs("Lift out the battery pack"),
            crash_after=6,
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _extract_chunked(
            ctx, windows, clip65,
            clients=crashing, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    assert len(crashing.spark.run_calls) == 7
    assert len(_jobs(lane_db)) == 4  # parent + 2 done + 1 crashed
    uploads_before_resume = len(meta.upload_calls)

    cuts: list[Path] = []
    real_cut = pipeline_mod.cut_chunk

    def _count_cut(src, window, dst, **kwargs):
        cuts.append(dst)
        return real_cut(src, window, dst, **kwargs)

    monkeypatch.setattr(pipeline_mod, "cut_chunk", _count_cut)
    resumed_clients = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Disconnect the battery cable")
            + _agreeing_runs("Lift out the battery pack")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=resumed_clients, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.number for s in stitched.steps] == [1, 2, 3, 4]
    # Chunks 0/1 re-cut locally to validate the resume (free), with
    # zero upload and zero Spark calls on resume.
    assert len(cuts) == 4
    assert len(meta.upload_calls) - uploads_before_resume == 2
    assert len(resumed_clients.spark.run_calls) == 6
    assert len(_jobs(lane_db)) == 6  # two new sub-jobs only
    parent_id = str(ctx.job_id)
    # The crashed chunk's re-attempt took a fresh idempotency key; the
    # finished chunks kept theirs and were never re-claimed.
    assert _claim_ids(lane_db) == [
        f"{parent_id}:chunk:0",
        f"{parent_id}:chunk:1",
        f"{parent_id}:chunk:2",
        f"{parent_id}:chunk:2:retry:1",
        f"{parent_id}:chunk:3",
    ]


def test_resume_refusal_says_what_the_run_knows(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    ctx = admit_job("resume-spent", "resume-spent", "video-resume-spent")
    crashing = PipelineClients(
        spark=_CrashAfterCalls(
            _disagreeing_runs()
            + _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws"),
            crash_after=6,
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _extract_chunked(
            ctx, windows, clip65,
            clients=crashing, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    with pytest.raises(_ChunkRetryRefused) as excinfo:
        _extract_chunked(
            ctx, windows, clip65,
            clients=rerun, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    # The earlier attempt crashed; this run never saw a disagreement,
    # so the refusal says so honestly instead of claiming one.
    assert str(excinfo.value) == (
        "step extraction failed: chunk 1 needs another attempt and the"
        " $0.26 per-video retry budget is spent"
    )
    assert rerun.spark.run_calls == []
    assert rerun.meta.upload_calls == []


def _saved_chunk_record(tmp_cache: Path, action: str) -> tuple[str, dict]:
    """(digest, record) of the saved chunk bundle holding one action.

    The resume digest binds the probed cut facts, which only the
    pipeline run observes — so tests locate a chunk's record by its
    bundle content, then recompute its digest from the recorded facts.
    """
    matches = []
    for path in sorted((tmp_cache / STAGE_CHUNK_BUNDLE).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        steps = record.get("bundle", {}).get("steps", [])
        if any(step.get("action") == action for step in steps):
            matches.append((path.stem, record))
    assert len(matches) == 1
    return matches[0]


def test_corrupt_bundle_reruns_only_that_chunk(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    from app.uploads import hash_file

    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert len(windows) == 2
    ctx = admit_job("resume-corrupt", "resume-corrupt", "video-resume-corrupt")
    parent_id = str(ctx.job_id)
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 2
    digest, record = _saved_chunk_record(tmp_cache, "Remove the rear screws")
    # The pipeline saves under the digest of the probed cut facts.
    assert digest == chunk_bundle_digest(
        parent_id, hash_file(clip65), 1,
        windows[1].start_s, windows[1].end_s, _LEGACY_CONFIG,
        chunk_tolerance_s=settings.contiguity_tolerance_s,
        **_tune_kwargs(settings),
        cut_start_s=record["cut_start_s"],
        cut_duration_s=record["cut_duration_s"],
    )
    bundle_path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    assert bundle_path.exists()
    bundle_path.write_text("{truncated", encoding="utf-8")

    uploads_before_rerun = len(meta.upload_calls)
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.action for s in stitched.steps] == [
        "Open the lid", "Remove the rear screws",
    ]
    # Only the corrupt chunk reran: one upload, three calls, and the
    # bundle file is valid again.
    assert len(meta.upload_calls) - uploads_before_rerun == 1
    assert len(rerun.spark.run_calls) == 3
    assert load_chunk_bundle(
        digest, DEFAULT_SPARK_CONFIG, ttl_s=settings.chunk_bundle_ttl_s
    ) is not None


def test_pre_fix_record_reruns_only_that_chunk(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert len(windows) == 2
    ctx = admit_job("resume-prefix", "resume-prefix", "video-resume-prefix")
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 2
    digest, record = _saved_chunk_record(tmp_cache, "Remove the rear screws")
    # Downgrade chunk 1's record to the pre-fix shape: format 1,
    # nominal duration, no cut facts — under the same key.
    bundle_path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    bundle_path.write_text(
        json.dumps(
            {"format": "partfidelity-chunk-bundle/1", "digest": digest,
             "stored_at": record["stored_at"], "bundle": record["bundle"]}
        ),
        encoding="utf-8",
    )
    assert load_chunk_bundle(
        digest, DEFAULT_SPARK_CONFIG, ttl_s=settings.chunk_bundle_ttl_s
    ) is None

    uploads_before_rerun = len(meta.upload_calls)
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.action for s in stitched.steps] == [
        "Open the lid", "Remove the rear screws",
    ]
    # Only the pre-fix chunk reran, and its record is rewritten with
    # the cut facts under the same digest.
    assert len(meta.upload_calls) - uploads_before_rerun == 1
    assert len(rerun.spark.run_calls) == 3
    rewritten = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert rewritten["format"] == CHUNK_BUNDLE_FORMAT
    assert rewritten["cut_start_s"] == record["cut_start_s"]
    assert rewritten["cut_duration_s"] == record["cut_duration_s"]


def test_recorded_cut_mismatch_reruns_only_that_chunk(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert len(windows) == 2
    ctx = admit_job("resume-cutedit", "resume-cutedit", "video-resume-cutedit")
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 2
    digest, record = _saved_chunk_record(tmp_cache, "Remove the rear screws")
    # Same key, valid but wrong cut facts: the record no longer
    # describes the bytes the rerun probes.
    bundle_path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    record["cut_duration_s"] = record["cut_duration_s"] + 5.0
    bundle_path.write_text(json.dumps(record), encoding="utf-8")

    uploads_before_rerun = len(meta.upload_calls)
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.action for s in stitched.steps] == [
        "Open the lid", "Remove the rear screws",
    ]
    assert len(meta.upload_calls) - uploads_before_rerun == 1
    assert len(rerun.spark.run_calls) == 3


# --- bundle records: digest scope, TTL, fail-closed reads ------------------


def test_bundle_digest_binds_parent_source_window_config(tmp_cache) -> None:
    base = {
        "parent_job_id": "parent-1",
        "source_hash": "aaa",
        "index": 2,
        "window_start_s": 116.0,
        "window_end_s": 176.0,
        "cut_start_s": 116.0,
        "cut_duration_s": 60.0,
    }
    first = chunk_bundle_digest(**base)
    assert chunk_bundle_digest(
        **{**base, "parent_job_id": "parent-2"}
    ) != first
    assert chunk_bundle_digest(**{**base, "source_hash": "bbb"}) != first
    assert chunk_bundle_digest(**{**base, "index": 3}) != first
    assert chunk_bundle_digest(**{**base, "window_start_s": 115.0}) != first
    assert chunk_bundle_digest(
        **base, config=SparkStageConfig(model_id="muse-spark-9.9")
    ) != first
    assert chunk_bundle_digest(
        **base, config=SparkStageConfig(spread_s=9.0)
    ) != first


def test_bundle_digest_changes_with_prompt_version(
    tmp_cache: Path, monkeypatch,
) -> None:
    from app import spark as spark_mod

    kwargs = {
        "parent_job_id": "parent-1",
        "source_hash": "aaa",
        "index": 0,
        "window_start_s": 0.0,
        "window_end_s": 60.0,
        "cut_start_s": 0.0,
        "cut_duration_s": 60.0,
    }
    first = chunk_bundle_digest(**kwargs)
    monkeypatch.setattr(spark_mod, "SPARK_PROMPT_VERSION", "l2-test")
    assert chunk_bundle_digest(**kwargs) != first


def test_config_change_invalidates_old_bundle(tmp_cache: Path) -> None:
    log: list = []
    steps = (make_step(start=10.0, end=20.0),)
    payloads = [make_payload(steps=steps) for _ in range(3)]
    bundle = extract_steps(
        RecordingContext(log), "vid",
        client=ScriptedSparkClient(
            log,
            [SparkRunResult(payload=p, cost_usd=Decimal("0.08"), latency_ms=9)
             for p in payloads],
        ),
    )
    old = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        old, bundle, now=1000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    new = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        config=SparkStageConfig(spread_s=9.0),
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    assert load_chunk_bundle(new, ttl_s=3600.0, now=1000.0) is None
    assert load_chunk_bundle(old, ttl_s=3600.0, now=1000.0) == bundle


def test_expired_bundle_is_dropped_and_deleted(tmp_cache: Path) -> None:
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        digest, bundle, now=1000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=4600.0) == bundle
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=4601.0) is None
    assert path.exists() is False


def test_future_dated_bundle_fails_closed_kept(tmp_cache) -> None:
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        digest, bundle, now=2000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None
    assert path.exists() is True


def test_mismatched_and_garbage_records_fail_closed(tmp_cache: Path) -> None:
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    stage_cache.cache_put(
        STAGE_CHUNK_BUNDLE, digest,
        {"format": CHUNK_BUNDLE_FORMAT, "digest": "0" * 64,
         "stored_at": 1000.0, "bundle": {},
         "cut_start_s": 0.0, "cut_duration_s": 60.0},
    )
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None
    stage_cache.cache_put(STAGE_CHUNK_BUNDLE, digest, {"format": "nope"})
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None


def test_saved_bundle_holds_no_footage_frames_or_keys(tmp_cache: Path) -> None:
    import json

    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        digest, bundle, now=1000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    record = json.loads(
        (tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json").read_text(
            encoding="utf-8"
        )
    )
    assert sorted(record) == [
        "bundle", "cut_duration_s", "cut_start_s",
        "digest", "format", "stored_at",
    ]
    text = json.dumps(record).lower()
    for banned in ("file_id", "api_key", "apikey", "frame", "footage",
                   "mp4", "authorization", "bearer"):
        assert banned not in text


def test_cut_window_record_round_trip_and_old_record_is_a_miss(
    tmp_cache: Path,
) -> None:
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-9", "aaa", 0, 58.0, 65.0,
        cut_start_s=58.0, cut_duration_s=7.0,
    )
    assert load_chunk_cut_window(digest, ttl_s=60.0) is None  # missing
    save_chunk_bundle(
        digest, bundle, now=1000.0, cut_start_s=58.0, cut_duration_s=7.0
    )
    assert load_chunk_cut_window(
        digest, ttl_s=60.0, now=1000.0
    ) == ChunkWindow(58.0, 65.0)
    assert load_chunk_bundle(digest, ttl_s=60.0, now=1000.0) == bundle
    # A pre-fix record (format 1, nominal duration, no cut facts) is a
    # cache miss for both loads, never reused.
    stage_cache.cache_put(
        STAGE_CHUNK_BUNDLE, digest,
        {"format": "partfidelity-chunk-bundle/1", "digest": digest,
         "stored_at": 1000.0, "bundle": bundle_to_json(bundle)},
    )
    assert load_chunk_cut_window(digest, ttl_s=60.0, now=1000.0) is None
    assert load_chunk_bundle(digest, ttl_s=60.0, now=1000.0) is None
    # The facts are mandatory and validated on the way in.
    with pytest.raises(TypeError):
        save_chunk_bundle(digest, bundle)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        save_chunk_bundle(digest, bundle, cut_start_s=-1.0, cut_duration_s=7.0)
    with pytest.raises(ValueError):
        save_chunk_bundle(digest, bundle, cut_start_s=58.0, cut_duration_s=0.0)


def test_bundle_digest_binds_the_decoded_cut_facts(tmp_cache) -> None:
    base = {
        "parent_job_id": "parent-1",
        "source_hash": "aaa",
        "index": 0,
        "window_start_s": 0.0,
        "window_end_s": 60.0,
        "cut_start_s": 0.0,
        "cut_duration_s": 60.0,
    }
    first = chunk_bundle_digest(**base)
    # The decoded extent re-keys the bundle: a cut that plays longer
    # (or opens earlier) never reuses the nominal record.
    assert chunk_bundle_digest(
        **{**base, "cut_duration_s": 60.2}
    ) != first
    assert chunk_bundle_digest(
        **{**base, "cut_start_s": 0.5}
    ) != first
    for bad in (
        {"cut_start_s": -1.0}, {"cut_duration_s": 0.0},
        {"cut_start_s": float("nan")}, {"cut_duration_s": float("inf")},
        {"cut_start_s": True}, {"cut_duration_s": "60"},
    ):
        with pytest.raises(ValueError):
            chunk_bundle_digest(**{**base, **bad})


def test_malformed_cut_metadata_is_a_cache_miss(tmp_cache: Path) -> None:
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    cases = [
        # Non-finite or impossible facts, including an end that
        # overflows to inf (1e308 + 1e308) and unconvertible JSON.
        (float("nan"), 60.0), (0.0, float("nan")),
        (float("inf"), 60.0), (0.0, float("inf")),
        (-1.0, 60.0), (0.0, 0.0), (0.0, -2.0),
        (1e308, 1e308), (True, 60.0), (0.0, False),
        ("0", 60.0), (0.0, "60"), (None, 60.0), (0.0, None),
        (10**1000, 60.0), (0.0, 10**1000),
    ]
    for start, duration in cases:
        stage_cache.cache_put(
            STAGE_CHUNK_BUNDLE, digest,
            {"format": CHUNK_BUNDLE_FORMAT, "digest": digest,
             "stored_at": 1000.0, "bundle": bundle_to_json(bundle),
             "cut_start_s": start, "cut_duration_s": duration},
        )
        assert load_chunk_bundle(
            digest, ttl_s=3600.0, now=1000.0
        ) is None
        assert load_chunk_cut_window(
            digest, ttl_s=3600.0, now=1000.0
        ) is None
    # Half the facts is malformed too.
    stage_cache.cache_put(
        STAGE_CHUNK_BUNDLE, digest,
        {"format": CHUNK_BUNDLE_FORMAT, "digest": digest,
         "stored_at": 1000.0, "bundle": bundle_to_json(bundle),
         "cut_start_s": 0.0},
    )
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None
    assert load_chunk_cut_window(digest, ttl_s=3600.0, now=1000.0) is None


def test_precision_collapsed_cut_facts_are_refused(tmp_cache: Path) -> None:
    # 1e308 + 1.0 rounds back to 1e308: a positive duration whose float
    # addition does not advance the start, so the facts describe no
    # window and must fail closed on read and refused on write.
    assert 1e308 + 1.0 == 1e308
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    stage_cache.cache_put(
        STAGE_CHUNK_BUNDLE, digest,
        {"format": CHUNK_BUNDLE_FORMAT, "digest": digest,
         "stored_at": 1000.0, "bundle": bundle_to_json(bundle),
         "cut_start_s": 1e308, "cut_duration_s": 1.0},
    )
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None
    assert load_chunk_cut_window(digest, ttl_s=3600.0, now=1000.0) is None
    with pytest.raises(ValueError):
        save_chunk_bundle(
            digest, bundle, cut_start_s=1e308, cut_duration_s=1.0
        )
    with pytest.raises(ValueError):
        chunk_bundle_digest(
            "parent-1", "aaa", 0, 0.0, 60.0,
            cut_start_s=1e308, cut_duration_s=1.0,
        )


def _extracted_bundle():
    log: list = []
    steps = (make_step(start=10.0, end=20.0),)
    payloads = [make_payload(steps=steps) for _ in range(3)]
    return extract_steps(
        RecordingContext(log), "vid",
        client=ScriptedSparkClient(
            log,
            [SparkRunResult(payload=p, cost_usd=Decimal("0.08"), latency_ms=9)
             for p in payloads],
        ),
    )


# --- raw evidence: every paid call survives consensus failure --------------


def test_disagreement_preserves_all_three_raws(tmp_cache: Path) -> None:
    log: list = []
    payloads = [
        make_payload(
            steps=(make_step(start=float(i * 10), end=float(i * 10 + 5),
                             action=f"Action {i}"),)
        )
        for i in range(3)
    ]
    results = [
        SparkRunResult(payload=p, cost_usd=Decimal("0.08"), latency_ms=9)
        for p in payloads
    ]
    # The sparse path still fails closed on wording splits (contiguous
    # phases mode would keep this disagreement as flagged segments).
    config = SparkStageConfig(span_mode="legacy")
    client = ScriptedSparkClient(log, results)
    client.spark_stage_config = config
    with pytest.raises(StepsDisagreement):
        extract_steps_cached(
            RecordingContext(log), "vid-disagree", client=client,
        )
    for index in range(3):
        digest = run_digest("job-l2-test", "vid-disagree", index, config)
        assert (tmp_cache / STAGE_SPARK_RUN / f"{digest}.json").exists()
    assert (
        tmp_cache / STAGE_SPARK_CONSENSUS
        / f"{consensus_digest('job-l2-test', 'vid-disagree', config)}.json"
    ).exists() is False


def test_uncached_extraction_writes_no_cache(tmp_cache: Path) -> None:
    log: list = []
    steps = (make_step(start=10.0, end=20.0),)
    payloads = [make_payload(steps=steps) for _ in range(3)]
    extract_steps(
        RecordingContext(log), "vid-live",
        client=ScriptedSparkClient(
            log,
            [SparkRunResult(payload=p, cost_usd=Decimal("0.08"), latency_ms=9)
             for p in payloads],
        ),
    )
    assert list(Path(tmp_cache).rglob("*.json")) == []


# --- settings: attempts, retry budget, bundle TTL --------------------------


def test_chunk_retry_settings_defaults() -> None:
    settings = load_settings({})
    assert settings.chunk_max_attempts == 2
    assert settings.chunk_retry_budget_usd == Decimal("0.26")
    assert settings.chunk_bundle_ttl_s == 7.0 * 24.0 * 3600.0


def test_chunk_retry_settings_from_env() -> None:
    settings = load_settings(
        {
            "PARTFIDELITY_CHUNK_MAX_ATTEMPTS": "3",
            "PARTFIDELITY_CHUNK_RETRY_BUDGET_USD": "0.52",
            "PARTFIDELITY_CHUNK_BUNDLE_TTL_S": "3600",
        }
    )
    assert settings.chunk_max_attempts == 3
    assert settings.chunk_retry_budget_usd == Decimal("0.52")
    assert settings.chunk_bundle_ttl_s == 3600.0
    assert load_settings(
        {"PARTFIDELITY_CHUNK_RETRY_BUDGET_USD": "0"}
    ).chunk_retry_budget_usd == Decimal("0")


@pytest.mark.parametrize(
    "env",
    [
        {"PARTFIDELITY_CHUNK_MAX_ATTEMPTS": "0"},
        {"PARTFIDELITY_CHUNK_MAX_ATTEMPTS": "-1"},
        {"PARTFIDELITY_CHUNK_MAX_ATTEMPTS": "abc"},
        {"PARTFIDELITY_CHUNK_RETRY_BUDGET_USD": "-1"},
        {"PARTFIDELITY_CHUNK_RETRY_BUDGET_USD": "free"},
        {"PARTFIDELITY_CHUNK_BUNDLE_TTL_S": "0"},
        {"PARTFIDELITY_CHUNK_BUNDLE_TTL_S": "abc"},
    ],
)
def test_chunk_retry_settings_reject_bad_values(env) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        load_settings(env)


# --- retry spend is durable ledger money, not count x price -------


def test_retry_spend_survives_a_price_change(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(
        tmp_path, lane_db, chunk_retry_budget_usd=Decimal("0.30"),
        job_cost_cap_usd=Decimal("5.00"),
    )
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert len(windows) == 2
    ctx = admit_job("spend-price", "spend-price", "video-spend-price")
    meta = ReplayMetaFilesClient()
    # Run 1 at the default $0.26 shares: chunk 0 disagrees, its retry
    # agrees with actuals exactly at the shares ($0.26 spent), and
    # chunk 1 crashes on its first call.
    pricey_retry = [
        _run([_payload_step(1, 0.5, 2.5, "Open the lid")], cost="0.09"),
        _run([_payload_step(1, 0.6, 2.6, "Open the lid")], cost="0.09"),
        _run([_payload_step(1, 0.4, 2.4, "Open the lid")], cost="0.08"),
    ]
    crashing = PipelineClients(
        spark=_CrashAfterCalls(
            _disagreeing_runs() + pricey_retry
            + _agreeing_runs("Remove the rear screws"),
            crash_after=6,
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _extract_chunked(
            ctx, windows, clip65,
            clients=crashing, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    # The provider repriced Spark to $0.09 a chunk. The digest binds
    # model/runs/spread, not shares, so chunk 0's bundle still hits —
    # but the $0.26 already spent still counts against the budget.
    cheap = _legacy_spark(_agreeing_runs("Remove the rear screws"))
    cheap.spark_stage_config = SparkStageConfig(
        run_shares=(Decimal("0.03"), Decimal("0.03"), Decimal("0.03")),
        span_mode="legacy",  # same digest scope as the crash run
    )
    rerun = PipelineClients(
        spark=cheap, sam=ReplaySamClient({}), meta=meta,
    )
    uploads_before_resume = len(meta.upload_calls)
    with pytest.raises(_ChunkRetryRefused) as excinfo:
        _extract_chunked(
            ctx, windows, clip65,
            clients=rerun, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    assert str(excinfo.value) == (
        "step extraction failed: chunk 1 needs another attempt and the"
        " $0.30 per-video retry budget is spent"
    )
    assert cheap.run_calls == []
    assert len(meta.upload_calls) == uploads_before_resume


def test_retry_spend_admits_when_actuals_undercut_count_price(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(
        tmp_path, lane_db, chunk_retry_budget_usd=Decimal("0.50")
    )
    windows = plan_chunks(65.0, 60.0, 2.0)
    ctx = admit_job("spend-actual", "spend-actual", "video-spend-actual")
    meta = ReplayMetaFilesClient()
    # One retry settled at $0.24 of actuals (3 x $0.08), under the
    # $0.26 count-price the old code charged for it.
    crashing = PipelineClients(
        spark=_CrashAfterCalls(
            _disagreeing_runs()
            + _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws"),
            crash_after=6,
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _extract_chunked(
            ctx, windows, clip65,
            clients=crashing, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    uploads_before_resume = len(meta.upload_calls)
    # $0.24 spent + the $0.2475 window estimate fits the $0.50
    # budget; a count-price $0.26 + $0.26 would refuse.
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.number for s in stitched.steps] == [1, 2]
    assert len(rerun.spark.run_calls) == 3
    assert len(meta.upload_calls) - uploads_before_resume == 1


# --- the attempt cap is durable across restarts --------------------


def test_resume_beyond_max_attempts_fails_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    settings = _settings(
        tmp_path, lane_db, chunk_retry_budget_usd=Decimal("2.00"),
        job_cost_cap_usd=Decimal("5.00"),
    )
    windows = plan_chunks(65.0, 60.0, 2.0)
    ctx = admit_job("spent-cap", "spent-cap", "video-spent-cap")
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _disagreeing_runs()
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    with pytest.raises((StepsDisagreement, _ChunkAttemptsExhausted)):
        _extract_chunked(
            ctx, windows, clip65,
            clients=first, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    assert len(first.spark.run_calls) == 6
    # Chunk 0 used both attempts with budget to spare; the resume must
    # not admit a third attempt.
    rerun = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    uploads_before_resume = len(meta.upload_calls)
    with pytest.raises(_ChunkAttemptsExhausted):
        _extract_chunked(
            ctx, windows, clip65,
            clients=rerun, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings,
        )
    assert rerun.spark.run_calls == []
    assert len(meta.upload_calls) == uploads_before_resume


# --- pending child settlement stops the retry ----------------------


def test_pending_child_settlement_stops_retry(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    clients = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _agreeing_runs("Open the lid")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("settle-stop", "settle-stop", "video-settle-stop")
    parent_id = str(ctx.job_id)
    real_settle = pipeline_mod._settle_failed

    def _pending_child(sub_ctx, error, *, settings):
        if str(sub_ctx.job_id) != parent_id:
            return "pending_settlement"
        return real_settle(sub_ctx, error, settings=settings)

    monkeypatch.setattr(pipeline_mod, "_settle_failed", _pending_child)
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "pending_settlement"
    assert result.error == (
        "step extraction failed: chunk 0 settlement is pending;"
        " retry not started (terminal settlement pending)"
    )
    # No retry started: three calls and one upload only.
    assert len(clients.spark.run_calls) == 3
    assert len(clients.meta.upload_calls) == 1
    states = sorted(row["state"] for row in _jobs(lane_db))
    assert states == ["active", "failed"]


def test_pending_child_settlement_propagates_on_unexpected_error(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    clients = PipelineClients(
        spark=_CrashAfterCalls(
            _agreeing_runs("Open the lid"), crash_after=0,
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("settle-err", "settle-err", "video-settle-err")
    parent_id = str(ctx.job_id)
    real_settle = pipeline_mod._settle_failed

    def _pending_child(sub_ctx, error, *, settings):
        if str(sub_ctx.job_id) != parent_id:
            return "pending_settlement"
        return real_settle(sub_ctx, error, settings=settings)

    monkeypatch.setattr(pipeline_mod, "_settle_failed", _pending_child)
    result = run_stages(
        ctx, "parent-file-id", clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=65.0,
    )
    assert result.state == "pending_settlement"
    assert len(clients.spark.run_calls) == 1


def test_interrupt_not_converted_when_child_settlement_pending(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    class _RaiseInterrupt:
        def __init__(self) -> None:
            self.run_calls: list[str] = []

        def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
            self.run_calls.append(video_file_id)
            raise KeyboardInterrupt("test: interrupt")

    settings = _settings(tmp_path, lane_db)
    clients = PipelineClients(
        spark=_RaiseInterrupt(),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("settle-ki", "settle-ki", "video-settle-ki")
    parent_id = str(ctx.job_id)
    real_settle = pipeline_mod._settle_failed

    def _pending_child(sub_ctx, error, *, settings):
        if str(sub_ctx.job_id) != parent_id:
            return "pending_settlement"
        return real_settle(sub_ctx, error, settings=settings)

    monkeypatch.setattr(pipeline_mod, "_settle_failed", _pending_child)
    with pytest.raises(KeyboardInterrupt):
        run_stages(
            ctx, "parent-file-id", clip65,
            clients=clients, lifecycle=lifecycle, uploads=uploads,
            replay=True, settings=settings, duration_s=65.0,
        )


# --- one claim per chunk, rechecked before any paid call -----------


def test_second_worker_pays_nothing_for_claimed_chunk(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    import threading

    settings = _settings(tmp_path, lane_db)
    windows = [plan_chunks(65.0, 60.0, 2.0)[0]]
    ctx = admit_job("claim-race", "claim-race", "video-claim-race")
    entered = threading.Event()
    release = threading.Event()
    inner_a = _legacy_spark(_agreeing_runs("Open the lid"))
    real_run = inner_a.run_steps

    def _blocking_run(video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
        result = real_run(video_file_id, clip_duration_s=clip_duration_s)
        entered.set()
        assert release.wait(timeout=30)
        return result

    inner_a.run_steps = _blocking_run
    # One provider for both workers, like production: separate replay
    # clients would mint colliding file ids and fail B at attach.
    shared_meta = ReplayMetaFilesClient()
    clients_a = PipelineClients(
        spark=inner_a, sam=ReplaySamClient({}),
        meta=shared_meta,
    )
    clients_b = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Open the lid")),
        sam=ReplaySamClient({}),
        meta=shared_meta,
    )
    errors: list[BaseException] = []
    done: list = []

    def _run_a() -> None:
        try:
            done.append(
                _extract_chunked(
                    ctx, windows, clip65,
                    clients=clients_a, lifecycle=lifecycle,
                    uploads=uploads, replay=True, settings=settings,
                )
            )
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)
        finally:
            entered.set()

    worker = threading.Thread(target=_run_a)
    worker.start()
    try:
        assert entered.wait(timeout=30)
        if errors:
            raise errors[0]
        # Worker A is mid-extraction holding the chunk claim: worker B
        # must pay nothing for the same chunk.
        with pytest.raises(_ChunkClaimHeld):
            _extract_chunked(
                ctx, windows, clip65,
                clients=clients_b, lifecycle=lifecycle,
                uploads=uploads, replay=True, settings=settings,
            )
        assert clients_b.spark.run_calls == []
        assert len(shared_meta.upload_calls) == 1
    finally:
        release.set()
        worker.join(timeout=30)
    assert not worker.is_alive()
    assert errors == []
    assert len(done) == 1
    assert len(clients_a.spark.run_calls) == 3
    assert len(shared_meta.upload_calls) == 1


# --- wild stored_at values fail closed to one rerun ----------------


def test_huge_stored_at_reruns_only_that_chunk(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    import json

    from app.uploads import hash_file

    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert len(windows) == 2
    ctx = admit_job("stored-huge", "stored-huge", "video-stored-huge")
    parent_id = str(ctx.job_id)
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 2
    digest, record = _saved_chunk_record(tmp_cache, "Remove the rear screws")
    assert digest == chunk_bundle_digest(
        parent_id, hash_file(clip65), 1,
        windows[1].start_s, windows[1].end_s, _LEGACY_CONFIG,
        chunk_tolerance_s=settings.contiguity_tolerance_s,
        **_tune_kwargs(settings),
        cut_start_s=record["cut_start_s"],
        cut_duration_s=record["cut_duration_s"],
    )
    bundle_path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    record["stored_at"] = 10**1000
    bundle_path.write_text(json.dumps(record), encoding="utf-8")

    uploads_before_rerun = len(meta.upload_calls)
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.action for s in stitched.steps] == [
        "Open the lid", "Remove the rear screws",
    ]
    assert len(meta.upload_calls) - uploads_before_rerun == 1
    assert len(rerun.spark.run_calls) == 3
    assert load_chunk_bundle(
        digest, DEFAULT_SPARK_CONFIG, ttl_s=settings.chunk_bundle_ttl_s
    ) is not None


def test_wild_timestamps_fail_closed_to_none(tmp_cache: Path) -> None:
    import json

    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        digest, bundle, now=1000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    path = tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["stored_at"] = 10**1000
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None
    record["stored_at"] = {"not": "a number"}
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None
    record["stored_at"] = 1000.0
    record["bundle"]["steps"][0]["start_s"] = 10**1000
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None


# --- chunk uploads drain on resume, attempt end, and sweep ---------


def test_resume_drains_prior_chunk_uploads(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    ctx = admit_job("drain-resume", "drain-resume", "video-drain-resume")
    meta = ReplayMetaFilesClient()
    # Run 1 records its deletes but performs none: the process died
    # after the bundle saves and before the cleanup sweep ran.
    logged: list[str] = []
    real_delete = pipeline_mod._delete_chunk_file

    def _record_only(file_id, **kwargs) -> None:
        logged.append(str(file_id))

    monkeypatch.setattr(
        pipeline_mod, "_delete_chunk_file", _record_only
    )
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 2
    leaked = sorted(str(key) for key in meta.files)
    assert len(leaked) == 2
    assert meta.delete_calls == []
    # Restore only this test's patch: undo() would also revert the
    # tmp_cache root the resume must read.
    monkeypatch.setattr(pipeline_mod, "_delete_chunk_file", real_delete)

    rerun = PipelineClients(
        spark=_legacy_spark([]),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    uploads_before_resume = len(meta.upload_calls)
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.number for s in stitched.steps] == [1, 2]
    assert len(meta.upload_calls) == uploads_before_resume
    assert rerun.spark.run_calls == []
    # Both prior uploads drained through the real deletion path.
    assert sorted(meta.delete_calls) == leaked
    assert meta.files == {}


def test_failed_attempt_upload_drained_before_retry(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    windows = [plan_chunks(65.0, 60.0, 2.0)[0]]
    ctx = admit_job("drain-eager", "drain-eager", "video-drain-eager")
    events: list[tuple] = []
    real_upload = pipeline_mod._upload_chunk_file

    def _log_upload(*args, **kwargs):
        file_id = real_upload(*args, **kwargs)
        events.append(
            ("upload", kwargs["idempotency_key"], str(file_id))
        )
        return file_id

    def _log_delete(file_id, **kwargs) -> None:
        events.append(("delete", str(file_id)))

    monkeypatch.setattr(pipeline_mod, "_upload_chunk_file", _log_upload)
    monkeypatch.setattr(pipeline_mod, "_delete_chunk_file", _log_delete)
    clients = PipelineClients(
        spark=_legacy_spark(
            _disagreeing_runs() + _agreeing_runs("Open the lid")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 1
    made = [event for event in events if event[0] == "upload"]
    gone = [event for event in events if event[0] == "delete"]
    assert len(made) == 2
    assert sorted(item[2] for item in made) == sorted(
        item[1] for item in gone
    )
    # The failed attempt's upload drained before the retry uploaded.
    first_id = made[0][2]
    assert events.index(("delete", first_id)) < events.index(made[1])


# --- claim contention never fails the shared parent -----------------


def test_claim_loser_returns_contention_while_winner_commits(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    import threading

    settings = _settings(tmp_path, lane_db)
    ctx = admit_job("claim-duo", "claim-duo", "video-claim-duo")
    parent_id = str(ctx.job_id)
    entered = threading.Event()
    release = threading.Event()
    inner_a = _legacy_spark(
        _agreeing_runs("Open the lid")
        + _agreeing_runs("Remove the rear screws")
    )
    real_run = inner_a.run_steps

    def _blocking_run(video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
        result = real_run(video_file_id, clip_duration_s=clip_duration_s)
        entered.set()
        assert release.wait(timeout=30)
        return result

    inner_a.run_steps = _blocking_run
    clients_a = PipelineClients(
        spark=inner_a, sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    clients_b = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    results: list = []
    errors: list[BaseException] = []

    def _run_a() -> None:
        try:
            results.append(
                run_stages(
                    ctx, "parent-file-id", clip65,
                    clients=clients_a, lifecycle=lifecycle,
                    uploads=uploads, replay=True, settings=settings,
                    duration_s=65.0,
                )
            )
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)
        finally:
            entered.set()

    worker = threading.Thread(target=_run_a)
    worker.start()
    try:
        assert entered.wait(timeout=30)
        if errors:
            raise errors[0]
        # Worker A is mid-extraction holding chunk 0's claim: worker B
        # must step aside without touching the shared parent.
        result_b = run_stages(
            ctx, "parent-file-id", clip65,
            clients=clients_b, lifecycle=lifecycle,
            uploads=uploads, replay=True, settings=settings,
            duration_s=65.0,
        )
        assert result_b.state == "contention"
        assert result_b.error == (
            "chunk 0 is being extracted by another worker; retry later"
        )
        assert result_b.guide is None
        assert clients_b.spark.run_calls == []
        assert clients_b.meta.upload_calls == []
        states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
        assert states[parent_id] == "active"
    finally:
        release.set()
        worker.join(timeout=60)
    assert not worker.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].state == "done", results[0].error
    assert results[0].guide is not None
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[parent_id] == "done"
    assert len(clients_a.spark.run_calls) == 6


# --- cancellation with pending child settlement ---------------------


def test_cancel_before_upload_with_pending_settlement_reports_pending(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod
    from app.uploads import prepare_preview

    settings = _settings(tmp_path, lane_db)
    clients = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("cancel-pre", "cancel-pre", "video-cancel-pre")
    parent_id = str(ctx.job_id)
    with open(clip65, "rb") as handle:
        preview = prepare_preview(
            ctx, handle, clip65.name, uploads=uploads,
            max_duration_s=settings.max_video_s,
        )
    real_cut = pipeline_mod.cut_chunk

    def _cancel_on_first_cut(src, window, dst, **kwargs):
        uploads.request_cancellation(ctx.job_id, db.utcnow())
        return real_cut(src, window, dst, **kwargs)

    monkeypatch.setattr(pipeline_mod, "cut_chunk", _cancel_on_first_cut)
    real_settle = pipeline_mod._settle_failed

    def _pending_child(sub_ctx, error, *, settings):
        if str(sub_ctx.job_id) != parent_id:
            return "pending_settlement"
        return real_settle(sub_ctx, error, settings=settings)

    monkeypatch.setattr(pipeline_mod, "_settle_failed", _pending_child)
    result = run_stages(
        ctx, "parent-file-id", preview.local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=preview.duration_s,
    )
    assert result.state == "pending_settlement"
    assert result.error == (
        "step extraction failed: chunk 0 settlement is pending;"
        " cancellation left it unsettled (terminal settlement pending)"
    )
    assert clients.spark.run_calls == []
    assert clients.meta.upload_calls == []
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[parent_id] == "failed"
    children = sorted(
        state for job_id, state in states.items() if job_id != parent_id
    )
    assert children == ["active"]


def test_cancel_after_upload_with_pending_settlement_reports_pending(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    from app import pipeline as pipeline_mod
    from app.uploads import prepare_preview

    settings = _settings(tmp_path, lane_db)
    clients = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    ctx = admit_job("cancel-post", "cancel-post", "video-cancel-post")
    parent_id = str(ctx.job_id)
    with open(clip65, "rb") as handle:
        preview = prepare_preview(
            ctx, handle, clip65.name, uploads=uploads,
            max_duration_s=settings.max_video_s,
        )
    real_upload = pipeline_mod._upload_chunk_file

    def _cancel_after_upload(*args, **kwargs):
        file_id = real_upload(*args, **kwargs)
        uploads.request_cancellation(ctx.job_id, db.utcnow())
        return file_id

    monkeypatch.setattr(
        pipeline_mod, "_upload_chunk_file", _cancel_after_upload
    )
    real_settle = pipeline_mod._settle_failed

    def _pending_child(sub_ctx, error, *, settings):
        if str(sub_ctx.job_id) != parent_id:
            return "pending_settlement"
        return real_settle(sub_ctx, error, settings=settings)

    monkeypatch.setattr(pipeline_mod, "_settle_failed", _pending_child)
    result = run_stages(
        ctx, "parent-file-id", preview.local_video,
        clients=clients, lifecycle=lifecycle, uploads=uploads, replay=True,
        settings=settings, duration_s=preview.duration_s,
    )
    assert result.state == "pending_settlement"
    assert result.error == (
        "step extraction failed: chunk 0 settlement is pending;"
        " cancellation left it unsettled (terminal settlement pending)"
    )
    assert clients.spark.run_calls == []
    assert len(clients.meta.upload_calls) == 1
    assert len(clients.meta.delete_calls) == 1
    states = {str(row["id"]): row["state"] for row in _jobs(lane_db)}
    assert states[parent_id] == "failed"
    children = sorted(
        state for job_id, state in states.items() if job_id != parent_id
    )
    assert children == ["active"]


# --- deeply nested corrupt JSON is a per-chunk cache miss ------------


def _write_nested_json(path: Path, depth: int = 20000) -> None:
    """Overwrite one cache file with JSON nested past the parser limit."""
    path.write_text("[" * depth + "]" * depth, encoding="utf-8")


def test_deeply_nested_bundle_json_is_a_cache_miss(tmp_cache: Path) -> None:
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-1", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        digest, bundle, now=1000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    _write_nested_json(tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json")
    assert load_chunk_bundle(digest, ttl_s=3600.0, now=1000.0) is None


def test_deeply_nested_resumed_bundle_is_a_cache_miss(
    tmp_cache: Path,
) -> None:
    bundle = bundle_from_json(bundle_to_json(_extracted_bundle()))
    digest = chunk_bundle_digest(
        "parent-2", "aaa", 0, 0.0, 60.0,
        cut_start_s=0.0, cut_duration_s=60.0,
    )
    save_chunk_bundle(
        digest, bundle, now=1000.0, cut_start_s=0.0, cut_duration_s=60.0
    )
    _write_nested_json(tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json")
    assert (
        _load_resumed_bundle(digest, DEFAULT_SPARK_CONFIG, 3600.0) is None
    )


def test_deeply_nested_bundle_reruns_only_that_chunk(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    from app.uploads import hash_file

    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert len(windows) == 2
    ctx = admit_job("stored-nested", "stored-nested", "video-stored-nested")
    parent_id = str(ctx.job_id)
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert len(stitched.steps) == 2
    digest, record = _saved_chunk_record(tmp_cache, "Remove the rear screws")
    assert digest == chunk_bundle_digest(
        parent_id, hash_file(clip65), 1,
        windows[1].start_s, windows[1].end_s, _LEGACY_CONFIG,
        chunk_tolerance_s=settings.contiguity_tolerance_s,
        **_tune_kwargs(settings),
        cut_start_s=record["cut_start_s"],
        cut_duration_s=record["cut_duration_s"],
    )
    _write_nested_json(tmp_cache / STAGE_CHUNK_BUNDLE / f"{digest}.json")
    uploads_before_rerun = len(meta.upload_calls)
    rerun = PipelineClients(
        spark=_legacy_spark(_agreeing_runs("Remove the rear screws")),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [s.action for s in stitched.steps] == [
        "Open the lid", "Remove the rear screws",
    ]
    assert len(meta.upload_calls) - uploads_before_rerun == 1
    assert len(rerun.spark.run_calls) == 3
    assert load_chunk_bundle(
        digest, DEFAULT_SPARK_CONFIG, ttl_s=settings.chunk_bundle_ttl_s
    ) is not None


# --- chunk extent: each chunk's prompt and consensus know its window ------


class _RecordingReplaySpark(ReplaySparkClient):
    """Replay runs that also record the clip duration of every call."""

    def __init__(self, runs: list[SparkRunResult]) -> None:
        super().__init__(runs)
        self.durations: list = []

    def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
        self.durations.append(clip_duration_s)
        return super().run_steps(
            video_file_id, clip_duration_s=clip_duration_s
        )


def test_chunked_contiguous_extent_excludes_overlong_run(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
) -> None:
    """Chunk 0 survives an over-long run: each call carries its window.

    The 65 s clip cuts into [0, 60] and [58, 65]; chunk 0's three
    scripted runs hold one over-long partition (to 100 s). The chunk
    still extracts — the run drops from consensus instead of
    stretching the tail into a ContiguityError — and every call
    observed its own chunk's duration for the prompt.
    """
    settings = _settings(tmp_path, lane_db, step_span_mode="contiguous")
    windows = plan_chunks(65.0, 60.0, 2.0)
    assert [(w.start_s, w.end_s) for w in windows] == [(0.0, 60.0), (58.0, 65.0)]
    ctx = admit_job("extent-chunked", "extent-chunked", "video-extent-chunked")
    good60 = [
        _payload_step(
            1, 0.0, 20.0, "Introduce laptop on table",
            [("laptop lid", "center")],
        ),
        _payload_step(
            2, 20.0, 40.0, "Remove the rear screws",
            [("screw", "rear edge")],
        ),
        _payload_step(
            3, 40.0, 60.0, "Disconnect the battery cable",
            [("battery cable", "lower right")],
        ),
    ]
    overlong = good60 + [
        _payload_step(
            4, 60.0, 100.0, "Pry off the base cover",
            [("base cover", "edge")],
        )
    ]
    good7 = [
        _payload_step(
            1, 0.0, 3.0, "Pry off the base cover",
            [("base cover", "edge")],
        ),
        _payload_step(
            2, 3.0, 7.0, "Lift out the battery",
            [("battery", "bay")],
        ),
    ]
    spark = _RecordingReplaySpark(
        [_run(good60), _run(good60), _run(overlong)]
        + [_run(good7)] * 3
    )
    clients = PipelineClients(
        spark=spark,
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=clients, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert spark.durations == [60.0] * 3 + [7.0] * 3
    require_contiguous_cover(
        stitched.steps, 65.0,
        tolerance_s=settings.contiguity_tolerance_s,
    )
    assert stitched.steps[0].start_s == 0.0
    assert stitched.steps[-1].end_s == 65.0


def _decoded_duration(path: Path) -> float:
    """Independent oracle: raw ffprobe format duration, never app.probe_cut."""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_entries", "format=duration", str(path),
        ],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def test_short_chunks_use_decoded_durations(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip61av: Path, tmp_cache: Path,
) -> None:
    """Short chunks prompt, validate, and record the decoded duration.

    The 0.5 s nominal middle chunk of a 61 s AV clip decodes longer:
    its end falls mid-frame on the 5 fps grid, so the straddling
    video frame is included fully (a structural floor, not packet
    phase). The 1.5 s last chunk is EOF-capped back to nominal — no
    content past EOF exists — and exercises the degenerate regime.
    """
    settings = _settings(tmp_path, lane_db, step_span_mode="contiguous")
    windows = [
        ChunkWindow(0.0, 59.0),
        ChunkWindow(59.0, 59.5),
        ChunkWindow(59.5, 61.0),
    ]
    # Independent expectation: cut the same windows test-side and probe
    # the bytes with raw ffprobe.
    cuts = [tmp_path / f"expect{i}.mp4" for i in range(3)]
    for window, cut in zip(windows, cuts):
        cut_chunk(clip61av, window, cut, timeout_s=120.0)
    first_len, second_len, third_len = (_decoded_duration(cut) for cut in cuts)
    # The premise: the short middle cut decodes longer by at least the
    # 0.1 s straddling frame (margin 2x under that floor).
    assert second_len - 0.5 > 0.05
    ctx = admit_job("extent-short", "extent-short", "video-extent-short")
    good59 = [
        _payload_step(
            1, 0.0, 20.0, "Introduce laptop on table",
            [("laptop lid", "center")],
        ),
        _payload_step(
            2, 20.0, 40.0, "Remove the rear screws",
            [("screw", "rear edge")],
        ),
        _payload_step(
            3, 40.0, 59.0, "Disconnect the battery cable",
            [("battery cable", "lower right")],
        ),
    ]
    good_middle = [
        _payload_step(
            1, 0.0, second_len / 2, "Pry off the base cover",
            [("base cover", "edge")],
        ),
        _payload_step(
            2, second_len / 2, second_len + 0.01, "Lift out the battery",
            [("battery", "bay")],
        ),
    ]
    good_last = [
        _payload_step(
            1, 0.0, third_len / 2, "Peel off the screen bezel",
            [("bezel", "edge")],
        ),
        _payload_step(
            2, third_len / 2, third_len + 0.01, "Unplug the display cable",
            [("display cable", "hinge")],
        ),
    ]
    spark = _RecordingReplaySpark(
        [_run(good59)] * 3 + [_run(good_middle)] * 3 + [_run(good_last)] * 3
    )
    clients = PipelineClients(
        spark=spark,
        sam=ReplaySamClient({}),
        meta=ReplayMetaFilesClient(),
    )
    stitched = _extract_chunked(
        ctx, windows, clip61av,
        clients=clients, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert spark.durations == [first_len] * 3 + [second_len] * 3 + [third_len] * 3
    parent_id = str(ctx.job_id)
    conn = db.connect(lane_db)
    try:
        recorded = {
            row["claim_id"]: row["duration_s"]
            for row in conn.execute(
                "SELECT claim_id, duration_s FROM uploads WHERE claim_id LIKE ?",
                (f"{parent_id}:%",),
            ).fetchall()
        }
    finally:
        conn.close()
    assert recorded[f"{parent_id}:chunk:0"] == first_len
    assert recorded[f"{parent_id}:chunk:1"] == second_len
    assert recorded[f"{parent_id}:chunk:2"] == third_len
    assert stitched.steps[0].start_s == 0.0
    assert stitched.steps[-1].end_s == 61.0


def test_resumed_chunk_stitches_at_its_recorded_cut_start(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, clip65: Path, tmp_cache: Path,
    monkeypatch,
) -> None:
    """Fresh and resumed runs stitch identically when the cut's real
    start differs from planned: the record carries the real window."""
    from app import pipeline as pipeline_mod

    settings = _settings(tmp_path, lane_db)
    windows = plan_chunks(65.0, 60.0, 2.0)
    real_probe = pipeline_mod.probe_cut

    def _shifted(path, *, timeout_s):
        probed = real_probe(path, timeout_s=timeout_s)
        return CutInfo(
            start_s=probed.start_s + 0.5,
            duration_s=probed.duration_s,
        )

    monkeypatch.setattr(pipeline_mod, "probe_cut", _shifted)
    ctx = admit_job("resume-window", "resume-window", "video-resume-window")
    meta = ReplayMetaFilesClient()
    first = PipelineClients(
        spark=_legacy_spark(
            _agreeing_runs("Open the lid")
            + _agreeing_runs("Remove the rear screws")
        ),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    stitched = _extract_chunked(
        ctx, windows, clip65,
        clients=first, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    uploads_before_rerun = len(meta.upload_calls)
    rerun = PipelineClients(
        spark=_legacy_spark([]),
        sam=ReplaySamClient({}),
        meta=meta,
    )
    again = _extract_chunked(
        ctx, windows, clip65,
        clients=rerun, lifecycle=lifecycle, uploads=uploads,
        replay=True, settings=settings,
    )
    assert [(s.start_s, s.end_s) for s in again.steps] == [
        (s.start_s, s.end_s) for s in stitched.steps
    ]
    assert len(rerun.spark.run_calls) == 0  # pure resume, no Spark calls
    assert len(meta.upload_calls) == uploads_before_rerun
    second = [s for s in again.steps if s.action == "Remove the rear screws"][0]
    assert second.start_s == 59.0  # median 0.5 over the recorded 58.5 start
