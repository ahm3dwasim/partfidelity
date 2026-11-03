"""Parent deadline extension for chunked live runs.

A 4-chunk live extraction runs ~5-7 wall minutes; the frozen 240 s job
deadline expired the parent mid-run (in one live retry the parent expired at
the chunk-3 admission, its later reserves refused, and the terminal
settlement landed pending). The service now extends a multi-chunk live
parent before building its live clients. All tests offline: fake
clients, no network, no key.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread
from time import sleep
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from app.hazards import refusal_copy

from app import db, service
from app.admission import admit_job
from app.contracts import JOB_DEADLINE_S, MetaFileId, MetaUpload
from app.ledger import (
    ContentionError,
    LifecycleRepositoryImpl,
    extend_job_deadline,
)
from app.pipeline import (
    PipelineClients,
    RecordingMetaClient,
    RecordingSparkClient,
    refusal_text_for_job,
)
from app.ledger import job_reservation_spend_usd, list_call_records
from app.readback import ReadbackResult
from app.sam import SamMask, SamStillResult
from app.settings import ProviderSettings, Settings, load_settings
from app.spark import SparkRunResult, SparkStageConfig
from app.uploads import confirm_and_send, hash_file, prepare_preview
from app.uploads_store import UploadStoreImpl

FAKE_KEY = "chunk-deadline-test-key-9f8e6d5c4b3a"


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
def live_settings(tmp_path: Path, lane_db: Path) -> Settings:
    # Sparse scripted runs predate the contiguous-phase contract: these
    # deadline tests pin the legacy rollback mode, and the 65 s clips
    # pin two-chunk planning (whole-video would single them).
    return Settings(
        data_root=tmp_path / "data",
        db_path=lane_db,
        production=False,
        meta_api_key=FAKE_KEY,
        pipeline_mode="live",
        providers=ProviderSettings(mode="live"),
        step_span_mode="legacy",
        whole_video_max_s=60.0,
    )


def _deadline_of(database: Path, job_id: str) -> str:
    conn = db.connect(database)
    try:
        row = conn.execute(
            "SELECT deadline_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        assert row is not None
        return str(row["deadline_at"])
    finally:
        conn.close()


def test_extend_job_deadline_pushes_active_job(lane_db: Path) -> None:
    ctx = admit_job("anon-ext", "ip-ext", "video-ext")
    before = _deadline_of(lane_db, str(ctx.job_id))
    assert extend_job_deadline(ctx.job_id, 240.0, path=lane_db) is True
    pushed = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    assert pushed - db.parse_dt(before) == timedelta(seconds=240)


def test_extend_job_deadline_never_resurrects(lane_db: Path) -> None:
    ctx = admit_job("anon-term", "ip-term", "video-term")
    ctx.finish("done")
    before = _deadline_of(lane_db, str(ctx.job_id))
    assert extend_job_deadline(ctx.job_id, 240.0, path=lane_db) is False
    assert _deadline_of(lane_db, str(ctx.job_id)) == before


def test_extend_job_deadline_rejects_bad_amounts(lane_db: Path) -> None:
    ctx = admit_job("anon-bad", "ip-bad", "video-bad")
    for bad in (0, -1, True, "240", float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            extend_job_deadline(ctx.job_id, bad, path=lane_db)  # type: ignore[arg-type]


def test_extend_job_deadline_refuses_past_deadline(lane_db: Path) -> None:
    """An active job past its deadline is expired, not extended."""
    ctx = admit_job("anon-past", "ip-past", "video-past")
    _force_past_deadline(lane_db, str(ctx.job_id))
    before = _deadline_of(lane_db, str(ctx.job_id))
    assert extend_job_deadline(ctx.job_id, 240.0, path=lane_db) is False
    assert _deadline_of(lane_db, str(ctx.job_id)) == before


def test_extend_refuses_when_lock_held_past_deadline(lane_db: Path) -> None:
    """Expiry is judged after the write lock is held, not before it.

    The extension blocks on a held write lock while the deadline
    passes; judging against a timestamp from before the wait would
    resurrect the expired job.
    """
    ctx = admit_job("anon-race", "ip-race", "video-race")
    _force_deadline_in(lane_db, str(ctx.job_id), 2.0)
    before = _deadline_of(lane_db, str(ctx.job_id))
    holder = db.connect(lane_db)
    holder.execute("BEGIN IMMEDIATE")
    outcome: dict[str, object] = {}

    def _extend() -> None:
        try:
            outcome["result"] = extend_job_deadline(
                ctx.job_id, 240.0, path=lane_db
            )
        except Exception as error:
            outcome["error"] = error

    worker = Thread(target=_extend, daemon=True)
    worker.start()
    try:
        deadline = db.parse_dt(before)
        while db.utcnow() <= deadline:
            sleep(0.05)
        assert worker.is_alive(), "extension returned before the lock released"
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "extension never finished after release"
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["result"] is False
    assert _deadline_of(lane_db, str(ctx.job_id)) == before


def test_extend_job_deadline_refuses_beyond_cap(lane_db: Path) -> None:
    """A push past created + 240 s + cap is refused, not clamped."""
    ctx = admit_job("anon-cap", "ip-cap", "video-cap")
    before = _deadline_of(lane_db, str(ctx.job_id))
    assert (
        extend_job_deadline(ctx.job_id, 100.0, path=lane_db, max_extra_s=50.0)
        is False
    )
    assert _deadline_of(lane_db, str(ctx.job_id)) == before
    assert (
        extend_job_deadline(ctx.job_id, 50.0, path=lane_db, max_extra_s=50.0)
        is True
    )
    pushed = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    assert pushed - db.parse_dt(before) == timedelta(seconds=50)


class _FakeLiveMeta:
    """Live-shaped Meta client: canned ids, real delete-then-gone reads."""

    def __init__(self) -> None:
        self.files: dict[str, MetaUpload] = {}
        self._counter = 0

    def upload_file(
        self, path: Path, *, expires_after_hours: int, idempotency_key: str
    ) -> MetaUpload:
        _ = (path, expires_after_hours, idempotency_key)
        self._counter += 1
        found = MetaUpload(
            file_id=MetaFileId(f"live-meta-{self._counter}"),
            expires_at=db.utcnow() + timedelta(hours=1),
        )
        self.files[str(found.file_id)] = found
        return found

    def find_upload(self, idempotency_key: str) -> MetaUpload | None:
        _ = idempotency_key
        return None

    def delete_file(self, file_id: MetaFileId) -> None:
        self.files.pop(str(file_id), None)

    def get_file(self, file_id: MetaFileId) -> MetaUpload | None:
        return self.files.get(str(file_id))


def _step(number: int, start: float, end: float, action: str, parts, kind: str = "repair") -> dict:
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


def _payload(steps: list[dict]) -> dict:
    return {"device_guess": "laptop", "confidence": 0.82, "steps": steps}


class _FakeChunkSpark:
    """Scripted step runs, consumed in order across every chunk."""

    # Legacy extraction like the legacy settings these tests run
    # under: the sparse payloads predate contiguous phases, and the
    # known chunk extent would exclude them outright.
    spark_stage_config = SparkStageConfig(span_mode="legacy")

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)
        self.calls: list[str] = []

    def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False) -> SparkRunResult:
        self.calls.append(video_file_id)
        assert self._payloads, "chunk spark ran out of scripted runs"
        return SparkRunResult(
            payload=self._payloads.pop(0),
            cost_usd=Decimal("0.03"),
            latency_ms=50,
        )


class _FakeParentSpark:
    """Parent bundle half: chunked extraction never calls run_steps here."""

    def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
        raise AssertionError("chunked path must not use the parent spark client")

    def read_back(self, image_path: Path, phrase: str, position_hint: str):
        _ = (image_path, phrase, position_hint)
        return ReadbackResult(
            answer="yes", cost_usd=Decimal("0.01"), latency_ms=30
        )


class _FakeLiveSam:
    """One confident box per still, masked at the calling frame's own size."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def segment_still(
        self, image_path: Path, phrase: str, *, include_confidence: bool = True
    ) -> SamStillResult:
        _ = include_confidence
        self.calls.append(phrase)
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert frame is not None
        height, width = frame.shape[:2]
        box = (10, 10, min(50, width), min(50, height))
        mask = np.zeros((height, width), np.uint8)
        mask[box[1] : box[3], box[0] : box[2]] = 255
        ok, encoded = cv2.imencode(".png", mask)
        assert ok
        return SamStillResult(
            masks=(SamMask(box_xyxy=box, c=0.80, mask_png=bytes(encoded)),)
        )


def _confirm_65s_job(
    uploads: UploadStoreImpl, tmp_path: Path, meta: _FakeLiveMeta | None = None
):
    """Admit, preview (65 s, two chunks), and confirm one live upload.

    The caller may share one fake Meta client with the later pipeline
    bundle, so chunk uploads mint ids distinct from the preview's.
    """
    clip = service.write_sample_clip(tmp_path / "sixtyfive.mp4", seconds=65.0, fps=5)
    ctx = admit_job("anon-chunk", "ip-chunk", hash_file(clip))
    with clip.open("rb") as handle:
        preview = prepare_preview(ctx, handle, clip.name, uploads=uploads)
    assert preview.duration_s > 60.0
    confirm_and_send(
        ctx.job_id,
        uploads=uploads,
        client=RecordingMetaClient(meta or _FakeLiveMeta(), ctx),
    )
    return ctx


def _force_deadline_in(database: Path, job_id: str, seconds: float) -> None:
    """Move one job's deadline to ``seconds`` from now, leaving it active."""
    forced = db.format_dt(db.utcnow() + timedelta(seconds=seconds))
    conn = db.connect(database)
    try:
        conn.execute(
            "UPDATE jobs SET deadline_at = ? WHERE id = ?", (forced, job_id)
        )
        conn.commit()
    finally:
        conn.close()


def _force_past_deadline(database: Path, job_id: str) -> None:
    """Simulate the live condition: past deadline, still active, no admit since."""
    _force_deadline_in(database, job_id, -5.0)


def _chunk_payloads() -> list[dict]:
    first = _payload(
        [
            _step(1, 1.0, 20.0, "Remove the rear screws", [("fan", "left side")]),
            _step(
                2, 20.0, 40.0, "Disconnect the battery cable",
                [("battery", "lower right")],
            ),
        ]
    )
    second = _payload(
        [
            _step(1, 0.5, 3.5, "Install the battery", [("battery", "lower right")]),
            _step(2, 3.5, 6.5, "Seat the fan cable", [("fan", "left side")]),
        ]
    )
    return [first] * 3 + [second] * 3


def test_chunked_live_parent_with_tight_deadline_finishes_done(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent that would expire mid-extraction still completes its guide."""
    import app.pipeline as pipeline_mod

    shared_meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, shared_meta)
    _force_deadline_in(lane_db, str(ctx.job_id), 30.0)

    chunk_spark = _FakeChunkSpark(_chunk_payloads())
    stage_during: list[str | None] = []
    actual_run_steps = chunk_spark.run_steps

    def checking_run_steps(*args, **kwargs):
        stage_during.append(pipeline_mod.stage_for_job(str(ctx.job_id)))
        return actual_run_steps(*args, **kwargs)

    monkeypatch.setattr(chunk_spark, "run_steps", checking_run_steps)
    actual_cut = pipeline_mod.cut_chunk

    def checking_cut(*args, **kwargs):
        stage_during.append(pipeline_mod.stage_for_job(str(ctx.job_id)))
        return actual_cut(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "cut_chunk", checking_cut)
    monkeypatch.setattr(
        pipeline_mod,
        "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(spark=chunk_spark),
    )
    parent_sam = _FakeLiveSam()
    service.set_client_factory(
        lambda settings: PipelineClients(
            spark=_FakeParentSpark(),  # type: ignore[arg-type]
            sam=parent_sam,  # type: ignore[arg-type]
            meta=shared_meta,  # type: ignore[arg-type]
        )
    )
    try:
        result = service.run_upload_job(
            ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "done", result.error
    assert len(chunk_spark.calls) == 6
    assert stage_during and all(stage == "extract" for stage in stage_during)
    assert parent_sam.calls, "SAM stage must run on the parent after stitching"


@pytest.mark.parametrize("failures,expected_state", ((1, "done"), (2, "failed")))
def test_extract_service_retry_has_own_receipt_and_true_reason(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch, failures: int, expected_state: str,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())
    original = spark.run_steps
    calls = []

    def flaky(*args, **kwargs):
        calls.append(args[0])
        if len(calls) <= failures:
            payload = "INDETERMINATE: timeout" if len(calls) == 2 else {}
            return SparkRunResult(payload, Decimal("0.03"), 50, "timeout")  # type: ignore[arg-type]
        return original(*args, **kwargs)

    monkeypatch.setattr(spark, "run_steps", flaky)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(
            spark=RecordingSparkClient(spark, sub_ctx)
        ),
    )
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == expected_state, result.error
    with db.connect(lane_db) as conn:
        jobs = conn.execute(
            "SELECT id, anon_id FROM jobs WHERE anon_id IN (?, ?)",
            (f"{ctx.job_id}/chunk/0", f"{ctx.job_id}/extract/retry/0"),
        ).fetchall()
    assert len(jobs) == 2
    receipts = [
        (row["anon_id"], receipt.status)
        for row in jobs
        for receipt in list_call_records(row["id"], path=lane_db)
        if receipt.stage == "extract"
    ]
    assert (f"{ctx.job_id}/chunk/0", "failed") in receipts
    assert (f"{ctx.job_id}/extract/retry/0", "succeeded" if failures == 1 else "failed") in receipts
    if failures == 2:
        assert len(calls) == 2
        assert "did not answer properly" in refusal_text_for_job(str(ctx.job_id), data_root=settings.data_root)
    else:
        assert len(calls) == 7


def test_direct_video_extract_retry_wiring(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    clip = service.write_sample_clip(tmp_path / "short.mp4", seconds=8.0, fps=5)
    ctx = admit_job("anon-direct", "ip-direct", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark([_chunk_payloads()[3]] * 3)
    spark.read_back = _FakeParentSpark().read_back  # type: ignore[attr-defined]
    original = spark.run_steps
    calls: list[str] = []

    def flaky(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            return SparkRunResult({}, Decimal("0.03"), 50, "timeout")
        return original(*args, **kwargs)

    monkeypatch.setattr(spark, "run_steps", flaky)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(
            spark=RecordingSparkClient(spark, sub_ctx)
        ),
    )
    service.set_client_factory(lambda settings: PipelineClients(
        spark=spark, sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "done", result.error
    assert len(calls) == 4
    with db.connect(lane_db) as conn:
        child = conn.execute(
            "SELECT id, state FROM jobs WHERE anon_id = ?",
            (f"{ctx.job_id}/extract/retry/0",),
        ).fetchone()
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE anon_id LIKE ?",
            (f"{ctx.job_id}/chunk/%",),
        ).fetchone()[0]
    assert child is not None and child["state"] == "done"
    assert chunk_count == 0
    assert any(
        record.stage == "extract" and record.status == "succeeded"
        for record in list_call_records(child["id"], path=lane_db)
    )


def test_schema_invalid_retry_settles_chunk_child_and_parent(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())
    calls: list[str] = []

    def invalid(*args, **kwargs):
        calls.append(args[0])
        return SparkRunResult({}, Decimal("0.03"), 50,
                              "timeout" if len(calls) == 1 else None)

    monkeypatch.setattr(spark, "run_steps", invalid)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(
            spark=RecordingSparkClient(spark, sub_ctx)
        ),
    )
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed" and len(calls) == 2
    with db.connect(lane_db) as conn:
        row = conn.execute(
            "SELECT id, state FROM jobs WHERE anon_id = ?",
            (f"{ctx.job_id}/extract/retry/0",),
        ).fetchone()
    assert row is not None and row["state"] == "failed"
    assert job_reservation_spend_usd(row["id"], path=lane_db) == (
        Decimal("0.03"), Decimal("0")
    )
    assert any(
        record.stage == "extract" and record.estimated_cost_usd == Decimal("0.03")
        for record in list_call_records(row["id"], path=lane_db)
    )


@pytest.mark.parametrize("block", ("cap", "job_cap", "deadline", "cancel"))
def test_extract_service_retry_stops_before_second_call(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch, block: str,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{
        **live_settings.__dict__, "extract_retry_backoff_s": 0.0,
        "chunk_retry_budget_usd": Decimal("0") if block == "cap" else live_settings.chunk_retry_budget_usd,
        "job_cost_cap_usd": Decimal("0.50") if block == "job_cap" else live_settings.job_cost_cap_usd,
    })
    spark = _FakeChunkSpark(_chunk_payloads())
    calls = []
    current_child = []

    def failing(*args, **kwargs):
        calls.append(args[0])
        if block == "deadline":
            _force_deadline_in(lane_db, str(current_child[0].job_id), 5.0)
        if block == "cancel":
            with db.connect(lane_db) as conn:
                conn.execute("UPDATE uploads SET cancel_requested_at = ? WHERE job_id = ?",
                             (db.format_dt(db.utcnow()), str(ctx.job_id)))
        return SparkRunResult({}, Decimal("0.03"), 50,
                              "timeout")

    monkeypatch.setattr(spark, "run_steps", failing)

    def provider(settings, sub_ctx, **kwargs):
        current_child[:] = [sub_ctx]
        return SimpleNamespace(spark=RecordingSparkClient(spark, sub_ctx))

    monkeypatch.setattr(pipeline_mod, "select_providers", provider)
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed"
    assert len(calls) == 1
    with db.connect(lane_db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE anon_id LIKE ?",
            (f"{ctx.job_id}/extract/retry/%",),
        ).fetchone()[0] == 0


@pytest.mark.parametrize("cause", ("local", "raises"))
def test_any_non_disagreement_extract_failure_shows_the_service_reason(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch, cause: str,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())

    def failing(*args, **kwargs):
        if cause == "raises":
            raise RuntimeError("provider returned something odd")
        return SparkRunResult({}, Decimal("0.03"), 50, "local")  # type: ignore[arg-type]

    monkeypatch.setattr(spark, "run_steps", failing)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(
            spark=RecordingSparkClient(spark, sub_ctx)
        ),
    )
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed"
    assert refusal_text_for_job(
        str(ctx.job_id), data_root=settings.data_root
    ) == refusal_copy("extract_service_failure")


def test_existing_refusal_marker_is_preserved(lane_db: Path, tmp_path: Path) -> None:
    from app.pipeline import _write_refusal_marker

    ctx = admit_job("original-reason", "original-reason", "original-reason")
    marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"
    marker.write_text('{"kind":"cancelled"}', encoding="utf-8")
    _write_refusal_marker(ctx, ctx.job_id, "extract_service_failure")
    assert marker.read_text(encoding="utf-8") == '{"kind":"cancelled"}'


def test_interrupted_refusal_write_does_not_publish_partial_marker(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    ctx = admit_job("write-interrupt", "write-interrupt", "write-interrupt")
    marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"
    original_fsync = pipeline_mod.os.fsync

    def interrupted(_fd: int) -> None:
        raise OSError("interrupted before publication")

    monkeypatch.setattr(pipeline_mod.os, "fsync", interrupted)
    pipeline_mod._write_refusal_marker(ctx, ctx.job_id, "extract_service_failure")
    assert not marker.exists()
    monkeypatch.setattr(pipeline_mod.os, "fsync", original_fsync)
    pipeline_mod._write_refusal_marker(ctx, ctx.job_id, "extract_service_failure")
    assert marker.read_text(encoding="utf-8") == '{"kind": "extract_service_failure"}'


def test_concurrent_extract_retries_claim_once(
    lane_db: Path, uploads: UploadStoreImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    ctx = admit_job("claim-race", "claim-race", "claim-race")
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())
    calls: list[str] = []
    barrier = Barrier(2)
    original_admit = pipeline_mod.admit_job

    def simultaneous_admit(*args, **kwargs):
        barrier.wait(timeout=5)
        return original_admit(*args, **kwargs)

    class RetrySpark:
        def run_steps(self, file_id, **kwargs):
            calls.append(file_id)
            return SparkRunResult(_chunk_payloads()[0], Decimal("0.01"), 5)

    monkeypatch.setattr(pipeline_mod, "admit_job", simultaneous_admit)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(spark=RetrySpark()),
    )
    retry = pipeline_mod._window_retry_call(
        ctx, ctx, spark, uploads, settings,
        window_index=0, window_s=60.0,
        planned_usd=lambda: Decimal("0.10"), chunk_count=0, replay=False,
    )
    assert retry is not None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(retry, "file", 60.0, None, False) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except Exception as error:
                outcomes.append(error)
    assert sum(isinstance(item, SparkRunResult) for item in outcomes) == 1
    assert sum(isinstance(item, pipeline_mod.SparkServiceError) for item in outcomes) == 1
    assert calls == ["file"]
    with db.connect(lane_db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE anon_id = ?",
            (f"{ctx.job_id}/extract/retry/0",),
        ).fetchone()[0] == 1


def test_distinct_extract_retry_windows_share_atomic_budget(
    lane_db: Path, uploads: UploadStoreImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    ctx = admit_job("budget-race", "budget-race", "budget-race")
    spark = _FakeChunkSpark(_chunk_payloads())
    share = spark.spark_stage_config.run_shares[0]
    settings = Settings(**{
        **live_settings.__dict__, "extract_retry_backoff_s": 0.0,
        "chunk_retry_budget_usd": share,
    })
    barrier = Barrier(2)
    original_admit = pipeline_mod.admit_job

    def simultaneous_admit(*args, **kwargs):
        barrier.wait(timeout=5)
        return original_admit(*args, **kwargs)

    calls: list[str] = []

    class RetrySpark:
        def run_steps(self, file_id, **kwargs):
            calls.append(file_id)
            return SparkRunResult(_chunk_payloads()[0], share, 5)

    monkeypatch.setattr(pipeline_mod, "admit_job", simultaneous_admit)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(spark=RetrySpark()),
    )
    retries = [pipeline_mod._window_retry_call(
        ctx, ctx, spark, uploads, settings,
        window_index=index, window_s=60.0,
        planned_usd=lambda: Decimal("0.10"), chunk_count=0, replay=False,
    ) for index in (0, 1)]
    assert all(retries)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(retry, "file", 60.0, None, False) for retry in retries]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except Exception as error:
                outcomes.append(error)
    assert sum(isinstance(item, SparkRunResult) for item in outcomes) == 1
    assert sum(isinstance(item, pipeline_mod.SparkServiceError) for item in outcomes) == 1
    assert calls == ["file"]
    with db.connect(lane_db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE anon_id LIKE ?",
            (f"{ctx.job_id}/extract/retry/%",),
        ).fetchone()[0] == 1


def test_schema_invalid_extract_retry_child_fails_with_paid_cost(
    lane_db: Path, uploads: UploadStoreImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    ctx = admit_job("schema-retry", "schema-retry", "schema-retry")
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(spark=SimpleNamespace(
            run_steps=lambda *args, **kwargs: SparkRunResult({}, Decimal("0.01"), 5)
        )),
    )
    retry = pipeline_mod._window_retry_call(
        ctx, ctx, spark, uploads, settings,
        window_index=0, window_s=60.0,
        planned_usd=lambda: Decimal("0.10"), chunk_count=0, replay=False,
    )
    assert retry is not None
    with pytest.raises(pipeline_mod.SparkClientError, match="expected keys"):
        retry("file", 60.0, None, False)
    with db.connect(lane_db) as conn:
        row = conn.execute(
            "SELECT id, state FROM jobs WHERE anon_id = ?",
            (f"{ctx.job_id}/extract/retry/0",),
        ).fetchone()
        assert row is not None
        hold = conn.execute(
            "SELECT state, actual_usd FROM reservations"
            " WHERE job_id = ? AND call_kind = 'spark_steps'",
            (row["id"],),
        ).fetchone()
    assert row["state"] == "failed"
    assert hold is not None and tuple(hold) == ("failed", "0.01")
    assert job_reservation_spend_usd(row["id"], path=lane_db) == (
        Decimal("0.01"), Decimal("0")
    )


def test_extract_retry_interrupt_settles_conservative_cost(
    lane_db: Path, uploads: UploadStoreImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    ctx = admit_job("interrupt-retry", "interrupt-retry", "interrupt-retry")
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())

    class Interrupted:
        def run_steps(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(spark=Interrupted()),
    )
    retry = pipeline_mod._window_retry_call(
        ctx, ctx, spark, uploads, settings,
        window_index=0, window_s=60.0,
        planned_usd=lambda: Decimal("0.10"), chunk_count=0, replay=False,
    )
    assert retry is not None
    with pytest.raises(KeyboardInterrupt):
        retry("file", 60.0, None, False)
    with db.connect(lane_db) as conn:
        row = conn.execute("SELECT id, state FROM jobs WHERE anon_id = ?",
                           (f"{ctx.job_id}/extract/retry/0",)).fetchone()
    assert row is not None and row["state"] == "failed"
    assert job_reservation_spend_usd(row["id"], path=lane_db) == (
        spark.spark_stage_config.run_shares[0], Decimal("0")
    )


def test_extension_lands_before_live_clients_build(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    live_settings: Settings,
) -> None:
    """The factory observes the pushed deadline: client budgets stamp it too."""
    from app.service import PipelineUnavailable

    ctx = _confirm_65s_job(uploads, tmp_path)
    _force_deadline_in(lane_db, str(ctx.job_id), 30.0)
    before = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    seen: dict[str, str] = {}

    def _factory(settings: Settings) -> PipelineClients:
        seen["deadline"] = _deadline_of(lane_db, str(ctx.job_id))
        raise PipelineUnavailable("stop before any provider call")

    service.set_client_factory(_factory)
    try:
        with pytest.raises(PipelineUnavailable):
            service.run_upload_job(
                ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle
            )
    finally:
        service.clear_client_factory()
    pushed = db.parse_dt(seen["deadline"])
    assert pushed - before == timedelta(seconds=2 * live_settings.chunk_deadline_allowance_s)


def test_extension_retries_contention_before_binding_clients(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _confirm_65s_job(uploads, tmp_path)
    real_extend = service.extend_job_deadline
    attempts = 0
    built = []

    def contended(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ContentionError("test race")
        return real_extend(*args, **kwargs)

    monkeypatch.setattr(service, "extend_job_deadline", contended)
    def factory(_settings):
        built.append(_deadline_of(lane_db, str(ctx.job_id)))
        raise service.PipelineUnavailable("test stop")

    service.set_client_factory(factory)
    try:
        with pytest.raises(service.PipelineUnavailable, match="test stop"):
            service.run_upload_job(ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle)
    finally:
        service.clear_client_factory()
    assert attempts == 2
    assert built and db.parse_dt(built[0]) > db.utcnow() + timedelta(seconds=JOB_DEADLINE_S)


@pytest.mark.parametrize("outcome", ("contention", "refused", "unpersisted"))
def test_extension_failure_fails_with_reason_before_clients(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{**live_settings.__dict__, "deadline_extension_max_attempts": 2})
    attempts = []
    built = []

    def contended(*args, **kwargs):
        attempts.append(1)
        if outcome == "contention":
            raise ContentionError("test race")
        return outcome == "unpersisted"

    monkeypatch.setattr(service, "extend_job_deadline", contended)
    service.set_client_factory(lambda settings: built.append(1) or service.default_replay_clients())
    try:
        result = service.run_confirmed_upload(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle, meta_client=meta,
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed"
    assert len(attempts) == (2 if outcome == "contention" else 1)
    assert built == []
    assert "service was unavailable" in refusal_text_for_job(str(ctx.job_id), data_root=settings.data_root)


def test_waiting_confirmed_job_starts_only_after_promotion(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = admit_job("anon-first", "ip-first", "video-first")
    ctx = _confirm_65s_job(uploads, tmp_path)
    second = admit_job("anon-second", "ip-second", "video-second")
    with db.connect(lane_db) as conn:
        conn.execute("UPDATE jobs SET state = 'waiting' WHERE id = ?", (str(ctx.job_id),))
        conn.execute("UPDATE jobs SET state = 'active' WHERE id = ?", (str(second.job_id),))
        assert conn.execute("SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)).fetchone()[0] == "waiting"
    settings = Settings(**{**live_settings.__dict__, "job_status_refresh_s": 1})
    built = Event()
    finished = Event()
    waiting_seen = Event()

    class ObservableWait:
        def wait(self, timeout):
            waiting_seen.set()
            return Event().wait(timeout)

    monkeypatch.setattr(service, "Event", ObservableWait)

    def factory(_settings):
        built.set()
        raise service.PipelineUnavailable("test stop")

    def worker():
        try:
            service.run_upload_job(ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle)
        except service.PipelineUnavailable:
            pass
        finally:
            finished.set()

    service.set_client_factory(factory)
    thread = Thread(target=worker)
    try:
        thread.start()
        assert waiting_seen.wait(5)
        assert not built.is_set()
        first.finish("done")
        assert finished.wait(5)
    finally:
        service.clear_client_factory()
        thread.join(timeout=5)
    assert built.is_set()
    with db.connect(lane_db) as conn:
        states = [conn.execute("SELECT state FROM jobs WHERE id = ?", (str(job.job_id),)).fetchone()[0]
                  for job in (second, ctx)]
    assert states == ["active", "active"]


@pytest.mark.parametrize("upload_state", ("cancelled", "failed"))
def test_terminal_waiting_job_keeps_original_refusal(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch, upload_state: str,
) -> None:
    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    with db.connect(lane_db) as conn:
        conn.execute("UPDATE jobs SET state = 'waiting' WHERE id = ?", (str(ctx.job_id),))
    original_kind = "cancelled" if upload_state == "cancelled" else "processing"
    marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"
    built = []
    polled = []

    class TerminalDuringWait:
        def wait(self, timeout):
            polled.append(timeout)
            with db.connect(lane_db) as conn:
                conn.execute("UPDATE jobs SET state = 'failed' WHERE id = ?", (str(ctx.job_id),))
                conn.execute("UPDATE uploads SET state = ? WHERE job_id = ?", (upload_state, str(ctx.job_id)))
            marker.write_text(f'{{"kind":"{original_kind}"}}', encoding="utf-8")
            return True

    monkeypatch.setattr(service, "Event", TerminalDuringWait)
    service.set_client_factory(lambda settings: built.append(1) or service.default_replay_clients())
    try:
        result = service.run_confirmed_upload(
            ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle, meta_client=meta,
        )
    finally:
        service.clear_client_factory()
    with db.connect(lane_db) as conn:
        job_state = conn.execute("SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)).fetchone()[0]
        final_upload_state = conn.execute("SELECT state FROM uploads WHERE job_id = ?", (str(ctx.job_id),)).fetchone()[0]
    assert polled and built == []
    assert result.state == "failed"
    assert result.error is not None
    assert result.error == refusal_text_for_job(str(ctx.job_id), data_root=live_settings.data_root)
    assert (job_state, final_upload_state) == ("failed", upload_state)
    assert marker.read_text(encoding="utf-8") == f'{{"kind":"{original_kind}"}}'
    assert "service_unavailable" not in marker.read_text(encoding="utf-8")


def test_waiting_confirmed_job_expires_with_reason_before_clients(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
) -> None:
    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    with db.connect(lane_db) as conn:
        conn.execute(
            "UPDATE jobs SET state = 'waiting', deadline_at = ? WHERE id = ?",
            (db.format_dt(db.utcnow() - timedelta(seconds=1)), str(ctx.job_id)),
        )
    built = []
    service.set_client_factory(lambda settings: built.append(1) or service.default_replay_clients())
    try:
        result = service.run_confirmed_upload(
            ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle,
            meta_client=meta,
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed" and built == []
    assert "took longer than the time limit" in refusal_text_for_job(
        str(ctx.job_id), data_root=live_settings.data_root,
    )


@pytest.mark.parametrize("delayed_preview", [False, True])
@pytest.mark.parametrize("persist_extension", [False, True])
def test_parent_extension_is_one_shot(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    delayed_preview: bool,
    persist_extension: bool,
) -> None:
    """Delayed consent still needs one persisted bounded extension, including on replay."""
    now = db.utcnow()
    monkeypatch.setattr(db, "utcnow", lambda: now)
    monkeypatch.setattr(uploads, "_clock", lambda: now)
    real_claim = uploads.claim_confirmation

    def delayed_claim(job_id, consented_at):
        nonlocal now
        if delayed_preview:
            now += timedelta(seconds=2 * live_settings.chunk_deadline_allowance_s + JOB_DEADLINE_S)
        return real_claim(job_id, consented_at)

    monkeypatch.setattr(uploads, "claim_confirmation", delayed_claim)
    ctx = _confirm_65s_job(uploads, tmp_path)
    before = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    if not persist_extension:
        monkeypatch.setattr(service, "extend_job_deadline", lambda *a, **kw: False)
        assert not service._extend_chunked_parent_deadline(live_settings, ctx, 65.0)
        assert _deadline_of(lane_db, str(ctx.job_id)) == db.format_dt(before)
        return
    assert service._extend_chunked_parent_deadline(live_settings, ctx, 65.0)
    once = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    assert once - before == timedelta(seconds=2 * live_settings.chunk_deadline_allowance_s)
    assert service._extend_chunked_parent_deadline(live_settings, ctx, 65.0)
    assert _deadline_of(lane_db, str(ctx.job_id)) == db.format_dt(once)


def test_whole_video_chunk_path_gets_scaled_parent_deadline(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _confirm_65s_job(uploads, tmp_path)
    # The 214 s production route plans one window, but still runs the
    # chunk extractor. A fixed 240 s parent expires before SAM starts.
    settings = Settings(**{**live_settings.__dict__, "whole_video_max_s": 240.0})
    before = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    service._extend_chunked_parent_deadline(settings, ctx, 214.0)
    after = db.parse_dt(_deadline_of(lane_db, str(ctx.job_id)))
    assert after - before == timedelta(seconds=settings.chunk_deadline_allowance_s)
    assert JOB_DEADLINE_S + settings.chunk_deadline_allowance_s > 214.0
    # At a wall time past the original deadline, the extended parent is
    # still authorized to begin its SAM work.
    monkeypatch.setattr(db, "utcnow", lambda: before + timedelta(seconds=1))
    assert ctx.remaining_s() > 0


def test_five_minute_clip_deadline_covers_every_window(live_settings: Settings) -> None:
    from app.pipeline import plan_extraction_windows

    settings = Settings(**{**live_settings.__dict__, "whole_video_max_s": 240.0})
    windows = plan_extraction_windows(settings.max_video_s, settings)
    needed = len(windows) * settings.chunk_deadline_allowance_s
    assert len(windows) == 6
    assert settings.deadline_extension_max_s >= needed
    assert JOB_DEADLINE_S + needed > settings.max_video_s


def test_loaded_extension_ceiling_tracks_chunk_configuration() -> None:
    configured = load_settings({
        "PARTFIDELITY_MAX_VIDEO_S": "120",
        "PARTFIDELITY_CHUNK_S": "30",
        "PARTFIDELITY_CHUNK_DEADLINE_ALLOWANCE_S": "900",
    })
    assert configured.deadline_extension_max_s == 4500
    with pytest.raises(ValueError, match="PARTFIDELITY_DEADLINE_EXTENSION_MAX_S"):
        load_settings({
            "PARTFIDELITY_MAX_VIDEO_S": "120",
            "PARTFIDELITY_CHUNK_S": "30",
            "PARTFIDELITY_CHUNK_DEADLINE_ALLOWANCE_S": "900",
            "PARTFIDELITY_DEADLINE_EXTENSION_MAX_S": "3600",
        })


def test_extension_skips_unconfirmed_upload(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    live_settings: Settings,
) -> None:
    """A preview with no confirmation and no Meta file is never extended."""
    clip = service.write_sample_clip(tmp_path / "unconfirmed.mp4", seconds=65.0, fps=5)
    ctx = admit_job("anon-unconf", "ip-unconf", hash_file(clip))
    with clip.open("rb") as handle:
        preview = prepare_preview(ctx, handle, clip.name, uploads=uploads)
    assert preview.duration_s > 60.0
    before = _deadline_of(lane_db, str(ctx.job_id))
    service._extend_chunked_parent_deadline(
        live_settings, ctx, preview.duration_s
    )
    assert _deadline_of(lane_db, str(ctx.job_id)) == before


def test_extension_skips_hidden_upload(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    live_settings: Settings,
) -> None:
    """A confirmed upload that has since been hidden is never extended."""
    ctx = _confirm_65s_job(uploads, tmp_path)
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "UPDATE uploads SET hidden = 1 WHERE job_id = ?",
            (str(ctx.job_id),),
        )
        conn.commit()
    finally:
        conn.close()
    before = _deadline_of(lane_db, str(ctx.job_id))
    service._extend_chunked_parent_deadline(live_settings, ctx, 65.0)
    assert _deadline_of(lane_db, str(ctx.job_id)) == before


def test_extension_contention_warns_and_leaves_deadline(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    live_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lost write race warns instead of claiming or moving the deadline."""
    ctx = _confirm_65s_job(uploads, tmp_path)
    before = _deadline_of(lane_db, str(ctx.job_id))
    holder = db.connect(lane_db)
    try:
        holder.execute("BEGIN IMMEDIATE")
        with caplog.at_level("WARNING", logger="partfidelity.service"):
            service._extend_chunked_parent_deadline(live_settings, ctx, 65.0)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert any("lost a race" in record.getMessage() for record in caplog.records)
    assert _deadline_of(lane_db, str(ctx.job_id)) == before


def test_child_deadline_covers_worst_case_spark_runs(
    tmp_path: Path,
    lane_db: Path,
    uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each child envelope covers the configured runs x timeout, exactly."""
    import app.pipeline as pipeline_mod

    providers = ProviderSettings(mode="live", spark_timeout_s=300.0)
    chunk_settings = Settings(
        data_root=tmp_path / "data",
        db_path=lane_db,
        production=False,
        meta_api_key=FAKE_KEY,
        pipeline_mode="live",
        providers=providers,
        step_span_mode="legacy",  # sparse payloads predate contiguous phases
        whole_video_max_s=60.0,  # the 65 s job pins two-chunk planning
    )
    expected_extra = providers.spark_runs * providers.spark_timeout_s
    assert expected_extra > JOB_DEADLINE_S

    shared_meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, shared_meta)
    chunk_spark = _FakeChunkSpark(_chunk_payloads())
    monkeypatch.setattr(
        pipeline_mod,
        "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(spark=chunk_spark),
    )
    service.set_client_factory(
        lambda settings: PipelineClients(
            spark=_FakeParentSpark(),  # type: ignore[arg-type]
            sam=_FakeLiveSam(),  # type: ignore[arg-type]
            meta=shared_meta,  # type: ignore[arg-type]
        )
    )
    try:
        result = service.run_upload_job(
            ctx.job_id, chunk_settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
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
        assert envelope == timedelta(
            seconds=JOB_DEADLINE_S + expected_extra
        )


@pytest.mark.parametrize("existing_reason", (False, True))
def test_real_waiting_cancellation_shows_reason(
    tmp_path, lane_db, uploads, lifecycle, live_settings, monkeypatch, existing_reason,
):
    from fastapi.testclient import TestClient
    from app.deletion import cancel_and_delete
    from app.main import create_app

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    with db.connect(lane_db) as conn:
        conn.execute("UPDATE jobs SET state = 'waiting' WHERE id = ?", (str(ctx.job_id),))

    class CancelDuringWait:
        def wait(self, timeout):
            result = cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=meta)
            assert result.recorded_state == "cancelled"
            if existing_reason:
                marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"
                marker.write_text('{"kind":"processing"}', encoding="utf-8")
            return True

    monkeypatch.setattr(service, "Event", CancelDuringWait)
    result = service.run_upload_job(ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle)
    page = TestClient(create_app(live_settings)).get(
        f"/jobs/{ctx.job_id}", headers={"accept": "text/html"}
    )
    assert result.error and ("process the video" if existing_reason else "cancelled") in result.error
    assert result.error in page.text


@pytest.mark.parametrize("existing_reason", (False, True))
def test_queued_cancellation_shows_reason_before_worker(
    tmp_path, lane_db, uploads, lifecycle, live_settings, existing_reason,
):
    import json
    from fastapi.testclient import TestClient
    from app.deletion import cancel_and_delete
    from app.main import create_app

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    with db.connect(lane_db) as conn:
        conn.execute("UPDATE jobs SET state = 'waiting' WHERE id = ?", (str(ctx.job_id),))
    marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"
    if existing_reason:
        marker.write_text('{"kind":"processing"}', encoding="utf-8")
    cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=meta)
    page = TestClient(create_app(live_settings)).get(f"/jobs/{ctx.job_id}")
    assert json.loads(marker.read_text())["kind"] == ("processing" if existing_reason else "cancelled")
    assert ("process the video" if existing_reason else "cancelled") in page.json()["reason"]


def test_cancelled_status_uses_durable_flag_when_marker_unwritten(
    tmp_path, uploads, lifecycle, live_settings, monkeypatch,
):
    from fastapi.testclient import TestClient
    from app.deletion import cancel_and_delete
    from app.main import create_app

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    monkeypatch.setattr(uploads, "_write_cancel_reason", lambda job_id: None)
    cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=meta)
    page = TestClient(create_app(live_settings)).get(f"/jobs/{ctx.job_id}")
    assert "cancelled" in page.json()["reason"]


@pytest.mark.parametrize("outcome", ("retry", "contention", "refused", "unpersisted"))
def test_child_extension_verified_before_clients(
    tmp_path, lane_db, uploads, lifecycle, live_settings, monkeypatch, outcome,
):
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{**live_settings.__dict__, "deadline_extension_max_attempts": 2})
    attempts = []
    built = []
    real_extend = pipeline_mod.extend_job_deadline

    def extension(*args, **kwargs):
        attempts.append(args[0])
        if outcome == "contention" or (outcome == "retry" and len(attempts) == 1):
            raise ContentionError("test race")
        if outcome == "refused":
            return False
        if outcome == "unpersisted":
            return True
        return real_extend(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "extend_job_deadline", extension)
    def select(*args, **kwargs):
        built.append(1)
        raise RuntimeError("test stop")
    monkeypatch.setattr(pipeline_mod, "select_providers", select)
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle)
    finally:
        service.clear_client_factory()
    assert result.state == "failed"
    assert len(attempts) == (2 if outcome in ("retry", "contention") else 1)
    if outcome == "retry":
        assert built == [1]
    else:
        assert built == []
        assert "service was unavailable" in refusal_text_for_job(str(attempts[0]), data_root=settings.data_root)
        with db.connect(lane_db) as conn:
            assert conn.execute("SELECT state FROM jobs WHERE id = ?", (str(attempts[0]),)).fetchone()[0] == "failed"


@pytest.mark.parametrize("failure", (RuntimeError, KeyboardInterrupt))
@pytest.mark.parametrize("stage", ("extension", "verification"))
def test_child_verification_exception_settles_before_clients(
    tmp_path, lane_db, uploads, lifecycle, live_settings, monkeypatch, failure, stage,
):
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    children = []
    built = []
    real_admit = pipeline_mod.admit_job
    def admit(*args, **kwargs):
        child = real_admit(*args, **kwargs)
        children.append(child.job_id)
        return child
    real_parse = db.parse_dt
    def fail(*args, **kwargs):
        monkeypatch.setattr(db, "parse_dt", real_parse)
        raise failure("verification failed")
    real_extend = pipeline_mod.extend_job_deadline
    def extension(*args, **kwargs):
        if stage == "extension":
            fail()
        result = real_extend(*args, **kwargs)
        monkeypatch.setattr(db, "parse_dt", fail)
        return result
    monkeypatch.setattr(pipeline_mod, "admit_job", admit)
    monkeypatch.setattr(pipeline_mod, "extend_job_deadline", extension)
    monkeypatch.setattr(pipeline_mod, "select_providers", lambda *a, **kw: built.append(1))
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        if failure is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                service.run_upload_job(ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle)
        else:
            assert service.run_upload_job(ctx.job_id, live_settings, uploads=uploads, lifecycle=lifecycle).state == "failed"
    finally:
        service.clear_client_factory()
    assert len(children) == 1 and not built
    with db.connect(lane_db) as conn:
        assert conn.execute("SELECT state FROM jobs WHERE id = ?", (str(children[0]),)).fetchone()[0] == "failed"
    assert "service was unavailable" in refusal_text_for_job(str(children[0]), data_root=live_settings.data_root)


def test_refusal_marker_is_written_where_hard_links_are_refused(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Modal volume answers os.link with PermissionError; the marker (and so the
    # reason a failed job shows) was silently dropped there.
    import app.pipeline as pipeline_mod

    ctx = admit_job("no-hardlink", "no-hardlink", "no-hardlink")
    marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"

    def refused(*_args: object) -> None:
        raise PermissionError("hard links not permitted")

    monkeypatch.setattr(pipeline_mod.os, "link", refused)
    pipeline_mod._write_refusal_marker(ctx, ctx.job_id, "extract_service_failure")
    assert marker.read_text(encoding="utf-8") == '{"kind": "extract_service_failure"}'
    pipeline_mod._write_refusal_marker(ctx, ctx.job_id, "cancelled")
    assert marker.read_text(encoding="utf-8") == '{"kind": "extract_service_failure"}'


def test_billed_empty_200_retries_once_then_ends_with_service_failure(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(**{**live_settings.__dict__, "extract_retry_backoff_s": 0.0})
    spark = _FakeChunkSpark(_chunk_payloads())
    calls: list[str] = []

    def empty(*args, **kwargs):
        calls.append(args[0])
        return SparkRunResult({}, Decimal("0.03"), 50, "empty")

    monkeypatch.setattr(spark, "run_steps", empty)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(
            spark=RecordingSparkClient(spark, sub_ctx)
        ),
    )
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed" and len(calls) == 2
    with db.connect(lane_db) as conn:
        retry = conn.execute(
            "SELECT id FROM jobs WHERE anon_id = ?",
            (f"{ctx.job_id}/extract/retry/0",),
        ).fetchone()
    assert retry is not None
    assert job_reservation_spend_usd(retry["id"], path=lane_db)[0] == Decimal("0.03")
    assert refusal_text_for_job(
        str(ctx.job_id), data_root=settings.data_root
    ) == refusal_copy("extract_service_failure")


@pytest.mark.parametrize(
    "override",
    [{"extract_retry_min_remaining_s": 1e9}, {"chunk_retry_budget_usd": Decimal("0")}],
    ids=["deadline", "retry-budget"],
)
def test_billed_empty_200_without_retry_room_ends_with_no_second_call(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch, override: dict,
) -> None:
    import app.pipeline as pipeline_mod

    meta = _FakeLiveMeta()
    ctx = _confirm_65s_job(uploads, tmp_path, meta)
    settings = Settings(
        **{**live_settings.__dict__, "extract_retry_backoff_s": 0.0, **override}
    )
    spark = _FakeChunkSpark(_chunk_payloads())
    calls: list[str] = []

    def empty(*args, **kwargs):
        calls.append(args[0])
        return SparkRunResult({}, Decimal("0.03"), 50, "empty")

    monkeypatch.setattr(spark, "run_steps", empty)
    monkeypatch.setattr(
        pipeline_mod, "select_providers",
        lambda settings, sub_ctx, **kwargs: SimpleNamespace(
            spark=RecordingSparkClient(spark, sub_ctx)
        ),
    )
    service.set_client_factory(lambda settings: PipelineClients(
        spark=_FakeParentSpark(), sam=_FakeLiveSam(), meta=meta,
    ))
    try:
        result = service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed" and len(calls) == 1
    with db.connect(lane_db) as conn:
        retries = conn.execute(
            "SELECT id FROM jobs WHERE anon_id LIKE ?",
            (f"{ctx.job_id}/extract/retry/%",),
        ).fetchall()
    assert retries == []
    assert refusal_text_for_job(
        str(ctx.job_id), data_root=settings.data_root
    ) == refusal_copy("extract_service_failure")


def test_refusal_marker_race_keeps_the_first_reason(
    lane_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.pipeline as pipeline_mod

    ctx = admit_job("marker-race", "marker-race", "marker-race")
    marker = tmp_path / "data" / "jobs" / str(ctx.job_id) / "refusal.json"
    real_open = pipeline_mod.os.open

    def second_writer_wins_the_gap(path, flags, *args):
        # The other writer lands between this writer's start and its create.
        if Path(path) == marker and not marker.exists():
            marker.write_text('{"kind": "cancelled"}', encoding="utf-8")
        return real_open(path, flags, *args)

    monkeypatch.setattr(pipeline_mod.os, "open", second_writer_wins_the_gap)
    pipeline_mod._write_refusal_marker(ctx, ctx.job_id, "extract_service_failure")
    assert marker.read_text(encoding="utf-8") == '{"kind": "cancelled"}'
