"""End-to-end pipeline tests: sample clip to guide in replay mode.

Builds a synthetic mp4 clip, runs it through app.pipeline.run_job with
recorded provider responses, and asserts the guide carries the fan and
battery steps with evidence links. Reservation and provider-error paths
must fail closed: holds settled, job terminal, nothing served as a guide.
"""

from __future__ import annotations

import json
import errno
import logging
import socket
import sqlite3
import threading
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db
from app import pipeline
from app.admission import admit_job
from app.contracts import GuideHazardAssessment, JobId
from app.hazards import assess_hazards
from app.ledger import LifecycleRepositoryImpl, list_call_records
from app.main import create_app
from app.meta_files import MetaFilesError
from app.pipeline import (
    SAM_SEGMENT_STAGE,
    PipelineClients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    deletion_schedule,
    guide_hazard_assessment,
    guide_path,
    load_guide,
    run_job,
    save_guide,
)
from app.readback import ReadbackResult
from app.settings import Settings
from app.sam import SamDeadlineExceeded
from app.spark import SparkRunResult, SparkStageConfig
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


def make_clip(path: Path, *, seconds: float = 6.0, fps: int = 10) -> Path:
    """Write a synthetic 320x240 mp4 with moving texture (sharpness varies)."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    assert writer.isOpened()
    count = int(seconds * fps)
    for index in range(count):
        frame = np.zeros((240, 320, 3), np.uint8)
        x = int(10 + (300 - 60) * index / max(count - 1, 1))
        cv2.rectangle(frame, (x, 60), (x + 60, 180), (240, 240, 240), -1)
        cv2.putText(
            frame,
            f"frame {index}",
            (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
        writer.write(frame)
    writer.release()
    return path


def _step(number, start, end, action, parts, tools=(), hazards=(), kind="repair"):
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


def sample_runs() -> list[SparkRunResult]:
    """Three recorded runs: fan step plus battery step on a laptop."""
    windows = [(0.5, 2.5), (0.6, 2.6), (0.4, 2.4)]
    # Contiguous partitions (time consensus excludes gapped runs).
    battery_windows = [(2.5, 5.5), (2.6, 5.6), (2.4, 5.4)]
    costs = ["0.08", "0.08", "0.07"]
    runs = []
    for index in range(3):
        start, end = windows[index]
        bstart, bend = battery_windows[index]
        runs.append(
            SparkRunResult(
                payload={
                    "device_guess": "laptop",
                    "confidence": 0.82,
                    "steps": [
                        _step(
                            1,
                            start,
                            end,
                            "Remove the rear screws",
                            [("fan", "left side")],
                            tools=("Phillips #0",),
                        ),
                        _step(
                            2,
                            bstart,
                            bend,
                            "Disconnect the battery cable",
                            [("battery", "lower right")],
                            hazards=("battery",),
                        ),
                    ],
                },
                cost_usd=Decimal(costs[index]),
                latency_ms=900,
            )
        )
    return runs


def sample_clients() -> PipelineClients:
    spark = ReplaySparkClient(
        sample_runs(),
        readbacks={"battery": ReadbackResult(answer="yes", cost_usd=Decimal("0.04"), latency_ms=120)},
    )
    sam = ReplaySamClient(
        {
            "fan": (FAN_BOX, [0.72, 0.60]),
            "battery": (BATTERY_BOX, [0.61, 0.55]),
        }
    )
    return PipelineClients(spark=spark, sam=sam, meta=ReplayMetaFilesClient())


def _job_state(database: Path, job_id: str) -> str | None:
    conn = db.connect(database)
    try:
        row = conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return None if row is None else str(row["state"])
    finally:
        conn.close()


def _held_reservations(database: Path, job_id: str) -> int:
    conn = db.connect(database)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM reservations WHERE job_id = ? AND state = 'held'",
                (job_id,),
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_part_check_logs_safe_filesystem_details(
    tmp_path, lane_db, uploads, lifecycle, monkeypatch, caplog,
):
    failed_path = tmp_path / "data" / "jobs" / "blocked.png"

    def fail_segment(*args, **kwargs):
        raise PermissionError(errno.EACCES, "secret provider headers", str(failed_path))

    monkeypatch.setattr(pipeline, "segment", fail_segment)
    caplog.set_level(logging.INFO, logger="partfidelity.pipeline")
    result = run_job(
        make_clip(tmp_path / "sample.mp4"), clients=sample_clients(),
        uploads=uploads, lifecycle=lifecycle, anon_id="sample-anon",
        ip_hash="sample-ip", replay=True,
    )
    assert result.error == "part check failed"
    assert str(failed_path) in caplog.text
    assert "Permission denied" in caplog.text
    assert "secret provider headers" not in caplog.text


def test_fresh_replay_clients_mint_distinct_file_ids(tmp_path: Path) -> None:
    ids = {
        str(
            sample_clients().meta.upload_file(
                tmp_path / "a.mp4", expires_after_hours=1, idempotency_key="k"
            ).file_id
        )
        for _ in range(2)
    }
    assert len(ids) == 2
    one = sample_clients().meta
    first = one.upload_file(tmp_path / "a.mp4", expires_after_hours=1, idempotency_key="k")
    assert one.upload_file(tmp_path / "a.mp4", expires_after_hours=1, idempotency_key="k") is first


def test_sample_flow_twice_against_one_store_both_succeed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    # Fresh clients per run simulate a restart over the persisted db.
    results = [
        run_job(
            make_clip(tmp_path / f"restart-{n}.mp4"),
            clients=sample_clients(),
            uploads=uploads,
            lifecycle=lifecycle,
            anon_id=f"restart-{n}",
            ip_hash=f"restart-{n}",
            replay=True,
        )
        for n in range(2)
    ]
    assert [r.state for r in results] == ["done", "done"], [r.error for r in results]
    assert results[0].job_id != results[1].job_id


def test_sample_clip_to_guide_in_replay_mode(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, monkeypatch,
) -> None:
    def _no_network(*args, **kwargs):
        raise AssertionError("replay mode must make zero network calls")

    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    with monkeypatch.context() as blocked:
        blocked.setattr(socket, "socket", _no_network)
        result = run_job(
            clip,
            clients=clients,
            uploads=uploads,
            lifecycle=lifecycle,
            anon_id="sample-anon",
            ip_hash="sample-ip",
            replay=True,
        )
    assert result.state == "done", result.error
    assert result.error is None
    assert result.replay is True
    assert result.job_id is not None
    assert result.guide is not None
    assert _job_state(lane_db, str(result.job_id)) == "done"
    assert _held_reservations(lane_db, str(result.job_id)) == 0

    guide = result.guide
    phrases = [part.phrase for step in guide.steps for part in step.parts]
    assert "fan" in phrases and "battery" in phrases
    assert len(guide.steps) == 2
    assert all(decision.checked for decision in guide.labels)
    assert len(guide.labels) == 2
    assert {d.readback for d in guide.labels} == {"not_run", "yes"}
    assert clients.spark.run_calls and len(clients.spark.run_calls) == 3
    assert len(clients.sam.calls) == 4  # two parts x two frames
    assert clients.meta.upload_calls  # one recorded Meta upload

    # The stored guide round-trips and serves with evidence links.
    stored = load_guide(result.job_id)
    assert stored is not None
    assert [s.action for s in stored.steps] == [s.action for s in guide.steps]
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False
    )
    client = TestClient(create_app(settings))
    page = client.get(f"/guides/{result.job_id}")
    assert page.status_code == 200
    assert "fan" in page.text and "battery" in page.text
    assert "No visual changes to this part" in page.text
    assert f"/calls/{result.job_id}" in page.text
    assert "How this guide was made" in page.text
    image = client.get(f"/guides/{result.job_id}/image/0")
    assert image.status_code == 200
    status = client.get(f"/jobs/{result.job_id}").json()
    assert status["state"] == "done"
    assert status["guide_url"] == f"/guides/{result.job_id}"
    assert status["calls_url"] == f"/calls/{result.job_id}"

    schedule = deletion_schedule(result.job_id, lifecycle=lifecycle)
    assert schedule.job_id == result.job_id
    assert isinstance(schedule.artifact_ids, tuple)


def test_refused_reservation_fails_closed_before_any_provider_call(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    for index in range(12):  # 2 running + 10 waiting fills the queue
        admit_job(f"fill-anon-{index}", f"fill-ip-{index}", f"fill-video-{index}")
    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    result = run_job(
        clip,
        clients=clients,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
    )
    assert result.state == "refused"
    assert result.job_id is None
    assert result.guide is None
    assert result.error
    assert clients.spark.run_calls == []
    assert clients.spark.readback_calls == []
    assert clients.sam.calls == []
    assert clients.meta.upload_calls == []


def _failing_spark_client(error: Exception) -> ReplaySparkClient:
    class _Failing(ReplaySparkClient):
        def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):  # type: ignore[override]
            self.run_calls.append(video_file_id)
            raise error

    return _Failing([])


def test_spark_provider_error_fails_closed_and_settles(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    failing = PipelineClients(
        spark=_failing_spark_client(RuntimeError("spark down")),
        sam=clients.sam,
        meta=clients.meta,
    )
    result = run_job(
        clip,
        clients=failing,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
    )
    assert result.state == "failed"
    assert result.guide is None
    assert result.job_id is not None
    assert _job_state(lane_db, str(result.job_id)) == "failed"
    assert _held_reservations(lane_db, str(result.job_id)) == 0
    assert load_guide(result.job_id) is None
    assert clients.sam.calls == []  # stopped before SAM
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False
    )
    status = TestClient(create_app(settings)).get(f"/jobs/{result.job_id}").json()
    assert status["state"] == "failed"
    assert status["guide_url"] is None


def test_spark_schema_violation_fails_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    bad = SparkRunResult(
        payload={"device_guess": "laptop", "confidence": 0.9, "steps": []},
        cost_usd=Decimal("0.08"),
        latency_ms=10,
    )
    failing = PipelineClients(
        spark=ReplaySparkClient([bad, bad, bad]),
        sam=clients.sam,
        meta=clients.meta,
    )
    result = run_job(
        clip,
        clients=failing,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
    )
    # Zero consensus steps is a content refusal, not a crash: no SAM calls.
    assert result.state == "refused"
    assert result.guide is None
    assert clients.sam.calls == []


def test_meta_provider_error_fails_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    class _FailingMeta(ReplayMetaFilesClient):
        def upload_file(self, path, *, expires_after_hours, idempotency_key):
            self.upload_calls.append(
                (str(path), expires_after_hours, idempotency_key)
            )
            raise MetaFilesError("meta down")

    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    failing = PipelineClients(
        spark=clients.spark, sam=clients.sam, meta=_FailingMeta()
    )
    result = run_job(
        clip,
        clients=failing,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
    )
    assert result.state == "failed"
    assert result.error == "confirmation failed"
    assert result.job_id is not None
    assert _job_state(lane_db, str(result.job_id)) == "failed"
    assert _held_reservations(lane_db, str(result.job_id)) == 0
    assert clients.spark.run_calls == []  # stopped before extraction
    assert load_guide(result.job_id) is None


def test_sam_provider_error_fails_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    class _FailingSam(ReplaySamClient):
        def segment_still(self, image_path, phrase, *, include_confidence=True):
            self.calls.append((str(image_path), phrase))
            raise RuntimeError("sam down")

    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    failing = PipelineClients(
        spark=clients.spark, sam=_FailingSam({}), meta=clients.meta
    )
    result = run_job(
        clip,
        clients=failing,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
    )
    assert result.state == "failed"
    assert result.error == "part check failed"
    assert result.job_id is not None
    assert _job_state(lane_db, str(result.job_id)) == "failed"
    assert _held_reservations(lane_db, str(result.job_id)) == 0
    assert len(clients.spark.run_calls) == 3  # extraction had succeeded
    assert clients.spark.readback_calls == []  # stopped before read-backs
    assert load_guide(result.job_id) is None


def test_readback_provider_error_fails_closed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    class _FailingJudge(ReplaySparkClient):
        def read_back(self, image_path, phrase, position_hint):  # type: ignore[override]
            self.readback_calls.append((phrase, position_hint))
            raise RuntimeError("judge down")

    clip = make_clip(tmp_path / "sample.mp4")
    clients = sample_clients()
    failing = PipelineClients(
        spark=_FailingJudge(sample_runs()),
        sam=clients.sam,
        meta=clients.meta,
    )
    result = run_job(
        clip,
        clients=failing,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
    )
    assert result.state == "failed"
    assert result.error == "read-back failed"
    assert result.job_id is not None
    assert _job_state(lane_db, str(result.job_id)) == "failed"
    assert _held_reservations(lane_db, str(result.job_id)) == 0
    assert failing.spark.readback_calls != []
    assert load_guide(result.job_id) is None


def test_hazard_refusal_authorizes_zero_sam_calls(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    clip = make_clip(tmp_path / "sample.mp4")
    run = SparkRunResult(
        payload={
            "device_guess": "smartwatch",
            "confidence": 0.9,
            "steps": [
                _step(1, 0.5, 2.5, "Open the case", [("fan", "left")]),
            ],
        },
        cost_usd=Decimal("0.08"),
        latency_ms=10,
    )
    clients = sample_clients()
    failing_spark = ReplaySparkClient([run, run, run])
    # Honest legacy extraction to match the legacy settings: the
    # contiguous consensus would exclude this sparse fragment run.
    failing_spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    failing = PipelineClients(
        spark=failing_spark,
        sam=clients.sam,
        meta=clients.meta,
    )
    result = run_job(
        clip,
        clients=failing,
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
        # Legacy spans: this test pins refusal behavior, which the
        # contiguous gate would shadow with an extraction failure.
        settings=Settings(
            data_root=tmp_path / "data", db_path=lane_db, production=False,
            step_span_mode="legacy",
        ),
    )
    assert result.state == "refused"
    assert result.guide is None
    assert clients.sam.calls == []


def test_guide_store_round_trip_and_fail_closed_reads(
    tmp_path: Path, lane_db: Path,
) -> None:
    from app.contracts import (
        ArtifactId,
        Detection,
        Guide,
        LabelDecision,
        LabelSlot,
        PartRequest,
        ProvisionalRender,
        Step,
    )

    step = Step(
        number=1,
        start_s=0.5,
        end_s=2.5,
        action="Remove the rear screws",
        tools=("Phillips #0",),
        parts=(PartRequest(phrase="fan", position_hint="left side"),),
        hazards=(),
        time_spread_s=0.2,
        time_needs_check=False,
    )
    root = tmp_path / "data"
    detection = Detection(
        phrase="fan",
        mask_path=root / "jobs" / "job-1" / "mask.png",
        score=0.72,
        box_xyxy=(12, 30, 140, 150),
    )
    decision = LabelDecision(
        slot=LabelSlot(
            step_index=0, part_index=0, part=step.parts[0]
        ),
        detection=detection,
        provisional=ProvisionalRender(
            detection=detection,
            artifact_id=ArtifactId("a" * 32),
            rendered_image_path=root / "jobs" / "job-1" / "label.png",
            position_hint="left",
        ),
        checked=True,
        status="part_pixels_unchanged",
        readback="not_run",
    )
    guide = Guide(
        job_id=JobId("job-1"),
        steps=(step,),
        frame_paths=(root / "jobs" / "job-1" / "frame.png",),
        labels=(decision,),
        difficulty="Easy",
        verdict="Try it",
        draft_reasons=(),
    )
    save_guide(guide, data_root=root)
    assert load_guide("job-1", data_root=root) == guide
    assert load_guide("missing", data_root=root) is None
    guide_file = root / "jobs" / "job-1" / "guide.json"
    guide_file.write_text("{not json", encoding="utf-8")
    assert load_guide("job-1", data_root=root) is None
    guide_file.write_text(json.dumps({"job_id": "other"}), encoding="utf-8")
    assert load_guide("job-1", data_root=root) is None
    for hostile in ("..", ".", "../x", "a/b", "a\\b", "", "x" * 65):
        assert load_guide(hostile, data_root=root) is None
    with pytest.raises(ValueError):
        guide_path("..", data_root=root)


def test_hazard_mapping_matches_ordered_rule() -> None:
    from app.contracts import PartRequest, Step

    safe = Step(
        number=1,
        start_s=0.5,
        end_s=2.5,
        action="Remove the rear screws",
        tools=(),
        parts=(PartRequest(phrase="fan", position_hint="left"),),
        hazards=(),
        time_spread_s=0.2,
        time_needs_check=False,
    )
    assessment = assess_hazards([safe], "laptop")
    assert assessment.refused is False
    mapped = guide_hazard_assessment([safe], assessment)
    assert (mapped.difficulty, mapped.verdict) == ("Easy", "Try it")
    assert mapped.draft_reasons == ()
    scoped = assess_hazards([safe], "smartwatch")
    assert scoped.refused is True
    assert guide_hazard_assessment([safe], scoped) == GuideHazardAssessment()


def test_overlapping_jobs_record_every_sam_receipt(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, monkeypatch,
) -> None:
    """Two thread-overlapped jobs each persist all four SAM receipts.

    The first job waits at its last SAM call until the second job is
    inside its own SAM stage; the second then parks until the first
    has fully finished — the exact interleave that strands a
    process-global sink's remaining receipts. The gates sit at
    ``segment`` entry: the pacer holds its lock across the enclosed
    client call, so parking inside the client would deadlock the peer.
    """
    import app.pipeline as pipeline_mod

    first_in = threading.Event()
    second_in = threading.Event()
    release = threading.Event()
    first_tid: list[int] = []
    second_tid: list[int] = []
    made: dict[int, int] = {}
    real_segment = pipeline_mod.segment

    def _gated_segment(ctx, image_path, phrase, *, client):
        tid = threading.get_ident()
        made[tid] = made.get(tid, 0) + 1
        if first_tid and tid == first_tid[0]:
            if made[tid] == 1:
                first_in.set()
            if made[tid] >= 4:
                assert second_in.wait(timeout=300), "overlap never happened"
        elif second_tid and tid == second_tid[0]:
            if made[tid] == 1:
                second_in.set()
            if made[tid] == 2:
                assert release.wait(timeout=300), "overlap gate stuck"
        else:  # pragma: no cover - only the two job threads segment
            raise AssertionError(f"segment from unknown thread {tid}")
        return real_segment(ctx, image_path, phrase, client=client)

    monkeypatch.setattr(pipeline_mod, "segment", _gated_segment)
    results: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def _run(tag: str, tids: list[int]) -> None:
        try:
            tids.append(threading.get_ident())
            base = sample_clients()
            results[tag] = run_job(
                make_clip(tmp_path / f"overlap-{tag}.mp4"),
                clients=base,
                uploads=uploads,
                lifecycle=lifecycle,
                anon_id=f"overlap-{tag}",
                ip_hash=f"overlap-{tag}",
                replay=True,
            )
        except BaseException as error:  # re-raised in the main thread
            errors[tag] = error

    first = threading.Thread(
        target=_run, args=("first", first_tid), daemon=True,
    )
    first.start()
    assert first_in.wait(timeout=300), "first job never reached SAM"
    second = threading.Thread(
        target=_run, args=("second", second_tid), daemon=True,
    )
    second.start()
    try:
        first.join(timeout=500)
        assert not first.is_alive(), "first job hung"
    finally:
        release.set()  # never strand the parked job on failure
    second.join(timeout=500)
    assert not second.is_alive(), "second job hung"
    assert not errors, {tag: repr(error) for tag, error in errors.items()}
    for tag in ("first", "second"):
        result = results[tag]
        assert result.state == "done", f"{tag}: {getattr(result, 'error', None)}"
        rows = list_call_records(result.job_id, path=lane_db)
        segment_rows = [
            row for row in rows if row.stage == SAM_SEGMENT_STAGE
        ]
        assert len(segment_rows) == 4, (
            f"job {tag}: {len(segment_rows)} of 4 SAM calls persisted"
        )


def test_scope_refusal_writes_marker_and_status_page_shows_reason(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    clip = make_clip(tmp_path / "sample.mp4")
    run = SparkRunResult(
        payload={
            "device_guess": "smartwatch",
            "confidence": 0.9,
            "steps": [_step(1, 0.5, 2.5, "Open the case", [("fan", "left")])],
        },
        cost_usd=Decimal("0.08"),
        latency_ms=10,
    )
    clients = sample_clients()
    spark = ReplaySparkClient([run, run, run])
    spark.spark_stage_config = SparkStageConfig(span_mode="legacy")
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        step_span_mode="legacy",
    )
    result = run_job(
        clip,
        clients=PipelineClients(spark=spark, sam=clients.sam, meta=clients.meta),
        uploads=uploads,
        lifecycle=lifecycle,
        anon_id="sample-anon",
        ip_hash="sample-ip",
        replay=True,
        settings=settings,
    )
    assert result.state == "refused"
    assert pipeline.refusal_text_for_job(str(result.job_id)) is not None
    page = TestClient(create_app(settings)).get(
        f"/jobs/{result.job_id}", headers={"accept": "text/html"}
    ).text
    assert "phones and laptops only" in page
    assert "could not be built" not in page


def test_sam_deadline_failure_shows_reason(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, monkeypatch,
) -> None:
    def deadline(*args, **kwargs):
        raise SamDeadlineExceeded("job deadline falls before the next SAM slot")

    monkeypatch.setattr(pipeline, "segment", deadline)
    settings = Settings(data_root=tmp_path / "data", db_path=lane_db, production=False)
    result = run_job(
        make_clip(tmp_path / "deadline.mp4"), clients=sample_clients(),
        uploads=uploads, lifecycle=lifecycle, anon_id="deadline-anon",
        ip_hash="deadline-ip", replay=True, settings=settings,
    )
    assert result.state == "failed"
    page = TestClient(create_app(settings)).get(
        f"/jobs/{result.job_id}", headers={"accept": "text/html"}
    ).text
    assert "took longer than the time limit" in page


def test_assembly_failure_queues_generated_images(tmp_path, lane_db, uploads, lifecycle, monkeypatch):
    from datetime import timedelta
    from app.deletion import sweep_expired
    from tests.test_deletion import FakeMetaFiles

    observed = []
    def fail_assembly(ctx, steps, frames, decisions, **kwargs):
        observed.extend(frames)
        observed.extend(d.provisional.detection.mask_path for d in decisions if d.provisional)
        assert observed and all(p.is_file() for p in observed)
        raise RuntimeError("assembly failed")

    monkeypatch.setattr(pipeline, "assemble_guide", fail_assembly)
    result = run_job(make_clip(tmp_path / "assembly.mp4"), clients=sample_clients(),
                     uploads=uploads, lifecycle=lifecycle, anon_id="assembly",
                     ip_hash="assembly", replay=True)
    assert result.error == "guide assembly failed"
    assert not guide_path(result.job_id).exists()
    now = db.utcnow() + timedelta(days=8)
    sweep_expired(now, uploads=uploads, lifecycle=lifecycle, client=FakeMetaFiles(lambda: now))
    assert all(not p.exists() for p in observed)


@pytest.mark.parametrize("cleanup_error", [
    sqlite3.OperationalError("database is locked"),
    OSError("filesystem unavailable"),
])
def test_assembly_failure_survives_cleanup_error(
    tmp_path, lane_db, uploads, lifecycle, monkeypatch, cleanup_error
):
    from datetime import timedelta
    from app import deletion
    from app.deletion import sweep_expired
    from tests.test_deletion import FakeMetaFiles

    observed = []
    def fail_assembly(ctx, steps, frames, decisions, **kwargs):
        observed.extend(frames)
        observed.extend(d.provisional.detection.mask_path for d in decisions if d.provisional)
        raise RuntimeError("assembly failed")

    def broken_queue(*args, **kwargs):
        raise cleanup_error

    monkeypatch.setattr(pipeline, "assemble_guide", fail_assembly)
    # The real cleanup runs; only its own queue transaction fails.
    monkeypatch.setattr(deletion, "queue_guide_images", broken_queue)
    result = run_job(make_clip(tmp_path / "assembly.mp4"), clients=sample_clients(),
                     uploads=uploads, lifecycle=lifecycle, anon_id="cleanup",
                     ip_hash="cleanup", replay=True)
    assert result.state == "failed" and result.error == "guide assembly failed"
    assert observed and all(p.is_file() for p in observed)
    now = db.utcnow() + timedelta(days=8)
    sweep_expired(now, uploads=uploads, lifecycle=lifecycle, client=FakeMetaFiles(lambda: now))
    assert all(not p.exists() for p in observed)


def test_failed_transition_listing_error_still_fails_and_cleanup_queues(
    tmp_path, lane_db, uploads, lifecycle, monkeypatch
):
    from datetime import timedelta
    from app import ledger
    from app.deletion import sweep_expired
    from tests.test_deletion import FakeMetaFiles

    observed = []
    def fail_assembly(ctx, steps, frames, decisions, **kwargs):
        observed.extend(frames)
        raise RuntimeError("assembly failed")

    def broken_listing(*args, **kwargs):
        raise OSError("listing failed")

    monkeypatch.setattr(pipeline, "assemble_guide", fail_assembly)
    monkeypatch.setattr(ledger, "queue_guide_images", broken_listing)
    result = run_job(make_clip(tmp_path / "listing.mp4"), clients=sample_clients(),
                     uploads=uploads, lifecycle=lifecycle, anon_id="listing",
                     ip_hash="listing", replay=True)
    assert result.state == "failed" and result.error == "guide assembly failed"
    now = db.utcnow() + timedelta(days=8)
    sweep_expired(now, uploads=uploads, lifecycle=lifecycle, client=FakeMetaFiles(lambda: now))
    assert observed and all(not p.exists() for p in observed)


@pytest.mark.parametrize("cleanup_error", [
    sqlite3.OperationalError("database is locked"),
    OSError("filesystem unavailable"),
])
def test_abandoned_publication_survives_second_cleanup_error(
    tmp_path, lane_db, uploads, lifecycle, monkeypatch, cleanup_error
):
    from datetime import timedelta
    from app import deletion, guide_clip
    from app.deletion import sweep_expired
    from tests.test_deletion import FakeMetaFiles

    def lose_clip(ctx, store, *, filename=None):
        raise guide_clip.GuideClipError("clip copy failed")

    def broken_queue(*args, **kwargs):
        raise cleanup_error

    with monkeypatch.context() as patch:
        patch.setattr(guide_clip, "retain_guide_clip", lose_clip)
        # Both cleanups (in _fail and after it) hit the same queue failure.
        patch.setattr(deletion, "queue_guide_images", broken_queue)
        result = run_job(make_clip(tmp_path / "abandon.mp4"), clients=sample_clients(),
                         uploads=uploads, lifecycle=lifecycle, anon_id="abandon",
                         ip_hash="abandon", replay=True)
    assert result.state == "failed" and result.error == "could not keep guide clip"
    saved = guide_path(result.job_id)
    assert saved.exists()  # the just-saved guide is still on disk, queued
    now = db.utcnow() + timedelta(days=8)
    for _ in range(2):
        sweep_expired(now, uploads=uploads, lifecycle=lifecycle, client=FakeMetaFiles(lambda: now))
    assert not saved.exists()
    assert not list(saved.parent.rglob("*.png"))


def test_listing_failure_without_cleanup_is_swept_later(
    tmp_path, lane_db, uploads, lifecycle, monkeypatch
):
    from datetime import timedelta
    from app import ledger
    from app.deletion import sweep_expired
    from tests.test_deletion import FakeMetaFiles

    observed = []
    def fail_assembly(ctx, steps, frames, decisions, **kwargs):
        observed.extend(frames)
        observed.extend(d.provisional.detection.mask_path for d in decisions if d.provisional)
        raise RuntimeError("assembly failed")

    def broken_listing(*args, **kwargs):
        raise OSError("listing failed")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "assemble_guide", fail_assembly)
        patch.setattr(ledger, "queue_guide_images", broken_listing)
        # The process exits after finish("failed"): no after-failure cleanup runs.
        patch.setattr(pipeline, "cleanup_failed_guide", lambda *a, **k: None)
        result = run_job(make_clip(tmp_path / "exit.mp4"), clients=sample_clients(),
                         uploads=uploads, lifecycle=lifecycle, anon_id="exit",
                         ip_hash="exit", replay=True)
    assert result.state == "failed"
    assert observed and all(Path(p).is_file() for p in observed)
    now = db.utcnow() + timedelta(days=8)
    sweep_expired(now, uploads=uploads, lifecycle=lifecycle, client=FakeMetaFiles(lambda: now))
    assert all(not Path(p).exists() for p in observed)
