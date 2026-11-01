"""Integration flow tests: mounted app from upload/sample to guide+calls.

Upload confirmation and the sample flow start the orchestrator through
the service; the guide pages render the stored guide the job produced
and the calls page renders its persisted receipts. Replay mode rejects
live clients, terminal contention reports pending settlement (never a
false failed), and a storage failure leaves the job failed with its
outputs removed.
"""

from __future__ import annotations

import json
import multiprocessing
import socket
import subprocess
import threading
from decimal import Decimal
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app import db, service
from app.admission import admit_job
from app.contracts import CallRecord, JobId
from app.evidence import list_calls, record_call
from app.ledger import (
    ContentionError,
    JobContextImpl,
    LifecycleRepositoryImpl,
    list_call_records,
)
from app.main import create_app
from app.pipeline import (
    PipelineClients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    default_replay_clients,
    load_guide,
    pending_settlement_path,
    run_job,
    translate_readback_record,
    translate_sam_record,
)
from app.routes import uploads as uploads_routes
from app.settings import Settings, load_settings
from app.spark import SparkRunResult
from app.uploads_store import UploadStoreImpl


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
def replay_settings(tmp_path: Path, lane_db: Path) -> Settings:
    return Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay",
    )


@pytest.fixture(autouse=True)
def _clean_service():
    service.clear_client_factory()
    yield
    service.clear_client_factory()
    uploads_routes.clear_test_overrides()


def _make_clip(path: Path, *, duration: float = 6.0) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc=size=320x240:rate=10:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-shortest", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def six_second_clip(tmp_path_factory) -> Path:
    return _make_clip(tmp_path_factory.mktemp("media") / "six.mp4")


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


def _job_hold_state(database: Path, job_id: str) -> str | None:
    conn = db.connect(database)
    try:
        row = conn.execute(
            "SELECT state FROM reservations WHERE job_id = ?"
            " AND state IN ('job_hold', 'job_released')",
            (job_id,),
        ).fetchone()
        return None if row is None else str(row["state"])
    finally:
        conn.close()


def _jobs_count(database: Path) -> int:
    conn = db.connect(database)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
    finally:
        conn.close()


class _LiveSpark:
    def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
        raise AssertionError("live client must never run in replay mode")

    def read_back(self, image_path, phrase, position_hint):
        raise AssertionError("live client must never run in replay mode")


class _LiveSam:
    def segment_still(self, image_path, phrase, *, include_confidence=True):
        raise AssertionError("live client must never run in replay mode")


class _LiveMeta:
    def upload_file(self, path, *, expires_after_hours, idempotency_key):
        raise AssertionError("live client must never run in replay mode")

    def find_upload(self, idempotency_key):
        return None

    def delete_file(self, file_id):
        pass

    def get_file(self, file_id):
        return None


def test_replay_rejects_live_clients(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, six_second_clip: Path,
) -> None:
    live = PipelineClients(
        spark=_LiveSpark(), sam=_LiveSam(), meta=_LiveMeta()  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="replay mode rejects"):
        run_job(
            six_second_clip, clients=live, uploads=uploads,
            lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=True,
        )
    canonical = default_replay_clients()
    mixed = PipelineClients(
        spark=canonical.spark, sam=_LiveSam(),  # type: ignore[arg-type]
        meta=canonical.meta,
    )
    with pytest.raises(ValueError, match="replay mode rejects"):
        run_job(
            six_second_clip, clients=mixed, uploads=uploads,
            lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=True,
        )
    assert _jobs_count(lane_db) == 0  # rejected before any side effect


def test_live_requires_injected_non_replay_clients(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, six_second_clip: Path,
) -> None:
    with pytest.raises(ValueError, match="requires an injected client set"):
        run_job(
            six_second_clip, clients=None, uploads=uploads,
            lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=False,
        )
    with pytest.raises(ValueError, match="live mode rejects"):
        run_job(
            six_second_clip, clients=default_replay_clients(),
            uploads=uploads, lifecycle=lifecycle, anon_id="a", ip_hash="i",
            replay=False,
        )
    assert _jobs_count(lane_db) == 0


def test_replay_with_no_clients_selects_canonical_set(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, six_second_clip: Path,
) -> None:
    result = run_job(
        six_second_clip, clients=None, uploads=uploads,
        lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=True,
    )
    assert result.state == "done", result.error
    assert result.job_id is not None
    assert _job_state(lane_db, str(result.job_id)) == "done"
    rows = list_call_records(result.job_id, path=lane_db)
    assert {row.stage for row in rows} == {"upload", "extract", "segment", "readback"}


def test_new_receipt_shapes_validate(lane_db: Path) -> None:
    ctx = admit_job("anon-vocab", "ip-vocab", "video-vocab")
    record_call(
        ctx,
        CallRecord(
            job_id=ctx.job_id, stage="extract", model="muse-spark-1.3",
            endpoint="/runs", latency_ms=900, units_in=None, units_out=None,
            estimated_cost_usd=Decimal("0.08"), status="succeeded",
            evidence={"run_index": 0},
        ),
    )
    record_call(
        ctx,
        CallRecord(
            job_id=ctx.job_id, stage="upload", model="meta-files",
            endpoint="/files", latency_ms=120, units_in=None, units_out=None,
            estimated_cost_usd=Decimal("0"), status="succeeded", evidence={},
        ),
    )
    sam_ok = translate_sam_record(
        CallRecord(
            job_id=ctx.job_id, stage="sam", model="sam-3.1",
            endpoint="segment_stills", latency_ms=400, units_in=None,
            units_out=None, estimated_cost_usd=Decimal("0.0025"), status="ok",
            evidence={"phrase": "fan", "c": 0.72, "request_index": 1},
        )
    )
    assert (sam_ok.stage, sam_ok.status, sam_ok.endpoint) == (
        "segment", "succeeded", "/segment_stills")
    assert sam_ok.evidence == {"phrase": "fan", "score": 0.72}
    record_call(ctx, sam_ok)
    sam_bad = translate_sam_record(
        CallRecord(
            job_id=ctx.job_id, stage="sam", model="sam-3.1",
            endpoint="segment_stills", latency_ms=400, units_in=None,
            units_out=None, estimated_cost_usd=Decimal("0.0025"),
            status="below_threshold",
            evidence={"phrase": "fan", "c": 0.30, "request_index": 2},
        )
    )
    assert sam_bad.status == "failed"
    record_call(ctx, sam_bad)
    rb = translate_readback_record(
        CallRecord(
            job_id=ctx.job_id, stage="readback", model="muse-spark-1.3",
            endpoint="read_back", latency_ms=120, units_in=None, units_out=None,
            estimated_cost_usd=Decimal("0.04"), status="yes",
            evidence={"phrase": "battery", "score": 0.61,
                      "cutoff": 0.66, "readback_fired_n": 1},
        )
    )
    assert (rb.stage, rb.status, rb.endpoint) == (
        "readback", "succeeded", "/read_back")
    assert rb.evidence == {"phrase": "battery", "decision": "yes", "score": 0.61}
    record_call(ctx, rb)
    rows = list_calls(ctx)
    assert [row.stage for row in rows] == [
        "extract", "upload", "segment", "segment", "readback"]
    # The raw module shapes still fail validation: translation is required.
    with pytest.raises(ValueError, match="known stages"):
        record_call(
            ctx,
            CallRecord(
                job_id=ctx.job_id, stage="sam", model="sam-3.1",
                endpoint="segment_stills", latency_ms=1, units_in=None,
                units_out=None, estimated_cost_usd=Decimal("0"),
                status="ok", evidence={},
            ),
        )


def test_terminal_contention_reports_pending_settlement(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, six_second_clip: Path,
    replay_settings: Settings, monkeypatch,
) -> None:
    attempts: list[str] = []

    def _contended(self, outcome, **kwargs):
        attempts.append(outcome)
        raise ContentionError("locked")

    monkeypatch.setattr(JobContextImpl, "finish", _contended)

    class _FailingSpark(ReplaySparkClient):
        def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):  # type: ignore[override]
            self.run_calls.append(video_file_id)
            raise RuntimeError("spark down")

    canonical = default_replay_clients()
    result = run_job(
        six_second_clip,
        clients=PipelineClients(
            spark=_FailingSpark([]), sam=canonical.sam, meta=canonical.meta),
        uploads=uploads, lifecycle=lifecycle, anon_id="a", ip_hash="i",
        replay=True, settings=replay_settings,
    )
    assert result.state == "pending_settlement"
    assert result.job_id is not None
    assert attempts == ["failed"] * 3  # bounded by config
    assert _job_state(lane_db, str(result.job_id)) == "active"
    assert _job_hold_state(lane_db, str(result.job_id)) == "job_hold"
    marker = pending_settlement_path(str(result.job_id))
    assert marker.is_file()
    body = json.loads(marker.read_text(encoding="utf-8"))
    assert body["job_id"] == str(result.job_id)
    assert body["attempts"] == 3
    status = TestClient(create_app(replay_settings)).get(
        f"/jobs/{result.job_id}").json()
    assert status["state"] == "pending_settlement"


def test_transient_contention_retries_to_failed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, six_second_clip: Path,
    replay_settings: Settings, monkeypatch,
) -> None:
    real_finish = JobContextImpl.finish
    calls: list[str] = []

    def _flaky(self, outcome, *args, **kwargs):
        calls.append(outcome)
        if len(calls) <= 2:
            raise ContentionError("locked")
        return real_finish(self, outcome, *args, **kwargs)

    monkeypatch.setattr(JobContextImpl, "finish", _flaky)

    class _FailingSpark(ReplaySparkClient):
        def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):  # type: ignore[override]
            self.run_calls.append(video_file_id)
            raise RuntimeError("spark down")

    canonical = default_replay_clients()
    result = run_job(
        six_second_clip,
        clients=PipelineClients(
            spark=_FailingSpark([]), sam=canonical.sam, meta=canonical.meta),
        uploads=uploads, lifecycle=lifecycle, anon_id="a", ip_hash="i",
        replay=True, settings=replay_settings,
    )
    assert result.state == "failed"
    assert result.job_id is not None
    assert calls == ["failed"] * 3
    assert _job_state(lane_db, str(result.job_id)) == "failed"
    assert _held_reservations(lane_db, str(result.job_id)) == 0
    assert _job_hold_state(lane_db, str(result.job_id)) == "job_released"


def test_guide_storage_failure_leaves_job_failed_and_outputs_removed(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, six_second_clip: Path,
    replay_settings: Settings, monkeypatch,
) -> None:
    import app.pipeline as pipeline_mod

    def _no_store(guide, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline_mod, "save_guide", _no_store)
    result = run_job(
        six_second_clip, clients=None, uploads=uploads,
        lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=True,
        settings=replay_settings,
    )
    assert result.state == "failed"
    assert result.error == "could not store guide"
    assert result.job_id is not None
    job_id = str(result.job_id)
    assert _job_state(lane_db, job_id) == "failed"
    assert _held_reservations(lane_db, job_id) == 0
    guide_file = tmp_path / "data" / "jobs" / job_id / "guide.json"
    assert not guide_file.exists()
    conn = db.connect(lane_db)
    try:
        rows = conn.execute(
            "SELECT kind, delete_state FROM artifacts WHERE job_id = ?",
            (job_id,),
        ).fetchall()
    finally:
        conn.close()
    provisionals = [
        row for row in rows
        if row["kind"] in ("provisional", "provisional_render")]
    assert provisionals  # the rendered outputs exist as rows...
    assert {row["delete_state"] for row in provisionals} == {"pending_delete"}


def test_upload_confirm_runs_job_and_serves_guide_and_calls(
    tmp_path: Path, lane_db: Path, replay_settings: Settings,
    six_second_clip: Path,
) -> None:
    uploads_routes.set_test_overrides(
        meta_client=ReplayMetaFilesClient())
    client = TestClient(create_app(replay_settings))
    posted = client.post(
        "/upload",
        files={"clip": ("clip.mp4", six_second_clip.read_bytes(), "video/mp4")},
        follow_redirects=False,
    )
    assert posted.status_code == 303
    preview_url = posted.headers["location"]
    assert preview_url.startswith("/preview/")
    job_id = preview_url.split("/preview/")[1]
    confirmed = client.post(
        f"/preview/{job_id}/confirm",
        data={"consent": "on", "ownership": "on"},
        follow_redirects=False,
    )
    assert confirmed.status_code == 303
    assert confirmed.headers["location"] == f"/jobs/{job_id}"
    client.app.state.job_runner.wait(job_id, timeout=120)
    guide = client.get(f"/guides/{job_id}")
    assert guide.status_code == 200
    assert "fan" in guide.text and "battery" in guide.text
    alias = client.get(f"/guide/{job_id}")
    assert alias.status_code == 200
    assert "fan" in alias.text and "battery" in alias.text
    calls = client.get(f"/calls/{job_id}")
    assert calls.status_code == 200
    for stage in ("upload", "extract", "segment", "readback"):
        assert stage in calls.text
    assert "battery" in calls.text  # the natural sample read-back row
    assert "measured example" not in calls.text
    rows = list_call_records(JobId(job_id), path=lane_db)
    assert len(rows) == 9  # 1 upload + 3 extract + 4 segment + 1 read-back
    status = client.get(f"/jobs/{job_id}").json()
    assert status["state"] == "done"
    assert status["guide_url"] == f"/guides/{job_id}"
    assert status["calls_url"] == f"/calls/{job_id}"


def test_sample_flow_builds_and_serves_guide_and_calls(
    tmp_path: Path, lane_db: Path, replay_settings: Settings,
) -> None:
    client = TestClient(create_app(replay_settings))
    first = client.get("/guide/sample", follow_redirects=False)
    assert first.status_code == 303
    location = first.headers["location"]
    assert location.startswith("/guides/")
    job_id = location.split("/guides/")[1]
    guide = client.get(location)
    assert guide.status_code == 200
    assert "Not shown in the video: the battery disconnect" in guide.text  # source gap stays visible
    calls = client.get(f"/calls/{job_id}")
    assert calls.status_code == 200
    for stage in ("upload", "extract", "segment", "readback"):
        assert stage in calls.text
    assert "battery" in calls.text and "measured example" not in calls.text
    rows = list_call_records(JobId(job_id), path=lane_db)
    assert len(rows) == 18 and sum(r.stage == "segment" for r in rows) == 12
    assert sum((r.estimated_cost_usd for r in rows), Decimal("0")) == Decimal("0.35357050")
    assert any(r.stage == "readback" and r.evidence.get("decision") == "yes" for r in rows)
    # Second hit reuses the stored sample job instead of rebuilding.
    second = client.get("/guide/sample", follow_redirects=False)
    assert second.headers["location"] == location
    pointer = tmp_path / "data" / "sample_job.json"
    assert json.loads(pointer.read_text(encoding="utf-8"))["job_id"] == job_id


def test_calls_page_unknown_job_renders_empty(
    tmp_path: Path, lane_db: Path, replay_settings: Settings,
) -> None:
    client = TestClient(create_app(replay_settings))
    page = client.get("/calls/does-not-exist")
    assert page.status_code == 200
    assert "No model calls recorded" in page.text


def test_settings_validation_fails_loudly(tmp_path: Path) -> None:
    base = dict(
        data_root=tmp_path / "data", db_path=tmp_path / "t.sqlite3",
        production=False,
    )
    with pytest.raises(ValueError, match="pipeline_mode"):
        Settings(**base, pipeline_mode="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="guide_filename"):
        Settings(**base, guide_filename="")
    with pytest.raises(ValueError, match="label_image_pattern"):
        Settings(**base, label_image_pattern="label.png")
    with pytest.raises(ValueError, match="settle_max_attempts"):
        Settings(**base, settle_max_attempts=0)
    settings = Settings(**base)
    assert settings.pipeline_mode == "live"
    assert settings.settle_max_attempts == 3
    assert settings.label_image_name(step=1, part=0) == "label_step1_part0.png"
    assert load_settings({"PARTFIDELITY_PIPELINE_MODE": "replay"}).pipeline_mode == (
        "replay"
    )
    assert load_settings(
        {"PARTFIDELITY_SETTLE_MAX_ATTEMPTS": "5"}
    ).settle_max_attempts == 5
    with pytest.raises(RuntimeError, match="PARTFIDELITY_PIPELINE_MODE"):
        load_settings({"PARTFIDELITY_PIPELINE_MODE": "sometimes"})
    with pytest.raises(RuntimeError, match="PARTFIDELITY_SETTLE_MAX_ATTEMPTS"):
        load_settings({"PARTFIDELITY_SETTLE_MAX_ATTEMPTS": "many"})


def test_replay_confirm_uses_replay_meta_with_zero_network_calls(
    tmp_path: Path, lane_db: Path, six_second_clip: Path, monkeypatch,
) -> None:
    """Replay + credentials, no client override: confirm stays offline.

    The first Meta upload (preview confirmation) must resolve through
    the replay-aware service, not the production client the credentials
    would otherwise select. Every socket creation fails the test.
    """
    attempts: list[str] = []
    real_socket = socket.socket

    def _no_network(family=socket.AF_INET, *args, **kwargs):
        # AF_UNIX stays: the test client's own portal needs socketpairs.
        if family == socket.AF_UNIX:
            return real_socket(family, *args, **kwargs)
        attempts.append("socket")
        raise AssertionError("replay mode must make zero network calls")

    def _no_connect(*args, **kwargs):
        attempts.append("connect")
        raise AssertionError("replay mode must make zero network calls")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_connect)
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay", meta_api_key="fake-present",
    )
    client = TestClient(create_app(settings))
    posted = client.post(
        "/upload",
        files={"clip": ("clip.mp4", six_second_clip.read_bytes(), "video/mp4")},
        follow_redirects=False,
    )
    assert posted.status_code == 303
    job_id = posted.headers["location"].split("/preview/")[1]
    confirmed = client.post(
        f"/preview/{job_id}/confirm",
        data={"consent": "on", "ownership": "on"},
        follow_redirects=False,
    )
    assert confirmed.status_code == 303
    assert confirmed.headers["location"] == f"/jobs/{job_id}"
    client.app.state.job_runner.wait(job_id, timeout=120)
    guide = client.get(f"/guides/{job_id}")
    assert guide.status_code == 200
    assert "fan" in guide.text and "battery" in guide.text
    assert attempts == []


def _assert_sample_clip_readable(clip: Path | None) -> None:
    assert clip is not None, "sample clip missing"
    capture = cv2.VideoCapture(str(clip))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame is not None, f"unreadable sample clip: {clip}"


def _assert_sample_job_usable(
    settings: Settings, job_id: JobId, *, where: str
) -> None:
    assert load_guide(
        job_id,
        data_root=Path(settings.data_root),
        guide_filename=settings.guide_filename,
    ) is not None, f"{where}: sample job {job_id} has no stored guide"


def test_concurrent_first_sample_hits_share_one_valid_clip(
    tmp_path: Path, lane_db: Path,
) -> None:
    """Racing first sample hits each finish with a readable clip/guide."""
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay",
    )
    barrier = threading.Barrier(4)
    results: dict[int, JobId] = {}
    errors: dict[int, BaseException] = {}

    def _hit(tag: int) -> None:
        try:
            barrier.wait(timeout=120)
            results[tag] = service.ensure_sample_job(settings)
        except BaseException as error:  # re-raised in the main thread
            errors[tag] = error

    threads = [
        threading.Thread(target=_hit, args=(tag,), daemon=True)
        for tag in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=500)
    assert all(not thread.is_alive() for thread in threads), "sample hit hung"
    assert not errors, {tag: repr(error) for tag, error in errors.items()}
    assert len(results) == 4
    for job_id in set(results.values()):
        _assert_sample_job_usable(settings, job_id, where="threads")
    assert len(set(results.values())) == 1
    _assert_sample_clip_readable(
        service.sample_clip_path(settings, str(next(iter(results.values()))))
    )


def _sample_job_worker(data_root: str, db_path: str, queue) -> None:
    """Process-spawn entry: build-or-reuse the sample job, report back."""
    from app import db as worker_db
    from app import service as worker_service
    from app.settings import Settings as WorkerSettings

    try:
        database = Path(db_path)
        worker_db.configure(database, Path(data_root))
        settings = WorkerSettings(
            data_root=Path(data_root), db_path=database, production=False,
            pipeline_mode="replay",
        )
        queue.put(("ok", str(worker_service.ensure_sample_job(settings))))
    except BaseException as error:
        queue.put(("error", repr(error)))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "app/sample_real/sample.mp4").is_file(),
    reason="real sample media absent; run python tools/sample_media.py unpack",
)
def test_concurrent_first_sample_hits_across_processes(
    tmp_path: Path, lane_db: Path,
) -> None:
    """Two processes racing the first build each finish with a guide."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_sample_job_worker,
            args=(str(tmp_path / "data"), str(lane_db), queue),
            daemon=True,
        )
        for _ in range(2)
    ]
    for proc in procs:
        proc.start()
    outcomes = [queue.get(timeout=500) for _ in procs]
    for proc in procs:
        proc.join(timeout=120)
    assert all(proc.exitcode == 0 for proc in procs), [
        proc.exitcode for proc in procs
    ]
    assert [status for status, _ in outcomes] == ["ok", "ok"], outcomes
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay",
    )
    for _, job_id in outcomes:
        _assert_sample_job_usable(settings, JobId(job_id), where="processes")
    assert len({job_id for _, job_id in outcomes}) == 1
    _assert_sample_clip_readable(service.sample_clip_path(settings, outcomes[0][1]))
