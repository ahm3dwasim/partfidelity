"""Wire providers: live provider factory wired into the pipeline.

Every test runs offline: the providers trio talks to a scripted
``HttpTransport`` and the Meta Files client to a scripted opener. The
scripted provider transport queries the real ledger on every POST and
refuses a call its reservation does not already cover, so the
end-to-end test proves reserve-before-send through the ledger itself,
not through the wrappers' good intentions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import db, service, stage_cache
from app.admission import admit_job
from app.contracts import (
    RECEIPT_MODEL_META_FILES,
    RECEIPT_MODEL_MUSE_SPARK,
    RECEIPT_MODEL_SAM,
    RECEIPT_MODEL_SPARK,
    RECEIPT_MODEL_SPARK_OPEN,
    RECEIPT_STAGE_EXTRACT,
    RECEIPT_STAGE_READBACK,
    RECEIPT_STAGE_SEGMENT,
    RECEIPT_STAGE_UPLOAD,
    ArtifactId,
    CallRecord,
    Detection,
    JobId,
    LabelSlot,
    PartRequest,
    ProvisionalRender,
)
from app.evidence import record_call
from app.ledger import LifecycleRepositoryImpl, ReserveRefused, list_call_records
from app.main import create_app
from app.meta_files import ProductionMetaFilesClient, select_meta_files_client
from app.pipeline import (
    PipelineClients,
    RecordingMetaClient,
    default_replay_clients,
    ReplayMetaFilesClient,
    ReplaySamClient,
    ReplaySparkClient,
    load_guide,
    readback_record_call,
    readback_record_event,
    run_confirmed_job,
    run_job,
    sam_receipt_sink,
)
from app.providers import select_providers
from app.providers._transport import HttpResponse
from app.providers.meta_sam import MetaSamClient
from app.readback import RESERVE_KIND as READBACK_RESERVE_KIND
from app.readback import ReadbackResult, read_back_label, run_readbacks
from app.sam import SAM_RESERVE_KIND
from app.sam import (
    SamDeadlineExceeded,
    SamMask,
    SamStillResult,
    install_call_sink,
    reset_call_sink,
    reset_sam_job_counts,
    segment,
)
from app.service import (
    CombinedLiveSparkClient,
    PipelineUnavailable,
    live_pipeline_clients,
)
from app.settings import ProviderSettings, Settings
from app.spark import RESERVE_KIND as SPARK_RESERVE_KIND
from app.spark import SparkStageConfig, extract_steps
from app.uploads import confirm_and_send, hash_file, prepare_preview
from app.uploads_store import UploadStoreImpl

# Distinctive fake key: greppable, and shaped nothing like a real one.
FAKE_KEY = "wire-providers-test-key-9f8e6d5c4b3a"


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
    return Settings(
        data_root=tmp_path / "data",
        db_path=lane_db,
        production=False,
        meta_api_key=FAKE_KEY,
        pipeline_mode="live",
        providers=ProviderSettings(mode="live"),
    )


@pytest.fixture(autouse=True)
def _clean_wiring(tmp_path: Path):
    stage_cache.configure_cache_root(tmp_path / "cache")
    reset_sam_job_counts()
    reset_call_sink()
    service.clear_client_factory()
    yield
    reset_call_sink()
    reset_sam_job_counts()
    service.clear_client_factory()
    stage_cache.configure_cache_root(None)


# ---------------------------------------------------------------------------
# Offline fakes
# ---------------------------------------------------------------------------


class LedgerOracleTransport:
    """Scripted HttpTransport that proves reserve-before-send per POST.

    Entries are ``HttpResponse`` (or an ``Exception`` to raise). On
    every POST the transport counts the job's metered reserves in the
    real ledger and refuses when the POST would outnumber them: a call
    that sends without its wrapper's prior reserve fails here, at send
    time. The Meta upload carries no reservation (documented $0
    shortcut in :class:`RecordingMetaClient`), so it never posts here.
    """

    KINDS = (SPARK_RESERVE_KIND, SAM_RESERVE_KIND, READBACK_RESERVE_KIND)

    def __init__(self, script, *, database: Path, job_id: str) -> None:
        self._script = list(script)
        self._database = database
        self._job_id = job_id
        self.posts: list[dict] = []

    @property
    def remaining(self) -> int:
        return len(self._script)

    def _reserve_count(self) -> int:
        conn = sqlite3.connect(self._database)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM reservations WHERE job_id = ?"
                " AND call_kind IN (?, ?, ?)",
                (self._job_id, *self.KINDS),
            ).fetchone()
            return int(row[0])
        finally:
            conn.close()

    def post(self, url, *, headers, body, timeout_s):
        self.posts.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_s": timeout_s,
            }
        )
        reserves = self._reserve_count()
        if len(self.posts) > reserves:
            raise AssertionError(
                f"provider POST #{len(self.posts)} sent with only "
                f"{reserves} ledger reserves"
            )
        if not self._script:
            raise AssertionError("transport script exhausted")
        entry = self._script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry


class _CannedReply:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode()
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _CannedReply:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class ScriptedOpener:
    """Opener-shaped fake for the Meta Files client (upload only)."""

    def __init__(self, *, file_id: str = "meta-live-1") -> None:
        self.calls: list[dict] = []
        self._file_id = file_id

    def __call__(self, request, timeout=None):
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
            }
        )
        assert request.get_method() == "POST", self.calls[-1]
        assert "/files" in request.full_url, self.calls[-1]
        expires_at = db.utcnow() + timedelta(hours=1)
        return _CannedReply(
            {"id": self._file_id, "expires_at": expires_at.isoformat()}
        )


def _ok(payload: dict) -> HttpResponse:
    return HttpResponse(
        status=200, body=json.dumps(payload).encode(), headers={}
    )


def _spark_response(text_obj: dict, in_tok: int = 16000, out_tok: int = 2000) -> dict:
    return {
        "output": [
            {"content": [{"type": "output_text", "text": json.dumps(text_obj)}]}
        ],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def _sam_response(text: str) -> dict:
    return {"output": [{"content": [{"type": "output_text", "text": text}]}]}


def _readback_response(visible: bool, in_tok: int = 500, out_tok: int = 50) -> dict:
    return {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"visible": visible}),
                    }
                ]
            }
        ],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def _box_token(c: float) -> str:
    return (
        "<|box;x1=12;y1=30;x2=140;y2=150;w=320;h=240"
        f";c={c}|>"
    )


def _step_payload(
    number: int,
    start: float,
    end: float,
    action: str,
    parts: list[tuple[str, str]],
    tools: tuple[str, ...] = (),
    hazards: tuple[str, ...] = (),
    kind: str = "repair",
) -> dict:
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


def _recorded_runs() -> list[dict]:
    """Recorded step shapes: the canonical replay windows as API text."""
    windows = [(0.5, 2.5), (0.6, 2.6), (0.4, 2.4)]
    # Contiguous partitions (time consensus excludes gapped runs).
    battery_windows = [(2.5, 5.5), (2.6, 5.6), (2.4, 5.4)]
    runs = []
    for (start, end), (bstart, bend) in zip(windows, battery_windows):
        runs.append(
            {
                "device_guess": "laptop",
                "confidence": 0.82,
                "steps": [
                    _step_payload(
                        1, start, end, "Remove the rear screws",
                        [("fan", "left side")], tools=("Phillips #0",),
                    ),
                    _step_payload(
                        2, bstart, bend, "Disconnect the battery cable",
                        [("battery", "lower right")], hazards=("battery",),
                    ),
                ],
            }
        )
    return runs


def _confirm_live_job(
    uploads: UploadStoreImpl,
    clip: Path,
    opener: ScriptedOpener,
    settings: Settings,
):
    """Admit, preview and confirm one upload against the fake opener."""
    ctx = admit_job("anon-live", "ip-live", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    meta = select_meta_files_client(settings, transport=opener)
    file_id = confirm_and_send(
        ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx)
    )
    return ctx, file_id


def _held_count(database: Path, job_id: str) -> int:
    conn = sqlite3.connect(database)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE job_id = ? AND state = 'held'",
            (job_id,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _write_frame(path: Path, size: int = 100) -> Path:
    frame = np.full((size, size), 200, np.uint8)
    assert cv2.imwrite(str(path), frame)
    return path


def _mask_bytes(box: tuple[int, int, int, int], size: int = 100) -> bytes:
    mask = np.zeros((size, size), np.uint8)
    mask[box[1] : box[3], box[0] : box[2]] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    return bytes(encoded)


class _ExpiredClock:
    """Monotonic clock already past any job deadline it mints."""

    def __init__(self) -> None:
        self._first = True

    def __call__(self) -> float:
        if self._first:
            self._first = False
            return 1000.0
        return 999999.0


# ---------------------------------------------------------------------------
# Step 1: the receipt vocabulary agrees end to end
# ---------------------------------------------------------------------------


def test_sam_and_readback_receipts_accepted_by_evidence(
    tmp_path: Path, lane_db: Path
) -> None:
    """A SAM call and a read-back call persist via real evidence.record_call.

    Both go through the pipeline seam (sam_receipt_sink /
    readback_record_call) against a real ledger context — no lambdas —
    and read back as the evidence module-valid segment/readback rows.
    """
    ctx = admit_job("anon-vocab-wire", "ip-vocab-wire", "video-vocab-wire")
    frame = _write_frame(tmp_path / "frame.png")

    class _FakeSam:
        def segment_still(
            self, image_path, phrase, *, include_confidence=True
        ):
            return SamStillResult(
                masks=(
                    SamMask(
                        box_xyxy=(10, 10, 50, 50),
                        c=0.72,
                        mask_png=_mask_bytes((10, 10, 50, 50)),
                    ),
                )
            )

    install_call_sink(sam_receipt_sink)
    try:
        detections = segment(ctx, frame, "fan", client=_FakeSam())  # type: ignore[arg-type]
    finally:
        reset_call_sink()
    assert len(detections) == 1

    class _FakeSpark:
        def read_back(self, image_path, phrase, position_hint):
            return ReadbackResult(
                answer="yes", cost_usd=Decimal("0.01"), latency_ms=9
            )

    provisional = ProvisionalRender(
        detection=Detection(
            phrase="battery",
            mask_path=tmp_path / "m.png",
            score=0.61,
            box_xyxy=(0, 0, 10, 10),
        ),
        artifact_id=ArtifactId("wire-rb-1"),
        rendered_image_path=frame,
        position_hint="lower right",
    )
    slot = LabelSlot(
        step_index=1,
        part_index=0,
        part=PartRequest(phrase="battery", position_hint="lower right"),
    )
    decisions = run_readbacks(
        ctx,
        (provisional,),
        slots=(slot,),
        client=_FakeSpark(),  # type: ignore[arg-type]
        record_call=readback_record_call,
        record_event=readback_record_event,
    )
    assert decisions[0].readback == "yes"

    rows = list_call_records(ctx.job_id, path=lane_db)
    segments = [row for row in rows if row.stage == RECEIPT_STAGE_SEGMENT]
    readbacks = [row for row in rows if row.stage == RECEIPT_STAGE_READBACK]
    assert len(segments) == 1
    assert len(readbacks) == 1
    segment_row = segments[0]
    assert segment_row.model == RECEIPT_MODEL_SAM
    assert segment_row.endpoint == "/segment_stills"
    assert segment_row.status == "succeeded"
    assert segment_row.evidence == {"phrase": "fan", "score": 0.72}
    readback_row = readbacks[0]
    assert readback_row.model == RECEIPT_MODEL_MUSE_SPARK
    assert readback_row.endpoint == "/read_back"
    assert readback_row.status == "succeeded"
    assert readback_row.evidence["decision"] == "yes"
    assert readback_row.evidence["phrase"] == "battery"

    # The seam is load-bearing: the raw module shapes are still rejected.
    with pytest.raises(ValueError, match="stage"):
        record_call(
            ctx,
            CallRecord(
                job_id=ctx.job_id, stage="sam", model=RECEIPT_MODEL_SAM,
                endpoint="/segment_stills", latency_ms=1, units_in=None,
                units_out=None, estimated_cost_usd=Decimal("0.0025"),
                status="succeeded", evidence={},
            ),
        )
    with pytest.raises(ValueError, match="endpoint"):
        record_call(
            ctx,
            CallRecord(
                job_id=ctx.job_id, stage=RECEIPT_STAGE_READBACK,
                model=RECEIPT_MODEL_MUSE_SPARK, endpoint="read_back",
                latency_ms=1, units_in=None, units_out=None,
                estimated_cost_usd=Decimal("0.01"), status="succeeded",
                evidence={},
            ),
        )


# ---------------------------------------------------------------------------
# Step 2: the service builds live sets from the provider factory
# ---------------------------------------------------------------------------


def test_resolve_clients_replay_ignores_context(live_settings: Settings) -> None:
    live_settings = Settings(
        data_root=live_settings.data_root,
        db_path=live_settings.db_path,
        production=False,
        meta_api_key=FAKE_KEY,
        pipeline_mode="replay",
        providers=ProviderSettings(mode="live"),
    )
    clients = service.resolve_clients(live_settings, ctx=None)
    assert isinstance(clients.spark, ReplaySparkClient)
    assert isinstance(clients.sam, ReplaySamClient)
    assert isinstance(clients.meta, ReplayMetaFilesClient)


def test_resolve_clients_live_needs_context(
    live_settings: Settings, lane_db: Path
) -> None:
    with pytest.raises(PipelineUnavailable, match="job context"):
        service.resolve_clients(live_settings, ctx=None)


def test_resolve_clients_live_builds_unmixed_set(
    live_settings: Settings, lane_db: Path, tmp_path: Path
) -> None:
    ctx = admit_job("anon-wire", "ip-wire", "video-wire")
    opener = ScriptedOpener()
    clients = service.resolve_clients(
        live_settings, ctx,
        transport=LedgerOracleTransport(
            [], database=lane_db, job_id=str(ctx.job_id)
        ),
        meta_transport=opener,
    )
    assert isinstance(clients.spark, CombinedLiveSparkClient)
    assert isinstance(clients.sam, MetaSamClient)
    assert isinstance(clients.meta, ProductionMetaFilesClient)
    for client in (clients.spark, clients.sam, clients.meta):
        assert not isinstance(
            client, (ReplaySparkClient, ReplaySamClient, ReplayMetaFilesClient)
        )


def test_resolve_clients_live_without_key(
    tmp_path: Path, lane_db: Path
) -> None:
    settings = Settings(
        data_root=tmp_path / "data",
        db_path=lane_db,
        production=False,
        meta_api_key=None,
        pipeline_mode="live",
        providers=ProviderSettings(mode="live"),
    )
    ctx = admit_job("anon-nokey", "ip-nokey", "video-nokey")
    with pytest.raises(PipelineUnavailable, match="META_API_KEY"):
        service.resolve_clients(settings, ctx)


def test_resolve_clients_live_mode_mismatch(
    tmp_path: Path, lane_db: Path
) -> None:
    settings = Settings(
        data_root=tmp_path / "data",
        db_path=lane_db,
        production=False,
        meta_api_key=FAKE_KEY,
        pipeline_mode="live",
        providers=ProviderSettings(mode="replay"),
    )
    ctx = admit_job("anon-mismatch", "ip-mismatch", "video-mismatch")
    with pytest.raises(PipelineUnavailable):
        service.resolve_clients(settings, ctx)


def test_resolve_clients_override_factory_wins(
    live_settings: Settings, lane_db: Path
) -> None:
    ctx = admit_job("anon-override", "ip-override", "video-override")
    sentinel = live_pipeline_clients(
        live_settings, ctx,
        transport=LedgerOracleTransport(
            [], database=lane_db, job_id=str(ctx.job_id)
        ),
        meta_transport=ScriptedOpener(),
    )
    service.set_client_factory(lambda settings: sentinel)
    try:
        assert service.resolve_clients(live_settings, ctx) is sentinel
    finally:
        service.clear_client_factory()


def test_run_upload_job_unknown_job_live_builds_nothing(
    live_settings: Settings, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    calls: list[Settings] = []

    def _factory(settings: Settings):
        calls.append(settings)
        raise AssertionError("factory must not run for an unknown job")

    service.set_client_factory(_factory)
    try:
        result = service.run_upload_job(
            JobId("does-not-exist"), live_settings,
            uploads=uploads, lifecycle=lifecycle,
        )
    finally:
        service.clear_client_factory()
    assert result.state == "failed"
    assert result.error == "unknown job"
    assert calls == []


def test_run_upload_job_live_without_key_is_unavailable(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data",
        db_path=lane_db,
        production=False,
        meta_api_key=None,
        pipeline_mode="live",
        providers=ProviderSettings(mode="live"),
    )
    clip = service.write_sample_clip(tmp_path / "clip.mp4")
    ctx = admit_job("anon-route", "ip-route", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    with pytest.raises(PipelineUnavailable):
        service.run_upload_job(
            ctx.job_id, settings, uploads=uploads, lifecycle=lifecycle
        )


def test_combined_spark_client_routes_and_hides_key() -> None:
    seen: list[tuple] = []

    class _Steps:
        spark_stage_config = SparkStageConfig()

        def run_steps(self, video_file_id: str, clip_duration_s=None, scene_cuts=None, clocked=False):
            seen.append(("run_steps", video_file_id))
            return "steps-result"

    class _Readback:
        def read_back(self, image_path, phrase, position_hint):
            seen.append(("read_back", phrase))
            return "readback-result"

    combined = CombinedLiveSparkClient(_Steps(), _Readback())
    assert combined.run_steps("file-1") == "steps-result"
    assert (
        combined.read_back(Path("r.png"), "fan", "left") == "readback-result"
    )
    assert seen == [("run_steps", "file-1"), ("read_back", "fan")]
    assert isinstance(combined.spark_stage_config, SparkStageConfig)
    assert FAKE_KEY not in repr(combined)


def test_select_meta_files_client_accepts_fake_transport(
    live_settings: Settings,
) -> None:
    opener = ScriptedOpener(file_id="meta-kwarg-1")
    client = select_meta_files_client(live_settings, transport=opener)
    assert isinstance(client, ProductionMetaFilesClient)
    assert FAKE_KEY not in repr(client)


# ---------------------------------------------------------------------------
# Step 4: a live-mode job end to end on fake transports
# ---------------------------------------------------------------------------


def test_live_job_end_to_end_fake_transports(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One live job, clip to guide: reserves first, receipts accepted.

    The provider transport refuses any POST its reservation does not
    already cover (see :class:`LedgerOracleTransport`); the Meta
    upload records its $0 receipt with no reservation (documented
    shortcut: the Files API is unmetered, and waiting jobs confirm
    before they activate).
    """
    caplog.set_level(logging.INFO)
    clip = service.write_sample_clip(tmp_path / "clip.mp4")

    confirm_opener = ScriptedOpener(file_id="meta-live-e2e")
    admitted, file_id = _confirm_live_job(
        uploads, clip, confirm_opener, live_settings
    )
    assert str(file_id) == "meta-live-e2e"
    assert len(confirm_opener.calls) == 1

    loaded = uploads.load_job(admitted.job_id)
    assert loaded is not None
    ctx, record = loaded
    assert record.state == "confirmed"
    assert record.meta_file_id is not None

    script = (
        [_ok(_spark_response(run)) for run in _recorded_runs()]
        + [_ok(_sam_response(_box_token(c))) for c in (0.72, 0.60)]
        + [_ok(_sam_response(_box_token(c))) for c in (0.61, 0.55)]
        + [_ok(_readback_response(True))]
    )
    transport = LedgerOracleTransport(
        script, database=lane_db, job_id=str(ctx.job_id)
    )
    stages_opener = ScriptedOpener(file_id="meta-must-not-send")
    clients = service.resolve_clients(
        live_settings, ctx,
        transport=transport, meta_transport=stages_opener,
    )
    result = run_confirmed_job(
        ctx.job_id, clients=clients, uploads=uploads,
        lifecycle=lifecycle, replay=False, settings=live_settings,
    )
    assert result.state == "done", result.error
    assert result.job_id == ctx.job_id
    assert load_guide(
        ctx.job_id, data_root=live_settings.data_root,
        guide_filename=live_settings.guide_filename,
    ) is not None

    # Every scripted call sent exactly once, each under its reserve
    # (the oracle refused otherwise at send time).
    assert len(transport.posts) == 8
    assert transport.remaining == 0
    assert stages_opener.calls == []
    for post in transport.posts:
        assert post["headers"].get("Authorization") == "Bearer " + FAKE_KEY
        assert post["url"].endswith("/responses")

    # One accepted receipt per call: 3 extract + 4 segment + 1 readback
    # from the live clients, plus the $0 upload receipt from confirm.
    rows = list_call_records(ctx.job_id, path=lane_db)
    by_stage: dict[str, list] = {}
    for row in rows:
        if row.model != RECEIPT_MODEL_SPARK_OPEN:  # the background second check, tested on its own
            by_stage.setdefault(row.stage, []).append(row)
    assert set(by_stage) == {
        RECEIPT_STAGE_EXTRACT,
        RECEIPT_STAGE_SEGMENT,
        RECEIPT_STAGE_READBACK,
        RECEIPT_STAGE_UPLOAD,
    }
    assert len(by_stage[RECEIPT_STAGE_EXTRACT]) == 3
    assert len(by_stage[RECEIPT_STAGE_SEGMENT]) == 4
    assert len(by_stage[RECEIPT_STAGE_READBACK]) == 1
    assert len(by_stage[RECEIPT_STAGE_UPLOAD]) == 1
    for row in by_stage[RECEIPT_STAGE_EXTRACT]:
        assert row.status == "succeeded"
        assert row.model == RECEIPT_MODEL_SPARK
        assert row.endpoint == "/responses"
    for row in by_stage[RECEIPT_STAGE_SEGMENT]:
        assert row.status == "succeeded"
        assert row.model == RECEIPT_MODEL_SAM
        assert row.endpoint == "/responses"
    (readback_row,) = by_stage[RECEIPT_STAGE_READBACK]
    assert readback_row.status == "succeeded"
    assert readback_row.model == RECEIPT_MODEL_SPARK
    assert readback_row.endpoint == "/responses"
    assert readback_row.evidence.get("decision") == "yes"
    (upload_row,) = by_stage[RECEIPT_STAGE_UPLOAD]
    assert upload_row.status == "succeeded"
    assert upload_row.model == RECEIPT_MODEL_META_FILES
    assert upload_row.endpoint == "/files"
    assert upload_row.estimated_cost_usd == Decimal("0")

    # Nothing held: every reserve settled.
    assert _held_count(lane_db, str(ctx.job_id)) == 0

    # The key travelled only in Authorization headers: never in logs,
    # receipts or reprs.
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in json.dumps(
        [
            {
                "stage": row.stage,
                "model": row.model,
                "endpoint": row.endpoint,
                "status": row.status,
                "evidence": row.evidence,
            }
            for row in rows
        ],
        sort_keys=True,
        default=str,
    )
    assert FAKE_KEY not in repr(clients.spark)
    assert FAKE_KEY not in repr(clients.sam)
    assert FAKE_KEY not in repr(clients.meta)
    assert FAKE_KEY not in repr(clients)


def _spend_envelope(ctx) -> None:
    """Hold the full metered envelope so the next reserve refuses."""
    ctx.reserve(SPARK_RESERVE_KIND, Decimal("0.26"))
    ctx.reserve(SAM_RESERVE_KIND, Decimal("0.08"))
    ctx.reserve(READBACK_RESERVE_KIND, Decimal("0.12"))


def test_live_reserve_refusal_sends_nothing(
    tmp_path: Path, lane_db: Path, live_settings: Settings
) -> None:
    """A refused reservation never reaches the provider transport."""
    frame = _write_frame(tmp_path / "refuse.png")

    spark_ctx = admit_job("anon-rs", "ip-rs", "video-rs")
    _spend_envelope(spark_ctx)
    spark_transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(spark_ctx.job_id)
    )
    spark_trio = select_providers(
        live_settings, spark_ctx, transport=spark_transport
    )
    with pytest.raises(ReserveRefused):
        extract_steps(spark_ctx, "file-refused", client=spark_trio.spark)  # type: ignore[arg-type]
    assert spark_transport.posts == []
    spark_ctx.finish("failed")

    sam_ctx = admit_job("anon-rm", "ip-rm", "video-rm")
    _spend_envelope(sam_ctx)
    sam_transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(sam_ctx.job_id)
    )
    sam_trio = select_providers(
        live_settings, sam_ctx, transport=sam_transport
    )
    with pytest.raises(ReserveRefused):
        segment(sam_ctx, frame, "fan", client=sam_trio.sam)  # type: ignore[arg-type]
    assert sam_transport.posts == []
    sam_ctx.finish("failed")

    rb_ctx = admit_job("anon-rb", "ip-rb", "video-rb")
    _spend_envelope(rb_ctx)
    rb_transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(rb_ctx.job_id)
    )
    rb_trio = select_providers(live_settings, rb_ctx, transport=rb_transport)
    provisional = ProvisionalRender(
        detection=Detection(
            phrase="battery",
            mask_path=tmp_path / "m.png",
            score=0.61,
            box_xyxy=(0, 0, 10, 10),
        ),
        artifact_id=ArtifactId("wire-rb-refuse"),
        rendered_image_path=frame,
        position_hint="lower right",
    )
    decision = read_back_label(
        rb_ctx, provisional,
        slot=LabelSlot(
            step_index=1, part_index=0,
            part=PartRequest(phrase="battery", position_hint="lower right"),
        ),
        client=rb_trio.readback,  # type: ignore[arg-type]
    )
    assert decision.readback == "not_run"
    assert rb_transport.posts == []


def test_live_deadline_ends_the_job_before_send(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
) -> None:
    """An expired job deadline fails the job with nothing sent."""
    clip = service.write_sample_clip(tmp_path / "clip.mp4")
    admitted, _ = _confirm_live_job(
        uploads, clip, ScriptedOpener(), live_settings
    )
    loaded = uploads.load_job(admitted.job_id)
    assert loaded is not None
    ctx, _record = loaded

    transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(ctx.job_id)
    )
    trio = select_providers(
        live_settings, ctx, transport=transport, monotonic=_ExpiredClock()
    )
    clients = PipelineClients(
        spark=CombinedLiveSparkClient(trio.spark, trio.readback),  # type: ignore[arg-type]
        sam=trio.sam,
        meta=select_meta_files_client(
            live_settings, transport=ScriptedOpener()
        ),
    )
    result = run_confirmed_job(
        ctx.job_id, clients=clients, uploads=uploads,
        lifecycle=lifecycle, replay=False, settings=live_settings,
    )
    assert result.state == "failed"
    assert transport.posts == []
    rows = list_call_records(ctx.job_id, path=lane_db)
    extracts = [row for row in rows if row.stage == RECEIPT_STAGE_EXTRACT]
    assert len(extracts) == 1
    assert extracts[0].status == "failed"
    assert extracts[0].estimated_cost_usd == Decimal("0")
    assert _held_count(lane_db, str(ctx.job_id)) == 0


def test_live_sam_deadline_refuses_before_send(
    tmp_path: Path, lane_db: Path, live_settings: Settings
) -> None:
    """A SAM pacing wait past the deadline sends nothing ($0 receipt)."""
    ctx = admit_job("anon-samd", "ip-samd", "video-samd")
    frame = _write_frame(tmp_path / "samd.png")
    transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(ctx.job_id)
    )
    trio = select_providers(
        live_settings, ctx, transport=transport, monotonic=_ExpiredClock()
    )
    with pytest.raises(SamDeadlineExceeded):
        segment(ctx, frame, "fan", client=trio.sam)  # type: ignore[arg-type]
    assert transport.posts == []
    rows = list_call_records(ctx.job_id, path=lane_db)
    segments = [row for row in rows if row.stage == RECEIPT_STAGE_SEGMENT]
    assert len(segments) == 1
    assert segments[0].status == "failed"
    assert segments[0].estimated_cost_usd == Decimal("0")


def test_live_mode_rejects_replay_clients(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
) -> None:
    """Live runs never mix: replay adapters are refused before any effect."""
    clip = service.write_sample_clip(tmp_path / "clip.mp4")
    with pytest.raises(ValueError, match="live mode rejects"):
        run_job(
            clip, clients=default_replay_clients(), uploads=uploads,
            lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=False,
            settings=live_settings,
        )


def _reservation_count(database: Path, job_id: str) -> int:
    conn = sqlite3.connect(database)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE job_id = ?", (job_id,)
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _jobs_count(database: Path) -> int:
    conn = sqlite3.connect(database)
    try:
        row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return int(row[0])
    finally:
        conn.close()


def test_live_bundle_from_other_job_cannot_run(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
) -> None:
    """A bundle bound to ctx_a refuses to run under ctx_b's job.

    The refusal lands before any reservation, request or receipt, so
    neither job gains a POST, a reservation row or a call record.
    """
    clip = service.write_sample_clip(tmp_path / "clip.mp4")
    ctx_a = admit_job("anon-ctxa", "ip-ctxa", "video-ctxa")
    transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(ctx_a.job_id)
    )
    trio = select_providers(live_settings, ctx_a, transport=transport)
    foreign = PipelineClients(
        spark=CombinedLiveSparkClient(trio.spark, trio.readback),  # type: ignore[arg-type]
        sam=trio.sam,
        meta=select_meta_files_client(
            live_settings, transport=ScriptedOpener()
        ),
    )
    admitted_b, _ = _confirm_live_job(
        uploads, clip, ScriptedOpener(file_id="meta-live-b"), live_settings
    )
    loaded = uploads.load_job(admitted_b.job_id)
    assert loaded is not None
    ctx_b, _record = loaded
    assert str(ctx_a.job_id) != str(ctx_b.job_id)
    res_a_before = _reservation_count(lane_db, str(ctx_a.job_id))
    res_b_before = _reservation_count(lane_db, str(ctx_b.job_id))
    records_b_before = list_call_records(ctx_b.job_id, path=lane_db)

    with pytest.raises(ValueError, match="bound to job"):
        run_confirmed_job(
            ctx_b.job_id, clients=foreign, uploads=uploads,
            lifecycle=lifecycle, replay=False, settings=live_settings,
        )

    assert transport.posts == []
    assert _reservation_count(lane_db, str(ctx_a.job_id)) == res_a_before
    assert _reservation_count(lane_db, str(ctx_b.job_id)) == res_b_before
    assert list(list_call_records(ctx_a.job_id, path=lane_db)) == []
    assert list_call_records(ctx_b.job_id, path=lane_db) == records_b_before


def test_combined_spark_client_refuses_mixed_jobs(
    lane_db: Path, live_settings: Settings
) -> None:
    """Steps and read-back halves bound to different jobs cannot combine."""
    ctx_a = admit_job("anon-mixa", "ip-mixa", "video-mixa")
    ctx_b = admit_job("anon-mixb", "ip-mixb", "video-mixb")
    trio_a = select_providers(
        live_settings, ctx_a,
        transport=LedgerOracleTransport(
            [], database=lane_db, job_id=str(ctx_a.job_id)
        ),
    )
    trio_b = select_providers(
        live_settings, ctx_b,
        transport=LedgerOracleTransport(
            [], database=lane_db, job_id=str(ctx_b.job_id)
        ),
    )
    with pytest.raises(ValueError, match="different jobs"):
        CombinedLiveSparkClient(trio_a.spark, trio_b.readback)


def test_run_job_live_refuses_before_admission(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings,
) -> None:
    """Direct live run_job calls are refused before admission or request.

    run_job admits its own job after the bundle was built, so any
    pre-bound live bundle would receipt against the wrong job; live
    jobs run only through the confirmed-job path.
    """
    clip = service.write_sample_clip(tmp_path / "clip.mp4")
    ctx_a = admit_job("anon-rjref", "ip-rjref", "video-rjref")
    transport = LedgerOracleTransport(
        [], database=lane_db, job_id=str(ctx_a.job_id)
    )
    trio = select_providers(live_settings, ctx_a, transport=transport)
    bundle = PipelineClients(
        spark=CombinedLiveSparkClient(trio.spark, trio.readback),  # type: ignore[arg-type]
        sam=trio.sam,
        meta=select_meta_files_client(
            live_settings, transport=ScriptedOpener()
        ),
    )
    jobs_before = _jobs_count(lane_db)
    with pytest.raises(ValueError, match="confirmed-job path"):
        run_job(
            clip, clients=bundle, uploads=uploads,
            lifecycle=lifecycle, anon_id="a", ip_hash="i", replay=False,
            settings=live_settings,
        )
    assert _jobs_count(lane_db) == jobs_before
    assert transport.posts == []
    assert list(list_call_records(ctx_a.job_id, path=lane_db)) == []


# ---------------------------------------------------------------------------
# Step 3: create_app binds the stage cache to the data root
# ---------------------------------------------------------------------------


def test_create_app_binds_stage_cache_to_data_root(tmp_path: Path) -> None:
    data = tmp_path / "appdata"
    settings = Settings(
        data_root=data, db_path=tmp_path / "app.sqlite3", production=False
    )
    create_app(settings)
    try:
        assert stage_cache.cache_root() == data / "stage_cache"
    finally:
        stage_cache.configure_cache_root(None)


# ---------------------------------------------------------------------------
# Spark Open second check: never on the user's path
# ---------------------------------------------------------------------------


def _live_guide_job(tmp_path, name, uploads, lifecycle, lane_db, settings):
    clip = service.write_sample_clip(tmp_path / f"{name}.mp4")
    ctx = admit_job(f"anon-{name}", f"ip-{name}", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    meta = select_meta_files_client(settings, transport=ScriptedOpener(file_id=f"meta-{name}"))
    confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    script = (
        [_ok(_spark_response(run)) for run in _recorded_runs()]
        + [_ok(_sam_response(_box_token(c))) for c in (0.72, 0.60)]
        + [_ok(_sam_response(_box_token(c))) for c in (0.61, 0.55)]
        + [_ok(_readback_response(True))]
    )
    clients = service.resolve_clients(
        settings, ctx,
        transport=LedgerOracleTransport(script, database=lane_db, job_id=str(ctx.job_id)),
        meta_transport=ScriptedOpener(file_id="meta-must-not-send"),
    )
    result = run_confirmed_job(ctx.job_id, clients=clients, uploads=uploads,
                               lifecycle=lifecycle, replay=False, settings=settings)
    assert result.state == "done", result.error
    guide = (settings.data_root / "jobs" / str(ctx.job_id) / settings.guide_filename)
    if not guide.exists():
        guide = next(settings.data_root.rglob(settings.guide_filename))
    text = guide.read_text().replace(str(ctx.job_id), "JOB")
    for artifact_id in {(a.get("provisional") or {}).get("artifact_id") for a in json.loads(text)["labels"]}:
        if artifact_id:
            text = text.replace(artifact_id, "ARTIFACT")  # fresh random id per job
    return ctx, text


def test_second_check_held_blocked_never_delays_or_changes_the_guide(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, live_settings: Settings, monkeypatch,
) -> None:
    import threading

    from app.providers import open_readback

    cfg = open_readback.load_config()
    monkeypatch.setattr(open_readback, "load_config", lambda: replace(cfg, enabled=False))
    _, disabled = _live_guide_job(tmp_path, "off", uploads, lifecycle, lane_db, live_settings)

    entered, release, posts = threading.Event(), threading.Event(), []

    class Held:
        def post(self, url, *, headers, body, timeout_s):
            posts.append(url)
            entered.set()
            assert release.wait(10)
            return HttpResponse(200, json.dumps({
                "choices": [{"message": {"content": "no"}}],
                "usage": {"prompt_tokens": 470, "completion_tokens": 60}}).encode(), {})

    monkeypatch.setattr(open_readback, "load_config", lambda: cfg)
    monkeypatch.setattr(open_readback, "UrllibTransport", Held)
    monkeypatch.setenv("BASETEN_API_KEY", "fake-baseten-test-key")
    try:
        ctx, enabled = _live_guide_job(tmp_path, "on", uploads, lifecycle, lane_db, live_settings)
        # The job is done and its guide stored while the second check is still held in its call.
        assert entered.wait(10) and not release.is_set()
        assert enabled == disabled
        assert not [r for r in list_call_records(ctx.job_id, path=lane_db)
                    if r.model == RECEIPT_MODEL_SPARK_OPEN]
        assert open_readback.cancel(ctx.job_id) is False  # returns at once while a call is held
    finally:
        release.set()
    assert open_readback.wait_quiet(ctx.job_id, 10) is True  # the held calls finish
    rows = [r for r in list_call_records(ctx.job_id, path=lane_db) if r.model == RECEIPT_MODEL_SPARK_OPEN]
    assert rows and len(rows) == len(posts)
    assert all(r.evidence.get("decision") == "no" for r in rows)  # a disagreement only flags
