"""Evidence-to-ledger receipt wiring: record_call persists through the real JobContextImpl."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import db, evidence
from app.admission import admit_job
from app.contracts import CallRecord, JobId
from app.routes.calls import resolve_job_context, router


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


def _record(job_id: JobId, **over) -> CallRecord:
    base = dict(
        job_id=job_id,
        stage="extract",
        model="spark-1.3",
        endpoint="/runs",
        latency_ms=1200,
        units_in=10,
        units_out=4,
        estimated_cost_usd=Decimal("0.0866"),
        status="succeeded",
        evidence={"run_index": 0},
    )
    base.update(over)
    return CallRecord(**base)


def _counts(database: Path) -> tuple[int, int]:
    conn = db.connect(database)
    try:
        calls = conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()
    return calls, events


def _pair(database: Path) -> tuple[sqlite3.Row, sqlite3.Row]:
    conn = db.connect(database)
    try:
        call = conn.execute("SELECT * FROM model_calls").fetchone()
        event = conn.execute("SELECT * FROM events").fetchone()
    finally:
        conn.close()
    assert call is not None and event is not None
    return call, event


def test_record_call_writes_one_shared_key_pair_and_replay_is_noop(
    lane_db: Path,
) -> None:
    ctx = admit_job("anon-e2e-1", "ip-e2e-1", "video-e2e-1")
    record = _record(ctx.job_id)
    evidence.record_call(ctx, record)
    assert _counts(lane_db) == (1, 1)
    call, event = _pair(lane_db)
    key = evidence.call_key(evidence._validated_call_record(record))
    assert call["id"] == event["id"] == key
    assert call["job_id"] == event["job_id"] == str(ctx.job_id)
    assert event["event"] == "model.call_recorded"
    assert json.loads(event["detail_json"]) == {
        "stage": "extract",
        "status": "succeeded",
    }
    evidence.record_call(ctx, record)
    assert _counts(lane_db) == (1, 1)


def test_conflicting_payload_for_the_key_raises(lane_db: Path) -> None:
    ctx = admit_job("anon-e2e-2", "ip-e2e-2", "video-e2e-2")
    record = _record(ctx.job_id)
    evidence.record_call(ctx, record)
    key = evidence.call_key(evidence._validated_call_record(record))
    good_event = ("model.call_recorded", {"stage": "extract", "status": "succeeded"})
    with pytest.raises(ValueError, match="does not match"):
        ctx.record_call_with_event(
            key, _record(ctx.job_id, latency_ms=1201), good_event
        )
    with pytest.raises(ValueError, match="receipt key conflict"):
        ctx.record_call_with_event(
            key,
            record,
            ("model.call_failed", {"stage": "extract", "status": "failed"}),
        )
    assert _counts(lane_db) == (1, 1)


def test_real_context_write_is_readable_through_list_calls_and_page(
    lane_db: Path,
) -> None:
    ctx = admit_job("anon-e2e-r", "ip-e2e-r", "video-e2e-r")
    record = _record(ctx.job_id)
    evidence.record_call(ctx, record)
    (found,) = evidence.list_calls(ctx)
    assert found == evidence._validated_call_record(record)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[resolve_job_context] = lambda: ctx
    body = TestClient(app).get(f"/calls/{ctx.job_id}").text
    assert "<td>extract</td>" in body
    assert "1200 ms" in body
    assert "$0.0866" in body
    assert "No model calls recorded" not in body


def test_real_context_read_ignores_in_memory_registry(lane_db: Path) -> None:
    ctx = admit_job("anon-e2e-n", "ip-e2e-n", "video-e2e-n")
    evidence.record_call(ctx, _record(ctx.job_id))
    try:
        evidence.get_store(ctx).save_call(_record(ctx.job_id, latency_ms=9999))
        (found,) = evidence.list_calls(ctx)
    finally:
        evidence.reset_stores()
    assert found.latency_ms == 1200


def test_receipt_failure_rolls_back_both_rows(lane_db: Path) -> None:
    ctx = admit_job("anon-e2e-3", "ip-e2e-3", "video-e2e-3")
    record = _record(ctx.job_id)
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "CREATE TRIGGER fail_receipt_event BEFORE INSERT ON events"
            " BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        evidence.record_call(ctx, record)
    assert _counts(lane_db) == (0, 0)
    conn = db.connect(lane_db)
    try:
        conn.execute("DROP TRIGGER fail_receipt_event")
    finally:
        conn.close()
    evidence.record_call(ctx, record)
    assert _counts(lane_db) == (1, 1)
