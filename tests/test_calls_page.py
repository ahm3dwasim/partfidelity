"""Calls-page tests: read-back first, labelled fallback, honest stats."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import evidence
from app import db
from app.ledger import JobContextImpl, insert_call_record
from app.admission import job_dir
from app.contracts import CallRecord, JobId
from app.routes.calls import (
    calls_link_html,
    render_calls_page,
    resolve_job_context,
    router,
)


@dataclass
class FakeCtx:
    job_id: JobId
    evidence_store: object = None

    def record_call_with_event(self, key, record, event) -> None:
        """Explicit fake receipt operation over this context's store."""
        evidence.get_store(self).record_call_with_event(key, record, event)


@pytest.fixture(autouse=True)
def _clean_registry():
    evidence.reset_stores()
    yield
    evidence.reset_stores()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _record(ctx, **over) -> None:
    base = dict(
        job_id=ctx.job_id,
        stage="extract",
        model="spark-1.3",
        endpoint="/extract",
        latency_ms=14500,
        units_in=3,
        units_out=8,
        estimated_cost_usd=Decimal("0.0866"),
        status="succeeded",
        evidence={"run_index": 1},
    )
    base.update(over)
    evidence.record_call(ctx, CallRecord(**base))  # type: ignore[arg-type]


def _seed_readback_job(job_id="job-rb"):
    ctx = FakeCtx(JobId(job_id))
    _record(ctx, stage="extract", latency_ms=14500)
    _record(ctx, stage="locate", model="sam-3.1", latency_ms=4300,
            estimated_cost_usd=Decimal("0.0025"), evidence={"phrase": "fan"})
    _record(
        ctx,
        stage="readback",
        model="spark-1.3",
        endpoint="/readback",
        latency_ms=5400,
        units_in=1,
        units_out=1,
        estimated_cost_usd=Decimal("0.0038"),
        evidence={"phrase": "fan", "decision": "yes", "score": 0.66},
    )
    return ctx


def test_sample_readback_visible_in_two_clicks(client):
    """Click 1 opens the linked page; the read-back row is already there."""
    ctx = _seed_readback_job()
    link = calls_link_html(str(ctx.job_id))
    assert "How this guide was made" in link
    assert f"/calls/{ctx.job_id}" in link
    body = client.get(f"/calls/{ctx.job_id}").text  # click 1
    assert "Sample read-back" in body
    assert "fan" in body and ">yes<" in body
    assert "5400 ms" in body and "$0.0038" in body
    assert "Read-back answered yes" in body and "Checked" not in body
    # The natural row comes before the full call table, not after it.
    assert body.index("Sample read-back") < body.index("Model calls")
    assert body.index("Sample read-back") < body.index("<td>extract</td>")


def test_sample_readback_no_badge_is_not_checked(client):
    ctx = FakeCtx(JobId("job-no"))
    _record(
        ctx,
        stage="readback",
        latency_ms=6100,
        estimated_cost_usd=Decimal("0.0041"),
        evidence={"phrase": "battery", "decision": "no", "score": 0.52},
    )
    body = client.get("/calls/job-no").text
    assert ">no<" in body and "Read-back answered no" in body and "Not checked" not in body


def test_fallback_example_is_separately_labelled(client):
    ctx = FakeCtx(JobId("job-norb"))
    _record(ctx, stage="extract", latency_ms=14500)
    body = client.get("/calls/job-norb").text
    assert "No measured read-back for this run" in body
    assert "/measured-example" not in body
    # The fallback is never presented as this run's own result.
    assert "Checked<" not in body
    assert "Sample read-back</h2>" in body  # section exists, row absent


def test_calls_page_shows_run_count_median_worst(client):
    _seed_readback_job("job-stats")
    body = client.get("/calls/job-stats").text
    assert "3" in body and "observed runs" in body
    assert "5400 ms" in body  # median of 14500/4300/5400
    assert "14500 ms" in body  # worst
    assert "p95" not in body.lower()
    assert "$0.0929" in body  # 0.0866 + 0.0025 + 0.0038


def test_calls_page_rejects_bad_job_id(client):
    assert client.get("/calls/../x").status_code == 404
    assert client.get("/calls/" + "j" * 65).status_code == 404


def test_readback_from_another_job_never_leaks(client):
    _seed_readback_job("job-a")
    other = FakeCtx(JobId("job-b"))
    _record(other, stage="extract", latency_ms=14500)
    body = client.get("/calls/job-b").text
    assert "No measured read-back for this run" in body
    assert "5400 ms" not in body  # job A's read-back latency stays in job A


def test_parent_calls_page_includes_its_chunk_model_receipts(tmp_path):
    database = tmp_path / "calls.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    now = db.format_dt(db.utcnow())
    with db.connect(database) as conn:
        for job_id, anon in (
            ("parentjob", "owner"),
            ("childjob", "parentjob/chunk/0"),
            ("retryjob", "parentjob/extract/retry/0"),
            ("otherjob", "otherparent/chunk/0"),
            ("spoofjob", "parentjob/chunk/1"),
        ):
            conn.execute(
                "INSERT INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at,"
                " deadline_at, confirmed_at, completed_at, cost_cap_usd, readback_fired_n)"
                " VALUES (?, 'active', ?, ?, 'video', ?, ?, NULL, NULL, '0.50', 0)",
                (job_id, anon, anon if job_id in ("childjob", "retryjob", "otherjob") else "ip", now, now),
            )
    for job_id, latency in (("childjob", 901), ("retryjob", 902), ("otherjob", 999), ("spoofjob", 777)):
        insert_call_record(CallRecord(
            job_id=JobId(job_id), stage="extract", model="spark-1.3",
            endpoint="/extract", latency_ms=latency, units_in=1, units_out=1,
            estimated_cost_usd=Decimal("0.01"), status="succeeded",
            evidence={"run_index": 1},
        ), path=database)
    ctx = JobContextImpl(database, JobId("parentjob"), job_dir("parentjob"))
    page = render_calls_page("parentjob", ctx)
    assert "901 ms" in page
    assert "902 ms" in page
    assert "999 ms" not in page
    assert "777 ms" not in page
    assert "2</span><span class=\"cap\">observed runs" in page


def test_calls_page_rejects_context_for_another_job(client):
    ctx_a = _seed_readback_job("job-a")
    with pytest.raises(ValueError, match="does not belong"):
        render_calls_page("job-b", ctx_a)  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[resolve_job_context] = lambda: ctx_a
    resp = TestClient(app).get("/calls/job-b")
    assert resp.status_code == 404
    assert "fan" not in resp.text  # job A's read-back never renders under B


def test_mismatched_context_is_rejected_before_any_read():
    store = evidence.InMemoryEvidenceStore()

    def _boom(job_id):
        raise AssertionError("store must not be read on a mismatched page")

    store.calls_for = _boom  # type: ignore[method-assign]
    store.events_for = _boom  # type: ignore[method-assign]
    ctx = FakeCtx(JobId("job-a"), evidence_store=store)
    with pytest.raises(ValueError, match="does not belong"):
        render_calls_page("job-b", ctx)  # type: ignore[arg-type]


def test_calls_page_reads_attached_store_through_frozen_interface(client):
    store = evidence.InMemoryEvidenceStore()
    ctx = FakeCtx(JobId("job-attached"), evidence_store=store)
    _record(
        ctx,
        stage="readback",
        latency_ms=5400,
        estimated_cost_usd=Decimal("0.0038"),
        evidence={"phrase": "fan", "decision": "yes", "score": 0.66},
    )
    assert len(evidence.list_calls(ctx)) == 1  # type: ignore[arg-type]
    # The default registry-backed dependency cannot see the attached store.
    assert "No model calls recorded" in client.get("/calls/job-attached").text
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[resolve_job_context] = lambda: ctx
    body = TestClient(app).get("/calls/job-attached").text
    assert "fan" in body and ">yes<" in body
    assert "Read-back answered yes" in body


def _ctx_with_rows(job_id, rows):
    """Exact owned store whose reader returns the given rows for any job."""
    store = evidence.InMemoryEvidenceStore()
    store.calls_for = lambda _job: tuple(rows)  # type: ignore[method-assign]
    return FakeCtx(JobId(job_id), evidence_store=store)


def _attached_body(ctx):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[resolve_job_context] = lambda: ctx
    return TestClient(app).get(f"/calls/{ctx.job_id}").text


def test_calls_page_drops_foreign_attached_row():
    foreign = CallRecord(
        job_id=JobId("job-a"),
        stage="readback",
        model="spark-1.3",
        endpoint="/readback",
        latency_ms=5400,
        units_in=1,
        units_out=1,
        estimated_cost_usd=Decimal("0.0038"),
        status="succeeded",
        evidence={"phrase": "fan", "decision": "yes", "score": 0.66},
    )
    ctx = _ctx_with_rows("job-b", [foreign])
    body = _attached_body(ctx)
    assert "No model calls recorded for this job yet" in body
    assert "No measured read-back for this run" in body
    assert "Checked" not in body
    assert "5400 ms" not in body  # job A's latency stays in job A


def test_calls_page_drops_adapter_row_with_unknown_model():
    # Isolated: the adapter row is valid except for the model id, so this
    # proves the model allowlist — not the endpoint check — drops the row.
    row = CallRecord(
        job_id=JobId("job-b"),
        stage="readback",
        model="Jane-Doe",
        endpoint="/readback",
        latency_ms=5400,
        units_in=1,
        units_out=1,
        estimated_cost_usd=Decimal("0.0038"),
        status="succeeded",
        evidence={"phrase": "fan", "decision": "yes", "score": 0.66},
    )
    ctx = _ctx_with_rows("job-b", [row])
    body = _attached_body(ctx)
    assert "Jane-Doe" not in body
    assert "No model calls recorded for this job yet" in body
    assert "Checked" not in body


def test_second_check_line_says_why_it_was_not_available():
    from types import SimpleNamespace
    from app.routes import calls as page
    call = SimpleNamespace(model=page.RECEIPT_MODEL_SPARK_OPEN, stage=page.RECEIPT_STAGE_READBACK, status="failed",
                           evidence={"phrase": "RAM", "reason": "provider key not configured"})
    html = page._render_second_check([call])
    assert "not available: provider key not configured" in html


def test_second_check_page_hides_secret_like_reason_but_shows_allowlisted_one():
    from app.contracts import RECEIPT_MODEL_SPARK_OPEN, RECEIPT_STAGE_READBACK
    from app.evidence import UNAVAILABLE_REASONS
    ctx = FakeCtx(JobId("job-reason"), evidence_store=evidence.InMemoryEvidenceStore())
    good = sorted(UNAVAILABLE_REASONS)[0]
    for phrase, reason in (("fan", "sk-live-SECRET123"), ("RAM", good)):
        _record(ctx, stage=RECEIPT_STAGE_READBACK, model=RECEIPT_MODEL_SPARK_OPEN, status="failed",
                endpoint="/readback", evidence={"phrase": phrase, "reason": reason})
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[resolve_job_context] = lambda: ctx
    body = TestClient(app).get("/calls/job-reason").text
    assert "SECRET123" not in body
    assert f"not available: {good}" in body
