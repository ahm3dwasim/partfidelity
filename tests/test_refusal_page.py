"""A hazard-refused job's status page says why, in plain words."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import db
from app.admission import job_dir
from app.hazards import refusal_copy
from app.pipeline import REFUSAL_MARKER_FILENAME
from app.pipeline import STAGE_MARKER_FILENAME
from app.admission import admit_job
from app.routes.jobs import router
from app.settings import Settings


def _client(tmp_path, job_id, marker, state="failed"):
    settings = Settings(
        data_root=tmp_path, db_path=tmp_path / "t.sqlite3", production=False
    )
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    now = db.format_dt(db.utcnow())
    with db.connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO jobs (id, state, anon_id, ip_hash, video_hash, created_at,"
            " deadline_at, confirmed_at, completed_at, cost_cap_usd, readback_fired_n)"
            " VALUES (?, ?, 'a', 'i', 'v', ?, ?, NULL, NULL, '0.50', 0)",
            (job_id, state, now, now),
        )
    if marker is not None:
        folder = job_dir(job_id, root=tmp_path)
        folder.mkdir(parents=True)
        (folder / REFUSAL_MARKER_FILENAME).write_text(json.dumps({"kind": marker}))
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    return TestClient(app)


def _page(client, job_id):
    return client.get(f"/jobs/{job_id}", headers={"accept": "text/html"}).text


def test_refused_job_page_shows_reason(tmp_path):
    page = _page(_client(tmp_path, "jobscope1", "scope"), "jobscope1")
    assert "phones and laptops only" in page
    assert "Try the sample guide" in page
    assert "could not be built" not in page


def test_failed_job_without_marker_keeps_generic_copy(tmp_path):
    page = _page(_client(tmp_path, "jobplain1", None), "jobplain1")
    assert "could not be built" in page


def test_deadline_failure_writes_reason_shown_on_job_page(tmp_path):
    client = _client(tmp_path, "seedjob", None)
    ctx = admit_job("anon-deadline", "ip-deadline", "video-deadline")
    from app.pipeline import _fail

    result = _fail(ctx, ctx.job_id, "part check failed", replay=True,
                   refusal_kind="deadline")
    assert result.state == "failed"
    page = _page(client, str(ctx.job_id))
    assert "took longer than the time limit" in page
    assert "Try a shorter clip" in page
    assert "took longer than the time limit" in client.get(f"/jobs/{ctx.job_id}").json()["reason"]


def test_chunk_parent_stage_marker_overrides_old_upload_receipt(tmp_path):
    client = _client(tmp_path, "jobstage1", None, state="active")
    folder = job_dir("jobstage1", root=tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / STAGE_MARKER_FILENAME
    for stage in ("extract", "segment", "readback"):
        marker.write_text(stage)
        page = _page(client, "jobstage1")
        assert f"Current stage: {stage}" in page
        assert client.get("/jobs/jobstage1").json()["stage"] == stage


def test_refusal_copy_kinds():
    assert refusal_copy("none") is None
    assert "teardown" in refusal_copy("not_a_teardown")


def test_pending_settlement_page_still_shows_reason(tmp_path):
    client = _client(tmp_path, "jobpend01", "scope")
    pending = job_dir("jobpend01", root=tmp_path) / "pending_settlement.json"
    pending.write_text(json.dumps({"job_id": "jobpend01"}))
    page = _page(client, "jobpend01")
    assert "phones and laptops only" in page


def test_refusal_copy_covers_every_user_visible_kind():
    for kind in (
        "scope", "not_a_teardown", "duration", "budget", "deadline",
        "part_check", "extraction", "upload_failed", "processing", "frames",
        "readback", "finalize", "settlement", "cancelled",
        "service_unavailable", "failure",
    ):
        assert refusal_copy(kind), kind
