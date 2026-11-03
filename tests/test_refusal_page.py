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
        (folder / REFUSAL_MARKER_FILENAME).write_text(json.dumps(marker if isinstance(marker, dict) else {"kind": marker}))
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
    assert "Trim it to the repair itself" in page
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
    assert "repair" in refusal_copy("not_a_teardown")


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


def test_failed_job_without_marker_still_gets_a_reason(tmp_path):
    client = _client(tmp_path, "jobnull01", None)
    body = client.get("/jobs/jobnull01").json()
    assert body["state"] == "failed"
    assert body["reason"] == refusal_copy("failure")
    assert refusal_copy("failure") in _page(client, "jobnull01")


def test_scope_stop_names_the_rule_the_clip_hit():
    from app.hazards import SCOPE_LEAD, SCOPE_RULE_TEXT, assess_hazards
    from test_hazards import make_step
    for rule, text in (("hazard:mains", SCOPE_RULE_TEXT["hazard"]), ("device_type:tablet", SCOPE_RULE_TEXT["device"]),
                       ("device_guess:unlisted", SCOPE_RULE_TEXT["device"])):
        copy = refusal_copy("scope", rule)
        assert copy.startswith(text) and SCOPE_LEAD in copy
    assert refusal_copy("scope").startswith(SCOPE_LEAD)  # old marker: no rule recorded
    cases = [("ThinkPad T480", make_step(1, hazards=("mains",)), "hazard:mains"),
             ("ThinkPad T480", make_step(1, "The cell looks swollen."), "hazard:swollen"),
             ("e-bike", make_step(1), "hazard:e-bike"),
             ("phone", make_step(1, "It was on fire yesterday."), "hazard:fire"),
             ("tablet", make_step(1), "device_type:tablet"),
             ("game controller", make_step(1), "device_guess:unlisted")]
    for guess, step, rule in cases:
        got = assess_hazards((step,), guess).refusal_rule
        assert got == rule, (guess, got)


def test_refused_job_page_names_recorded_rule(tmp_path):
    text = _page(_client(tmp_path, "jobrule01", {"kind": "scope", "rule": "device_type:tablet"}), "jobrule01")
    assert "The video did not look like a phone or laptop repair." in text


def test_pipeline_persists_refusal_rule_on_marker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app import pipeline
    client = _client(tmp_path, "jobwire01", None)
    monkeypatch.setattr(pipeline, "_settle_failed", lambda *a, **k: "failed")
    ctx = SimpleNamespace(job_id="jobwire01")
    monkeypatch.setattr(pipeline, "job_dir", lambda job_id, **k: job_dir(job_id, root=tmp_path))
    pipeline._refuse(ctx, None, "not a teardown", replay=False, refusal_kind="scope", refusal_rule="hazard:mains")
    marker = job_dir("jobwire01", root=tmp_path) / REFUSAL_MARKER_FILENAME
    assert json.loads(marker.read_text()) == {"kind": "scope", "rule": "hazard:mains"}


def test_refusal_branch_passes_assessment_rule_to_refuse():
    import ast
    import inspect
    from app import pipeline
    tree = ast.parse(inspect.getsource(pipeline._run_stages_inner))
    wired = [
        kw for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_refuse"
        for kw in node.keywords
        if kw.arg == "refusal_rule" and ast.unparse(kw.value) == "assessment.refusal_rule"
    ]
    assert wired, "hazard refusal branch no longer passes assessment.refusal_rule to _refuse"
