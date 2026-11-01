"""Confirmed jobs start off the request and never fail silently.

The hosted root cause: production set only the pipeline mode (live by
default) while the provider mode defaulted to replay, so every live
client build refused and the confirm route rendered "Confirmed" forever.
These tests pin the fixed settings, the pre-upload client check, the
background runner, the honest failed job with its Meta copy deleted,
cancel during a run, and the boot-time fail of interrupted jobs.
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import db, deletion, guide_clip, pipeline, service
from app.admission import admit_job, job_dir
from app.deletion import sweep_expired
from app.hazards import refusal_copy
from app.job_runner import JobRunner
from app.contracts import MetaFilesError, MetaUpload
from app.meta_files import MetaClientUnavailableError
from app.ledger import LifecycleRepositoryImpl
from app.main import create_app
from app.pipeline import Guide, RecordingMetaClient, ReplayMetaFilesClient, guide_path, save_guide
from app.routes import uploads as uploads_routes
from app.settings import ProviderSettings, Settings, load_settings
from app.uploads import (
    confirm_and_send, hash_file,
    prepare_preview, send_claimed_confirmation,
)
from app.uploads_store import UploadStoreImpl

# Distinctive fake key: shaped nothing like a real one.
FAKE_KEY = "live-start-test-key-0a1b2c3d"
RELEASE_TIMEOUT_S = 120


def test_deferred_finish_forwards_failed_output_names():
    inner = Mock()
    proxy = pipeline._DeferredFinish(inner)

    proxy.finish("failed", failed_output_names=("guide.json", "video.mp4"))

    inner.finish.assert_called_once_with(
        "failed", keep_artifact_ids=(), failed_output_names=("guide.json", "video.mp4")
    )


def test_deferred_finish_preserves_keyword_until_commit():
    inner = Mock()
    proxy = pipeline._DeferredFinish(inner)
    proxy.finish("done", failed_output_names=("guide.json", "video.mp4"))
    inner.finish.assert_not_called()

    proxy.commit_done()

    inner.finish.assert_called_once_with(
        "done", keep_artifact_ids=(), failed_output_names=("guide.json", "video.mp4")
    )


@pytest.fixture(autouse=True)
def _clean():
    service.clear_client_factory()
    yield
    service.clear_client_factory()
    uploads_routes.clear_test_overrides()


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "six.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=6",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return path


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        data_root=tmp_path / "data", db_path=tmp_path / "lane.sqlite3",
        production=False,
    )
    values.update(overrides)
    return Settings(**values)


def _job_state(settings: Settings, job_id: str) -> str:
    conn = db.connect(settings.db_path)
    try:
        return conn.execute(
            "SELECT state FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()["state"]
    finally:
        conn.close()


def _upload(client: TestClient, clip: Path) -> str:
    posted = client.post(
        "/upload",
        files={"clip": ("clip.mp4", clip.read_bytes(), "video/mp4")},
        follow_redirects=False,
    )
    assert posted.status_code == 303
    return posted.headers["location"].split("/preview/")[1]


def _confirm(client: TestClient, job_id: str):
    return client.post(
        f"/preview/{job_id}/confirm",
        data={"consent": "on", "ownership": "on"},
        follow_redirects=False,
    )


def _html(client: TestClient, job_id: str):
    return client.get(
        f"/jobs/{job_id}", headers={"accept": "text/html"},
        follow_redirects=False,
    )


# --- the root cause -------------------------------------------------------


def test_hosted_production_env_builds_live_clients(tmp_path: Path) -> None:
    """The Modal env (no provider mode set) now builds a live client set."""
    env = {
        "PARTFIDELITY_PRODUCTION": "1",
        "META_API_KEY": FAKE_KEY,
        "META_FILES_BASE_URL": "https://files.example.invalid/v1",
        "PARTFIDELITY_DATA_ROOT": str(tmp_path / "data"),
        "PARTFIDELITY_DB_PATH": str(tmp_path / "lane.sqlite3"),
    }
    settings = load_settings(env)
    assert settings.pipeline_mode == "live"
    assert settings.providers.mode == "live"
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    ctx = admit_job("anon-host", "ip-host", "video-host")
    clients = service.resolve_clients(settings, ctx)
    assert str(clients.spark.bound_job_id) == str(ctx.job_id)
    assert FAKE_KEY not in repr(clients)
    # Replay stays selectable, and an explicit mismatch refuses at boot.
    replay = load_settings({**env, "PARTFIDELITY_PIPELINE_MODE": "replay"})
    assert replay.providers.mode == "replay"
    with pytest.raises(RuntimeError, match="no live job could start"):
        load_settings({**env, "PARTFIDELITY_PROVIDER_MODE": "replay"})


def test_concurrency_and_refresh_settings_load_loudly(tmp_path: Path) -> None:
    assert load_settings({}).max_concurrent_jobs == 1
    tuned = load_settings({
        "PARTFIDELITY_MAX_CONCURRENT_JOBS": "2",
        "PARTFIDELITY_JOB_STATUS_REFRESH_S": "9",
    })
    assert (tuned.max_concurrent_jobs, tuned.job_status_refresh_s) == (2, 9)
    with pytest.raises(RuntimeError, match="PARTFIDELITY_MAX_CONCURRENT_JOBS"):
        load_settings({"PARTFIDELITY_MAX_CONCURRENT_JOBS": "0"})
    with pytest.raises(ValueError, match="max_concurrent_jobs"):
        _settings(tmp_path, max_concurrent_jobs=0)


# --- never silent ---------------------------------------------------------


def test_unbuildable_clients_refuse_before_any_upload(
    tmp_path: Path, clip: Path, caplog
) -> None:
    settings = _settings(tmp_path, providers=ProviderSettings(mode="live"))
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    with caplog.at_level(logging.WARNING):
        refused = _confirm(client, job_id)
    assert refused.status_code == 503
    assert meta.by_key == {}  # nothing reached Meta
    assert "META_API_KEY" in caplog.text and "refused before upload" in caplog.text
    assert FAKE_KEY not in caplog.text
    record = UploadStoreImpl(settings.db_path).load_job(job_id)[1]
    assert record.state == "preview" and record.meta_file_id is None


def test_pipeline_unavailable_after_upload_fails_job_and_deletes_meta(
    tmp_path: Path, clip: Path, caplog
) -> None:
    settings = _settings(tmp_path, pipeline_mode="live")
    calls: list[int] = []

    def _factory(_settings: Settings):
        calls.append(1)
        if len(calls) == 1:  # the pre-upload check passes
            return service.default_replay_clients()
        raise service.PipelineUnavailable("live mode cannot build: test cause")

    service.set_client_factory(_factory)
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    with caplog.at_level(logging.INFO):
        confirmed = _confirm(client, job_id)
        assert confirmed.status_code == 303
        assert confirmed.headers["location"] == f"/jobs/{job_id}"
        client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert len(calls) == 2
    assert _job_state(settings, job_id) == "failed"
    assert "cannot start: PipelineUnavailable" in caplog.text
    file_id = UploadStoreImpl(settings.db_path).load_job(job_id)[1].meta_file_id
    assert file_id is not None and meta.delete_calls == [str(file_id)]
    assert "verified_absent" in caplog.text
    status = client.get(f"/jobs/{job_id}").json()
    assert status["state"] == "failed" and status["guide_url"] is None
    page = _html(client, job_id)
    assert page.status_code == 200
    assert "The guide service was unavailable before processing started" in page.text
    assert "http-equiv='refresh'" not in page.text


def test_unexpected_provider_error_is_sanitized_and_settled(
    tmp_path: Path, clip: Path, caplog
) -> None:
    settings = _settings(tmp_path, pipeline_mode="live")
    calls = 0

    def _factory(_settings: Settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            return service.default_replay_clients()
        raise RuntimeError(f"secret path /private/{FAKE_KEY}")

    service.set_client_factory(_factory)
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    with caplog.at_level(logging.WARNING):
        assert _confirm(client, job_id).status_code == 303
        client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert _job_state(settings, job_id) == "failed"
    assert meta.delete_calls
    assert "RuntimeError" in caplog.text
    assert FAKE_KEY not in caplog.text


def test_submit_after_shutdown_fails_and_deletes_meta(tmp_path: Path, clip: Path) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    client.app.state.job_runner.shutdown()
    confirmed = _confirm(client, job_id)
    assert confirmed.status_code == 303
    assert _job_state(settings, job_id) == "failed"
    assert meta.upload_calls == []
    assert "The job could not start" in _html(client, job_id).text
    assert client.get(f"/jobs/{job_id}").json()["guide_url"] is None


def test_confirm_redirects_during_meta_upload_and_replay_never_uploads_twice(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    meta = ReplayMetaFilesClient()
    real_upload = meta.upload_file

    def held_upload(*args, **kwargs):
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)
        return real_upload(*args, **kwargs)

    monkeypatch.setattr(meta, "upload_file", held_upload)
    settings = _settings(tmp_path, pipeline_mode="replay")
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    try:
        before = time.monotonic()
        first = _confirm(client, job_id)
        elapsed = time.monotonic() - before
        assert first.status_code == 303
        assert elapsed < 1.0
        assert first.headers["location"] == f"/jobs/{job_id}"
        assert started.wait(RELEASE_TIMEOUT_S)
        assert len(meta.upload_calls) == 0
        assert "Sending your video" in _html(client, job_id).text
        assert _confirm(client, job_id).status_code in (303, 409)
    finally:
        release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert len(meta.upload_calls) == 1
    assert _confirm(client, job_id).status_code in (200, 303)
    assert len(meta.upload_calls) == 1


def test_cancel_queued_upload_never_reaches_meta(tmp_path: Path, clip: Path) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay", max_concurrent_jobs=1)
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    started = threading.Event()
    release = threading.Event()

    def blocker():
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)

    assert client.app.state.job_runner.submit("blocker", blocker)
    assert started.wait(RELEASE_TIMEOUT_S)
    job_id = _upload(client, clip)
    try:
        assert _confirm(client, job_id).status_code == 303
        assert _confirm(client, job_id).status_code in (303, 409)
        assert client.post(f"/preview/{job_id}/cancel").status_code in (200, 202)
    finally:
        release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert meta.upload_calls == []
    assert _job_state(settings, job_id) == "failed"
    assert "cancelled" in _html(client, job_id).text.lower()


def test_meta_client_unavailable_refuses_on_post_before_claim(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)

    def unavailable(_request):
        raise MetaClientUnavailableError("unavailable")

    monkeypatch.setattr(uploads_routes, "_meta_client", unavailable)
    response = _confirm(client, job_id)
    assert response.status_code == 503
    record = UploadStoreImpl(settings.db_path).load_job(job_id)[1]
    assert record.state == "preview" and record.claim_id is None


def test_standalone_router_still_confirms_on_request(
    tmp_path: Path, clip: Path
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    ctx = admit_job("anon-standalone", "ip-standalone", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(uploads=uploads, meta_client=meta)
    app = FastAPI()
    app.include_router(uploads_routes.router)
    response = _confirm(TestClient(app), str(ctx.job_id))
    assert response.status_code == 200
    assert len(meta.upload_calls) == 1
    assert uploads.load_job(ctx.job_id)[1].state == "confirmed"


def test_cancel_during_meta_upload_never_starts_pipeline(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    started = threading.Event()
    release = threading.Event()
    real_upload = meta.upload_file
    ran = []

    def held_upload(*args, **kwargs):
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)
        return real_upload(*args, **kwargs)

    monkeypatch.setattr(meta, "upload_file", held_upload)
    monkeypatch.setattr(service, "run_confirmed_upload", lambda *a, **k: ran.append(1))
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    try:
        assert _confirm(client, job_id).status_code == 303
        assert started.wait(RELEASE_TIMEOUT_S)
        assert client.post(f"/preview/{job_id}/cancel").status_code in (200, 202)
    finally:
        release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert not ran
    assert _job_state(settings, job_id) == "failed"
    assert "cancelled" in _html(client, job_id).text.lower()


@pytest.mark.parametrize("bad_expiry", [False, True])
def test_background_upload_failure_writes_reason_and_skips_pipeline(
    tmp_path: Path, clip: Path, monkeypatch, bad_expiry: bool
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    ran = []

    def broken_upload(*args, **kwargs):
        if bad_expiry:
            uploaded = ReplayMetaFilesClient.upload_file(meta, *args, **kwargs)
            return MetaUpload(uploaded.file_id, db.utcnow() - timedelta(seconds=1))
        raise MetaFilesError("upload failed")

    monkeypatch.setattr(meta, "upload_file", broken_upload)
    monkeypatch.setattr(service, "run_confirmed_upload", lambda *a, **k: ran.append(1))
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    assert _confirm(client, job_id).status_code == 303
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert _job_state(settings, job_id) == "failed"
    assert not ran
    assert "could not be sent" in _html(client, job_id).text


@pytest.mark.parametrize("in_flight", [False, True])
def test_restart_fails_unattached_claim_without_resending(
    tmp_path: Path, clip: Path, in_flight: bool
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    meta = ReplayMetaFilesClient()
    ctx = admit_job("anon-restart", "ip-restart", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    claim = uploads.claim_confirmation(ctx.job_id, db.utcnow())
    if in_flight:
        meta.upload_file(
            claim.record.preview.local_video, expires_after_hours=23,
            idempotency_key=claim.record.claim_id,
        )
    before = len(meta.upload_calls)
    assert service.fail_interrupted_jobs(settings, meta_client=meta) == 1
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    assert len(meta.upload_calls) == before
    assert claim.record.claim_id == uploads.load_job(ctx.job_id)[1].claim_id
    page = TestClient(create_app(settings)).get(
        f"/jobs/{ctx.job_id}", headers={"accept": "text/html"},
    )
    assert "restarted" in page.text
    assert service.fail_interrupted_jobs(settings, meta_client=meta) == 0


def test_create_app_settles_interrupted_claim_and_hides_preview(
    tmp_path: Path, clip: Path
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    ctx = admit_job("anon-factory-restart", "ip-factory-restart", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    claim = uploads.claim_confirmation(ctx.job_id, db.utcnow())

    app = create_app(settings)
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    after = uploads.load_job(ctx.job_id)[1]
    assert after.state == "failed" and after.hidden
    assert after.claim_id == claim.record.claim_id
    assert TestClient(app).get(f"/preview/{ctx.job_id}/video").status_code in (404, 410)


def test_create_app_deletes_attached_meta_after_restart(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    ctx = admit_job("anon-factory-meta", "ip-factory-meta", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    meta = ReplayMetaFilesClient()
    file_id = confirm_and_send(
        ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx)
    )
    monkeypatch.setattr(service, "resolve_clients", lambda *a, **k: Mock(meta=meta))

    create_app(settings)
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    assert meta.delete_calls == [str(file_id)]


def test_failure_settlement_hides_unattached_claim_and_keeps_first_reason(
    tmp_path: Path, clip: Path
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    uploads = UploadStoreImpl(settings.db_path)
    claim = uploads.claim_confirmation(job_id, db.utcnow())
    marker = job_dir(job_id, root=settings.data_root) / "refusal.json"
    marker.write_text('{"kind":"cancelled"}', encoding="utf-8")
    lifecycle = LifecycleRepositoryImpl(settings.db_path)

    service.fail_confirmed_upload(
        job_id, "background job could not start", settings,
        uploads=uploads, lifecycle=lifecycle, meta_client=None,
    )
    after = uploads.load_job(job_id)[1]
    assert after.state == "failed" and after.hidden
    assert after.claim_id == claim.record.claim_id
    assert marker.read_text(encoding="utf-8") == '{"kind":"cancelled"}'
    assert client.get(f"/preview/{job_id}/video").status_code in (404, 410)


def test_restart_repairs_crash_after_job_failure_before_upload_failure(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    uploads = UploadStoreImpl(settings.db_path)
    claim = uploads.claim_confirmation(job_id, db.utcnow())
    lifecycle = LifecycleRepositoryImpl(settings.db_path)
    real_record_failure = uploads.record_failure

    def crash(*args, **kwargs):
        raise SystemExit("crash between writes")

    monkeypatch.setattr(uploads, "record_failure", crash)
    with pytest.raises(SystemExit, match="crash between writes"):
        service.fail_confirmed_upload(
            job_id, "background job could not start", settings,
            uploads=uploads, lifecycle=lifecycle, meta_client=None,
        )
    monkeypatch.setattr(uploads, "record_failure", real_record_failure)
    assert _job_state(settings, job_id) == "failed"
    assert uploads.load_job(job_id)[1].state == "confirming"
    marker = job_dir(job_id, root=settings.data_root) / "refusal.json"
    first_reason = marker.read_text(encoding="utf-8")

    app = create_app(settings)
    after = uploads.load_job(job_id)[1]
    assert after.state == "failed" and after.hidden
    assert after.claim_id == claim.record.claim_id
    assert marker.read_text(encoding="utf-8") == first_reason
    assert TestClient(app).get(f"/preview/{job_id}/video").status_code in (404, 410)


@pytest.mark.parametrize("closed_cancel", [False, True])
@pytest.mark.parametrize("cancel_before_failure", [False, True])
def test_recovery_cancel_copy_preserves_first_reason(
    tmp_path: Path, clip: Path, monkeypatch, closed_cancel: bool,
    cancel_before_failure: bool,
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    uploads = UploadStoreImpl(settings.db_path)
    claim = uploads.claim_confirmation(job_id, db.utcnow())
    real_fail_job = service.fail_job

    def cancel_around_failure(*args, **kwargs):
        if cancel_before_failure:
            uploads.request_cancellation(job_id, db.utcnow())
        result = real_fail_job(*args, **kwargs)
        if not cancel_before_failure:
            uploads.request_cancellation(job_id, db.utcnow())
        uploads.recover_job(
            job_id,
            claim.record.guard_deadline_utc + timedelta(seconds=1),
            no_match_claim_id=claim.record.claim_id,
        )
        if closed_cancel:
            # The deletion sweeper may close cancellation before recovery resumes.
            with db.connect(settings.db_path) as conn:
                conn.execute(
                    "UPDATE uploads SET state = 'cancelled' WHERE job_id = ?",
                    (job_id,),
                )
        return result

    monkeypatch.setattr(service, "fail_job", cancel_around_failure)
    assert service.fail_interrupted_jobs(settings, meta_client=None) == 1
    # The cancel queued the preview; the upload closes only once it is deleted.
    assert uploads.load_job(job_id)[1].state == ("cancelled" if closed_cancel else "cancelling")
    if cancel_before_failure and not closed_cancel:
        sweep_expired(db.utcnow(), uploads=uploads,
                      lifecycle=LifecycleRepositoryImpl(settings.db_path), client=None)
        assert uploads.load_job(job_id)[1].state == "cancelled"
    assert refusal_copy("cancelled" if cancel_before_failure else "interrupted") in client.get(
        f"/jobs/{job_id}", headers={"accept": "text/html"}
    ).text


def test_one_bad_recovery_row_does_not_abort_startup(
    tmp_path: Path, clip: Path, monkeypatch, caplog
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    client = TestClient(create_app(settings))
    first, second = (_upload(client, clip) for _ in range(2))
    uploads = UploadStoreImpl(settings.db_path)
    uploads.claim_confirmation(first, db.utcnow())
    uploads.claim_confirmation(second, db.utcnow())
    bad = min(first, second)
    good = max(first, second)
    real_fail = service._fail_and_drop_meta

    def fail_one(job_id, *args, **kwargs):
        if str(job_id) == bad:
            raise RuntimeError("bad row")
        return real_fail(job_id, *args, **kwargs)

    monkeypatch.setattr(service, "_fail_and_drop_meta", fail_one)
    create_app(settings)
    assert uploads.load_job(good)[1].state == "failed"
    assert uploads.load_job(bad)[1].state == "confirming"
    assert f"job={bad} restart recovery failed: RuntimeError" in caplog.text
    monkeypatch.setattr(service, "_fail_and_drop_meta", real_fail)
    create_app(settings)
    assert uploads.load_job(bad)[1].state == "failed"


def test_expired_parent_refuses_meta_before_upload(tmp_path: Path, clip: Path) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    ctx = admit_job("anon-expired-send", "ip-expired-send", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    claim = uploads.claim_confirmation(ctx.job_id, db.utcnow())
    with db.connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET deadline_at = ? WHERE id = ?",
            (db.format_dt(db.utcnow() - timedelta(seconds=1)), str(ctx.job_id)),
        )
    meta = ReplayMetaFilesClient()
    with pytest.raises(Exception, match="deadline"):
        send_claimed_confirmation(
            ctx.job_id, claim.record.claim_id, uploads=uploads,
            client=RecordingMetaClient(meta, ctx),
        )
    assert meta.upload_calls == []


def test_queued_claim_past_parent_deadline_fails_without_meta(
    tmp_path: Path, clip: Path
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay", max_concurrent_jobs=1)
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    started = threading.Event()
    release = threading.Event()

    def blocker():
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)

    assert client.app.state.job_runner.submit("blocker", blocker)
    assert started.wait(RELEASE_TIMEOUT_S)
    job_id = _upload(client, clip)
    try:
        assert _confirm(client, job_id).status_code == 303
        with db.connect(settings.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET deadline_at = ? WHERE id = ?",
                (db.format_dt(db.utcnow() - timedelta(seconds=1)), job_id),
            )
    finally:
        release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert meta.upload_calls == []
    assert _job_state(settings, job_id) == "failed"
    after = UploadStoreImpl(settings.db_path).load_job(job_id)[1]
    assert after.state == "failed" and after.hidden and after.claim_id is not None
    assert "time limit" in _html(client, job_id).text


def test_parent_expiring_after_meta_skips_pipeline_and_writes_deadline_reason(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    real_send = uploads_routes.send_claimed_confirmation
    ran = []

    def expire_after_send(*args, **kwargs):
        file_id = real_send(*args, **kwargs)
        with db.connect(settings.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET deadline_at = ? WHERE id = ?",
                (db.format_dt(db.utcnow() - timedelta(seconds=1)), job_id),
            )
        return file_id

    monkeypatch.setattr(uploads_routes, "send_claimed_confirmation", expire_after_send)
    monkeypatch.setattr(service, "run_confirmed_upload", lambda *a, **k: ran.append(1))
    assert _confirm(client, job_id).status_code == 303
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert len(meta.upload_calls) == 1
    assert not ran
    assert _job_state(settings, job_id) == "failed"
    assert "time limit" in _html(client, job_id).text


def test_second_factory_does_not_fail_another_workers_live_claim(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    started = threading.Event()
    release = threading.Event()
    real_upload = meta.upload_file

    def held_upload(*args, **kwargs):
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)
        return real_upload(*args, **kwargs)

    monkeypatch.setattr(meta, "upload_file", held_upload)
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    try:
        assert _confirm(client, job_id).status_code == 303
        assert started.wait(RELEASE_TIMEOUT_S)
        create_app(settings)
        assert _job_state(settings, job_id) == "active"
    finally:
        release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)


def test_expired_attachment_never_starts_extraction(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    lifecycle = LifecycleRepositoryImpl(settings.db_path)
    meta = ReplayMetaFilesClient()
    ctx = admit_job("anon-expired-meta", "ip-expired-meta", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    file_id = confirm_and_send(
        ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx),
    )
    with db.connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE meta_files SET expires_at = ? WHERE id = ?",
            (db.format_dt(db.utcnow() - timedelta(seconds=1)), str(file_id)),
        )
    ran = []
    monkeypatch.setattr(pipeline, "run_stages", lambda *a, **k: ran.append(1))
    result = pipeline.run_confirmed_job(
        ctx.job_id, clients=service.default_replay_clients(),
        uploads=uploads, lifecycle=lifecycle, replay=True, settings=settings,
    )
    assert result.state == "failed" and not ran
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    page = TestClient(create_app(settings)).get(
        f"/jobs/{ctx.job_id}", headers={"accept": "text/html"},
    )
    assert "expired before processing" in page.text


def test_shutdown_rejection_settles_confirmed_waiting_job(tmp_path: Path, clip: Path) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    uploads = UploadStoreImpl(settings.db_path)
    ctx, _ = uploads.load_job(job_id)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    # Reproduce a confirmed runner-queued row after both admission slots fill.
    first = admit_job("anon-first", "ip-first", "video-first")
    second = admit_job("anon-second", "ip-second", "video-second")
    with db.connect(settings.db_path) as conn:
        conn.execute("UPDATE jobs SET state = 'waiting' WHERE id = ?", (job_id,))
        conn.execute("UPDATE jobs SET state = 'active' WHERE id = ?", (str(second.job_id),))
    assert _job_state(settings, job_id) == "waiting"
    assert _job_state(settings, str(first.job_id)) == "active"
    assert _job_state(settings, str(second.job_id)) == "active"
    client.app.state.job_runner.shutdown()
    response = uploads_routes._start_confirmed_job(client, uploads, job_id, uploads.load_job(ctx.job_id)[1], meta)
    assert response.status_code == 303
    assert _job_state(settings, job_id) == "failed"
    assert meta.delete_calls == [str(file_id)]


@pytest.mark.parametrize("cleanup_error", [
    sqlite3.OperationalError("database is locked"),
    OSError("filesystem unavailable"),
])
def test_cleanup_error_keeps_meta_deletion_and_output_retry(
    tmp_path: Path, clip: Path, monkeypatch, cleanup_error: Exception
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    lifecycle = LifecycleRepositoryImpl(settings.db_path)
    meta = ReplayMetaFilesClient()
    ctx = admit_job("anon-clean", "ip-clean", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    guide_file = save_guide(
        Guide(job_id=ctx.job_id, steps=(), frame_paths=(), labels=(),
              difficulty="Medium", verdict="draft", draft_reasons=()),
        data_root=settings.data_root, guide_filename=settings.guide_filename,
    )
    frame = guide_file.parent / "frames" / "frame-000.png"
    frame.parent.mkdir()
    frame.write_bytes(b"png")
    cleanup_calls = []

    def _busy(*args, **kwargs):
        assert _job_state(settings, str(ctx.job_id)) == "failed"
        cleanup_calls.append(args)
        raise cleanup_error
    # The real cleanup runs once; its own queue transaction fails.
    monkeypatch.setattr(deletion, "queue_guide_images", _busy)
    result = service.fail_confirmed_upload(
        ctx.job_id, "start rejected", settings, uploads=uploads,
        lifecycle=lifecycle, meta_client=meta,
    )
    assert result.state == "failed" and result.error == "start rejected"
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    assert len(cleanup_calls) == 1
    assert meta.delete_calls == [str(file_id)]
    assert guide_file.exists() and frame.exists()
    due = {str(row.id) for row in lifecycle.list_due_artifacts(db.utcnow())}
    assert f"cancel_guide:{ctx.job_id}" in due
    assert f"cancel_frame:{ctx.job_id}:frames/frame-000.png" in due
    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=meta)
    assert not guide_file.exists() and not frame.exists()


def test_exception_after_done_keeps_committed_guide(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    meta = ReplayMetaFilesClient()
    ctx = admit_job("anon-done", "ip-done", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    guide_file = save_guide(
        Guide(job_id=ctx.job_id, steps=(), frame_paths=(), labels=(),
              difficulty="Medium", verdict="draft", draft_reasons=()),
        data_root=settings.data_root, guide_filename=settings.guide_filename,
    )
    ctx.finish("done")

    def _boom(*args, **kwargs):
        raise RuntimeError("post-commit snapshot failed")

    monkeypatch.setattr(service, "run_upload_job", _boom)
    result = service.run_confirmed_upload(
        ctx.job_id, settings, uploads=uploads,
        lifecycle=LifecycleRepositoryImpl(settings.db_path), meta_client=meta,
    )
    assert result.state == "done" and guide_file.exists()
    assert meta.delete_calls == []


# --- off the request ------------------------------------------------------


class _Gate:
    """Hold the real background run until the test releases it."""

    def __init__(self, monkeypatch) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        real = service.run_upload_job

        def _held(*args, **kwargs):
            self.started.set()
            assert self.release.wait(RELEASE_TIMEOUT_S)
            return real(*args, **kwargs)

        monkeypatch.setattr(service, "run_upload_job", _held)


def test_confirm_returns_at_once_and_job_finishes_in_background(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    gate = _Gate(monkeypatch)
    settings = _settings(tmp_path, pipeline_mode="replay")
    uploads_routes.set_test_overrides(meta_client=ReplayMetaFilesClient())
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    confirmed = _confirm(client, job_id)
    # The response came back while the run is still held.
    assert confirmed.status_code == 303
    assert confirmed.headers["location"] == f"/jobs/{job_id}"
    assert gate.started.wait(RELEASE_TIMEOUT_S)
    assert _job_state(settings, job_id) == "active"
    running = _html(client, job_id)
    assert running.status_code == 200
    assert "http-equiv='refresh'" in running.text
    assert f"/preview/{job_id}/cancel" in running.text
    assert client.get(f"/jobs/{job_id}").json()["state"] == "active"
    gate.release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert _job_state(settings, job_id) == "done"
    done = _html(client, job_id)
    assert done.status_code == 303
    assert done.headers["location"] == f"/guides/{job_id}"
    assert client.get(f"/guides/{job_id}").status_code == 200


def test_cancel_while_job_runs(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    gate = _Gate(monkeypatch)
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    assert _confirm(client, job_id).status_code == 303
    assert gate.started.wait(RELEASE_TIMEOUT_S)
    cancelled = client.post(f"/preview/{job_id}/cancel")
    assert cancelled.status_code == 200
    record = UploadStoreImpl(settings.db_path).load_job(job_id)[1]
    assert record.state == "cancelled" and record.hidden
    gate.release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert _job_state(settings, job_id) == "failed"
    assert client.get(f"/jobs/{job_id}").json()["guide_url"] is None
    assert meta.delete_calls  # the cancel path deleted the Meta copy


@pytest.mark.parametrize("stop", ["cancel", "expire"])
def test_terminal_during_assembly_never_publishes(
    tmp_path: Path, clip: Path, monkeypatch, stop: str
) -> None:
    started = threading.Event()
    release = threading.Event()
    real = pipeline.assemble_guide

    def _held(*args, **kwargs):
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "assemble_guide", _held)
    # Isolate guide publication; the clip path has its own guarded copy.
    monkeypatch.setattr(guide_clip, "retain_guide_clip", lambda *args, **kwargs: True)
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    job_id = _upload(client, clip)
    assert _confirm(client, job_id).status_code == 303
    try:
        assert started.wait(RELEASE_TIMEOUT_S)
        if stop == "cancel":
            assert client.post(f"/preview/{job_id}/cancel").status_code in (200, 202)
            expected = "failed"
        else:
            now = db.utcnow() + timedelta(days=1)
            sweep = sweep_expired(
                now, uploads=UploadStoreImpl(settings.db_path),
                lifecycle=LifecycleRepositoryImpl(settings.db_path), client=meta,
            )
            assert sweep.expired_jobs >= 1
            expected = "expired"
    finally:
        release.set()
    client.app.state.job_runner.wait(job_id, timeout=RELEASE_TIMEOUT_S)
    assert _job_state(settings, job_id) == expected
    assert not guide_path(job_id, data_root=settings.data_root,
                          guide_filename=settings.guide_filename).exists()
    assert not (job_dir(job_id, root=settings.data_root) / settings.guide_clip_filename).exists()
    assert not (job_dir(job_id, root=settings.data_root) / settings.corrections_filename).exists()
    assert client.get(f"/guides/{job_id}").status_code == 404
    assert client.get(f"/guides/{job_id}/clip").status_code == 404
    assert meta.delete_calls


def test_cancel_during_clip_copy_prevents_publication(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    meta = ReplayMetaFilesClient()
    uploads_routes.set_test_overrides(meta_client=meta)
    client = TestClient(create_app(settings))
    uploads = UploadStoreImpl(settings.db_path)
    ctx = admit_job("anon-clip-race", "ip-clip-race", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    clip_file = job_dir(str(ctx.job_id), root=settings.data_root) / settings.guide_clip_filename
    started = threading.Event()
    release = threading.Event()
    published: list[Path] = []
    errors: list[Exception] = []
    real_copy = guide_clip.shutil.copyfile
    real_replace = Path.replace

    def _held_copy(source, target):
        real_copy(source, target)
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)

    def _watch_replace(source, target):
        if Path(target) == clip_file:
            published.append(clip_file)
        return real_replace(source, target)

    monkeypatch.setattr(guide_clip.shutil, "copyfile", _held_copy)
    monkeypatch.setattr(Path, "replace", _watch_replace)

    def _retain():
        try:
            guide_clip.retain_guide_clip(ctx, uploads, filename=settings.guide_clip_filename)
        except guide_clip.GuideClipError as error:
            errors.append(error)

    thread = threading.Thread(target=_retain)
    thread.start()
    try:
        assert started.wait(RELEASE_TIMEOUT_S)
        assert client.post(f"/preview/{ctx.job_id}/cancel").status_code in (200, 202)
    finally:
        release.set()
    thread.join(timeout=RELEASE_TIMEOUT_S)
    assert not thread.is_alive() and errors
    assert not published and not clip_file.exists()
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    assert client.get(f"/guides/{ctx.job_id}/clip").status_code == 404
    assert meta.delete_calls


def test_runner_caps_concurrent_jobs() -> None:
    runner = JobRunner(1)
    first_in = threading.Event()
    first_out = threading.Event()
    order: list[str] = []

    def _first():
        first_in.set()
        assert first_out.wait(RELEASE_TIMEOUT_S)
        order.append("first")

    try:
        assert runner.submit("a", _first)
        assert first_in.wait(RELEASE_TIMEOUT_S)
        assert not runner.submit("a", lambda: None)  # same job never twice
        assert runner.submit("b", lambda: order.append("second"))
        assert order == []  # the second waits for the only slot
        first_out.set()
        runner.wait("a", timeout=RELEASE_TIMEOUT_S)
        runner.wait("b", timeout=RELEASE_TIMEOUT_S)
        assert order == ["first", "second"]
    finally:
        first_out.set()
        runner.shutdown()


def test_runner_shutdown_with_queued_job_returns() -> None:
    runner = JobRunner(1)
    started = threading.Event()
    release = threading.Event()
    stopped = threading.Event()

    def _running():
        started.set()
        assert release.wait(RELEASE_TIMEOUT_S)

    try:
        assert runner.submit("running", _running)
        assert started.wait(RELEASE_TIMEOUT_S)
        assert runner.submit("queued", lambda: None)
        thread = threading.Thread(target=lambda: (runner.shutdown(), stopped.set()))
        thread.start()
        assert stopped.wait(RELEASE_TIMEOUT_S)
        with pytest.raises(RuntimeError, match="closed"):
            runner.submit("rejected", lambda: None)
        thread.join(timeout=RELEASE_TIMEOUT_S)
    finally:
        release.set()
        runner.wait("running", timeout=RELEASE_TIMEOUT_S)


# --- restart --------------------------------------------------------------


def test_boot_fails_jobs_interrupted_mid_run(tmp_path: Path, clip: Path) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    meta = ReplayMetaFilesClient()
    ctx = admit_job("anon-boot", "ip-boot", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    file_id = confirm_and_send(
        ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx)
    )
    # An unconfirmed preview is not a running job and must survive boot.
    idle = admit_job("anon-idle", "ip-idle", "video-idle")
    with clip.open("rb") as handle:
        prepare_preview(idle, handle, clip.name, uploads=uploads)
    assert _job_state(settings, str(ctx.job_id)) == "active"
    guide_file = save_guide(
        Guide(job_id=ctx.job_id, steps=(), frame_paths=(), labels=(),
              difficulty="Medium", verdict="draft", draft_reasons=(),
              excluded_spans=()),
        data_root=settings.data_root, guide_filename=settings.guide_filename,
    )
    assert guide_file.exists()
    sidecar = guide_file.parent / settings.corrections_filename
    sidecar.write_text("{}", encoding="utf-8")
    assert guide_clip.retain_guide_clip(ctx, uploads, filename=settings.guide_clip_filename)
    clip_file = guide_file.parent / settings.guide_clip_filename
    assert clip_file.exists()
    assert service.fail_interrupted_jobs(settings, meta_client=meta) == 1
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    assert not guide_file.exists()
    assert not clip_file.exists() and not sidecar.exists()
    assert TestClient(create_app(settings)).get(f"/guides/{ctx.job_id}").status_code == 404
    assert _job_state(settings, str(idle.job_id)) == "active"
    assert meta.delete_calls == [str(file_id)]
    # Idempotent: a second boot finds nothing left to fail.
    assert service.fail_interrupted_jobs(settings, meta_client=meta) == 0


def test_boot_queues_meta_delete_for_confirmed_waiting_job_without_client(
    tmp_path: Path, clip: Path
) -> None:
    settings = _settings(tmp_path, pipeline_mode="replay")
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    uploads = UploadStoreImpl(settings.db_path)
    lifecycle = LifecycleRepositoryImpl(settings.db_path)
    meta = ReplayMetaFilesClient()
    ctx = admit_job("anon-wait-boot", "ip-wait-boot", hash_file(clip))
    with clip.open("rb") as handle:
        prepare_preview(ctx, handle, clip.name, uploads=uploads)
    file_id = confirm_and_send(ctx.job_id, uploads=uploads, client=RecordingMetaClient(meta, ctx))
    with db.connect(settings.db_path) as conn:
        conn.execute("UPDATE jobs SET state = 'waiting' WHERE id = ?", (str(ctx.job_id),))
    assert service.fail_interrupted_jobs(settings, meta_client=None) == 1
    assert _job_state(settings, str(ctx.job_id)) == "failed"
    assert meta.delete_calls == []
    assert any(row.id == file_id for row in lifecycle.list_due_meta_files(db.utcnow()))
    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=meta)
    assert meta.delete_calls == [str(file_id)]
    assert lifecycle.list_due_meta_files(db.utcnow()) == ()
