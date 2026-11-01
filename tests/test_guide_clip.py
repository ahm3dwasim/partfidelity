"""Guide-clip retention tests.

The published guide plays its own clip: retained at publish beside the
guide, recorded as a guide_clip artifact row, served with Range seeks,
kept outside the 24-hour preview promise, deleted with the guide.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db
from app import guide_clip
from app import export_links
from app.admission import admit_job, job_dir
from app.contracts import Guide, JobId, UploadPreview
from app.deletion import cancel_and_delete, sweep_expired
from app.ledger import LifecycleRepositoryImpl
from app.main import create_app
from app.pipeline import load_guide, save_guide
from app.routes import uploads as routes_mod
from app.routes.guides import router, set_guide_clip_loader, set_guide_loader
from app.settings import Settings
from app.uploads_store import UploadStoreImpl
from test_guide import _step
from test_pipeline import make_clip, sample_clients

CONSENT_SENTENCE = (
    "If you publish a guide, your video stays with it until you delete the guide."
)
# The upload page says the guide retention once, at the consent step; the
# page foot keeps only the two 24-hour limits.
RETENTION_SENTENCE = "about 24 hours after you upload it"


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


@pytest.fixture(scope="module")
def six_second_clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("media") / "six.mp4")


def _admit(tag: str):
    return admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")


def _job_root(job_id: JobId) -> Path:
    return job_dir(str(job_id), root=db.data_root())


def _register_preview(uploads, ctx, *, payload: bytes = b"preview-bytes"):
    root = _job_root(ctx.job_id)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "preview.mp4"
    target.write_bytes(payload)
    artifact_id = ctx.artifacts.register("upload_preview", target)
    preview = UploadPreview(
        job_id=ctx.job_id,
        local_video=target,
        duration_s=12.5,
        size_bytes=len(payload),
        audio_removed=True,
        confirmed_at=None,
    )
    record = uploads.register_preview(
        ctx, preview, artifact_id, db.utcnow() + timedelta(hours=24)
    )
    return record, artifact_id, target


def _guide(job_id: JobId, *, count: int = 1) -> Guide:
    steps = tuple(_step(number, ()) for number in range(1, count + 1))
    return Guide(
        job_id=job_id,
        steps=steps,
        frame_paths=(),
        labels=(),
        difficulty="Easy",
        verdict="TRY",
        draft_reasons=(),
    )


def _clip_app(*, guide=None, clip: Path | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    set_guide_loader(app, lambda job_id: guide)
    set_guide_clip_loader(app, lambda job_id: clip)
    return TestClient(app)


def _row_states(lane_db: Path, job_id: JobId) -> dict:
    conn = db.connect(lane_db)
    try:
        return {
            row["kind"]: row["delete_state"]
            for row in conn.execute(
                "SELECT kind, delete_state FROM artifacts WHERE job_id = ?",
                (str(job_id),),
            ).fetchall()
        }
    finally:
        conn.close()


# Retention at publish.


def test_publish_retains_clip_end_to_end(
    tmp_path: Path, lane_db: Path, six_second_clip: Path
) -> None:
    """Upload, confirm, and the guide owns a playable clip copy."""
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay",
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
    root = tmp_path / "data" / "jobs" / job_id
    preview = (root / "preview.mp4").read_bytes()
    clip = root / "guide_clip.mp4"
    assert clip.is_file()
    assert clip.read_bytes() == preview
    assert guide_clip.find_live_clips(job_id, db_path=lane_db)
    served = client.get(f"/guides/{job_id}/clip")
    assert served.status_code == 200
    assert served.headers["content-type"] == "video/mp4"
    assert served.content == preview
    page = client.get(f"/guides/{job_id}")
    assert page.status_code == 200
    assert f'src="/guides/{job_id}/clip" preload="metadata"' in page.text
    first = client.get(f"/guides/{job_id}/clip", headers={"Range": "bytes=0-99"})
    assert first.status_code == 206
    assert first.content == preview[:100]


def test_retain_skips_jobs_without_upload_row(uploads: UploadStoreImpl) -> None:
    """No upload row (the sample build) keeps no clip and raises nothing."""
    ctx = _admit("clip-skip")
    assert guide_clip.retain_guide_clip(ctx, uploads) is False
    assert guide_clip.find_live_clips(ctx.job_id) == ()
    assert not (_job_root(ctx.job_id) / "guide_clip.mp4").exists()


def test_retain_is_idempotent(uploads: UploadStoreImpl) -> None:
    ctx = _admit("clip-idem")
    _register_preview(uploads, ctx)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    assert len(guide_clip.find_live_clips(ctx.job_id)) == 1


def test_retain_fails_closed_when_preview_file_gone(
    uploads: UploadStoreImpl,
) -> None:
    ctx = _admit("clip-gone")
    _, _, target = _register_preview(uploads, ctx)
    target.unlink()
    with pytest.raises(guide_clip.GuideClipError):
        guide_clip.retain_guide_clip(ctx, uploads)
    assert guide_clip.find_live_clips(ctx.job_id) == ()


def test_retain_rejects_bad_filenames(uploads: UploadStoreImpl) -> None:
    ctx = _admit("clip-name")
    _register_preview(uploads, ctx)
    for bad in ("", ".", "..", "a/b", "a\\b", "a\x00b"):
        with pytest.raises(guide_clip.GuideClipError):
            guide_clip.retain_guide_clip(ctx, uploads, filename=bad)
    assert guide_clip.find_live_clips(ctx.job_id) == ()


def test_retain_removes_partial_copy_when_registration_fails(
    uploads: UploadStoreImpl,
) -> None:
    from types import SimpleNamespace

    ctx = _admit("clip-regfail")
    _register_preview(uploads, ctx)

    class _FailingStore:
        def register(self, kind: str, path: Path):
            raise RuntimeError("disk is full")

    broken = SimpleNamespace(job_id=ctx.job_id, artifacts=_FailingStore())
    with pytest.raises(guide_clip.GuideClipError):
        guide_clip.retain_guide_clip(broken, uploads)
    assert not (_job_root(ctx.job_id) / "guide_clip.mp4").exists()
    assert guide_clip.find_live_clips(ctx.job_id) == ()


def test_retain_tombstones_pending_row_when_copy_fails(
    lane_db: Path, uploads: UploadStoreImpl, monkeypatch,
) -> None:
    """A failed copy removes its pending row; nothing stays half-kept."""
    ctx = _admit("clip-copyfail")
    _register_preview(uploads, ctx, payload=PAYLOAD)

    def _boom(source: Path, dest: Path, job_id) -> None:
        raise OSError("disk is full")

    monkeypatch.setattr(guide_clip, "_copy_atomic", _boom)
    with pytest.raises(guide_clip.GuideClipError, match="clip copy failed"):
        guide_clip.retain_guide_clip(ctx, uploads)
    assert not (_job_root(ctx.job_id) / "guide_clip.mp4").exists()
    assert guide_clip.find_live_clips(ctx.job_id) == ()
    assert _row_states(lane_db, ctx.job_id) == {
        "upload_preview": "live",
        "guide_clip": "tombstoned",
    }


def test_retain_drops_copy_when_delete_wins_promotion_race(
    lane_db: Path, uploads: UploadStoreImpl, monkeypatch,
) -> None:
    """A delete racing the promotion keeps the tombstone; our copy is dropped."""
    ctx = _admit("clip-promoterace")
    _register_preview(uploads, ctx, payload=PAYLOAD)
    real_promote = guide_clip._promote_clip_row

    def _deleted_first(artifact_id) -> bool:
        ctx.artifacts.mark_deleted(artifact_id)  # cancel/sweep won the race
        return real_promote(artifact_id)  # sees no pending row now

    monkeypatch.setattr(guide_clip, "_promote_clip_row", _deleted_first)
    with pytest.raises(guide_clip.GuideClipError, match="deleted during retain"):
        guide_clip.retain_guide_clip(ctx, uploads)
    assert not (_job_root(ctx.job_id) / "guide_clip.mp4").exists()
    assert guide_clip.find_live_clips(ctx.job_id) == ()
    assert _row_states(lane_db, ctx.job_id)["guide_clip"] == "tombstoned"


def test_crash_between_copy_and_promote_sweeps_clip(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files, monkeypatch,
) -> None:
    """A crash after the copy leaves a tracked pending row the sweep deletes."""
    ctx = _admit("clip-crash")
    _register_preview(uploads, ctx, payload=PAYLOAD)

    def _crash(artifact_id) -> bool:
        raise KeyboardInterrupt("simulated crash between copy and promote")

    monkeypatch.setattr(guide_clip, "_promote_clip_row", _crash)
    with pytest.raises(KeyboardInterrupt):
        guide_clip.retain_guide_clip(ctx, uploads)
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    assert clip_file.is_file()  # the copy landed; no cleanup ran
    assert guide_clip.find_live_clips(ctx.job_id) == ()
    assert _row_states(lane_db, ctx.job_id)["guide_clip"] == "pending_delete"
    ctx.finish("failed")
    swept = sweep_expired(
        db.utcnow() + timedelta(seconds=1), uploads=uploads,
        lifecycle=lifecycle, client=fake_meta_files,
    )
    assert swept.artifacts_tombstoned == 1
    assert not clip_file.exists()
    assert _row_states(lane_db, ctx.job_id)["guide_clip"] == "tombstoned"


def test_pipeline_fails_closed_when_retention_fails(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, monkeypatch,
) -> None:
    """A lost clip fails the publish instead of shipping half a guide."""
    from app.pipeline import guide_path, run_job

    def _boom(ctx, store, *, filename=None):
        raise guide_clip.GuideClipError("clip copy failed")

    monkeypatch.setattr(guide_clip, "retain_guide_clip", _boom)
    clip = make_clip(tmp_path / "boom.mp4")
    result = run_job(
        clip, clients=sample_clients(), uploads=uploads,
        lifecycle=lifecycle, anon_id="anon-boom", ip_hash="ip-boom",
        replay=True,
    )
    assert result.state == "failed"
    assert result.error == "could not keep guide clip"
    assert result.job_id is not None
    assert not guide_path(result.job_id).exists()


# Route, ranges, access.

PAYLOAD = bytes(range(256)) * 4


def _payload_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(PAYLOAD)
    return clip


def test_clip_route_serves_full_and_partial_bytes(tmp_path: Path) -> None:
    guide = _guide(JobId("job-clip"))
    client = _clip_app(guide=guide, clip=_payload_clip(tmp_path))
    full = client.get("/guides/job-clip/clip")
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    assert "attachment" not in full.headers.get("content-disposition", "")
    assert full.content == PAYLOAD
    first = client.get("/guides/job-clip/clip", headers={"Range": "bytes=0-99"})
    assert first.status_code == 206
    assert first.headers["content-range"] == f"bytes 0-99/{len(PAYLOAD)}"
    assert first.content == PAYLOAD[:100]
    tail = client.get("/guides/job-clip/clip", headers={"Range": "bytes=-100"})
    assert tail.status_code == 206
    assert tail.content == PAYLOAD[-100:]
    rest = client.get("/guides/job-clip/clip", headers={"Range": "bytes=100-"})
    assert rest.status_code == 206
    assert rest.content == PAYLOAD[100:]


def test_clip_route_rejects_bad_and_malformed_ranges(tmp_path: Path) -> None:
    guide = _guide(JobId("job-clip"))
    client = _clip_app(guide=guide, clip=_payload_clip(tmp_path))
    past = client.get(
        "/guides/job-clip/clip", headers={"Range": f"bytes={len(PAYLOAD)}-"}
    )
    assert past.status_code == 416
    assert past.headers["content-range"] == f"bytes */{len(PAYLOAD)}"
    malformed = client.get(
        "/guides/job-clip/clip", headers={"Range": "bytes=abc"}
    )
    assert malformed.status_code == 400


def test_clip_singular_alias_serves_same_bytes(tmp_path: Path) -> None:
    guide = _guide(JobId("job-clip"))
    client = _clip_app(guide=guide, clip=_payload_clip(tmp_path))
    response = client.get("/guide/job-clip/clip")
    assert response.status_code == 200
    assert response.content == PAYLOAD
    ranged = client.get("/guide/job-clip/clip", headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == PAYLOAD[:10]


def test_clip_route_404s_without_guide_or_clip(tmp_path: Path) -> None:
    clip = _payload_clip(tmp_path)
    guide = _guide(JobId("job-clip"))
    assert _clip_app(guide=None, clip=clip).get("/guides/job-clip/clip").status_code == 404
    assert _clip_app(guide=None, clip=clip).get("/guide/job-clip/clip").status_code == 404
    assert _clip_app(guide=guide, clip=None).get("/guides/job-clip/clip").status_code == 404
    missing = tmp_path / "missing.mp4"
    assert _clip_app(guide=guide, clip=missing).get("/guides/job-clip/clip").status_code == 404


def test_clip_route_never_builds_paths_from_job_id(tmp_path: Path) -> None:
    """The id only selects loader rows; it never touches the filesystem."""
    seen: list[str] = []
    app = FastAPI()
    app.include_router(router)

    def _guide_loader(job_id: str):
        seen.append(job_id)
        return None

    set_guide_loader(app, _guide_loader)
    set_guide_clip_loader(app, lambda job_id: tmp_path / "clip.mp4")
    client = TestClient(app)
    # A routable hostile id reaches the loader verbatim and 404s there.
    response = client.get("/guides/..%5Cx/clip")
    assert response.status_code == 404
    assert seen == ["..\\x"]
    # An encoded slash never even routes: 404 with no loader call.
    assert client.get("/guides/..%2F..%2Fx/clip").status_code == 404
    assert seen == ["..\\x"]
    assert not (tmp_path / "clip.mp4").exists()


def test_resolve_clip_path_refuses_hostile_rows(tmp_path: Path) -> None:
    from app.contracts import ArtifactId, ArtifactRecord

    data_root = tmp_path / "data"
    job_root = data_root / "jobs" / "job-clip"
    job_root.mkdir(parents=True)
    (job_root / "guide_clip.mp4").write_bytes(b"clip")

    def _record(relative: Path) -> ArtifactRecord:
        return ArtifactRecord(
            id=ArtifactId("clip-row"),
            job_id=JobId("job-clip"),
            kind="guide_clip",
            relative_path=relative,
            expires_at=db.utcnow(),
        )

    def _resolve(relative: Path):
        return guide_clip.resolve_clip_path(_record(relative), data_root=data_root)

    assert _resolve(Path("nope.mp4")) is None
    assert _resolve(Path("/etc/passwd")) is None
    assert _resolve(Path("../evil.mp4")) is None
    assert _resolve(Path("a/../../evil.mp4")) is None
    assert _resolve(Path("guide_clip.mp4")) == job_root / "guide_clip.mp4"
    link = job_root / "link.mp4"
    try:
        link.symlink_to(job_root / "guide_clip.mp4")
    except OSError:
        pytest.skip("symlinks unavailable")
    assert _resolve(Path("link.mp4")) is None


def test_load_guide_clip_skips_rows_without_files(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl
) -> None:
    assert guide_clip.load_guide_clip("unknown-job", db_path=lane_db) is None
    ctx = _admit("clip-load")
    _register_preview(uploads, ctx)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    found = guide_clip.load_guide_clip(ctx.job_id, db_path=lane_db)
    assert found == _job_root(ctx.job_id) / "guide_clip.mp4"
    found.unlink()
    assert guide_clip.load_guide_clip(ctx.job_id, db_path=lane_db) is None


def test_guide_page_wires_clip_src_or_bare_player(tmp_path: Path) -> None:
    guide = _guide(JobId("job-clip"))
    with_clip = _clip_app(guide=guide, clip=_payload_clip(tmp_path)).get(
        "/guides/job-clip"
    )
    assert with_clip.status_code == 200
    assert 'src="/guides/job-clip/clip" preload="metadata"' in with_clip.text
    bare = _clip_app(guide=guide, clip=None).get("/guides/job-clip")
    assert bare.status_code == 200
    assert '<video id="guide-video" controls playsinline></video>' in bare.text
    assert "GUIDE_VIDEO_ATTRS" not in bare.text
    singular = _clip_app(guide=guide, clip=_payload_clip(tmp_path)).get(
        "/guide/job-clip"
    )
    assert 'src="/guides/job-clip/clip" preload="metadata"' in singular.text


def test_guide_page_escapes_hostile_job_id_in_clip_src(tmp_path: Path) -> None:
    from urllib.parse import quote

    hostile = 'a"><img src=x onerror=alert(1)>'
    guide = _guide(JobId(hostile))
    body = _clip_app(guide=guide, clip=_payload_clip(tmp_path)).get(
        f"/guides/{quote(hostile, safe='')}"
    ).text
    assert hostile not in body
    assert f'src="/guides/{quote(hostile, safe="")}/clip"' in body


# Deletion, sweep, preview expiry.


def test_cancel_deletes_clip_file_and_row(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files,
) -> None:
    ctx = _admit("clip-cancel")
    _, _, preview_file = _register_preview(uploads, ctx)
    guide_file = save_guide(_guide(ctx.job_id))
    sidecar = guide_file.parent / "corrections.json"
    sidecar.write_text('{"ops": []}')
    token = export_links.create(str(ctx.job_id), hours=24, path=lane_db)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"

    app = FastAPI()
    app.include_router(router)
    assert load_guide(ctx.job_id) is not None
    set_guide_loader(app, lambda job_id: load_guide(job_id))
    set_guide_clip_loader(
        app, lambda job_id: guide_clip.load_guide_clip(job_id, db_path=lane_db)
    )
    client = TestClient(app)
    assert client.get(f"/guides/{ctx.job_id}/clip").status_code == 200

    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle,
        client=fake_meta_files,
    )
    assert result.local_deleted is True
    assert result.recorded_state == "cancelled"
    assert not clip_file.exists()
    assert not preview_file.exists()
    assert not guide_file.exists() and not sidecar.exists()
    assert not export_links.resolve(token, str(ctx.job_id), path=lane_db)
    assert _row_states(lane_db, ctx.job_id) == {
        "upload_preview": "tombstoned",
        "guide_clip": "tombstoned",
        "cancel_guide": "tombstoned",
        "cancel_corrections": "tombstoned",
        "cancel_export_links": "tombstoned",
    }
    assert client.get(f"/guides/{ctx.job_id}/clip").status_code == 404
    assert client.get(f"/guides/{ctx.job_id}").status_code == 404


def test_deferred_cancel_of_a_done_guide_sweeps_everything_once_quiet(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files, monkeypatch,
) -> None:
    from app.providers import open_readback

    ctx = _admit("clip-deferred")
    _, _, preview_file = _register_preview(uploads, ctx)
    guide_file = save_guide(_guide(ctx.job_id))
    token = export_links.create(str(ctx.job_id), hours=24, path=lane_db)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    ctx.finish("done")
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    quiet = []  # a second-check call is in flight until this holds True
    monkeypatch.setattr(open_readback, "cancel", lambda job_id: bool(quiet))
    monkeypatch.setattr(open_readback, "wait_quiet", lambda job_id, timeout_s=None: bool(quiet))

    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake_meta_files,
    )
    assert result.local_deleted is False and result.recorded_state == "cancelling"
    assert clip_file.is_file() and preview_file.is_file() and guide_file.is_file()
    assert not export_links.resolve(token, str(ctx.job_id), path=lane_db)  # revoked at once

    quiet.append(True)
    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=fake_meta_files)
    assert not clip_file.exists() and not preview_file.exists() and not guide_file.exists()
    assert set(_row_states(lane_db, ctx.job_id).values()) == {"tombstoned"}
    assert uploads.load_job(ctx.job_id)[1].state == "cancelled"


def test_exit_right_after_cancel_commit_still_sweeps_everything_once_quiet(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files, monkeypatch,
) -> None:
    from app.providers import open_readback

    ctx = _admit("clip-crash")
    _, _, preview_file = _register_preview(uploads, ctx)
    guide_file = save_guide(_guide(ctx.job_id))
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    ctx.finish("done")
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    quiet = []  # a second-check call is in flight until this holds True
    monkeypatch.setattr(open_readback, "cancel", lambda job_id: bool(quiet))
    monkeypatch.setattr(open_readback, "wait_quiet", lambda job_id, timeout_s=None: bool(quiet))

    class ProcessExit(BaseException):
        pass

    real = uploads.request_cancellation

    def commit_then_exit(*args, **kwargs):
        real(*args, **kwargs)
        raise ProcessExit  # no code after the commit runs

    monkeypatch.setattr(uploads, "request_cancellation", commit_then_exit)
    with pytest.raises(ProcessExit):
        cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake_meta_files)
    monkeypatch.setattr(uploads, "request_cancellation", real)
    states = _row_states(lane_db, ctx.job_id)
    assert states["upload_preview"] == "pending_delete" and states["guide_clip"] == "pending_delete"

    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=fake_meta_files)
    assert clip_file.is_file() and preview_file.is_file() and guide_file.is_file()  # busy: retry

    quiet.append(True)
    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle, client=fake_meta_files)
    assert not clip_file.exists() and not preview_file.exists() and not guide_file.exists()
    assert set(_row_states(lane_db, ctx.job_id).values()) == {"tombstoned"}


def test_failed_guide_unlink_stays_pending_until_sweep(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files, monkeypatch,
) -> None:
    ctx = _admit("guide-unlink-retry")
    _register_preview(uploads, ctx)
    guide_file = save_guide(_guide(ctx.job_id))
    original_unlink = Path.unlink
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if self == guide_file and not failed:
            failed = True
            raise PermissionError("injected unlink failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=fake_meta_files,
    )
    assert result.recorded_state == "cancelling"
    assert guide_file.exists()
    app = FastAPI()
    app.state.settings = Settings(data_root=db.data_root(), db_path=lane_db,
                                  production=False)
    app.include_router(router)
    set_guide_loader(app, lambda job_id: load_guide(job_id))
    assert TestClient(app).get(f"/guides/{ctx.job_id}").status_code == 404
    assert _row_states(lane_db, ctx.job_id)["cancel_guide"] == "pending_delete"
    sweep_expired(db.utcnow(), uploads=uploads, lifecycle=lifecycle,
                  client=fake_meta_files)
    assert not guide_file.exists()
    assert TestClient(app).get(f"/guides/{ctx.job_id}").status_code == 404
    assert uploads.load_job(ctx.job_id)[1].state == "cancelled"
    assert _row_states(lane_db, ctx.job_id)["cancel_guide"] == "tombstoned"


def test_sweep_takes_preview_but_never_guide_clip(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files,
) -> None:
    ctx = _admit("clip-sweep")
    _, preview_id, preview_file = _register_preview(uploads, ctx)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    ctx.finish("done")
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    past = db.format_dt(db.utcnow() - timedelta(days=1))
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "UPDATE artifacts SET expires_at = ? WHERE job_id = ?",
            (past, str(ctx.job_id)),
        )
    finally:
        conn.close()
    due = {row.id for row in lifecycle.list_due_artifacts(db.utcnow())}
    assert preview_id in due
    assert len(guide_clip.find_live_clips(ctx.job_id, db_path=lane_db)) == 1
    swept = sweep_expired(
        db.utcnow(), uploads=uploads, lifecycle=lifecycle,
        client=fake_meta_files,
    )
    assert swept.artifacts_tombstoned == 1
    assert not preview_file.exists()
    assert clip_file.is_file()
    assert _row_states(lane_db, ctx.job_id) == {
        "upload_preview": "tombstoned",
        "guide_clip": "live",
    }


def test_sweep_takes_crashed_publish_clip_orphan(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files,
) -> None:
    """A failed job's clip row is an orphan: it sweeps like the rest."""
    ctx = _admit("clip-orphan")
    _register_preview(uploads, ctx)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    ctx.finish("failed")
    past = db.format_dt(db.utcnow() - timedelta(days=1))
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "UPDATE artifacts SET expires_at = ? WHERE job_id = ?",
            (past, str(ctx.job_id)),
        )
    finally:
        conn.close()
    kinds = {row.kind for row in lifecycle.list_due_artifacts(db.utcnow())}
    assert "guide_clip" in kinds
    swept = sweep_expired(
        db.utcnow(), uploads=uploads, lifecycle=lifecycle,
        client=fake_meta_files,
    )
    assert swept.artifacts_tombstoned == 2
    assert not clip_file.exists()
    assert _row_states(lane_db, ctx.job_id)["guide_clip"] == "tombstoned"


def test_preview_expiry_hides_preview_but_not_guide_clip(
    tmp_path: Path, lane_db: Path, uploads: UploadStoreImpl
) -> None:
    """The 24-hour promise still hides the preview on a guided job."""
    ctx = _admit("clip-expire")
    _register_preview(uploads, ctx, payload=PAYLOAD)
    save_guide(_guide(ctx.job_id))
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    ctx.finish("done")
    past = db.format_dt(db.utcnow() - timedelta(hours=1))
    conn = db.connect(lane_db)
    try:
        conn.execute(
            "UPDATE uploads SET preview_expires_at = ? WHERE job_id = ?",
            (past, str(ctx.job_id)),
        )
    finally:
        conn.close()
    settings = Settings(
        data_root=tmp_path / "data", db_path=lane_db, production=False,
        pipeline_mode="replay",
    )
    client = TestClient(create_app(settings))
    assert client.get(f"/preview/{ctx.job_id}").status_code == 410
    assert client.get(f"/preview/{ctx.job_id}/video").status_code == 410
    clip = client.get(f"/guides/{ctx.job_id}/clip")
    assert clip.status_code == 200
    assert clip.content == PAYLOAD
    page = client.get(f"/guides/{ctx.job_id}")
    assert page.status_code == 200
    assert f'src="/guides/{ctx.job_id}/clip" preload="metadata"' in page.text


# Consent copy and settings.


def test_upload_page_states_guide_retention() -> None:
    body = routes_mod._upload_page()
    assert "own footage" in body
    assert body.count(CONSENT_SENTENCE) == 1
    assert RETENTION_SENTENCE in body


def test_guide_clip_filename_setting(tmp_path: Path) -> None:
    base = {
        "data_root": tmp_path, "db_path": tmp_path / "test.sqlite3",
        "production": False,
    }
    assert Settings(**base).guide_clip_filename == "guide_clip.mp4"
    assert Settings(**base, guide_clip_filename="clip.mp4").guide_clip_filename == "clip.mp4"
    for bad in ("", ".", "..", "a/b", "a\\b", "a\x00b", "x" * 65):
        with pytest.raises(ValueError, match="guide_clip_filename"):
            Settings(**base, guide_clip_filename=bad)


# Cancel and preview failure paths.


def test_cancel_preview_failure_still_deletes_clip(
    lane_db: Path, uploads: UploadStoreImpl,
    lifecycle: LifecycleRepositoryImpl, fake_meta_files,
) -> None:
    """A preview unlink failure never strands the clip; both stay retryable."""
    ctx = _admit("clip-case1-previewfail")
    _, _, preview_file = _register_preview(uploads, ctx, payload=PAYLOAD)
    save_guide(_guide(ctx.job_id))
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    preview_file.unlink()
    preview_file.mkdir()

    result = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle,
        client=fake_meta_files,
    )
    assert result.local_deleted is False
    assert result.recorded_state == "cancelling"
    assert "preview delete failed" in (result.error or "")
    assert not clip_file.exists()
    assert _row_states(lane_db, ctx.job_id) == {
        "upload_preview": "pending_delete",
        "guide_clip": "tombstoned",
        "cancel_guide": "tombstoned",
        "cancel_corrections": "tombstoned",
        "cancel_export_links": "tombstoned",
    }

    preview_file.rmdir()
    preview_file.write_bytes(PAYLOAD)
    retried = cancel_and_delete(
        ctx.job_id, uploads=uploads, lifecycle=lifecycle,
        client=fake_meta_files,
    )
    assert retried.local_deleted is True
    assert retried.recorded_state == "cancelled"
    assert _row_states(lane_db, ctx.job_id) == {
        "upload_preview": "tombstoned",
        "guide_clip": "tombstoned",
        "cancel_guide": "tombstoned",
        "cancel_corrections": "tombstoned",
        "cancel_export_links": "tombstoned",
    }


def test_delete_job_clips_retries_after_unlink_failure(
    lane_db: Path, uploads: UploadStoreImpl
) -> None:
    """A failed first unlink leaves a pending row the retry deletes at once."""
    ctx = _admit("clip-case1-retry")
    _register_preview(uploads, ctx, payload=PAYLOAD)
    assert guide_clip.retain_guide_clip(ctx, uploads) is True
    clip_file = _job_root(ctx.job_id) / "guide_clip.mp4"
    clip_file.unlink()
    clip_file.mkdir()

    with pytest.raises(OSError):
        guide_clip.delete_job_clips(ctx.job_id, artifacts=ctx.artifacts)
    assert _row_states(lane_db, ctx.job_id)["guide_clip"] == "pending_delete"

    clip_file.rmdir()
    clip_file.write_bytes(PAYLOAD)
    assert (
        guide_clip.delete_job_clips(ctx.job_id, artifacts=ctx.artifacts) == 1
    )
    assert not clip_file.exists()
    assert _row_states(lane_db, ctx.job_id)["guide_clip"] == "tombstoned"
    assert guide_clip.load_guide_clip(ctx.job_id, db_path=lane_db) is None
