"""A published guide keeps every image it shows until the guide is deleted."""

from __future__ import annotations

import json
import io
import re
import sqlite3
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import db, deletion
from app.admission import admit_job, job_dir
from app.deletion import cancel_and_delete, sweep_expired
from app.ledger import LifecycleRepositoryImpl
from app.pipeline import guide_path
from app.pipeline import save_guide, load_guide
from app.guides import assemble_guide
from tests.test_guide import RecordingContext, _render, _decision, _slot, _step, _frame
from app.uploads import confirm_and_send
from app.uploads_store import UploadStoreImpl
from tests.test_deletion import FakeClock, FakeMetaFiles, _prepare  # noqa: F401
from tests.test_deletion import short_clip  # noqa: F401  (fixture)


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture(autouse=True)
def _fake_clock(monkeypatch, clock):
    monkeypatch.setattr(db, "utcnow", clock)


@pytest.fixture
def uploads(lane_db, clock):
    return UploadStoreImpl(lane_db, clock=clock)


@pytest.fixture
def lifecycle(lane_db):
    return LifecycleRepositoryImpl(lane_db)


def _publish(tag: str, uploads, clip: Path, *, finish: str = "done"):
    """A job holding one kept and one discarded render, a frame and a mask."""
    ctx = admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")
    _prepare(uploads, ctx, clip)
    confirm_and_send(ctx.job_id, uploads=uploads, client=FakeMetaFiles())
    directory = job_dir(str(ctx.job_id))
    paths = {}
    render = _render(RecordingContext(), directory, "kept", "fan")
    paths = {"kept.png": render.rendered_image_path,
             "frame0.png": directory / "kept-frame.png",
             "mask0.png": render.detection.mask_path,
             "discarded.png": _frame(directory / "discarded.png")}
    kept = ctx.artifacts.register("provisional_render", paths["kept.png"])
    ctx.artifacts.register("provisional_render", paths["discarded.png"])
    guide = assemble_guide(RecordingContext(), [_step(1, ("fan",))],
                           [paths["frame0.png"]], [_decision(render, slot=_slot())])
    save_guide(replace(guide, job_id=ctx.job_id))
    if finish == "done":
        ctx.finish("done", keep_artifact_ids=[kept])
    else:
        ctx.finish("failed")
    return ctx, paths


def _rows(lane_db: Path, job_id) -> dict[str, dict]:
    conn = sqlite3.connect(lane_db)
    conn.row_factory = sqlite3.Row
    try:
        return {r["id"]: dict(r) for r in conn.execute(
            "SELECT * FROM artifacts WHERE job_id = ?", (str(job_id),))}
    finally:
        conn.close()


def _sweep(clock, uploads, lifecycle, **advance):
    clock.advance(**advance)
    return sweep_expired(clock(), uploads=uploads, lifecycle=lifecycle,
                         client=FakeMetaFiles(clock))


def test_published_guide_images_survive_past_retention(
    lane_db, uploads, lifecycle, clock, short_clip
):
    ctx, paths = _publish("keep", uploads, short_clip)
    _sweep(clock, uploads, lifecycle, days=8)
    _sweep(clock, uploads, lifecycle, days=30)
    for name in ("kept.png", "frame0.png", "mask0.png"):
        assert paths[name].is_file(), name
    kept = [r for r in _rows(lane_db, ctx.job_id).values() if r["kept_at"]]
    assert len(kept) == 1 and kept[0]["delete_state"] == "live"


def test_discarded_renders_still_sweep_at_seven_days(
    lane_db, uploads, lifecycle, clock, short_clip
):
    _ctx, paths = _publish("disc", uploads, short_clip)
    _sweep(clock, uploads, lifecycle, days=8)
    assert not paths["discarded.png"].exists()
    assert paths["kept.png"].is_file()


@pytest.mark.parametrize("finish", ["failed"])
def test_failed_job_renders_still_sweep_at_seven_days(
    lane_db, uploads, lifecycle, clock, short_clip, finish
):
    _ctx, paths = _publish("fail", uploads, short_clip, finish=finish)
    _sweep(clock, uploads, lifecycle, days=8)
    assert not paths["kept.png"].exists()
    assert not paths["discarded.png"].exists()


def test_guide_deletion_removes_renders_frames_and_masks(
    lane_db, uploads, lifecycle, clock, short_clip
):
    ctx, paths = _publish("del", uploads, short_clip)
    cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle,
                      client=FakeMetaFiles(clock))
    _sweep(clock, uploads, lifecycle, minutes=5)
    for name in ("kept.png", "frame0.png", "mask0.png"):
        assert not paths[name].exists(), name


def test_guide_deletion_while_second_check_runs_waits_then_removes_images(
    lane_db, uploads, lifecycle, clock, short_clip, monkeypatch
):
    from app.providers import open_readback

    ctx, paths = _publish("busy", uploads, short_clip)
    guide_file = guide_path(ctx.job_id)
    quiet = []  # a second-check call is in flight until this holds True
    monkeypatch.setattr(open_readback, "cancel", lambda job_id: bool(quiet))
    monkeypatch.setattr(open_readback, "wait_quiet", lambda job_id, timeout_s=None: bool(quiet))
    result = cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle,
                               client=FakeMetaFiles(clock))
    assert result.local_deleted is False
    rows = _rows(lane_db, ctx.job_id).values()
    for name in ("kept.png", "frame0.png", "mask0.png"):
        row = next(r for r in rows if r["relative_path"] == paths[name].name)
        assert row["delete_state"] == "pending_delete" and row["kept_at"] is None, name
    _sweep(clock, uploads, lifecycle, minutes=5)
    for name in ("kept.png", "frame0.png", "mask0.png"):
        assert paths[name].is_file(), name  # never removed under a running check
    assert guide_file.is_file()
    quiet.append(True)
    _sweep(clock, uploads, lifecycle, minutes=5)
    for name in ("kept.png", "frame0.png", "mask0.png", "discarded.png"):
        assert not paths[name].exists(), name
    assert not guide_file.exists()


def test_guide_deletion_retries_after_crash_and_failed_unlink(
    lane_db, uploads, lifecycle, clock, short_clip, monkeypatch
):
    ctx, paths = _publish("crash", uploads, short_clip)
    # Crash: only the durable queueing committed, nothing was unlinked.
    uploads.request_cancellation(ctx.job_id, clock())
    queued = [r for r in _rows(lane_db, ctx.job_id).values()
              if r["delete_state"] == "pending_delete"]
    assert all(any(r["relative_path"] == paths[name].name for r in queued)
               for name in ("frame0.png", "mask0.png", "kept.png"))
    real = deletion._delete_one_file
    monkeypatch.setattr(deletion, "_delete_one_file", lambda p: False)
    _sweep(clock, uploads, lifecycle, minutes=5)
    assert paths["frame0.png"].exists()
    monkeypatch.setattr(deletion, "_delete_one_file", real)
    _sweep(clock, uploads, lifecycle, hours=2)
    for name in ("kept.png", "frame0.png", "mask0.png"):
        assert not paths[name].exists(), name


def test_prechange_published_guide_is_safe_at_next_sweep(
    lane_db, uploads, lifecycle, clock, short_clip
):
    """A guide published before this change already has kept_at set."""
    ctx, paths = _publish("old", uploads, short_clip)
    conn = sqlite3.connect(lane_db)
    conn.execute("UPDATE artifacts SET expires_at = ? WHERE job_id = ?",
                 (db.format_dt(clock() - timedelta(days=1)), str(ctx.job_id)))
    conn.commit()
    conn.close()
    _sweep(clock, uploads, lifecycle, minutes=1)
    assert paths["kept.png"].is_file()


def test_retention_config_fails_loud(tmp_path):
    from app.retention_config import RetentionConfigError, guide_lifetime_kinds

    assert "provisional_render" in guide_lifetime_kinds()
    for body in ('', 'guide_lifetime_kind = ["x"]\n', 'guide_lifetime_kinds = []\n',
                 'guide_lifetime_kinds = [1]\n'):
        bad = tmp_path / "bad.toml"
        bad.write_text(body)
        with pytest.raises(RetentionConfigError):
            guide_lifetime_kinds(bad)
    for path in (tmp_path / "missing.toml", tmp_path / "syntax.toml"):
        if path.name == "syntax.toml":
            path.write_text("invalid = [")
        with pytest.raises(RetentionConfigError):
            guide_lifetime_kinds(path)


@pytest.mark.parametrize("failure", ["json", "oserror"])
def test_bad_guide_rolls_back_cancellation(lane_db, uploads, clock, short_clip, monkeypatch, failure):
    ctx, paths = _publish(failure, uploads, short_clip)
    before = _rows(lane_db, ctx.job_id)
    path = guide_path(ctx.job_id)
    if failure == "json":
        path.write_text("{")
    else:
        original = Path.read_text
        def read_text(self, *args, **kwargs):
            if self == path:
                raise OSError("unreadable")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises((ValueError, OSError)):
        uploads.request_cancellation(ctx.job_id, clock())
    assert path.exists() and all(p.exists() for p in paths.values())
    assert _rows(lane_db, ctx.job_id) == before


@pytest.mark.parametrize("guide_exists", [True, False])
def test_failed_publication_queues_images(lane_db, uploads, lifecycle, clock, short_clip, guide_exists):
    ctx, paths = _publish(f"failure-{guide_exists}", uploads, short_clip, finish="failed")
    if not guide_exists:
        guide_path(ctx.job_id).unlink()
    deletion.cleanup_failed_guide(ctx.job_id, artifacts=ctx.artifacts)
    _sweep(clock, uploads, lifecycle, days=8)
    assert all(not p.exists() for p in paths.values())


def test_kept_guide_delivery_survives_sweep_and_disappears_on_deletion(
    lane_db, uploads, lifecycle, clock, short_clip
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes.guides import router, set_guide_loader
    from app.settings import Settings
    from tests.test_export import MemoryStore
    from app.routes.guides import set_corrections_store

    ctx, paths = _publish("delivery", uploads, short_clip)
    app = FastAPI()
    app.state.settings = Settings(data_root=db.data_root(), db_path=lane_db, production=False)
    app.include_router(router)
    set_guide_loader(app, lambda job: load_guide(job))
    set_corrections_store(app, MemoryStore())
    client = TestClient(app)
    base = f"/guides/{ctx.job_id}"
    before = client.get(base)
    image_token = re.search(r'/images.zip\?token=([^"&]+)', before.text).group(1)
    before_images = client.get(base + "/images.zip?token=" + image_token).content
    with zipfile.ZipFile(io.BytesIO(before_images)) as archive:
        expected_composite = archive.read("step-01.png")
    _sweep(clock, uploads, lifecycle, days=8)
    page = client.get(base)
    assert page.status_code == 200
    image_urls = set(re.findall(rf'{base}/image/\d+', page.text))
    assert image_urls == {base + "/image/0", base + "/image/1"}
    assert {client.get(url).content for url in image_urls} == {
        paths["frame0.png"].read_bytes(), paths["kept.png"].read_bytes()
    }
    urls = [base + f"/{kind}.zip?token=" + re.search(rf'/{kind}.zip\?token=([^"&]+)', page.text).group(1)
            for kind in ("images", "export")]
    images, export = [client.get(url) for url in urls]
    assert images.status_code == export.status_code == 200
    with zipfile.ZipFile(io.BytesIO(images.content)) as archive:
        assert set(archive.namelist()) == {"step-01.png"}
        assert archive.read("step-01.png") == expected_composite
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        assert set(archive.namelist()) == {
            "guide.html", "images/step-01-original.png",
            "images/step-01-part1-labelled.png",
        }
        assert archive.read("images/step-01-original.png") == paths["frame0.png"].read_bytes()
        assert archive.read("images/step-01-part1-labelled.png") == paths["kept.png"].read_bytes()
    cancel_and_delete(ctx.job_id, uploads=uploads, lifecycle=lifecycle, client=FakeMetaFiles(clock))
    _sweep(clock, uploads, lifecycle, minutes=5)
    assert client.get(base).status_code == 404
    assert all(client.get(url).status_code == 404 for url in image_urls)
    assert all(client.get(url).status_code == 404 for url in urls)
    assert all(not p.exists() for p in paths.values())


def _failed_job(tag: str, **names) -> Path:
    """A job failed at admission time, its folder left behind."""
    from app.admission import MAX_RUNNING
    from app.ledger import _finalize_job_failed

    ctx = admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}", **names)
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _finalize_job_failed(conn, str(ctx.job_id), db.format_dt(db.utcnow()), MAX_RUNNING)
        conn.execute("COMMIT")
    finally:
        conn.close()
    directory = job_dir(str(ctx.job_id))
    (directory / "frames").mkdir(parents=True, exist_ok=True)
    (directory / "frames" / "a.png").write_bytes(b"png")
    (directory / "b.png").write_bytes(b"png")
    return directory


def _rescan_done(job_id: str) -> bool:
    conn = db.connect()
    try:
        return conn.execute("SELECT rescan_done_at FROM jobs WHERE id = ?",
                            (job_id,)).fetchone()[0] is not None
    finally:
        conn.close()


def test_rescan_uses_each_jobs_own_filenames(lane_db, uploads, lifecycle, clock):
    directory = _failed_job("names", guide_filename="custom-guide.json",
                            corrections_filename="custom-fixes.json")
    (directory / "custom-guide.json").write_text(json.dumps({"frame_paths": [], "labels": []}))
    (directory / "custom-fixes.json").write_text("{}")
    _sweep(clock, uploads, lifecycle, days=8)
    for _ in range(2):
        _sweep(clock, uploads, lifecycle, minutes=5)
    assert not (directory / "custom-guide.json").exists()
    assert not (directory / "custom-fixes.json").exists()
    assert not list(directory.rglob("*.png"))
    assert _rescan_done(directory.name)


def test_old_jobs_get_the_configured_filenames(tmp_path):
    database = tmp_path / "old.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    job = admit_job("anon-old", "ip-old", "video-old")
    conn = db.connect(database)
    conn.execute("DROP INDEX idx_jobs_rescan_pending")
    for column in ("guide_filename", "corrections_filename", "rescan_seen_at", "rescan_done_at"):
        conn.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
    conn.close()
    for _ in range(2):  # the migration is safe to re-run
        db.init_db(database, guide_filename="g.json", corrections_filename="c.json")
    conn = db.connect(database)
    row = conn.execute("SELECT guide_filename, corrections_filename, rescan_done_at"
                       " FROM jobs WHERE id = ?", (str(job.job_id),)).fetchone()
    conn.close()
    assert tuple(row) == ("g.json", "c.json", None)


def test_completed_rescan_opens_no_transaction(lane_db, uploads, lifecycle, clock, monkeypatch):
    directory = _failed_job("done")
    calls = []
    real = deletion.queue_failed_outputs
    monkeypatch.setattr(deletion, "queue_failed_outputs",
                        lambda conn, job_id, *a, **k: (calls.append(job_id), real(conn, job_id, *a, **k)))
    _sweep(clock, uploads, lifecycle, days=8)
    assert calls == [directory.name] and _rescan_done(directory.name)
    assert not list(directory.rglob("*.png"))
    calls.clear()
    connects = []
    real_connect = db.connect
    monkeypatch.setattr(db, "connect", lambda *a, **k: (connects.append(1), real_connect(*a, **k))[1])
    before = len(connects)
    deletion._rescan_failed_jobs(clock())
    assert calls == [] and len(connects) - before == 1  # the one selecting query only


def test_rescan_batches_cover_every_job(lane_db, uploads, lifecycle, clock, monkeypatch):
    monkeypatch.setattr(deletion, "failed_rescan_batch", lambda: 2)
    directories = [_failed_job(f"batch{i}") for i in range(5)]
    _sweep(clock, uploads, lifecycle, days=8)
    assert sum(_rescan_done(d.name) for d in directories) == 2
    for _ in range(2):
        _sweep(clock, uploads, lifecycle, minutes=5)
    assert all(_rescan_done(d.name) for d in directories)
    assert not any(list(d.rglob("*.png")) for d in directories)


def test_stuck_job_does_not_starve_the_batch(lane_db, uploads, lifecycle, clock, monkeypatch):
    monkeypatch.setattr(deletion, "failed_rescan_batch", lambda: 1)
    stuck = _failed_job("stuck")
    later = _failed_job("later")
    real = deletion.queue_guide_images

    def refuse(conn, job_id, *a, **k):
        if job_id == stuck.name:
            raise OSError("listing failed")
        return real(conn, job_id, *a, **k)

    monkeypatch.setattr(deletion, "queue_guide_images", refuse)
    for minutes in (0, 5, 5):
        _sweep(clock, uploads, lifecycle, days=8 if minutes == 0 else 0, minutes=minutes)
    assert _rescan_done(later.name) and not _rescan_done(stuck.name)
    assert list(stuck.rglob("*.png")) and not list(later.rglob("*.png"))


def test_rescan_batch_config_fails_loud(tmp_path):
    from app.retention_config import RetentionConfigError, failed_rescan_batch

    assert failed_rescan_batch() >= 1
    kinds = 'guide_lifetime_kinds = ["provisional"]\n'
    for body in (kinds, kinds + "failed_rescan_batch = 0\n", kinds + 'failed_rescan_batch = "9"\n',
                 kinds + "failed_rescan_batch = true\n", kinds + "failed_rescan_batch_size = 9\n"):
        bad = tmp_path / "bad.toml"
        bad.write_text(body)
        with pytest.raises(RetentionConfigError):
            failed_rescan_batch(bad)
