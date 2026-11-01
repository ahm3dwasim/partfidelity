"""Snapshot-and-reconcile persistence: consistency, restore order, crash cases.

A crash is simulated by restoring an older snapshot against newer files
(rows lost, files kept) and the reverse (rows kept, files lost). Neither
may leave a half-state: a lost row's files are deleted, a lost file's row
becomes expired.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from app import db, persistence
from app.admission import admit_job, job_dir
from app.contracts import UploadPreview
from app.uploads_store import UploadStoreImpl


@pytest.fixture
def host(tmp_path: Path):
    """Volume-like dirs plus a container-local live database."""
    data_root = tmp_path / "vol" / "data"
    snap_dir = tmp_path / "vol" / "snapshots"
    live = tmp_path / "local" / "live.sqlite3"
    data_root.mkdir(parents=True)
    live.parent.mkdir()
    db.configure(live, data_root)
    db.init_db(live)
    return data_root, snap_dir, live


def _admit(tag: str):
    return admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")


def _job_with_preview(live: Path, tag: str):
    ctx = _admit(tag)
    root = job_dir(str(ctx.job_id), root=db.data_root())
    root.mkdir(parents=True, exist_ok=True)
    target = root / "preview.mp4"
    target.write_bytes(b"preview-bytes")
    artifact_id = ctx.artifacts.register("upload_preview", target)
    preview = UploadPreview(
        job_id=ctx.job_id, local_video=target, duration_s=12.5,
        size_bytes=13, audio_removed=True, confirmed_at=None,
    )
    UploadStoreImpl(live).register_preview(
        ctx, preview, artifact_id, db.utcnow() + timedelta(hours=24)
    )
    return ctx, target


def _count(path: Path, sql: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def _restore_fresh(snap_dir: Path, tmp_path: Path, name: str = "booted.sqlite3"):
    booted = tmp_path / "boot" / name
    result = persistence.restore_latest(booted, snap_dir)
    return booted, result


def test_snapshot_is_whole_file_with_no_sidecars_and_rotates(host):
    _, snap_dir, live = host
    _job_with_preview(live, "a")
    first = persistence.take_snapshot(live, snap_dir)
    assert first == snap_dir / "db.snapshot"
    assert persistence.is_good_database(first)
    assert _count(first, "SELECT COUNT(*) FROM jobs") == 1
    _job_with_preview(live, "b")
    persistence.take_snapshot(live, snap_dir)
    assert _count(snap_dir / "db.snapshot", "SELECT COUNT(*) FROM jobs") == 2
    assert _count(snap_dir / "db.snapshot.prev", "SELECT COUNT(*) FROM jobs") == 1
    # Only whole files on the Volume: no WAL/SHM/journal, no leftover temp.
    assert sorted(p.name for p in snap_dir.iterdir()) == [
        "db.snapshot", "db.snapshot.prev",
    ]


def test_snapshot_under_concurrent_writes_is_transaction_consistent(host):
    _, snap_dir, live = host
    conn = db.connect(live)
    conn.execute("CREATE TABLE pairs (n INTEGER, half INTEGER)")
    stop = threading.Event()

    def writer() -> None:
        wconn = db.connect(live)
        n = 0
        while not stop.is_set():
            n += 1
            wconn.execute("BEGIN IMMEDIATE")
            wconn.execute("INSERT INTO pairs VALUES (?, 0)", (n,))
            wconn.execute("INSERT INTO pairs VALUES (?, 1)", (n,))
            wconn.execute("COMMIT")
        wconn.close()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(15):
            snap = persistence.take_snapshot(live, snap_dir)
            sconn = sqlite3.connect(snap)
            try:
                rows = sconn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
                halves = sconn.execute(
                    "SELECT COUNT(*) FROM (SELECT n FROM pairs GROUP BY n"
                    " HAVING COUNT(*) != 2)"
                ).fetchone()[0]
            finally:
                sconn.close()
            assert halves == 0, "snapshot caught half of a transaction"
            assert rows % 2 == 0
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join()
        conn.close()


def test_failed_snapshot_keeps_previous_and_leaves_no_temp(host, tmp_path):
    _, snap_dir, live = host
    persistence.take_snapshot(live, snap_dir)
    before = (snap_dir / "db.snapshot").read_bytes()
    with pytest.raises(persistence.SnapshotError):
        persistence.take_snapshot(tmp_path / "missing" / "nope.sqlite3", snap_dir)
    assert (snap_dir / "db.snapshot").read_bytes() == before
    assert [p.name for p in snap_dir.iterdir()] == ["db.snapshot"]


def test_corrupt_current_snapshot_never_overwrites_good_prev(host):
    _, snap_dir, live = host
    _job_with_preview(live, "a")
    persistence.take_snapshot(live, snap_dir)
    persistence.take_snapshot(live, snap_dir)  # prev now holds a good copy
    good_prev = (snap_dir / "db.snapshot.prev").read_bytes()
    (snap_dir / "db.snapshot").write_bytes(b"torn-by-a-bad-disk" * 100)
    persistence.take_snapshot(live, snap_dir)
    assert (snap_dir / "db.snapshot.prev").read_bytes() == good_prev
    assert persistence.is_good_database(snap_dir / "db.snapshot")


def test_restore_prefers_snapshot_then_prev_then_empty(host, tmp_path):
    _, snap_dir, live = host
    _job_with_preview(live, "a")
    persistence.take_snapshot(live, snap_dir)
    _job_with_preview(live, "b")
    persistence.take_snapshot(live, snap_dir)

    booted, result = _restore_fresh(snap_dir, tmp_path, "one.sqlite3")
    assert (result.source, result.rejected) == ("snapshot", ())
    assert _count(booted, "SELECT COUNT(*) FROM jobs") == 2

    (snap_dir / "db.snapshot").write_bytes(b"not sqlite at all" * 64)
    booted, result = _restore_fresh(snap_dir, tmp_path, "two.sqlite3")
    assert (result.source, result.rejected) == ("prev", ("db.snapshot",))
    assert _count(booted, "SELECT COUNT(*) FROM jobs") == 1

    (snap_dir / "db.snapshot.prev").write_bytes(b"also broken" * 64)
    booted, result = _restore_fresh(snap_dir, tmp_path, "three.sqlite3")
    assert result.source == "empty"
    assert result.rejected == ("db.snapshot", "db.snapshot.prev")
    assert not booted.exists()


def test_restore_rejects_a_snapshot_that_fails_integrity_check(host, tmp_path):
    _, snap_dir, live = host
    for tag in ("a", "b", "c"):
        _job_with_preview(live, tag)
    persistence.take_snapshot(live, snap_dir)
    snap = snap_dir / "db.snapshot"
    data = bytearray(snap.read_bytes())
    # Zero a page in the middle: still opens as SQLite, fails integrity.
    for index in range(4096 * 3, 4096 * 4):
        data[index] = 0xFF
    snap.write_bytes(bytes(data))
    assert not persistence.is_good_database(snap)
    _, result = _restore_fresh(snap_dir, tmp_path)
    assert result.source == "empty"


def test_restore_never_opens_or_changes_the_volume_files(host, tmp_path):
    _, snap_dir, live = host
    persistence.take_snapshot(live, snap_dir)
    before = {p.name: p.read_bytes() for p in snap_dir.iterdir()}
    _restore_fresh(snap_dir, tmp_path)
    assert {p.name: p.read_bytes() for p in snap_dir.iterdir()} == before


def test_older_snapshot_with_newer_files_deletes_unreferenced_files(host, tmp_path):
    data_root, snap_dir, live = host
    ctx_old, preview_old = _job_with_preview(live, "old")
    persistence.take_snapshot(live, snap_dir)
    # Written after the last snapshot; the "crash" loses its row.
    ctx_new, preview_new = _job_with_preview(live, "new")
    assert preview_new.is_file()

    booted, restored = _restore_fresh(snap_dir, tmp_path)
    assert restored.source == "snapshot"
    summary = persistence.reconcile(booted, data_root)
    assert summary.orphan_dirs == 1
    assert summary.expired_rows == 0
    assert preview_old.is_file()
    assert not preview_new.parent.exists()
    assert _count(booted, "SELECT COUNT(*) FROM jobs") == 1


def test_newer_snapshot_with_missing_files_expires_the_row(host, tmp_path):
    data_root, snap_dir, live = host
    ctx, preview = _job_with_preview(live, "lost")
    persistence.take_snapshot(live, snap_dir)
    preview.unlink()  # the Volume never persisted this file

    booted, _ = _restore_fresh(snap_dir, tmp_path)
    summary = persistence.reconcile(booted, data_root)
    assert summary.expired_rows == 1
    conn = sqlite3.connect(booted)
    try:
        state, hidden = conn.execute(
            "SELECT delete_state, hidden FROM artifacts"
        ).fetchone()
        upload_hidden = conn.execute("SELECT hidden FROM uploads").fetchone()[0]
        hold = conn.execute("SELECT state FROM reservations WHERE job_id = ? AND state = 'job_hold'",
                            (str(ctx.job_id),)).fetchone()
        job_state = conn.execute("SELECT state FROM jobs WHERE id = ?", (str(ctx.job_id),)).fetchone()[0]
    finally:
        conn.close()
    assert (state, hidden, upload_hidden) == ("tombstoned", 1, 1)
    assert job_state == "expired" and hold is None


def test_expired_row_link_shows_the_expired_page(host, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.settings import load_settings

    data_root, snap_dir, live = host
    ctx, preview = _job_with_preview(live, "page")
    persistence.take_snapshot(live, snap_dir)
    preview.unlink()
    booted, _ = _restore_fresh(snap_dir, tmp_path)
    settings = load_settings({
        "PARTFIDELITY_DATA_ROOT": str(data_root),
        "PARTFIDELITY_DB_PATH": str(booted),
    })
    app = create_app(settings)
    persistence.reconcile(settings.db_path, settings.data_root)
    client = TestClient(app)
    assert client.get(f"/preview/{ctx.job_id}").status_code == 410
    assert client.get(f"/preview/{ctx.job_id}/video").status_code in (404, 410)


def test_reconcile_is_idempotent_and_keeps_referenced_files(host):
    data_root, _, live = host
    ctx, preview = _job_with_preview(live, "keep")
    guide = preview.parent / "guide.json"
    guide.write_text("{}")
    first = persistence.reconcile(live, data_root)
    second = persistence.reconcile(live, data_root)
    assert first == second == persistence.ReconcileResult(0, 0, 0)
    assert preview.is_file() and guide.is_file()


def test_reconcile_removes_interrupted_atomic_write_leftovers(host):
    data_root, _, live = host
    ctx, preview = _job_with_preview(live, "tmp")
    leftovers = [
        preview.parent / ".guide-abc123.json",
        preview.parent / "frame_1.png.tmp",
        preview.parent / ".tmp-xyz.json",
    ]
    for item in leftovers:
        item.write_text("partial")
    summary = persistence.reconcile(live, data_root)
    assert summary.temp_files == 3
    assert not any(item.exists() for item in leftovers)
    assert preview.is_file()


def test_reconcile_clears_crashed_png_publish_so_retry_succeeds(host):
    from app.render import _write_png_atomic, write_png_rgb

    data_root, _, live = host
    ctx = _admit("png")
    root = job_dir(str(ctx.job_id), root=db.data_root())
    root.mkdir(parents=True, exist_ok=True)
    good = root / "good.png"
    write_png_rgb(good, 2, 2, bytes(12))
    ctx.artifacts.register("rendered", good)
    out = root / "label.png"
    out.write_bytes(b"")
    ctx.artifacts.register("rendered", out)
    hidden = [root / ".label.png.claim", root / ".render-abc.png"]
    for item in hidden:
        item.write_bytes(b"")
    with pytest.raises(FileExistsError):
        _write_png_atomic(out, 2, 2, bytes(12))
    persistence.reconcile(live, data_root)
    assert not out.exists() and not any(i.exists() for i in hidden)
    assert good.is_file()
    _write_png_atomic(out, 2, 2, bytes(12))
    assert out.stat().st_size > 0


def test_reconcile_treats_an_escaping_path_as_missing(host):
    data_root, _, live = host
    ctx, preview = _job_with_preview(live, "esc")
    outside = data_root / "outside.bin"
    outside.write_bytes(b"x")
    conn = db.connect(live)
    conn.execute("UPDATE artifacts SET relative_path = '../../outside.bin'")
    conn.close()
    summary = persistence.reconcile(live, data_root)
    assert summary.expired_rows == 1
    assert outside.is_file()  # reconcile only reads the path, never follows it


def test_reconcile_leaves_no_symlinked_orphan_target_deleted(host, tmp_path):
    data_root, _, live = host
    target = tmp_path / "precious"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    (data_root / "jobs").mkdir(exist_ok=True)
    (data_root / "jobs" / "ghost").symlink_to(target, target_is_directory=True)
    summary = persistence.reconcile(live, data_root)
    assert summary.orphan_dirs == 1
    assert not (data_root / "jobs" / "ghost").exists()
    assert (target / "keep.txt").read_text() == "keep"


def test_snapshotter_snapshots_on_request_then_commits(host):
    _, snap_dir, live = host
    commits: list[bool] = []
    snapper = persistence.Snapshotter(
        live, snap_dir, interval_s=3600, settle_s=0.05,
        commit=lambda: commits.append((snap_dir / "db.snapshot").is_file()),
    )
    snapper.start()
    try:
        snapper.request()
        deadline = time.monotonic() + 10
        while not commits and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        snapper.stop()
    # The snapshot was already renamed into place when commit ran.
    assert commits and commits[0] is True


def test_snapshotter_survives_a_failed_pass_and_counts_it(host, tmp_path):
    _, snap_dir, _ = host
    snapper = persistence.Snapshotter(
        tmp_path / "gone.sqlite3", snap_dir, interval_s=3600, settle_s=0.01
    )
    assert snapper.snapshot_now() is False
    assert snapper.failures == 1


def test_request_snapshot_is_a_noop_without_a_snapshotter():
    persistence.install(None)
    persistence.request_snapshot()  # must not raise


def test_reconcile_removes_new_artifact_in_surviving_job(host, tmp_path):
    data_root, snap_dir, live = host
    ctx, preview = _job_with_preview(live, 'survivor')
    persistence.take_snapshot(live, snap_dir)
    newer = preview.parent / 'newer.mp4'
    newer.write_bytes(b'new')
    ctx.artifacts.register('guide_clip', newer)
    booted, _ = _restore_fresh(snap_dir, tmp_path)
    persistence.reconcile(booted, data_root)
    assert preview.exists()
    assert not newer.exists()


def test_restore_inaccessible_snapshot_falls_back_or_aborts(host, tmp_path, monkeypatch):
    _, snap_dir, live = host
    persistence.take_snapshot(live, snap_dir)
    persistence.take_snapshot(live, snap_dir)
    original_copy = persistence.shutil.copyfile
    def inaccessible(src, dst):
        if Path(src).name == persistence.SNAPSHOT_NAME:
            raise OSError('inaccessible')
        return original_copy(src, dst)
    monkeypatch.setattr(persistence.shutil, 'copyfile', inaccessible)
    booted, result = _restore_fresh(snap_dir, tmp_path)
    assert result.source == 'prev'
    previous = booted.read_bytes()
    (snap_dir / persistence.PREV_NAME).write_bytes(b'corrupt')
    with pytest.raises(persistence.SnapshotError):
        persistence.restore_latest(booted, snap_dir)
    assert booted.read_bytes() == previous


def test_stop_does_not_wait_for_snapshot_lock_when_worker_alive(host):
    _, snap_dir, live = host
    snapper = persistence.Snapshotter(live, snap_dir, 10, 1)
    class Stuck:
        def join(self, timeout):
            pass
        def is_alive(self):
            return True
    snapper._thread = Stuck()
    called = []
    snapper.snapshot_now = lambda: called.append(True)
    snapper.stop()
    assert not called


def test_reconcile_keeps_guide_frame_reference(host):
    import json
    data_root, _, live = host
    _, preview = _job_with_preview(live, 'guide-frame')
    frame = preview.parent / 'frame.png'
    frame.write_bytes(b'frame')
    (preview.parent / 'guide.json').write_text(json.dumps({'frame_paths': [str(frame)], 'labels': []}))
    stray = preview.parent / 'stray.png'
    stray.write_bytes(b'extra')
    persistence.reconcile(live, data_root)
    assert frame.exists()
    assert not stray.exists()


@pytest.mark.parametrize("failure", ["json", "oserror"])
def test_reconcile_reports_unreadable_guide_and_keeps_files(host, monkeypatch, caplog, failure):
    data_root, _, live = host
    _, preview = _job_with_preview(live, 'bad-guide')
    guide = preview.parent / 'guide.json'
    guide.write_text('{')
    frame = preview.parent / 'frame.png'
    frame.write_bytes(b'frame')
    if failure == "oserror":
        original = Path.read_text
        def read_text(self, *args, **kwargs):
            if self == guide:
                raise OSError('unreadable')
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, 'read_text', read_text)
    persistence.reconcile(live, data_root)
    assert 'reconcile could not read guide references' in caplog.text
    assert guide.exists() and frame.exists()


def test_inaccessible_artifact_aborts_reconcile_without_expiring_jobs(host, monkeypatch):
    data_root, _, live = host
    ctx, preview = _job_with_preview(live, 'io-error')
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        if path == preview:
            raise OSError('volume unavailable')
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    with pytest.raises(OSError):
        persistence.reconcile(live, data_root)
    conn = sqlite3.connect(live)
    try:
        assert conn.execute('SELECT state FROM jobs WHERE id = ?', (str(ctx.job_id),)).fetchone()[0] == 'active'
        assert conn.execute('SELECT delete_state FROM artifacts').fetchone()[0] == 'live'
    finally:
        conn.close()


def test_reconcile_removes_corrupt_body_png_with_valid_magic_and_iend(host):
    from app.render import _write_png_atomic, write_png_rgb

    data_root, _, live = host
    ctx = _admit("corrupt")
    root = job_dir(str(ctx.job_id), root=db.data_root())
    root.mkdir(parents=True, exist_ok=True)
    out = root / "label.png"
    write_png_rgb(out, 2, 2, bytes(12))
    raw = bytearray(out.read_bytes())
    raw[40] ^= 0xFF  # inside IDAT: magic, IHDR and IEND stay intact
    out.write_bytes(bytes(raw))
    ctx.artifacts.register("rendered", out)
    persistence.reconcile(live, data_root)
    assert not out.exists()
    _write_png_atomic(out, 2, 2, bytes(12))
    assert out.stat().st_size > 0
