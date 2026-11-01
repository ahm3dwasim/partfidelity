"""One-container hosting: config, boot, restart-after-crash, 24 h re-proof, dry run."""

from __future__ import annotations

import http.server
import logging
import os
import signal
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from app import db, guide_clip, persistence
from app.admission import admit_job, job_dir
from app.contracts import UploadPreview
from app.deletion import cancel_and_delete
from app.deploy_config import HostConfig, load_deploy_config
from app.hosting import SweepLoop, build_hosted_app
from app.ledger import LifecycleRepositoryImpl
from app.pipeline import save_guide
from app.settings import load_settings
from app.uploads import confirm_and_send
from app.uploads_store import UploadStoreImpl
import monitor_sample
from test_guide import _step
from test_sweeper import FakeClock, FakeMetaFiles

REPLAY_ENV = {"PARTFIDELITY_PIPELINE_MODE": "replay"}


def test_sample_with_read_only_code_root(tmp_path, synthetic_sample_bundle):
    code = tmp_path / "code"
    code.mkdir()
    modal = load_deploy_config(ROOT / "deploy" / "modal_deploy.toml").modal
    for folder in modal.image_folders:
        shutil.copytree(ROOT / folder, code / folder, ignore=shutil.ignore_patterns("__pycache__"))
    for name in modal.image_files:
        if (ROOT / name).is_file():  # the private sample media is gitignored; the synthetic bundle stands in for it
            (code / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, code / name)
    shutil.copytree(synthetic_sample_bundle, code / "app/sample_real", dirs_exist_ok=True)
    for item in sorted(code.rglob("*"), reverse=True):
        item.chmod(0o555 if item.is_dir() else 0o444)
    code.chmod(0o555)
    script = r'''
import os
import socket
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import create_app
from app.retention_config import guide_lifetime_kinds

assert "provisional_render" in guide_lifetime_kinds()
try:
    Path('write-probe').write_text('test')
except PermissionError:
    pass
else:
    raise AssertionError('code root is writable')
app = create_app()
def no_network(*args, **kwargs):
    raise AssertionError("network forbidden")
socket.create_connection = no_network
real_socket = socket.socket
def guarded_socket(family=socket.AF_INET, *args, **kwargs):
    if family != socket.AF_UNIX:  # the test client's portal needs socketpairs
        raise AssertionError("network forbidden")
    return real_socket(family, *args, **kwargs)
socket.socket = guarded_socket
response = TestClient(app).get("/guide/sample", follow_redirects=False)
assert response.status_code in (302, 303, 307), (response.status_code, response.text)
assert list((Path(os.environ['PARTFIDELITY_DATA_ROOT']) / 'jobs').rglob('step-06-part-1.png'))
'''
    child_env = {
        "PATH": str(Path(shutil.which("ffmpeg")).parent) + ":/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(code),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PARTFIDELITY_PRODUCTION": "1",
        "PARTFIDELITY_DATA_ROOT": str(tmp_path / "data"),
        "PARTFIDELITY_DB_PATH": str(tmp_path / "local.sqlite3"),
        "PARTFIDELITY_PIPELINE_MODE": "replay",
        "META_API_KEY": "test-only",
        "META_FILES_BASE_URL": "https://example.invalid",
    }
    try:
        result = subprocess.run([sys.executable, "-c", script], cwd=code, env=child_env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        code.chmod(0o755)
        for item in code.rglob("*"):
            item.chmod(0o755 if item.is_dir() else 0o644)


@pytest.fixture(autouse=True)
def _no_installed_snapshotter():
    yield
    persistence.install(None)


def _host(tmp_path: Path, name: str = "host", **overrides) -> HostConfig:
    base = tmp_path / name
    values = dict(
        data_root=tmp_path / "vol" / "data",
        snapshot_dir=tmp_path / "vol" / "snapshots",
        local_db=base / "local" / "partfidelity.sqlite3",
        snapshot_interval_s=3600.0,
        snapshot_settle_s=0.05,
        sweep_interval_s=3600.0,
        snapshot_max_age_s=7200.0,
    )
    values.update(overrides)
    return HostConfig(**values)


def _boot(host: HostConfig, **kwargs):
    return build_hosted_app(host, env=REPLAY_ENV, start_loops=False, **kwargs)


def _job_with_preview(db_path: Path, tag: str, *, payload: bytes = b"preview"):
    ctx = admit_job(f"anon-{tag}", f"ip-{tag}", f"video-{tag}")
    root = job_dir(str(ctx.job_id), root=db.data_root())
    root.mkdir(parents=True, exist_ok=True)
    target = root / "preview.mp4"
    target.write_bytes(payload)
    artifact_id = ctx.artifacts.register("upload_preview", target)
    preview = UploadPreview(
        job_id=ctx.job_id, local_video=target, duration_s=12.5,
        size_bytes=len(payload), audio_removed=True, confirmed_at=None,
    )
    UploadStoreImpl(db_path).register_preview(
        ctx, preview, artifact_id, db.utcnow() + timedelta(hours=24)
    )
    return ctx, target


def _job_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("SELECT id FROM jobs")}
    finally:
        conn.close()


# --- config -----------------------------------------------------------------

def test_shipped_deploy_config_loads_and_is_single_writer_sane():
    cfg = load_deploy_config(ROOT / "deploy" / "modal_deploy.toml")
    assert cfg.modal.secret_name and cfg.modal.volume_name
    # The live database must not sit under the Volume mount.
    assert not str(cfg.host.local_db).startswith("/vol")
    assert str(cfg.host.data_root).startswith("/vol")
    assert str(cfg.host.snapshot_dir).startswith("/vol")
    assert cfg.host.snapshot_interval_s <= 300


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t.replace("sweep_interval_s", "sweep_intervals"), "missing"),
        (lambda t: t.replace("[host]", "[host]\nsurprise = 1"), "unknown keys"),
        (lambda t: t.replace("snapshot_interval_s = 60", "snapshot_interval_s = 0"), "positive"),
        (lambda t: t.replace('secret_name = "partfidelity-meta"', 'secret_name = ""'), "non-empty"),
        (lambda t: t.replace("[modal]", "[modl]"), r"\[modal\]"),
    ],
)
def test_deploy_config_fails_loudly(tmp_path, edit, message):
    text = (ROOT / "deploy" / "modal_deploy.toml").read_text()
    bad = tmp_path / "bad.toml"
    bad.write_text(edit(text))
    with pytest.raises(RuntimeError, match=message):
        load_deploy_config(bad)


def test_db_path_setting_overrides_the_data_root_default(tmp_path):
    settings = load_settings({
        "PARTFIDELITY_DATA_ROOT": str(tmp_path / "vol"),
        "PARTFIDELITY_DB_PATH": str(tmp_path / "local.sqlite3"),
    })
    assert settings.db_path == tmp_path / "local.sqlite3"
    assert settings.data_root == tmp_path / "vol"
    default = load_settings({"PARTFIDELITY_DATA_ROOT": str(tmp_path / "vol")})
    assert default.db_path == tmp_path / "vol" / "partfidelity.sqlite3"


# --- boot and crash recovery --------------------------------------------------

def test_boot_serves_keeps_db_off_the_volume_and_snapshots_first(tmp_path):
    host = _host(tmp_path)
    app = _boot(host)
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
    assert host.local_db.is_file()
    assert not list(host.data_root.rglob("*.sqlite3*"))
    assert (host.snapshot_dir / "db.snapshot").is_file()


def test_graceful_shutdown_takes_a_final_snapshot(tmp_path):
    host = _host(tmp_path)
    app = _boot(host)
    with TestClient(app):
        ctx, _ = _job_with_preview(host.local_db, "late")
    snap = host.snapshot_dir / "db.snapshot"
    assert str(ctx.job_id) in _job_ids(snap)


def test_crash_restart_restores_rows_and_drops_unsnapshotted_work(tmp_path, caplog):
    host_a = _host(tmp_path, "a")
    app_a = _boot(host_a)
    ctx_kept, preview_kept = _job_with_preview(host_a.local_db, "kept")
    app_a.state.snapshotter.snapshot_now()
    ctx_lost, preview_lost = _job_with_preview(host_a.local_db, "lost")
    # SIGKILL: no shutdown handler runs and the local disk is gone.
    persistence.install(None)

    with caplog.at_level(logging.INFO, logger="app.hosting"):
        host_b = _host(tmp_path, "b")
        _boot(host_b)
    assert _job_ids(host_b.local_db) == {str(ctx_kept.job_id)}
    assert preview_kept.is_file()
    assert not preview_lost.parent.exists()
    assert (
        "reconcile: restored=snapshot expired_rows=0 orphan_dirs=1 temp_files=0"
        in caplog.text
    )


@pytest.mark.parametrize("orphan_kind", ["job", "file"])
def test_boot_aborts_when_orphan_removal_fails(tmp_path, monkeypatch, orphan_kind):
    host = _host(tmp_path, "a")
    app = _boot(host)
    _, preview = _job_with_preview(host.local_db, "cleanup")
    if orphan_kind == "file":
        assert app.state.snapshotter.snapshot_now()
        orphan = preview.parent / "unreferenced.txt"
        orphan.write_text("orphan")
        original = Path.unlink

        def fail_unlink(path, *args, **kwargs):
            if path == orphan:
                raise PermissionError("cleanup denied")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    else:
        orphan = preview.parent
        original = persistence.shutil.rmtree

        def fail_rmtree(path, *args, **kwargs):
            if path == orphan:
                raise PermissionError("cleanup denied")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(persistence.shutil, "rmtree", fail_rmtree)
    persistence.install(None)
    with pytest.raises(OSError, match="reconcile could not remove .*cleanup denied"):
        _boot(_host(tmp_path, "b"))
    assert orphan.exists()


def test_rowless_guide_is_hidden_and_its_directory_removed_on_boot(tmp_path):
    from app.contracts import Guide
    from app.main import create_app

    host = _host(tmp_path, "a")
    app = _boot(host)
    ctx, preview = _job_with_preview(host.local_db, "lost-guide")
    save_guide(Guide(job_id=ctx.job_id, steps=(_step(1, ()),), frame_paths=(), labels=(),
                     difficulty="Easy", verdict="TRY", draft_reasons=()))
    assert TestClient(app).get(f"/guides/{ctx.job_id}").status_code == 200
    persistence.install(None)
    host_b = _host(tmp_path, "b")
    persistence.restore_latest(host_b.local_db, host_b.snapshot_dir)
    settings = load_settings({"PARTFIDELITY_DATA_ROOT": str(host_b.data_root),
                              "PARTFIDELITY_DB_PATH": str(host_b.local_db), **REPLAY_ENV})
    rowless_app = create_app(settings)
    assert (preview.parent / settings.guide_filename).is_file()
    assert TestClient(rowless_app).get(f"/guides/{ctx.job_id}").status_code == 404
    booted = _boot(host_b)
    assert TestClient(booted).get(f"/guides/{ctx.job_id}").status_code == 404
    assert not preview.parent.exists()


def test_restart_after_file_loss_serves_the_expired_page(tmp_path):
    host_a = _host(tmp_path, "a")
    app_a = _boot(host_a)
    ctx, preview = _job_with_preview(host_a.local_db, "gone")
    app_a.state.snapshotter.snapshot_now()
    preview.unlink()  # the row was snapshotted, the file never reached the Volume
    persistence.install(None)

    app_b = _boot(_host(tmp_path, "b"))
    client = TestClient(app_b)
    assert client.get(f"/preview/{ctx.job_id}").status_code == 410


def test_restart_keeps_corrections_of_a_retained_guide_and_drops_orphan_sidecars(tmp_path):
    from app.corrections import CorrectionStore, RENAME

    host = _host(tmp_path, "a")
    app = _boot(host)
    settings = load_settings({**REPLAY_ENV, "PARTFIDELITY_DATA_ROOT": str(host.data_root),
                              "PARTFIDELITY_DB_PATH": str(host.local_db)})
    ctx, preview = _job_with_preview(host.local_db, "corr")
    job = str(ctx.job_id)
    (preview.parent / settings.guide_filename).write_text("{}")
    store = CorrectionStore(lambda _: preview.parent, settings.corrections_filename, max_ops=5)
    store.write(job, [{"op": RENAME, "step": 0, "part": 1, "phrase": "left bracket"}])
    sidecar = preview.parent / settings.corrections_filename
    before = sidecar.read_bytes()
    with store.locked(job):
        pass
    assert app.state.snapshotter.snapshot_now()
    persistence.install(None)

    _boot(host)  # the same reconcile a hosted restart runs
    assert sidecar.read_bytes() == before
    assert store.read(job) == [{"op": RENAME, "step": 0, "part": 1, "phrase": "left bracket"}]

    (preview.parent / settings.guide_filename).unlink()
    persistence.install(None)
    _boot(host)
    assert not sidecar.exists()


def test_boot_with_corrupt_snapshots_starts_empty_and_clears_orphans(tmp_path):
    host_a = _host(tmp_path, "a")
    app_a = _boot(host_a)
    ctx, preview = _job_with_preview(host_a.local_db, "x")
    app_a.state.snapshotter.snapshot_now()
    persistence.install(None)
    (host_a.snapshot_dir / "db.snapshot").write_bytes(b"junk" * 500)
    (host_a.snapshot_dir / "db.snapshot.prev").write_bytes(b"junk" * 500)

    host_b = _host(tmp_path, "b")
    _boot(host_b)
    assert _job_ids(host_b.local_db) == set()
    assert not preview.parent.exists()


def test_background_loops_snapshot_on_request_and_sweep_on_interval(tmp_path):
    host = _host(tmp_path, snapshot_settle_s=0.05, sweep_interval_s=0.1)
    app = build_hosted_app(host, env=REPLAY_ENV, start_loops=True)
    try:
        sweeper = app.state.sweeper
        deadline = time.monotonic() + 15
        while sweeper.passes < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sweeper.passes >= 2
        ctx, _ = _job_with_preview(host.local_db, "req")
        persistence.request_snapshot()
        deadline = time.monotonic() + 15
        snap = host.snapshot_dir / "db.snapshot"
        while time.monotonic() < deadline:
            if str(ctx.job_id) in _job_ids(snap):
                break
            time.sleep(0.05)
        assert str(ctx.job_id) in _job_ids(snap)
    finally:
        with TestClient(app):
            pass  # runs the shutdown handler: loops stop, final snapshot


# --- 24 h re-proof through the in-process sweep --------------------------------

def test_in_process_sweep_deletes_everything_at_24h_but_keeps_a_published_guide(
    tmp_path, monkeypatch
):
    clock = FakeClock(datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(db, "utcnow", clock)
    host = _host(tmp_path)
    app = _boot(host)
    settings = app.state.settings
    uploads = UploadStoreImpl(settings.db_path, clock=clock)
    fake = FakeMetaFiles(clock)

    # A published guide: keeps its clip until deleted.
    ctx_g, preview_g = _job_with_preview(settings.db_path, "guide")
    assert guide_clip.retain_guide_clip(ctx_g, uploads) is True
    from app.contracts import Guide
    save_guide(Guide(
        job_id=ctx_g.job_id, steps=(_step(1, ()),), frame_paths=(), labels=(),
        difficulty="Easy", verdict="TRY", draft_reasons=(),
    ))
    ctx_g.finish("done")
    clip_g = preview_g.parent / "guide_clip.mp4"
    guide_json = preview_g.parent / "guide.json"
    assert clip_g.is_file() and guide_json.is_file()


    # Preview of a guide-less job.
    ctx_p, preview_p = _job_with_preview(settings.db_path, "plain")
    # Uploaded to Meta.
    ctx_u, preview_u = _job_with_preview(settings.db_path, "meta")
    meta_id = confirm_and_send(ctx_u.job_id, uploads=uploads, client=fake)
    assert meta_id in fake.files
    # Crashed-upload staging leftover (raw footage).
    staging = settings.data_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    raw = staging / "raw-upload.bin"
    raw.write_bytes(b"raw footage")
    stamp = (clock() - timedelta(hours=2)).timestamp()
    os.utime(raw, (stamp, stamp))
    sweeper = SweepLoop(
        settings, 3600, client_factory=lambda _s: fake,
        after_sweep=app.state.snapshotter.request,
    )
    # 24 h plus one sweep interval later.
    clock.advance(hours=25, minutes=1)
    result = sweeper.run_once()
    assert result is not None and result.failures == ()

    for gone in (preview_p, preview_u, preview_g, raw):
        assert not gone.exists(), gone
    assert meta_id in fake.delete_calls and meta_id not in fake.files
    assert clip_g.is_file() and guide_json.is_file()
    assert app.state.snapshotter._wake.is_set()  # sweep asked for a snapshot

    # Deleting the guide deletes the clip with it.
    cancel_and_delete(
        ctx_g.job_id, uploads=uploads,
        lifecycle=LifecycleRepositoryImpl(settings.db_path), client=fake,
    )
    assert not clip_g.exists()


def test_sweep_pass_with_unavailable_meta_client_is_retried_not_fatal(tmp_path):
    app = _boot(_host(tmp_path))

    def unavailable(_settings):
        raise RuntimeError("no meta endpoint")

    sweeper = SweepLoop(app.state.settings, 3600, client_factory=unavailable)
    assert sweeper.run_once() is None
    assert sweeper.errors == 1 and sweeper.passes == 0


# --- local dry run: the same app object under uvicorn -----------------------------

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_local_uvicorn_dry_run_serves_snapshots_and_passes_the_monitor(tmp_path):
    port = _free_port()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "local_hosted_run.py"),
         "--root", str(tmp_path / "dry"), "--port", str(port)],
        env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"dry run exited: {proc.stderr.read().decode()[-2000:]}")
            try:
                with urllib.request.urlopen(f"{base}/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("dry run never answered /healthz")
        state = tmp_path / "monitor.json"
        code = monitor_sample.main(
            ["--base-url", base, "--state-file", str(state)]
        )
        assert code == monitor_sample.OK
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    assert (tmp_path / "dry" / "vol" / "snapshots" / "db.snapshot").is_file()
    assert not list((tmp_path / "dry" / "vol" / "data").rglob("*.sqlite3*"))


# --- monitor ------------------------------------------------------------------------

class _Stub(http.server.BaseHTTPRequestHandler):
    healthy = True

    def do_GET(self):  # noqa: N802
        if not _Stub.healthy:
            self.send_response(503)
            self.end_headers()
            return
        body = b"ok" if self.path == "/healthz" else b"<h1>Sample guide: x</h1>"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Stub.healthy = True
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_monitor_alerts_only_on_the_second_consecutive_miss(tmp_path, stub_server, capsys):
    state = tmp_path / "state.json"
    args = ["--base-url", stub_server, "--state-file", str(state)]
    assert monitor_sample.main(args) == monitor_sample.OK
    _Stub.healthy = False
    assert monitor_sample.main(args) == monitor_sample.MISS
    assert monitor_sample.main(args) == monitor_sample.ALERT
    assert monitor_sample.main(args) == monitor_sample.ALERT
    _Stub.healthy = True
    assert monitor_sample.main(args) == monitor_sample.OK
    _Stub.healthy = False
    assert monitor_sample.main(args) == monitor_sample.MISS  # count was reset
    assert "reason=healthz answered 503" in capsys.readouterr().out


def test_monitor_flags_a_sample_page_missing_its_marker(tmp_path, stub_server):
    args = ["--base-url", stub_server, "--state-file", str(tmp_path / "s.json"),
            "--expect", "never on the page"]
    assert monitor_sample.main(args) == monitor_sample.MISS


def test_monitor_treats_an_unreachable_host_as_a_miss(tmp_path):
    port = _free_port()
    args = ["--base-url", f"http://127.0.0.1:{port}", "--timeout", "1",
            "--state-file", str(tmp_path / "s.json")]
    assert monitor_sample.main(args) == monitor_sample.MISS

@pytest.mark.parametrize("key", ["timeout_s", "scaledown_window_s", "max_inputs"])
@pytest.mark.parametrize("value", ["0", "-1", "true", "0.5", "1.0"])
def test_modal_integer_config_rejects_invalid_values(tmp_path, key, value):
    import re
    text = (ROOT / "deploy" / "modal_deploy.toml").read_text()
    text = re.sub(rf"(?m)^{key} = [^ #]+", f"{key} = {value}", text)
    path = tmp_path / "deploy.toml"
    path.write_text(text)
    with pytest.raises(RuntimeError):
        load_deploy_config(path)


@pytest.mark.parametrize("key", ["timeout_s", "scaledown_window_s", "max_inputs"])
def test_modal_integer_config_accepts_one(tmp_path, key):
    import re
    text = (ROOT / "deploy" / "modal_deploy.toml").read_text()
    text = re.sub(rf"(?m)^{key} = [^ #]+", f"{key} = 1", text)
    path = tmp_path / "deploy.toml"
    path.write_text(text)
    assert getattr(load_deploy_config(path).modal, key) == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_snapshot_freshness_requires_finite_limit(tmp_path, value):
    text = (ROOT / "deploy" / "modal_deploy.toml").read_text()
    path = tmp_path / "deploy.toml"
    path.write_text(text.replace("snapshot_max_age_s = 180", f"snapshot_max_age_s = {value}"))
    with pytest.raises(RuntimeError):
        load_deploy_config(path)


def test_deploy_config_rejects_unknown_section(tmp_path):
    text = (ROOT / "deploy" / "modal_deploy.toml").read_text()
    path = tmp_path / "deploy.toml"
    path.write_text(text + "\n[hots]\ntypo = 1\n")
    with pytest.raises(RuntimeError):
        load_deploy_config(path)


@pytest.mark.parametrize('db_path', ['/vol', '/vol/data/live.sqlite3'])
def test_deploy_config_rejects_volume_database(tmp_path, db_path):
    text = (ROOT / 'deploy' / 'modal_deploy.toml').read_text()
    path = tmp_path / 'deploy.toml'
    path.write_text(text.replace('/tmp/pf/partfidelity.sqlite3', db_path))
    with pytest.raises(RuntimeError):
        load_deploy_config(path)


def test_deploy_config_loads_custom_deploy_paths(tmp_path):
    text = (ROOT / 'deploy' / 'modal_deploy.toml').read_text()
    path = tmp_path / 'deploy.toml'
    path.write_text(text.replace('/srv/partfidelity', '/srv/custom').replace('/vol', '/mount'))
    config = load_deploy_config(path)
    assert config.modal.remote_root == Path('/srv/custom')
    assert config.modal.volume_mount == Path('/mount')


def test_boot_fails_when_initial_commit_fails(tmp_path):
    def fail():
        raise OSError("commit unavailable")
    with pytest.raises(RuntimeError):
        _boot(_host(tmp_path), commit=fail)


def test_healthz_reports_stale_snapshot(tmp_path):
    host = _host(tmp_path)
    app = _boot(host)
    client = TestClient(app)
    assert client.get('/healthz').status_code == 200
    app.state.snapshotter.commit = lambda: (_ for _ in ()).throw(OSError("commit failed"))
    assert not app.state.snapshotter.snapshot_now()
    app.state.snapshotter.last_ok = time.monotonic() - host.snapshot_max_age_s - 1
    assert client.get('/healthz').status_code == 503


def test_missing_clip_expires_job_and_guide_routes(tmp_path):
    from app.contracts import Guide
    host = _host(tmp_path)
    app = _boot(host)
    ctx, preview = _job_with_preview(host.local_db, 'missing-clip')
    assert guide_clip.retain_guide_clip(ctx, UploadStoreImpl(host.local_db))
    save_guide(Guide(job_id=ctx.job_id, steps=(_step(1, ()),), frame_paths=(), labels=(),
                     difficulty='Easy', verdict='TRY', draft_reasons=()))
    ctx.finish('done')
    assert TestClient(app).get(f'/guides/{ctx.job_id}').status_code == 200
    assert app.state.snapshotter.snapshot_now()
    (preview.parent / 'guide_clip.mp4').unlink()
    app = _boot(_host(tmp_path, 'restart'))
    conn = sqlite3.connect(app.state.settings.db_path)
    try:
        assert conn.execute('SELECT state FROM jobs WHERE id = ?', (str(ctx.job_id),)).fetchone()[0] == 'expired'
    finally:
        conn.close()
    client = TestClient(app)
    for path in (f'/guides/{ctx.job_id}', f'/guide/{ctx.job_id}', f'/guides/{ctx.job_id}/clip'):
        assert client.get(path).status_code == 404


@pytest.mark.parametrize('record', ['marker', 'pointer'])
def test_critical_json_records_fsync_before_atomic_replace(tmp_path, monkeypatch, record):
    from app import pipeline, service
    settings = load_settings({'PARTFIDELITY_DATA_ROOT': str(tmp_path),
                              'PARTFIDELITY_DB_PATH': str(tmp_path / 'db.sqlite3')})
    db.configure(settings.db_path, settings.data_root)
    db.init_db(settings.db_path)
    events = []
    original_fsync, original_replace = os.fsync, os.replace
    def fsync(fd):
        events.append('fsync')
        original_fsync(fd)
    def replace(src, dst):
        assert Path(src).parent == Path(dst).parent
        assert events and events[-1] == 'fsync'
        events.append('replace')
        original_replace(src, dst)
    monkeypatch.setattr(os, 'fsync', fsync)
    monkeypatch.setattr(os, 'replace', replace)
    if record == 'marker':
        pipeline._write_pending_marker('123', 'error', settings=settings, attempts=1, guide_saved=False)
    else:
        service._build_sample_job(settings, tmp_path / settings.sample_pointer_filename)
    # every replace (staged media, then guide, then pointer) is directly preceded by an fsync of what it publishes
    assert events.count('replace') >= (3 if record == 'pointer' else 1)
    assert events[0] == 'fsync' and events[-1] in ('fsync', 'replace')
    if record == 'marker':
        assert events == ['fsync', 'replace']
