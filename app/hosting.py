"""One-container hosting: restore, reconcile, serve, snapshot, sweep.

``build_hosted_app`` is what the Modal entry module and the local dry run
both call, so the dry run exercises the same boot path. Boot order is
fixed: restore the newest good snapshot onto local disk, build the app
(migrations), reconcile rows against files, take a first snapshot, and
only then start the snapshot and sweep loops and hand the app back.

The expiry sweep runs in this process (one container, min=max=1), never
in a second container, so there is exactly one SQLite writer.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Callable, Mapping

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app import db, persistence, sample_readback, service
from app.deploy_config import HostConfig
from app.deletion import SweepResult, sweep_expired
from app.ledger import LifecycleRepositoryImpl
from app.main import create_app
from app.providers import open_readback
from app.settings import DATA_ROOT_ENV_VAR, DB_PATH_ENV_VAR, Settings, load_settings
from app.uploads_store import UploadStoreImpl

logger = logging.getLogger(__name__)


MetaClientFactory = Callable[[Settings], Any]


def default_meta_client(settings: Settings) -> Any:
    """Same client choice as the upload routes: replay set, else production."""
    from app import service
    from app.meta_files import select_meta_files_client

    if service.replay_flag(settings):
        return service.resolve_clients(settings).meta
    return select_meta_files_client(settings)


class SweepLoop:
    """Runs ``sweep_expired`` every ``interval_s`` inside this process."""

    def __init__(
        self,
        settings: Settings,
        interval_s: float,
        *,
        client_factory: MetaClientFactory = default_meta_client,
        after_sweep: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        self._interval_s = interval_s
        self._client_factory = client_factory
        self._after_sweep = after_sweep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_result: SweepResult | None = None
        self.passes = 0
        self.errors = 0

    def run_once(self) -> SweepResult | None:
        """One pass; None when the Meta client is unavailable (retried later)."""
        path = self._settings.db_path
        try:
            client = self._client_factory(self._settings)
            result = sweep_expired(
                db.utcnow(),
                uploads=UploadStoreImpl(path),
                lifecycle=LifecycleRepositoryImpl(path),
                client=client,
            )
        except Exception as error:
            self.errors += 1
            logger.warning("sweep pass failed: %s", type(error).__name__)
            return None
        self.passes += 1
        self.last_result = result
        if result.failures:
            logger.warning("sweep pass finished with failures: %s", ",".join(result.failures))
        if self._after_sweep is not None:
            self._after_sweep()
        return result

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.run_once()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pf-sweep", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)


def build_hosted_app(
    host: HostConfig,
    *,
    commit: Callable[[], None] | None = None,
    env: Mapping[str, str] | None = None,
    client_factory: MetaClientFactory = default_meta_client,
    start_loops: bool = True,
) -> FastAPI:
    """Boot one hosted instance and return the ASGI app to serve."""
    restored = persistence.restore_latest(host.local_db, host.snapshot_dir)
    if restored.rejected:
        logger.warning("snapshot rejected: %s", ",".join(restored.rejected))
    merged = dict(os.environ if env is None else env)
    merged[DATA_ROOT_ENV_VAR] = str(host.data_root)
    merged[DB_PATH_ENV_VAR] = str(host.local_db)
    settings = load_settings(merged)
    app = create_app(settings, reconcile_interrupted=False)
    summary = persistence.reconcile(settings.db_path, settings.data_root,
                                    canonical=(settings.guide_filename, settings.pending_settlement_filename),
                                    # The corrections log and its lock live and die with the guide.
                                    guide_sidecars=(settings.corrections_filename,
                                                    os.path.splitext(settings.corrections_filename)[0] + ".lock"))
    logger.info(summary.log_line(restored.source))
    # One container: any confirmed job still active ran in the stopped
    # container's pool, so it fails honestly before the first request.
    try:
        meta_client = client_factory(settings)
    except Exception as error:
        logger.warning("interrupted jobs: Meta client unavailable: %s", type(error).__name__)
        meta_client = None
    interrupted = service.fail_interrupted_jobs(settings, meta_client=meta_client)
    if interrupted:
        logger.warning("failed %d job(s) interrupted by the last stop", interrupted)

    snapshotter = persistence.Snapshotter(
        live_db=settings.db_path,
        snap_dir=host.snapshot_dir,
        interval_s=host.snapshot_interval_s,
        settle_s=host.snapshot_settle_s,
        commit=commit,
    )
    sweeper = SweepLoop(
        settings,
        host.sweep_interval_s,
        client_factory=client_factory,
        after_sweep=snapshotter.request,
    )
    app.state.snapshotter = snapshotter
    app.state.sweeper = sweeper
    # The reconciled state is durable before the first request is served.
    if not snapshotter.snapshot_now():
        raise RuntimeError("initial snapshot or Volume commit failed")
    app.state.snapshot_max_age_s = host.snapshot_max_age_s
    persistence.install(snapshotter)

    sample_batch: list = []  # the one-shot sample batch, if started

    def _shutdown() -> None:
        app.state.job_runner.shutdown()
        # Drain the in-flight paid call (bounded by cancel_wait_s) so its receipt exists before the final snapshot.
        wait_s = open_readback.load_config().cancel_wait_s
        drained = open_readback.shutdown(wait_s)
        for batch in sample_batch:
            drained = batch.join(wait_s) and drained
        if not drained and sample_batch:
            try:  # a call outlived the wait: say so in the receipts before the final snapshot
                sample_readback.mark_interrupted(settings)
            except Exception as error:
                logger.warning("sample read-back interrupt receipt failed: %s", type(error).__name__)
        sweeper.stop()
        snapshotter.stop()
        persistence.install(None)

    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(scope_app: FastAPI):
        async with inner(scope_app) as state:
            try:
                yield state
            finally:
                await run_in_threadpool(_shutdown)

    app.router.lifespan_context = lifespan
    if start_loops:
        snapshotter.start()
        sweeper.start()
        if settings.providers.mode == "live":
            try:  # one-shot, idempotent; never blocks or fails start-up
                batch = sample_readback.run_once(settings, snapshot=snapshotter.snapshot_now)
                if batch is not None:
                    sample_batch.append(batch)
                    threading.Thread(target=lambda: sample_readback.persist_when_done(batch, snapshotter),
                                     name="sample-readback-snap", daemon=True).start()
            except Exception as error:
                logger.warning("sample read-back not started: %s", type(error).__name__)
    return app
