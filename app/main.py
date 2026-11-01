"""Single-service application factory (one VM, two workers)."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import db
from app import guide_clip
from app import stage_cache
from app import service
from app.admission import job_dir
from app.body_limit import BodyLimitMiddleware
from app.corrections import CorrectionStore
from app.job_runner import JobRunner
from app.meta_files import select_meta_files_client
from app.providers import open_readback
from app.pipeline import load_guide
from app.routes import calls, catch, guides, how, jobs, site, steps, uploads
from app.routes.guides import (
    set_corrections_store,
    set_guide_clip_loader,
    set_guide_loader,
)
from app.settings import Settings, load_settings
from app.uploads import run_startup_cleanup

WORKERS = 2

ROOT = Path(__file__).resolve().parents[1]


def create_app(
    settings: Settings | None = None, *, reconcile_interrupted: bool = True
) -> FastAPI:
    open_readback.load_config()
    resolved = settings if settings is not None else load_settings()
    resolved.data_root.mkdir(parents=True, exist_ok=True)
    db.configure(resolved.db_path, resolved.data_root)
    stage_cache.configure_cache_root(resolved.data_root / "stage_cache")
    db.init_db(resolved.db_path, guide_filename=resolved.guide_filename,
               corrections_filename=resolved.corrections_filename)
    run_startup_cleanup(data_root=resolved.data_root, db_path=resolved.db_path)
    service.prepare_sample(resolved)  # before any request: an old install never serves old sample media
    if reconcile_interrupted:
        try:
            meta_client = (
                service.resolve_clients(resolved).meta if service.replay_flag(resolved)
                else select_meta_files_client(resolved)
            )
        except Exception as error:
            logging.getLogger("partfidelity.jobs").warning(
                "interrupted jobs: Meta client unavailable: %s", type(error).__name__
            )
            meta_client = None
        service.fail_interrupted_jobs(resolved, meta_client=meta_client)
    app = FastAPI(title="PartFidelity")
    app.state.settings = resolved
    app.state.job_runner = JobRunner(resolved.max_concurrent_jobs)
    app.add_middleware(BodyLimitMiddleware)
    site.add_html_cache_control(app)
    app.include_router(site.router)
    app.include_router(how.router)
    app.include_router(guides.router)
    app.include_router(calls.router)
    app.include_router(catch.router)
    app.include_router(steps.router)
    app.include_router(uploads.router)
    app.include_router(jobs.router)
    app.mount("/static", site.CachedStaticFiles(directory=ROOT / "static", html=False))
    def visible_guide(job_id: str):
        conn = db.connect(resolved.db_path)
        try:
            job = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        if job is None or job["state"] in ("failed", "expired"):
            return None
        guide = load_guide(job_id, data_root=resolved.data_root,
                           guide_filename=resolved.guide_filename)
        if guide is not None and service.sample_job_id(resolved) == job_id:
            # Existing installations retain the old sample data; use the shipped source-gap wording.
            try:
                bundle = json.loads((service._sample_bundle(resolved) / resolved.guide_filename).read_text())
                action = next(step["action"] for step in bundle["steps"] if step["number"] == 5)
                if not isinstance(action, str) or not action.strip():
                    return None
                guide = replace(guide, steps=tuple(
                    replace(step, action=action) if step.number == 5 else step for step in guide.steps
                ))
            except (OSError, ValueError, KeyError, TypeError, StopIteration):
                return None
        return guide

    set_guide_loader(
        app,
        visible_guide,
    )
    set_guide_clip_loader(
        app,
        lambda job_id: guide_clip.load_guide_clip(
            job_id, db_path=resolved.db_path,
            data_root=resolved.data_root,
        ) or service.sample_clip_path(resolved, job_id),
    )

    set_corrections_store(
        app,
        CorrectionStore(
            lambda job_id: job_dir(job_id, root=resolved.data_root),
            resolved.corrections_filename,
            shared_job=lambda: service.sample_job_id(resolved),
            max_ops=resolved.correction_max_ops,
            max_chars=resolved.correction_phrase_max_chars,
            guide_filename=resolved.guide_filename,
        ),
    )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        snapshotter = getattr(app.state, "snapshotter", None)
        if snapshotter is not None and (snapshotter.last_ok is None or
                time.monotonic() - snapshotter.last_ok > app.state.snapshot_max_age_s):
            return JSONResponse({"status": "snapshot stale"}, status_code=503)
        return JSONResponse({"status": "ok"})

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
        workers=WORKERS,
    )
