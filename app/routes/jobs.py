"""Job status endpoint: ``GET /jobs/{job_id}`` (integration-owned).

Reports one job's plain ledger state, its current parent stage and
links to its guide (when a stored pipeline guide exists) and its
model-call receipts. Reads the jobs table through the app settings
database; unknown ids 404. A browser (``Accept: text/html``) gets a
plain status page instead: it reloads itself while the job runs,
offers cancel, redirects to the guide when done, and says plainly
when the job failed. Only constant copy is shown, never a raw error.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from app.brand_templates import make_env

from app import db, wait_view
from app.hazards import refusal_copy
from app.pipeline import (
    guide_path, read_pending_settlement, refusal_text_for_job, stage_for_job,
)
from app.settings import DEFAULT_JOB_STATUS_REFRESH_S
from app.wait_config import config as wait_config

router = APIRouter()

_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

STATE_COPY = {
    "sending": "Sending your video… This page refreshes on its own.",
    "waiting": "Your job is waiting for a free slot.",
    "active": "Building your guide. This page refreshes on its own.",
    "failed": "This guide could not be built. Nothing more will run for this upload.",
    "expired": "This job ran out of time before its guide was built.",
    "pending_settlement": "This job stopped; its final accounting is still settling.",
}
DONE_WITHOUT_GUIDE_COPY = "This job finished, but its guide is not available."
RUNNING_STATES = ("waiting", "active", "sending")


def _settings(request: Request):
    return getattr(request.app.state, "settings", None)


def _db_path(request: Request) -> Path:
    settings = _settings(request)
    if settings is not None and getattr(settings, "db_path", None) is not None:
        return Path(settings.db_path)
    return Path("var") / "partfidelity.sqlite3"


def _data_root(request: Request) -> Path:
    settings = _settings(request)
    if settings is not None and getattr(settings, "data_root", None) is not None:
        return Path(settings.data_root)
    return Path("var")


def _job_state(database: Path, job_id: str) -> str | None:
    try:
        conn = db.connect(database)
    except Exception:
        return None
    try:
        try:
            row = conn.execute(
                "SELECT state FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return str(row["state"])
    finally:
        conn.close()


def _latest_stage(database: Path, job_id: str) -> str | None:
    """The stage of the job's newest model-call receipt, if any."""
    try:
        conn = db.connect(database)
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT stage FROM model_calls WHERE job_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (job_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return None if row is None else str(row["stage"])


def _sending(database: Path, job_id: str) -> bool:
    with db.connect(database) as conn:
        row = conn.execute(
            "SELECT 1 FROM uploads WHERE job_id = ? AND state = 'confirming'"
            " AND hidden = 0 LIMIT 1", (job_id,),
        ).fetchone()
    return row is not None


_env = make_env(Path(__file__).resolve().parents[2] / "templates")
_NUMERALS = ("I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.")
_STATUS_WORDS = {
    "done": "Done", "working": "Working", "queued": "In queue", "waiting": "Waiting",
    "stopped": "Stopped",
}


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "0:00"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _status_page(
    job_id: str, state: str, stage: str | None, refresh_s: int, reason: str | None = None,
    view: wait_view.WaitView | None = None,
) -> str:
    message = STATE_COPY.get(state, state)
    if state in ("failed", "pending_settlement") and reason:
        message = reason
    if state == "done":
        message = DONE_WITHOUT_GUIDE_COPY
    running = state in RUNNING_STATES
    page = wait_config()["page"]
    if view is None:
        view = wait_view.WaitView((), None, None, None, 0, 0, ())
    if state == "done":
        title = page["done_title"]
    elif running:
        title = page["title"]
    else:
        title = page["stopped_title"]
    return _env.get_template("wait.html").render(
        job_id=job_id, state=state, stage=stage if running else None,
        refresh_s=int(refresh_s), running=running, message=message,
        title=title, lede=page["lede"], view=view,
        clock_label=(
            "Waiting for" if state == "waiting" else "Running for" if running else "Ran for"
        ),
        elapsed_text=_clock(view.elapsed_s),
        durations=[_clock(s.took_s) for s in view.stages],
        numerals=_NUMERALS, status_words=_STATUS_WORDS,
    )


@router.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> Response:
    """Return one job's state and its guide/receipt links (404 when unknown)."""
    wants_html = "text/html" in request.headers.get("accept", "")
    if not _JOB_ID_RE.match(job_id):
        return JSONResponse({"detail": "job not found"}, status_code=404)
    database = _db_path(request)
    state = _job_state(database, job_id)
    if state is None:
        return JSONResponse({"detail": "job not found"}, status_code=404)
    settings = _settings(request)
    guide_filename = getattr(settings, "guide_filename", None)
    pending_filename = getattr(settings, "pending_settlement_filename", None)
    data_root = _data_root(request)
    if read_pending_settlement(
        job_id, data_root=data_root, filename=pending_filename
    ) is not None:
        # The terminal transition never confirmed: report the pending
        # settlement, never the still-active row beneath it.
        state = "pending_settlement"
    if state in ("waiting", "active") and _sending(database, job_id):
        state = "sending"
    guide_url = None
    if state == "done" and guide_path(
        job_id, data_root=data_root, guide_filename=guide_filename
    ).is_file():
        guide_url = f"/guides/{job_id}"
    stage = stage_for_job(job_id, data_root=data_root) or _latest_stage(database, job_id)
    reason = (
        refusal_text_for_job(job_id, data_root=data_root)
        if state in ("failed", "expired", "pending_settlement") else None
    )
    if state == "failed" and reason is None:
        with db.connect(database) as conn:
            cancelled = conn.execute(
                "SELECT cancel_requested_at FROM uploads WHERE job_id = ?", (job_id,)
            ).fetchone()
        if cancelled is not None and cancelled["cancel_requested_at"] is not None:
            reason = refusal_copy("cancelled")
    if wants_html:
        if guide_url is not None:
            return RedirectResponse(guide_url, status_code=303)
        refresh_s = getattr(
            settings, "job_status_refresh_s", DEFAULT_JOB_STATUS_REFRESH_S
        )
        marker = stage_for_job(job_id, data_root=data_root)
        view = wait_view.build(
            job_id, state, marker, database=database, data_root=data_root,
            now=datetime.now(timezone.utc),
        )
        return HTMLResponse(_status_page(job_id, state, stage, refresh_s, reason, view))
    return JSONResponse(
        {
            "job_id": job_id,
            "state": state,
            "stage": stage,
            "reason": reason,
            "guide_url": guide_url,
            "calls_url": f"/calls/{job_id}",
        }
    )


__all__ = ("job_status", "router")
