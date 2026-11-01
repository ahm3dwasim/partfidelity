"""Calls evidence page: ``GET /calls/{job_id}``.

Renders this job's, chunk children's and extract retry children's model-call receipts with observed run count, median,
and worst latency plus per-call cost/status. The natural sample read-back row
comes first; when none qualifies, the page shows no read-back row rather than
fabricating one. ``/calls/sample`` and ``/measured-example``
install the sample if needed, then redirect to its receipts page.

Reads go through the frozen ``list_calls(ctx)`` interface only. The page
resolves its job context through the :func:`resolve_job_context` dependency
(registry-backed by default; deployments override it with the ledger context).

Other pages link here through :func:`calls_link_html`, labelled
"How this guide was made". Rendering uses the stdlib only (no template engine).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.admission import job_dir
from app import db, proof_links, service
from app.brand_config import fill
from app.contracts import RECEIPT_MODEL_SPARK_OPEN, RECEIPT_STAGE_READBACK, JobContext, JobId
from app.evidence import (
    CallSummary,
    ReadbackRow,
    escape_cell,
    list_calls,
    select_sample_readback,
    summarize_calls,
)
from app.ledger import JobContextImpl

router = APIRouter()

CALLS_LINK_LABEL = "How this guide was made"
CALLS_PAGE_TITLE = "Calls"

_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "calls.html"


def calls_url(job_id: str) -> str:
    """Return this job's evidence-page URL."""
    return f"/calls/{job_id}"


def calls_link_html(job_id: str) -> str:
    """Return the anchor other pages embed to reach this job's receipts."""
    return (
        f'<a class="calls-link" href="{escape_cell(calls_url(job_id))}">'
        f"{escape_cell(CALLS_LINK_LABEL)}</a>"
    )


def _money(amount) -> str:
    return f"${amount:.4f}"


def _render_summary(summary: CallSummary) -> str:
    if summary.run_count == 0:
        return ""
    median = f"{summary.median_latency_ms} ms" if summary.median_latency_ms is not None else "—"
    worst = f"{summary.worst_latency_ms} ms" if summary.worst_latency_ms is not None else "—"
    return (
        '<section class="summary" aria-label="Observed runs">'
        f'<div><span class="num">{summary.run_count}</span>'
        f'<span class="cap">observed run{"s" if summary.run_count != 1 else ""}</span></div>'
        f'<div><span class="num">{escape_cell(median)}</span>'
        '<span class="cap">median latency</span></div>'
        f'<div><span class="num">{escape_cell(worst)}</span>'
        '<span class="cap">worst latency</span></div>'
        f'<div><span class="num">{escape_cell(_money(summary.total_cost_usd))}</span>'
        '<span class="cap">estimated cost</span></div>'
        "</section>"
    )


def _render_readback(row: ReadbackRow | None) -> str:
    if row is None:
        return (
            '<section class="readback fallback" aria-label="Sample read-back">'
            "<h2>Sample read-back</h2>"
            "<p>No measured read-back for this run: none was recorded, so none is shown.</p>"
            "</section>"
        )
    badge_class = "badge-ok" if row.decision == "yes" else "badge-not"
    return (
        '<section class="readback" aria-label="Sample read-back">'
        "<h2>Sample read-back</h2>"
        '<div class="table-wrap"><table class="readback-row"><thead><tr>'
        "<th>Phrase</th><th>Read-back</th><th>Latency</th><th>Cost</th><th>Shown as</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{escape_cell(row.phrase)}</td>"
        f"<td>{escape_cell(row.decision)}</td>"
        f"<td>{row.latency_ms} ms</td>"
        f"<td>{escape_cell(_money(row.cost_usd))}</td>"
        f'<td><span class="badge {badge_class}">{escape_cell(row.badge)}</span></td>'
        "</tr></tbody></table></div>"
        "</section>"
    )


SECOND_CHECK_LABEL = "Second check (Meta open weights, Spark Open)"


def _render_second_check(calls) -> str:
    """One line per part the background Spark Open check covered; "" when it never ran."""
    items = []
    for c in calls:
        if c.model != RECEIPT_MODEL_SPARK_OPEN or c.stage != RECEIPT_STAGE_READBACK:
            continue
        phrase = escape_cell(c.evidence.get("phrase", "part"))
        decision = c.evidence.get("decision") if c.status == "succeeded" else None
        if decision == "yes":
            verdict, css = "agrees", "badge-ok"
        elif decision == "no":
            verdict, css = "disagrees (please check this label yourself)", "badge-not"
        else:
            why = c.evidence.get("reason")
            verdict, css = "not available" + (f": {escape_cell(why)}" if isinstance(why, str) else ""), "badge-not"
        items.append(
            f"<li>{phrase}: {SECOND_CHECK_LABEL}: "
            f'<span class="badge {css}">{verdict}</span></li>'
        )
    if not items:
        return ""
    return (
        '<section class="second-check" aria-label="Second check">'
        "<h2>Second check</h2><ul>" + "".join(items) + "</ul></section>"
    )


def _render_table(summary: CallSummary, rows_html: str) -> str:
    if summary.run_count == 0:
        return (
            '<section class="calls" aria-label="Model calls">'
            "<h2>Model calls</h2>"
            "<p>No model calls recorded for this job yet.</p>"
            "</section>"
        )
    return (
        '<section class="calls" aria-label="Model calls">'
        "<h2>Model calls</h2>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Stage</th><th>Model</th><th>Endpoint</th><th>Latency</th>"
        "<th>Units in/out</th><th>Cost</th><th>Status</th>"
        "</tr></thead><tbody>"
        f"{rows_html}"
        "</tbody></table></div>"
        "</section>"
    )


def _render_call_row(
    stage: str,
    model: str,
    endpoint: str,
    latency_ms: int,
    units_in: int | None,
    units_out: int | None,
    cost_usd,
    status: str,
) -> str:
    units = (
        f"{units_in if units_in is not None else '—'} / "
        f"{units_out if units_out is not None else '—'}"
    )
    status_class = "st-ok" if status == "succeeded" else "st-bad"
    return (
        "<tr>"
        f"<td>{escape_cell(stage)}</td>"
        f"<td>{escape_cell(model)}</td>"
        f"<td>{escape_cell(endpoint)}</td>"
        f"<td>{latency_ms} ms</td>"
        f"<td>{escape_cell(units)}</td>"
        f"<td>{escape_cell(_money(cost_usd))}</td>"
        f'<td><span class="{status_class}">{escape_cell(status)}</span></td>'
        "</tr>"
    )


@dataclass(frozen=True)
class _RegistryJobContext:
    """Minimal read context over the module registry (no a ledger adapter)."""

    job_id: JobId


def resolve_job_context(job_id: str, request: Request) -> JobContext:
    """Page dependency: the job context whose store backs this job's reads.

    Under app settings the page resolves a database-backed ``JobContextImpl``
    from the configured paths, so persisted receipts render; without
    settings (standalone routers) it keeps the module registry context.
    Deployments and tests may still override this dependency; the page
    itself always reads through the frozen ``list_calls(ctx)`` interface.
    """
    settings = getattr(request.app.state, "settings", None)
    db_path = getattr(settings, "db_path", None)
    data_root = getattr(settings, "data_root", None)
    if db_path is not None and data_root is not None:
        return JobContextImpl(
            Path(db_path), JobId(job_id),
            job_dir(job_id, root=Path(data_root)),
        )
    return _RegistryJobContext(JobId(job_id))  # type: ignore[return-value]


def render_calls_page(job_id: str, ctx: JobContext) -> str:
    """Render the evidence page for one job id (pure string building).

    The context must belong to the URL job; a mismatch is rejected before
    anything is read, so one job's receipts can never render under another
    job's page.
    """
    if str(ctx.job_id) != job_id:
        raise ValueError("job context does not belong to this page")
    calls = list_calls(ctx)
    if isinstance(ctx, JobContextImpl):
        prefix = f"{job_id}/"
        with db.connect(ctx._db) as conn:
            children = conn.execute(
                "SELECT id, anon_id FROM jobs WHERE substr(anon_id, 1, ?) = ?"
                " AND ip_hash = anon_id"
                " ORDER BY created_at, id",
                (len(prefix), prefix),
            ).fetchall()
        for child in children:
            if re.fullmatch(
                r"(?:chunk/[0-9]+(?:/retry/[0-9]+)?|extract/retry/[0-9]+)",
                child["anon_id"][len(prefix):],
            ):
                child_id = JobId(child["id"])
                child_ctx = JobContextImpl(ctx._db, child_id, job_dir(child_id))
                calls += list_calls(child_ctx)
    summary = summarize_calls(calls)
    readback = select_sample_readback(calls)
    rows_html = "".join(
        _render_call_row(
            c.stage,
            c.model,
            c.endpoint,
            c.latency_ms,
            c.units_in,
            c.units_out,
            c.estimated_cost_usd,
            c.status,
        )
        for c in calls
    )
    template = fill(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return (
        template.replace("{{PAGE_TITLE}}", escape_cell(f"{CALLS_PAGE_TITLE} · {job_id}"))
        .replace("{{JOB_ID}}", escape_cell(job_id))
        .replace("{{SUMMARY}}", _render_summary(summary))
        .replace("{{READBACK}}", _render_readback(readback) + _render_second_check(calls))
        .replace("{{TABLE}}", _render_table(summary, rows_html))
    )


@router.get("/calls/sample", include_in_schema=False)
@router.get("/measured-example", include_in_schema=False)
def sample_calls_redirect(request: Request) -> RedirectResponse:
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:  # receipts exist only once the sample is installed
        with contextlib.suppress(service.SampleUnavailable):
            service.ensure_sample_job(settings)
    return RedirectResponse(proof_links.load()["sample_calls"], status_code=303)


@router.get("/calls/{job_id}", response_class=HTMLResponse)
def calls_page(
    job_id: str, ctx: JobContext = Depends(resolve_job_context)
) -> HTMLResponse:
    if not _JOB_ID_RE.match(job_id):
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    if str(ctx.job_id) != job_id:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return HTMLResponse(render_calls_page(job_id, ctx))
