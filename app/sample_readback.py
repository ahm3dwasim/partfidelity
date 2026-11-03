"""One-shot real Spark Open second check of the sample's kept fan and battery labels.

Runs inside the web container at start (single SQLite writer), through the same
``open_readback.start`` path and spend book a new job uses, so the per-job caps in
``config/open_readback.toml`` hold. It never edits or locks a label. Labels that already
have a Spark Open row are skipped, so a second run does nothing.
"""
from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.providers import open_readback

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sample_readback.toml"
logger = logging.getLogger(__name__)
PENDING_REASON = "check started, receipt not yet recorded"
INTERRUPTED_REASON = "interrupted at shutdown, not retried"


class SampleReadbackConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SampleReadbackConfig:
    enabled: bool
    labels: tuple[str, ...]


def load_config(path: Path = CONFIG) -> SampleReadbackConfig:
    with open(path, "rb") as fh:
        top = tomllib.load(fh)
    table = top.get("sample_readback")
    if set(top) != {"sample_readback"} or not isinstance(table, dict) or set(table) != {"enabled", "labels"}:
        raise SampleReadbackConfigError("expected one [sample_readback] table with enabled and labels")
    labels = table["labels"]
    if (not isinstance(table["enabled"], bool) or not isinstance(labels, list) or not labels
            or any(not isinstance(x, str) or not x.strip() for x in labels)):
        raise SampleReadbackConfigError("sample_readback: enabled must be a bool, labels a non-empty list of text")
    return SampleReadbackConfig(table["enabled"], tuple(labels))


def persist_when_done(batch, snapshotter) -> bool:
    """Wait for the whole batch, then snapshot at once (no debounce); if that fails, fall back to the debounced snapshot."""
    if not batch.join():
        return False
    if snapshotter.snapshot_now():
        return True
    logger.warning("sample read-back: end-of-run snapshot failed, requesting a debounced snapshot")
    snapshotter.request()
    return False


def pending(decisions, cfg: SampleReadbackConfig, done_labels) -> list:
    """Kept, eligible labels whose phrase is configured and that have no Spark Open row yet."""
    return [d for d in decisions
            if open_readback.eligible(d) and d.detection is not None and d.detection.phrase in cfg.labels
            and d.provisional.rendered_image_path.name not in done_labels]


def plan(decisions, cfg: SampleReadbackConfig, ocfg: open_readback.OpenReadbackConfig, done_labels=()) -> dict:
    """What a run would send: the labels, the call count and the worst-case cost (capped by the job cap)."""
    todo = pending(decisions, cfg, set(done_labels))
    calls = min(len(todo), ocfg.max_calls_per_job)
    worst = min(calls * open_readback.reservation_usd(ocfg), ocfg.max_cost_usd_per_job)
    return {"labels": [(d.provisional.rendered_image_path.name, d.detection.phrase) for d in todo],
            "calls": calls, "worst_case_usd": worst}


def run_once(settings, *, cfg: SampleReadbackConfig | None = None, ocfg=None, snapshot=None, **start_kwargs):
    """Queue the check for the sample job; returns the Batch, or None when nothing was queued.

    No job-level lock, on purpose: the read-back is off by default and turned on only for a single maintainer
    run, and the web app runs exactly one container (min_containers=max_containers=1 in deploy/modal_app.py),
    so there is one process and one SQLite writer. tests/test_sample_readback.py guards the container count.
    """
    from app import service
    from app.evidence import list_calls
    from app.ledger import JobContextImpl
    from app.admission import job_dir
    from app.contracts import JobId
    from app.pipeline import _second_check_budget, load_guide

    cfg = cfg or load_config()  # inside the caller's start-up guard: a bad file never blocks boot
    if not cfg.enabled:
        return None
    ocfg = ocfg or open_readback.load_config()
    job_id = service.sample_job_id(settings)
    guide = load_guide(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename) if job_id else None
    if guide is None:
        logger.info("sample read-back: no sample guide")
        return None
    ctx = JobContextImpl(Path(settings.db_path), JobId(job_id), job_dir(job_id, root=Path(settings.data_root)))
    done = set(open_readback.outcomes(list_calls(ctx)))
    todo = pending(guide.labels, cfg, done)
    if not todo:
        return None
    logger.info("sample read-back: %s", plan(guide.labels, cfg, ocfg, done))
    # Marker first: a restart after an interrupted call finds a row and never sends the label again.
    # The real receipt lands later and wins (same generation, later row).
    for d in todo:
        open_readback.record_unavailable(ctx, d, 0, reason=PENDING_REASON)
    # Durable before any paid call: a crash mid-call restores with the markers and never re-sends.
    if snapshot is not None and not snapshot():
        raise RuntimeError("sample read-back: marker snapshot failed, nothing sent")
    return open_readback.start(ctx, todo, budget=_second_check_budget(settings), cfg=ocfg, **start_kwargs)


def mark_interrupted(settings, *, cfg: SampleReadbackConfig | None = None) -> int:
    """After a drain that timed out: give each label still holding only its pending marker an explicit
    "interrupted, not retried" row (the marker stays, so a restart never re-sends). Returns the count."""
    from app.evidence import list_calls
    from app.ledger import JobContextImpl
    from app.admission import job_dir
    from app.contracts import JobId, RECEIPT_MODEL_SPARK_OPEN
    from app import service
    from app.pipeline import load_guide

    cfg = cfg or load_config()
    job_id = service.sample_job_id(settings)
    guide = load_guide(job_id, data_root=settings.data_root, guide_filename=settings.guide_filename) if job_id else None
    if guide is None:
        return 0
    ctx = JobContextImpl(Path(settings.db_path), JobId(job_id), job_dir(job_id, root=Path(settings.data_root)))
    real = {c.evidence.get("label") for c in list_calls(ctx)
            if c.model == RECEIPT_MODEL_SPARK_OPEN and c.evidence.get("reason") != PENDING_REASON}
    todo = pending(guide.labels, cfg, real)
    for d in todo:
        open_readback.record_unavailable(ctx, d, 0, reason=INTERRUPTED_REASON)
    return len(todo)
