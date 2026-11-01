"""What the waiting screen shows, read only from the job's real state.

The wait screen: named stages marked Done / Working / Waiting with
the time each took, a clock that counts from the confirm (never a finish
estimate), a FRAME n / N counter of frames outlined so far, and the last
lines of the run log. Every figure comes from the jobs row, the job's
model-call receipts, its stage marker and the frames on disk; a figure the
job has not produced yet is left out, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sqlite3

from app import db
from app.admission import job_dir
from app.wait_config import config

# Staged writes look like step1_t0.00s_0.png.abcd.tmp.png; they are not frames yet.
_STAGED = re.compile(r"\.tmp\.")

TERMINAL_STATES = ("done", "failed", "expired", "pending_settlement")


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    detail: str
    status: str  # "done" | "working" | "queued" | "waiting" | "stopped"
    took_s: float | None


@dataclass(frozen=True)
class WaitView:
    stages: tuple[Stage, ...]
    started_at: str | None  # ISO time the clock counts from
    ended_at: str | None  # ISO time the clock stops at (terminal jobs)
    elapsed_s: float | None  # server-side elapsed, for the no-script page
    frames_done: int
    frames_total: int
    log: tuple[tuple[str, str, str], ...]  # (ISO time, hh:mm:ss UTC, line), newest last


def _parse(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _rows(database: Path, job_id: str):
    try:
        conn = db.connect(database)
    except Exception:
        return None, []
    try:
        job = conn.execute(
            "SELECT created_at, confirmed_at, completed_at FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        calls = conn.execute(
            "SELECT stage, status, created_at FROM model_calls WHERE job_id = ?"
            " ORDER BY created_at, id",
            (job_id,),
        ).fetchall()
    except sqlite3.Error:
        return None, []
    finally:
        conn.close()
    return job, calls


def frame_counts(job_id: str, data_root: Path) -> tuple[int, int]:
    """(frames outlined, frames picked) from the job's frame folder."""
    picked: set[str] = set()
    outlined: set[str] = set()
    try:
        folders = [p for p in job_dir(job_id, root=data_root).glob("*_frames") if p.is_dir()]
    except (OSError, ValueError):
        return 0, 0
    for folder in folders:
        for path in folder.glob("step*.png"):
            if path.name.endswith(".mask.png") or _STAGED.search(path.name):
                continue
            picked.add(path.stem)
        for path in folder.glob("step*.mask.png"):
            if _STAGED.search(path.name):
                continue
            for stem in picked:
                if path.name.startswith(stem + "_"):
                    outlined.add(stem)
    return len(outlined), len(picked)


def build(
    job_id: str, state: str, marker: str | None, *, database: Path, data_root: Path,
    now: datetime,
) -> WaitView:
    cfg = config()
    stages_cfg = cfg["stages"]
    job, calls = _rows(database, job_id)
    owner = {r: i for i, s in enumerate(stages_cfg) for r in s["receipts"]}
    by_marker = {s["marker"]: i for i, s in enumerate(stages_cfg) if s["marker"]}

    # The furthest stage the job has evidence for: its receipts or its marker.
    current = 0
    for call in calls:
        current = max(current, owner.get(str(call["stage"]), 0))
    if marker in by_marker:
        current = max(current, by_marker[marker])
    if state == "sending":
        current = 0
    if state == "waiting":
        # Queued: stages with receipts finished; the next one has no slot yet.
        receipted = [owner[str(c["stage"])] for c in calls if str(c["stage"]) in owner]
        current = min(max(receipted, default=-1) + 1, len(stages_cfg) - 1)
    if state == "done":
        current = len(stages_cfg)

    # Each finished stage ends at its last receipt; it took the time since
    # the previous stage ended (or since the confirm, for the first).
    started = _parse(job["confirmed_at"] or job["created_at"]) if job else None
    ended = _parse(job["completed_at"]) if job and state in TERMINAL_STATES else None
    last_at: dict[int, datetime] = {}
    for call in calls:
        when = _parse(call["created_at"])
        stage = owner.get(str(call["stage"]))
        if when is not None and stage is not None:
            last_at[stage] = max(last_at.get(stage, when), when)

    stopped = state in ("failed", "expired", "pending_settlement")
    stages = []
    prev = started
    for i, s in enumerate(stages_cfg):
        if i < current:
            status = "done"
            end = last_at.get(i)
            if end is None and i == len(stages_cfg) - 1:
                end = ended
            took = (end - prev).total_seconds() if end and prev and end >= prev else None
            # No end time (e.g. receipts held by child jobs): the next stage
            # has no known start either, so it gets no duration.
            prev = end
        elif i == current:
            if stopped:
                status = "stopped"
            elif state == "waiting":
                status = "queued"  # no slot yet: nothing is working on it
            else:
                status = "working"
            took = None
        else:
            status = "waiting"
            took = None
        stages.append(Stage(s["id"], s["label"], s["detail"], status, took))

    log: list[tuple[datetime, str]] = []
    confirmed = _parse(job["confirmed_at"]) if job else None
    created = _parse(job["created_at"]) if job else None
    if confirmed is not None:
        log.append((confirmed, "Clip confirmed"))
    elif created is not None:
        log.append((created, "Job created"))
    for call in calls:
        when = _parse(call["created_at"])
        if when is None:
            continue
        status = str(call["status"])
        log.append((when, f"{call['stage']} call {status}"))
    done_frames, total_frames = frame_counts(job_id, data_root)
    for i, s in enumerate(stages_cfg):
        succeeded = any(owner.get(str(c["stage"])) == i and c["status"] == "succeeded" for c in calls)
        if stages[i].status == "done" and i in last_at and succeeded and (s["id"] != "outline" or done_frames > 0):
            log.append((last_at[i], s["log_done"]))
    if state == "done" and ended is not None:
        log.append((ended, stages_cfg[-1]["log_done"]))
    log.sort(key=lambda row: row[0])
    tail = tuple(
        (when.isoformat(), when.strftime("%H:%M:%S"), line)
        for when, line in log[-cfg["page"]["log_lines"]:]
    )

    # A stopped job with no completion time stops its clock at its last
    # recorded activity, so a reload never adds time nothing spent.
    stop_at = ended
    if stop_at is None and stopped:
        stop_at = log[-1][0] if log else started
    elapsed = ((stop_at or now) - started).total_seconds() if started else None
    return WaitView(
        stages=tuple(stages),
        started_at=started.isoformat() if started else None,
        ended_at=(stop_at.isoformat() if stop_at else None),
        elapsed_s=max(0.0, elapsed) if elapsed is not None else None,
        frames_done=done_frames,
        frames_total=total_frames,
        log=tail,
    )
