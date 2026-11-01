"""Durable, short-lived bearer links for in-memory guide downloads."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from pathlib import Path

from app import db


def create(guide_id: str, *, hours: int, path: Path | str, now: datetime | None = None) -> str:
    moment = db.utcnow() if now is None else now
    token = secrets.token_urlsafe(32)
    with db.connect(path) as conn:
        conn.execute(
            "INSERT INTO export_links (token, guide_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, guide_id, db.format_dt(moment), db.format_dt(moment + timedelta(hours=hours))),
        )
    return token


def get_or_create(guide_id: str, *, hours: int, path: Path | str,
                  now: datetime | None = None) -> str:
    """Reuse a live token; serialize lookup and insertion across workers."""
    moment = db.utcnow() if now is None else now
    with db.connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cancelled = conn.execute(
            "SELECT cancel_requested_at FROM uploads WHERE job_id = ?", (guide_id,)
        ).fetchone()
        if cancelled is not None and cancelled["cancel_requested_at"] is not None:
            raise ValueError("guide is cancelling")
        row = conn.execute(
            "SELECT token FROM export_links WHERE guide_id = ? AND expires_at > ?"
            " ORDER BY expires_at DESC LIMIT 1", (guide_id, db.format_dt(moment))
        ).fetchone()
        if row is not None:
            return row["token"]
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO export_links (token, guide_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, guide_id, db.format_dt(moment), db.format_dt(moment + timedelta(hours=hours))),
        )
        return token


def resolve(token: str, guide_id: str, *, path: Path | str, now: datetime | None = None) -> bool:
    moment = db.utcnow() if now is None else now
    with db.connect(path) as conn:
        return conn.execute(
            "SELECT 1 FROM export_links WHERE token = ? AND guide_id = ? AND expires_at > ?",
            (token, guide_id, db.format_dt(moment)),
        ).fetchone() is not None


def purge_expired(now: datetime, *, path: Path | str | None = None) -> int:
    with db.connect(path) as conn:
        return conn.execute(
            "DELETE FROM export_links WHERE expires_at <= ?", (db.format_dt(now),)
        ).rowcount


def delete_guide(guide_id: str, *, path: Path | str | None = None) -> int:
    with db.connect(path) as conn:
        return conn.execute("DELETE FROM export_links WHERE guide_id = ?", (guide_id,)).rowcount
