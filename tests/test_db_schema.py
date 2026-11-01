"""Ledger schema ownership: WAL/busy-timeout on every connection, exact tables."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from app import db
from app.admission import admit_job

EXPECTED_COLUMNS = {
    "jobs": (
        "id",
        "state",
        "anon_id",
        "ip_hash",
        "video_hash",
        "created_at",
        "deadline_at",
        "confirmed_at",
        "completed_at",
        "cost_cap_usd",
        "readback_fired_n",
        "guide_filename",
        "corrections_filename",
        "rescan_seen_at",
        "rescan_done_at",
    ),
    "budget_rows": ("name", "hard_cap_usd", "spent_usd", "held_usd", "day_utc", "breaker_usd"),
    "reservations": (
        "id",
        "job_id",
        "budget_row",
        "call_kind",
        "estimated_usd",
        "actual_usd",
        "state",
        "created_at",
        "settled_at",
    ),
    "meta_files": (
        "id",
        "job_id",
        "expires_at",
        "delete_state",
        "delete_requested_at",
        "verified_absent_at",
        "last_error",
    ),
    "model_calls": (
        "id",
        "job_id",
        "stage",
        "model",
        "endpoint",
        "latency_ms",
        "units_in",
        "units_out",
        "estimated_cost_usd",
        "status",
        "evidence_json",
        "created_at",
    ),
    "events": ("id", "job_id", "anon_id", "session_id", "event", "detail_json", "created_at"),
    "artifacts": (
        "id",
        "job_id",
        "kind",
        "relative_path",
        "expires_at",
        "hidden",
        "kept_at",
        "delete_state",
        "deleted_at",
        "created_at",
    ),
    "uploads": (
        "job_id",
        "preview_artifact_id",
        "state",
        "preview_expires_at",
        "duration_s",
        "size_bytes",
        "audio_removed",
        "consented_at",
        "claim_id",
        "claim_started_at",
        "guard_deadline_utc",
        "meta_file_id",
        "confirmed_at",
        "cancel_requested_at",
        "hidden",
        "created_at",
        "updated_at",
    ),
    "upload_failures": (
        "failure_key",
        "job_id",
        "stage",
        "error",
        "created_at",
        "resolved_at",
    ),
    "delete_requests": (
        "id",
        "job_id",
        "contact",
        "requested_at",
        "expires_at",
    ),
}


@pytest.fixture
def lane_db(tmp_path: Path) -> Path:
    database = tmp_path / "lane.sqlite3"
    db.configure(database, tmp_path / "data")
    db.init_db(database)
    return database


def _pragmas(conn: sqlite3.Connection) -> dict[str, object]:
    return {
        "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
        "busy_timeout": conn.execute("PRAGMA busy_timeout").fetchone()[0],
        "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
    }


def test_wal_busy_timeout_on_every_connection(lane_db: Path) -> None:
    first = db.connect(lane_db)
    second = db.connect(lane_db)
    try:
        for conn in (first, second):
            settings = _pragmas(conn)
            assert settings["journal_mode"] == "wal"
            assert settings["busy_timeout"] == db.BUSY_TIMEOUT_MS
            assert settings["foreign_keys"] == 1
    finally:
        first.close()
        second.close()
    # A fresh connection after close gets the same settings, not pool residue.
    reopened = db.connect(lane_db)
    try:
        assert _pragmas(reopened) == {
            "journal_mode": "wal",
            "busy_timeout": db.BUSY_TIMEOUT_MS,
            "foreign_keys": 1,
        }
    finally:
        reopened.close()


def test_schema_tables_and_columns_match_interfaces(lane_db: Path) -> None:
    conn = db.connect(lane_db)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert set(EXPECTED_COLUMNS) <= tables
        for table, columns in EXPECTED_COLUMNS.items():
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            assert [column[1] for column in info] == list(columns), table
    finally:
        conn.close()


def test_model_calls_and_events_ids_store_text_receipt_keys(
    lane_db: Path, tmp_path: Path
) -> None:
    from app.admission import admit_job

    conn = db.connect(lane_db)
    try:
        for table in ("model_calls", "events"):
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            assert (info[0][1], info[0][2], info[0][5]) == ("id", "TEXT", 1), table
    finally:
        conn.close()
    # The uploads state CHECK rejects anything outside the six literals.
    ctx = admit_job("anon-schema", "ip-schema", "video-schema")
    target = tmp_path / "data" / "jobs" / str(ctx.job_id) / "clip.mp4"
    artifact_id = ctx.artifacts.register("upload_preview", target)
    conn = db.connect(lane_db)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            conn.execute(
                "INSERT INTO uploads (job_id, preview_artifact_id, state,"
                " preview_expires_at, duration_s, size_bytes, audio_removed,"
                " hidden, created_at, updated_at)"
                " VALUES (?, ?, 'bogus', ?, 1.0, 1, 1, 0, ?, ?)",
                (
                    str(ctx.job_id),
                    str(artifact_id),
                    db.format_dt(db.utcnow()),
                    db.format_dt(db.utcnow()),
                    db.format_dt(db.utcnow()),
                ),
            )
    finally:
        conn.close()


def test_init_db_is_idempotent_and_seeds_budget_rows(lane_db: Path) -> None:
    db.init_db(lane_db)
    conn = db.connect(lane_db)
    try:
        rows = {
            row["name"]: row
            for row in conn.execute("SELECT * FROM budget_rows").fetchall()
        }
        for name in ("spark", "sam", "readback", "aggregate"):
            assert rows[name]["spent_usd"] == "0"
            assert rows[name]["held_usd"] == "0"
            assert rows[name]["breaker_usd"] is None
        assert rows["aggregate"]["hard_cap_usd"] == "100"
    finally:
        conn.close()


def test_interrupted_rebuild_resumes_with_intact_data(tmp_path: Path) -> None:
    database = tmp_path / "interrupted.sqlite3"
    db.configure(database, tmp_path / "data")
    conn = db.connect(database)
    try:
        conn.executescript(db.MIGRATION_001.read_text(encoding="utf-8"))
        now_text = db.format_dt(db.utcnow())
        conn.execute(
            "INSERT INTO jobs (id, state, anon_id, ip_hash, video_hash,"
            " created_at, deadline_at, confirmed_at, completed_at,"
            " cost_cap_usd, readback_fired_n)"
            " VALUES ('job-1', 'active', 'a', 'i', 'v', ?, ?, NULL, NULL,"
            " '0.50', 0)",
            (now_text, now_text),
        )
        conn.execute(
            "INSERT INTO model_calls (job_id, stage, model, endpoint,"
            " latency_ms, units_in, units_out, estimated_cost_usd, status,"
            " evidence_json, created_at)"
            " VALUES ('job-1', 's', 'm', 'e', 5, 1, 2, '0.01', 'ok', '{}', ?)",
            (now_text,),
        )
        conn.execute(
            "INSERT INTO events (job_id, anon_id, session_id, event,"
            " detail_json, created_at)"
            " VALUES ('job-1', 'a', 's', 'fired', '{}', ?)",
            (now_text,),
        )
        # Simulate a crash after populating events_new: run the migration's
        # own statements in order up to that copy, then stop.
        migration = db.MIGRATION_002.read_text(encoding="utf-8")
        marker = "FROM events;"
        prefix = migration[: migration.index(marker) + len(marker)]
        conn.executescript(prefix)
        conn.execute("COMMIT")
    finally:
        conn.close()
    # The next init_db discards the stale staging table and redoes the copy.
    db.init_db(database)
    conn = db.connect(database)
    try:
        calls = conn.execute(
            "SELECT id, job_id, stage, evidence_json FROM model_calls"
        ).fetchall()
        events = conn.execute(
            "SELECT id, job_id, event, detail_json FROM events"
        ).fetchall()
        staging = conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN"
            " ('model_calls_new', 'events_new')"
        ).fetchall()
        uploads_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'uploads'"
        ).fetchone()
    finally:
        conn.close()
    assert [(row["id"], row["job_id"]) for row in calls] == [("1", "job-1")]
    assert calls[0]["stage"] == "s"
    assert [(row["id"], row["job_id"]) for row in events] == [("1", "job-1")]
    assert events[0]["event"] == "fired"
    assert staging == []
    assert uploads_exists is not None
    # And a further re-run stays idempotent with the same rows.
    db.init_db(database)
    conn = db.connect(database)
    try:
        again = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()
    assert again == 1


def test_retry_claim_upgrade_settles_duplicate_jobs_and_holds(lane_db: Path) -> None:
    label = "parent/extract/retry/0"
    winner = admit_job(label, label, label, internal=True)
    with db.connect(lane_db) as conn:
        conn.execute("DROP INDEX idx_jobs_extract_retry_claim")
    loser = admit_job(label, label, label, internal=True)
    loser.reserve("spark_steps", Decimal("0.03")).settle(
        Decimal("0.01"), "succeeded"
    )
    loser.reserve("spark_steps", Decimal("0.03"))

    db.init_db(lane_db)
    with db.connect(lane_db) as conn:
        rows = conn.execute(
            "SELECT id, state, anon_id FROM jobs WHERE id IN (?, ?)",
            (str(winner.job_id), str(loser.job_id)),
        ).fetchall()
        reservations = conn.execute(
            "SELECT call_kind, state FROM reservations WHERE job_id = ?",
            (str(loser.job_id),),
        ).fetchall()
        reason = conn.execute(
            "SELECT detail_json FROM events WHERE job_id = ? AND event = 'retry_claim_duplicate'",
            (str(loser.job_id),),
        ).fetchone()
        spark_row = conn.execute(
            "SELECT held_usd, spent_usd FROM budget_rows WHERE name = 'spark'"
        ).fetchone()
        aggregate = conn.execute(
            "SELECT held_usd, spent_usd FROM budget_rows WHERE name = 'aggregate'"
        ).fetchone()
    by_id = {row["id"]: row for row in rows}
    assert by_id[str(winner.job_id)]["anon_id"] == label
    assert by_id[str(loser.job_id)]["state"] == "failed"
    assert by_id[str(loser.job_id)]["anon_id"] != label
    assert {tuple(row) for row in reservations} == {
        ("job", "job_released"), ("spark_steps", "released"),
        ("spark_steps", "settled"),
    }
    assert reason is not None and "duplicate" in reason["detail_json"]
    assert Decimal(spark_row["held_usd"]) == 0
    assert Decimal(spark_row["spent_usd"]) == Decimal("0.01")
    assert Decimal(aggregate["held_usd"]) == Decimal("0.46")
    assert Decimal(aggregate["spent_usd"]) == Decimal("0.01")
    db.init_db(lane_db)


def test_foreign_keys_reject_orphan_ledger_rows(lane_db: Path) -> None:
    conn = db.connect(lane_db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reservations (id, job_id, budget_row, call_kind,"
                " estimated_usd, actual_usd, state, created_at, settled_at)"
                " VALUES ('r1', 'no-such-job', 'sam', 'sam', '0.01', NULL,"
                " 'held', '2026-10-27T00:00:00.000000Z', NULL)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO artifacts (id, job_id, kind, relative_path, expires_at,"
                " hidden, kept_at, delete_state, deleted_at, created_at)"
                " VALUES ('row-one', 'no-such-job', 'provisional', 'p.png',"
                " '2026-11-03T00:00:00.000000Z', 0, NULL, 'live', NULL,"
                " '2026-10-27T00:00:00.000000Z')"
            )
    finally:
        conn.close()


def test_storage_is_private_for_new_and_existing_installations(
    tmp_path: Path,
) -> None:
    import os
    import stat

    # New installation under a permissive umask: the directory lands
    # 0700 and the database 0600, never 0755/0644.
    target = tmp_path / "var" / "partfidelity.sqlite3"
    old_umask = os.umask(0o022)
    try:
        conn = db.connect(target)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
            conn.execute("INSERT INTO t (id) VALUES (1)")
        finally:
            conn.close()
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    # Existing installation with lax modes is repaired on next connect.
    os.chmod(target.parent, 0o755)
    os.chmod(target, 0o644)
    wal = Path(str(target) + "-wal")
    if wal.exists():
        os.chmod(wal, 0o644)
    conn = db.connect(target)
    conn.close()
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target) + suffix)
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600, suffix
