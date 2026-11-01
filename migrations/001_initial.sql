-- PartFidelity initial schema. The ledger module is the sole owner; other modules never add
-- migrations. Column lists mirror the frozen contract exactly. Money is fixed-point
-- decimal text, never float. Timestamps are UTC ISO-8601 with microseconds and
-- a 'Z' suffix so TEXT comparison is chronological.
--
-- jobs.state: 'waiting' | 'active' | 'done' | 'failed' | 'expired'.
-- reservations.state: 'job_hold' | 'job_released' (job-level $0.46 hold) or
--   'held' | 'settled' | 'failed' | 'released' (per-call holds).
-- artifacts.delete_state: 'live' | 'pending_delete' | 'tombstoned' | 'retry'.
-- meta_files.delete_state: 'live' | 'requested' | 'verified_absent'
--   | 'acknowledged_only' | 'retry'.

CREATE TABLE IF NOT EXISTS jobs (
  id            TEXT PRIMARY KEY,
  state         TEXT NOT NULL CHECK (state IN ('waiting', 'active', 'done', 'failed', 'expired')),
  anon_id       TEXT NOT NULL,
  ip_hash       TEXT NOT NULL,
  video_hash    TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  deadline_at   TEXT NOT NULL,
  confirmed_at  TEXT,
  completed_at  TEXT,
  cost_cap_usd  TEXT NOT NULL,
  readback_fired_n INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs (state);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at);

CREATE TABLE IF NOT EXISTS budget_rows (
  name          TEXT PRIMARY KEY,
  hard_cap_usd  TEXT NOT NULL,
  spent_usd     TEXT NOT NULL,
  held_usd      TEXT NOT NULL,
  day_utc       TEXT,
  breaker_usd   TEXT
);

CREATE TABLE IF NOT EXISTS reservations (
  id            TEXT PRIMARY KEY,
  job_id        TEXT NOT NULL REFERENCES jobs (id) ON DELETE RESTRICT,
  budget_row    TEXT NOT NULL,
  call_kind     TEXT NOT NULL,
  estimated_usd TEXT NOT NULL,
  actual_usd    TEXT,
  state         TEXT NOT NULL CHECK (state IN ('job_hold', 'job_released', 'held', 'settled', 'failed', 'released')),
  created_at    TEXT NOT NULL,
  settled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_reservations_job ON reservations (job_id);

CREATE TABLE IF NOT EXISTS meta_files (
  id                  TEXT PRIMARY KEY,
  job_id              TEXT NOT NULL REFERENCES jobs (id) ON DELETE RESTRICT,
  expires_at          TEXT NOT NULL,
  delete_state        TEXT NOT NULL,
  delete_requested_at TEXT,
  verified_absent_at  TEXT,
  last_error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_meta_files_job ON meta_files (job_id);

CREATE TABLE IF NOT EXISTS model_calls (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id            TEXT NOT NULL REFERENCES jobs (id) ON DELETE RESTRICT,
  stage             TEXT NOT NULL,
  model             TEXT NOT NULL,
  endpoint          TEXT NOT NULL,
  latency_ms        INTEGER NOT NULL,
  units_in          INTEGER,
  units_out         INTEGER,
  estimated_cost_usd TEXT NOT NULL,
  status            TEXT NOT NULL,
  evidence_json     TEXT NOT NULL,
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_calls_job ON model_calls (job_id);

CREATE TABLE IF NOT EXISTS events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id      TEXT REFERENCES jobs (id) ON DELETE SET NULL,
  anon_id     TEXT,
  session_id  TEXT,
  event       TEXT NOT NULL,
  detail_json TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events (job_id);

CREATE TABLE IF NOT EXISTS artifacts (
  id            TEXT PRIMARY KEY,
  job_id        TEXT NOT NULL REFERENCES jobs (id) ON DELETE RESTRICT,
  kind          TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  expires_at    TEXT NOT NULL,
  hidden        INTEGER NOT NULL DEFAULT 0,
  kept_at       TEXT,
  delete_state  TEXT NOT NULL,
  deleted_at    TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_job ON artifacts (job_id);
