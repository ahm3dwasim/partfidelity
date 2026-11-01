-- PartFidelity migration 002: durable uploads plus TEXT receipt ids. The ledger module sole owner.
-- 001 is untouched; init_db applies 001 then 002. Safe to re-run: every
-- statement is IF NOT EXISTS, or a rebuild that maps old and new shapes alike.
--
-- Crash safety: each table rebuild runs in one transaction, and any leftover
-- *_new staging table from an interrupted run is discarded first (the
-- originals stay intact until their own rebuild commits, so a re-run simply
-- redoes the copy). An interrupt therefore resumes cleanly on next init_db.
--
-- Receipt keys the caller's key is stored as both
-- model_calls.id and events.id, so both columns become TEXT PRIMARY KEY.
-- Standalone vocabulary events keep generated (random hex) ids. Existing rows
-- are preserved with their integer ids cast to text.
-- Meta delete_state gains 'pending' (queued for immediate deletion, due
-- regardless of expiry); LifecycleRepository lists it. The 001 comment still
-- lists the original five states because 001 is frozen.

DROP TABLE IF EXISTS model_calls_new;
DROP TABLE IF EXISTS events_new;

BEGIN IMMEDIATE;
CREATE TABLE model_calls_new (
  id                TEXT PRIMARY KEY,
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
INSERT INTO model_calls_new
  (id, job_id, stage, model, endpoint, latency_ms, units_in, units_out,
   estimated_cost_usd, status, evidence_json, created_at)
  SELECT CAST(id AS TEXT), job_id, stage, model, endpoint, latency_ms,
   units_in, units_out, estimated_cost_usd, status, evidence_json, created_at
  FROM model_calls;
DROP TABLE model_calls;
ALTER TABLE model_calls_new RENAME TO model_calls;
CREATE INDEX IF NOT EXISTS idx_model_calls_job ON model_calls (job_id);
COMMIT;

BEGIN IMMEDIATE;
CREATE TABLE events_new (
  id          TEXT PRIMARY KEY,
  job_id      TEXT REFERENCES jobs (id) ON DELETE SET NULL,
  anon_id     TEXT,
  session_id  TEXT,
  event       TEXT NOT NULL,
  detail_json TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
INSERT INTO events_new
  (id, job_id, anon_id, session_id, event, detail_json, created_at)
  SELECT CAST(id AS TEXT), job_id, anon_id, session_id, event, detail_json,
   created_at
  FROM events;
DROP TABLE events;
ALTER TABLE events_new RENAME TO events;
CREATE INDEX IF NOT EXISTS idx_events_job ON events (job_id);
COMMIT;

-- uploads.job_id is the primary/foreign key to jobs; state is closed over the
-- six UploadRecord.state literals; preview_artifact_id, claim_id, and non-null
-- meta_file_id are unique; artifact/Meta ids are RESTRICT foreign keys. The
-- committed claim_id is both local claim ownership and the exact provider
-- idempotency/lookup key. Column order mirrors the frozen contract exactly.
CREATE TABLE IF NOT EXISTS uploads (
  job_id              TEXT PRIMARY KEY REFERENCES jobs (id) ON DELETE RESTRICT,
  preview_artifact_id TEXT NOT NULL UNIQUE REFERENCES artifacts (id) ON DELETE RESTRICT,
  state               TEXT NOT NULL CHECK (state IN ('preview', 'confirming', 'confirmed', 'cancelling', 'cancelled', 'failed')),
  preview_expires_at  TEXT NOT NULL,
  duration_s          REAL NOT NULL,
  size_bytes          INTEGER NOT NULL,
  audio_removed       INTEGER NOT NULL,
  consented_at        TEXT,
  claim_id            TEXT UNIQUE,
  claim_started_at    TEXT,
  guard_deadline_utc  TEXT,
  meta_file_id        TEXT UNIQUE REFERENCES meta_files (id) ON DELETE RESTRICT,
  confirmed_at        TEXT,
  cancel_requested_at TEXT,
  hidden              INTEGER NOT NULL DEFAULT 0,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL
);

-- upload_failures.failure_key is the caller's failure-attempt idempotency key
-- and primary key. A row counts toward the public block until resolved_at is
-- set by recover_job. Column order mirrors the frozen contract exactly.
CREATE TABLE IF NOT EXISTS upload_failures (
  failure_key TEXT PRIMARY KEY,
  job_id      TEXT NOT NULL REFERENCES jobs (id) ON DELETE RESTRICT,
  stage       TEXT NOT NULL CHECK (stage IN ('meta_upload', 'meta_persist', 'local_delete', 'meta_delete', 'recovery')),
  error       TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_upload_failures_job ON upload_failures (job_id);

-- Bounded operator delete-request queue: contact address only, no name or
-- message. job_id is nullable (no FK: a request must survive a lost or
-- mistyped job id). Column order mirrors the frozen contract exactly.
CREATE TABLE IF NOT EXISTS delete_requests (
  id           TEXT PRIMARY KEY,
  job_id       TEXT,
  contact      TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  expires_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_delete_requests_expires ON delete_requests (expires_at);
