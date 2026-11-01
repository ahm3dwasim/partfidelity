-- PartFidelity migration 006: per-job output names and failed-job rescan markers.
-- init_db applies it once, statement by statement, inside its BEGIN IMMEDIATE
-- transaction when jobs lacks these columns, then fills the names of older
-- rows with the deployment's configured names. admit_job writes both names.
-- rescan_seen_at rotates the bounded rescan batch, rescan_done_at ends it.
-- No semicolon may appear in a comment: init_db splits this file on them.
ALTER TABLE jobs ADD COLUMN guide_filename TEXT;
ALTER TABLE jobs ADD COLUMN corrections_filename TEXT;
ALTER TABLE jobs ADD COLUMN rescan_seen_at TEXT;
ALTER TABLE jobs ADD COLUMN rescan_done_at TEXT;
CREATE INDEX IF NOT EXISTS idx_jobs_rescan_pending ON jobs (rescan_seen_at, completed_at)
WHERE state = 'failed' AND rescan_done_at IS NULL;
