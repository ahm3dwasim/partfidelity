-- A retry child is the durable, unique claim for one parent/window.
-- init_db settles and renames legacy duplicate claims under the same
-- BEGIN IMMEDIATE transaction before executing this index statement.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_extract_retry_claim
ON jobs (anon_id)
WHERE anon_id LIKE '%/extract/retry/%' AND ip_hash = anon_id;
