-- PartFidelity migration 003: delete-request throttle buckets.
-- Safe to re-run: CREATE TABLE IF NOT EXISTS only; init_db applies
-- 001 then 002 then 003. One row per active source/session bucket with
-- a fixed-window start and count; the store purges buckets outside the
-- window in the same transaction that checks them, so the table stays
-- bounded without a sweeper of its own. Bucket keys are "ip:<hex>" and
-- "sess:<id>" strings; no contact text is stored here.

CREATE TABLE IF NOT EXISTS delete_throttle (
  bucket      TEXT PRIMARY KEY,
  window_start TEXT NOT NULL,
  count       INTEGER NOT NULL
);
