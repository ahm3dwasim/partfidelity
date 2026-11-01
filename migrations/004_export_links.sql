-- Download links expire independently of published guides. Archive bytes are never stored.
CREATE TABLE IF NOT EXISTS export_links (
  token      TEXT PRIMARY KEY,
  guide_id   TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_export_links_expiry ON export_links (expires_at);
CREATE INDEX IF NOT EXISTS idx_export_links_guide ON export_links (guide_id);
