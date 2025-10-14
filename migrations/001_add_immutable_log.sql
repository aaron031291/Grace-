-- Migration: Immutable log table and indices (Postgres)
CREATE TABLE IF NOT EXISTS immutable_entries (
  entry_id UUID PRIMARY KEY,
  entry_cid TEXT UNIQUE NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  who_actor_id TEXT,
  who_actor_type TEXT,
  who_actor_display TEXT,
  what TEXT,
  where_host TEXT,
  where_region TEXT,
  where_service_path TEXT,
  when_ts TIMESTAMP WITH TIME ZONE,
  why TEXT,
  how TEXT,
  error_code TEXT,
  severity TEXT,
  tags TEXT[],
  text_summary TEXT,
  related_cids TEXT[],
  signature TEXT,
  payload_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_immutable_entries_timestamp ON immutable_entries (timestamp);
CREATE INDEX IF NOT EXISTS idx_immutable_entries_error_code ON immutable_entries (error_code);
CREATE INDEX IF NOT EXISTS idx_immutable_entries_tags ON immutable_entries USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_immutable_entries_fulltext ON immutable_entries USING GIN (to_tsvector('english', coalesce(what,'') || ' ' || coalesce(why,'') || ' ' || coalesce(how,'')));
