-- Phase A core tables for Grace
-- File: migrations/002_phase_a_core_tables.sql
-- Creates: immutable_entries, evidence_blobs, event_logs, user_accounts, kpi_snapshots, search_index_meta

BEGIN;

-- immutable_entries: canonical log
CREATE TABLE IF NOT EXISTS immutable_entries (
    entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_cid TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor JSONB,
    operation TEXT,
    error_code TEXT,
    severity TEXT,
    what TEXT,
    why TEXT,
    how TEXT,
    where JSONB,
    who JSONB,
    text_summary TEXT,
    payload_path TEXT,
    signature TEXT,
    tags TEXT[],
    payload_json JSONB
);

-- GIN index for tags
CREATE INDEX IF NOT EXISTS idx_immutable_entries_tags ON immutable_entries USING GIN (tags);

-- Full text index for combined searchable fields
ALTER TABLE immutable_entries ADD COLUMN IF NOT EXISTS tsv tsvector;
CREATE INDEX IF NOT EXISTS idx_immutable_entries_tsv ON immutable_entries USING GIN (tsv);

-- Populate tsv column and add trigger
CREATE FUNCTION immutable_entries_tsv_trigger() RETURNS trigger AS $$
begin
  new.tsv := to_tsvector('english', coalesce(new.what,'') || ' ' || coalesce(new.why,'') || ' ' || coalesce(new.how,'') || ' ' || coalesce(new.text_summary,''));
  return new;
end
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
    ON immutable_entries FOR EACH ROW EXECUTE PROCEDURE immutable_entries_tsv_trigger();

-- evidence_blobs: binary artifact mapping
CREATE TABLE IF NOT EXISTS evidence_blobs (
    blob_cid TEXT PRIMARY KEY,
    storage_path TEXT NOT NULL,
    content_type TEXT,
    size BIGINT,
    checksum TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    access_policy JSONB
);

-- event_logs: raw kernel events
CREATE TABLE IF NOT EXISTS event_logs (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    payload_cid TEXT,
    timestamp TIMESTAMPTZ DEFAULT now(),
    severity TEXT,
    payload_json JSONB
);

-- user_accounts: users and roles
CREATE TABLE IF NOT EXISTS user_accounts (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    role TEXT,
    public_key TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    active_bool BOOLEAN DEFAULT TRUE
);

-- kpi_snapshots: time-series KPIs
CREATE TABLE IF NOT EXISTS kpi_snapshots (
    kpi_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_cid TEXT,
    metrics_json JSONB,
    timestamp TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_kpi_snapshots_ts ON kpi_snapshots (timestamp);

-- search_index_meta: mapping for vector embeddings
CREATE TABLE IF NOT EXISTS search_index_meta (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_cid TEXT NOT NULL,
    embed_model TEXT,
    vector_store_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_search_index_meta_entry ON search_index_meta (entry_cid);

COMMIT;
