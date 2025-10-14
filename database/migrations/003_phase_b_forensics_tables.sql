-- Phase B Forensics & Self-Heal tables for Grace
-- File: migrations/003_phase_b_forensics_tables.sql
-- Creates: specialist_analyses, forensic_reports, incidents, remediation_actions, sandbox_runs, patches

BEGIN;

-- specialist_analyses: per-specialist analysis outputs
CREATE TABLE IF NOT EXISTS specialist_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_cid TEXT UNIQUE NOT NULL,
    entry_cid TEXT NOT NULL,
    specialist_id TEXT NOT NULL,
    domain TEXT,
    confidence NUMERIC,
    analysis_summary TEXT,
    analysis_blob_path TEXT,
    signature TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_specialist_analyses_entry ON specialist_analyses (entry_cid);
CREATE INDEX IF NOT EXISTS idx_specialist_analyses_specialist ON specialist_analyses (specialist_id);

-- forensic_reports: aggregated quorum reports
CREATE TABLE IF NOT EXISTS forensic_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_cid TEXT UNIQUE NOT NULL,
    entry_cid TEXT NOT NULL,
    canonical_rca_cid TEXT,
    recommended_actions_json JSONB,
    consensus_score NUMERIC,
    dissent_json JSONB,
    signed_by JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_forensic_reports_entry ON forensic_reports (entry_cid);
CREATE INDEX IF NOT EXISTS idx_forensic_reports_report ON forensic_reports (report_cid);

-- incidents: group related immutable entries
CREATE TABLE IF NOT EXISTS incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    primary_error_code TEXT,
    entry_cids JSONB,
    first_seen TIMESTAMPTZ DEFAULT now(),
    last_seen TIMESTAMPTZ DEFAULT now(),
    severity TEXT,
    status TEXT,
    incident_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_error ON incidents (primary_error_code);
CREATE INDEX IF NOT EXISTS idx_incidents_entrycids ON incidents USING GIN (entry_cids);

-- remediation_actions: proposed/executed remediations
CREATE TABLE IF NOT EXISTS remediation_actions (
    remediation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    remediation_cid TEXT UNIQUE NOT NULL,
    proposal_id UUID,
    action_type TEXT,
    params_json JSONB,
    executor TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    result_cid TEXT,
    success_bool BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_remediation_actions_proposal ON remediation_actions (proposal_id);
CREATE INDEX IF NOT EXISTS idx_remediation_actions_executor ON remediation_actions (executor);

-- sandbox_runs: sandbox/CI runs
CREATE TABLE IF NOT EXISTS sandbox_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_cid TEXT UNIQUE NOT NULL,
    remediation_id UUID,
    env_matrix JSONB,
    test_results_cid TEXT,
    kpi_snapshot_cid TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT
);
CREATE INDEX IF NOT EXISTS idx_sandbox_runs_remediation ON sandbox_runs (remediation_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_runs_status ON sandbox_runs (status);

-- patches: code patches / generated changes
CREATE TABLE IF NOT EXISTS patches (
    patch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patch_cid TEXT UNIQUE NOT NULL,
    source_bundle_cid TEXT,
    author TEXT,
    diff_summary TEXT,
    applied_at TIMESTAMPTZ,
    status TEXT,
    validation_cid TEXT
);
CREATE INDEX IF NOT EXISTS idx_patches_patch_cid ON patches (patch_cid);
CREATE INDEX IF NOT EXISTS idx_patches_status ON patches (status);

COMMIT;
