-- Phase C Learning & Governance tables for Grace
-- File: migrations/004_phase_c_learning_tables.sql
-- Creates: training_bundles, model_artifacts, trust_ledger, governance_proposals, approvals, canary_rollouts, dependency_snapshots, model_validation_results, federation_references

BEGIN;

-- training_bundles: deterministic, auditable dataset manifests
CREATE TABLE IF NOT EXISTS training_bundles (
    bundle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_cid TEXT UNIQUE NOT NULL,
    selection_logic TEXT,
    example_cids JSONB,
    label_source_cids JSONB,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_training_bundles_cid ON training_bundles (bundle_cid);
CREATE INDEX IF NOT EXISTS idx_training_bundles_example_gin ON training_bundles USING GIN (example_cids);

-- model_artifacts: models and adaptors
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_cid TEXT UNIQUE NOT NULL,
    type TEXT,
    version TEXT,
    training_bundle_cid TEXT,
    validation_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    promoted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_cid ON model_artifacts (artifact_cid);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_version ON model_artifacts (version);

-- trust_ledger: track dynamic trust for entities
CREATE TABLE IF NOT EXISTS trust_ledger (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT,
    alpha NUMERIC,
    beta NUMERIC,
    trust_score NUMERIC,
    last_updated TIMESTAMPTZ DEFAULT now(),
    history_cid TEXT
);
CREATE INDEX IF NOT EXISTS idx_trust_ledger_entity ON trust_ledger (entity_id);

-- governance_proposals: governance transactions
CREATE TABLE IF NOT EXISTS governance_proposals (
    proposal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_cid TEXT UNIQUE NOT NULL,
    report_cid TEXT,
    action_json JSONB,
    risk_score NUMERIC,
    required_approvals INT,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_governance_proposals_cid ON governance_proposals (proposal_cid);
CREATE INDEX IF NOT EXISTS idx_governance_proposals_status ON governance_proposals (status);

-- approvals: signed approvals for proposals
CREATE TABLE IF NOT EXISTS approvals (
    approval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id UUID NOT NULL,
    approver_id TEXT,
    role TEXT,
    signature TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_approvals_proposal ON approvals (proposal_id);
CREATE INDEX IF NOT EXISTS idx_approvals_approver ON approvals (approver_id);

-- canary_rollouts: canary promotion records
CREATE TABLE IF NOT EXISTS canary_rollouts (
    canary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_cid TEXT,
    target_nodes JSONB,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    kpi_before JSONB,
    kpi_after JSONB,
    result_cid TEXT
);
CREATE INDEX IF NOT EXISTS idx_canary_rollouts_artifact ON canary_rollouts (artifact_cid);

-- dependency_snapshots: fingerprinted dependency manifests
CREATE TABLE IF NOT EXISTS dependency_snapshots (
    dep_snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dep_snapshot_cid TEXT UNIQUE NOT NULL,
    manifest_json JSONB,
    os_fingerprint TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    image_cid TEXT
);
CREATE INDEX IF NOT EXISTS idx_dependency_snapshots_cid ON dependency_snapshots (dep_snapshot_cid);
CREATE INDEX IF NOT EXISTS idx_dependency_snapshots_os ON dependency_snapshots (os_fingerprint);

-- model_validation_results: validation runs per artifact
CREATE TABLE IF NOT EXISTS model_validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_cid TEXT,
    validation_bundle_cid TEXT,
    metrics_json JSONB,
    passed_bool BOOLEAN,
    run_cid TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_model_validation_artifact ON model_validation_results (artifact_cid);

-- federation_references: pointers to remote entries in federated instances
CREATE TABLE IF NOT EXISTS federation_references (
    federation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    remote_entry_cid TEXT,
    local_reference_cid TEXT,
    peer_url TEXT,
    signature TEXT,
    fetched_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_federation_remote ON federation_references (remote_entry_cid);
CREATE INDEX IF NOT EXISTS idx_federation_peer ON federation_references (peer_url);

COMMIT;
