-- Grace ML Database Schema
-- Core tables for MLT (Memory, Learning, Trust) operations

-- Experiences table
CREATE TABLE mlt_experiences (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL CHECK (source IN ('training', 'inference', 'governance', 'ops')),
    task TEXT NOT NULL CHECK (task IN ('classification', 'regression', 'clustering', 'dimred', 'rl')),
    context_json JSONB NOT NULL,
    signals_json JSONB NOT NULL,
    gt_lag_s INTEGER NOT NULL CHECK (gt_lag_s >= 0),
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_experience_id CHECK (id ~ '^[a-z][a-z0-9_-]{4,64}$')
);

-- Create index for efficient querying by source and task
CREATE INDEX idx_mlt_experiences_source_task ON mlt_experiences(source, task);
CREATE INDEX idx_mlt_experiences_ts ON mlt_experiences(ts);

-- Insights table
CREATE TABLE mlt_insights (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('performance', 'drift', 'fairness', 'calibration', 'stability', 'governance_alignment')),
    scope TEXT NOT NULL CHECK (scope IN ('model', 'specialist', 'policy', 'dataset', 'segment')),
    target_ref TEXT,
    evidence_json JSONB NOT NULL,
    confidence NUMERIC NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    recommendation TEXT CHECK (recommendation IN ('retrain', 'reweight', 'recalibrate', 'hpo', 'policy_tune', 'segment_route')),
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_insight_id CHECK (id ~ '^[a-z][a-z0-9_-]{4,64}$')
);

-- Create indexes for efficient querying
CREATE INDEX idx_mlt_insights_type_scope ON mlt_insights(type, scope);
CREATE INDEX idx_mlt_insights_ts ON mlt_insights(ts);
CREATE INDEX idx_mlt_insights_confidence ON mlt_insights(confidence DESC);

-- Plans table
CREATE TABLE mlt_plans (
    id TEXT PRIMARY KEY,
    plan_json JSONB NOT NULL,
    expected_effect JSONB NOT NULL,
    risk_controls JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'applied')),
    rationale TEXT,
    correlation_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_plan_id CHECK (id ~ '^[a-z][a-z0-9_-]{4,64}$'),
    CONSTRAINT valid_correlation_id CHECK (correlation_id IS NULL OR correlation_id ~ '^[a-z][a-z0-9_-]{4,64}$')
);

-- Create indexes for plan operations
CREATE INDEX idx_mlt_plans_status ON mlt_plans(status);
CREATE INDEX idx_mlt_plans_correlation_id ON mlt_plans(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX idx_mlt_plans_created_at ON mlt_plans(created_at);

-- Snapshots table
CREATE TABLE mlt_snapshots (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('governance', 'mlt')),
    payload_json JSONB NOT NULL,
    hash TEXT NOT NULL CHECK (hash ~ '^sha256:[a-f0-9]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_snapshot_id CHECK (id ~ '^[a-z][a-z0-9_-]{4,64}$')
);

-- Create indexes for snapshot operations
CREATE INDEX idx_mlt_snapshots_type ON mlt_snapshots(type);
CREATE INDEX idx_mlt_snapshots_created_at ON mlt_snapshots(created_at);
CREATE INDEX idx_mlt_snapshots_hash ON mlt_snapshots(hash);

-- Specialist reports table (for tracking ML model evaluations)
CREATE TABLE mlt_specialist_reports (
    id TEXT PRIMARY KEY,
    specialist TEXT NOT NULL,
    task TEXT NOT NULL CHECK (task IN ('classification', 'regression', 'clustering', 'dimred', 'rl')),
    candidates_json JSONB NOT NULL,
    dataset_id TEXT,
    notes TEXT,
    version TEXT NOT NULL CHECK (version ~ '^[0-9]+\.[0-9]+\.[0-9]+$'),
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_report_id CHECK (id ~ '^[a-z][a-z0-9_-]{4,64}$')
);

-- Create indexes for specialist reports
CREATE INDEX idx_mlt_specialist_reports_specialist ON mlt_specialist_reports(specialist);
CREATE INDEX idx_mlt_specialist_reports_task ON mlt_specialist_reports(task);
CREATE INDEX idx_mlt_specialist_reports_ts ON mlt_specialist_reports(ts);

-- Trigger to update updated_at timestamp on plan changes
CREATE OR REPLACE FUNCTION update_mlt_plans_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER mlt_plans_updated_at_trigger
    BEFORE UPDATE ON mlt_plans
    FOR EACH ROW
    EXECUTE FUNCTION update_mlt_plans_updated_at();

-- Grant permissions (assuming application role 'grace_app')
-- GRANT SELECT, INSERT, UPDATE ON mlt_experiences, mlt_insights, mlt_plans, mlt_snapshots, mlt_specialist_reports TO grace_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO grace_app;