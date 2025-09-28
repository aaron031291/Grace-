-- Intelligence Kernel Database Schema
-- Stores operational data for task requests, plans, results, and snapshots

-- Task requests table
CREATE TABLE intel_requests (
    req_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    modality TEXT,
    context_json JSONB NOT NULL,
    input_data_json JSONB,
    constraints_json JSONB,
    user_ctx_json JSONB,
    latency_budget_ms INTEGER,
    cost_budget_units DECIMAL(10,4),
    explanation_required BOOLEAN DEFAULT FALSE,
    canary_allowed BOOLEAN DEFAULT TRUE,
    segment TEXT,
    env TEXT CHECK (env IN ('dev', 'staging', 'prod')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'received'
);

-- Execution plans table
CREATE TABLE intel_plans (
    plan_id TEXT PRIMARY KEY,
    req_id TEXT NOT NULL REFERENCES intel_requests(req_id),
    route_json JSONB NOT NULL, -- specialists, models, ensemble, canary_pct, shadow
    policy_json JSONB NOT NULL, -- min_confidence, min_calibration, fairness_delta_max
    risk_level TEXT CHECK (risk_level IN ('low', 'medium', 'high')),
    governance_approved BOOLEAN DEFAULT FALSE,
    governance_decision_id TEXT,
    pre_flight_status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'ready'
);

-- Inference results table
CREATE TABLE intel_results (
    req_id TEXT PRIMARY KEY REFERENCES intel_requests(req_id),
    plan_id TEXT REFERENCES intel_plans(plan_id),
    outputs_json JSONB, -- y_hat, proba, top_k, actions
    metrics_json JSONB, -- runtime metrics
    explanations_json JSONB, -- optional explanations
    uncertainties_json JSONB, -- calibration, variance, entropy
    lineage_json JSONB, -- plan_id, models, ensemble, feature_view
    governance_json JSONB, -- approved, policy_version, redactions
    timing_json JSONB, -- total_ms, per_stage
    shadow_result_json JSONB, -- shadow deployment results
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Specialist reports table (for tracking specialist performance)
CREATE TABLE intel_specialist_reports (
    report_id TEXT PRIMARY KEY,
    req_id TEXT REFERENCES intel_requests(req_id),
    specialist_name TEXT NOT NULL,
    model_key TEXT,
    candidates_json JSONB, -- candidate models and their metrics
    metrics_json JSONB, -- specialist-specific metrics
    uncertainties_json JSONB, -- calibration, variance, entropy
    explanations_json JSONB, -- specialist explanations
    risks_json JSONB, -- drift, bias, instability warnings
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Snapshots table
CREATE TABLE intel_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    snapshot_type TEXT DEFAULT 'intelligence', -- intelligence, governance, mlt
    payload_json JSONB NOT NULL,
    version TEXT,
    hash TEXT NOT NULL,
    size_bytes INTEGER,
    compressed BOOLEAN DEFAULT FALSE,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    restored_at TIMESTAMPTZ,
    restore_count INTEGER DEFAULT 0,
    description TEXT
);

-- Experience tracking for meta-learning
CREATE TABLE intel_experiences (
    exp_id TEXT PRIMARY KEY,
    req_id TEXT REFERENCES intel_requests(req_id),
    stage TEXT CHECK (stage IN ('route', 'plan', 'infer', 'explain', 'eval', 'canary', 'shadow')),
    metrics_json JSONB, -- p95_latency_ms, success_rate, calibration, etc.
    segment TEXT,
    model_keys TEXT[], -- array of model keys used
    ensemble_type TEXT,
    canary_pct INTEGER,
    shadow_enabled BOOLEAN DEFAULT FALSE,
    governance_approved BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Canary deployment tracking
CREATE TABLE intel_canary_deployments (
    deployment_id TEXT PRIMARY KEY,
    model_key TEXT NOT NULL,
    target_version TEXT NOT NULL,
    current_step INTEGER DEFAULT 0,
    steps INTEGER[] NOT NULL, -- [5, 25, 50, 100] traffic percentages
    success_metrics_json JSONB, -- success criteria
    current_metrics_json JSONB, -- current performance
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'promoted', 'rolled_back', 'paused')),
    auto_promote_threshold DECIMAL(3,2) DEFAULT 0.95,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Shadow deployment tracking  
CREATE TABLE intel_shadow_deployments (
    deployment_id TEXT PRIMARY KEY,
    primary_model_key TEXT NOT NULL,
    shadow_model_key TEXT NOT NULL,
    traffic_pct INTEGER DEFAULT 100, -- percentage of traffic to shadow
    agreement_threshold DECIMAL(3,2) DEFAULT 0.90,
    current_agreement DECIMAL(3,2),
    sample_count INTEGER DEFAULT 0,
    agreement_history_json JSONB, -- time series of agreement scores
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'terminated')),
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Policy violations and governance events
CREATE TABLE intel_policy_violations (
    violation_id TEXT PRIMARY KEY,
    req_id TEXT REFERENCES intel_requests(req_id),
    plan_id TEXT REFERENCES intel_plans(plan_id),
    violation_type TEXT NOT NULL, -- confidence_threshold, calibration_threshold, fairness_delta
    severity TEXT CHECK (severity IN ('warning', 'error', 'critical')),
    required_value DECIMAL(10,4),
    actual_value DECIMAL(10,4),
    policy_version TEXT,
    remediation_json JSONB, -- suggested fixes
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics aggregation
CREATE TABLE intel_metrics_hourly (
    metric_hour TIMESTAMPTZ NOT NULL, -- hour bucket
    segment TEXT,
    env TEXT,
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    avg_latency_ms DECIMAL(8,2),
    p95_latency_ms DECIMAL(8,2),
    avg_confidence DECIMAL(4,3),
    calibration_error DECIMAL(4,3),
    fairness_delta DECIMAL(4,3),
    canary_deployments INTEGER DEFAULT 0,
    shadow_deployments INTEGER DEFAULT 0,
    policy_violations INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (metric_hour, COALESCE(segment, ''), COALESCE(env, ''))
);

-- Indexes for performance
CREATE INDEX idx_intel_requests_created_at ON intel_requests(created_at);
CREATE INDEX idx_intel_requests_status ON intel_requests(status);
CREATE INDEX idx_intel_requests_task ON intel_requests(task);
CREATE INDEX idx_intel_requests_env ON intel_requests(env);

CREATE INDEX idx_intel_plans_req_id ON intel_plans(req_id);
CREATE INDEX idx_intel_plans_status ON intel_plans(status);
CREATE INDEX idx_intel_plans_created_at ON intel_plans(created_at);

CREATE INDEX idx_intel_results_created_at ON intel_results(created_at);
CREATE INDEX idx_intel_results_success ON intel_results(success);
CREATE INDEX idx_intel_results_plan_id ON intel_results(plan_id);

CREATE INDEX idx_intel_experiences_stage ON intel_experiences(stage);
CREATE INDEX idx_intel_experiences_created_at ON intel_experiences(created_at);
CREATE INDEX idx_intel_experiences_segment ON intel_experiences(segment);

CREATE INDEX idx_intel_snapshots_created_at ON intel_snapshots(created_at);
CREATE INDEX idx_intel_snapshots_type ON intel_snapshots(snapshot_type);

CREATE INDEX idx_intel_policy_violations_severity ON intel_policy_violations(severity);
CREATE INDEX idx_intel_policy_violations_resolved ON intel_policy_violations(resolved);
CREATE INDEX idx_intel_policy_violations_created_at ON intel_policy_violations(created_at);

CREATE INDEX idx_intel_metrics_hourly_hour ON intel_metrics_hourly(metric_hour);
CREATE INDEX idx_intel_metrics_hourly_env ON intel_metrics_hourly(env);

-- Views for common queries
CREATE VIEW intel_active_requests AS
SELECT r.*, p.plan_id, p.status as plan_status, p.governance_approved
FROM intel_requests r
LEFT JOIN intel_plans p ON r.req_id = p.req_id
WHERE r.status IN ('received', 'processing') OR p.status IN ('ready', 'executing');

CREATE VIEW intel_performance_summary AS
SELECT 
    DATE_TRUNC('day', created_at) as day,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE success = true) as successful_requests,
    ROUND(AVG(CAST((timing_json->>'total_ms') AS DECIMAL)), 2) as avg_latency_ms,
    ROUND(AVG(CAST((outputs_json->>'confidence') AS DECIMAL)), 3) as avg_confidence,
    COUNT(*) FILTER (WHERE shadow_result_json IS NOT NULL) as shadow_requests
FROM intel_results
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY day DESC;

-- Stored procedure for metrics aggregation (PostgreSQL syntax)
CREATE OR REPLACE FUNCTION aggregate_hourly_metrics()
RETURNS void AS $$
BEGIN
    INSERT INTO intel_metrics_hourly (
        metric_hour, segment, env, total_requests, successful_requests,
        avg_latency_ms, p95_latency_ms, avg_confidence
    )
    SELECT 
        DATE_TRUNC('hour', ir.created_at) as metric_hour,
        COALESCE(ir_req.segment, '') as segment,
        COALESCE(ir_req.env, '') as env,
        COUNT(*) as total_requests,
        COUNT(*) FILTER (WHERE ir.success = true) as successful_requests,
        ROUND(AVG(CAST((ir.timing_json->>'total_ms') AS DECIMAL)), 2) as avg_latency_ms,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CAST((ir.timing_json->>'total_ms') AS DECIMAL)), 2) as p95_latency_ms,
        ROUND(AVG(CAST((ir.outputs_json->>'confidence') AS DECIMAL)), 3) as avg_confidence
    FROM intel_results ir
    JOIN intel_requests ir_req ON ir.req_id = ir_req.req_id
    WHERE ir.created_at >= CURRENT_TIMESTAMP - INTERVAL '2 hours'
        AND ir.created_at < DATE_TRUNC('hour', CURRENT_TIMESTAMP)
    GROUP BY DATE_TRUNC('hour', ir.created_at), ir_req.segment, ir_req.env
    ON CONFLICT (metric_hour, segment, env) DO UPDATE SET
        total_requests = EXCLUDED.total_requests,
        successful_requests = EXCLUDED.successful_requests,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        p95_latency_ms = EXCLUDED.p95_latency_ms,
        avg_confidence = EXCLUDED.avg_confidence,
        created_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;