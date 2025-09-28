-- Grace Resilience Kernel Database Schema
-- Manages circuit breaker state, degradation policies, and incident tracking

-- Circuit Breaker States
CREATE TABLE IF NOT EXISTS circuit_breakers (
    id TEXT PRIMARY KEY,
    component_name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'closed',  -- closed, open, half_open
    failure_count INTEGER NOT NULL DEFAULT 0,
    failure_threshold INTEGER NOT NULL DEFAULT 5,
    timeout_duration_seconds INTEGER NOT NULL DEFAULT 60,
    last_failure_time TIMESTAMP,
    last_success_time TIMESTAMP,
    next_attempt_time TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Degradation Policies
CREATE TABLE IF NOT EXISTS degradation_policies (
    id TEXT PRIMARY KEY,
    policy_name TEXT UNIQUE NOT NULL,
    component_pattern TEXT NOT NULL,
    trigger_conditions JSON NOT NULL,  -- JSON object with conditions
    degradation_actions JSON NOT NULL,  -- JSON array of actions
    priority INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Active Degradations
CREATE TABLE IF NOT EXISTS active_degradations (
    id TEXT PRIMARY KEY,
    component_name TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    degradation_level TEXT NOT NULL,  -- mild, moderate, severe
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expected_duration_seconds INTEGER,
    metadata JSON,
    FOREIGN KEY (policy_id) REFERENCES degradation_policies(id)
);

-- Incidents
CREATE TABLE IF NOT EXISTS resilience_incidents (
    id TEXT PRIMARY KEY,
    incident_type TEXT NOT NULL,  -- circuit_break, degradation, failure
    component_name TEXT NOT NULL,
    severity TEXT NOT NULL,  -- low, medium, high, critical
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',  -- open, investigating, resolved
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_summary TEXT,
    metadata JSON
);

-- Rate Limiting
CREATE TABLE IF NOT EXISTS rate_limits (
    id TEXT PRIMARY KEY,
    resource_name TEXT NOT NULL,
    limit_type TEXT NOT NULL,  -- per_second, per_minute, per_hour
    max_requests INTEGER NOT NULL,
    current_requests INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reset_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Service Level Indicators (SLI) measurements
CREATE TABLE IF NOT EXISTS sli_measurements (
    id TEXT PRIMARY KEY,
    sli_name TEXT NOT NULL,
    component_name TEXT NOT NULL,
    measurement_type TEXT NOT NULL,  -- latency, availability, error_rate
    value REAL NOT NULL,
    target_value REAL NOT NULL,
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Recovery Actions
CREATE TABLE IF NOT EXISTS recovery_actions (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL,
    action_type TEXT NOT NULL,  -- restart, rollback, scale, manual
    action_details JSON NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_summary TEXT,
    FOREIGN KEY (incident_id) REFERENCES resilience_incidents(id)
);

-- Resilience Snapshots
CREATE TABLE IF NOT EXISTS resilience_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    component_states JSON NOT NULL,
    circuit_breaker_states JSON NOT NULL,
    active_policies JSON NOT NULL,
    degradation_states JSON NOT NULL,
    rate_limit_states JSON NOT NULL,
    snapshot_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_component ON circuit_breakers(component_name);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_state ON circuit_breakers(state);
CREATE INDEX IF NOT EXISTS idx_degradation_policies_pattern ON degradation_policies(component_pattern);
CREATE INDEX IF NOT EXISTS idx_active_degradations_component ON active_degradations(component_name);
CREATE INDEX IF NOT EXISTS idx_incidents_component ON resilience_incidents(component_name);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON resilience_incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON resilience_incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_rate_limits_resource ON rate_limits(resource_name);
CREATE INDEX IF NOT EXISTS idx_sli_measurements_component ON sli_measurements(component_name);
CREATE INDEX IF NOT EXISTS idx_sli_measurements_measured ON sli_measurements(measured_at);
CREATE INDEX IF NOT EXISTS idx_recovery_actions_incident ON recovery_actions(incident_id);

-- Triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_circuit_breakers_timestamp 
    AFTER UPDATE ON circuit_breakers
    BEGIN
        UPDATE circuit_breakers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_degradation_policies_timestamp
    AFTER UPDATE ON degradation_policies  
    BEGIN
        UPDATE degradation_policies SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_rate_limits_timestamp
    AFTER UPDATE ON rate_limits
    BEGIN
        UPDATE rate_limits SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;