-- Grace Orchestration Kernel Database Schema
-- DDL for orchestration state persistence and management

-- Core orchestration state table
CREATE TABLE IF NOT EXISTS orchestration_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL,
    context TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Orchestration loops definition and status
CREATE TABLE IF NOT EXISTS orchestration_loops (
    loop_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    priority INTEGER NOT NULL CHECK(priority BETWEEN 1 AND 10),
    interval_s INTEGER NOT NULL CHECK(interval_s > 0),
    kernels TEXT NOT NULL, -- JSON array of kernel names
    policies TEXT NOT NULL, -- JSON object of policies
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_run REAL DEFAULT 0,
    next_run REAL DEFAULT 0,
    run_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    average_duration REAL DEFAULT 0.0
);

-- Orchestration tasks
CREATE TABLE IF NOT EXISTS orchestration_tasks (
    task_id TEXT PRIMARY KEY,
    loop_id TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN ('start', 'run', 'validate', 'complete', 'rollback')),
    inputs TEXT NOT NULL, -- JSON object
    outputs TEXT, -- JSON object
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'succeeded', 'failed', 'rolled_back')),
    error TEXT, -- JSON object
    priority INTEGER NOT NULL DEFAULT 5 CHECK(priority BETWEEN 1 AND 10),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    timeout_minutes INTEGER DEFAULT 30,
    FOREIGN KEY (loop_id) REFERENCES orchestration_loops(loop_id)
);

-- System policies
CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('security', 'governance', 'performance', 'reliability', 'compliance')),
    scope TEXT NOT NULL CHECK(scope IN ('global', 'kernel', 'loop', 'task')),
    rules TEXT NOT NULL, -- JSON object
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    applied_count INTEGER DEFAULT 0,
    violation_count INTEGER DEFAULT 0
);

-- State transitions history
CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    trigger TEXT NOT NULL,
    context TEXT NOT NULL, -- JSON object
    timestamp TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- System snapshots
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('manual', 'scheduled', 'pre_rollback', 'pre_upgrade', 'emergency')),
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('creating', 'completed', 'failed', 'validating', 'corrupted')),
    description TEXT DEFAULT '',
    tags TEXT, -- JSON array
    size_bytes INTEGER DEFAULT 0,
    hash TEXT DEFAULT '',
    validation_errors TEXT, -- JSON array
    validated_at TEXT,
    -- Snapshot content
    loops TEXT, -- JSON array of loop definitions
    active_tasks TEXT, -- JSON array of task IDs
    policies TEXT, -- JSON object of policies
    component_states TEXT, -- JSON object of component states
    system_metrics TEXT -- JSON object of metrics
);

-- Rollback operations
CREATE TABLE IF NOT EXISTS rollback_operations (
    operation_id TEXT PRIMARY KEY,
    target_snapshot_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('preparing', 'draining', 'restoring', 'validating', 'completed', 'failed')),
    steps_completed INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    current_step TEXT DEFAULT '',
    pre_rollback_snapshot_id TEXT,
    errors TEXT, -- JSON array
    warnings TEXT, -- JSON array
    FOREIGN KEY (target_snapshot_id) REFERENCES snapshots(snapshot_id),
    FOREIGN KEY (pre_rollback_snapshot_id) REFERENCES snapshots(snapshot_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_state_timestamp ON orchestration_state(timestamp);
CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON state_transitions(timestamp);
CREATE INDEX IF NOT EXISTS idx_tasks_loop_id ON orchestration_tasks(loop_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON orchestration_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON orchestration_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_type ON snapshots(type);
CREATE INDEX IF NOT EXISTS idx_policies_type ON policies(type);
CREATE INDEX IF NOT EXISTS idx_policies_scope ON policies(scope);

-- Initial data - Default governance policies
INSERT OR IGNORE INTO policies (policy_id, name, type, scope, rules, created_at, updated_at) VALUES
('orch_resource_limits', 'Orchestration Resource Limits', 'performance', 'global', 
 '{"max_instances_per_loop": 10, "max_concurrent_tasks": 100, "max_memory_per_instance": "4GB"}', 
 datetime('now'), datetime('now')),
 
('orch_security_controls', 'Orchestration Security Controls', 'security', 'global',
 '{"require_authorization": ["rollback", "policy_update"], "audit_sensitive_operations": true, "encryption_required": true}',
 datetime('now'), datetime('now')),
 
('orch_operational_limits', 'Orchestration Operational Limits', 'governance', 'global',
 '{"max_rollbacks_per_day": 5, "min_snapshot_interval_hours": 1, "max_loop_execution_time_minutes": 60}',
 datetime('now'), datetime('now'));

COMMIT;