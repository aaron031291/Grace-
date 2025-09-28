-- Multi-OS Kernel Database Schema
-- Supports PostgreSQL and SQLite

CREATE TABLE IF NOT EXISTS mos_hosts (
    host_id TEXT PRIMARY KEY,
    os TEXT NOT NULL CHECK (os IN ('linux', 'windows', 'macos')),
    arch TEXT NOT NULL CHECK (arch IN ('x86_64', 'arm64')),
    agent_version TEXT NOT NULL,
    capabilities JSONB,  -- SQLite: TEXT with JSON validation
    labels JSONB,        -- SQLite: TEXT with JSON validation
    status TEXT NOT NULL CHECK (status IN ('online', 'degraded', 'offline')),
    control_url TEXT,
    metrics_url TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mos_tasks (
    task_id TEXT PRIMARY KEY,
    host_id TEXT,
    spec JSONB NOT NULL,  -- SQLite: TEXT with JSON validation
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'running', 'completed', 'failed', 'timeout', 'killed')),
    exit_code INTEGER,
    logs_uri TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES mos_hosts(host_id)
);

CREATE TABLE IF NOT EXISTS mos_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (scope IN ('agent', 'image', 'vm', 'container')),
    payload_json JSONB NOT NULL,  -- SQLite: TEXT with JSON validation
    hash TEXT NOT NULL,
    host_id TEXT,
    uri TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES mos_hosts(host_id)
);

CREATE TABLE IF NOT EXISTS mos_fs_actions (
    action_id TEXT PRIMARY KEY,
    host_id TEXT,
    type TEXT NOT NULL CHECK (type IN ('read', 'write', 'list', 'move', 'copy', 'delete', 'hash')),
    path TEXT NOT NULL,
    content_b64 TEXT,
    recursive BOOLEAN DEFAULT FALSE,
    result JSONB,  -- SQLite: TEXT with JSON validation
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    FOREIGN KEY (host_id) REFERENCES mos_hosts(host_id)
);

CREATE TABLE IF NOT EXISTS mos_net_actions (
    action_id TEXT PRIMARY KEY,
    host_id TEXT,
    type TEXT NOT NULL CHECK (type IN ('fetch', 'post', 'port_check')),
    url TEXT NOT NULL,
    body TEXT,
    timeout_s INTEGER DEFAULT 30,
    result JSONB,  -- SQLite: TEXT with JSON validation
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    FOREIGN KEY (host_id) REFERENCES mos_hosts(host_id)
);

CREATE TABLE IF NOT EXISTS mos_runtime_specs (
    spec_id TEXT PRIMARY KEY,
    host_id TEXT,
    runtime TEXT NOT NULL CHECK (runtime IN ('python', 'conda', 'node', 'java', 'system')),
    version TEXT NOT NULL,
    env JSONB,       -- SQLite: TEXT with JSON validation
    packages JSONB,  -- SQLite: TEXT with JSON validation
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'installing', 'ready', 'failed')),
    job_id TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES mos_hosts(host_id)
);

CREATE TABLE IF NOT EXISTS mos_experiences (
    exp_id TEXT PRIMARY KEY,
    host_id TEXT,
    stage TEXT NOT NULL CHECK (stage IN ('schedule', 'pkg_setup', 'exec', 'fs', 'net', 'sandbox', 'snapshot', 'rollback')),
    metrics JSONB NOT NULL,  -- SQLite: TEXT with JSON validation
    labels JSONB,            -- SQLite: TEXT with JSON validation
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES mos_hosts(host_id)
);

CREATE TABLE IF NOT EXISTS mos_agent_rollouts (
    rollout_id TEXT PRIMARY KEY,
    from_version TEXT NOT NULL,
    to_version TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('blue', 'green')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    status TEXT NOT NULL DEFAULT 'started' CHECK (status IN ('started', 'in_progress', 'completed', 'failed', 'rolled_back')),
    affected_hosts JSONB,  -- SQLite: TEXT with JSON validation
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_mos_hosts_status ON mos_hosts(status);
CREATE INDEX IF NOT EXISTS idx_mos_hosts_os_arch ON mos_hosts(os, arch);
CREATE INDEX IF NOT EXISTS idx_mos_tasks_host_state ON mos_tasks(host_id, state);
CREATE INDEX IF NOT EXISTS idx_mos_tasks_created_at ON mos_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_mos_snapshots_scope ON mos_snapshots(scope);
CREATE INDEX IF NOT EXISTS idx_mos_snapshots_created_at ON mos_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_mos_fs_actions_host_status ON mos_fs_actions(host_id, status);
CREATE INDEX IF NOT EXISTS idx_mos_net_actions_host_status ON mos_net_actions(host_id, status);
CREATE INDEX IF NOT EXISTS idx_mos_runtime_specs_host_status ON mos_runtime_specs(host_id, status);
CREATE INDEX IF NOT EXISTS idx_mos_experiences_host_stage ON mos_experiences(host_id, stage);
CREATE INDEX IF NOT EXISTS idx_mos_experiences_timestamp ON mos_experiences(timestamp);

-- Views for common queries
CREATE VIEW IF NOT EXISTS v_host_summary AS
SELECT 
    h.host_id,
    h.os,
    h.arch,
    h.status,
    h.agent_version,
    COUNT(DISTINCT t.task_id) as total_tasks,
    COUNT(DISTINCT CASE WHEN t.state = 'running' THEN t.task_id END) as running_tasks,
    COUNT(DISTINCT CASE WHEN t.state = 'completed' THEN t.task_id END) as completed_tasks,
    COUNT(DISTINCT CASE WHEN t.state = 'failed' THEN t.task_id END) as failed_tasks,
    h.updated_at as last_seen
FROM mos_hosts h
LEFT JOIN mos_tasks t ON h.host_id = t.host_id
GROUP BY h.host_id, h.os, h.arch, h.status, h.agent_version, h.updated_at;

CREATE VIEW IF NOT EXISTS v_task_summary AS
SELECT 
    DATE(created_at) as date,
    state,
    COUNT(*) as task_count,
    AVG(CASE WHEN finished_at IS NOT NULL AND started_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (finished_at - started_at)) END) as avg_duration_seconds
FROM mos_tasks
GROUP BY DATE(created_at), state;

-- Trigger to update updated_at timestamp (PostgreSQL syntax)
-- For SQLite, this would need to be adapted
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_mos_hosts_updated_at
    BEFORE UPDATE ON mos_hosts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mos_runtime_specs_updated_at
    BEFORE UPDATE ON mos_runtime_specs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing (optional)
-- INSERT INTO mos_hosts (host_id, os, arch, agent_version, capabilities, labels, status, control_url, metrics_url)
-- VALUES 
--     ('host-linux-001', 'linux', 'x86_64', '2.4.1', '["process", "fs", "net", "pkg", "sandbox"]', '["region:us-west", "gpu:none"]', 'online', 'http://host1:8080/control', 'http://host1:8080/metrics'),
--     ('host-win-001', 'windows', 'x86_64', '2.3.0', '["process", "fs", "net", "pkg"]', '["region:us-east", "gpu:nvidia"]', 'online', 'http://host2:8080/control', 'http://host2:8080/metrics'),
--     ('host-mac-001', 'macos', 'arm64', '2.2.5', '["process", "fs", "net", "pkg", "sandbox"]', '["region:eu-west", "gpu:apple"]', 'online', 'http://host3:8080/control', 'http://host3:8080/metrics');