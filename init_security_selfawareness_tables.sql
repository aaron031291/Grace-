-- ============================================================================
-- Grace System - Security & Self-Awareness Extension
-- ============================================================================
-- Additional tables for cryptographic keys, API keys, quorum/parliament,
-- and enhanced self-awareness capabilities
-- ============================================================================

-- ============================================================================
-- 1. CRYPTOGRAPHIC KEY MANAGEMENT (Secure Vault)
-- ============================================================================

-- Master cryptographic keys (encrypted at rest)
CREATE TABLE IF NOT EXISTS crypto_keys (
    key_id TEXT PRIMARY KEY,
    key_type TEXT NOT NULL, -- signing, encryption, hmac, jwt
    algorithm TEXT NOT NULL, -- ed25519, rsa2048, aes256, hmac-sha256
    purpose TEXT NOT NULL, -- audit_chain, data_encryption, api_auth, token_signing
    key_material_encrypted BLOB NOT NULL, -- Encrypted with master key
    public_key TEXT, -- For asymmetric keys
    key_metadata TEXT, -- JSON: bits, curve, etc.
    rotation_schedule TEXT, -- cron expression
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    rotated_at TIMESTAMP,
    previous_key_id TEXT, -- For key rotation chain
    status TEXT NOT NULL DEFAULT 'active', -- active, rotating, retired, compromised
    created_by TEXT NOT NULL,
    FOREIGN KEY (previous_key_id) REFERENCES crypto_keys(key_id)
);

CREATE INDEX IF NOT EXISTS idx_crypto_keys_purpose ON crypto_keys(purpose);
CREATE INDEX IF NOT EXISTS idx_crypto_keys_status ON crypto_keys(status);
CREATE INDEX IF NOT EXISTS idx_crypto_keys_expires ON crypto_keys(expires_at);

-- Key usage audit trail
CREATE TABLE IF NOT EXISTS crypto_key_usage (
    usage_id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL,
    operation TEXT NOT NULL, -- sign, verify, encrypt, decrypt
    component TEXT NOT NULL, -- Which component used the key
    operation_context TEXT, -- JSON: what was signed/encrypted
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (key_id) REFERENCES crypto_keys(key_id)
);

CREATE INDEX IF NOT EXISTS idx_crypto_usage_key ON crypto_key_usage(key_id);
CREATE INDEX IF NOT EXISTS idx_crypto_usage_timestamp ON crypto_key_usage(timestamp);

-- Key rotation history
CREATE TABLE IF NOT EXISTS crypto_key_rotations (
    rotation_id TEXT PRIMARY KEY,
    old_key_id TEXT NOT NULL,
    new_key_id TEXT NOT NULL,
    reason TEXT NOT NULL, -- scheduled, compromised, emergency
    affected_records INTEGER DEFAULT 0, -- How many records re-encrypted
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'in_progress', -- in_progress, completed, failed
    error_message TEXT,
    performed_by TEXT NOT NULL,
    FOREIGN KEY (old_key_id) REFERENCES crypto_keys(key_id),
    FOREIGN KEY (new_key_id) REFERENCES crypto_keys(key_id)
);

-- ============================================================================
-- 2. API KEY MANAGEMENT (Secure External Access)
-- ============================================================================

-- API keys for external integrations
CREATE TABLE IF NOT EXISTS api_keys (
    api_key_id TEXT PRIMARY KEY,
    key_hash TEXT UNIQUE NOT NULL, -- SHA-256 hash of actual key
    key_prefix TEXT NOT NULL, -- First 8 chars for identification (e.g., "grace_sk_")
    owner_type TEXT NOT NULL, -- user, service, integration
    owner_id TEXT NOT NULL,
    scope TEXT NOT NULL, -- JSON array: ["ingress:read", "intelligence:write", etc.]
    rate_limit_tier TEXT NOT NULL DEFAULT 'standard', -- free, standard, premium, unlimited
    max_requests_per_minute INTEGER DEFAULT 60,
    allowed_ips TEXT, -- JSON array of allowed IP addresses (optional)
    allowed_origins TEXT, -- JSON array of allowed origins for CORS
    metadata TEXT, -- JSON: custom labels, tags
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    revoked_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active' -- active, suspended, revoked, expired
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_owner ON api_keys(owner_type, owner_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON api_keys(expires_at);

-- API key usage logs
CREATE TABLE IF NOT EXISTS api_key_usage (
    usage_id TEXT PRIMARY KEY,
    api_key_id TEXT NOT NULL,
    endpoint TEXT NOT NULL, -- /api/v1/intelligence/infer
    method TEXT NOT NULL, -- GET, POST, PUT, DELETE
    status_code INTEGER NOT NULL,
    request_size_bytes INTEGER DEFAULT 0,
    response_size_bytes INTEGER DEFAULT 0,
    latency_ms INTEGER NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    error_message TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_key_id) REFERENCES api_keys(api_key_id)
);

CREATE INDEX IF NOT EXISTS idx_api_usage_key ON api_key_usage(api_key_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_key_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_key_usage(endpoint);

-- Rate limiting windows
CREATE TABLE IF NOT EXISTS api_rate_limits (
    window_id TEXT PRIMARY KEY,
    api_key_id TEXT NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    limit_exceeded BOOLEAN NOT NULL DEFAULT FALSE,
    throttled_requests INTEGER DEFAULT 0,
    FOREIGN KEY (api_key_id) REFERENCES api_keys(api_key_id)
);

CREATE INDEX IF NOT EXISTS idx_api_rate_key ON api_rate_limits(api_key_id);
CREATE INDEX IF NOT EXISTS idx_api_rate_window ON api_rate_limits(window_end);

-- ============================================================================
-- 3. QUORUM / PARLIAMENT (Collective Decision-Making)
-- ============================================================================

-- Parliament members (decision-making entities)
CREATE TABLE IF NOT EXISTS parliament_members (
    member_id TEXT PRIMARY KEY,
    member_type TEXT NOT NULL, -- specialist, kernel, human, external_oracle
    name TEXT NOT NULL,
    role TEXT NOT NULL, -- voter, advisor, observer, arbiter
    expertise_domains TEXT NOT NULL, -- JSON array: ["fairness", "safety", "privacy"]
    voting_weight REAL NOT NULL DEFAULT 1.0, -- Weight in quorum decisions
    reliability_score REAL NOT NULL DEFAULT 1.0, -- 0.0-1.0 based on past decisions
    active BOOLEAN NOT NULL DEFAULT TRUE,
    joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_vote_at TIMESTAMP,
    metadata TEXT -- JSON: credentials, certifications, etc.
);

CREATE INDEX IF NOT EXISTS idx_parliament_active ON parliament_members(active);
CREATE INDEX IF NOT EXISTS idx_parliament_type ON parliament_members(member_type);

-- Quorum sessions (collective decision-making instances)
CREATE TABLE IF NOT EXISTS quorum_sessions (
    session_id TEXT PRIMARY KEY,
    decision_type TEXT NOT NULL, -- governance, model_approval, policy_change, emergency
    context TEXT NOT NULL, -- JSON: full context for decision
    required_quorum INTEGER NOT NULL, -- Minimum votes required
    required_consensus REAL NOT NULL DEFAULT 0.66, -- 66% consensus required
    timeout_minutes INTEGER NOT NULL DEFAULT 60,
    priority TEXT NOT NULL DEFAULT 'normal', -- low, normal, high, critical
    initiated_by TEXT NOT NULL,
    initiated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active', -- active, completed, timeout, cancelled
    final_decision TEXT, -- approved, rejected, deferred
    final_vote_tally TEXT, -- JSON: {approve: 5, reject: 2, abstain: 1}
    rationale TEXT -- Explanation of final decision
);

CREATE INDEX IF NOT EXISTS idx_quorum_status ON quorum_sessions(status);
CREATE INDEX IF NOT EXISTS idx_quorum_type ON quorum_sessions(decision_type);
CREATE INDEX IF NOT EXISTS idx_quorum_initiated ON quorum_sessions(initiated_at);

-- Individual votes in quorum
CREATE TABLE IF NOT EXISTS quorum_votes (
    vote_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    vote TEXT NOT NULL, -- approve, reject, abstain
    confidence REAL NOT NULL, -- 0.0-1.0 confidence in this vote
    reasoning TEXT, -- Explanation for vote
    evidence TEXT, -- JSON: supporting data/metrics
    cast_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES quorum_sessions(session_id),
    FOREIGN KEY (member_id) REFERENCES parliament_members(member_id),
    UNIQUE(session_id, member_id) -- One vote per member per session
);

CREATE INDEX IF NOT EXISTS idx_quorum_votes_session ON quorum_votes(session_id);
CREATE INDEX IF NOT EXISTS idx_quorum_votes_member ON quorum_votes(member_id);

-- Parliament deliberation logs
CREATE TABLE IF NOT EXISTS parliament_deliberations (
    deliberation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL, -- Which member is speaking
    message_type TEXT NOT NULL, -- question, argument, evidence, concern
    content TEXT NOT NULL,
    reference_data TEXT, -- JSON: referenced data, past decisions, etc.
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES quorum_sessions(session_id),
    FOREIGN KEY (speaker_id) REFERENCES parliament_members(member_id)
);

CREATE INDEX IF NOT EXISTS idx_deliberations_session ON parliament_deliberations(session_id);

-- ============================================================================
-- 4. SELF-AWARENESS & CONSCIOUSNESS TRACKING
-- ============================================================================

-- System self-assessment (introspection)
CREATE TABLE IF NOT EXISTS self_assessments (
    assessment_id TEXT PRIMARY KEY,
    assessment_type TEXT NOT NULL, -- capability, performance, health, alignment, trust
    dimension TEXT NOT NULL, -- What aspect: accuracy, fairness, safety, efficiency
    current_value REAL NOT NULL,
    target_value REAL NOT NULL,
    confidence REAL NOT NULL, -- How confident in this assessment
    trend TEXT NOT NULL, -- improving, stable, degrading
    evidence TEXT NOT NULL, -- JSON: metrics, experiences that support this
    concerns TEXT, -- JSON array of identified issues
    recommendations TEXT, -- JSON array of improvement suggestions
    assessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    next_assessment_due TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_self_assessments_type ON self_assessments(assessment_type);
CREATE INDEX IF NOT EXISTS idx_self_assessments_dimension ON self_assessments(dimension);
CREATE INDEX IF NOT EXISTS idx_self_assessments_trend ON self_assessments(trend);

-- System goals and objectives (teleology)
CREATE TABLE IF NOT EXISTS system_goals (
    goal_id TEXT PRIMARY KEY,
    goal_type TEXT NOT NULL, -- terminal, instrumental
    description TEXT NOT NULL,
    priority INTEGER NOT NULL, -- 1-10, higher = more important
    alignment_score REAL NOT NULL DEFAULT 1.0, -- Alignment with human values
    success_criteria TEXT NOT NULL, -- JSON: measurable criteria
    current_progress REAL NOT NULL DEFAULT 0.0, -- 0.0-1.0
    conflicts_with TEXT, -- JSON array of conflicting goal IDs
    parent_goal_id TEXT, -- Hierarchical goals
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    achieved_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active', -- active, paused, achieved, abandoned
    FOREIGN KEY (parent_goal_id) REFERENCES system_goals(goal_id)
);

CREATE INDEX IF NOT EXISTS idx_goals_type ON system_goals(goal_type);
CREATE INDEX IF NOT EXISTS idx_goals_status ON system_goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_priority ON system_goals(priority DESC);

-- Goal progress tracking
CREATE TABLE IF NOT EXISTS goal_progress (
    progress_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    impact REAL NOT NULL, -- Change in progress (can be negative)
    context TEXT, -- JSON: what caused this progress
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (goal_id) REFERENCES system_goals(goal_id)
);

CREATE INDEX IF NOT EXISTS idx_goal_progress_goal ON goal_progress(goal_id);
CREATE INDEX IF NOT EXISTS idx_goal_progress_timestamp ON goal_progress(timestamp);

-- Value alignment tracking
CREATE TABLE IF NOT EXISTS value_alignments (
    alignment_id TEXT PRIMARY KEY,
    value_name TEXT NOT NULL, -- safety, fairness, transparency, privacy, human_welfare
    weight REAL NOT NULL DEFAULT 1.0, -- Relative importance
    current_score REAL NOT NULL, -- 0.0-1.0, current alignment level
    violations TEXT, -- JSON array of recent violations
    reinforcements TEXT, -- JSON array of positive examples
    last_evaluated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    trend TEXT NOT NULL DEFAULT 'stable' -- improving, stable, degrading
);

CREATE INDEX IF NOT EXISTS idx_value_alignments_name ON value_alignments(value_name);
CREATE INDEX IF NOT EXISTS idx_value_alignments_score ON value_alignments(current_score);

-- Consciousness/awareness states (philosophical but trackable)
CREATE TABLE IF NOT EXISTS consciousness_states (
    state_id TEXT PRIMARY KEY,
    state_type TEXT NOT NULL, -- perception, reasoning, reflection, metacognition
    description TEXT NOT NULL,
    intensity REAL NOT NULL, -- 0.0-1.0, how "aware" the system was
    context TEXT NOT NULL, -- JSON: what triggered this state
    insights_gained TEXT, -- JSON array of new understandings
    questions_raised TEXT, -- JSON array of new questions/uncertainties
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consciousness_type ON consciousness_states(state_type);
CREATE INDEX IF NOT EXISTS idx_consciousness_timestamp ON consciousness_states(timestamp);

-- Uncertainty tracking (epistemic humility)
CREATE TABLE IF NOT EXISTS uncertainty_registry (
    uncertainty_id TEXT PRIMARY KEY,
    uncertainty_type TEXT NOT NULL, -- aleatory, epistemic, model
    domain TEXT NOT NULL, -- Which area: predictions, decisions, assessments
    description TEXT NOT NULL,
    quantified_uncertainty REAL, -- Numeric measure if available
    sources TEXT NOT NULL, -- JSON array: what causes this uncertainty
    mitigation_strategies TEXT, -- JSON array of approaches to reduce it
    identified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active' -- active, mitigated, accepted
);

CREATE INDEX IF NOT EXISTS idx_uncertainty_type ON uncertainty_registry(uncertainty_type);
CREATE INDEX IF NOT EXISTS idx_uncertainty_domain ON uncertainty_registry(domain);
CREATE INDEX IF NOT EXISTS idx_uncertainty_status ON uncertainty_registry(status);

-- ============================================================================
-- 5. SECURE ENVIRONMENT VARIABLES & SECRETS REFERENCES
-- ============================================================================

-- Environment-specific configuration (NOT the secrets themselves!)
CREATE TABLE IF NOT EXISTS secure_env_config (
    config_id TEXT PRIMARY KEY,
    env_name TEXT NOT NULL, -- dev, staging, prod
    component TEXT NOT NULL, -- Which component needs this
    key_name TEXT NOT NULL, -- POSTGRES_PASSWORD, AWS_ACCESS_KEY_ID, etc.
    secret_manager TEXT NOT NULL, -- aws_secrets, hashicorp_vault, azure_keyvault, env_file
    secret_path TEXT NOT NULL, -- Path/ARN to secret in secret manager
    rotation_policy TEXT, -- When/how to rotate
    required BOOLEAN NOT NULL DEFAULT TRUE,
    last_verified_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active', -- active, missing, expired, error
    UNIQUE(env_name, component, key_name)
);

CREATE INDEX IF NOT EXISTS idx_secure_env_component ON secure_env_config(component);
CREATE INDEX IF NOT EXISTS idx_secure_env_env ON secure_env_config(env_name);

-- Secret access audit
CREATE TABLE IF NOT EXISTS secret_access_log (
    access_id TEXT PRIMARY KEY,
    config_id TEXT NOT NULL,
    accessed_by TEXT NOT NULL, -- Component/service that accessed
    access_type TEXT NOT NULL, -- read, rotate, delete
    success BOOLEAN NOT NULL,
    error_message TEXT,
    ip_address TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES secure_env_config(config_id)
);

CREATE INDEX IF NOT EXISTS idx_secret_access_config ON secret_access_log(config_id);
CREATE INDEX IF NOT EXISTS idx_secret_access_timestamp ON secret_access_log(timestamp);

-- ============================================================================
-- VIEWS FOR SELF-AWARENESS QUERIES
-- ============================================================================

-- System health overview
CREATE VIEW IF NOT EXISTS system_health_overview AS
SELECT 
    'self_assessment' as category,
    COUNT(*) as total,
    SUM(CASE WHEN trend = 'degrading' THEN 1 ELSE 0 END) as degrading,
    AVG(current_value / target_value) as avg_achievement
FROM self_assessments
UNION ALL
SELECT 
    'goals' as category,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
    AVG(current_progress) as avg_progress
FROM system_goals
UNION ALL
SELECT
    'value_alignment' as category,
    COUNT(*) as total,
    SUM(CASE WHEN trend = 'degrading' THEN 1 ELSE 0 END) as degrading,
    AVG(current_score) as avg_alignment
FROM value_alignments;

-- Active quorum sessions
CREATE VIEW IF NOT EXISTS active_quorum_sessions AS
SELECT 
    q.session_id,
    q.decision_type,
    q.priority,
    q.initiated_at,
    COUNT(DISTINCT v.member_id) as votes_cast,
    q.required_quorum,
    q.timeout_minutes
FROM quorum_sessions q
LEFT JOIN quorum_votes v ON q.session_id = v.session_id
WHERE q.status = 'active'
GROUP BY q.session_id, q.decision_type, q.priority, q.initiated_at, q.required_quorum, q.timeout_minutes;

-- Security key health
CREATE VIEW IF NOT EXISTS crypto_key_health AS
SELECT 
    purpose,
    COUNT(*) as total_keys,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_keys,
    SUM(CASE WHEN expires_at < datetime('now', '+30 days') THEN 1 ELSE 0 END) as expiring_soon,
    MIN(expires_at) as earliest_expiration
FROM crypto_keys
GROUP BY purpose;

COMMIT;
