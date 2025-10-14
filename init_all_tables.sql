-- ============================================================================
-- Grace System - Complete Database Initialization
-- ============================================================================
-- This script creates all database tables for all Grace components and kernels
-- Compatible with SQLite for development and PostgreSQL for production
-- ============================================================================

-- ============================================================================
-- 1. AUDIT & GOVERNANCE CORE
-- ============================================================================

-- Immutable audit log for all system operations
CREATE TABLE IF NOT EXISTS audit_logs (
    entry_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    data_json TEXT NOT NULL,
    transparency_level TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    hash TEXT NOT NULL,
    previous_hash TEXT,
    chain_hash TEXT,
    chain_position INTEGER,
    verified BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_category ON audit_logs(category);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_chain_position ON audit_logs(chain_position);

-- Chain verification tracking
CREATE TABLE IF NOT EXISTS chain_verification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_position INTEGER NOT NULL,
    end_position INTEGER NOT NULL,
    chain_hash TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    verification_result TEXT NOT NULL
);

-- Log categories metadata
CREATE TABLE IF NOT EXISTS log_categories (
    category TEXT PRIMARY KEY,
    entry_count INTEGER DEFAULT 0,
    last_entry_timestamp TEXT,
    transparency_level TEXT
);

-- ============================================================================
-- 2. INGRESS KERNEL - Data Pipeline (Bronze/Silver/Gold)
-- ============================================================================

-- Source registry
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    auth_mode TEXT NOT NULL DEFAULT 'none',
    secrets_ref TEXT,
    schedule TEXT NOT NULL DEFAULT 'manual',
    parser TEXT NOT NULL,
    parser_opts TEXT, -- JSON
    target_contract TEXT NOT NULL,
    retention_days INTEGER NOT NULL DEFAULT 365,
    pii_policy TEXT NOT NULL DEFAULT 'mask',
    governance_label TEXT NOT NULL DEFAULT 'internal',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sources_kind ON sources(kind);
CREATE INDEX IF NOT EXISTS idx_sources_enabled ON sources(enabled);

-- Source health monitoring
CREATE TABLE IF NOT EXISTS source_health (
    source_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'unknown',
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latency_ms INTEGER DEFAULT 0,
    backlog INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_ok TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- Bronze tier: Raw events (append-only)
CREATE TABLE IF NOT EXISTS bronze_raw_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    source_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    offset TEXT NOT NULL,
    watermark TIMESTAMP NOT NULL,
    content_hash TEXT NOT NULL,
    headers TEXT, -- JSON
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_uri TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bronze_source_id ON bronze_raw_events(source_id);
CREATE INDEX IF NOT EXISTS idx_bronze_content_hash ON bronze_raw_events(content_hash);
CREATE INDEX IF NOT EXISTS idx_bronze_ingestion_ts ON bronze_raw_events(ingestion_ts);

-- Silver tier: Normalized records
CREATE TABLE IF NOT EXISTS silver_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT UNIQUE NOT NULL,
    contract TEXT NOT NULL,
    source_id TEXT NOT NULL,
    raw_event_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    parser TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    validity_score REAL NOT NULL DEFAULT 0.0,
    completeness REAL NOT NULL DEFAULT 0.0,
    freshness_minutes REAL NOT NULL DEFAULT 0.0,
    trust_score REAL NOT NULL DEFAULT 0.0,
    pii_flags TEXT, -- JSON array
    transforms TEXT NOT NULL, -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_uri TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    FOREIGN KEY (raw_event_id) REFERENCES bronze_raw_events(event_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_silver_contract ON silver_records(contract);
CREATE INDEX IF NOT EXISTS idx_silver_source_id ON silver_records(source_id);
CREATE INDEX IF NOT EXISTS idx_silver_trust_score ON silver_records(trust_score);

-- Silver: Articles
CREATE TABLE IF NOT EXISTS silver_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    published_at TIMESTAMP,
    url TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    text_uri TEXT NOT NULL,
    topics TEXT, -- JSON array
    entities TEXT, -- JSON object
    embeddings_ref TEXT,
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_silver_articles_published ON silver_articles(published_at);

-- Silver: Transcripts
CREATE TABLE IF NOT EXISTS silver_transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT UNIQUE NOT NULL,
    media_id TEXT NOT NULL,
    start_at TIMESTAMP,
    duration_s REAL,
    lang TEXT NOT NULL DEFAULT 'en',
    segments_uri TEXT NOT NULL,
    summary TEXT,
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE
);

-- Silver: Tabular data
CREATE TABLE IF NOT EXISTS silver_tabular (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT UNIQUE NOT NULL,
    dataset_id TEXT NOT NULL,
    columns TEXT NOT NULL, -- JSON array
    rows_uri TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE
);

-- Gold tier: Article topics
CREATE TABLE IF NOT EXISTS gold_article_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_record_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    extraction_method TEXT NOT NULL DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_record_id) REFERENCES silver_articles(record_id) ON DELETE CASCADE,
    UNIQUE(article_record_id, topic)
);

-- Gold tier: Entity mentions
CREATE TABLE IF NOT EXISTS gold_entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    context TEXT,
    position_start INTEGER,
    position_end INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_gold_entity ON gold_entity_mentions(entity);

-- Gold tier: Feature datasets
CREATE TABLE IF NOT EXISTS gold_feature_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT UNIQUE NOT NULL,
    dataset_version TEXT NOT NULL DEFAULT '1.0.0',
    feature_count INTEGER NOT NULL DEFAULT 0,
    source_contracts TEXT NOT NULL, -- JSON array
    storage_uri TEXT NOT NULL,
    train_split_uri TEXT,
    validation_split_uri TEXT,
    test_split_uri TEXT,
    metadata TEXT, -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ingress metrics
CREATE TABLE IF NOT EXISTS processing_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    throughput_rps REAL DEFAULT 0.0,
    error_rate REAL DEFAULT 0.0,
    avg_latency_ms REAL DEFAULT 0.0,
    schema_violations INTEGER DEFAULT 0,
    pii_incidents INTEGER DEFAULT 0,
    dedup_rate REAL DEFAULT 0.0,
    trust_mean REAL DEFAULT 0.0,
    completeness_mean REAL DEFAULT 0.0,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- Validation failures
CREATE TABLE IF NOT EXISTS validation_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT,
    raw_event_id TEXT,
    source_id TEXT NOT NULL,
    policy_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'error',
    reasons TEXT NOT NULL, -- JSON array
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- Trust score history
CREATE TABLE IF NOT EXISTS source_trust_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    old_reputation REAL NOT NULL,
    new_reputation REAL NOT NULL,
    outcome_score REAL NOT NULL,
    context TEXT, -- JSON
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- ============================================================================
-- 3. INTELLIGENCE KERNEL - Inference & Routing
-- ============================================================================

-- Intelligence requests
CREATE TABLE IF NOT EXISTS intel_requests (
    req_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    modality TEXT,
    context_json TEXT NOT NULL,
    input_data_json TEXT,
    constraints_json TEXT,
    user_ctx_json TEXT,
    latency_budget_ms INTEGER,
    cost_budget_units REAL,
    explanation_required BOOLEAN DEFAULT FALSE,
    canary_allowed BOOLEAN DEFAULT TRUE,
    segment TEXT,
    env TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'received'
);

CREATE INDEX IF NOT EXISTS idx_intel_requests_created ON intel_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_intel_requests_status ON intel_requests(status);

-- Execution plans
CREATE TABLE IF NOT EXISTS intel_plans (
    plan_id TEXT PRIMARY KEY,
    req_id TEXT NOT NULL,
    route_json TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    risk_level TEXT,
    governance_approved BOOLEAN DEFAULT FALSE,
    governance_decision_id TEXT,
    pre_flight_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'ready',
    FOREIGN KEY (req_id) REFERENCES intel_requests(req_id)
);

-- Intelligence results
CREATE TABLE IF NOT EXISTS intel_results (
    req_id TEXT PRIMARY KEY,
    plan_id TEXT,
    outputs_json TEXT,
    metrics_json TEXT,
    explanations_json TEXT,
    uncertainties_json TEXT,
    lineage_json TEXT,
    governance_json TEXT,
    timing_json TEXT,
    shadow_result_json TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (req_id) REFERENCES intel_requests(req_id),
    FOREIGN KEY (plan_id) REFERENCES intel_plans(plan_id)
);

-- Specialist reports
CREATE TABLE IF NOT EXISTS intel_specialist_reports (
    report_id TEXT PRIMARY KEY,
    req_id TEXT,
    specialist_name TEXT NOT NULL,
    model_key TEXT,
    candidates_json TEXT,
    metrics_json TEXT,
    uncertainties_json TEXT,
    explanations_json TEXT,
    risks_json TEXT,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (req_id) REFERENCES intel_requests(req_id)
);

-- Intelligence experiences
CREATE TABLE IF NOT EXISTS intel_experiences (
    exp_id TEXT PRIMARY KEY,
    req_id TEXT,
    stage TEXT,
    metrics_json TEXT,
    segment TEXT,
    model_keys TEXT,
    ensemble_type TEXT,
    canary_pct INTEGER,
    shadow_enabled BOOLEAN DEFAULT FALSE,
    governance_approved BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (req_id) REFERENCES intel_requests(req_id)
);

-- Canary deployments
CREATE TABLE IF NOT EXISTS intel_canary_deployments (
    deployment_id TEXT PRIMARY KEY,
    model_key TEXT NOT NULL,
    target_version TEXT NOT NULL,
    current_step INTEGER DEFAULT 0,
    steps TEXT NOT NULL, -- JSON array
    success_metrics_json TEXT,
    current_metrics_json TEXT,
    status TEXT DEFAULT 'active',
    auto_promote_threshold REAL DEFAULT 0.95,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Shadow deployments
CREATE TABLE IF NOT EXISTS intel_shadow_deployments (
    deployment_id TEXT PRIMARY KEY,
    primary_model_key TEXT NOT NULL,
    shadow_model_key TEXT NOT NULL,
    traffic_pct INTEGER DEFAULT 100,
    agreement_threshold REAL DEFAULT 0.90,
    current_agreement REAL,
    sample_count INTEGER DEFAULT 0,
    agreement_history_json TEXT,
    status TEXT DEFAULT 'active',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Policy violations
CREATE TABLE IF NOT EXISTS intel_policy_violations (
    violation_id TEXT PRIMARY KEY,
    req_id TEXT,
    plan_id TEXT,
    violation_type TEXT NOT NULL,
    severity TEXT,
    required_value REAL,
    actual_value REAL,
    policy_version TEXT,
    remediation_json TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (req_id) REFERENCES intel_requests(req_id),
    FOREIGN KEY (plan_id) REFERENCES intel_plans(plan_id)
);

-- Intelligence snapshots
CREATE TABLE IF NOT EXISTS intel_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    snapshot_type TEXT DEFAULT 'intelligence',
    payload_json TEXT NOT NULL,
    version TEXT,
    hash TEXT NOT NULL,
    size_bytes INTEGER,
    compressed BOOLEAN DEFAULT FALSE,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    restored_at TIMESTAMP,
    restore_count INTEGER DEFAULT 0,
    description TEXT
);

-- ============================================================================
-- 4. LEARNING KERNEL - Data-Centric ML
-- ============================================================================

-- Dataset registry
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    modality TEXT NOT NULL,
    schema_json TEXT,
    default_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dataset versions
CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    source_refs_json TEXT NOT NULL,
    row_count INTEGER,
    byte_size INTEGER,
    stats_json TEXT,
    train_split REAL DEFAULT 0.8,
    valid_split REAL DEFAULT 0.1,
    test_split REAL DEFAULT 0.1,
    feature_view TEXT,
    lineage_hash TEXT,
    governance_label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    UNIQUE(dataset_id, version)
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset ON dataset_versions(dataset_id);

-- Label policies
CREATE TABLE IF NOT EXISTS label_policies (
    policy_id TEXT PRIMARY KEY,
    rubric_json TEXT NOT NULL,
    gold_ratio REAL DEFAULT 0.05,
    dual_label_ratio REAL DEFAULT 0.1,
    min_agreement REAL DEFAULT 0.8,
    allow_pii BOOLEAN DEFAULT FALSE,
    mask_pii BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Label tasks
CREATE TABLE IF NOT EXISTS label_tasks (
    task_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    assign_strategy TEXT DEFAULT 'auto',
    items_json TEXT NOT NULL,
    priority TEXT DEFAULT 'normal',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (policy_id) REFERENCES label_policies(policy_id)
);

-- Labels
CREATE TABLE IF NOT EXISTS labels (
    label_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    y_value TEXT NOT NULL,
    evidence_json TEXT,
    weak_sources_json TEXT,
    agreement REAL,
    gold_correct BOOLEAN,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES label_tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_labels_task ON labels(task_id);
CREATE INDEX IF NOT EXISTS idx_labels_item ON labels(item_id);

-- Active learning queries
CREATE TABLE IF NOT EXISTS active_queries (
    query_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    strategy TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    segment_filters_json TEXT,
    min_confidence REAL,
    results_json TEXT,
    query_gain_f1 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Curriculum learning specs
CREATE TABLE IF NOT EXISTS curriculum_specs (
    spec_id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    rules_json TEXT NOT NULL,
    dataset_id TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Augmentation specs
CREATE TABLE IF NOT EXISTS augment_specs (
    spec_id TEXT PRIMARY KEY,
    modality TEXT NOT NULL,
    ops_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS augment_applications (
    application_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    spec_id TEXT NOT NULL,
    delta_rows INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (spec_id) REFERENCES augment_specs(spec_id)
);

-- Feature views
CREATE TABLE IF NOT EXISTS feature_views (
    view_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    view_uri TEXT NOT NULL,
    format TEXT DEFAULT 'parquet',
    build_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    built_at TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    UNIQUE(dataset_id, version)
);

-- Quality reports
CREATE TABLE IF NOT EXISTS quality_reports (
    report_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT,
    leakage_flags INTEGER DEFAULT 0,
    bias_metrics_json TEXT,
    coverage_ratio REAL,
    label_agreement REAL,
    drift_psi REAL,
    noise_estimate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Learning experiences
CREATE TABLE IF NOT EXISTS learning_experiences (
    exp_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    segment TEXT,
    dataset_id TEXT,
    version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Weak labelers
CREATE TABLE IF NOT EXISTS weak_labelers (
    labeler_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    labeler_type TEXT DEFAULT 'rule',
    threshold REAL DEFAULT 0.65,
    rules_json TEXT,
    active BOOLEAN DEFAULT TRUE,
    precision REAL,
    recall REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weak_predictions (
    prediction_id TEXT PRIMARY KEY,
    labeler_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (labeler_id) REFERENCES weak_labelers(labeler_id)
);

CREATE INDEX IF NOT EXISTS idx_weak_predictions_labeler ON weak_predictions(labeler_id);
CREATE INDEX IF NOT EXISTS idx_weak_predictions_item ON weak_predictions(item_id);

-- Learning snapshots
CREATE TABLE IF NOT EXISTS learning_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    datasets_json TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    feature_views_json TEXT,
    policies_version TEXT,
    active_query_config_json TEXT,
    weak_labelers_json TEXT,
    augmentation_config_json TEXT,
    hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 5. MEMORY KERNEL - Lightning Cache & Fusion Storage
-- ============================================================================

-- Lightning memory (fast cache)
CREATE TABLE IF NOT EXISTS lightning_memory (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/json',
    ttl_seconds INTEGER,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lightning_expires ON lightning_memory(expires_at);
CREATE INDEX IF NOT EXISTS idx_lightning_accessed ON lightning_memory(last_accessed_at);

-- Fusion memory (persistent storage)
CREATE TABLE IF NOT EXISTS fusion_memory (
    entry_id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    content BLOB NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/json',
    size_bytes INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    compressed BOOLEAN NOT NULL DEFAULT FALSE,
    compression_type TEXT,
    metadata TEXT, -- JSON
    tags TEXT, -- JSON array
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    accessed_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fusion_key ON fusion_memory(key);
CREATE INDEX IF NOT EXISTS idx_fusion_created ON fusion_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_fusion_accessed ON fusion_memory(last_accessed_at);

-- Memory access patterns
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    access_type TEXT NOT NULL,
    access_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    response_time_ms REAL NOT NULL,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
    user_context TEXT,
    query_context TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_access_patterns_key ON memory_access_patterns(key);
CREATE INDEX IF NOT EXISTS idx_access_patterns_timestamp ON memory_access_patterns(access_timestamp);

-- Librarian search index
CREATE TABLE IF NOT EXISTS librarian_index (
    id TEXT PRIMARY KEY,
    entry_key TEXT NOT NULL,
    index_type TEXT NOT NULL,
    index_value TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_librarian_key ON librarian_index(entry_key);
CREATE INDEX IF NOT EXISTS idx_librarian_type ON librarian_index(index_type);

-- Memory statistics
CREATE TABLE IF NOT EXISTS memory_stats (
    id TEXT PRIMARY KEY,
    stat_type TEXT NOT NULL,
    component TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_memory_stats_component ON memory_stats(component);
CREATE INDEX IF NOT EXISTS idx_memory_stats_recorded ON memory_stats(recorded_at);

-- Memory operations log
CREATE TABLE IF NOT EXISTS memory_operations (
    id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    key TEXT NOT NULL,
    component TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    duration_ms REAL NOT NULL,
    size_bytes INTEGER,
    error_message TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_memory_ops_timestamp ON memory_operations(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_ops_component ON memory_operations(component);

-- Memory snapshots
CREATE TABLE IF NOT EXISTS memory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    lightning_state TEXT NOT NULL, -- JSON
    fusion_index TEXT NOT NULL, -- JSON
    librarian_config TEXT NOT NULL, -- JSON
    access_patterns TEXT NOT NULL, -- JSON
    statistics TEXT NOT NULL, -- JSON
    snapshot_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    size_bytes INTEGER
);

-- Memory cleanup tasks
CREATE TABLE IF NOT EXISTS memory_cleanup_tasks (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    component TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    scheduled_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    items_processed INTEGER DEFAULT 0,
    items_total INTEGER DEFAULT 0,
    error_message TEXT,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_cleanup_tasks_status ON memory_cleanup_tasks(status);

-- ============================================================================
-- 6. ORCHESTRATION KERNEL - State Management & Scheduling
-- ============================================================================

-- Orchestration state
CREATE TABLE IF NOT EXISTS orchestration_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL,
    context TEXT NOT NULL, -- JSON
    timestamp TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_state_timestamp ON orchestration_state(timestamp);

-- Orchestration loops
CREATE TABLE IF NOT EXISTS orchestration_loops (
    loop_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    priority INTEGER NOT NULL,
    interval_s INTEGER NOT NULL,
    kernels TEXT NOT NULL, -- JSON array
    policies TEXT NOT NULL, -- JSON
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
    stage TEXT NOT NULL,
    inputs TEXT NOT NULL, -- JSON
    outputs TEXT, -- JSON
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT, -- JSON
    priority INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    timeout_minutes INTEGER DEFAULT 30,
    FOREIGN KEY (loop_id) REFERENCES orchestration_loops(loop_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_loop_id ON orchestration_tasks(loop_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON orchestration_tasks(status);

-- System policies
CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    rules TEXT NOT NULL, -- JSON
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    applied_count INTEGER DEFAULT 0,
    violation_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_policies_type ON policies(type);
CREATE INDEX IF NOT EXISTS idx_policies_scope ON policies(scope);

-- State transitions
CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    trigger TEXT NOT NULL,
    context TEXT NOT NULL, -- JSON
    timestamp TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON state_transitions(timestamp);

-- ============================================================================
-- 7. RESILIENCE KERNEL - Circuit Breakers & Degradation
-- ============================================================================

-- Circuit breaker states
CREATE TABLE IF NOT EXISTS circuit_breakers (
    id TEXT PRIMARY KEY,
    component_name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'closed',
    failure_count INTEGER NOT NULL DEFAULT 0,
    failure_threshold INTEGER NOT NULL DEFAULT 5,
    timeout_duration_seconds INTEGER NOT NULL DEFAULT 60,
    last_failure_time TIMESTAMP,
    last_success_time TIMESTAMP,
    next_attempt_time TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_circuit_breakers_component ON circuit_breakers(component_name);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_state ON circuit_breakers(state);

-- Degradation policies
CREATE TABLE IF NOT EXISTS degradation_policies (
    id TEXT PRIMARY KEY,
    policy_name TEXT UNIQUE NOT NULL,
    component_pattern TEXT NOT NULL,
    trigger_conditions TEXT NOT NULL, -- JSON
    degradation_actions TEXT NOT NULL, -- JSON
    priority INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Active degradations
CREATE TABLE IF NOT EXISTS active_degradations (
    id TEXT PRIMARY KEY,
    component_name TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    degradation_level TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expected_duration_seconds INTEGER,
    metadata TEXT, -- JSON
    FOREIGN KEY (policy_id) REFERENCES degradation_policies(id)
);

-- Resilience incidents
CREATE TABLE IF NOT EXISTS resilience_incidents (
    id TEXT PRIMARY KEY,
    incident_type TEXT NOT NULL,
    component_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_summary TEXT,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_incidents_component ON resilience_incidents(component_name);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON resilience_incidents(status);

-- Rate limits
CREATE TABLE IF NOT EXISTS rate_limits (
    id TEXT PRIMARY KEY,
    resource_name TEXT NOT NULL,
    limit_type TEXT NOT NULL,
    max_requests INTEGER NOT NULL,
    current_requests INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reset_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- SLI measurements
CREATE TABLE IF NOT EXISTS sli_measurements (
    id TEXT PRIMARY KEY,
    sli_name TEXT NOT NULL,
    component_name TEXT NOT NULL,
    measurement_type TEXT NOT NULL,
    value REAL NOT NULL,
    target_value REAL NOT NULL,
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_sli_measurements_component ON sli_measurements(component_name);

-- Recovery actions
CREATE TABLE IF NOT EXISTS recovery_actions (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_details TEXT NOT NULL, -- JSON
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_summary TEXT,
    FOREIGN KEY (incident_id) REFERENCES resilience_incidents(id)
);

-- Resilience snapshots
CREATE TABLE IF NOT EXISTS resilience_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    component_states TEXT NOT NULL, -- JSON
    circuit_breaker_states TEXT NOT NULL, -- JSON
    active_policies TEXT NOT NULL, -- JSON
    degradation_states TEXT NOT NULL, -- JSON
    rate_limit_states TEXT NOT NULL, -- JSON
    snapshot_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- ============================================================================
-- 8. MLT (Memory, Learning, Trust) CORE
-- ============================================================================

-- MLT experiences
CREATE TABLE IF NOT EXISTS mlt_experiences (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    task TEXT NOT NULL,
    context_json TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    gt_lag_s INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mlt_experiences_source_task ON mlt_experiences(source, task);
CREATE INDEX IF NOT EXISTS idx_mlt_experiences_ts ON mlt_experiences(ts);

-- MLT insights
CREATE TABLE IF NOT EXISTS mlt_insights (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    scope TEXT NOT NULL,
    target_ref TEXT,
    evidence_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    recommendation TEXT,
    ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mlt_insights_type_scope ON mlt_insights(type, scope);
CREATE INDEX IF NOT EXISTS idx_mlt_insights_ts ON mlt_insights(ts);

-- MLT plans
CREATE TABLE IF NOT EXISTS mlt_plans (
    id TEXT PRIMARY KEY,
    plan_json TEXT NOT NULL,
    expected_effect TEXT NOT NULL, -- JSON
    risk_controls TEXT NOT NULL, -- JSON
    status TEXT NOT NULL DEFAULT 'pending',
    rationale TEXT,
    correlation_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mlt_plans_status ON mlt_plans(status);
CREATE INDEX IF NOT EXISTS idx_mlt_plans_created_at ON mlt_plans(created_at);

-- MLT snapshots
CREATE TABLE IF NOT EXISTS mlt_snapshots (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mlt_snapshots_type ON mlt_snapshots(type);
CREATE INDEX IF NOT EXISTS idx_mlt_snapshots_created_at ON mlt_snapshots(created_at);

-- MLT specialist reports
CREATE TABLE IF NOT EXISTS mlt_specialist_reports (
    id TEXT PRIMARY KEY,
    specialist TEXT NOT NULL,
    task TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    dataset_id TEXT,
    notes TEXT,
    version TEXT NOT NULL,
    ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mlt_specialist_reports_specialist ON mlt_specialist_reports(specialist);
CREATE INDEX IF NOT EXISTS idx_mlt_specialist_reports_task ON mlt_specialist_reports(task);

-- ============================================================================
-- 9. MLDL (ML Deployment & Lifecycle) COMPONENTS
-- ============================================================================

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    model_id TEXT UNIQUE NOT NULL,
    task TEXT NOT NULL,
    version TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    metrics TEXT NOT NULL, -- JSON
    params TEXT NOT NULL, -- JSON
    tags TEXT, -- JSON
    signature TEXT NOT NULL, -- JSON
    registered_by TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'development',
    governance_approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_models_model_id ON models(model_id);
CREATE INDEX IF NOT EXISTS idx_models_task ON models(task);
CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage);

-- Model approvals
CREATE TABLE IF NOT EXISTS model_approvals (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    approver TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    policy_checks TEXT, -- JSON
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- Model deployments
CREATE TABLE IF NOT EXISTS model_deployments (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    deployment_type TEXT NOT NULL,
    endpoint TEXT,
    replicas INTEGER DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending',
    deployed_by TEXT NOT NULL,
    deployed_at TIMESTAMP,
    undeployed_at TIMESTAMP,
    config TEXT, -- JSON
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- Deployment metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT, -- JSON
    FOREIGN KEY (deployment_id) REFERENCES model_deployments(id)
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_deployment ON model_metrics(deployment_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp);

-- SLO violations
CREATE TABLE IF NOT EXISTS slo_violations (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    slo_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    threshold_value REAL NOT NULL,
    actual_value REAL NOT NULL,
    severity TEXT NOT NULL,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution TEXT,
    FOREIGN KEY (deployment_id) REFERENCES model_deployments(id)
);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_alerts_triggered ON alerts(triggered_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

-- Canary progress
CREATE TABLE IF NOT EXISTS canary_progress (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    stage INTEGER NOT NULL,
    traffic_percentage INTEGER NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'running',
    metrics TEXT, -- JSON
    FOREIGN KEY (deployment_id) REFERENCES model_deployments(id)
);

-- Rollback history
CREATE TABLE IF NOT EXISTS deployment_rollback_history (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL,
    from_version TEXT NOT NULL,
    to_version TEXT NOT NULL,
    reason TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'in_progress',
    FOREIGN KEY (deployment_id) REFERENCES model_deployments(id)
);

-- ============================================================================
-- 10. COMMUNICATIONS & MESSAGING
-- ============================================================================

-- Dead letter queue
CREATE TABLE IF NOT EXISTS dlq_entries (
    id TEXT PRIMARY KEY,
    original_topic TEXT NOT NULL,
    payload TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    first_failure_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_failure_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'failed',
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dlq_status ON dlq_entries(status);
CREATE INDEX IF NOT EXISTS idx_dlq_topic ON dlq_entries(original_topic);

-- Message deduplication
CREATE TABLE IF NOT EXISTS message_dedupe (
    message_id TEXT PRIMARY KEY,
    message_hash TEXT NOT NULL,
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dedupe_hash ON message_dedupe(message_hash);
CREATE INDEX IF NOT EXISTS idx_dedupe_expires ON message_dedupe(expires_at);

-- ============================================================================
-- 11. GENERAL SNAPSHOTS & ROLLBACK
-- ============================================================================

-- Unified snapshots table
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    snapshot_type TEXT NOT NULL,
    payload TEXT NOT NULL, -- JSON
    hash TEXT NOT NULL,
    size_bytes INTEGER,
    compressed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    metadata TEXT -- JSON
);

CREATE INDEX IF NOT EXISTS idx_snapshots_component ON snapshots(component);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON snapshots(created_at);

-- Unified rollback history
CREATE TABLE IF NOT EXISTS rollback_history (
    id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    target_snapshot_id TEXT NOT NULL,
    from_state TEXT NOT NULL, -- JSON
    to_state TEXT NOT NULL, -- JSON
    triggered_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'in_progress',
    error_message TEXT,
    FOREIGN KEY (target_snapshot_id) REFERENCES snapshots(snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_rollback_component ON rollback_history(component);
CREATE INDEX IF NOT EXISTS idx_rollback_started ON rollback_history(started_at);

-- ============================================================================
-- 12. CORE MEMORY & GOVERNANCE
-- ============================================================================

-- Structured memory (core)
CREATE TABLE IF NOT EXISTS structured_memory (
    id TEXT PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    type TEXT NOT NULL,
    ttl_seconds INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_structured_memory_key ON structured_memory(key);
CREATE INDEX IF NOT EXISTS idx_structured_memory_expires ON structured_memory(expires_at);

-- Governance decisions
CREATE TABLE IF NOT EXISTS governance_decisions (
    id TEXT PRIMARY KEY,
    decision_id TEXT UNIQUE NOT NULL,
    context TEXT NOT NULL, -- JSON
    decision TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    approved BOOLEAN NOT NULL,
    reasons TEXT, -- JSON array
    risk_level TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_governance_decisions_decision_id ON governance_decisions(decision_id);
CREATE INDEX IF NOT EXISTS idx_governance_decisions_approved ON governance_decisions(approved);
CREATE INDEX IF NOT EXISTS idx_governance_decisions_created_at ON governance_decisions(created_at);

-- Instance states (for stateful components)
CREATE TABLE IF NOT EXISTS instance_states (
    id TEXT PRIMARY KEY,
    instance_id TEXT UNIQUE NOT NULL,
    component TEXT NOT NULL,
    state TEXT NOT NULL, -- JSON
    version TEXT NOT NULL,
    last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_instance_states_component ON instance_states(component);
CREATE INDEX IF NOT EXISTS idx_instance_states_heartbeat ON instance_states(last_heartbeat);

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Commit all changes
COMMIT;
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
    references TEXT, -- JSON: referenced data, past decisions, etc.
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
