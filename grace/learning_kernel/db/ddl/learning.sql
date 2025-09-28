-- Learning Kernel Database Schema
-- Data-centric learning: datasets, labeling, active learning, weak supervision

-- Dataset registry and versioning
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    task TEXT NOT NULL CHECK (task IN ('classification', 'regression', 'clustering', 'dimred', 'rl', 'nlp', 'vision', 'timeseries')),
    modality TEXT NOT NULL CHECK (modality IN ('tabular', 'text', 'image', 'audio', 'video', 'graph')),
    schema_json TEXT,
    default_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    source_refs_json TEXT NOT NULL, -- JSON array of source references
    row_count INTEGER,
    byte_size INTEGER,
    stats_json TEXT, -- JSON stats object
    train_split REAL DEFAULT 0.8,
    valid_split REAL DEFAULT 0.1,
    test_split REAL DEFAULT 0.1,
    feature_view TEXT,
    lineage_hash TEXT,
    governance_label TEXT CHECK (governance_label IN ('public', 'internal', 'restricted')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    UNIQUE(dataset_id, version)
);

-- Labeling tasks and policies
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

CREATE TABLE IF NOT EXISTS label_tasks (
    task_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    assign_strategy TEXT DEFAULT 'auto' CHECK (assign_strategy IN ('auto', 'round_robin', 'skill_based')),
    items_json TEXT NOT NULL, -- JSON array of item IDs/URIs
    priority TEXT DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'critical')),
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'paused', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (policy_id) REFERENCES label_policies(policy_id)
);

CREATE TABLE IF NOT EXISTS labels (
    label_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    y_value TEXT NOT NULL, -- JSON serialized label value
    evidence_json TEXT, -- JSON evidence object
    weak_sources_json TEXT, -- JSON array of weak source IDs
    agreement REAL,
    gold_correct BOOLEAN,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES label_tasks(task_id)
);

-- Active learning and query strategies
CREATE TABLE IF NOT EXISTS active_queries (
    query_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    strategy TEXT NOT NULL CHECK (strategy IN ('uncertainty', 'margin', 'entropy', 'diversity', 'coresets', 'hybrid')),
    batch_size INTEGER NOT NULL,
    segment_filters_json TEXT,
    min_confidence REAL,
    results_json TEXT, -- JSON array of selected item IDs
    query_gain_f1 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Curriculum learning specifications  
CREATE TABLE IF NOT EXISTS curriculum_specs (
    spec_id TEXT PRIMARY KEY,
    objective TEXT NOT NULL CHECK (objective IN ('improve_tail', 'reduce_bias', 'boost_calibration', 'reduce_regret')),
    rules_json TEXT NOT NULL, -- JSON array of curriculum rules
    dataset_id TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Data augmentation specifications
CREATE TABLE IF NOT EXISTS augment_specs (
    spec_id TEXT PRIMARY KEY,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image', 'audio', 'tabular')),
    ops_json TEXT NOT NULL, -- JSON array of augmentation operations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS augment_applications (
    application_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    spec_id TEXT NOT NULL,
    delta_rows INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (spec_id) REFERENCES augment_specs(spec_id)
);

-- Feature store views for train/serve parity
CREATE TABLE IF NOT EXISTS feature_views (
    view_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT NOT NULL,
    view_uri TEXT NOT NULL,
    format TEXT DEFAULT 'parquet',
    build_status TEXT DEFAULT 'pending' CHECK (build_status IN ('pending', 'building', 'ready', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    built_at TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    UNIQUE(dataset_id, version)
);

-- Data quality and evaluation metrics
CREATE TABLE IF NOT EXISTS quality_reports (
    report_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version TEXT,
    leakage_flags INTEGER DEFAULT 0,
    bias_metrics_json TEXT, -- JSON object with bias/fairness metrics
    coverage_ratio REAL,
    label_agreement REAL,
    drift_psi REAL,
    noise_estimate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Learning experiences for MLT integration
CREATE TABLE IF NOT EXISTS learning_experiences (
    exp_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL CHECK (stage IN ('labeling', 'active_query', 'weak_supervision', 'augmentation', 'curriculum', 'version_publish', 'eval')),
    metrics_json TEXT NOT NULL, -- JSON metrics object
    segment TEXT,
    dataset_id TEXT,
    version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

-- Snapshots for rollback capability
CREATE TABLE IF NOT EXISTS learning_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    datasets_json TEXT NOT NULL, -- JSON array of dataset IDs
    versions_json TEXT NOT NULL, -- JSON object mapping dataset->version
    feature_views_json TEXT, -- JSON object mapping dataset@version->uri
    policies_version TEXT,
    active_query_config_json TEXT,
    weak_labelers_json TEXT,
    augmentation_config_json TEXT,
    hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weak supervision and rules
CREATE TABLE IF NOT EXISTS weak_labelers (
    labeler_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    labeler_type TEXT DEFAULT 'rule' CHECK (labeler_type IN ('rule', 'model', 'heuristic')),
    threshold REAL DEFAULT 0.65,
    rules_json TEXT, -- JSON rules/config
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
    prediction TEXT NOT NULL, -- JSON serialized prediction
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (labeler_id) REFERENCES weak_labelers(labeler_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset ON dataset_versions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_labels_task ON labels(task_id);
CREATE INDEX IF NOT EXISTS idx_labels_item ON labels(item_id);
CREATE INDEX IF NOT EXISTS idx_active_queries_dataset ON active_queries(dataset_id);
CREATE INDEX IF NOT EXISTS idx_learning_experiences_dataset ON learning_experiences(dataset_id);
CREATE INDEX IF NOT EXISTS idx_learning_experiences_stage ON learning_experiences(stage);
CREATE INDEX IF NOT EXISTS idx_weak_predictions_labeler ON weak_predictions(labeler_id);
CREATE INDEX IF NOT EXISTS idx_weak_predictions_item ON weak_predictions(item_id);