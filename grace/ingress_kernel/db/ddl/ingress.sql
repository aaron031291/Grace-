-- Ingress Kernel Database Schema
-- DDL for Bronze/Silver/Gold data tiers and operational tables

-- Create database and basic setup
CREATE DATABASE IF NOT EXISTS grace_ingress;
USE grace_ingress;

-- Enable foreign key constraints
SET FOREIGN_KEY_CHECKS = 1;

-- =============================================================================
-- OPERATIONAL TABLES (System Management)
-- =============================================================================

-- Source registry table
CREATE TABLE sources (
    source_id VARCHAR(50) PRIMARY KEY,
    kind ENUM('http', 'rss', 's3', 'gcs', 'azure_blob', 'github', 'youtube', 'podcast', 'social', 'kafka', 'mqtt', 'sql', 'csv_local') NOT NULL,
    uri TEXT NOT NULL,
    auth_mode ENUM('none', 'bearer', 'basic', 'api_key', 'oauth', 'signed_url', 'aws_iam', 'gcp_sa') NOT NULL DEFAULT 'none',
    secrets_ref VARCHAR(100),
    schedule VARCHAR(100) NOT NULL DEFAULT 'manual',
    parser ENUM('json', 'csv', 'html', 'pdf', 'audio', 'video', 'xml') NOT NULL,
    parser_opts JSON,
    target_contract VARCHAR(100) NOT NULL,
    retention_days INTEGER NOT NULL DEFAULT 365,
    pii_policy ENUM('block', 'mask', 'hash', 'allow_with_consent') NOT NULL DEFAULT 'mask',
    governance_label ENUM('public', 'internal', 'restricted') NOT NULL DEFAULT 'internal',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_kind (kind),
    INDEX idx_enabled (enabled),
    INDEX idx_governance_label (governance_label),
    INDEX idx_created_at (created_at)
);

-- Source health status
CREATE TABLE source_health (
    source_id VARCHAR(50) NOT NULL,
    status ENUM('ok', 'degraded', 'down', 'unknown') NOT NULL DEFAULT 'unknown',
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latency_ms INTEGER DEFAULT 0,
    backlog INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_ok TIMESTAMP,
    
    PRIMARY KEY (source_id),
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    INDEX idx_status (status),
    INDEX idx_last_check (last_check)
);

-- System snapshots for rollback
CREATE TABLE snapshots (
    snapshot_id VARCHAR(100) PRIMARY KEY,
    active_sources JSON NOT NULL,
    registry_hash VARCHAR(64) NOT NULL,
    parser_versions JSON NOT NULL,
    dedupe_threshold DECIMAL(3,2) NOT NULL DEFAULT 0.87,
    pii_policy_defaults VARCHAR(50) NOT NULL DEFAULT 'mask',
    offsets JSON NOT NULL,
    watermarks JSON NOT NULL,
    gold_views_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_created_at (created_at),
    INDEX idx_hash (hash)
);

-- =============================================================================
-- BRONZE TIER (Raw Events)
-- =============================================================================

-- Raw events table (append-only)
CREATE TABLE bronze_raw_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_id VARCHAR(50) UNIQUE NOT NULL,
    source_id VARCHAR(50) NOT NULL,
    kind ENUM('json', 'csv', 'html', 'pdf', 'audio', 'video', 'xml', 'bin') NOT NULL,
    offset VARCHAR(200) NOT NULL,
    watermark TIMESTAMP NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    headers JSON,
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_uri TEXT NOT NULL, -- Points to actual content storage (S3, local file, etc.)
    size_bytes INTEGER NOT NULL DEFAULT 0,
    
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    INDEX idx_source_id (source_id),
    INDEX idx_content_hash (content_hash),
    INDEX idx_ingestion_ts (ingestion_ts),
    INDEX idx_watermark (watermark),
    INDEX idx_kind (kind)
);

-- =============================================================================
-- SILVER TIER (Normalized Records)
-- =============================================================================

-- Main normalized records table
CREATE TABLE silver_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id VARCHAR(50) UNIQUE NOT NULL,
    contract VARCHAR(100) NOT NULL,
    source_id VARCHAR(50) NOT NULL,
    raw_event_id VARCHAR(50) NOT NULL,
    
    -- Source information (denormalized for query performance)
    source_uri TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    parser VARCHAR(20) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    
    -- Quality metrics (denormalized)
    validity_score DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    completeness DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    freshness_minutes DECIMAL(10,2) NOT NULL DEFAULT 0.0,
    trust_score DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    pii_flags JSON, -- Array of detected PII types
    
    -- Lineage information
    transforms JSON NOT NULL, -- Array of transformation steps
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_uri TEXT NOT NULL, -- Points to actual record body storage
    
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    FOREIGN KEY (raw_event_id) REFERENCES bronze_raw_events(event_id) ON DELETE CASCADE,
    
    INDEX idx_contract (contract),
    INDEX idx_source_id (source_id),
    INDEX idx_trust_score (trust_score),
    INDEX idx_validity_score (validity_score),
    INDEX idx_created_at (created_at),
    INDEX idx_freshness (freshness_minutes)
);

-- Contract-specific tables for structured queries

-- Articles (contract:article.v1)
CREATE TABLE silver_articles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(200),
    published_at TIMESTAMP NULL,
    url TEXT NOT NULL,
    language CHAR(2) NOT NULL DEFAULT 'en',
    text_uri TEXT NOT NULL, -- Points to full text storage
    topics JSON, -- Array of topic strings
    entities JSON, -- Object with persons, orgs, locations arrays
    embeddings_ref VARCHAR(100), -- Reference to embeddings storage
    
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE,
    
    INDEX idx_title (title(100)),
    INDEX idx_author (author),
    INDEX idx_published_at (published_at),
    INDEX idx_language (language),
    FULLTEXT idx_title_fulltext (title)
);

-- Transcripts (contract:transcript.v1)
CREATE TABLE silver_transcripts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id VARCHAR(50) UNIQUE NOT NULL,
    media_id VARCHAR(100) NOT NULL,
    start_at TIMESTAMP NULL,
    duration_s DECIMAL(10,3),
    lang CHAR(2) NOT NULL DEFAULT 'en',
    segments_uri TEXT NOT NULL, -- Points to segments JSON storage
    summary TEXT,
    
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE,
    
    INDEX idx_media_id (media_id),
    INDEX idx_start_at (start_at),
    INDEX idx_duration (duration_s),
    INDEX idx_lang (lang)
);

-- Tabular data (contract:tabular.v1)
CREATE TABLE silver_tabular (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id VARCHAR(50) UNIQUE NOT NULL,
    dataset_id VARCHAR(100) NOT NULL,
    columns JSON NOT NULL, -- Array of column definitions
    rows_uri TEXT NOT NULL, -- Points to Parquet/CSV storage
    row_count INTEGER NOT NULL DEFAULT 0,
    
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE,
    
    INDEX idx_dataset_id (dataset_id),
    INDEX idx_row_count (row_count)
);

-- =============================================================================
-- GOLD TIER (Curated Features)
-- =============================================================================

-- Article topics (curated)
CREATE TABLE gold_article_topics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_record_id VARCHAR(50) NOT NULL,
    topic VARCHAR(100) NOT NULL,
    weight DECIMAL(4,3) NOT NULL DEFAULT 0.0,
    extraction_method VARCHAR(50) NOT NULL DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (article_record_id) REFERENCES silver_articles(record_id) ON DELETE CASCADE,
    
    UNIQUE KEY unique_article_topic (article_record_id, topic),
    INDEX idx_topic (topic),
    INDEX idx_weight (weight)
);

-- Entity mentions across all content
CREATE TABLE gold_entity_mentions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id VARCHAR(50) NOT NULL,
    entity VARCHAR(200) NOT NULL,
    entity_type ENUM('person', 'organization', 'location', 'other') NOT NULL,
    confidence DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    context TEXT, -- Surrounding text context
    position_start INTEGER,
    position_end INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (record_id) REFERENCES silver_records(record_id) ON DELETE CASCADE,
    
    INDEX idx_entity (entity),
    INDEX idx_entity_type (entity_type),
    INDEX idx_confidence (confidence),
    INDEX idx_record_entity (record_id, entity)
);

-- Feature store metadata
CREATE TABLE gold_feature_datasets (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(100) UNIQUE NOT NULL,
    dataset_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    feature_count INTEGER NOT NULL DEFAULT 0,
    source_contracts JSON NOT NULL, -- Array of contracts contributing to this dataset
    storage_uri TEXT NOT NULL,
    train_split_uri TEXT,
    validation_split_uri TEXT,
    test_split_uri TEXT,
    metadata JSON, -- Schema, statistics, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_dataset_name (dataset_name),
    INDEX idx_dataset_version (dataset_version),
    INDEX idx_created_at (created_at)
);

-- =============================================================================
-- OPERATIONAL METRICS AND AUDITING
-- =============================================================================

-- Processing pipeline metrics
CREATE TABLE processing_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    stage ENUM('capture', 'parse', 'normalize', 'validate', 'enrich', 'persist', 'publish') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance metrics
    throughput_rps DECIMAL(8,2) DEFAULT 0.0,
    error_rate DECIMAL(5,4) DEFAULT 0.0,
    avg_latency_ms DECIMAL(8,2) DEFAULT 0.0,
    
    -- Quality metrics
    schema_violations INTEGER DEFAULT 0,
    pii_incidents INTEGER DEFAULT 0,
    dedup_rate DECIMAL(5,4) DEFAULT 0.0,
    trust_mean DECIMAL(3,2) DEFAULT 0.0,
    completeness_mean DECIMAL(3,2) DEFAULT 0.0,
    
    -- Stage-specific metrics
    parser_accuracy DECIMAL(3,2),
    ner_precision DECIMAL(3,2),
    ner_recall DECIMAL(3,2),
    
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    
    INDEX idx_source_stage (source_id, stage),
    INDEX idx_timestamp (timestamp),
    INDEX idx_error_rate (error_rate)
);

-- Validation failures log
CREATE TABLE validation_failures (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id VARCHAR(50),
    raw_event_id VARCHAR(50),
    source_id VARCHAR(50) NOT NULL,
    policy_type ENUM('pii', 'schema', 'format', 'governance') NOT NULL,
    severity ENUM('warn', 'error') NOT NULL DEFAULT 'error',
    reasons JSON NOT NULL, -- Array of failure reasons
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    
    INDEX idx_source_id (source_id),
    INDEX idx_policy_type (policy_type),
    INDEX idx_severity (severity),
    INDEX idx_timestamp (timestamp),
    INDEX idx_resolved (resolved)
);

-- Trust score history for sources
CREATE TABLE source_trust_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_id VARCHAR(50) NOT NULL,
    old_reputation DECIMAL(3,2) NOT NULL,
    new_reputation DECIMAL(3,2) NOT NULL,
    outcome_score DECIMAL(3,2) NOT NULL,
    context JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE,
    
    INDEX idx_source_timestamp (source_id, timestamp),
    INDEX idx_reputation (new_reputation)
);

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Recent high-quality articles
CREATE VIEW recent_quality_articles AS
SELECT 
    a.record_id,
    a.title,
    a.author,
    a.published_at,
    a.url,
    a.language,
    sr.trust_score,
    sr.validity_score,
    sr.created_at
FROM silver_articles a
JOIN silver_records sr ON a.record_id = sr.record_id
WHERE sr.trust_score >= 0.7 
    AND sr.validity_score >= 0.8
    AND sr.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY sr.created_at DESC;

-- Source performance summary
CREATE VIEW source_performance_summary AS
SELECT 
    s.source_id,
    s.kind,
    s.enabled,
    sh.status,
    sh.last_check,
    sh.latency_ms,
    sh.error_count,
    COUNT(DISTINCT bre.event_id) AS total_events_24h,
    COUNT(DISTINCT sr.record_id) AS successful_records_24h,
    AVG(sr.trust_score) AS avg_trust_score,
    AVG(sr.validity_score) AS avg_validity_score
FROM sources s
LEFT JOIN source_health sh ON s.source_id = sh.source_id
LEFT JOIN bronze_raw_events bre ON s.source_id = bre.source_id 
    AND bre.ingestion_ts >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
LEFT JOIN silver_records sr ON s.source_id = sr.source_id 
    AND sr.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY s.source_id, s.kind, s.enabled, sh.status, sh.last_check, sh.latency_ms, sh.error_count;

-- Top entities across all content
CREATE VIEW top_entities AS
SELECT 
    entity,
    entity_type,
    COUNT(*) as mention_count,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT record_id) as unique_records
FROM gold_entity_mentions
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY entity, entity_type
HAVING mention_count >= 5
ORDER BY mention_count DESC, avg_confidence DESC;