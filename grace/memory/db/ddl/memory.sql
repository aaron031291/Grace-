-- Grace Memory Kernel Database Schema  
-- Manages memory operations, fusion storage, and librarian indexing

-- Lightning Memory (Fast Cache) Entries
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

-- Fusion Memory (Persistent Storage) Entries
CREATE TABLE IF NOT EXISTS fusion_memory (
    entry_id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    content BLOB NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/json',
    size_bytes INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    compressed BOOLEAN NOT NULL DEFAULT FALSE,
    compression_type TEXT,  -- gzip, lz4, zstd, etc.
    metadata JSON,
    tags JSON,  -- Array of tags for categorization
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    accessed_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Memory Access Patterns (for librarian optimization)
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    access_type TEXT NOT NULL,  -- read, write, search
    access_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    response_time_ms REAL NOT NULL,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
    user_context TEXT,
    query_context JSON
);

-- Librarian Search Index
CREATE TABLE IF NOT EXISTS librarian_index (
    id TEXT PRIMARY KEY,
    entry_key TEXT NOT NULL,
    index_type TEXT NOT NULL,  -- content, semantic, tag, metadata
    index_value TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Memory Usage Statistics
CREATE TABLE IF NOT EXISTS memory_stats (
    id TEXT PRIMARY KEY,
    stat_type TEXT NOT NULL,  -- storage_size, cache_hit_rate, avg_response_time
    component TEXT NOT NULL,  -- lightning, fusion, librarian
    value REAL NOT NULL,
    unit TEXT NOT NULL,  -- bytes, percentage, milliseconds
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Memory Operations Log (for debugging and analytics)
CREATE TABLE IF NOT EXISTS memory_operations (
    id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,  -- write, read, search, delete, expire
    key TEXT NOT NULL,
    component TEXT NOT NULL,  -- lightning, fusion, librarian
    success BOOLEAN NOT NULL DEFAULT TRUE,
    duration_ms REAL NOT NULL,
    size_bytes INTEGER,
    error_message TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Memory Snapshots
CREATE TABLE IF NOT EXISTS memory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    lightning_state JSON NOT NULL,  -- Current lightning memory state
    fusion_index JSON NOT NULL,     -- Fusion memory entry index
    librarian_config JSON NOT NULL, -- Librarian configuration
    access_patterns JSON NOT NULL,  -- Recent access pattern summary
    statistics JSON NOT NULL,       -- Current memory statistics
    snapshot_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    size_bytes INTEGER
);

-- Memory Cleanup Tasks
CREATE TABLE IF NOT EXISTS memory_cleanup_tasks (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,  -- expire_cache, compress_old, index_rebuild
    component TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    scheduled_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    items_processed INTEGER DEFAULT 0,
    items_total INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSON
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_lightning_expires ON lightning_memory(expires_at);
CREATE INDEX IF NOT EXISTS idx_lightning_accessed ON lightning_memory(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_lightning_ttl ON lightning_memory(ttl_seconds);

CREATE INDEX IF NOT EXISTS idx_fusion_key ON fusion_memory(key);
CREATE INDEX IF NOT EXISTS idx_fusion_created ON fusion_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_fusion_accessed ON fusion_memory(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_fusion_size ON fusion_memory(size_bytes);
CREATE INDEX IF NOT EXISTS idx_fusion_tags ON fusion_memory(tags);

CREATE INDEX IF NOT EXISTS idx_access_patterns_key ON memory_access_patterns(key);
CREATE INDEX IF NOT EXISTS idx_access_patterns_timestamp ON memory_access_patterns(access_timestamp);
CREATE INDEX IF NOT EXISTS idx_access_patterns_type ON memory_access_patterns(access_type);

CREATE INDEX IF NOT EXISTS idx_librarian_key ON librarian_index(entry_key);
CREATE INDEX IF NOT EXISTS idx_librarian_type ON librarian_index(index_type);
CREATE INDEX IF NOT EXISTS idx_librarian_value ON librarian_index(index_value);
CREATE INDEX IF NOT EXISTS idx_librarian_weight ON librarian_index(weight);

CREATE INDEX IF NOT EXISTS idx_memory_stats_component ON memory_stats(component);
CREATE INDEX IF NOT EXISTS idx_memory_stats_recorded ON memory_stats(recorded_at);
CREATE INDEX IF NOT EXISTS idx_memory_stats_type ON memory_stats(stat_type);

CREATE INDEX IF NOT EXISTS idx_memory_ops_timestamp ON memory_operations(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_ops_component ON memory_operations(component);
CREATE INDEX IF NOT EXISTS idx_memory_ops_key ON memory_operations(key);

CREATE INDEX IF NOT EXISTS idx_cleanup_tasks_status ON memory_cleanup_tasks(status);
CREATE INDEX IF NOT EXISTS idx_cleanup_tasks_scheduled ON memory_cleanup_tasks(scheduled_at);

-- Triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_lightning_accessed
    AFTER UPDATE ON lightning_memory
    BEGIN
        UPDATE lightning_memory SET last_accessed_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
    END;

CREATE TRIGGER IF NOT EXISTS update_fusion_timestamps
    AFTER UPDATE ON fusion_memory
    BEGIN
        UPDATE fusion_memory 
        SET updated_at = CURRENT_TIMESTAMP,
            last_accessed_at = CURRENT_TIMESTAMP
        WHERE entry_id = NEW.entry_id;
    END;

CREATE TRIGGER IF NOT EXISTS update_librarian_timestamp
    AFTER UPDATE ON librarian_index
    BEGIN
        UPDATE librarian_index SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

-- Views for common queries
CREATE VIEW IF NOT EXISTS memory_overview AS
SELECT 
    'lightning' as component,
    COUNT(*) as total_entries,
    SUM(LENGTH(value)) as total_bytes,
    AVG(access_count) as avg_access_count,
    COUNT(*) FILTER (WHERE expires_at > CURRENT_TIMESTAMP OR expires_at IS NULL) as active_entries
FROM lightning_memory
UNION ALL
SELECT 
    'fusion' as component,
    COUNT(*) as total_entries,
    SUM(size_bytes) as total_bytes,
    AVG(accessed_count) as avg_access_count,
    COUNT(*) as active_entries
FROM fusion_memory;

CREATE VIEW IF NOT EXISTS recent_memory_activity AS
SELECT 
    operation_type,
    component,
    COUNT(*) as operation_count,
    AVG(duration_ms) as avg_duration_ms,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_operations
FROM memory_operations 
WHERE timestamp >= datetime('now', '-1 hour')
GROUP BY operation_type, component;

-- Cleanup expired lightning memory entries (scheduled task support)
CREATE TRIGGER IF NOT EXISTS cleanup_expired_lightning
    AFTER INSERT ON memory_cleanup_tasks
    WHEN NEW.task_type = 'expire_cache' AND NEW.component = 'lightning'
    BEGIN
        DELETE FROM lightning_memory 
        WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP;
        
        UPDATE memory_cleanup_tasks 
        SET status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            items_processed = changes()
        WHERE id = NEW.id;
    END;