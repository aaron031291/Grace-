-- Grace Memory System - PostgreSQL Schema
-- Production database schema for structured memories

-- Create extension for vector operations (if using pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Structured Memories Table
CREATE TABLE IF NOT EXISTS structured_memories (
    memory_id VARCHAR(255) PRIMARY KEY,
    content JSONB NOT NULL,
    memory_type VARCHAR(100) NOT NULL,
    embedding vector(1536),  -- OpenAI ada-002 embedding size
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_memory_type ON structured_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_created_at ON structured_memories(created_at);
CREATE INDEX IF NOT EXISTS idx_content_gin ON structured_memories USING GIN(content);
CREATE INDEX IF NOT EXISTS idx_metadata_gin ON structured_memories USING GIN(metadata);

-- Vector similarity search index (requires pgvector)
CREATE INDEX IF NOT EXISTS idx_embedding_vector ON structured_memories 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Memory Relationships Table
CREATE TABLE IF NOT EXISTS memory_relationships (
    relationship_id SERIAL PRIMARY KEY,
    source_memory_id VARCHAR(255) REFERENCES structured_memories(memory_id) ON DELETE CASCADE,
    target_memory_id VARCHAR(255) REFERENCES structured_memories(memory_id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_memory_id, target_memory_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_source_memory ON memory_relationships(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_target_memory ON memory_relationships(target_memory_id);

-- Memory Access Log (for analytics)
CREATE TABLE IF NOT EXISTS memory_access_log (
    log_id SERIAL PRIMARY KEY,
    memory_id VARCHAR(255) REFERENCES structured_memories(memory_id) ON DELETE CASCADE,
    access_type VARCHAR(50) NOT NULL,  -- 'read', 'write', 'update', 'delete'
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_access_timestamp ON memory_access_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_access_memory ON memory_access_log(memory_id);

-- Health Check Table
CREATE TABLE IF NOT EXISTS system_health (
    check_id SERIAL PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    metrics JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_health_component ON system_health(component);
CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-updating updated_at
CREATE TRIGGER update_memory_modtime
    BEFORE UPDATE ON structured_memories
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Function to log memory access
CREATE OR REPLACE FUNCTION log_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO memory_access_log (memory_id, access_type, context)
        VALUES (NEW.memory_id, 'write', NEW.metadata);
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO memory_access_log (memory_id, access_type, context)
        VALUES (NEW.memory_id, 'update', NEW.metadata);
        NEW.access_count = OLD.access_count + 1;
        NEW.last_accessed = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for access logging
CREATE TRIGGER log_memory_operations
    AFTER INSERT OR UPDATE ON structured_memories
    FOR EACH ROW
    EXECUTE FUNCTION log_memory_access();

-- View for memory statistics
CREATE OR REPLACE VIEW memory_statistics AS
SELECT 
    memory_type,
    COUNT(*) as total_count,
    AVG(access_count) as avg_access_count,
    MAX(created_at) as latest_created,
    COUNT(DISTINCT metadata->>'domain') as unique_domains
FROM structured_memories
GROUP BY memory_type;

-- View for health dashboard
CREATE OR REPLACE VIEW health_dashboard AS
SELECT 
    component,
    status,
    metrics,
    timestamp,
    ROW_NUMBER() OVER (PARTITION BY component ORDER BY timestamp DESC) as rn
FROM system_health
WHERE timestamp > NOW() - INTERVAL '1 hour';

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO grace_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO grace_user;

COMMENT ON TABLE structured_memories IS 'Core table for Grace structured memory storage';
COMMENT ON TABLE memory_relationships IS 'Relationships between memory nodes';
COMMENT ON TABLE memory_access_log IS 'Audit log for memory access patterns';
COMMENT ON TABLE system_health IS 'System health monitoring data';
