-- Initialize Grace Governance Database
-- This script sets up the initial database schema for Grace

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create main database if not exists
-- Note: This runs inside the grace_governance database

-- Create schema for governance system
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS memory;
CREATE SCHEMA IF NOT EXISTS mldl;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA governance TO grace_user;
GRANT ALL PRIVILEGES ON SCHEMA audit TO grace_user;
GRANT ALL PRIVILEGES ON SCHEMA memory TO grace_user;  
GRANT ALL PRIVILEGES ON SCHEMA mldl TO grace_user;

-- Create initial governance tables
SET search_path = governance;

-- Constitutional framework table
CREATE TABLE IF NOT EXISTS constitutional_framework (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    principle_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    weight DECIMAL(3,2) NOT NULL DEFAULT 1.0,
    required BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert constitutional principles
INSERT INTO constitutional_framework (principle_name, description, weight, required) VALUES
('transparency', 'All decisions must be transparent and auditable', 1.0, true),
('fairness', 'Decisions must be fair and unbiased', 1.0, true),
('accountability', 'Decision makers must be accountable', 0.9, true),
('consistency', 'Similar cases should have similar outcomes', 0.8, true),
('harm_prevention', 'Decisions must not cause unnecessary harm', 1.0, true)
ON CONFLICT (principle_name) DO NOTHING;

-- Governance instances table
CREATE TABLE IF NOT EXISTS governance_instances (
    instance_id VARCHAR(255) PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    config_hash VARCHAR(64) NOT NULL,
    is_shadow BOOLEAN DEFAULT false
);

-- Audit schema
SET search_path = audit;

-- Immutable audit logs
CREATE TABLE IF NOT EXISTS immutable_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    component_id VARCHAR(255) NOT NULL,
    correlation_id VARCHAR(255),
    event_data JSONB NOT NULL,
    hash_chain VARCHAR(64),
    transparency_level INTEGER DEFAULT 0,
    retention_until TIMESTAMP WITH TIME ZONE
);

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_immutable_logs_timestamp ON immutable_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_immutable_logs_component ON immutable_logs(component_id);
CREATE INDEX IF NOT EXISTS idx_immutable_logs_correlation ON immutable_logs(correlation_id);

-- KPI and Trust monitoring
CREATE TABLE IF NOT EXISTS kpi_trust_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    instance_id VARCHAR(255) NOT NULL,
    component_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    threshold_warning DECIMAL(10,4),
    threshold_critical DECIMAL(10,4),
    correlation_id VARCHAR(255)
);

-- Memory schema
SET search_path = memory;

-- Structured memory store
CREATE TABLE IF NOT EXISTS structured_memory (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_type VARCHAR(50) NOT NULL,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    content JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    importance_score DECIMAL(3,2) DEFAULT 0.5
);

-- Create indices for memory queries
CREATE INDEX IF NOT EXISTS idx_structured_memory_type ON structured_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_structured_memory_hash ON structured_memory(content_hash);
CREATE INDEX IF NOT EXISTS idx_structured_memory_importance ON structured_memory(importance_score);

-- MLDL schema 
SET search_path = mldl;

-- Specialist registry
CREATE TABLE IF NOT EXISTS specialists (
    specialist_id VARCHAR(255) PRIMARY KEY,
    specialist_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    confidence_threshold DECIMAL(3,2) DEFAULT 0.6,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    performance_metrics JSONB
);

-- Consensus records
CREATE TABLE IF NOT EXISTS consensus_records (
    consensus_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_context VARCHAR(255) NOT NULL,
    participating_specialists TEXT[] NOT NULL,
    consensus_score DECIMAL(3,2) NOT NULL,
    final_decision JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    correlation_id VARCHAR(255)
);

-- Reset search path
SET search_path = public;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for constitutional_framework
CREATE TRIGGER trigger_constitutional_framework_updated_at
    BEFORE UPDATE ON governance.constitutional_framework
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();