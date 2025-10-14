# Grace System - Database Schema Documentation

## Overview

This document describes the complete database schema for all Grace system components and kernels. The schema has been designed to support multiple database backends (SQLite, PostgreSQL, MySQL) and follows best practices for data governance, auditability, and operational resilience.

## Quick Start

### Build All Tables

```bash
# SQLite (default - recommended for development)
python build_all_tables.py --db sqlite --path grace_system.db

# PostgreSQL (recommended for production)
python build_all_tables.py --db postgres --host localhost --dbname grace --user grace --password <password>

# MySQL
python build_all_tables.py --db mysql --host localhost --dbname grace --user grace --password <password>

# List all components and tables
python build_all_tables.py --list
```

### Files

- **`init_all_tables.sql`** - Complete SQL DDL for all 81 tables
- **`build_all_tables.py`** - Python script to build tables across different database backends
- **`grace_system.db`** - Pre-built SQLite database with all tables initialized

## Architecture

The Grace system uses a distributed, kernel-based architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Grace System Database                     │
├─────────────────────────────────────────────────────────────┤
│  Core (6 tables)                                             │
│  ├─ Audit Logs (immutable, blockchain-style chain)         │
│  ├─ Governance Decisions                                    │
│  └─ Instance States                                         │
├─────────────────────────────────────────────────────────────┤
│  Ingress Kernel (13 tables)                                 │
│  ├─ Bronze Tier: Raw events                                │
│  ├─ Silver Tier: Normalized records                        │
│  └─ Gold Tier: Curated features                            │
├─────────────────────────────────────────────────────────────┤
│  Intelligence Kernel (9 tables)                             │
│  ├─ Requests, Plans, Results                               │
│  ├─ Canary & Shadow Deployments                            │
│  └─ Policy Violations                                       │
├─────────────────────────────────────────────────────────────┤
│  Learning Kernel (15 tables)                                │
│  ├─ Datasets & Versions                                     │
│  ├─ Labels & Active Learning                               │
│  └─ Weak Supervision                                        │
├─────────────────────────────────────────────────────────────┤
│  Memory Kernel (8 tables)                                   │
│  ├─ Lightning (cache)                                       │
│  ├─ Fusion (persistent)                                     │
│  └─ Librarian (indexing)                                    │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Kernel (5 tables)                            │
│  Resilience Kernel (8 tables)                               │
│  MLT Core (5 tables)                                        │
│  MLDL Components (8 tables)                                 │
│  Communications (2 tables)                                  │
│  Common (2 tables)                                          │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Core (6 tables)

**Purpose**: Foundational governance, audit, and state management

| Table | Description | Key Features |
|-------|-------------|--------------|
| `audit_logs` | Immutable audit log with blockchain-style chaining | Tamper-evident, cryptographic hashing |
| `chain_verification` | Verification records for audit chain integrity | Periodic chain validation |
| `log_categories` | Metadata for log categorization | Category statistics |
| `structured_memory` | Core memory for frequently accessed data | TTL support, access counting |
| `governance_decisions` | Records of all governance decisions | Approval tracking, risk levels |
| `instance_states` | State tracking for stateful components | Heartbeat monitoring |

**Key Patterns**:
- Audit logs use SHA-256 chaining: each entry includes hash of previous entry
- All timestamps use ISO 8601 format
- Governance decisions are immutable after creation

### 2. Ingress Kernel (13 tables)

**Purpose**: Data ingestion pipeline with Bronze/Silver/Gold architecture

#### Bronze Tier (Raw Data)
- `bronze_raw_events` - Append-only raw events from all sources
- `sources` - Source registry with auth, scheduling, governance labels
- `source_health` - Real-time health monitoring

#### Silver Tier (Normalized)
- `silver_records` - Normalized, validated records
- `silver_articles` - Contract-specific: articles
- `silver_transcripts` - Contract-specific: audio/video transcripts
- `silver_tabular` - Contract-specific: structured data

#### Gold Tier (Curated)
- `gold_article_topics` - Extracted topics from articles
- `gold_entity_mentions` - Named entities across all content
- `gold_feature_datasets` - ML-ready feature datasets

#### Operational
- `processing_metrics` - Pipeline performance metrics
- `validation_failures` - Schema/PII/governance violations
- `source_trust_history` - Source reputation tracking

**Key Patterns**:
- Content stored externally (S3, local files), tables contain URIs
- Trust scoring for all sources (0.0-1.0)
- PII detection and policy enforcement at ingestion
- Lineage tracking through transforms JSON

### 3. Intelligence Kernel (9 tables)

**Purpose**: ML inference routing, planning, and execution

| Table | Description |
|-------|-------------|
| `intel_requests` | Incoming inference requests with constraints |
| `intel_plans` | Execution plans (route, ensemble, governance) |
| `intel_results` | Inference results with explanations |
| `intel_specialist_reports` | Performance reports from specialists |
| `intel_experiences` | Meta-learning data |
| `intel_canary_deployments` | Progressive canary rollouts |
| `intel_shadow_deployments` | Shadow testing for model validation |
| `intel_policy_violations` | Confidence/calibration/fairness violations |
| `intel_snapshots` | Kernel state snapshots for rollback |

**Key Patterns**:
- Pre-flight governance approval required for high-risk requests
- Canary deployments use [5%, 25%, 50%, 100%] traffic stages
- Shadow deployments track agreement between primary/shadow models
- All results include uncertainties (calibration, variance, entropy)

### 4. Learning Kernel (15 tables)

**Purpose**: Data-centric ML with active learning and weak supervision

#### Dataset Management
- `datasets` - Dataset registry
- `dataset_versions` - Immutable versioned datasets
- `feature_views` - Train/serve parity views

#### Labeling
- `label_policies` - Labeling rubrics and quality controls
- `label_tasks` - Labeling assignments
- `labels` - Human labels with agreement tracking

#### Active Learning
- `active_queries` - Query strategies (uncertainty, diversity, etc.)
- `curriculum_specs` - Curriculum learning rules
- `augment_specs` - Data augmentation pipelines
- `augment_applications` - Augmentation executions

#### Quality & Weak Supervision
- `quality_reports` - Leakage, bias, drift detection
- `weak_labelers` - Rule-based/model-based weak labelers
- `weak_predictions` - Weak labels for semi-supervised learning
- `learning_experiences` - MLT integration data
- `learning_snapshots` - Kernel state snapshots

**Key Patterns**:
- Datasets are immutable after version creation
- Governance labels: public, internal, restricted
- Weak supervision combines multiple noisy labelers
- Active learning prioritizes high-value samples

### 5. Memory Kernel (8 tables)

**Purpose**: Multi-tier memory system (cache + persistent storage)

| Table | Description | Tier |
|-------|-------------|------|
| `lightning_memory` | Fast in-memory cache with TTL | Cache |
| `fusion_memory` | Persistent storage with compression | Storage |
| `librarian_index` | Search index for content/semantic queries | Index |
| `memory_access_patterns` | Access pattern tracking for optimization | Analytics |
| `memory_stats` | Component performance statistics | Analytics |
| `memory_operations` | Audit log for all memory operations | Audit |
| `memory_snapshots` | Full memory state snapshots | Backup |
| `memory_cleanup_tasks` | Scheduled cleanup operations | Maintenance |

**Key Patterns**:
- Lightning memory: sub-millisecond access, automatic expiration
- Fusion memory: supports compression (gzip, lz4, zstd)
- Librarian: content + semantic + tag indexing
- Access patterns inform caching decisions

### 6. Orchestration Kernel (5 tables)

**Purpose**: State machine, loop scheduling, task management

| Table | Description |
|-------|-------------|
| `orchestration_state` | Current system state |
| `orchestration_loops` | Loop definitions (7-phase loops) |
| `orchestration_tasks` | Task execution tracking |
| `policies` | System-wide policies (security, governance, performance) |
| `state_transitions` | State machine transition history |

**Key Patterns**:
- Loops run on configurable schedules (cron-like)
- Tasks have timeout and rollback support
- Policies enforce system-wide constraints

### 7. Resilience Kernel (8 tables)

**Purpose**: Circuit breakers, graceful degradation, incident management

| Table | Description |
|-------|-------------|
| `circuit_breakers` | Circuit breaker states per component |
| `degradation_policies` | Graceful degradation rules |
| `active_degradations` | Currently active degradations |
| `resilience_incidents` | Incident tracking |
| `rate_limits` | Rate limiting windows |
| `sli_measurements` | Service Level Indicators |
| `recovery_actions` | Automated recovery attempts |
| `resilience_snapshots` | Resilience state snapshots |

**Key Patterns**:
- Circuit breakers: closed → open → half_open states
- Degradation levels: mild, moderate, severe
- SLI targets for latency, availability, error rates

### 8. MLT Core (5 tables)

**Purpose**: Memory-Learning-Trust meta-learning system

| Table | Description |
|-------|-------------|
| `mlt_experiences` | Experience signals from all sources |
| `mlt_insights` | Detected patterns (drift, bias, calibration) |
| `mlt_plans` | Improvement plans (retrain, reweight, recalibrate) |
| `mlt_snapshots` | MLT state snapshots |
| `mlt_specialist_reports` | Specialist evaluation reports |

**Key Patterns**:
- Experiences flow from training, inference, governance, ops
- Insights trigger automated improvement plans
- Plans require governance approval before execution

### 9. MLDL Components (8 tables)

**Purpose**: ML model registry, deployment, monitoring

| Table | Description |
|-------|-------------|
| `models` | Model registry with versioning |
| `model_approvals` | Governance approval records |
| `model_deployments` | Active deployments |
| `model_metrics` | Runtime performance metrics |
| `slo_violations` | SLO breach tracking |
| `alerts` | Alert management |
| `canary_progress` | Canary deployment stages |
| `deployment_rollback_history` | Rollback tracking |

**Key Patterns**:
- Model stages: development → staging → production
- Governance approval required for production
- Canary deployments with automatic rollback on SLO violations

### 10. Communications (2 tables)

**Purpose**: Messaging, deduplication, dead letter queue

| Table | Description |
|-------|-------------|
| `dlq_entries` | Failed messages with retry logic |
| `message_dedupe` | Message deduplication cache |

### 11. Common (2 tables)

**Purpose**: Unified snapshot and rollback across all components

| Table | Description |
|-------|-------------|
| `snapshots` | Unified snapshots for all components |
| `rollback_history` | Rollback operation tracking |

## Data Governance

### PII Protection
- All ingress data scanned for PII
- Policies: block, mask, hash, allow_with_consent
- PII flags stored in JSON arrays

### Audit Trail
- All operations logged to `audit_logs`
- Blockchain-style chaining prevents tampering
- Regular chain verification

### Governance Labels
- **Public**: No restrictions
- **Internal**: Organization-only
- **Restricted**: Need-to-know access

### Immutability
- Audit logs are append-only
- Dataset versions are immutable after creation
- Governance decisions cannot be modified

## Performance Considerations

### Indexing Strategy
- Primary keys on all tables
- Foreign keys enforce referential integrity
- Composite indexes on common query patterns
- Full-text indexes on searchable text fields

### Partitioning (PostgreSQL)
- Partition large tables by time (audit_logs, metrics)
- Partition by hash for high-volume tables (events, requests)

### Compression
- BLOB fields support compression (gzip, lz4, zstd)
- JSON fields are stored as TEXT (SQLite) or JSONB (PostgreSQL)

### Retention Policies
- Bronze tier: configurable per source (default 365 days)
- Silver tier: indefinite (compress after 90 days)
- Gold tier: indefinite
- Metrics: aggregate hourly after 7 days, daily after 30 days

## Migration Strategy

### Schema Versioning
- All DDL changes tracked in `migrations/` directory
- Use Alembic (Python) or Flyway (Java) for migrations
- Never drop columns - mark as deprecated instead

### Backward Compatibility
- New optional columns default to NULL
- New tables don't affect existing code
- Use views to provide backward-compatible interfaces

## Monitoring & Observability

### Key Metrics to Track
1. **Ingress**: throughput, error_rate, trust_score, latency
2. **Intelligence**: p95_latency, confidence, calibration, fairness_delta
3. **Learning**: label_agreement, drift_psi, coverage_ratio
4. **Memory**: cache_hit_rate, avg_response_time, storage_size
5. **Orchestration**: loop_execution_time, task_failure_rate
6. **Resilience**: circuit_breaker_trips, degradation_activations, sli_violations

### Health Checks
```sql
-- Check for stuck tasks
SELECT * FROM orchestration_tasks 
WHERE status='running' AND started_at < datetime('now', '-1 hour');

-- Check for high error rates
SELECT source_id, AVG(error_rate) as avg_error_rate
FROM processing_metrics
WHERE timestamp >= datetime('now', '-1 hour')
GROUP BY source_id
HAVING avg_error_rate > 0.05;

-- Check circuit breaker status
SELECT component_name, state, failure_count
FROM circuit_breakers
WHERE state != 'closed';
```

## Security

### Authentication & Authorization
- All database operations require authentication
- Use role-based access control (RBAC)
- Separate read-only and read-write roles

### Encryption
- **At rest**: Encrypt database files (SQLite), use TDE (PostgreSQL/MySQL)
- **In transit**: Always use TLS/SSL for database connections
- **Application level**: Encrypt sensitive fields before storage

### Secrets Management
- Never store secrets in database (use references: `secrets_ref`)
- Use external secret managers (AWS Secrets Manager, HashiCorp Vault)

## Backup & Recovery

### Backup Strategy
- **SQLite**: Use `.backup` command or file-level backups
- **PostgreSQL/MySQL**: Use `pg_dump`/`mysqldump` or continuous archiving

### Frequency
- **Full backups**: Daily
- **Incremental backups**: Hourly (PostgreSQL WAL archiving)
- **Snapshots**: On-demand via snapshot tables

### Retention
- **Daily backups**: 30 days
- **Weekly backups**: 1 year
- **Snapshots**: Indefinite (user-managed)

### Testing
- Test restores monthly
- Automated restore validation
- Document RTO (Recovery Time Objective) and RPO (Recovery Point Objective)

## Troubleshooting

### Common Issues

1. **Foreign key constraint violations**
   - Enable foreign key checks: `PRAGMA foreign_keys = ON;` (SQLite)
   - Check for orphaned records before deletion

2. **Slow queries**
   - Run `EXPLAIN QUERY PLAN` to identify missing indexes
   - Consider materialized views for complex aggregations

3. **Disk space issues**
   - Run `VACUUM` periodically (SQLite)
   - Enable auto-vacuum: `PRAGMA auto_vacuum = INCREMENTAL;`
   - Implement retention policies

4. **Deadlocks**
   - Always acquire locks in consistent order
   - Use shorter transactions
   - Consider read replicas for read-heavy workloads

## Development Workflow

### Local Development
```bash
# Create fresh database
python build_all_tables.py --db sqlite --path dev.db

# Run application
export GRACE_DB_PATH=dev.db
python demo_ingress_kernel.py
```

### Testing
```bash
# Create test database
python build_all_tables.py --db sqlite --path test.db

# Run tests
export GRACE_DB_PATH=test.db
pytest tests/
```

### Production Deployment
```bash
# Use PostgreSQL for production
python build_all_tables.py \
  --db postgres \
  --host db.example.com \
  --dbname grace_prod \
  --user grace_app \
  --password ${GRACE_DB_PASSWORD}
```

## Future Enhancements

1. **Time-series optimization**: Migrate metrics tables to TimescaleDB
2. **Graph queries**: Add Neo4j for lineage/provenance queries
3. **Vector search**: Add Milvus/Pinecone for embedding search
4. **Multi-tenancy**: Add `tenant_id` column to all tables
5. **Geo-distribution**: Implement multi-region replication

## Support

For questions or issues:
- Documentation: See `grace/contracts/README.md`
- Schema validation: `python grace/contracts/validate_contracts.py`
- Database health check: `python build_all_tables.py --health-check` (coming soon)

---

**Last Updated**: 2025-10-14  
**Schema Version**: 1.0.0  
**Total Tables**: 81 across 11 components
