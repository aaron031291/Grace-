# Grace Database - Quick Reference

## 🚀 One-Liner Commands

```bash
# Build database (SQLite)
python build_all_tables.py --db sqlite --path grace_system.db

# Verify database
python verify_database.py grace_system.db

# List components
python build_all_tables.py --list
```

## 📊 Component Summary

| Component | Tables | Purpose |
|-----------|--------|---------|
| Core | 6 | Audit logs, governance, instance state |
| Ingress Kernel | 13 | Bronze/Silver/Gold data pipeline |
| Intelligence Kernel | 9 | ML inference routing & deployments |
| Learning Kernel | 15 | Data-centric ML & active learning |
| Memory Kernel | 8 | Lightning cache + Fusion storage |
| Orchestration Kernel | 5 | State machine & scheduling |
| Resilience Kernel | 8 | Circuit breakers & degradation |
| MLT Core | 5 | Memory-Learning-Trust meta-learning |
| MLDL Components | 8 | Model registry & deployment |
| Communications | 2 | DLQ & message deduplication |
| Common | 2 | Snapshots & rollback |
| **TOTAL** | **81** | **All Grace components** |

## 🗄️ Key Tables by Use Case

### Audit & Governance
- `audit_logs` - Immutable log with blockchain chaining
- `governance_decisions` - Policy approvals
- `chain_verification` - Audit chain validation

### Data Pipeline
- `bronze_raw_events` - Raw ingestion
- `silver_records` - Normalized data
- `gold_feature_datasets` - ML-ready features

### ML Operations
- `intel_requests` - Inference requests
- `intel_plans` - Execution plans
- `intel_results` - Inference results
- `models` - Model registry
- `model_deployments` - Active deployments

### Active Learning
- `datasets` - Dataset registry
- `labels` - Human labels
- `active_queries` - Sample selection
- `weak_labelers` - Weak supervision

### Memory
- `lightning_memory` - Fast cache (TTL)
- `fusion_memory` - Persistent storage
- `librarian_index` - Search index

### Resilience
- `circuit_breakers` - Circuit breaker states
- `resilience_incidents` - Incident tracking
- `sli_measurements` - SLI metrics

## 🔍 Useful Queries

```sql
-- Check table sizes
SELECT name, 
       COUNT(*) as rows,
       pg_size_pretty(pg_total_relation_size(name::regclass)) as size
FROM (SELECT table_name as name FROM information_schema.tables 
      WHERE table_schema='public') t
GROUP BY name ORDER BY rows DESC;

-- View recent audit logs
SELECT category, timestamp, transparency_level 
FROM audit_logs 
ORDER BY timestamp DESC LIMIT 10;

-- Check source health
SELECT s.source_id, s.kind, sh.status, sh.error_count
FROM sources s
LEFT JOIN source_health sh ON s.source_id = sh.source_id
WHERE s.enabled = TRUE;

-- View active intelligence requests
SELECT req_id, task, status, created_at
FROM intel_requests
WHERE status IN ('received', 'processing')
ORDER BY created_at DESC;

-- Check circuit breaker status
SELECT component_name, state, failure_count, last_failure_time
FROM circuit_breakers
WHERE state != 'closed';

-- View model deployments
SELECT m.model_id, m.version, d.status, d.deployed_at
FROM models m
JOIN model_deployments d ON m.model_id = d.model_id
WHERE d.status = 'running';
```

## 📁 File Reference

```
/workspaces/Grace-/
├── init_all_tables.sql      # SQL DDL (1,315 lines)
├── build_all_tables.py      # Table builder (SQLite/PG/MySQL)
├── verify_database.py       # Verification tool
├── grace_system.db          # Pre-built database (1 MB)
├── DATABASE_SCHEMA.md       # Complete documentation
└── DB_SETUP_README.md       # Quick start guide
```

## 🛠️ Maintenance

```bash
# Backup (SQLite)
sqlite3 grace_system.db ".backup grace_backup.db"

# Vacuum
sqlite3 grace_system.db "VACUUM;"

# Check integrity
sqlite3 grace_system.db "PRAGMA integrity_check;"

# Foreign key check
sqlite3 grace_system.db "PRAGMA foreign_key_check;"

# Enable WAL mode (better concurrency)
sqlite3 grace_system.db "PRAGMA journal_mode=WAL;"
```

## 🔐 Security Checklist

- [ ] Enable encryption at rest
- [ ] Use TLS/SSL for connections
- [ ] Store secrets externally (not in DB)
- [ ] Enable foreign key constraints
- [ ] Set up RBAC
- [ ] Configure audit logging
- [ ] Regular backups
- [ ] Test restore procedures

## 📈 Performance Tips

1. **Indexes**: Already created for common queries
2. **JSON**: Use JSONB in PostgreSQL for better performance
3. **Partitioning**: Partition large tables by time (PostgreSQL)
4. **Compression**: Enable for BLOB fields
5. **Vacuum**: Run periodically (SQLite/PostgreSQL)
6. **Connection pooling**: Use for production

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Database locked | Check `lsof grace_system.db`, enable WAL mode |
| Slow queries | Run `EXPLAIN`, add indexes, use VACUUM |
| Disk full | Check retention policies, compress old data |
| Foreign key error | Enable FK checks: `PRAGMA foreign_keys=ON` |
| Schema mismatch | Run `verify_database.py` |

## 📞 Support

- **Documentation**: [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)
- **Quick Start**: [DB_SETUP_README.md](DB_SETUP_README.md)
- **Verification**: `python verify_database.py grace_system.db`
- **Contract validation**: `python grace/contracts/validate_contracts.py`

---

**Version**: 1.0.0 | **Last Updated**: 2025-10-14 | **Tables**: 81
