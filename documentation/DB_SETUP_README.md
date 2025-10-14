# Grace System - Database Setup

Complete database initialization for all Grace components and kernels.

## 📋 Quick Start

```bash
# 1. Build all tables (SQLite)
python build_all_tables.py --db sqlite --path grace_system.db

# 2. Verify database
python verify_database.py grace_system.db

# 3. View documentation
cat DATABASE_SCHEMA.md
```

## 📁 Files

| File | Description |
|------|-------------|
| `init_all_tables.sql` | Complete SQL DDL for all 81 tables (1800+ lines) |
| `build_all_tables.py` | Python script to build tables (SQLite/PostgreSQL/MySQL) |
| `verify_database.py` | Database verification script |
| `grace_system.db` | Pre-built SQLite database (ready to use) |
| `DATABASE_SCHEMA.md` | Complete schema documentation |

## 🗄️ Database Structure

```
81 tables across 11 components:
├─ Core (6): Audit logs, governance, instance state
├─ Ingress Kernel (13): Bronze/Silver/Gold data pipeline
├─ Intelligence Kernel (9): ML inference & routing
├─ Learning Kernel (15): Data-centric ML
├─ Memory Kernel (8): Lightning cache + Fusion storage
├─ Orchestration Kernel (5): State machine & scheduling
├─ Resilience Kernel (8): Circuit breakers & degradation
├─ MLT Core (5): Memory-Learning-Trust meta-learning
├─ MLDL Components (8): Model registry & deployment
├─ Communications (2): DLQ & deduplication
└─ Common (2): Snapshots & rollback
```

## 🚀 Usage Examples

### SQLite (Development)
```bash
python build_all_tables.py --db sqlite --path dev.db
export GRACE_DB_PATH=dev.db
python demo_ingress_kernel.py
```

### PostgreSQL (Production)
```bash
python build_all_tables.py \
  --db postgres \
  --host localhost \
  --dbname grace_prod \
  --user grace \
  --password "${DB_PASSWORD}"
```

### MySQL
```bash
python build_all_tables.py \
  --db mysql \
  --host localhost \
  --dbname grace \
  --user grace \
  --password "${DB_PASSWORD}"
```

### List All Components
```bash
python build_all_tables.py --list
```

## 🔍 Verification

```bash
# Run full verification
python verify_database.py grace_system.db

# Check specific tables
sqlite3 grace_system.db "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"

# Check row counts
sqlite3 grace_system.db "SELECT 'sources', COUNT(*) FROM sources UNION ALL SELECT 'intel_requests', COUNT(*) FROM intel_requests;"

# Verify foreign keys
sqlite3 grace_system.db "PRAGMA foreign_key_check;"
```

## 📊 Key Features

### ✅ Governance & Audit
- **Immutable audit logs** with blockchain-style chaining (SHA-256)
- **Governance decisions** tracked for all policy changes
- **PII detection** and policy enforcement at ingestion
- **Data lineage** tracking through all transformations

### ✅ Data Pipeline (Ingress)
- **Bronze tier**: Raw events (append-only)
- **Silver tier**: Normalized, validated records
- **Gold tier**: Curated, ML-ready features
- **Trust scoring** for all data sources

### ✅ ML Operations
- **Intelligence Kernel**: Inference routing with canary/shadow deployments
- **Learning Kernel**: Data-centric ML with active learning
- **MLT Core**: Meta-learning for continuous improvement
- **Model Registry**: Full lifecycle management

### ✅ Resilience
- **Circuit breakers** per component (closed/open/half_open)
- **Graceful degradation** policies (mild/moderate/severe)
- **SLI/SLO tracking** with automatic alerts
- **Automated recovery** actions

### ✅ Memory System
- **Lightning memory**: Fast cache with TTL
- **Fusion memory**: Persistent storage with compression
- **Librarian**: Content + semantic indexing
- **Access patterns**: Optimization insights

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Ingress     Intelligence    Learning    Orchestration      │
│  Kernel      Kernel          Kernel      Kernel             │
├─────────────────────────────────────────────────────────────┤
│              Memory Kernel + Resilience Kernel              │
├─────────────────────────────────────────────────────────────┤
│                    Core (Audit + Governance)                │
├─────────────────────────────────────────────────────────────┤
│              Database (SQLite/PostgreSQL/MySQL)             │
└─────────────────────────────────────────────────────────────┘
```

## 🔐 Security

- **Encryption at rest**: Database file encryption (SQLite) or TDE (PostgreSQL)
- **Encryption in transit**: TLS/SSL for all connections
- **Secrets management**: External secret managers (never store in DB)
- **RBAC**: Role-based access control
- **Audit trail**: All operations logged

## 📈 Performance

### Indexing
- Primary keys on all tables
- Foreign keys for referential integrity
- Composite indexes for common queries
- Full-text indexes for searchable fields

### Optimization
- JSON fields (SQLite TEXT, PostgreSQL JSONB)
- BLOB compression support (gzip, lz4, zstd)
- Materialized views for complex aggregations
- Partitioning for time-series data (PostgreSQL)

## 🔧 Maintenance

### Backups
```bash
# SQLite
sqlite3 grace_system.db ".backup grace_backup.db"

# PostgreSQL
pg_dump grace_prod > grace_backup.sql

# MySQL
mysqldump grace > grace_backup.sql
```

### Vacuum (SQLite)
```bash
sqlite3 grace_system.db "VACUUM;"
```

### Health Checks
```bash
# Check for stuck tasks
sqlite3 grace_system.db "SELECT * FROM orchestration_tasks WHERE status='running' AND started_at < datetime('now', '-1 hour');"

# Check circuit breaker status
sqlite3 grace_system.db "SELECT component_name, state, failure_count FROM circuit_breakers WHERE state != 'closed';"

# Check for high error rates
sqlite3 grace_system.db "SELECT source_id, AVG(error_rate) FROM processing_metrics WHERE timestamp >= datetime('now', '-1 hour') GROUP BY source_id HAVING AVG(error_rate) > 0.05;"
```

## 🐛 Troubleshooting

### Database locked
```bash
# Check for processes holding locks
lsof grace_system.db

# Enable WAL mode (SQLite)
sqlite3 grace_system.db "PRAGMA journal_mode=WAL;"
```

### Slow queries
```bash
# Analyze query plan
sqlite3 grace_system.db "EXPLAIN QUERY PLAN SELECT * FROM intel_requests WHERE status='pending';"

# Rebuild indexes
sqlite3 grace_system.db "REINDEX;"
```

### Disk space
```bash
# Check database size
ls -lh grace_system.db

# Check table sizes
sqlite3 grace_system.db "SELECT name, (pgsize*pageno)/1024/1024 AS size_mb FROM dbstat WHERE aggregate=TRUE ORDER BY size_mb DESC LIMIT 10;"
```

## 📚 Documentation

- **[DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)** - Complete schema documentation
- **[init_all_tables.sql](init_all_tables.sql)** - Raw SQL DDL
- **grace/contracts/** - Data contracts and validation

## 🧪 Testing

```bash
# Create test database
python build_all_tables.py --db sqlite --path test.db

# Run tests with test database
export GRACE_DB_PATH=test.db
pytest tests/ -v

# Verify test database
python verify_database.py test.db
```

## 🔄 Migration

For production migrations:
1. Use migration tools (Alembic for Python, Flyway for Java)
2. Never drop columns - mark as deprecated
3. Test migrations on staging first
4. Take backup before migration
5. Run verification after migration

## 📞 Support

- Issues with table creation? Check `build_all_tables.py` logs
- Schema questions? See [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)
- Contract validation? Run `python grace/contracts/validate_contracts.py`

---

**Status**: ✅ All 81 tables built and verified  
**Database Size**: ~1 MB (empty)  
**Last Updated**: 2025-10-14  
**Version**: 1.0.0
