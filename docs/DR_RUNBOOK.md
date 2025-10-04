# Grace Disaster Recovery Runbook & Rehearsal Checklist

## Purpose
Ensure rapid, reliable recovery from major incidents, data loss, or system failures. This runbook covers backup, restore, failover, and periodic rehearsal procedures for the Grace platform.

---

## 1. DR Principles
- Immutable logs and memory snapshots are the source of truth.
- All backups are signed, versioned, and stored in multiple regions.
- Recovery is always auditable and must be tested quarterly.

---

## 2. Backup Procedures
- **Memory (CAS + Qdrant + Postgres):**
  - Nightly snapshot to local and remote vaults.
  - Qdrant: `curl -X POST http://qdrant:6333/collections/<COLLECTION>/snapshots`
  - Postgres: `pg_dump grace_governance > /backups/pg_<date>.sql`
  - CAS: `rsync -a grace/memory/cas/ /backups/cas_<date>/`
- **Immutable Logs:**
  - Archive logs daily: `cp logs/grace.log /backups/log_<date>.log`
- **Config & Policies:**
  - Backup all YAMLs: `tar czf /backups/policies_<date>.tar.gz grace/policy/`

---

## 3. Restore Procedures
- **Qdrant:**
  - `curl -X POST http://qdrant:6333/collections/<COLLECTION>/snapshots/recover`
- **Postgres:**
  - `psql grace_governance < /backups/pg_<date>.sql`
- **CAS:**
  - `rsync -a /backups/cas_<date>/ grace/memory/cas/`
- **Logs:**
  - `cat /backups/log_<date>.log >> logs/grace.log`

---

## 4. Failover & Federation
- If primary node fails, promote standby (see federation config).
- Use gossip protocol for state sync: `grace_federation_sync.py`
- Validate region-based sovereignty before promotion.

---

## 5. Rehearsal Checklist
- [ ] Quarterly restore test (all components)
- [ ] Validate backup integrity (hash/signature)
- [ ] Simulate failover and federation sync
- [ ] Audit log recovery and replay
- [ ] Document all steps and outcomes

---

## 6. Emergency Contacts & Escalation
- On-call SRE: <update contact>
- Escalation: page via monitoring alert (Prometheus severity: page)
- Governance approval required for destructive restores

---

## 7. References
- [Qdrant snapshot policy](../docs/qdrant_snapshot_policy.md)
- [Unified Operating Spec](../docs/GRACE_UNIFIED_OPERATING_SPEC_COMPLETE.md)
