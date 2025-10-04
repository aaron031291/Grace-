# Grace PROD_RUNBOOK.md — Production Operations Guide

## Purpose
Guide for running, monitoring, and maintaining Grace in production. Covers monitoring, alerting, backup, recovery, and escalation.

---

## 1. Monitoring & Alerting
- Prometheus + Grafana dashboards for API, worker, DB, memory, trust, governance.
- Key metrics: api_availability, worker_job_success, api_p50_latency, trust_component, governance_latency.
- Alerts: SLO breaches, error spikes, rollback triggers.
- Health endpoints: `/health`, `/health/full`

---

## 2. Backup & Recovery
- Nightly snapshots: Qdrant, Postgres, CAS, logs, policies.
- Store backups in multiple regions, signed and versioned.
- Restore procedures: see [DR_RUNBOOK.md](DR_RUNBOOK.md)

---

## 3. Rollback & Auto-Heal
- Blue/green deploy with shadow compare.
- Automated rollback on SLO regression (see alert rules).
- Self-healing pipeline: anomaly detection, RCA, patch proposal, consensus, sandbox, governance, deploy.

---

## 4. Governance & Security
- Quorum approvals for risky ops.
- Policy enforcement: data sovereignty, blast-radius, secrets redaction.
- RBAC per namespace; region-bound access.
- All changes logged immutably.

---

## 5. On-Call & Escalation
- On-call SRE: <update contact>
- Escalation via Prometheus alert (severity: page)
- Governance approval required for destructive actions

---

## 6. References
- [Unified Operating Spec](GRACE_UNIFIED_OPERATING_SPEC_COMPLETE.md)
- [DR_RUNBOOK.md](DR_RUNBOOK.md)
- [ONE_HOUR_SETUP.md](ONE_HOUR_SETUP.md)
