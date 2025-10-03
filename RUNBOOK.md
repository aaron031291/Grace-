# Grace Governance Kernel Runbook

## SLOs & Alert Matrix
- **Uptime SLO:** 99.9%
- **Governance Decision Latency:** < 2s
- **Audit Integrity:** 100%
- **Alert Matrix:**
  - Kernel crash: PagerDuty, Slack
  - Health check fail: Email, dashboard
  - Audit chain break: Immediate escalation

## Snapshot/Rollback
- Use `scripts/snapshot_create.py` to capture state
- Use `scripts/snapshot_rollback.py` to restore state in <5 min

## Blue/Green Deployment Steps
- Use `scripts/shadow_compare.py` to run shadow governance
- Compare decisions, validate accuracy
- Roll out new kernel if shadow passes

## Clean Shutdown
- Use canonical runner or API endpoint to trigger graceful shutdown

## Monitoring
- Heartbeat emitted to `/monitoring/heartbeat.log` by watchdog
- Grafana dashboard: `/monitoring/grafana/provisioning/dashboards/grace_governance_dashboard.json`

## Troubleshooting
- Check logs in `/logs/`
- Use `scripts/verify_chain.py` for audit integrity
- Use health check scripts for subsystem status

## Links
- [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md)
- [API Docs](docs/api/openapi.yaml)
- [Advanced Features](docs/features/ADVANCED_GOVERNANCE.md)
- [Security](docs/security/SECURITY_HARDENING.md)
