# Grace BusinessOps Kernel

The BusinessOps Kernel is Grace's execution layer: it takes an approved GovernedDecision from Governance and carries out the plan via sandboxed, allow-listed plugins, with full W5H tagging, trust updates, immutable audit logs, and trigger events through the MTL kernel.

## Core Responsibilities

1. Execute approved plans (sequential/parallel steps) from GovernedDecision
2. Sandbox & policy enforcement for each step (network, FS, process limits)  
3. Plugin registry & allowlist with versioning and capability tags
4. Idempotent, observable runs: timeouts, retries, backoff, and dedupe
5. Full telemetry: W5H memories, trust deltas, immutable logs, trigger events
6. Run reports: per-step results + overall outcome for IDE and Learning loops
7. Operational health: readiness/liveness, plugin health, config snapshot

## Installation

```bash
pip install -e .[dev]
```

## Usage

```python
from grace.business_ops import BusinessOpsKernel
from grace.mtl import MTLKernel

# Initialize
mtl = MTLKernel()
kernel = BusinessOpsKernel(mtl)
await kernel.initialize()

# Execute approved decision
decision = {
    "decision_id": "01JABC...",
    "approved": True,
    "plan": {
        "mode": "sequential",
        "steps": [
            {"name": "notify", "plugin": "email_send", "args": {...}},
            {"name": "index", "plugin": "http_call", "args": {...}}
        ]
    }
}

report = await kernel.execute(decision)
print(f"Status: {report['overall_status']}")
```

## API

The kernel exposes REST endpoints:

- `POST /api/business_ops/execute` - Execute approved decision
- `GET /api/business_ops/health` - Health check

## Architecture

- **Kernel**: Core execution engine with plugin registry
- **Plugins**: Sandboxed execution units for specific capabilities
- **MTL Integration**: Audit logging and W5H tracking
- **Sandbox**: Policy enforcement and resource limits
- **Observatory**: Metrics and monitoring 
