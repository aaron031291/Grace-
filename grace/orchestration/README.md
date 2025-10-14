# Grace TriggerMesh Orchestration Layer

## Overview

The TriggerMesh orchestration layer acts as the **event-driven nervous system** for Grace, connecting tables, APIs, and kernels through intelligent event routing. This layer ensures that KPI/trust metric changes automatically trigger the appropriate self-healing workflows.

## Architecture Principles

### 1️⃣ Repository Alignment
- **Module Placement**: TriggerMesh orchestrations live in `orchestration/` folder
- **Separation of Concerns**:
  - `tables/` (data layer) → emit events
  - `api/` (interface layer) → expose events
  - `orchestration/` (routing layer) → route events to kernels
  - `kernels/` (logic layer) → execute workflows
- **Interface Contracts**: All events match schemas in `schemas/` folder
- **Versioning**: Workflow definitions stored in repo for testing & rollback

### 2️⃣ Cohesion with Existing Architecture
- **Event-Driven Consistency**: TriggerMesh enhances (not bypasses) existing kernels
- **Schema Awareness**: Mapping file links metrics → events → kernels
- **Decoupling**: Kernels respond to inputs without knowing about TriggerMesh

### 3️⃣ Effective Operation
- **Event Filtering**: Only meaningful events fire workflows (threshold-based)
- **Parallelization**: Multiple workflows run concurrently
- **Logging & Traceability**: All actions logged to ImmutableLogs

## Directory Structure

```
grace/orchestration/
├── README.md                           # This file
├── __init__.py                         # Module exports
├── workflows/                          # TriggerMesh workflow definitions
│   ├── __init__.py
│   ├── trigger_mesh_workflows.yaml     # Main workflow configurations
│   ├── kpi_workflows.yaml              # KPI-triggered workflows
│   ├── trust_workflows.yaml            # Trust metric workflows
│   └── self_healing_workflows.yaml     # Self-healing orchestrations
├── event_router.py                     # Core event routing engine
├── workflow_engine.py                  # Workflow execution engine
├── workflow_registry.py                # Workflow registration & discovery
├── event_filters.py                    # Threshold & delta-based filtering
├── bridges/                            # Existing bridges
│   ├── mesh_bridge.py                  # Event mesh integration
│   ├── governance_bridge.py            # Governance kernel bridge
│   └── memory_bridge.py                # Memory orchestration bridge
└── logs/                               # TriggerMesh execution logs
    └── trigger_mesh.log                # Event routing & execution logs

grace/schemas/
├── event_schemas.yaml                  # Event payload schemas
├── workflow_schemas.yaml               # Workflow definition schemas
└── governance_events.yaml              # Existing governance events

grace/api/
├── metrics_api.py                      # KPI/metrics API (to create)
├── trust_api.py                        # Trust score API (to create)
└── events_api.py                       # Event subscription API (to create)
```

## Event Flow

```
┌─────────────────┐
│  Tables Layer   │ metrics_table.py, trust_table.py
└────────┬────────┘
         │ emit event
         ▼
┌─────────────────┐
│   API Layer     │ metrics_api.py, trust_api.py
└────────┬────────┘
         │ publish event
         ▼
┌─────────────────┐
│ TriggerMesh     │ event_router.py, workflow_engine.py
│ Orchestration   │ ← reads trigger_mesh_workflows.yaml
└────────┬────────┘
         │ route to kernel
         ▼
┌─────────────────┐
│ Kernels Layer   │ ingress_kernel, learning_kernel, etc.
└────────┬────────┘
         │ execute & emit result
         ▼
┌─────────────────┐
│ Tables/Logs     │ results → tables, logs → immutable_logs
└─────────────────┘
```

## Workflow Definition Example

```yaml
# trigger_mesh_workflows.yaml
workflows:
  - name: "kpi_degradation_healing"
    trigger:
      event_type: "kpi.threshold_breach"
      filters:
        severity: ["WARNING", "CRITICAL"]
        metric_type: "test_quality_score"
    actions:
      - condition: "payload.severity == 'CRITICAL'"
        target_kernel: "avn_core"
        action: "escalate_healing"
        parameters:
          component_id: "{{ payload.component_id }}"
          current_score: "{{ payload.value }}"
          threshold: "{{ payload.threshold }}"
          
      - condition: "payload.severity == 'WARNING'"
        target_kernel: "learning_kernel"
        action: "trigger_adaptive_learning"
        parameters:
          component_id: "{{ payload.component_id }}"
          focus_areas: ["error_pattern_analysis", "coverage_improvement"]
    
    logging:
      source_metric: "{{ payload.metric_name }}"
      triggered_kernel: "{{ action.target_kernel }}"
      outcome: "{{ action.result }}"
```

## Integration with Existing Systems

### KPITrustMonitor Integration
```python
# In kpi_trust_monitor.py
async def record_metric(self, name, value, component_id, threshold_warning, threshold_critical):
    # ... existing logic ...
    
    # Emit event if threshold breached
    if value < threshold_critical:
        await self.event_publisher(
            "kpi.threshold_breach",
            {
                "metric_name": name,
                "component_id": component_id,
                "value": value,
                "threshold": threshold_critical,
                "severity": "CRITICAL"
            }
        )
```

### TestQualityMonitor Integration
```python
# In test_quality_monitor.py
async def _check_self_healing_trigger(self, component_id, status, score):
    # Publish event instead of direct escalation
    await self.event_bus.publish(
        "test_quality.healing_required",
        {
            "component_id": component_id,
            "status": status.value,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    # TriggerMesh will route to appropriate kernel
```

## Key Features

### 1. Event Filtering
```python
# event_filters.py
class EventFilter:
    def threshold_filter(self, event, threshold):
        """Only trigger if metric crosses threshold"""
        
    def delta_filter(self, event, min_change):
        """Only trigger if change > min_change"""
        
    def rate_limit_filter(self, event, max_per_minute):
        """Prevent event storms"""
```

### 2. Parallel Execution
```python
# workflow_engine.py
async def execute_workflow(self, workflow, event):
    # Execute all actions in parallel where possible
    tasks = [
        self._execute_action(action, event)
        for action in workflow.actions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Traceability
```python
# All TriggerMesh actions log to ImmutableLogs
await immutable_logs.log_event(
    event_type="workflow_executed",
    component_id="trigger_mesh",
    event_data={
        "workflow_name": workflow.name,
        "source_event": event.type,
        "target_kernel": action.target_kernel,
        "result": result,
        "latency_ms": latency
    }
)
```

## Getting Started

### 1. Load Workflows
```python
from grace.orchestration import WorkflowRegistry, EventRouter

# Load workflow definitions from YAML
registry = WorkflowRegistry()
await registry.load_workflows("grace/orchestration/workflows/")

# Create event router
router = EventRouter(registry, event_bus, immutable_logs)
await router.start()
```

### 2. Subscribe to Events
```python
# Router automatically subscribes to all workflow trigger events
# No manual subscription needed
```

### 3. Monitor Execution
```bash
# View TriggerMesh logs
tail -f grace/orchestration/logs/trigger_mesh.log

# Query ImmutableLogs for workflow execution
python scripts/query_workflow_logs.py --workflow kpi_degradation_healing
```

## Testing

```python
# tests/test_orchestration.py
async def test_kpi_workflow_triggers_avn():
    # Emit KPI threshold breach event
    await event_bus.publish(
        "kpi.threshold_breach",
        {"severity": "CRITICAL", "component_id": "test_comp"}
    )
    
    # Verify AVN core was called
    assert avn_core.escalations[-1]["component_id"] == "test_comp"
```

## Versioning & Rollback

All workflow definitions are in Git:
```bash
# View workflow history
git log -- grace/orchestration/workflows/

# Rollback to previous version
git checkout <commit> -- grace/orchestration/workflows/kpi_workflows.yaml

# Reload workflows without restart
python scripts/reload_workflows.py --hot-reload
```

## Performance Metrics

- **Event Latency**: < 5ms routing overhead
- **Workflow Execution**: Logged with timestamps
- **Success Rate**: Tracked per workflow
- **Kernel Response Time**: Measured end-to-end

## Security & Governance

- **Event Validation**: All events validated against schemas
- **RBAC Integration**: Workflow execution requires permissions
- **Audit Trail**: Full traceability in ImmutableLogs
- **Constitutional Compliance**: Workflows can't bypass governance

## Next Steps

1. Create workflow YAML definitions
2. Implement EventRouter and WorkflowEngine
3. Create metrics/trust APIs
4. Wire up KPITrustMonitor event publishing
5. Update TestQualityMonitor to use events
6. Add workflow tests
7. Deploy and monitor
