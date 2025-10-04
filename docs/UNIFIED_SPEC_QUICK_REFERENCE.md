# Grace Unified Operating Specification - Quick Reference

## Quick Start

```python
from grace.interface.orb_interface import GraceUnifiedOrbInterface

# Initialize
orb = GraceUnifiedOrbInterface()

# Start unified orchestrator
await orb.start_unified_orchestrator()

# Get comprehensive stats
stats = orb.get_unified_stats()
print(f"Version: {stats['version']}")
```

## Core Components

### 1. Context Bus - Unified "Now"

```python
# Get current context state
context = orb.get_context_bus_state()

# Context is automatically shared across:
# - Loop Orchestrator
# - Memory Explorer
# - AVN Engine
# - Trust Metrics
```

### 2. Loop Orchestrator - Tick Scheduler

```python
# Start (auto-starts on orb initialization if available)
await orb.start_unified_orchestrator()

# Check health
health = orb.get_orchestrator_health()
print(f"Ticks: {health['tick_count']}, Loops: {health['loops_count']}")

# Stop
await orb.stop_unified_orchestrator()
```

### 3. Memory Explorer - File-Explorer Brain

```python
# Create memory folder
folder = await orb.create_memory_folder(
    folder_id="patterns",
    purpose="System patterns",
    domain="system",
    policies=["trust_threshold_0.7"]
)

# Search with trust ranking
results = await orb.search_unified_memory(
    query="api timeout",
    filters={"min_trust": 0.7, "limit": 10}
)
```

### 4. AVN/RCA/Healing - Self-Healing

```python
# Detect anomaly
anomaly = await orb.detect_anomaly(
    metric_name="api_latency",
    current_value=950.0,
    expected_value=200.0,
    threshold=0.2
)

# Perform RCA
if anomaly:
    hypotheses = await orb.perform_root_cause_analysis(anomaly['anomaly_id'])
    
    # Execute healing in sandbox
    if hypotheses:
        result = await orb.execute_healing_action(
            hypotheses[0]['hypothesis_id'],
            sandbox=True
        )
```

### 5. Trust & KPI Framework

```python
# Update component trust
trust = await orb.update_component_trust(
    component_id="api_service",
    trust_score=0.85,
    confidence=0.9
)

# Record KPI
kpi = await orb.record_performance_kpi(
    metric_type="mttr",  # or mttu, governance_latency, learning_gain
    value=3.5,
    unit="minutes",
    target=5.0
)
```

### 6. Voice Modes

```python
# Set voice mode
await orb.set_unified_voice_mode("co_partner")
# Options: solo_voice, text_only, co_partner, silent_autonomous

# Execute voice command
command = await orb.execute_unified_voice_command(
    intent="optimize API performance",
    user_id="user123"
)
```

## REST API Quick Reference

### Get Stats & Health

```bash
# Comprehensive stats
GET /api/orb/v1/unified/stats

# Orchestrator health
GET /api/orb/v1/unified/health

# Context bus state
GET /api/orb/v1/unified/context
```

### Memory Operations

```bash
# Create folder
POST /api/orb/v1/unified/memory/folder
{
  "folder_id": "api_patterns",
  "purpose": "API patterns",
  "domain": "api",
  "policies": ["trust_threshold_0.7"]
}

# Search memory
POST /api/orb/v1/unified/memory/search
{
  "query": "timeout",
  "filters": {"min_trust": 0.7}
}
```

### Anomaly & Healing

```bash
# Detect anomaly
POST /api/orb/v1/unified/anomaly/detect
{
  "metric_name": "api_latency",
  "current_value": 950.0,
  "expected_value": 200.0,
  "threshold": 0.2
}

# Perform RCA
POST /api/orb/v1/unified/anomaly/rca
{
  "anomaly_id": "anom_12345"
}

# Execute healing
POST /api/orb/v1/unified/healing/execute
{
  "hypothesis_id": "hyp_67890",
  "sandbox": true
}
```

### Voice & Trust

```bash
# Set voice mode
POST /api/orb/v1/unified/voice/mode
{
  "mode": "co_partner"
}

# Execute voice command
POST /api/orb/v1/unified/voice/command
{
  "intent": "optimize performance",
  "user_id": "user123"
}

# Update trust
POST /api/orb/v1/unified/trust/update
{
  "component_id": "api_service",
  "trust_score": 0.85,
  "confidence": 0.9
}

# Record KPI
POST /api/orb/v1/unified/kpi/record
{
  "metric_type": "mttr",
  "value": 3.5,
  "unit": "minutes",
  "target": 5.0
}
```

## Data Structures

### Event Envelope

```python
{
  "event_type": "grace.event.v1",
  "actor": "human|grace",
  "component": "memory|avn|governance|orb",
  "payload": {...},
  "kpi_deltas": {"latency_ms": -420},
  "trust_before": 0.74,
  "trust_after": 0.80,
  "confidence": 0.82,
  "immutable_hash": "sha256:...",
  "trace_id": "...",
  "timestamp": "2025-10-04T12:00Z",
  "correlation_id": "..."
}
```

### Memory Item

```python
{
  "id": "mem_123",
  "path": "knowledge/patterns/api_resilience/",
  "tags": ["fastapi", "timeouts"],
  "trust": 0.89,
  "last_used": "2025-10-04T12:00Z",
  "policy_refs": ["secrets_redaction"],
  "vector_ref": "vec://api_patterns/a1b2"
}
```

### Trust Metric

```python
{
  "metric_id": "trust_component_api_service",
  "component_id": "api_service",
  "trust_score": 0.85,
  "trust_drift": 0.02,
  "samples": 100,
  "rolling_window_days": 30
}
```

### KPI Metric

```python
{
  "metric_id": "kpi_mttr_123",
  "metric_type": "mttr",
  "value": 3.5,
  "unit": "minutes",
  "target": 5.0,
  "status": "normal"  # or warning, critical
}
```

## Voice Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `solo_voice` | Grace listens, executes, narrates | Single user voice control |
| `text_only` | Command palette + keyboard | Identical semantics without voice |
| `co_partner` | Multi-user collaborative | Team discussions with Grace |
| `silent_autonomous` | Acts within delegated scope | Background automation |

## Success Metrics

| Category | KPI | Target | Type |
|----------|-----|--------|------|
| Comprehension | MTTU | < 10 s | `mttu` |
| Healing | MTTR | < 5 min | `mttr` |
| Governance | Latency | < 60 s | `governance_latency` |
| Learning | Gain | Track | `learning_gain` |
| Performance | General | Track | `kpi_perf` |

## Common Patterns

### Complete Healing Workflow

```python
# 1. Detect
anomaly = await orb.detect_anomaly("api_latency", 950, 200, 0.2)

# 2. Analyze
hypotheses = await orb.perform_root_cause_analysis(anomaly['anomaly_id'])

# 3. Heal
result = await orb.execute_healing_action(hypotheses[0]['hypothesis_id'], sandbox=True)

# 4. Update Trust
await orb.update_component_trust("api_service", 0.88, 0.9)

# 5. Record MTTR
await orb.record_performance_kpi("mttr", 2.8, "minutes", 5.0)
```

### Memory Pattern Storage

```python
# 1. Create folder
await orb.create_memory_folder("api_patterns", "API patterns", "api")

# 2. Store items (via unified_spec)
item = MemoryItem(
    id="pattern_001",
    path="api_patterns/timeout",
    tags=["timeout", "api"],
    trust=0.9,
    metadata={"fix": "Increase timeout"}
)
await orb.unified_spec.memory_explorer.store_memory_item(item)

# 3. Search
results = await orb.search_unified_memory("timeout", {"min_trust": 0.7})

# 4. Update trust based on success
await orb.unified_spec.memory_explorer.update_trust_feedback(
    "pattern_001", success=True
)
```

## Testing

```bash
# Run validation test
python demo_and_tests/tests/test_unified_spec_validation.py

# Run interactive demo
python demo_and_tests/unified_spec_demo.py
```

## Troubleshooting

### Unified Spec Not Available

```python
if not orb.unified_spec:
    print("Unified spec integration not available")
    # Import error - module not found
```

**Solution**: The unified_spec module loads independently. Check that `grace/interface/unified_spec_integration.py` exists.

### Trust Drift Warning

```
Trust drift exceeded bounds for api_gateway: 0.078
```

**Meaning**: Trust changed by more than ±0.05 in the 30-day rolling window. This is logged as a warning but allowed.

### KPI Status

- `normal`: Value within target or no target set
- `warning`: Value > target * 1.2
- `critical`: Value > target * 1.5

## References

- [Full Documentation](./UNIFIED_SPEC_INTEGRATION.md)
- [North-Star Spec](../GRACE_UNIFIED_OPERATING_SPECIFICATION.md)
- [Orb Interface](../README.md#orb-interface)
