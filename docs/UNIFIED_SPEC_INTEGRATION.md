# Grace Unified Operating Specification - Integration Documentation

## Overview

This document describes the integration of the Grace Unified Operating Specification with the Orb Interface. The integration implements the North-Star Architecture that transforms Grace into a self-healing, self-learning, transparent co-partner.

## Architecture

The Unified Operating Specification integrates the following components:

```
        ┌─────────────┐
        │    ORB UI   │  ← Voice / Text / Hover
        └─────┬───────┘
              │
     ┌────────▼────────┐
     │  Intent Parser  │
     └────────┬────────┘
              │
     ┌────────▼───────────┐
     │  Core Orchestrator │  ← tick scheduler + context bus
     └────────┬───────────┘
              │
 ┌────────────▼────────────┐
 │   AVN / RCA / Healing   │
 └────────────┬────────────┘
              │
 ┌────────────▼────────────┐
 │    Memory Explorer      │  ← File-Explorer-like brain
 └────────────┬────────────┘
              │
 ┌────────────▼────────────┐
 │  Governance & Ledger    │  ← Immutable logs, policies
 └────────────┬────────────┘
              │
 ┌────────────▼────────────┐
 │ Observability / Metrics │
 └─────────────────────────┘
```

## Core Components

### 1. Context Bus

The Context Bus provides unified "now" across all components, ensuring synchronization and shared state.

**Key Features:**
- Unified context sharing across all components
- Subscriber pattern for context changes
- History tracking for audit and replay
- Thread-safe operations with async locks

**Usage:**
```python
from grace.interface.orb_interface import GraceUnifiedOrbInterface

orb = GraceUnifiedOrbInterface()

# Get current context state
context = orb.get_context_bus_state()
```

### 2. Loop Orchestrator

The Loop Orchestrator provides tick-based scheduling for orchestration loops with priority management.

**Key Features:**
- Tick-based execution (default 1s intervals)
- Priority-based loop scheduling
- Health monitoring and metrics
- Async loop execution with error handling

**Usage:**
```python
# Start the orchestrator
await orb.start_unified_orchestrator()

# Check health status
health = orb.get_orchestrator_health()
print(f"Tick count: {health['tick_count']}")
print(f"Running loops: {len(health['loops'])}")

# Stop the orchestrator
await orb.stop_unified_orchestrator()
```

### 3. Enhanced Memory Explorer

File-Explorer interface for Grace's cognition with trust feedback and auto-classification.

**Key Features:**
- Folder context manifests define purpose, domain, and policies
- Auto-classification based on folder domain
- Trust feedback from success/failure loops
- Adjacency graph for related items
- Search with trust ranking

**Usage:**
```python
# Create memory folder with context manifest
folder = await orb.create_memory_folder(
    folder_id="api_patterns",
    purpose="API resilience patterns",
    domain="api_resilience",
    policies=["secrets_redaction", "trust_threshold_0.7"]
)

# Search memory with trust ranking
results = await orb.search_unified_memory(
    query="timeout handling",
    filters={"min_trust": 0.7, "tags": ["resilience"], "limit": 10}
)

for item in results:
    print(f"{item['path']} (trust={item['trust']:.2f})")
```

### 4. AVN/RCA/Healing Engine

Anomaly detection, root cause analysis, and automated healing with sandbox proofing.

**Key Features:**
- Anomaly detection from metrics
- Root cause analysis using Memory Explorer patterns
- Sandbox proof execution before production
- Self-healing with trust feedback
- Healing history tracking

**Usage:**
```python
# Detect anomaly
anomaly = await orb.detect_anomaly(
    metric_name="api_latency",
    current_value=850.0,
    expected_value=200.0,
    threshold=0.2  # 20% deviation threshold
)

if anomaly:
    print(f"Anomaly: {anomaly['metric_name']} (severity={anomaly['severity']})")
    
    # Perform root cause analysis
    hypotheses = await orb.perform_root_cause_analysis(anomaly['anomaly_id'])
    
    for h in hypotheses:
        print(f"Hypothesis: {h['root_cause']}")
        print(f"Confidence: {h['confidence']:.2%}")
        print(f"Suggested fix: {h['suggested_fix']}")
        
        # Execute healing in sandbox
        result = await orb.execute_healing_action(
            h['hypothesis_id'],
            sandbox=True
        )
        print(f"Sandbox result: {result['success']}")
```

### 5. Trust & KPI Framework

Comprehensive metrics tracking for trust, performance, and governance.

**Key Features:**
- Component trust tracking with drift detection (bounded ±0.05)
- Memory item trust tracking
- Performance KPIs (MTTR, MTTU, governance latency)
- Learning gain metrics
- Status indicators (normal, warning, critical)

**Usage:**
```python
# Update component trust
trust = await orb.update_component_trust(
    component_id="api_service",
    trust_score=0.85,
    confidence=0.9
)
print(f"Trust: {trust['trust_score']}, Drift: {trust['trust_drift']}")

# Record KPI metrics
kpi = await orb.record_performance_kpi(
    metric_type="mttr",  # Mean Time to Repair
    value=3.5,
    unit="minutes",
    target=5.0
)
print(f"MTTR: {kpi['value']}{kpi['unit']} (status={kpi['status']})")

# Record governance latency
gov_kpi = await orb.record_performance_kpi(
    metric_type="governance_latency",
    value=45.0,
    unit="seconds",
    target=60.0
)
```

### 6. Voice & Collaboration Modes

Multi-modal interaction with voice commands and collaboration support.

**Voice Modes:**
- **solo_voice**: Grace listens, executes, narrates
- **text_only**: Command palette + keyboard shortcuts (identical semantics)
- **co_partner**: Multi-user collaborative session with real-time approvals
- **silent_autonomous**: Grace acts within delegated scope

**Usage:**
```python
# Set voice mode
await orb.set_unified_voice_mode("solo_voice")

# Execute voice command with intent logging
command = await orb.execute_unified_voice_command(
    intent="optimize API latency",
    user_id="user123"
)
print(f"Command executed: {command['intent']}")
print(f"Trace ID: {command['trace_id']}")
```

## Unified Data Contracts

### Event Envelope

All events use a unified envelope structure:

```python
{
  "event_type": "grace.event.v1",
  "actor": "human|grace",
  "component": "memory|avn|governance|orb",
  "payload": { ... },
  "kpi_deltas": { "latency_ms": -420 },
  "trust_before": 0.74,
  "trust_after": 0.80,
  "confidence": 0.82,
  "immutable_hash": "sha256:...",
  "trace_id": "...",
  "timestamp": "2025-10-04T12:00:00Z",
  "correlation_id": "..."
}
```

### Memory Item

Enhanced memory items with governance and trust:

```python
{
  "id": "mem_123",
  "path": "knowledge/patterns/api_resilience/",
  "tags": ["fastapi", "timeouts"],
  "trust": 0.89,
  "last_used": "2025-10-04T12:00Z",
  "policy_refs": ["secrets_redaction"],
  "vector_ref": "vec://api_patterns/a1b2",
  "content": "...",
  "metadata": { ... }
}
```

## Interaction Lifecycle

The complete interaction lifecycle follows this flow:

1. **Intent** → User says "optimize API latency"
2. **Parsing** → Grace builds structured plan + KPIs
3. **Context linking** → Relevant Memory folders auto-attached
4. **Simulation** → AVN proposes fixes, runs sandbox proof
5. **Presentation** → Orb displays results, rationale, trust deltas
6. **Decision** → Human approves / Grace auto-promotes (per policy)
7. **Execution** → Changes applied, metrics tracked, log written
8. **Learning** → Memory + trust updated, meta-loop tick logged

## Success Metrics (Operational KPIs)

| Category | KPI | Target |
|----------|-----|--------|
| Comprehension | Mean Time to Understand (MTTU) | < 10 s |
| Healing | Mean Time to Repair (MTTR) | < 5 min |
| Governance | Approval latency | < 60 s |
| Memory | Retrieval precision (top-3) | ≥ 90 % |
| Voice | Intent accuracy | ≥ 95 % |
| Trust | Drift stability | ± 0.05 / 30 days |

## Comprehensive Stats

Get all unified stats:

```python
stats = orb.get_unified_stats()

print(f"Version: {stats['version']}")
print(f"Orchestrator ticks: {stats['orchestrator']['tick_count']}")
print(f"Memory items: {stats['memory_explorer']['total_items']}")
print(f"Anomalies detected: {stats['avn_engine']['anomalies_detected']}")
print(f"Trust metrics: {stats['trust_metrics']['total']}")
print(f"Voice mode: {stats['voice']['mode']}")
```

## Integration with Existing Orb Interface

The Unified Spec integration is **optional** and **backward compatible**. The orb interface will work with or without it:

```python
from grace.interface.orb_interface import GraceUnifiedOrbInterface

orb = GraceUnifiedOrbInterface()

# Check if unified spec is available
if orb.unified_spec:
    # Use unified spec features
    await orb.start_unified_orchestrator()
    stats = orb.get_unified_stats()
else:
    # Fallback to basic features
    print("Unified spec not available")
```

## Testing

Run the validation tests:

```bash
# Basic validation test
python demo_and_tests/tests/test_unified_spec_validation.py

# Comprehensive integration test (requires dependencies)
python demo_and_tests/tests/test_unified_spec_integration.py
```

## North-Star Principles

The integration follows these core principles:

1. **Transparency**: Every metric, log, and memory is inspectable and explainable
2. **Co-ownership**: Grace and humans share authority; either can propose, test, or approve
3. **Progressive disclosure**: Surface simplicity; reveal depth on demand
4. **Safety by default**: Every change has preview → simulation → apply → undo
5. **Learning loop**: Each interaction updates trust, confidence, and context
6. **Multimodal access**: Voice, text, UI, or API — identical semantics

## API Endpoints

When using with the Orb API (orb_api.py), the following endpoints are available:

```
GET  /api/orb/v1/unified/stats - Get unified stats
GET  /api/orb/v1/unified/health - Get orchestrator health
GET  /api/orb/v1/unified/context - Get context bus state
POST /api/orb/v1/unified/memory/folder - Create memory folder
POST /api/orb/v1/unified/memory/search - Search unified memory
POST /api/orb/v1/unified/anomaly/detect - Detect anomaly
POST /api/orb/v1/unified/anomaly/rca - Perform RCA
POST /api/orb/v1/unified/healing/execute - Execute healing action
POST /api/orb/v1/unified/voice/mode - Set voice mode
POST /api/orb/v1/unified/voice/command - Execute voice command
POST /api/orb/v1/unified/trust/update - Update trust metric
POST /api/orb/v1/unified/kpi/record - Record KPI metric
```

## Future Enhancements

Planned enhancements for future versions:

- [ ] Federation sync between Grace nodes
- [ ] Energy metrics integration
- [ ] Economic & resource awareness
- [ ] Advanced causal graph visualization
- [ ] Storybook exporter for incidents
- [ ] Multi-OS runner integration
- [ ] Backup & vault snapshots

## References

- [North-Star Architecture Specification](../GRACE_UNIFIED_OPERATING_SPECIFICATION.md)
- [Orb Interface Documentation](./ORB_INTERFACE.md)
- [Memory Explorer Guide](./MEMORY_EXPLORER.md)
- [Trust & KPI Framework](./TRUST_KPI_FRAMEWORK.md)
