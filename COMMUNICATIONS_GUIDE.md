# Grace Communications Schema Pack - Implementation Guide

## Overview

The Grace Communications Schema pack provides a unified, transport-agnostic messaging standard for all Grace kernels. It implements the Grace Message Envelope (GME) format with comprehensive governance, observability, and reliability features.

## 🏗️ Architecture

### Core Components

1. **Grace Message Envelope (GME)** - Standardized message wrapper
2. **Schema Registry** - Version management and evolution
3. **Topic & Routing** - Priority lanes and partition management  
4. **Quality of Service** - Retry policies and backpressure handling
5. **Security & Governance** - RBAC, encryption, and audit trails
6. **Observability** - OpenTelemetry integration and metrics
7. **Transport Abstraction** - Kafka, NATS, AMQP, HTTP, WebSocket support

## 📁 Directory Structure

```
contracts/comms/
├── envelope.schema.json      # GME core schema
├── rpc.schema.json          # Command/Query/Reply contracts  
├── errors.schema.json       # Error taxonomy
├── topics.yaml              # Routing and lanes
├── events.master.yaml       # Event catalog
├── registry.yaml            # Schema evolution rules
├── efficiency.yaml          # Batching/compression
├── qos.yaml                 # Quality of service
├── security.yaml            # Auth/RBAC/encryption
├── observability.yaml       # Tracing/metrics/logs
├── transports.yaml          # Transport mappings
├── bindings.yaml            # Kernel communication patterns
├── experience.schema.json   # MLT optimization data
├── snapshot.schema.json     # Rollback capability
└── defaults.yaml            # Drop-in configuration

grace/comms/
├── __init__.py             # Module exports
├── envelope.py             # GME implementation
└── validator.py            # Schema validation
```

## 🚀 Quick Start

### 1. Create a GME Message

```python
from grace.comms import create_envelope, MessageKind, Priority, QoSClass

# Create an event
envelope = create_envelope(
    kind=MessageKind.EVENT,
    domain="intelligence", 
    name="INTEL_INFER_COMPLETED",
    payload={
        "request_id": "req_123",
        "result": {"confidence": 0.95}
    },
    priority=Priority.P0,
    qos=QoSClass.REALTIME,
    rbac=["intel.read"]
)
```

### 2. Validate Messages

```python
from grace.comms import validate_envelope

result = validate_envelope(envelope.model_dump())
if result.passed:
    print("✅ Valid GME message")
else:
    print(f"❌ Validation errors: {result.errors}")
```

### 3. Update Mesh Bridges

The Multi-OS mesh bridge has been updated to use GME format automatically:

```python
# In mesh bridges - automatically uses GME if available
await bridge.publish_event(
    "MOS_TASK_COMPLETED",
    {"task_id": "task_123", "status": "success"}
)
```

## 🎯 Key Features Implemented

### ✅ One Envelope, Many Payloads
- Consistent headers across events, commands, queries, and replies
- Transport-agnostic message format
- Payload/payload_ref distinction for large content

### ✅ Zero-Loss Semantics  
- Idempotency keys prevent duplicate processing
- Dead Letter Queue (DLQ) for failed messages
- Exponential/linear retry with jitter
- At-least-once delivery with deduplication

### ✅ Ordering Where It Matters
- Per partition_key message ordering
- Causal chains via causation_id
- Correlation tracking across workflows

### ✅ Governance-Grade Security
- RBAC permissions in message headers
- PII flags and consent scope tracking
- Message signatures for high-risk content
- Governance labels (public/internal/restricted)

### ✅ Full Observability
- W3C trace context propagation (traceparent/tracestate)
- Hop count tracking to detect loops
- Timing headers and performance metrics
- OpenTelemetry ready spans and attributes

### ✅ Adaptive Quality of Service
- Priority lanes (P0-P3) with different SLAs
- QoS classes (realtime/standard/bulk)
- Backpressure signals and circuit breakers
- MLT feedback for auto-optimization

## 🔄 Event Flow Examples

### Intelligence Request/Response
```
1. UI → INTEL_REQUESTED (P0, realtime)
2. Intelligence → INTEL_INFER_COMPLETED (P0, realtime) 
3. MLT ← MLT_EXPERIENCE (P3, bulk)
```

### MLDL Deployment Workflow  
```
1. MLDL → MLDL_DEPLOYMENT_REQUESTED (P0, standard)
2. Governance → GOVERNANCE_APPROVED (P0, standard)
3. MLDL → MLDL_DEPLOYMENT_STARTED (P1, standard)
4. Multi-OS ← ORCH_TASK_DISPATCHED (P1, standard)
```

## 🧪 Testing

Run the test suite:

```bash
# Test GME implementation
python test_comms_schema.py

# Test system integration  
python tests/smoke_tests.py

# See GME workflows
python demo_gme_workflows.py
```

Expected output: All tests pass ✅

## 📊 Schema Evolution

The system supports backward-compatible schema evolution:

- **Additive changes** (new optional fields) → Minor version bump
- **Breaking changes** (required fields, removed fields) → Major version bump
- **Enum extensions** → Minor version with warnings

Schema compatibility matrix in `registry.yaml`:
- `default: backward` - Old consumers work with new schemas
- `intelligence.result: full` - Strict compatibility required
- `mlt.experience: none` - Breaking changes allowed

## 🔧 Transport Configuration

### Kafka Integration
- Topic prefix: `grace.*`
- Idempotent producers with transactions
- Header mapping for GME fields
- Partitioning by `partition_key`

### NATS Integration  
- JetStream for durability
- Queue groups for load balancing
- Duplicate detection window

### HTTP/WebSocket
- REST endpoints at `/api/comms/v1`
- WebSocket hubs for real-time streams
- GME envelope in message body

## 🛡️ Security Features

### Authentication
- mTLS for transport security
- JWT tokens with required claims
- Signature verification for critical messages

### Authorization
- RBAC permissions in message headers
- Deny-by-default enforcement
- Role-based topic access

### Privacy
- PII flag detection and handling
- Consent scope validation
- Automatic redaction in logs

## 📈 MLT Integration

The system feeds communication experiences to MLT for optimization:

- **Batching tuning** - Optimal batch sizes per topic
- **Compression selection** - Best codec per content type  
- **Retry curves** - Adaptive backoff strategies
- **Lane rebalancing** - Dynamic partition allocation
- **Transport selection** - Route based on performance

## 🔄 Rollback Support

Complete snapshot/rollback capability:

1. **Snapshot creation** - Capture routing, QoS, and registry state
2. **Graceful degradation** - Freeze changes, drain P0 traffic
3. **State restoration** - Reload configuration from snapshot
4. **Cache rehydration** - Restore idempotency windows
5. **Resume operations** - Emit ROLLBACK_COMPLETED event

## 🎛️ Configuration

### Environment Overrides
- **Development**: Relaxed validation, full tracing  
- **Testing**: No DLQ, higher timeouts
- **Production**: Full security, minimal tracing

### Drop-in Defaults
```yaml
# contracts/comms/defaults.yaml
defaults:
  priority: "P2"
  qos: "standard" 
  compression: "zstd"
  retries:
    max_attempts: 5
    exp_base_ms: 50
```

## 🏆 Benefits Delivered

1. **Unified Language** - All kernels speak the same GME format
2. **Zero Vendor Lock-in** - Transport-agnostic design
3. **Production Ready** - Enterprise security and reliability  
4. **Self-Optimizing** - MLT feedback loops improve performance
5. **Governance Compliant** - Built-in audit and approval workflows
6. **Developer Friendly** - Rich type safety and validation
7. **Backward Compatible** - Works alongside existing event bus

The Grace Communications Schema pack is now ready for production use across all Grace kernels! 🎉