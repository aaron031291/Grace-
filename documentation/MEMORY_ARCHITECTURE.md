# Grace MemoryCore Architecture

## Overview

MemoryCore is the central memory coordinator that orchestrates writes across multiple storage layers with automatic fan-out to trust attestations, audit logs, and event triggers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         MemoryCore                           │
│                    (Unified Coordinator)                     │
└────────────┬───────────────────────────────────────┬────────┘
             │                                        │
             ▼                                        ▼
    ┌────────────────┐                      ┌────────────────┐
    │  Write Fan-Out │                      │  Read Hierarchy│
    └────────┬───────┘                      └────────┬───────┘
             │                                        │
    ┌────────┴────────────────────────────┐          │
    │                                     │          │
    ▼                                     ▼          ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  
│  Lightning  │  │   Fusion    │  │   Vector    │  ← Storage Layers
│   (Cache)   │  │  (Durable)  │  │ (Semantic)  │  
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  
       │                │                │
       └────────┬───────┴────────┬───────┘
                │                │
                ▼                ▼
       ┌────────────────┐  ┌────────────────┐
       │ Trust          │  │ Immutable      │  ← Attestation Layer
       │ Attestations   │  │ Logs           │
       └────────────────┘  └────────────────┘
                │
                ▼
       ┌────────────────┐
       │ Trigger        │  ← Event Layer
       │ Ledgers        │
       └────────────────┘
```

## Write Fan-Out Flow

When `MemoryCore.write(key, value)` is called:

### 1. Lightning (Cache Layer)
- **Purpose**: Fast in-memory access
- **Backend**: Redis (async)
- **TTL**: Configurable (default 3600s)
- **Failure**: Non-critical, continues to other layers

```python
await lightning.set(key, value, ttl=ttl_seconds)
```

### 2. Fusion (Durable Store)
- **Purpose**: Persistent storage
- **Backend**: PostgreSQL (async)
- **Tables**: 
  - `learned_patterns` - for pattern data
  - `interactions` - for general writes
- **Failure**: Critical - write fails if Fusion fails

```python
if metadata.get("is_pattern"):
    await fusion.store_pattern(pattern_type, pattern_data, confidence)
else:
    await fusion.record_interaction(action="memory_write", context={...})
```

### 3. Vector (Semantic Search)
- **Purpose**: Embedding-based search
- **Backend**: FAISS / pgvector
- **Data**: Text/string values are embedded
- **Failure**: Non-critical

```python
if isinstance(value, str):
    # Create embedding and index
    vector.index(key, embedding)
```

### 4. Trust Attestations
- **Purpose**: Update actor trust scores
- **Backend**: TrustCoreKernel
- **Updates**: Based on write success/failure
- **Failure**: Non-critical

```python
outcome = {"success": True, "error_rate": 0.0, ...}
await trust.update_trust(entity_id=actor, outcome=outcome)
```

### 5. Immutable Logs
- **Purpose**: Audit trail
- **Backend**: PostgreSQL append-only table
- **Chain**: Cryptographically chained hashes
- **Failure**: Critical - write fails if logging fails

```python
await logs.log(
    operation_type="memory_write",
    actor=actor,
    action={"key": key, ...},
    result={...}
)
```

### 6. Trigger Ledgers
- **Purpose**: Event-driven downstream processing
- **Backend**: EventBus
- **Event Type**: `memory.write`
- **Failure**: Non-critical

```python
event = factory.create_event(
    event_type="memory.write",
    payload={"key": key, "results": {...}}
)
await event_bus.emit(event)
```

## Read Hierarchy

Reads follow a cache hierarchy:

1. **Lightning** (fast cache) - O(1) lookup
2. **Fusion** (durable store) - O(log n) database query
3. **Vector** (semantic) - O(n) similarity search

```python
value = await memory_core.read(key, use_cache=True)

# Flow:
# 1. Check Lightning → CACHE HIT? Return
# 2. Check Fusion → FOUND? Populate Lightning, Return
# 3. Check Vector → FOUND? Populate Lightning, Return
# 4. Return None
```

## Success Criteria

A write is considered successful if:
- ✅ Fusion (durable store) write succeeds
- ✅ Immutable log write succeeds

Other layers are best-effort:
- ⚠️ Lightning failure → continues (degraded performance)
- ⚠️ Vector failure → continues (no semantic search)
- ⚠️ Trust failure → continues (no trust update)
- ⚠️ Trigger failure → continues (no downstream events)

## Statistics

MemoryCore tracks:
- `writes_total` - Total write attempts
- `writes_failed` - Failed writes
- `write_success_rate` - Success percentage
- `cache_hits` - Lightning cache hits
- `cache_misses` - Lightning cache misses
- `cache_hit_rate` - Cache effectiveness

```python
stats = memory_core.get_stats()
```

## Configuration

```bash
# Enable/disable layers
MEMORY_LIGHTNING_ENABLED=true
MEMORY_FUSION_ENABLED=true
MEMORY_VECTOR_ENABLED=false
MEMORY_FANOUT_ENABLED=true

# Redis (Lightning)
REDIS_URL=redis://localhost:6379/0

# PostgreSQL (Fusion, Logs)
DATABASE_URL=postgresql://user:pass@localhost/grace_db
```

## API Usage

### Write with Full Fan-Out

```python
success = await memory_core.write(
    key="user_preference",
    value={"theme": "dark", "language": "en"},
    metadata={"user_id": "123"},
    ttl_seconds=3600,
    actor="user_123",
    trust_attestation=True
)
```

### Read with Cache Hierarchy

```python
value = await memory_core.read(
    key="user_preference",
    actor="user_123",
    use_cache=True
)
```

### Store Learned Pattern

```python
await memory_core.write(
    key="behavior_pattern_1",
    value={"action": "click", "frequency": 10},
    metadata={
        "is_pattern": True,
        "pattern_type": "user_behavior",
        "confidence": 0.85
    },
    actor="ml_engine"
)
```

## Integration with Kernels

Kernels can write to memory and trigger downstream processing:

```python
class MyKernel:
    def __init__(self, memory_core):
        self.memory = memory_core
    
    async def process_event(self, event):
        # Write to memory triggers fan-out
        await self.memory.write(
            key=f"event_{event.event_id}",
            value=event.payload,
            actor=event.source
        )
        
        # Fan-out automatically:
        # - Caches in Lightning
        # - Persists to Fusion
        # - Updates Trust for event.source
        # - Logs to immutable audit
        # - Emits memory.write trigger event
```

## Testing

See `tests/test_memory_integration.py` for comprehensive integration tests:

```bash
pytest tests/test_memory_integration.py -v
```

Tests verify:
- ✅ Write fan-out to all layers
- ✅ Read cache hierarchy
- ✅ Statistics tracking
- ✅ Pattern storage
- ✅ Trust integration
- ✅ Delete operations
