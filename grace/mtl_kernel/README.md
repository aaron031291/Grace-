# MTL Kernel - Memory, Trust, Learning

## Overview

The MTL (Memory, Trust, Learning) Kernel is Grace's core infrastructure for managing memory operations, trust scoring, and audit logging. It provides a unified, high-performance API for all memory-related operations across the Grace AI system.

## Architecture

The MTL Kernel consists of 7 core components:

### 1. **Lightning Memory** (`lightning_memory.py`)
- **Purpose**: High-speed in-memory cache (Redis-like)
- **Performance**: <1ms operations
- **Features**:
  - TTL management (default 1 hour, configurable)
  - LRU eviction when max size reached
  - Batch operations (get_many, set_many)
  - Prefix-based key organization
  - JSON serialization for complex objects

### 2. **Fusion Memory** (`fusion_memory.py`)
- **Purpose**: Long-term structured storage (PostgreSQL-like)
- **Performance**: <50ms query operations
- **Schema**:
  - `learned_patterns`: ML patterns and insights
  - `interactions`: User interactions and responses
  - `governance_decisions`: Policy and decision records
  - `constitutional_precedents`: Legal/ethical precedents
  - `trust_ledger`: Trust score history
- **Features**:
  - Transaction support
  - Time-range queries
  - Full-text search capability
  - Trust score storage

### 3. **Vector Memory** (`vector_memory.py`)
- **Purpose**: Semantic search (ChromaDB-like)
- **Collections**: patterns, precedents, knowledge, interactions
- **Features**:
  - Cosine similarity search
  - Metadata filtering
  - Batch vector operations
  - Mock embedding generation (384D default)

### 4. **Trust Core** (`trust_core.py`)
- **Purpose**: Real-time trust calculation (0.0-1.0 scale)
- **Features**:
  - Entity trust registry (components, data sources, agents)
  - Performance-based trust updates
  - Trust decay algorithms (1% per day without activity)
  - Context-aware trust thresholds
  - Trust history tracking
- **Trust Factors**:
  - Historical performance: 30%
  - Constitutional compliance: 25%
  - Error frequency: 20%
  - Response time consistency: 15%
  - Governance approval rate: 10%

### 5. **Immutable Logger** (`immutable_logger.py`)
- **Purpose**: Blockchain-style audit trail
- **Features**:
  - Chain-based log structure (each entry references previous hash)
  - Cryptographic sealing (SHA-256)
  - Tamper detection via chain integrity verification
  - Event type taxonomy (9 standard types)
  - Constitutional compliance tagging
- **Event Types**:
  - MEMORY_STORED, MEMORY_RETRIEVED
  - DECISION_MADE, TRUST_UPDATED
  - CONSTITUTIONAL_CHECK, POLICY_VIOLATION
  - GOVERNANCE_ACTION, USER_INTERACTION, SYSTEM_EVENT

### 6. **Memory Orchestrator** (`memory_orchestrator.py`)
- **Purpose**: Unified API with smart routing
- **Features**:
  - Automatic tier selection (Lightning → Fusion → Vector)
  - Fallback chain for resilience
  - Performance monitoring
  - Health checks for all backends
  - Batch operations support

### 7. **MTL Service** (`mtl_service.py`)
- **Purpose**: Unified API combining Memory + Trust + Logs
- **Features**:
  - Constitutional validation hooks
  - Event mesh integration points
  - Comprehensive health checks
  - Performance metrics collection

## Performance Targets

| Component | Target | Alert Threshold |
|-----------|--------|-----------------|
| Lightning Memory | <1ms | >2ms |
| Fusion Memory | <50ms | >100ms |
| Trust Calculation | <100ms | - |
| Audit Logging | <50ms | - |

## Usage Examples

### Quick Start

```python
import asyncio
from grace.mtl_kernel import MTLService

async def main():
    # Initialize MTL Service
    mtl = MTLService()
    
    # Store data with governance
    entry_id = await mtl.store_with_governance(
        data={'key': 'user_data', 'value': 'sensitive'},
        trust_score=0.9,
        constitutional_check=True,
        component_id='user_service'
    )
    
    # Retrieve with trust validation
    data = await mtl.retrieve_with_trust(
        key='user_data',
        min_trust=0.7,
        component_id='analytics'
    )
    
    # Update trust based on performance
    new_trust = await mtl.update_entity_trust(
        entity_id='ml_model_v1',
        performance_data={
            'success': True,
            'error_count': 0,
            'response_time_ms': 45,
            'constitutional_compliant': True
        }
    )
    
    # Health check
    health = await mtl.health_check()
    print(f"System health: {health}")

asyncio.run(main())
```

### Individual Components

#### Lightning Memory (Hot Cache)
```python
from grace.mtl_kernel import LightningMemory

lightning = LightningMemory(max_size=10000, default_ttl=3600)

# Store with TTL
await lightning.set('session:abc123', {'user_id': 42}, ttl_seconds=1800)

# Retrieve
session = await lightning.get('session:abc123')

# Batch operations
await lightning.set_many({
    'config:theme': 'dark',
    'config:lang': 'en'
})
```

#### Fusion Memory (Structured Storage)
```python
from grace.mtl_kernel import FusionMemory

fusion = FusionMemory()

# Insert learned pattern
await fusion.insert(
    'learned_patterns',
    {
        'pattern_type': 'user_behavior',
        'pattern_data': {'action': 'login', 'frequency': 'daily'},
        'confidence': 0.92
    },
    trust_score=0.85
)

# Query with filters
results = await fusion.query(
    'learned_patterns',
    filters={'pattern_type': 'user_behavior'},
    limit=10
)
```

#### Vector Memory (Semantic Search)
```python
from grace.mtl_kernel import VectorMemory

vector = VectorMemory()

# Generate embedding and store
text = "Machine learning best practices"
embedding = VectorMemory.generate_mock_embedding(text)
await vector.add('knowledge', embedding, {'text': text, 'category': 'docs'})

# Semantic search
query_emb = VectorMemory.generate_mock_embedding("AI guidelines")
results = await vector.search('knowledge', query_emb, top_k=5)
```

#### Trust Core
```python
from grace.mtl_kernel import TrustCore

trust = TrustCore()

# Register entity
await trust.register_entity('ml_model_v1', initial_trust=0.7)

# Update based on performance
new_trust = await trust.update_trust(
    'ml_model_v1',
    {
        'success': True,
        'error_count': 0,
        'constitutional_compliant': True
    }
)

# Context-aware calculation
trust_score = await trust.calculate_trust(
    'ml_model_v1',
    {'sensitivity': 0.8, 'risk_level': 0.3}
)
```

#### Immutable Logger
```python
from grace.mtl_kernel import ImmutableLogger

logger = ImmutableLogger()

# Log event
audit_id = await logger.log(
    event_type='DECISION_MADE',
    component_id='governance_kernel',
    payload={'decision': 'approve', 'policy': 'data_retention'},
    trust_score=0.95,
    constitutional_compliance=1.0
)

# Verify chain integrity
is_valid = await logger.verify_chain_integrity()

# Query audit trail
logs = await logger.query_logs(
    filters={'event_type': 'DECISION_MADE'},
    limit=50
)
```

## Integration with Grace

### Event Mesh Integration

The MTL Kernel integrates with Grace's event mesh for distributed operations:

**Subscribed Events:**
- `MEMORY_STORE_REQUEST`
- `MEMORY_RETRIEVE_REQUEST`
- `TRUST_CALCULATION_REQUEST`
- `AUDIT_LOG_REQUEST`

**Emitted Events:**
- `MEMORY_STORED`
- `MEMORY_RETRIEVED`
- `TRUST_SCORE_CALCULATED`
- `AUDIT_LOGGED`
- `TRUST_VIOLATION_DETECTED`

### Constitutional Hooks

All storage operations can be validated for constitutional compliance:
- Privacy checks on data storage
- Trust violations escalated to governance
- Audit logs for all constitutional decisions

## Testing

Run the demo to see all components in action:

```bash
cd /path/to/Grace-
PYTHONPATH=. python demo_and_tests/demos/demo_mtl_quickstart.py
```

Run tests (requires pytest):

```bash
pytest demo_and_tests/tests/test_mtl_kernel.py
```

## Statistics and Monitoring

All components provide detailed statistics:

```python
stats = await mtl_service.get_stats()

# Access component stats
print(f"Lightning: {stats['memory']['lightning']['size']} entries")
print(f"Fusion: {stats['memory']['fusion']['total_entries']} entries")
print(f"Trust: {stats['trust']['avg_trust_score']} avg score")
print(f"Audit: {stats['logger']['chain_length']} logs")
```

## Production Deployment

For production use, replace mock implementations with real backends:

1. **Lightning Memory**: Replace with Redis
   - Install: `pip install redis`
   - Use: `redis.asyncio.Redis()`

2. **Fusion Memory**: Replace with PostgreSQL
   - Install: `pip install asyncpg sqlalchemy`
   - Use: `create_async_engine()`

3. **Vector Memory**: Replace with ChromaDB
   - Install: `pip install chromadb`
   - Use: `chromadb.AsyncClient()`

4. **Embedding Generation**: Use real model
   - Install: `pip install sentence-transformers`
   - Use: `SentenceTransformer('all-MiniLM-L6-v2')`

## License

Part of the Grace AI Governance System.
