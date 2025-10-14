# Grace Meta-Control Protocol (MCP) System

## Overview

The **Meta-Control Protocol (MCP)** is Grace's canonical API façade layer that provides:

- **Unified API contracts** for all domain tables and operations
- **Governance integration** at every access point
- **Automatic vectorization** and semantic search capabilities
- **Event emission** for distributed system coordination
- **Immutable audit logging** for full traceability
- **Trust scoring** and provenance tracking
- **Meta-Loop integration** for observation and learning

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    External Clients                         │
│              (Kernels, Agents, Human APIs)                  │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│                   MCP Gateway Layer                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │
│  │  Auth &  │ Gov      │  Rate    │ Contract │ Metrics  │ │
│  │  AuthZ   │ Validate │ Limiting │ Validate │ Collect  │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘ │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│              Domain MCP Handlers (Per-Table)                │
│  ┌─────────────┬─────────────┬─────────────┬────────────┐ │
│  │ Patterns    │ Experiences │ Decisions   │ Knowledge  │ │
│  │ MCP         │ MCP         │ MCP         │ MCP        │ │
│  └─────────────┴─────────────┴─────────────┴────────────┘ │
└──────────────────────┬─────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐
│   Database   │ │  Vector  │ │   Meta-    │
│   (Fusion)   │ │  Store   │ │   Loops    │
│              │ │(Librarian)│ │  (OODA)    │
└──────────────┘ └──────────┘ └────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
           ┌──────────────────────┐
           │   Event Bus          │
           │   (TriggerMesh)      │
           └──────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐
│  Immutable   │ │   AVN    │ │ Governance │
│    Logs      │ │ (Immune) │ │  Kernel    │
└──────────────┘ └──────────┘ └────────────┘
```

## Key Responsibilities

1. **API Contract Enforcement**: All CRUD and semantic operations follow versioned contracts
2. **Authentication & Authorization**: Token validation, role checks, identity verification
3. **Constitutional Validation**: Governance kernel integration before sensitive operations
4. **Vectorization Management**: Embed, index, and semantic search coordination
5. **Event Emission**: Publish domain events to TriggerMesh/EventBus
6. **Immutable Audit Logging**: Full traceability of all operations
7. **Trust & Provenance**: Attach trust scores and provenance metadata
8. **Meta-Loop Integration**: Feed observations to O-Loop, record decisions to D-Loop
9. **Rate Limiting & Cost Control**: Throttle expensive embedding operations
10. **Pushback Handling**: Consistent error responses with forensic escalation

## MCP vs Direct Access

| Aspect | Direct DB Access | Through MCP |
|--------|------------------|-------------|
| **Governance** | ❌ Not enforced | ✅ Always validated |
| **Audit Trail** | ❌ Manual | ✅ Automatic |
| **Vectorization** | ❌ Caller implements | ✅ MCP handles |
| **Events** | ❌ Caller emits | ✅ MCP emits |
| **Trust Scoring** | ❌ Not tracked | ✅ Automatic |
| **Contract Versioning** | ❌ Schema drift | ✅ Versioned API |
| **Meta-Loop Integration** | ❌ Not connected | ✅ Feeds OODA |

## Manifest-Driven Architecture

Each domain (table/collection) has a YAML manifest that defines:

- **Ownership**: System-owned vs Agent-owned (MTL)
- **Endpoints**: CRUD, semantic search, metadata, domain actions
- **Authentication**: Required roles and permissions
- **Governance**: Sensitivity levels and validation requirements
- **Vectorization**: Whether to embed, which collection, trust filters
- **Events**: Which events to emit on operations
- **Observability**: Metrics, logs, traces

## Hybrid Model: System vs Agent Ownership

### System-Owned Tables
MCP executes full pipeline (auth → gov → persist → vectorize → emit → audit)

**Examples**: `observations`, `decisions`, `actions`, `trust_scores`

### Agent-Owned Tables
MCP routes to agent's MTL endpoint; agent implements semantic logic

**Examples**: `patterns` (Pattern MTL), `intel_results` (Intelligence MTL)

## Meta-Loop Integration

Every MCP operation feeds the Meta-Loop system:

```python
# On CREATE
1. MCP validates and persists
2. O-Loop: INSERT INTO observations (operation='mcp.create', data=...)
3. Event emitted: {domain}.created
4. Immutable log: append with provenance

# On SEMANTIC_SEARCH
1. MCP embeds query
2. O-Loop: INSERT INTO observations (operation='mcp.semantic_search', query=...)
3. Vector search + trust filtering
4. E-Loop: INSERT INTO evaluations (results_quality, latency, trust_scores)
5. Event emitted: {domain}.searched
6. F-Loop learns from search patterns

# On GOVERNANCE_REJECTION
1. MCP calls governance kernel
2. D-Loop: INSERT INTO decisions (rejected=true, reason=...)
3. T-Loop: INSERT INTO trust_measurements (violation_type=...)
4. Event emitted: GOVERNANCE.REJECTED
5. Forensic escalation if critical
```

## Quick Start

### 1. Define Domain Manifest

```yaml
# grace/mcp/manifests/patterns.yaml
mcp_version: "1.0"
domain: "patterns"
table: "observations"  # or patterns if separate table
owner: "system"  # or "pattern_mtl"

auth:
  required: true
  roles_allowed: ["kernel", "agent", "admin"]

endpoints:
  - name: create
    method: POST
    path: /api/v1/patterns
    body_schema: PatternCreateRequest
    side_effects:
      - vectorize: true
      - emit_event: PATTERN.CREATED
      - log_immutable: true
      - observe_meta_loop: true
    governance:
      sensitivity: low
      require_validation: false

  - name: semantic_search
    method: POST
    path: /api/v1/patterns/search
    body_schema: SemanticSearchRequest
    semantics:
      vector_collection: patterns_vectors
      trust_filter: true
      default_top_k: 10
    governance:
      sensitivity: medium
      require_validation_threshold: 0.8
```

### 2. Implement MCP Handler

```python
from grace.mcp import BaseMCP, mcp_endpoint

class PatternsMCP(BaseMCP):
    domain = "patterns"
    
    @mcp_endpoint(manifest="patterns.yaml", endpoint="create")
    async def create_pattern(self, request: PatternCreateRequest, caller: Caller):
        # MCP framework handles: auth, governance, observation
        # You implement: business logic
        
        record = await self.db.insert('observations', request.dict())
        embedding = await self.vectorize(request.raw_text)
        await self.vector_store.upsert(self.collection, record.id, embedding)
        
        # Framework auto-emits event and logs to immutable store
        return PatternCreateResponse(id=record.id, trust_score=0.5)
```

### 3. Register with Gateway

```python
# grace/mcp/gateway.py
from grace.mcp.handlers import PatternsMCP, ExperiencesMCP, DecisionsMCP

app = FastAPI()
gateway = MCPGateway()

gateway.register(PatternsMCP())
gateway.register(ExperiencesMCP())
gateway.register(DecisionsMCP())

app.include_router(gateway.router, prefix="/api/v1")
```

## Directory Structure

```
grace/mcp/
├── README.md                    # This file
├── __init__.py
├── gateway.py                   # Main gateway with middleware
├── base_mcp.py                  # Base MCP class with decorators
├── pushback.py                  # Unified error handling (upgraded)
├── manifests/                   # YAML contracts
│   ├── patterns.yaml
│   ├── experiences.yaml
│   ├── decisions.yaml
│   ├── observations.yaml
│   └── knowledge.yaml
├── handlers/                    # Per-domain MCP implementations
│   ├── __init__.py
│   ├── patterns_mcp.py
│   ├── experiences_mcp.py
│   ├── decisions_mcp.py
│   ├── observations_mcp.py
│   └── knowledge_mcp.py
├── middleware/                  # Cross-cutting concerns
│   ├── __init__.py
│   ├── auth.py
│   ├── governance.py
│   ├── rate_limiter.py
│   ├── metrics.py
│   └── meta_loop_observer.py
├── schemas/                     # Pydantic models
│   ├── __init__.py
│   ├── common.py
│   ├── patterns.py
│   ├── experiences.py
│   └── decisions.py
└── tests/
    ├── test_gateway.py
    ├── test_patterns_mcp.py
    ├── test_governance_integration.py
    └── test_meta_loop_integration.py
```

## Benefits

### For Developers
- **Single source of truth** for API contracts
- **Automatic governance** integration
- **Built-in observability** (metrics, logs, traces)
- **Consistent error handling** with forensic escalation
- **Testable contracts** with generated mocks

### For Operations
- **Centralized security** (auth, rate limiting, encryption)
- **Unified audit trail** for compliance
- **Service mesh integration** ready
- **Auto-scaling** based on cost metrics
- **Incident response** via AVN integration

### For Governance
- **Policy enforcement** at API boundary
- **Trust score** tracking per operation
- **Constitutional checks** before sensitive ops
- **Democratic oversight** via quorum integration
- **Immutable evidence** for arbitration

### For Learning (Meta-Loops)
- **Every operation observed** (O-Loop)
- **Decision rationale captured** (D-Loop)
- **Outcomes evaluated** (E-Loop)
- **Patterns detected** (F-Loop)
- **System evolves** (V-Loop)

## Next Steps

1. ✅ Read this README
2. 📝 Create manifests for your domains (see `manifests/`)
3. 🔨 Implement MCP handlers (see `handlers/`)
4. 🧪 Write integration tests (see `tests/`)
5. 🚀 Deploy gateway with TLS and auth
6. 📊 Connect to monitoring and alerting
7. 🔄 Iterate based on Meta-Loop learning

## See Also

- [Database Schema Documentation](../DATABASE_SCHEMA.md)
- [Grace Cognitive Architecture](../GRACE_COGNITIVE_ARCHITECTURE.md)
- [Meta-Loop Integration Guide](../docs/meta_loop_integration.md)
- [Governance Kernel API](../governance/README.md)
- [Vector Store Integration](../mlt_kernel/README.md)
