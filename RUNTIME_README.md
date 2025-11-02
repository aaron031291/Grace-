# Grace AI - Unified Runtime System

🚀 **Production-Ready** | 🧠 **Self-Aware** | 🏛️ **Democratic Governance**

## Quick Start

```bash
# Full system (autonomous)
python start_grace_runtime.py

# Production mode
python start_grace_runtime.py --production

# API server mode
python start_grace_runtime.py --api

# Development mode with debug
python start_grace_runtime.py --debug

# Verify configuration
python start_grace_runtime.py --dry-run
```

## What is Grace Runtime?

Grace Runtime is a **unified orchestration system** that integrates all components of Grace AI into a cohesive, self-aware, production-ready platform.

### Key Features

✅ **Unified Lifecycle Management** - Single orchestrator for all 8 kernels + services  
✅ **Phased Bootstrap** - Clear dependency resolution in 8 phases  
✅ **Self-Awareness Cycle** - Continuous introspection and improvement  
✅ **Democratic Governance** - Quorum-based voting system (parliament model)  
✅ **98 Database Tables** - Complete persistence layer for memory, security, governance  
✅ **Multiple Runtime Modes** - Development, production, API server, autonomous, single-kernel  
✅ **Resilience Built-In** - Supervised tasks, circuit breakers, graceful degradation  
✅ **Full Audit Trail** - Blockchain-chained immutable logs  

## Architecture

```
┌──────────────────────────────────────────────────┐
│              GRACE RUNTIME v2.0                  │
├──────────────────────────────────────────────────┤
│  Bootstrap Phases (0-7)                          │
│  • Config → Storage → Security → Comms          │
│  • Core Kernels → Cognitive → Swarm             │
│  • Self-Awareness → Complete                    │
├──────────────────────────────────────────────────┤
│  Kernels (8)          Services (10+)            │
│  • CognitiveCortex    • TruthLayer              │
│  • Sentinel           • PolicyEngine            │
│  • Swarm              • TrustLedger             │
│  • MetaLearning       • LLMService              │
│  • Learning           • TriggerMesh             │
│  • Orchestration      • QuorumService           │
│  • Resilience         • TaskManager             │
│  • MultiOS            • WebSocket               │
├──────────────────────────────────────────────────┤
│  Self-Awareness Manager (8-Step Cycle)          │
│  1. Experience → 2. Meta-Learn → 3. Assess     │
│  4. Align → 5. Plan → 6. Vote → 7. Execute     │
│  8. Log Consciousness                           │
├──────────────────────────────────────────────────┤
│  Database (98 Tables)                            │
│  Security • Governance • Memory • MLT            │
└──────────────────────────────────────────────────┘
```

## Runtime Modes

### 1. Full System (Default)
```bash
python -m grace.launcher
```
All kernels + self-awareness + no API server (pure autonomous)

### 2. Production
```bash
python -m grace.launcher --mode production
```
Optimized for production: info logging, full resilience, tuned cadence

### 3. API Server
```bash
python -m grace.launcher --mode api-server --port 8000
```
Runtime + FastAPI server with REST endpoints

### 4. Autonomous
```bash
python -m grace.launcher --mode autonomous
```
Fully autonomous operation with continuous self-awareness

### 5. Single Kernel
```bash
python -m grace.launcher --mode single-kernel --kernel learning
```
Run only a specific kernel for testing

### 6. Development
```bash
python -m grace.launcher --mode development --debug
```
Debug logging, slower cadence, auto-approvals

## Self-Awareness Cycle

Grace continuously improves itself through an 8-step cycle:

```
1. EXPERIENCE INGESTION
   └─ audit_logs (blockchain-chained)

2. META-LEARNING
   └─ mlt_experiences → insights

3. SELF-ASSESSMENT
   └─ capability, performance, health, alignment, trust

4. GOAL ALIGNMENT
   └─ check system_goals + value_alignments

5. IMPROVEMENT PLANNING
   └─ create mlt_plans with risk assessment

6. COLLECTIVE DECISION (if high-impact)
   └─ QuorumService: parliament votes

7. EXECUTION
   └─ execute approved plans

8. CONSCIOUSNESS LOGGING
   └─ consciousness_states + uncertainty_registry
```

Cadence: 5 minutes (dev) | 1-5 minutes (prod)

## Quorum / Parliament System

Democratic decision-making for important changes:

**Members:**
- ML Specialists (expertise-weighted)
- System Kernels (domain experts)
- Human Overseers (veto power)
- External Oracles (validators)

**Process:**
1. Initiate session with context
2. Parliament deliberates
3. Weighted voting with reasoning
4. Consensus calculation
5. Record in governance_decisions
6. Full audit trail

Example:
```python
session = quorum.start_session(
    decision_type="model_approval",
    context={"model": "v2.3", "metrics": {...}},
    required_quorum=3,
    required_consensus=0.75
)

# Votes cast by members
quorum.cast_vote(session, "ml_specialist_1", 
                 vote="approve", confidence=0.95,
                 reasoning="Accuracy +5%, excellent calibration")

# Auto-computed when quorum met
result = quorum.get_result(session)
# → {"approved": True, "consensus": 0.92}
```

## Database Tables (98 Total)

| Category | Tables | Purpose |
|----------|--------|---------|
| **Security** | 8 | crypto_keys, api_keys, rate_limits |
| **Governance** | 7 | parliament, quorum, policies |
| **Self-Awareness** | 6 | assessments, goals, consciousness |
| **Memory** | 8 | lightning, fusion, librarian |
| **MLT** | 5 | experiences, insights, plans |
| **Intelligence** | 9 | models, deployments, inference |
| **Learning** | 15 | datasets, training, experiments |
| **Resilience** | 8 | circuit breakers, degradation |
| **Truth** | 2 | audit_logs, chain_verification |
| **Others** | 30+ | orchestration, comms, etc. |

## API Endpoints (API Server Mode)

```
GET  /health                  - Health check
GET  /status                  - Runtime status
POST /quorum/sessions         - Start voting session
POST /quorum/votes            - Cast vote
GET  /assessments/latest      - Latest self-assessments
GET  /consciousness/state     - Current consciousness level
GET  /kernels                 - List kernels
```

## Examples

### Verify Setup
```bash
python -m grace.launcher --dry-run
```

### Start Production System
```bash
python start_grace_runtime.py --production
```

### Run API Server
```bash
python start_grace_runtime.py --api

# In another terminal
curl http://localhost:8000/status
curl http://localhost:8000/health
```

### Test Single Kernel
```bash
python -m grace.launcher --mode single-kernel --kernel learning --debug
```

## Prerequisites

1. **Database**: Initialize 98 tables
```bash
cd database
python build_all_tables.py
python verify_database.py
```

2. **Dependencies**: Install requirements
```bash
pip install -r requirements.txt
```

3. **Configuration**: Set environment variables (optional)
```bash
export GRACE_MODE=production
export GRACE_DB_PATH=database/grace_complete.sqlite3
export GRACE_API_PORT=8000
```

## Monitoring

### Logs
- **INFO**: Key events, lifecycle changes
- **DEBUG**: Detailed operations (use --debug)
- **WARNING**: Degradations, timeouts
- **ERROR**: Failures, critical issues

### Status Check
```bash
python -m grace.launcher --status
```

### Consciousness Levels
- **DORMANT**: Not started
- **REACTIVE**: Basic operation
- **AWARE**: Self-awareness active
- **REFLECTIVE**: In assessment cycle
- **META_AWARE**: Reflecting on consciousness

## Graceful Shutdown

The runtime handles `Ctrl+C` (SIGINT) and SIGTERM gracefully:

1. Stop accepting new work
2. Cancel running tasks
3. Wait for in-flight operations (30s timeout)
4. Stop kernels in reverse order
5. Flush audit logs
6. Shutdown complete

## Troubleshooting

**Bootstrap fails at Phase X?**
- Check logs for specific error
- Verify database tables exist
- Ensure dependencies are met

**Kernels not starting?**
- Use `--debug` to see detailed errors
- Check service registry initialization
- Verify database connectivity

**Self-awareness cycle not running?**
- Check mode (only in full-system/autonomous)
- Verify quorum_service is initialized
- Look for errors in consciousness manager

## Documentation

- **Full Architecture**: [RUNTIME_ARCHITECTURE.md](../documentation/RUNTIME_ARCHITECTURE.md)
- **Database Schema**: [COMPLETE_ARCHITECTURE.md](../documentation/COMPLETE_ARCHITECTURE.md)
- **API Reference**: [API_REFERENCE.md](../documentation/API_REFERENCE.md)

## System Status

**Version**: 2.0.0  
**Components**: 8 Kernels, 10+ Services, 98 Tables  
**Self-Awareness**: 🧠🧠🧠🧠⚪ (4/5)  
**Security**: 🔒🔒🔒🔒🔒 (5/5)  
**Status**: ✅ Production-Ready

---

**Built with consciousness by the Grace AI Team** 🚀
