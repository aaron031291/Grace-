# Grace Governance Kernel

The **Grace Governance Kernel** is an advanced AI governance system that implements constitutional decision-making, multi-specialist consensus, and democratic oversight for AI systems. It provides tamper-proof audit trails, real-time health monitoring, and sophisticated trust management.

## Architecture Overview

Grace implements a comprehensive governance architecture with the following components:

### Core Governance Layer (`grace/governance/`)
- **Verification Engine** - Truth validation and claim analysis with constitutional reasoning
- **Unified Logic** - Cross-layer synthesis and arbitration of specialist inputs
- **Governance Engine** - Main orchestrator with policy enforcement and event management
- **Parliament** - Democratic review system for major decisions
- **Trust Core Kernel** - Trust and credibility weighting for sources and components

### Event Infrastructure (`grace/layer_02_event_mesh/`)
- **Trigger Mesh** - Sub-millisecond event routing with priority queues and constitutional validators

### Audit System (`grace/layer_04_audit_logs/` → `grace/audit/`)
- **Immutable Logs** - Blockchain-like tamper-proof audit trail with transparency controls
- **Golden Path Auditor** - Concrete append/verify implementation for all memory operations
- **Accessible via**: `grace/audit/immutable_logs.py` (symlinked for convenience)

### Health Monitoring (`grace/immune/`)
- **Enhanced AVN Core** - Anomaly detection, predictive alerts, and automated healing

### ML/DL Consensus (`grace/mldl/`)
- **21-Specialist Quorum** - Expert consensus system with weighted voting
- **Governance Liaison** - Ensures ML/DL model compliance with constitutional principles

### Core Infrastructure (`core/`)
- **Event Bus** - Central event routing and correlation tracking
- **Memory Core** - Persistent storage with precedent-based reasoning
- **Contracts** - Shared data structures and type definitions

## Key Features

### 🏛️ Constitutional Governance
- Enforces transparency, fairness, accountability, consistency, and harm prevention
- Democratic parliamentary review for high-impact decisions
- Precedent-based case reasoning

### 🤖 21-Specialist AI Consensus
- Tabular classification/regression, NLP, computer vision, time series
- Reinforcement learning, Bayesian modeling, causal inference
- Fairness auditing, privacy/security scanning, AutoML planning
- Meta-ensembling with governance liaison oversight

### 🔒 Immutable Audit Trail
- Blockchain-inspired chain verification
- Configurable transparency levels (public → security-sensitive)
- Constitutional compliance logging for all decisions

### 📊 Real-time Health Monitoring
- Anomaly detection with predictive failure alerts
- Component performance tracking and trust scoring
- Automated healing and failover capabilities

### 🔄 Blue/Green Governance
- Shadow mode testing of governance instances
- Snapshot/rollback capabilities with state verification
- Hot-swap governance with delta comparison

## Quick Start

### Automated Setup (Recommended)
```bash
git clone https://github.com/aaron031291/Grace-.git
cd Grace-
./scripts/setup.sh
```

The setup script will automatically:
- Install Python and Git if needed (Linux/macOS)
- Create a virtual environment
- Install all dependencies
- Configure Git settings
- Set up development tools

### Manual Installation
```bash
git clone https://github.com/aaron031291/Grace-.git
cd Grace-
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env  # Edit with your configuration
```

### Basic Usage
```python
import asyncio
from grace.governance.grace_governance_kernel import GraceGovernanceKernel

async def main():
    # Initialize and start governance kernel
    kernel = GraceGovernanceKernel()
    await kernel.start()
    
    # Process a governance request
    result = await kernel.process_governance_request(
        "policy",  # decision type
        {
            "claims": [...],     # Claims to validate
            "context": {...}     # Decision context
        }
    )
    
    print(f"Decision: {result['outcome']}")
    await kernel.shutdown()

asyncio.run(main())
```

### Governance Enforcement Hooks

Grace provides runtime enforcement through several key components:

#### Constitutional Decorator
```python
from grace.governance.constitutional_decorator import constitutional_check

@constitutional_check(policy="strict", transparency_level="public")
async def sensitive_operation(data, rationale="Automated operation"):
    # Your sensitive operation here
    return {"result": "processed", "governance": {...}}
```

#### Trust Middleware
```python
from grace.governance.constitutional_decorator import trust_middleware

@trust_middleware(min_trust_score=0.8)
async def high_trust_operation(data):
    # Operation requiring high trust score
    return data
```

#### Golden Path Auditing
```python
from grace.audit.golden_path_auditor import append_audit

# In your API endpoints or operations:
audit_id = await append_audit(
    operation_type="memory_write",
    operation_data={"action": "data_ingestion"},
    user_id="user123",
    transparency_level="democratic_oversight"
)
```

#### Policy Enforcement Middleware
Applied automatically to all API routes in `grace.api.api_service`:
- Validates operations against policy rules
- Blocks unauthorized actions
- Logs policy violations
- Enforces constitutional compliance

## Development Workflow

Grace provides automated Git workflow tools to streamline development across environments:

### Git Workflow Helper
```bash
# Linux/macOS
./scripts/git-workflow.sh <command>

# Windows PowerShell
.\scripts\git-workflow.ps1 <command>
```

Common commands:
- `setup` - Configure Git settings
- `new-branch feature/my-feature` - Create new feature branch
- `workflow fix governance "fix validation bug"` - Complete workflow (test, commit, push)
- `sync` - Sync with main branch
- `status` - Show repository status
- `test` - Run tests

### For Detailed Setup Instructions
See [DEVELOPMENT_SETUP.md](./DEVELOPMENT_SETUP.md) for comprehensive development environment setup across all platforms.


## Architecture Diagrams

### Governance Flow
```
MLDL Specialists (21) → Unified Logic → Verification Engine → Governance Engine
                                     ↓
Parliament ← Trigger Mesh ← Constitutional Validators ← Immutable Logs
     ↓                                                        ↑
Trust Core ← AVN Health Monitor ← Memory Core ← Event Bus ←─┘
```

### Event Routing
```
External Request → Trigger Mesh → Priority Queues → Constitutional Validators
                                      ↓
                     Component Routing → Shadow Mirroring → Audit Logging
```

## Configuration

Governance behavior is configurable via `config/governance_config.py`:

```python
GOVERNANCE_THRESHOLDS = {
    "min_confidence": 0.78,
    "min_trust": 0.72,
    "constitutional_compliance_min": 0.85
}
```

## Components Detail

### Verification Engine
- Multi-source claim validation
- Constitutional reasoning and compliance checking  
- Logical chain analysis and contradiction detection
- Confidence scoring with evidence quality assessment

### Unified Logic  
- Weighted synthesis of 21 specialist outputs
- Cross-layer conflict arbitration
- Domain-specific expertise weighting
- Threshold-based recommendation generation

### Parliament
- Democratic review with configurable voting thresholds
- Expertise-based reviewer assignment
- Deadline-driven decision processes
- Vote tracking and audit trails

### Trust Core
- Dynamic trust scoring with performance feedback
- Source credibility assessment
- Domain expertise weighting
- Trust decay and maintenance

### MLDL Quorum
- 21 specialized AI models covering ML/DL landscape
- Weighted consensus with confidence thresholds
- Performance-based specialist weight adjustment
- Governance liaison for constitutional compliance

## Event Types

Grace uses a comprehensive event system:

- `GOVERNANCE_VALIDATION` - Request governance decision
- `GOVERNANCE_APPROVED/REJECTED` - Decision outcomes  
- `GOVERNANCE_NEEDS_REVIEW` - Escalation to parliament
- `GOVERNANCE_SNAPSHOT_CREATED` - State snapshots
- `GOVERNANCE_ROLLBACK` - Rollback operations
- `ANOMALY_DETECTED` - Health alerts
- `TRUST_UPDATED` - Trust score changes
- `MLDL_CONSENSUS_REACHED` - Specialist consensus

## Database Schema

Grace maintains several SQLite databases:

- **grace_governance.db** - Decisions, snapshots, experiences, precedents
- **governance_audit.db** - Immutable audit trail with chain verification

## Transparency Levels

Audit logs support configurable transparency:

1. **Public** - Fully accessible (7-year retention)
2. **Democratic Oversight** - Parliament/oversight access (5-year retention)  
3. **Governance Internal** - Internal operations (1-year retention)
4. **Audit Only** - Audit purposes (7-year retention)
5. **Security Sensitive** - Restricted access (90-day retention)

## Architecture Evolution

> **⚠️ Deprecation Notice**: The "layer_*" model references in older documentation are deprecated. 
> The current Grace architecture uses an **11-kernel structure** as the single source of truth:
> 
> - **Governance Kernel** - Constitutional decision-making
> - **Intelligence Kernel** - AI reasoning and validation  
> - **Learning Kernel** - Adaptive learning and improvement
> - **Memory Kernel** - Persistent storage and retrieval
> - **Trust Kernel** - Credibility and reliability management
> - **Interface Kernel** - User and system interfaces
> - **Orchestration Kernel** - Workflow coordination
> - **Resilience Kernel** - Health monitoring and recovery
> - **Ingress Kernel** - Data input and validation
> - **Multi-OS Kernel** - Cross-platform compatibility
> - **MLdL Kernel** - Machine learning specialists
>
> For legacy compatibility, some layer references remain but point to the kernel implementations.

## Documentation Links

- [Development Setup](DEVELOPMENT_SETUP.md) - Getting started with Grace development
- [Communications Guide](COMMUNICATIONS_GUIDE.md) - Event system and messaging
- [Implementation Details](IMPLEMENTATION.md) - Technical implementation notes
- [Security Improvements](SECURITY_IMPROVEMENTS.md) - Security architecture
- [Enhanced Features](ENHANCED_FEATURES_DOCUMENTATION.md) - Advanced capabilities
- [System Overview](GRACE_COMPLETE_SYSTEM_OVERVIEW.md) - Comprehensive system documentation

## Meta-Learning

Grace continuously improves through experience collection:

- **Verification Results** - Claim validation performance
- **Consensus Quality** - Specialist agreement effectiveness  
- **Constitutional Compliance** - Principle adherence tracking
- **Trust Updates** - Entity reliability evolution

## Shadow Mode & Rollback

Grace supports blue/green governance deployment:

1. **Shadow Mode** - Run new governance instance alongside current
2. **Delta Tracking** - Compare decisions between instances
3. **Switchover Criteria** - Accuracy, compliance, and latency gates
4. **Rollback** - Revert to last known good snapshot

## Development

### Project Structure
```
Grace-/
├── core/                    # Core infrastructure
├── grace/                   # Grace kernels and components
│   ├── governance/          # Unified governance system (merged)
│   ├── intelligence/       # Intelligence service and kernel (merged)
│   ├── mtl_kernel/          # Memory, trust, and logging
│   └── ...                  # Other kernel components
├── layer_02_event_mesh/     # Event routing
├── layer_04_audit_logs/     # Audit system
├── immune/                  # Health monitoring
├── mldl/                    # ML/DL specialists
│   └── specialists/         # Specialist implementations
├── config/                  # Configuration
├── schemas/                 # Event schemas
└── test_governance_kernel.py   # Test suite
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure constitutional compliance
5. Submit a pull request

## License

This project implements governance principles of transparency and democratic oversight.
