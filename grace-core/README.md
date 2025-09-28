# Grace Core - AI Governance & Intelligence System

Grace is a comprehensive AI-powered system for distributed governance, intelligence gathering, and memory management. It features a multi-kernel architecture designed for scalability, resilience, and autonomous operation.

## Architecture Overview

Grace consists of 11 specialized kernels that work together to provide a complete AI governance solution:

### Core Kernels

1. **Memory–Trust–Log (MTL) Kernel** - Unified memory, trust ledger, and immutable logging
2. **Governance Kernel** - Policy enforcement and decision making 
3. **Intelligence Kernel** - Quorum consensus and specialist analysis
4. **Ingress Kernel** - Data intake and trigger processing
5. **Resilience Kernel** - Health monitoring and self-healing
6. **Learning Kernel** - Adaptation and pattern recognition
7. **Consciousness Kernel** - Self-awareness and priority management
8. **Orchestration Kernel** - Job scheduling and resource management
9. **Interface Kernel** - REST/WebSocket API gateway
10. **Business Ops Kernel** - External action execution
11. **Multi-OS Kernel** - Cross-platform runtime management

### Supporting Systems

- **Verification Engine** - Truth validation and reasoning
- **Swarm Coordination** - Distributed consensus (future-ready)
- **Grace IDE** - Web-based management interface

## Features

- **Distributed Architecture**: Multi-kernel design for high availability
- **AI-Powered Governance**: Automated policy enforcement and decision making
- **Comprehensive Memory**: Lightning (Redis), Fusion (Postgres), Vector (Chroma/FAISS)
- **Trust Management**: Trust ledgers with decay mechanisms
- **Immutable Logging**: Append-only chains with Merkle proofs
- **Quorum Intelligence**: Consensus-based decision making with specialists
- **Self-Healing**: Autonomous health monitoring and recovery
- **Cross-Platform**: Multi-OS support with dependency management

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run Grace system
grace serve

# Access web interface
open http://localhost:8000
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black .

# Type checking
mypy grace/
```

## Architecture Details

### MTL Kernel (Memory-Trust-Log)
The foundational kernel providing unified memory, trust tracking, and immutable logging capabilities.

### Governance Kernel
Handles policy evaluation, verification, quorum consensus, and decision synthesis.

### Intelligence Kernel
Manages specialist consensus and quorum-based decision making.

### W5H Indexing
All content is indexed using Who/What/Where/When/Why/How extraction for comprehensive searchability.

## Configuration

Grace uses environment-based configuration. See `grace/config/` for available settings.

## API Documentation

REST API documentation is available at `/docs` when running the server.

## Contributing

Please read our contributing guidelines and submit pull requests for improvements.

## License

MIT License - see LICENSE file for details.