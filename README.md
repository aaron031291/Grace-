# Grace AI System

**Advanced Multi-Agent AI System with Distributed Intelligence, Memory, and Consciousness**

Grace is a production-ready AI system featuring:
- 🧠 **Enhanced Memory System** with PostgreSQL and Redis
- 🔍 **Clarity Framework** for LLM consistency and governance
- 🤝 **Swarm Intelligence** for distributed coordination
- 🚀 **Transcendence Layer** for advanced reasoning
- 🎯 **Consciousness Layer** with self-awareness
- 📊 **Health Monitoring** and AVN self-diagnostics

## Architecture

```
grace/
├── clarity/           # Clarity Framework (Classes 5-10)
│   ├── memory_scoring.py
│   ├── governance_validation.py
│   ├── feedback_integration.py
│   ├── specialist_consensus.py
│   ├── output_schema.py
│   └── drift_detection.py
├── swarm/            # Swarm Intelligence
│   ├── node_coordinator.py
│   ├── consensus_engine.py
│   ├── knowledge_graph_manager.py
│   └── swarm_orchestrator.py
├── transcendent/     # Transcendence Layer
│   ├── quantum_algorithms.py
│   ├── scientific_discovery.py
│   ├── societal_impact.py
│   └── orchestrator.py
├── memory/           # Enhanced Memory Core
│   ├── enhanced_memory_core.py
│   ├── db_setup.sql
│   └── redis_setup.py
├── integration/      # Integration Layer
│   ├── event_bus.py
│   ├── quorum_integration.py
│   └── avn_reporter.py
└── core/            # Core Runtime
    ├── grace_core_runtime.py
    └── unified_logic_extensions.py
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- PostgreSQL 14+ (with pgvector extension)
- Redis 7+

### Installation

1. **Clone and setup:**
```bash
git clone <repository>
cd Grace-
chmod +x setup.sh
./setup.sh
```

2. **Configure environment:**
```bash
cp .env.template .env
# Edit .env with your API keys and settings
```

3. **Start services:**
```bash
docker-compose up -d
```

4. **Initialize database:**
```bash
docker exec grace-postgres psql -U grace_user -d grace_db -f /docker-entrypoint-initdb.d/init.sql
```

### Running Grace

**Memory System Demo:**
```bash
python grace/memory/production_demo.py
```

**Clarity Framework Demo:**
```bash
python grace/clarity/clarity_demo.py
```

**Swarm Intelligence Demo:**
```bash
python grace/swarm/integration_example.py
```

**Full System:**
```bash
python grace/core/grace_core_runtime.py
```

## Features

### 1. Enhanced Memory Core

Production-ready memory system with:
- ✅ PostgreSQL transactions with ACID compliance
- ✅ Redis caching with automatic invalidation
- ✅ Vector embeddings with OpenAI API
- ✅ Graceful fallback for API failures
- ✅ Health monitoring and metrics
- ✅ Integration with Clarity Framework

```python
from grace.memory import EnhancedMemoryCore

memory = EnhancedMemoryCore(
    db_connection=db,
    redis_client=redis,
    clarity_memory_bank=clarity
)

memory.initialize()  # Health checks embedded

memory.store_structured_memory(
    memory_id="mem_001",
    content={"decision": "approved"},
    memory_type="episodic"
)
```

### 2. Clarity Framework (Classes 5-10)

LLM clarity and consistency management:

**Class 5: Memory Scoring**
```python
from grace.clarity import LoopMemoryBank

memory_bank = LoopMemoryBank()
scores = memory_bank.score(memory_id, context)
# Returns: clarity, relevance, ambiguity, composite scores
```

**Class 6: Governance Validation**
```python
from grace.clarity import ConstitutionValidator

validator = ConstitutionValidator()
result = validator.validate_against_constitution(action, context)
# Returns: passed, violations, warnings, recommendations
```

**Class 7: Feedback Integration**
```python
from grace.clarity import FeedbackIntegrator

feedback = FeedbackIntegrator(memory_bank)
feedback.loop_output_to_memory(loop_id, output)
```

**Class 8: Specialist Consensus**
```python
from grace.clarity import MLDLSpecialist

specialist = MLDLSpecialist()
quorum = specialist.evaluate(proposal, required_specialists)
# Returns: decision, confidence, consensus_strength
```

**Class 9: Universal Output Format**
```python
from grace.clarity import GraceLoopOutput

output = GraceLoopOutput(
    loop_id="loop_001",
    iteration=1,
    result=result,
    confidence=0.95
)
```

**Class 10: Loop Drift Detection**
```python
from grace.clarity import GraceCognitionLinter

linter = GraceCognitionLinter()
alerts = linter.lint_loop_output(loop_id, iteration, output)
```

### 3. Swarm Intelligence

Distributed coordination and consensus:

```python
from grace.swarm import SwarmOrchestrator

orchestrator = SwarmOrchestrator()
orchestrator.connect_event_bus(event_bus)
orchestrator.connect_quorum(quorum)

# Distribute task
task_id = orchestrator.distribute_task(
    task_type="optimization",
    payload={"problem": "route_optimization"}
)

# Create consensus proposal
proposal_id = orchestrator.propose_to_swarm(
    proposer_id="node_1",
    proposal_type="model_selection",
    content={"model": "gpt-4"}
)
```

### 4. Transcendence Layer

Advanced reasoning capabilities:

```python
from grace.transcendent import TranscendenceOrchestrator

transcendence = TranscendenceOrchestrator()

# Quantum-inspired search
result = transcendence.quantum.quantum_annealing(
    cost_function=cost_fn,
    initial_state=state
)

# Scientific discovery
hypothesis = transcendence.discovery.generate_hypothesis(
    observations=data,
    domain="healthcare"
)

# Societal impact
assessment = transcendence.impact.assess_impact(
    action="AI deployment",
    context={"stakeholders": 5}
)
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://grace_user:password@localhost:5432/grace_db
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=your_api_key_here

# Grace Settings
GRACE_ENV=production
GRACE_LOG_LEVEL=INFO
MEMORY_CACHE_TTL=3600
```

### Docker Services

```yaml
# Start all services
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Stop services
docker-compose down
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run integration tests
pytest grace/memory/integration_test.py -v

# Run with coverage
pytest --cov=grace tests/
```

## Monitoring

### Health Checks

```python
# Memory system health
health = memory_core.get_health_status()

# System-wide health
system_health = avn.get_system_health()
```

### Metrics (with Prometheus)

- Access Prometheus: http://localhost:9090
- Access Grafana: http://localhost:3000

## Development

### Repository Organization

```bash
# Organize repository structure
chmod +x organize_repo.sh
./organize_repo.sh
```

### Adding New Components

1. Create module in `grace/`
2. Add to `__init__.py`
3. Register in orchestrator manifest
4. Add tests in `tests/`
5. Update documentation

## Production Deployment

### Database Setup

```bash
# Production PostgreSQL with pgvector
docker run -d \
  --name grace-postgres \
  -e POSTGRES_DB=grace_db \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -p 5432:5432 \
  ankane/pgvector:latest
```

### Redis Setup

```bash
# Production Redis with persistence
docker run -d \
  --name grace-redis \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --appendonly yes
```

### Application Deployment

```bash
# Build production image
docker build -t grace-ai:latest .

# Run with production settings
docker run -d \
  -e DATABASE_URL=$DATABASE_URL \
  -e REDIS_URL=$REDIS_URL \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -p 8000:8000 \
  grace-ai:latest
```

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL
docker exec grace-postgres pg_isready

# View logs
docker logs grace-postgres
```

### Redis Connection Issues

```bash
# Check Redis
docker exec grace-redis redis-cli ping

# View logs
docker logs grace-redis
```

### Memory System Issues

```bash
# Run diagnostics
python grace/memory/production_demo.py

# Check health
python -c "from grace.memory import EnhancedMemoryCore; print(EnhancedMemoryCore().initialize())"
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## License

[Your License Here]

## Support

- Documentation: `/documentation`
- Issues: [GitHub Issues]
- Email: [support email]

---

**Grace AI System** - Built with ❤️ for advanced AI research and production deployment
