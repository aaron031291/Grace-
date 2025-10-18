# Grace AI System

**Status**: 🚧 **Active Development** - Core features complete, advanced features in progress

Grace is a constitutional AI system with multi-agent coordination, governance framework, and advanced reasoning capabilities.

---

## 🎯 Project Status

### ✅ Completed Components (Production Ready)

- [x] Authentication & Authorization (JWT, RBAC)
- [x] Document Management & Vector Search
- [x] Basic Governance & Policy Management
- [x] WebSocket Real-time Communication
- [x] Middleware (Logging, Rate Limiting, Metrics)
- [x] Database Models & Migrations
- [x] API Documentation (OpenAPI/Swagger)
- [x] Centralized Configuration Management

### 🚧 In Progress

- [ ] **Clarity Framework** (Classes 5-10) - 60% complete
  - [x] Memory Bank (Class 5)
  - [x] Governance Validator (Class 6)
  - [x] Feedback Integrator (Class 7)
  - [x] Specialist Consensus (Class 8)
  - [x] Unified Output (Class 9)
  - [x] Drift Detector (Class 10)
  - [ ] Full integration testing

- [ ] **MLDL Specialists** - 70% complete
  - [x] Quorum Aggregator
  - [x] Uncertainty Estimation (MC Dropout, Ensembles)
  - [ ] Production model integration
  - [ ] Real-time inference

- [ ] **AVN Self-Healing** - 50% complete
  - [x] Health monitoring
  - [x] Healing strategies
  - [ ] Automated deployment
  - [ ] Cross-component healing

- [ ] **Swarm Intelligence** - 40% complete
  - [x] Node coordinator
  - [x] Transport protocols (HTTP)
  - [ ] gRPC implementation
  - [ ] Kafka integration
  - [ ] Production peer discovery

- [ ] **Transcendence Layer** - 30% complete
  - [x] Quantum algorithms (basic)
  - [x] Scientific discovery (prototype)
  - [x] Impact evaluation (prototype)
  - [ ] Production optimization
  - [ ] Real-world validation

### 📋 Planned Features

- [ ] GraphQL API
- [ ] Multi-region deployment
- [ ] Advanced analytics dashboard
- [ ] Mobile SDK
- [ ] Federated learning
- [ ] Enhanced quantum algorithms

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or SQLite for development)
- Redis 7+ (optional, for rate limiting)
- 4GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/grace.git
cd grace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy example configuration
cp .env.example .env

# Edit configuration (see Configuration section)
nano .env
```

### Configuration

Create `.env` file with your settings:

```bash
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=sqlite:///./grace.db
# DATABASE_URL=postgresql://user:pass@localhost/grace

# Authentication
AUTH_SECRET_KEY=your-super-secret-key-min-32-chars
AUTH_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Embedding Provider
EMBEDDING_PROVIDER=huggingface
# EMBEDDING_OPENAI_API_KEY=sk-...  # If using OpenAI

# Vector Store
VECTOR_TYPE=faiss
VECTOR_FAISS_INDEX_PATH=./data/vectors/index.bin

# Features (set to true to enable)
SWARM_ENABLED=false
TRANSCENDENCE_QUANTUM_ENABLED=false
TRANSCENDENCE_DISCOVERY_ENABLED=false
TRANSCENDENCE_IMPACT_ENABLED=false

# Observability
OBSERVABILITY_LOG_LEVEL=INFO
OBSERVABILITY_METRICS_ENABLED=true
```

### Initialize Database

```bash
# Initialize database schema
python -c "from grace.database import init_db; init_db()"

# Create admin user (optional)
python scripts/create_admin.py
```

### Run API Server

```bash
# Development
uvicorn grace.api:app --reload --port 8000

# Production
uvicorn grace.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 📚 Documentation

### Core Concepts

1. **Authentication**: JWT-based with role-based access control
2. **Documents**: Semantic search using vector embeddings
3. **Governance**: Constitutional constraints and policy validation
4. **WebSocket**: Real-time communication with pub/sub
5. **Observability**: Structured logging and Prometheus metrics

### API Examples

#### Authentication

```bash
# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user",
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'

# Login
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=user&password=SecurePass123!"
```

#### Document Search

```bash
# Create document
TOKEN="your-jwt-token"
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "ML Guide",
    "content": "Machine learning is...",
    "tags": ["ml", "ai"]
  }'

# Semantic search
curl -X POST http://localhost:8000/api/v1/documents/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "k": 10
  }'
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
python test_integration_full.py

# Run with coverage
pytest --cov=grace --cov-report=html
```

### Test Files

- `test_integration_full.py` - Full system integration
- `test_clarity_framework_complete.py` - Clarity Framework
- `test_orchestration_complete.py` - Scheduling & orchestration
- `test_swarm_transcendence.py` - Advanced features
- `test_observability_complete.py` - Logging & metrics

---

## 📊 Monitoring

### Prometheus Metrics

```bash
# Access metrics endpoint
curl http://localhost:8000/metrics
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Logs

Structured JSON logs with trace IDs:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "component": "auth",
  "trace_id": "uuid-here",
  "user_id": "user123",
  "severity": "INFO",
  "message": "User authenticated successfully"
}
```

---

## 🏗️ Architecture

```
grace/
├── api/           # FastAPI endpoints
├── auth/          # Authentication system
├── config/        # Centralized configuration
├── database/      # Database models
├── documents/     # Document management
├── embeddings/    # Embedding providers
├── governance/    # Policy & governance
├── middleware/    # Logging, rate limiting
├── observability/ # Metrics & monitoring
└── websocket/     # Real-time communication

Advanced (In Progress):
├── clarity/       # Clarity Framework
├── mldl/          # ML specialists
├── avn/           # Self-healing
├── swarm/         # Multi-node coordination
└── transcendence/ # Quantum & discovery
```

---

## 🔒 Security

- JWT authentication with refresh tokens
- Role-based access control (RBAC)
- Rate limiting (per-user/IP)
- Password hashing (bcrypt)
- PII detection in governance
- Audit logging (immutable)

---

## 🤝 Contributing

This project is under active development. Contributions are welcome for:

- Bug fixes
- Documentation improvements
- Test coverage
- Feature implementations (see "In Progress" section)

---

## 📝 License

[Your License Here]

---

## 📧 Contact

- **Documentation**: `/docs`
- **Issues**: [GitHub Issues]
- **Email**: support@grace-ai.example

---

## ⚠️ Important Notes

### Current Limitations

1. **Swarm Coordination**: Only HTTP transport fully implemented
2. **Transcendence Features**: Prototypes only, not production-ready
3. **Advanced ML**: Integration with real models pending
4. **Multi-region**: Single-region deployment only
5. **Scalability**: Optimized for small-to-medium scale

### Roadmap

**Q1 2024**
- [ ] Complete Clarity Framework integration
- [ ] Production ML model integration
- [ ] Enhanced AVN self-healing

**Q2 2024**
- [ ] gRPC/Kafka for swarm
- [ ] Multi-region deployment
- [ ] Advanced analytics dashboard

**Q3 2024**
- [ ] Mobile SDK
- [ ] Federated learning
- [ ] Enhanced quantum algorithms

---

**Last Updated**: January 2024
**Version**: 1.0.0-beta
**Status**: Active Development
