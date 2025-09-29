# Grace Infra Quickstart - Complete Implementation

This implementation provides a complete infrastructure quickstart for the Grace AI Governance System with enterprise-grade features.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.12+
- Git

### 1. Environment Setup
```bash
# Clone and setup
git clone https://github.com/aaron031291/Grace-.git
cd Grace-

# Copy environment template
cp .env.template .env

# Edit .env with your configuration
# At minimum, set:
# - DATABASE_URL=postgresql://grace_user:grace_pass@localhost:5432/grace_governance  
# - REDIS_URL=redis://:grace_redis_pass@localhost:6379
# - S3_ENDPOINT=http://localhost:9000
# - VECTOR_URL=http://localhost:6333
# - JWT_SECRET_KEY=your-secret-key-here
```

### 2. Start All Services
```bash
# Start full infrastructure
docker-compose up -d

# Check service health
docker-compose ps
```

### 3. Run Database Migrations
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run migrations
python -m alembic upgrade head
```

### 4. Create Authentication Token
```bash
# Using curl to create a token
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "admin", "roles": ["admin"]}'
```

### 5. Test the System
```bash
# Test memory ingestion (use token from step 4)
curl -X POST http://localhost:8080/api/v1/memory/ingest \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"text": "Grace is an AI governance system", "title": "Test Document"}'

# Test semantic search
curl "http://localhost:8080/api/v1/search?q=governance" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🏗️ Architecture

### Services Overview
- **API Service**: FastAPI-based REST API with JWT authentication
- **Worker Service**: Background task processing (ingestion, embeddings, media)
- **PostgreSQL**: Primary database with Alembic migrations
- **Redis**: Caching and task queues
- **MinIO**: S3-compatible object storage
- **Qdrant**: Vector database for semantic search
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization and dashboards

### Key Features Implemented

#### 1. Complete Infrastructure (Phase 1)
- ✅ Full docker-compose with all required services
- ✅ Health checks and service discovery
- ✅ Environment-based configuration
- ✅ Prometheus and Grafana monitoring

#### 2. Database & Persistence (Phase 2)  
- ✅ SQLAlchemy models for all data structures
- ✅ Alembic database migrations
- ✅ Repository pattern with LRU caching
- ✅ Support for Sessions, Messages, Knowledge Entries, Tasks, etc.

#### 3. Memory Ingestion System (Phase 3)
- ✅ File text extraction (multiple formats)
- ✅ Intelligent text chunking (1-2k tokens)
- ✅ Vector embedding generation
- ✅ Semantic search with trust filtering
- ✅ Background processing via worker queues

#### 4. Policy & Security (Phase 4)
- ✅ YAML-based policy rules
- ✅ Dangerous operation detection
- ✅ CI integration with GitHub Actions
- ✅ Sandbox branch enforcement for IDE changes

#### 5. Authentication & RBAC (Phase 5)
- ✅ JWT-based authentication
- ✅ Scopes: `read:chat`, `write:memory`, `govern:tasks`, `sandbox:build`
- ✅ WebSocket authentication support
- ✅ Role-based access control

#### 6. Observability (Phase 6)
- ✅ Request ID tracking
- ✅ Structured JSON logging
- ✅ Prometheus metrics
- ✅ Distributed tracing support

## 🔐 Authentication & Authorization

### Scopes
- `read:chat` - Read chat messages and search
- `write:memory` - Ingest content into memory system
- `govern:tasks` - Execute governance and administrative tasks  
- `sandbox:build` - Build and deploy in sandbox environments
- `network:access` - Make network requests
- `admin` - Full system administration

### Creating Tokens
```python
# Example: Create admin token
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "admin",
    "roles": ["admin"]
  }'

# Example: Create developer token  
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "developer",
    "roles": ["developer"]
  }'
```

## 📊 Monitoring & Observability

### Accessing Dashboards
- **Grafana**: http://localhost:3000 (admin/grace_grafana_pass)
- **Prometheus**: http://localhost:9090
- **API Docs**: http://localhost:8080/docs (when debug=true)

### Key Metrics
- Request counts and duration
- Queue sizes and processing times
- Memory ingestion success rates
- Authentication success/failure rates
- Policy violation counts

## 🔍 Memory System Usage

### Ingest Documents
```bash
# Text content
curl -X POST http://localhost:8080/api/v1/memory/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document content here",
    "title": "Document Title",
    "tags": ["category1", "category2"],
    "trust_score": 0.8
  }'

# File path (worker processes asynchronously)
curl -X POST http://localhost:8080/api/v1/memory/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "trust_score": 0.7
  }'
```

### Search Knowledge
```bash
curl "http://localhost:8080/api/v1/search?q=your%20search%20query&trust_threshold=0.6&limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

## 🛡️ Policy System

### Policy Rules Location
- Default policies: `grace/policy/default_policies.yml`
- Custom policies: Set `POLICY_FILE` environment variable

### Example Policy Rule
```yaml
dangerous_code_execution:
  description: 'Block dangerous code execution patterns'
  severity: 'critical'
  enabled: true
  conditions:
    - type: 'content_pattern'
      pattern: '(exec|eval|subprocess|os\.system)'
      regex: true
  actions: ['block', 'log', 'alert']
  exceptions:
    - type: 'scope_required'
      scopes: ['sandbox:build']
```

### CI Integration
The system automatically validates all code changes in CI via `.github/workflows/policy-validation.yml`:
- Scans changed files for policy violations
- Blocks dangerous operations
- Adds `policy:pass` labels to compliant PRs
- Requires human approval for IDE changes

## 🔧 Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start database only (for local API development)
docker-compose up -d postgres redis minio qdrant

# Run API locally
python -m grace.api.api_service

# Run worker locally  
SERVICE_MODE=worker python -m grace.worker.worker_service
```

### Database Migrations
```bash
# Create new migration
python -m alembic revision --autogenerate -m "Description"

# Apply migrations
python -m alembic upgrade head

# Check migration status
python -m alembic current
```

### Policy Validation
```bash
# Validate policies on changed files
python -m grace.policy.ci_integration file1.py file2.py

# Generate policy report
python -m grace.policy.ci_integration --output report.md *.py
```

## 🚀 Production Deployment

### Environment Variables
Set these in production:
- `JWT_SECRET_KEY` - Strong secret for JWT signing
- `DATABASE_URL` - Production PostgreSQL URL  
- `REDIS_URL` - Production Redis URL
- `S3_ENDPOINT` - Production S3/MinIO endpoint
- `VECTOR_URL` - Production Qdrant URL
- `GOVERNANCE_STRICT_MODE=true` - Enable strict governance
- `ENABLE_TELEMETRY=true` - Enable monitoring

### Security Considerations
- Use strong JWT secrets
- Configure CORS appropriately  
- Set up proper TLS/SSL
- Review and customize policy rules
- Monitor security logs

### Scaling
- Run multiple API instances behind load balancer
- Scale worker instances based on queue sizes
- Use external managed services for databases
- Configure Redis clustering for high availability

## 📖 API Documentation

When running in debug mode, API documentation is available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Key Endpoints
- `POST /api/v1/auth/token` - Create authentication token
- `GET /api/v1/auth/me` - Get current user info
- `POST /api/v1/memory/ingest` - Ingest content
- `GET /api/v1/search` - Semantic search
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `WebSocket /ws` - Real-time communication

This implementation provides a complete, production-ready foundation for the Grace AI Governance System with enterprise-grade security, observability, and scalability features.