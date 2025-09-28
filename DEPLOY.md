# Grace Deployment Guide

This guide covers deploying Grace Governance System in local development and production environments.

## Quick Start (One-Click Local Deployment)

```bash
# Clone and setup
git clone <repository>
cd Grace-
make up
```

That's it! Grace will be running at http://localhost:8080

## Prerequisites

- Docker and Docker Compose
- Python 3.12+
- 4GB RAM minimum, 8GB recommended
- 20GB disk space

## Local Development Deployment

### 1. Environment Setup

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your configuration
# At minimum, set API keys for AI providers
nano .env
```

### 2. One-Click Deployment

```bash
make up
```

This command will:
- Install Python dependencies
- Create necessary directories
- Start PostgreSQL, Redis, and ChromaDB containers
- Initialize databases and run migrations
- Start the Grace orchestrator
- Run health checks

### 3. Verify Deployment

```bash
# Check service health
make health-check

# View logs
make logs

# Run smoke tests
python scripts/smoke_test.py
```

### 4. Access Points

- **Web UI**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Status**: http://localhost:8080/health/status
- **Metrics**: http://localhost:8080/metrics
- **WebSocket Events**: ws://localhost:8080/ws/events

## Production Deployment

### Option 1: Docker Compose (Simple)

```bash
# Set production environment variables
export GRACE_VERSION=1.0.0
export DATABASE_URL=postgresql://user:pass@prod-db:5432/grace
export REDIS_URL=redis://prod-redis:6379
export CHROMA_URL=http://prod-chroma:8000

# Build and deploy
make build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Option 2: Fly.io

1. Install Fly CLI
2. Set secrets:
```bash
fly secrets set DATABASE_URL=postgresql://...
fly secrets set REDIS_URL=redis://...
fly secrets set OPENAI_API_KEY=sk-...
```
3. Deploy:
```bash
fly deploy
```

### Option 3: Render

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy automatically on push to main branch

### Option 4: Kubernetes

```bash
# Apply Kubernetes manifests (create these based on your cluster)
kubectl apply -f k8s/
```

## Database Setup

### PostgreSQL

The system requires PostgreSQL 15+ with the following:
- Database: `grace_governance`
- Schemas: `governance`, `audit`, `memory`, `mldl`
- Extensions: `uuid-ossp`

Initialization is automated via `init_db/01_init_grace_db.sql`.

### Redis

Used for:
- Event coordination
- Caching
- Session management

No special configuration required beyond authentication.

### ChromaDB

Vector database for:
- Document embeddings
- Similarity search
- ML model storage

Default collection: `grace_vectors`

## Configuration Management

### Environment Variables

Critical variables (see `.env.template` for all options):

```bash
# AI Providers
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Databases
DATABASE_URL=postgresql://user:pass@host:5432/grace_governance
REDIS_URL=redis://host:6379
CHROMA_URL=http://host:8000

# Security
JWT_SECRET_KEY=your_32_char_secret_here
ENCRYPTION_KEY=your_32_char_encryption_key_here

# Governance
GOVERNANCE_STRICT_MODE=true
CONSTITUTIONAL_ENFORCEMENT=true
```

### Configuration Validation

```bash
python -c "from grace.config.environment import validate_environment; print(validate_environment())"
```

## Health Monitoring

### Health Endpoints

- `/health/status` - Detailed health information
- `/health/live` - Liveness probe (for K8s)
- `/health/ready` - Readiness probe (for K8s)

### Metrics

Prometheus metrics available at `/metrics`:
- HTTP request metrics
- Database connection health
- Component status
- Custom Grace metrics

### Logging

Structured JSON logs are output to:
- Console (for container logs)
- `logs/grace.log` (application logs)
- `logs/grace_errors.log` (error logs)
- `logs/grace_audit.log` (governance audit trail)

## Troubleshooting

### Common Issues

**"Import failed" errors**
```bash
pip install -r requirements.txt
```

**Database connection failed**
```bash
# Check database is running
docker-compose ps postgres

# Check connection manually
psql $DATABASE_URL
```

**Services not starting**
```bash
# Check logs for specific service
docker-compose logs grace_orchestrator

# Restart specific service
docker-compose restart grace_orchestrator
```

### Debug Mode

```bash
# Enable debug mode
echo "GRACE_DEBUG=true" >> .env
make down && make up
```

### Reset Everything

```bash
# WARNING: This destroys all data
make db-reset
make clean
make up
```

## Performance Tuning

### Resource Limits

Default container limits:
- Memory: 2GB
- CPU: 2 cores

Adjust in `docker-compose.yml` or environment variables.

### Database Tuning

PostgreSQL settings for production:
- `shared_buffers` = 25% of RAM
- `effective_cache_size` = 75% of RAM
- `work_mem` = 4MB per connection
- `maintenance_work_mem` = 64MB

### Scaling

For high load:
1. Use database connection pooling
2. Deploy multiple Grace instances behind a load balancer
3. Use Redis Cluster for distributed caching
4. Scale ChromaDB horizontally

## Security Considerations

### Network Security
- Use HTTPS in production
- Restrict database access to application networks
- Enable firewall rules

### API Security
- Set strong JWT secrets
- Use API keys for service-to-service communication
- Enable CORS restrictions

### Data Security
- Enable encryption at rest for databases
- Use encrypted connections (SSL/TLS)
- Regular security updates

## Backup and Recovery

### Database Backups

```bash
# PostgreSQL backup
pg_dump $DATABASE_URL > grace_backup.sql

# Redis backup (automatic with persistence)
# ChromaDB backup (copy data directory)
```

### Recovery

```bash
# Restore PostgreSQL
psql $DATABASE_URL < grace_backup.sql

# Restart services
make down && make up
```

## Updates and Maintenance

### Updating Grace

```bash
git pull origin main
make down
make build
make up
```

### Database Migrations

```bash
# Run pending migrations
make migrate

# Rollback if needed
python -m alembic downgrade -1
```

## Support

For issues and support:
- Check logs: `make logs`
- Run diagnostics: `python scripts/smoke_test.py`
- Review health status: `curl localhost:8080/health/status`

## Next Steps

After deployment:
1. Configure AI provider API keys
2. Set up monitoring and alerting
3. Configure backup schedules
4. Review and customize governance policies
5. Integrate with your applications via the API