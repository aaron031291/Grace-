# Grace Configuration Guide

This document covers all configuration options for the Grace Governance System.

## Environment Variables

### Required Configuration

These variables must be set for production deployment:

```bash
# Database URLs
DATABASE_URL=postgresql://grace_user:password@host:5432/grace_governance
REDIS_URL=redis://:password@host:6379
CHROMA_URL=http://host:8000

# Security keys (minimum 32 characters)
JWT_SECRET_KEY=your_jwt_secret_key_here_min_32_chars
ENCRYPTION_KEY=your_encryption_key_here_min_32_chars
```

### AI Provider Configuration

```bash
# OpenAI (recommended)
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_ORG_ID=org-your_openai_org_id_here

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Service Configuration

```bash
# Network binding
API_HOST=0.0.0.0                    # Interface to bind to
API_PORT=8080                       # Main API port
ORCHESTRATOR_HOST=0.0.0.0           # Orchestrator interface
ORCHESTRATOR_PORT=8081              # Orchestrator port

# CORS and security
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
API_KEY_HEADER=X-Grace-API-Key
```

### Grace System Configuration

```bash
# Instance identification
GRACE_INSTANCE_ID=grace_main_001    # Unique instance identifier
GRACE_VERSION=1.0.0                 # Version for tracking
GRACE_LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
GRACE_DEBUG=false                   # Enable debug mode
```

### Governance Settings

```bash
# Constitutional governance
GOVERNANCE_STRICT_MODE=true         # Enforce strict constitutional compliance
CONSTITUTIONAL_ENFORCEMENT=true      # Enable constitutional validation
AUTO_ROLLBACK_ENABLED=true          # Auto-rollback on violations

# Decision thresholds
PARLIAMENT_QUORUM_THRESHOLD=0.6     # Parliament voting quorum (0.0-1.0)
TRUST_SCORE_THRESHOLD=0.7           # Minimum trust score for actions
```

### ML/DL Quorum Configuration

```bash
# Specialist consensus
MLDL_MIN_SPECIALISTS=3              # Minimum specialists for consensus
MLDL_MAX_SPECIALISTS=5              # Maximum specialists to consult
MLDL_CONSENSUS_THRESHOLD=0.65       # Consensus threshold (0.0-1.0)
MLDL_TIMEOUT_SECONDS=30             # Timeout for specialist responses
```

### Vector Database Configuration

```bash
# ChromaDB settings
CHROMA_COLLECTION_NAME=grace_vectors # Collection name for vectors
EMBEDDING_MODEL=all-MiniLM-L6-v2     # Sentence transformer model
VECTOR_DIMENSIONS=384                # Vector dimensions (model-dependent)
```

### Monitoring and Telemetry

```bash
# Monitoring
ENABLE_TELEMETRY=true               # Enable telemetry collection
ENABLE_HEALTH_MONITORING=true       # Enable health monitoring
METRICS_EXPORT_INTERVAL=30          # Metrics export interval (seconds)
PROMETHEUS_PORT=9090                # Prometheus metrics port
HEALTH_CHECK_INTERVAL=10            # Health check interval (seconds)

# Observability
SENTRY_DSN=your_sentry_dsn_here_optional  # Error tracking (optional)
```

### Audit and Logging

```bash
# Audit configuration
AUDIT_LOG_RETENTION_DAYS=90         # How long to keep audit logs
STRUCTURED_LOGGING=true             # Use structured JSON logging
LOG_FORMAT=json                     # json or text

# Log levels per component
GRACE_GOVERNANCE_LOG_LEVEL=DEBUG    # Governance kernel logging
GRACE_INGRESS_LOG_LEVEL=INFO        # Ingress kernel logging
GRACE_ORCHESTRATION_LOG_LEVEL=INFO  # Orchestration logging
```

### Performance Configuration

```bash
# Request handling
MAX_CONCURRENT_REQUESTS=100         # Max concurrent API requests
REQUEST_TIMEOUT_SECONDS=30          # Request timeout
MEMORY_CACHE_SIZE_MB=512            # In-memory cache size

# Database performance
DB_POOL_SIZE=20                     # Database connection pool size
DB_MAX_OVERFLOW=0                   # Max overflow connections
DB_POOL_TIMEOUT=30                  # Connection pool timeout
```

### Development Settings

```bash
# Development mode
DEV_MODE=false                      # Enable development features
MOCK_AI_RESPONSES=false             # Mock AI API responses (for testing)
ENABLE_MOCK_SERVICES=false          # Enable mock external services
HOT_RELOAD=false                    # Enable hot reload (dev only)
```

### Container Settings

```bash
# Resource limits (for containerized deployment)
CONTAINER_MEMORY_LIMIT=2g           # Memory limit
CONTAINER_CPU_LIMIT=2               # CPU limit
GRACEFUL_SHUTDOWN_TIMEOUT=30        # Graceful shutdown timeout
```

## Configuration Files

### Main Configuration Structure

Grace uses a hierarchical configuration system:

1. **Default values** (in code)
2. **Configuration files** (`grace/config/`)
3. **Environment variables** (highest priority)

### Governance Configuration

Located at `grace/config/governance_config.py`:

```python
GOVERNANCE_CONFIG = {
    "constitutional_principles": {
        "transparency": {"weight": 1.0, "required": True},
        "fairness": {"weight": 1.0, "required": True},
        "accountability": {"weight": 0.9, "required": True},
        "consistency": {"weight": 0.8, "required": True},
        "harm_prevention": {"weight": 1.0, "required": True}
    },
    "decision_thresholds": {
        "approval_threshold": 0.7,
        "strict_enforcement": True,
        "allow_overrides": False
    }
}
```

### Logging Configuration

See `logging.yaml` for structured logging setup:

- **Console output**: JSON format for container logs
- **File logging**: Rotating logs with retention
- **Audit logs**: Separate audit trail
- **Error tracking**: Dedicated error logging

## Configuration Validation

### Environment Validation

```bash
# Check required environment variables
python -c "
from grace.config.environment import validate_environment
missing = validate_environment()
if missing:
    print('Missing:', missing)
else:
    print('Environment valid')
"
```

### Configuration Testing

```bash
# Test configuration loading
python -c "
from grace.config.environment import get_grace_config
config = get_grace_config()
print('Configuration loaded successfully')
print(f'Instance: {config[\"environment_config\"][\"instance_id\"]}')
print(f'Version: {config[\"environment_config\"][\"version\"]}')
"
```

## Security Best Practices

### Secret Management

**Never commit secrets to version control!**

- Use environment variables for secrets
- Consider using secret management tools:
  - Docker secrets
  - Kubernetes secrets
  - HashiCorp Vault
  - AWS Secrets Manager
  - Azure Key Vault

### Key Generation

```bash
# Generate secure random keys
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"
```

### Database Security

```bash
# Use strong database passwords
DATABASE_URL=postgresql://grace_user:$(openssl rand -base64 32)@host:5432/grace_governance

# Enable SSL for database connections
DATABASE_URL=postgresql://user:pass@host:5432/grace_governance?sslmode=require
```

## Environment-Specific Configuration

### Development Environment

```bash
# .env.dev
GRACE_DEBUG=true
GRACE_LOG_LEVEL=DEBUG
DEV_MODE=true
HOT_RELOAD=true
MOCK_AI_RESPONSES=true
ENABLE_MOCK_SERVICES=true
```

### Testing Environment

```bash
# .env.test
GRACE_DEBUG=true
GRACE_LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/grace_test
REDIS_URL=redis://localhost:6379/1
MOCK_AI_RESPONSES=true
```

### Production Environment

```bash
# .env.prod
GRACE_DEBUG=false
GRACE_LOG_LEVEL=INFO
GOVERNANCE_STRICT_MODE=true
CONSTITUTIONAL_ENFORCEMENT=true
STRUCTURED_LOGGING=true
ENABLE_TELEMETRY=true
SENTRY_DSN=https://your-sentry-dsn
```

## Configuration Monitoring

### Health Checks

Configuration status is included in health checks:

```bash
curl http://localhost:8080/health/status
```

Returns configuration validation status and any missing required variables.

### Configuration Changes

Grace supports runtime configuration reloading for non-critical settings:

```bash
# Reload configuration (if supported)
curl -X POST http://localhost:8080/api/v1/admin/reload-config
```

## Troubleshooting Configuration

### Common Issues

1. **Missing environment variables**
   - Check `.env` file exists and is readable
   - Verify environment variable names are correct

2. **Database connection issues**
   - Test database URL manually: `psql $DATABASE_URL`
   - Check network connectivity and credentials

3. **AI provider errors**
   - Verify API keys are correct and active
   - Check API quotas and billing

4. **Permission errors**
   - Ensure log directories are writable
   - Check file permissions for configuration files

### Debug Configuration

```bash
# Enable verbose configuration logging
GRACE_DEBUG=true GRACE_LOG_LEVEL=DEBUG python -m grace.config.environment
```

### Configuration Export

```bash
# Export current configuration (sanitized)
python -c "
from grace.config.environment import get_grace_config
import json
config = get_grace_config()
# Remove sensitive data
for section in config.values():
    if isinstance(section, dict):
        for key in section.keys():
            if 'key' in key.lower() or 'secret' in key.lower() or 'pass' in key.lower():
                section[key] = '***'
print(json.dumps(config, indent=2))
"
```

This guide covers all major configuration aspects of Grace. For specific deployment scenarios, refer to the [DEPLOY.md](DEPLOY.md) documentation.