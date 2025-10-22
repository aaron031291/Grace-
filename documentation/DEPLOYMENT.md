# Grace Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Hardening](#security-hardening)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU:** 4+ cores
- **RAM:** 8GB+ (16GB recommended)
- **Storage:** 50GB+ SSD
- **OS:** Ubuntu 20.04+, RHEL 8+, or macOS

### Software Requirements

- Docker 24.0+
- Docker Compose 2.0+
- Python 3.11+
- PostgreSQL 14+ (or SQLite for dev)
- Redis 7+ (optional but recommended)

## Docker Deployment

### Quick Start

1. **Clone Repository**

```bash
git clone https://github.com/yourorg/grace.git
cd grace
```

2. **Configure Environment**

```bash
cp .env.example .env
nano .env  # Edit configuration
```

Required environment variables:

```bash
# Grace Configuration
GRACE_ENV=production
GRACE_DEBUG=false
AUTH_SECRET_KEY=your-secure-secret-key-min-32-chars

# Database
DATABASE_URL=postgresql://grace:password@postgres:5432/grace_db

# Redis
REDIS_URL=redis://redis:6379/0

# Security
ENCRYPTION_KEY=your-encryption-key-32-bytes-base64
```

3. **Build and Run**

```bash
docker-compose up -d
```

4. **Verify Deployment**

```bash
curl http://localhost:8000/health
```

### Docker Compose Configuration

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  grace-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - AUTH_SECRET_KEY=${AUTH_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: grace_db
      POSTGRES_USER: grace
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (optional)

### Deploy to Kubernetes

1. **Create Namespace**

```bash
kubectl create namespace grace
```

2. **Create Secrets**

```bash
kubectl create secret generic grace-secrets \
  --from-literal=auth-secret-key=your-secret-key \
  --from-literal=db-password=your-db-password \
  --from-literal=encryption-key=your-encryption-key \
  -n grace
```

3. **Apply Configurations**

```bash
kubectl apply -f k8s/ -n grace
```

### Kubernetes Manifests

`k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grace-api
  namespace: grace
spec:
  replicas: 3
  selector:
    matchLabels:
      app: grace-api
  template:
    metadata:
      labels:
        app: grace-api
    spec:
      containers:
      - name: grace-api
        image: grace:latest
        ports:
        - containerPort: 8000
        env:
        - name: GRACE_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: grace-secrets
              key: database-url
        - name: AUTH_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: grace-secrets
              key: auth-secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: grace-api
  namespace: grace
spec:
  selector:
    app: grace-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GRACE_ENV` | Environment (development/production) | `development` | Yes |
| `GRACE_DEBUG` | Enable debug mode | `false` | No |
| `DATABASE_URL` | PostgreSQL connection string | `sqlite:///./grace.db` | Yes |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` | No |
| `AUTH_SECRET_KEY` | JWT signing key (min 32 chars) | - | Yes |
| `ENCRYPTION_KEY` | Data encryption key (base64) | - | Yes |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `KERNELS_ENABLED` | Comma-separated kernel list | `multi_os,mldl,resilience` | No |

### Configuration File

`config/grace.yaml`:

```yaml
service:
  name: grace-unified
  host: 0.0.0.0
  port: 8000

database:
  pool_size: 10
  max_overflow: 20
  echo: false

redis:
  enabled: true
  max_connections: 50

event_bus:
  max_queue_size: 10000
  dlq_max_size: 1000
  ttl_cleanup: true

security:
  rate_limit:
    enabled: true
    default_limit: 100
    window_seconds: 60
  rbac:
    enabled: true
    minimum_trust: 0.5

monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

## Database Setup

### PostgreSQL

1. **Create Database**

```sql
CREATE DATABASE grace_db;
CREATE USER grace WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE grace_db TO grace;
```

2. **Run Migrations**

```bash
python scripts/apply_migrations.py
```

3. **Verify**

```bash
psql -U grace -d grace_db -c "\dt"
```

### SQLite (Development Only)

```bash
# Automatic on first run
python main.py service
```

## Monitoring Setup

### Prometheus

`config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'grace'
    static_configs:
      - targets: ['grace-api:8000']
    metrics_path: '/api/v1/metrics/prometheus'
```

### Grafana

1. **Access Grafana:** http://localhost:3000
2. **Login:** admin / (your password)
3. **Add Data Source:** Prometheus (http://prometheus:9090)
4. **Import Dashboard:** Upload `config/grafana/grace-dashboard.json`

## Security Hardening

### SSL/TLS Configuration

Use reverse proxy (Nginx/Traefik) for HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name grace.example.com;

    ssl_certificate /etc/ssl/certs/grace.crt;
    ssl_certificate_key /etc/ssl/private/grace.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://grace-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Network Security

```bash
# Firewall rules (UFW)
ufw allow 443/tcp  # HTTPS
ufw deny 8000/tcp  # Block direct API access
ufw enable
```

### Secrets Management

Use Kubernetes secrets or vault:

```bash
# Kubernetes
kubectl create secret generic grace-secrets \
  --from-file=auth-secret-key=./secrets/auth.key \
  --from-file=encryption-key=./secrets/enc.key
```

## Troubleshooting

### Check Logs

```bash
# Docker
docker-compose logs -f grace-api

# Kubernetes
kubectl logs -f deployment/grace-api -n grace
```

### Common Issues

**Issue: Connection Refused**

```bash
# Check if service is running
docker ps | grep grace
kubectl get pods -n grace

# Check ports
netstat -tlnp | grep 8000
```

**Issue: Database Connection Failed**

```bash
# Test database connection
psql -U grace -h localhost -d grace_db

# Check DATABASE_URL format
echo $DATABASE_URL
```

**Issue: High Memory Usage**

```bash
# Check memory limits
docker stats grace-api
kubectl top pods -n grace

# Adjust in docker-compose.yml or k8s manifests
```

### Health Checks

```bash
# API Health
curl http://localhost:8000/health

# Detailed health
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/monitoring/health

# Metrics
curl http://localhost:8000/api/v1/metrics/prometheus
```
