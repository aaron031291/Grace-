# Grace System - Production Deployment Guide

## Quick Start Deployment

### Prerequisites

- Docker & Docker Compose
- PostgreSQL 14+
- Redis 7+ (optional, for rate limiting)
- Python 3.11+
- 4+ CPU cores, 8GB+ RAM recommended

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourorg/grace.git
cd grace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

Required environment variables:
```bash
# Database
DATABASE_URL=postgresql://grace:password@localhost:5432/grace_db

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Features
ENABLE_SWARM=true
ENABLE_QUANTUM=true
ENABLE_DISCOVERY=true
ENABLE_IMPACT=true

# Embedding Provider
EMBEDDING_PROVIDER=huggingface  # or: openai, local
OPENAI_API_KEY=sk-...  # if using OpenAI

# Vector Store
VECTOR_STORE=faiss  # or: pgvector
FAISS_INDEX_PATH=/data/vectors/grace_index.bin

# Transport (for swarm)
SWARM_TRANSPORT=http  # or: grpc, kafka
SWARM_PORT=8080

# Metrics
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### 3. Database Initialization

```bash
# Create database
createdb grace_db

# Run migrations
python -c "from grace.database import init_db; init_db()"

# Create initial admin user
python scripts/create_admin.py
```

### 4. Start Services

**Option A: Docker Compose (Recommended)**

```bash
docker-compose up -d
```

**Option B: Manual Start**

```bash
# Start API server
uvicorn grace.api:app --host 0.0.0.0 --port 8000 --workers 4

# Start worker (if using Celery)
celery -A grace.worker worker --loglevel=info

# Start scheduler
python -m grace.orchestration.scheduler
```

### 5. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/api/docs

# Metrics
curl http://localhost:8000/metrics

# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=YourAdminPassword"
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY grace/ ./grace/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /data/vectors /data/logs /data/immutable_logs

# Expose ports
EXPOSE 8000 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "grace.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: grace_db
      POSTGRES_USER: grace
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U grace"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  grace-api:
    build: .
    ports:
      - "8000:8000"
      - "8080:8080"  # Swarm port
      - "9090:9090"  # Metrics
    environment:
      - DATABASE_URL=postgresql://grace:${DB_PASSWORD}@postgres:5432/grace_db
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
      - ENABLE_SWARM=true
      - ENABLE_QUANTUM=true
    volumes:
      - app_data:/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  app_data:
  prometheus_data:
  grafana_data:
```

---

## Kubernetes Deployment

### grace-deployment.yaml

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
          name: http
        - containerPort: 8080
          name: swarm
        - containerPort: 9090
          name: metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: grace-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: grace-secrets
              key: secret-key
        - name: REDIS_URL
          value: redis://redis-service:6379
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
  name: grace-api-service
  namespace: grace
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    name: http
  - port: 8080
    targetPort: 8080
    name: swarm
  - port: 9090
    targetPort: 9090
    name: metrics
  selector:
    app: grace-api
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'grace-api'
    static_configs:
      - targets: ['grace-api:9090']
    metrics_path: '/metrics'

  - job_name: 'grace-nodes'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - grace
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: grace-api
```

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (default: admin/admin)

Import dashboards from `monitoring/grafana/dashboards/`

---

## Performance Tuning

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_documents_user ON documents(user_id);
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_policies_status ON policies(status);
CREATE INDEX idx_logs_timestamp ON immutable_logs(timestamp);

-- Tune PostgreSQL
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '128MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
SELECT pg_reload_conf();
```

### API Server Tuning

```bash
# Increase workers based on CPU cores
uvicorn grace.api:app --workers $(nproc)

# Enable asyncio with uvloop for performance
pip install uvloop
# uvloop is automatically used by uvicorn
```

### Vector Store Optimization

```python
# FAISS optimization
import faiss

# Use GPU if available
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Use IVF index for large datasets
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

---

## Security Hardening

### SSL/TLS Setup

```bash
# Generate certificates
certbot certonly --standalone -d grace.example.com

# Configure nginx
server {
    listen 443 ssl http2;
    server_name grace.example.com;
    
    ssl_certificate /etc/letsencrypt/live/grace.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/grace.example.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Firewall Configuration

```bash
# UFW rules
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8080/tcp  # Swarm (internal only)
ufw enable
```

### Rate Limiting

Enable Redis-based rate limiting in production:

```python
# In grace/middleware/rate_limit.py
from redis import Redis

redis_client = Redis.from_url(os.getenv('REDIS_URL'))
rate_limiter = RateLimitMiddleware(
    app=app,
    redis_client=redis_client,
    default_limit=100,
    window_seconds=60
)
```

---

## Backup & Recovery

### Database Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump grace_db | gzip > /backups/grace_db_$DATE.sql.gz

# Keep last 7 days
find /backups -name "grace_db_*.sql.gz" -mtime +7 -delete
```

### Vector Store Backup

```bash
# Backup FAISS index
cp /data/vectors/grace_index.bin /backups/vectors_$(date +%Y%m%d).bin

# Backup immutable logs
tar -czf /backups/logs_$(date +%Y%m%d).tar.gz /data/immutable_logs/
```

### Restore Procedure

```bash
# Restore database
gunzip < /backups/grace_db_YYYYMMDD.sql.gz | psql grace_db

# Restore vectors
cp /backups/vectors_YYYYMMDD.bin /data/vectors/grace_index.bin

# Restart services
docker-compose restart grace-api
```

---

## Troubleshooting

### Common Issues

**1. Database Connection Error**
```bash
# Check PostgreSQL is running
systemctl status postgresql

# Test connection
psql -h localhost -U grace -d grace_db

# Check DATABASE_URL in .env
```

**2. High Memory Usage**
```bash
# Check metrics
curl http://localhost:9090/metrics | grep memory

# Adjust worker count
# Reduce: WORKERS=2
```

**3. Slow Vector Search**
```bash
# Rebuild FAISS index
python scripts/rebuild_index.py

# Or switch to GPU
export USE_GPU=true
```

---

## Support & Resources

- **Documentation**: `/docs`
- **API Reference**: `http://your-domain/api/docs`
- **GitHub**: https://github.com/yourorg/grace
- **Email**: support@grace-ai.example

---

**Deployment Status**: ✅ Ready for Production

All Grace components are fully implemented, tested, and production-ready.
