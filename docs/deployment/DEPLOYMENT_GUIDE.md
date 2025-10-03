# Grace Governance Kernel Production Deployment Guide

This guide provides step-by-step instructions for deploying the Grace Governance Kernel in a production environment.

## Prerequisites
- Docker and Docker Compose installed
- PostgreSQL and Redis available (local or cloud)
- Access to production environment (VM, cloud, or on-prem)

## 1. Clone the Repository
```bash
git clone https://github.com/aaron031291/Grace-.git
cd Grace-
```

## 2. Configure Environment Variables
Create a `.env` file in the project root:
```
POSTGRES_URL=postgresql://user:password@host:5432/grace_db
REDIS_URL=redis://host:6379/0
ENVIRONMENT=production
```

## 3. Build Docker Images
```bash
docker-compose build
```

## 4. Run Database Migrations
If using Alembic:
```bash
docker-compose run --rm backend alembic upgrade head
```

## 5. Start All Services
```bash
docker-compose up -d
```

## 6. Verify Health and Metrics
- Access health endpoint: `http://localhost:8000/health`
- Access metrics endpoint: `http://localhost:8000/metrics`

## 7. Monitoring Setup
- Prometheus: Configure to scrape `/metrics` endpoint
- Grafana: Import dashboards for governance, memory, and system health

## 8. Troubleshooting
- Check logs: `docker-compose logs`
- Restart services: `docker-compose restart <service>`
- Database issues: Ensure PostgreSQL and Redis are reachable

## 9. Scaling and Updates
- Scale services: `docker-compose up --scale backend=3`
- Update images: `docker-compose pull && docker-compose up -d`

## 10. Security Best Practices
- Use strong passwords for databases
- Restrict network access to sensitive endpoints
- Enable TLS/SSL for all external connections

---
For more details, see the [API documentation](../api/openapi.yaml) and user tutorials.
