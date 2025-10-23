# Grace Operational Runbook

## Quick Start

### Development

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f grace-api

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Production

```bash
# Set environment variables
cp .env.production.example .env
nano .env  # Configure

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

## Common Operations

### Scaling Services

```bash
# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale grace-api=5

# Check scaled services
docker-compose -f docker-compose.prod.yml ps
```

### Database Operations

#### Backup

```bash
# Manual backup
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U grace_prod grace_prod | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore from backup
gunzip < backup_20240115.sql.gz | \
  docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U grace_prod grace_prod
```

#### Migrations

```bash
# Apply migrations
docker-compose -f docker-compose.prod.yml exec grace-api \
  python scripts/apply_migrations.py

# Create new migration
docker-compose -f docker-compose.prod.yml exec grace-api \
  python scripts/create_migration.py "add_new_table"
```

### Log Management

#### View Logs

```bash
# All services
docker-compose -f docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker-compose.prod.yml logs -f grace-api

# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 grace-api
```

#### Export Logs

```bash
# Export to file
docker-compose -f docker-compose.prod.yml logs grace-api > api_logs.txt

# Search logs
docker-compose -f docker-compose.prod.yml logs grace-api | grep ERROR
```

### Health Checks

```bash
# Check service health
curl http://localhost:8000/health

# Check all services
docker-compose -f docker-compose.prod.yml ps

# Detailed health
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/monitoring/health
```

### Updates and Rollbacks

#### Zero-Downtime Update

```bash
# Pull new image
docker-compose -f docker-compose.prod.yml pull grace-api

# Update with rolling restart
docker-compose -f docker-compose.prod.yml up -d --no-deps --build grace-api
```

#### Rollback

```bash
# Tag before update
docker tag grace:latest grace:backup-$(date +%Y%m%d)

# Rollback
docker-compose -f docker-compose.prod.yml down grace-api
docker tag grace:backup-20240115 grace:latest
docker-compose -f docker-compose.prod.yml up -d grace-api
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs grace-api

# Check environment
docker-compose -f docker-compose.prod.yml exec grace-api env

# Restart service
docker-compose -f docker-compose.prod.yml restart grace-api
```

### Database Connection Issues

```bash
# Test database connection
docker-compose -f docker-compose.prod.yml exec postgres \
  psql -U grace_prod -c "SELECT 1;"

# Check connection string
docker-compose -f docker-compose.prod.yml exec grace-api \
  python -c "import os; print(os.getenv('DATABASE_URL'))"
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Restart specific service
docker-compose -f docker-compose.prod.yml restart grace-api

# Scale down if needed
docker-compose -f docker-compose.prod.yml up -d --scale grace-api=2
```

### Rate Limiting Issues

```bash
# Check rate limit status
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/monitoring/health

# Reset user limits (via API)
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/api/v1/admin/users/{user_id}/reset-limits
```

## Monitoring

### Prometheus Queries

```promql
# Event throughput
rate(grace_events_published_total[5m])

# Error rate
rate(grace_events_failed_total[5m]) / rate(grace_events_published_total[5m])

# P95 latency
histogram_quantile(0.95, grace_latency_event_processing_ms_bucket)

# Queue size
grace_pending_queue_size
```

### Grafana Dashboards

- Main Dashboard: http://localhost:3000/d/grace-main
- KPI Dashboard: http://localhost:3000/d/grace-kpis
- System Health: http://localhost:3000/d/grace-health

### Alerts

Check active alerts:

```bash
curl http://localhost:9090/api/v1/alerts
```

## Maintenance Windows

### Planned Maintenance

1. **Notify Users** (24-48 hours advance)
2. **Backup Database**
   ```bash
   bash scripts/backup.sh
   ```
3. **Enable Maintenance Mode**
   ```bash
   docker-compose -f docker-compose.prod.yml exec nginx \
     cp /etc/nginx/maintenance.conf /etc/nginx/nginx.conf
   docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload
   ```
4. **Perform Maintenance**
5. **Disable Maintenance Mode**
   ```bash
   docker-compose -f docker-compose.prod.yml exec nginx \
     cp /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf
   docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload
   ```

## Disaster Recovery

### Complete System Failure

1. **Restore from Backup**
   ```bash
   # Start database
   docker-compose -f docker-compose.prod.yml up -d postgres
   
   # Restore backup
   gunzip < latest_backup.sql.gz | \
     docker-compose -f docker-compose.prod.yml exec -T postgres \
     psql -U grace_prod grace_prod
   ```

2. **Start Services**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Verify**
   ```bash
   curl http://localhost:8000/health
   ```

### Data Corruption

1. **Stop affected services**
2. **Restore from last known good backup**
3. **Replay event logs if available**
4. **Verify data integrity**

## Security Incidents

### Compromised API Key

1. **Revoke key immediately**
   ```bash
   curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://localhost:8000/api/v1/admin/keys/{key_id}
   ```

2. **Review access logs**
3. **Generate new key**
4. **Update dependent services**

### Suspected Breach

1. **Enable audit mode**
2. **Isolate affected systems**
3. **Review immutable logs**
4. **Follow incident response plan**

## Contacts

- **On-Call Engineer:** [Contact info]
- **Database Admin:** [Contact info]
- **Security Team:** [Contact info]
- **Escalation:** [Contact info]
