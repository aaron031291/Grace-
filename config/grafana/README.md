# Grafana Dashboard Setup

## Import Dashboard

1. **Open Grafana**
   - Navigate to: http://localhost:3000

2. **Import Dashboard**
   - Click **+** → **Import**
   - Upload `grace-dashboard.json`
   - Select Prometheus data source
   - Click **Import**

## Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'grace'
    static_configs:
      - targets: ['grace-api:8000']
    metrics_path: '/api/v1/metrics/prometheus'
    scrape_interval: 10s
```

## Dashboard Panels

### 1. Event Throughput
- **Metric**: Rate of published/processed/failed events
- **Format**: Events per second
- **Use**: Monitor system load

### 2. Queue Sizes
- **Metrics**: Pending queue, DLQ size
- **Format**: Count
- **Use**: Detect backpressure, DLQ issues

### 3. Event Latency (p95)
- **Metrics**: Processing, emit, consensus latencies
- **Format**: Milliseconds
- **Use**: Performance monitoring

### 4. TTL Drops & Deduplication
- **Metrics**: Expired events, deduplicated events
- **Format**: Events per second
- **Use**: Configuration tuning

### 5. Events by Type
- **Metric**: Event counts by type
- **Format**: Pie chart
- **Use**: Traffic distribution

### 6. DLQ Events Over Time
- **Metric**: DLQ ingestion rate
- **Format**: Events per second
- **Use**: Error detection

### 7. Event Success Rate
- **Metric**: Processed / Published ratio
- **Format**: Percentage
- **Use**: Health check

### 8. Active Subscribers
- **Metric**: Number of subscribers
- **Format**: Count
- **Use**: System state

## Alerting Rules

Create alerts for:

- **High DLQ Size**: `grace_dlq_size > 100`
- **Low Success Rate**: `success_rate < 95`
- **High Latency**: `grace_latency_p95 > 1000`
- **TTL Drops**: `rate(grace_events_expired_total[5m]) > 10`

## Custom Metrics

Add custom panels by querying:
```promql
grace_events_by_type{event_type="your.event"}
grace_latency_your_operation_ms{quantile="0.99"}
```
