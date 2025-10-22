# Grace API Reference

## Table of Contents

1. [Authentication](#authentication)
2. [Events API](#events-api)
3. [Kernels API](#kernels-api)
4. [Memory API](#memory-api)
5. [Metrics & Monitoring](#metrics--monitoring)
6. [Security & RBAC](#security--rbac)
7. [Error Handling](#error-handling)

## Authentication

All API requests require authentication via JWT token.

### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using Token

Include token in all subsequent requests:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

## Events API

### Emit Event

```http
POST /api/v1/events/emit
Authorization: Bearer {token}
Content-Type: application/json

{
  "event_type": "user.action",
  "source": "web_app",
  "payload": {
    "action": "click",
    "target": "button_1"
  },
  "priority": "normal",
  "ttl_seconds": 300
}
```

**Response:**
```json
{
  "success": true,
  "event_id": "evt_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Query Events

```http
GET /api/v1/events?event_type=user.action&limit=10
Authorization: Bearer {token}
```

**Response:**
```json
{
  "events": [
    {
      "event_id": "evt_abc123",
      "event_type": "user.action",
      "timestamp": "2024-01-15T10:30:00Z",
      "source": "web_app",
      "payload": {...}
    }
  ],
  "total": 42,
  "page": 1
}
```

## Kernels API

### List Kernels

```http
GET /api/v1/kernels
Authorization: Bearer {token}
```

**Response:**
```json
{
  "kernels": ["multi_os", "mldl", "resilience"]
}
```

### Get Kernel Status

```http
GET /api/v1/kernels/{kernel_name}/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "name": "multi_os",
  "running": true,
  "uptime_seconds": 3600,
  "events_processed": 1234,
  "last_activity": "2024-01-15T10:30:00Z"
}
```

### Control Kernel

**Required Permission:** `manage:kernels`

```http
POST /api/v1/kernels/{kernel_name}/control
Authorization: Bearer {token}
Content-Type: application/json

{
  "action": "restart"
}
```

**Response:**
```json
{
  "status": "restarted",
  "kernel": "multi_os"
}
```

### Get Kernel Health

```http
GET /api/v1/kernels/{kernel_name}/health
Authorization: Bearer {token}
```

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "errors": 0,
  "warnings": 2,
  "mcp_stats": {
    "messages_sent": 100,
    "messages_received": 95,
    "validation_failures": 0
  }
}
```

## Memory API

### Write to Memory

```http
POST /api/v1/memory/write
Authorization: Bearer {token}
Content-Type: application/json

{
  "key": "user_preferences",
  "value": {
    "theme": "dark",
    "language": "en"
  },
  "ttl_seconds": 3600,
  "trust_attestation": true
}
```

**Response:**
```json
{
  "success": true,
  "key": "user_preferences",
  "layers_written": {
    "lightning": true,
    "fusion": true,
    "vector": true,
    "trust": true,
    "immutable_log": true,
    "trigger": true
  }
}
```

### Read from Memory

```http
POST /api/v1/memory/read
Authorization: Bearer {token}
Content-Type: application/json

{
  "key": "user_preferences",
  "use_cache": true
}
```

**Response:**
```json
{
  "key": "user_preferences",
  "value": {
    "theme": "dark",
    "language": "en"
  },
  "found": true,
  "source": "lightning"
}
```

### Get Memory Stats

```http
GET /api/v1/memory/stats
Authorization: Bearer {token}
```

**Response:**
```json
{
  "writes_total": 1000,
  "writes_failed": 5,
  "cache_hits": 850,
  "cache_misses": 150,
  "cache_hit_rate": 0.85
}
```

## Metrics & Monitoring

### Get Metrics (JSON)

```http
GET /api/v1/metrics
Authorization: Bearer {token}
```

**Response:**
```json
{
  "grace_events_published_total": 10000,
  "grace_events_processed_total": 9800,
  "grace_events_failed_total": 200,
  "grace_pending_queue_size": 10,
  "grace_dlq_size": 5,
  "grace_latency_percentiles": {
    "event_processing": {
      "p50": 45.2,
      "p95": 85.7,
      "p99": 120.3
    }
  }
}
```

### Get Metrics (Prometheus)

```http
GET /api/v1/metrics/prometheus
```

**Response:** (text/plain)
```
# HELP grace_events_published_total Total events published
# TYPE grace_events_published_total counter
grace_events_published_total 10000

# HELP grace_pending_queue_size Current pending queue size
# TYPE grace_pending_queue_size gauge
grace_pending_queue_size 10
```

### Get KPIs

```http
GET /api/v1/monitoring/kpis
Authorization: Bearer {token}
```

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "overall_health": "healthy",
  "health_percentage": 95.0,
  "kpis": {
    "event_success_rate": {
      "value": 98.0,
      "target": 95.0,
      "met": true
    }
  }
}
```

### Health Dashboard

```http
GET /api/v1/monitoring/health
```

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "overall_health": "healthy",
  "health_percentage": 95.0,
  "kpis": {...},
  "metrics": {...},
  "event_bus": {...},
  "trigger_mesh": {...}
}
```

## Security & RBAC

### Get User Permissions

```http
GET /api/v1/auth/me/permissions
Authorization: Bearer {token}
```

**Response:**
```json
{
  "user_id": "user123",
  "roles": ["user", "operator"],
  "permissions": [
    "read:events",
    "write:events",
    "manage:kernels"
  ]
}
```

### Assign Role (Admin Only)

**Required Permission:** `manage:users`

```http
POST /api/v1/admin/users/{user_id}/roles
Authorization: Bearer {token}
Content-Type: application/json

{
  "role": "operator"
}
```

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "details": {
    "field": "specific_field",
    "constraint": "validation_rule"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Missing/invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### Rate Limiting

Rate limit headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705315800
```

When rate limited:

```json
{
  "error": "Rate limit exceeded",
  "message": "100 requests per 60s. Retry after 45.0s",
  "retry_after": 45.0
}
```

## WebSocket API

### Connect

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/events?token=YOUR_JWT_TOKEN');
```

### Subscribe to Events

```json
{
  "action": "subscribe",
  "event_types": ["kernel.heartbeat", "system.error"]
}
```

### Receive Events

```json
{
  "event_id": "evt_xyz",
  "event_type": "kernel.heartbeat",
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {...}
}
```
