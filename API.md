# Grace API Documentation

This document describes the REST API and WebSocket interfaces provided by the Grace Governance Service.

## Base URL

- **Local Development**: `http://localhost:8080`
- **Production**: Your deployed URL

## Authentication

### API Key Authentication (Optional)

Include the API key in the header:

```http
X-Grace-API-Key: your-api-key-here
```

### CORS Support

The API supports Cross-Origin Resource Sharing (CORS) for web applications.

## REST API Endpoints

### Root Endpoint

#### GET /

Get basic service information.

**Response:**
```json
{
  "status": "success",
  "message": "Grace Governance Service is running",
  "data": {
    "version": "1.0.0",
    "docs": "/docs",
    "health": "/health/status",
    "metrics": "/metrics"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Health and Monitoring

### GET /health/status

Get detailed health status of all components.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "components": {
    "governance_kernel": "healthy",
    "ingress_kernel": "healthy",
    "orchestration_service": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "vector_db": "healthy"
  },
  "metrics": {
    "cpu_usage_percent": 25.5,
    "memory_usage_percent": 45.2,
    "memory_available_bytes": 2147483648,
    "disk_usage_percent": 15.8,
    "disk_free_bytes": 85899345920,
    "load_average": [0.5, 0.3, 0.2]
  }
}
```

### GET /health/live

Simple liveness probe for container orchestrators.

**Response:**
```json
{
  "status": "alive",
  "timestamp": 1704110400.123
}
```

### GET /health/ready

Readiness probe indicating if the service is ready to accept requests.

**Response:**
```json
{
  "status": "ready",
  "timestamp": 1704110400.123
}
```

### GET /metrics

Prometheus-compatible metrics endpoint.

**Response:** Plain text Prometheus metrics format

## Governance API

### POST /api/v1/governance/validate

Validate an action against constitutional governance rules.

**Request Body:**
```json
{
  "action": "data_access",
  "context": {
    "user": "alice",
    "resource": "customer_data",
    "purpose": "analytics",
    "duration": "1_hour"
  },
  "user_id": "user_12345",
  "priority": "normal"
}
```

**Response:**
```json
{
  "approved": true,
  "decision_id": "dec_67890",
  "compliance_score": 0.95,
  "violations": [],
  "recommendations": [
    "Consider adding additional audit logging for this action"
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Response (with violations):**
```json
{
  "approved": false,
  "decision_id": "dec_67891",
  "compliance_score": 0.45,
  "violations": [
    {
      "principle": "transparency",
      "severity": "high",
      "description": "Action lacks sufficient transparency documentation",
      "recommendation": "Add detailed logging and audit trail"
    }
  ],
  "recommendations": [
    "Add transparency documentation",
    "Implement additional audit controls",
    "Consider lower-privilege alternatives"
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /api/v1/governance/status/{decision_id}

Get the status of a specific governance decision.

**Response:**
```json
{
  "status": "success",
  "message": "Decision status retrieved",
  "data": {
    "decision_id": "dec_67890",
    "status": "completed",
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

### GET /api/v1/governance/pending

Get list of pending governance decisions.

**Response:**
```json
{
  "status": "success",
  "message": "Pending decisions retrieved",
  "data": {
    "pending_count": 2,
    "decisions": [
      {
        "decision_id": "dec_67892",
        "action": "system_modification",
        "created_at": "2024-01-01T11:30:00Z",
        "priority": "high"
      }
    ]
  }
}
```

## Data Ingestion API

### POST /api/v1/ingest/data

Ingest data through the Grace system.

**Request Body:**
```json
{
  "source_id": "news_feed_01",
  "data": {
    "title": "Breaking News",
    "content": "News article content...",
    "published_at": "2024-01-01T10:00:00Z"
  },
  "metadata": {
    "content_type": "article",
    "language": "en",
    "classification": "public"
  },
  "priority": "normal"
}
```

**Response:**
```json
{
  "event_id": "evt_123456",
  "status": "processed",
  "trust_score": 0.85,
  "processing_time_ms": 150
}
```

### POST /api/v1/ingest/source/register

Register a new data source.

**Request Body:**
```json
{
  "source_id": "api_feed_01",
  "kind": "http",
  "uri": "https://api.example.com/data",
  "auth_mode": "bearer",
  "parser": "json",
  "target_contract": "contract:data.v1",
  "pii_policy": "mask",
  "governance_label": "internal"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Source registered successfully",
  "data": {
    "source_id": "api_feed_01"
  }
}
```

### GET /api/v1/ingest/source/{source_id}/status

Get the status of a registered data source.

**Response:**
```json
{
  "status": "success",
  "message": "Source status retrieved",
  "data": {
    "source_id": "api_feed_01",
    "status": "active",
    "last_ingestion": "2024-01-01T11:45:00Z",
    "total_events": 1250
  }
}
```

## Event Streaming API

### GET /api/v1/events/stream/status

Get the status of the event streaming system.

**Response:**
```json
{
  "status": "success",
  "message": "Event stream status retrieved",
  "data": {
    "active_connections": 5,
    "stream_enabled": true,
    "supported_events": [
      "governance.decision",
      "governance.violation",
      "ingress.data_received",
      "ingress.validation_complete",
      "orchestration.task_started",
      "orchestration.task_completed",
      "health.component_status_changed"
    ]
  }
}
```

### GET /api/v1/events/history

Get recent events from the system.

**Query Parameters:**
- `limit` (optional): Maximum number of events to return (default: 50)
- `event_type` (optional): Filter by specific event type

**Response:**
```json
{
  "status": "success",
  "message": "Retrieved 10 recent events",
  "data": {
    "events": [
      {
        "event_id": "evt_001",
        "event_type": "governance.decision",
        "timestamp": "2024-01-01T12:00:00Z",
        "data": {
          "decision_id": "dec_001",
          "approved": true,
          "compliance_score": 0.95
        }
      }
    ],
    "total": 10,
    "filtered_by": null
  }
}
```

### POST /api/v1/events/emit

Emit a custom event to all connected WebSocket clients.

**Request Body:**
```json
{
  "type": "event",
  "event_type": "custom.notification",
  "data": {
    "message": "System maintenance scheduled",
    "scheduled_time": "2024-01-02T02:00:00Z"
  },
  "correlation_id": "maint_001"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Event emitted successfully",
  "data": {
    "event_type": "custom.notification",
    "timestamp": "2024-01-01T12:00:00Z",
    "broadcast_to": 5
  }
}
```

## WebSocket Interface

### Connection

**Endpoint:** `ws://localhost:8080/ws/events`

### Message Format

All WebSocket messages follow this format:

```json
{
  "type": "event",
  "event_type": "governance.decision",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "decision_id": "dec_123",
    "approved": true,
    "compliance_score": 0.95
  }
}
```

### Event Types

#### Governance Events

- **`governance.decision`**: New governance decision made
- **`governance.violation`**: Constitutional violation detected
- **`governance.override`**: Manual override applied

#### Ingress Events

- **`ingress.data_received`**: New data received for processing
- **`ingress.validation_complete`**: Data validation completed
- **`ingress.trust_scored`**: Trust score calculated

#### Orchestration Events

- **`orchestration.task_started`**: New task started
- **`orchestration.task_completed`**: Task completed
- **`orchestration.error`**: Orchestration error occurred

#### Health Events

- **`health.component_status_changed`**: Component health status changed
- **`health.threshold_exceeded`**: Monitoring threshold exceeded

### Client Example

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/events');

ws.onopen = function() {
    console.log('Connected to Grace events');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received event:', message);
    
    switch(message.event_type) {
        case 'governance.decision':
            handleGovernanceDecision(message.data);
            break;
        case 'ingress.data_received':
            handleDataIngestion(message.data);
            break;
        // Handle other event types...
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};

ws.onclose = function() {
    console.log('Disconnected from Grace events');
    // Implement reconnection logic
};
```

## Error Handling

### Standard Error Response

```json
{
  "status": "error",
  "message": "Detailed error message",
  "error_code": "GRACE_001",
  "details": {
    "field": "Invalid field value",
    "suggestion": "Use valid values: x, y, z"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### HTTP Status Codes

- **200**: Success
- **201**: Created
- **400**: Bad Request (invalid input)
- **401**: Unauthorized (missing/invalid API key)
- **403**: Forbidden (governance denied)
- **404**: Not Found
- **429**: Too Many Requests (rate limited)
- **500**: Internal Server Error
- **503**: Service Unavailable (component not ready)

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default limit**: 100 requests per minute per IP
- **Governance validation**: 50 requests per minute per user
- **Data ingestion**: 200 requests per minute per source

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## API Versioning

The API uses URL-based versioning:
- Current version: `v1`
- Future versions: `v2`, etc.

Backward compatibility is maintained within major versions.

## Interactive Documentation

Visit `/docs` on your Grace instance for interactive OpenAPI documentation with:
- Try-it-out functionality
- Request/response examples
- Schema documentation
- Authentication testing

## Client Libraries

Official client libraries (planned):
- **Python**: `grace-client`
- **JavaScript/TypeScript**: `@grace/client`
- **Go**: `go-grace-client`
- **Java**: `grace-java-client`

## Examples

### Complete Governance Workflow

```bash
# 1. Check system health
curl http://localhost:8080/health/status

# 2. Validate a governance action
curl -X POST http://localhost:8080/api/v1/governance/validate \
  -H "Content-Type: application/json" \
  -d '{
    "action": "user_data_access",
    "context": {
      "user": "analyst_01",
      "data_type": "customer_analytics",
      "purpose": "quarterly_report"
    },
    "user_id": "usr_123",
    "priority": "normal"
  }'

# 3. Check decision status
curl http://localhost:8080/api/v1/governance/status/dec_67890
```

### Data Ingestion Workflow

```bash
# 1. Register a data source
curl -X POST http://localhost:8080/api/v1/ingest/source/register \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "crm_system",
    "kind": "database",
    "parser": "json"
  }'

# 2. Ingest data
curl -X POST http://localhost:8080/api/v1/ingest/data \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "crm_system",
    "data": {"customer_id": "12345", "action": "purchase"},
    "priority": "normal"
  }'
```

This API documentation covers all major endpoints and functionality of the Grace Governance Service. For more detailed information about specific features, refer to the governance and ingress kernel documentation.