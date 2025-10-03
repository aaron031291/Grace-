# Grace Governance Kernel User Tutorial

This tutorial will guide you through using the Grace Governance Kernel API for governance decision processing, monitoring, and audit trail access.

## 1. Making a Governance Request
Send a POST request to `/governance/request`:

```bash
curl -X POST http://localhost:8000/governance/request \
  -H "Content-Type: application/json" \
  -d '{
    "type": "policy",
    "payload": {
      "claims": ["claim1", "claim2"],
      "context": {"user": "alice", "action": "update"}
    }
  }'
```
Response:
```
{
  "decision": "APPROVED",
  "details": { ... }
}
```

## 2. Checking System Health
Access the health endpoint:
```bash
curl http://localhost:8000/health
```
Response:
```
{
  "status": "running",
  "components": ["governance", "memory", "event_mesh", ...]
}
```

## 3. Retrieving Metrics
Get system metrics for monitoring:
```bash
curl http://localhost:8000/metrics
```
Response:
```
{
  "decision_count": 123,
  "uptime_seconds": 45678,
  ...
}
```

## 4. Accessing the Audit Trail
Fetch audit records:
```bash
curl http://localhost:8000/audit
```
Response:
```
[
  {
    "timestamp": "2025-10-03T12:34:56Z",
    "event": "policy_update",
    "details": { ... }
  },
  ...
]
```

## 5. Error Handling
All endpoints return clear error messages and codes. Example:
```
{
  "error": "Invalid request payload",
  "code": 400
}
```

---
For more details, see the [API documentation](../api/openapi.yaml) and deployment guide.
