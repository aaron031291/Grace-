#!/bin/bash
# Fire-Drill Alert Script for Grace Monitoring
# Simulates SLO breach to test alerting and on-call response

PROMETHEUS_URL="http://localhost:9090"
ALERT_NAME="APIAvailabilityLow"

# Simulate API downtime by pushing a custom metric (if possible)
# Or use Prometheus API to fire an alert

curl -X POST "$PROMETHEUS_URL/-/reload"

# Optionally, send a test alert via Alertmanager
# curl -XPOST -d '{"alerts":[{"labels":{"alertname":"FireDrillTest"}}]}' http://localhost:9093/api/v1/alerts

echo "Fire-drill alert triggered. Check on-call response and alerting integrations."
