#!/bin/bash
# Auto-rollback on SLO regression (triggered by Prometheus alert)
# Usage: ./scripts/ci_auto_rollback.sh <rollback_artifact>
set -e

ROLLBACK_ARTIFACT="$1"
if [ -z "$ROLLBACK_ARTIFACT" ]; then
  echo "Usage: $0 <rollback_artifact>"
  exit 1
fi

echo "[Grace CI] SLO regression detected. Rolling back to $ROLLBACK_ARTIFACT..."
# Example: docker image rollback

docker pull "$ROLLBACK_ARTIFACT"
docker tag "$ROLLBACK_ARTIFACT" grace_api:latest
docker-compose up -d grace-api

echo "[Grace CI] Rollback complete. System running $ROLLBACK_ARTIFACT."
