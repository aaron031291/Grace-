#!/bin/bash

# Docker entrypoint script to handle different service modes

set -e

# Default SERVICE_MODE to api if not set
SERVICE_MODE=${SERVICE_MODE:-api}

echo "Starting Grace in $SERVICE_MODE mode..."

case "$SERVICE_MODE" in
  api)
    echo "Starting Grace API service..."
    exec python -m grace.api.api_service
    ;;
  worker)
    echo "Starting Grace Worker service with queues: $WORKER_QUEUES"
    exec python -m grace.worker.worker_service
    ;;
  orchestrator)
    echo "Starting Grace Orchestrator service..."
    exec python -m grace.orchestration.orchestration_service
    ;;
  *)
    echo "Unknown SERVICE_MODE: $SERVICE_MODE"
    echo "Valid modes: api, worker, orchestrator"
    exit 1
    ;;
esac