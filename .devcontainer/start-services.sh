#!/bin/bash
# Start services on codespace start

echo "🔄 Starting Grace services..."

# Wait for database
echo "⏳ Waiting for PostgreSQL..."
until pg_isready -h postgres -p 5432 -U grace; do
  sleep 1
done

echo "⏳ Waiting for Redis..."
until redis-cli -h redis ping 2>/dev/null; do
  sleep 1
done

echo "✅ All services ready!"
