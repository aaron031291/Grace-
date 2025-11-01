#!/bin/bash
# Simple Grace Startup - Get Frontend and Backend Working

echo "🚀 Starting Grace - Simple Mode"
echo "================================"

# Check if in correct directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Run from Grace root directory"
    exit 1
fi

# Start PostgreSQL and Redis with Docker
echo "📦 Starting infrastructure..."
docker-compose -f docker-compose-working.yml up -d postgres redis

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 5

# Start backend
echo "🔧 Starting backend on port 8000..."
cd backend
export PYTHONPATH=..
export DATABASE_URL="postgresql://grace:grace_dev_password@localhost:5432/grace_dev"
export REDIS_URL="redis://localhost:6379/0"
export DEBUG=true
export JWT_SECRET_KEY="dev-secret-key-change-in-production-minimum-32-characters-long"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

sleep 3

# Start frontend
echo "🎨 Starting frontend on port 5173..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
cd ..

sleep 3

echo ""
echo "================================"
echo "✅ Grace is starting!"
echo "================================"
echo ""
echo "🌐 Access points:"
echo "   Backend API:  http://localhost:8000"
echo "   API Docs:     http://localhost:8000/api/docs"
echo "   Frontend:     http://localhost:5173"
echo "   Health:       http://localhost:8000/api/health"
echo ""
echo "📊 Infrastructure:"
echo "   PostgreSQL:   localhost:5432"
echo "   Redis:        localhost:6379"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
wait $BACKEND_PID $FRONTEND_PID
