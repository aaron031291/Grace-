#!/bin/bash
# Run Grace AI System API Server

set -e

echo "🚀 Starting Grace AI System API Server"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📚 Installing dependencies..."
    pip install -r requirements.txt
fi

# Export environment variables
export PYTHONPATH=/workspaces/Grace-:$PYTHONPATH
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./grace_data.db"}

echo ""
echo "✅ Environment ready"
echo "📍 Database: $DATABASE_URL"
echo ""
echo "🌐 Starting server on http://0.0.0.0:8000"
echo "📖 API Docs: http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the server
python -m grace.main
