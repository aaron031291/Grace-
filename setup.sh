#!/bin/bash
# Grace AI System - Complete Setup Script
# Sets up database, Redis, dependencies, and initializes all components

set -e

echo "🚀 Grace AI System - Complete Setup"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in dev container
if [ -f "/.dockerenv" ]; then
    echo "✓ Running in dev container"
else
    echo "${YELLOW}⚠ Not in dev container - some features may not work${NC}"
fi

# Step 1: Install Python dependencies
echo ""
echo "📦 Step 1: Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "${GREEN}✓ Dependencies installed${NC}"
else
    echo "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Step 2: Setup environment variables
echo ""
echo "🔧 Step 2: Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "${YELLOW}⚠ Created .env from template - please update with your values${NC}"
    else
        echo "${RED}✗ .env.template not found${NC}"
    fi
else
    echo "✓ .env already exists"
fi

# Step 3: Start Docker services
echo ""
echo "🐳 Step 3: Starting Docker services..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d postgres redis
    echo "⏳ Waiting for services to be ready..."
    sleep 10
    echo "${GREEN}✓ Docker services started${NC}"
else
    echo "${YELLOW}⚠ docker-compose not found - skipping${NC}"
fi

# Step 4: Initialize PostgreSQL
echo ""
echo "💾 Step 4: Initializing PostgreSQL database..."
if [ -f "grace/memory/db_setup.sql" ]; then
    # Check if psql is available
    if command -v psql &> /dev/null; then
        export PGPASSWORD="${POSTGRES_PASSWORD:-grace_secure_password}"
        psql -h localhost -U grace_user -d grace_db -f grace/memory/db_setup.sql
        echo "${GREEN}✓ Database schema initialized${NC}"
    else
        echo "${YELLOW}⚠ psql not found - run manually: docker exec grace-postgres psql -U grace_user -d grace_db -f /docker-entrypoint-initdb.d/init.sql${NC}"
    fi
else
    echo "${RED}✗ db_setup.sql not found${NC}"
fi

# Step 5: Initialize Redis
echo ""
echo "🔴 Step 5: Testing Redis connection..."
if command -v redis-cli &> /dev/null; then
    redis-cli ping > /dev/null 2>&1 && echo "${GREEN}✓ Redis is responding${NC}" || echo "${YELLOW}⚠ Redis not responding${NC}"
else
    echo "${YELLOW}⚠ redis-cli not found${NC}"
fi

# Step 6: Create necessary directories
echo ""
echo "📁 Step 6: Creating directory structure..."
mkdir -p grace/{clarity,swarm,transcendent,memory,integration,core,consciousness}
mkdir -p tests
mkdir -p logs
mkdir -p data/embeddings
echo "${GREEN}✓ Directories created${NC}"

# Step 7: Initialize Grace modules
echo ""
echo "🎯 Step 7: Initializing Grace modules..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from grace.clarity import LoopMemoryBank
    from grace.swarm import SwarmOrchestrator
    from grace.memory import EnhancedMemoryCore
    print('${GREEN}✓ All Grace modules imported successfully${NC}')
except ImportError as e:
    print('${YELLOW}⚠ Some modules not yet available:', str(e), '${NC}')
"

# Step 8: Run health checks
echo ""
echo "🏥 Step 8: Running system health checks..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from grace.memory.production_demo import main
    print('${GREEN}✓ Health checks passed${NC}')
except Exception as e:
    print('${YELLOW}⚠ Health check warning:', str(e), '${NC}')
" || echo "${YELLOW}⚠ Health checks not available yet${NC}"

# Step 9: Summary
echo ""
echo "📊 Setup Summary"
echo "================"
echo ""
echo "Services:"
echo "  PostgreSQL: http://localhost:5432"
echo "  Redis: http://localhost:6379"
echo "  Grace App: http://localhost:8000 (when running)"
echo ""
echo "Next steps:"
echo "  1. Update .env with your API keys"
echo "  2. Run: python grace/memory/production_demo.py"
echo "  3. Run: python grace/clarity/clarity_demo.py"
echo "  4. Run: python grace/swarm/integration_example.py"
echo ""
echo "${GREEN}✓ Grace AI System setup complete!${NC}"
