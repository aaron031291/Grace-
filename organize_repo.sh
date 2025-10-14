#!/bin/bash
# Grace Repository Organization Script
# This script tidies up the repository structure

set -e

echo "🧹 Starting Grace Repository Organization..."

# Create folder structure
echo "📁 Creating organized folder structure..."
mkdir -p demos
mkdir -p documentation
mkdir -p database
mkdir -p scripts
mkdir -p config/docker
mkdir -p config/deployment
mkdir -p config/alembic

# Move demo files
echo "📦 Moving demo files..."
mv demo_*.py demos/ 2>/dev/null || true
mv *_demo.html demos/ 2>/dev/null || true
mv grace_enhanced_interface.html demos/ 2>/dev/null || true
mv demo_and_tests demos/ 2>/dev/null || true

# Move documentation
echo "📚 Moving documentation files..."
mv *_ARCHITECTURE.md documentation/ 2>/dev/null || true
mv *_COMPLETE.md documentation/ 2>/dev/null || true
mv *_SUMMARY.md documentation/ 2>/dev/null || true
mv *_STATUS.md documentation/ 2>/dev/null || true
mv *_CHECKLIST.md documentation/ 2>/dev/null || true
mv *_REFERENCE.md documentation/ 2>/dev/null || true
mv *_README.md documentation/ 2>/dev/null || true
mv DATABASE_SCHEMA.md documentation/ 2>/dev/null || true
mv PR_DESCRIPTION.md documentation/ 2>/dev/null || true
mv GRACE_KNOWLEDGE_BASE documentation/ 2>/dev/null || true

# Move database files
echo "💾 Moving database files..."
mv *.db database/ 2>/dev/null || true
mv *.sqlite3 database/ 2>/dev/null || true
mv init_*.sql database/ 2>/dev/null || true
mv build_all_tables.py database/ 2>/dev/null || true
mv verify_database.py database/ 2>/dev/null || true
mv init_db database/ 2>/dev/null || true

# Move test files
echo "🧪 Moving test files..."
mv test_*.py tests/ 2>/dev/null || true
mv test_run_log.txt tests/ 2>/dev/null || true
mv test_reports tests/ 2>/dev/null || true
mv conftest.py tests/ 2>/dev/null || true

# Move configuration files
echo "⚙️  Moving configuration files..."
mv docker-compose*.yml config/docker/ 2>/dev/null || true
mv Dockerfile config/docker/ 2>/dev/null || true
mv docker-entrypoint.sh config/docker/ 2>/dev/null || true
mv alembic.ini config/alembic/ 2>/dev/null || true
mv alembic config/alembic/ 2>/dev/null || true
mv pytest.ini config/ 2>/dev/null || true
mv .env.template config/ 2>/dev/null || true

# Move migration files
echo "🔄 Moving migration files..."
mv migrations database/migrations 2>/dev/null || true

# Move utility scripts
echo "🔧 Moving utility scripts..."
mv system_check.py scripts/ 2>/dev/null || true
mv watchdog.py scripts/ 2>/dev/null || true
mv grace_*_runner.py scripts/ 2>/dev/null || true
mv grace_loop_engine.py scripts/ 2>/dev/null || true
mv grace_*_analysis.py scripts/ 2>/dev/null || true
mv governance_examples.py scripts/ 2>/dev/null || true
mv *_server.py scripts/ 2>/dev/null || true

# Move contract/policy files
echo "📜 Moving contract and policy files..."
mv contracts scripts/contracts 2>/dev/null || true
mv policies scripts/policies 2>/dev/null || true
mv grace_build_policy_contract.yaml scripts/ 2>/dev/null || true

# Clean up Python cache
echo "🗑️  Cleaning up Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

echo "✅ Repository organization complete!"
echo ""
echo "📊 New structure:"
echo "  demos/          - All demo files and examples"
echo "  documentation/  - All markdown documentation"
echo "  database/       - Database files, schemas, migrations"
echo "  scripts/        - Utility scripts and tools"
echo "  config/         - Configuration files (Docker, pytest, etc.)"
echo "  tests/          - All test files"
echo "  grace/          - Core application code"
echo ""
echo "⚠️  Next steps:"
echo "  1. Update import paths in affected files"
echo "  2. Update docker-compose paths"
echo "  3. Update documentation references"
echo "  4. Test the application"
