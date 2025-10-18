.PHONY: help setup install test clean docker-up docker-down demo health

# Default target
help:
	@echo "Grace AI System - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Complete system setup"
	@echo "  make install      - Install Python dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    - Start all Docker services"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make docker-logs  - View Docker logs"
	@echo ""
	@echo "Running:"
	@echo "  make demo         - Run all demos"
	@echo "  make memory-demo  - Run memory system demo"
	@echo "  make clarity-demo - Run clarity framework demo"
	@echo "  make swarm-demo   - Run swarm intelligence demo"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-memory  - Run memory tests"
	@echo "  make coverage     - Run tests with coverage"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Clean Python cache"
	@echo "  make health       - Run health checks"
	@echo "  make organize     - Organize repository"
	@echo ""

# Setup
setup:
	@echo "🚀 Setting up Grace AI System..."
	@chmod +x setup.sh
	@./setup.sh

install:
	@echo "📦 Installing dependencies..."
	@pip install -r requirements.txt

# Docker
docker-up:
	@echo "🐳 Starting Docker services..."
	@docker-compose up -d
	@echo "⏳ Waiting for services..."
	@sleep 5
	@docker-compose ps

docker-down:
	@echo "🛑 Stopping Docker services..."
	@docker-compose down

docker-logs:
	@docker-compose logs -f

# Demos
demo: memory-demo clarity-demo swarm-demo

memory-demo:
	@echo "💾 Running Memory System Demo..."
	@python grace/memory/production_demo.py

clarity-demo:
	@echo "🔍 Running Clarity Framework Demo..."
	@python grace/clarity/clarity_demo.py

swarm-demo:
	@echo "🤝 Running Swarm Intelligence Demo..."
	@python grace/swarm/integration_example.py

# Testing
test:
	@echo "🧪 Running tests..."
	@pytest tests/ -v

test-memory:
	@echo "🧪 Running memory tests..."
	@pytest grace/memory/integration_test.py -v

coverage:
	@echo "📊 Running tests with coverage..."
	@pytest --cov=grace tests/ --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Maintenance
clean:
	@echo "🗑️  Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✓ Clean complete"

health:
	@echo "🏥 Running health checks..."
	@python -c "from grace.memory import EnhancedMemoryCore; from grace.integration import AVNReporter; avn = AVNReporter(); print(avn.get_system_health())"

organize:
	@echo "📁 Organizing repository..."
	@chmod +x organize_repo.sh
	@./organize_repo.sh

# Development
dev-setup:
	@echo "🔧 Setting up development environment..."
	@pip install -r requirements.txt
	@pre-commit install || echo "pre-commit not available"

lint:
	@echo "🔍 Linting code..."
	@ruff check grace/ || echo "ruff not available"
	@black --check grace/ || echo "black not available"

format:
	@echo "✨ Formatting code..."
	@black grace/
	@ruff check --fix grace/
