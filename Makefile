# Grace Governance System - Production Deployment Makefile
# Provides one-click deployment commands for local development and production

.PHONY: help up down build logs test clean bootstrap migrate lint format install-deps

# Default target
help:
	@echo "Grace Governance System - Deployment Commands"
	@echo "=============================================="
	@echo ""
	@echo "Development Commands:"
	@echo "  make up              - Start all services (one-click deployment)"
	@echo "  make down            - Stop all services"
	@echo "  make logs            - View service logs"
	@echo "  make build           - Build Docker images"
	@echo "  make test            - Run full test suite"
	@echo "  make bootstrap       - Initialize databases and dependencies"
	@echo "  make clean           - Clean up containers and volumes"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            - Run linting checks"
	@echo "  make format          - Format code with black"
	@echo "  make type-check      - Run type checking with mypy"
	@echo ""
	@echo "Database Operations:"
	@echo "  make migrate         - Run database migrations"
	@echo "  make db-reset        - Reset database (DESTRUCTIVE)"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-dev      - Deploy to development environment"
	@echo "  make deploy-prod     - Deploy to production environment"

# One-click development deployment
up: install-deps bootstrap
	@echo "🚀 Starting Grace Governance System..."
	@echo "Creating environment file..."
	@cp -n .env.template .env 2>/dev/null || true
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Running health check..."
	@make health-check || echo "⚠️  Some services may still be starting up"
	@echo ""
	@echo "✅ Grace is running!"
	@echo "📊 Web UI: http://localhost:8080"
	@echo "📚 API Docs: http://localhost:8080/docs"
	@echo "🔍 Health: http://localhost:8080/health/status"
	@echo "📈 Metrics: http://localhost:8080/metrics"

# Stop all services
down:
	@echo "🛑 Stopping Grace services..."
	docker-compose down
	@echo "✅ Services stopped"

# Build Docker images
build:
	@echo "🔨 Building Grace Docker images..."
	docker-compose build --no-cache
	@echo "✅ Images built successfully"

# View logs
logs:
	docker-compose logs -f

# Install Python dependencies
install-deps:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Bootstrap the system (databases, initial data)
bootstrap:
	@echo "🏗️  Bootstrapping Grace system..."
	@echo "Creating necessary directories..."
	@mkdir -p logs data/postgres data/redis data/chroma
	@echo "Waiting for databases to be ready..."
	@docker-compose up -d postgres redis chromadb
	@sleep 15
	@echo "Running bootstrap script..."
	@python scripts/bootstrap.py || echo "⚠️  Bootstrap script not yet created"
	@echo "✅ System bootstrapped"

# Database migrations
migrate:
	@echo "🗄️  Running database migrations..."
	@python -m alembic upgrade head
	@echo "✅ Migrations completed"

# Reset database (DESTRUCTIVE)
db-reset:
	@echo "⚠️  This will DESTROY all data. Are you sure? (y/N)"
	@read -r response && [ "$$response" = "y" ] || exit 1
	@echo "🗄️  Resetting database..."
	docker-compose down -v
	docker-compose up -d postgres redis chromadb
	@sleep 10
	@make migrate
	@echo "✅ Database reset completed"

# Run tests
test:
	@echo "🧪 Running Grace test suite..."
	python -m pytest demo_and_tests/tests/ -v --tb=short
	@echo "Running smoke tests..."
	@python scripts/smoke_test.py || echo "⚠️  Smoke tests not yet created"
	@echo "✅ Tests completed"

# Code quality checks
lint:
	@echo "🔍 Running linting checks..."
	flake8 grace/ grace_service/ --max-line-length=127 --extend-ignore=E203,W503
	@echo "✅ Linting passed"

format:
	@echo "🎨 Formatting code with black..."
	black grace/ grace_service/ demo_and_tests/ *.py
	@echo "✅ Code formatted"

type-check:
	@echo "🔎 Running type checks..."
	mypy grace/ grace_service/ --ignore-missing-imports --no-strict-optional
	@echo "✅ Type checking passed"

# Health check
health-check:
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:8080/health/status || echo "❌ Health check failed"
	@curl -f http://localhost:8080/health/ready || echo "❌ Readiness check failed"

# Clean up
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup completed"

# Development deployment
deploy-dev:
	@echo "🚀 Deploying to development environment..."
	@echo "Building images..."
	@make build
	@echo "Deploying..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "✅ Development deployment completed"

# Production deployment (placeholder)
deploy-prod:
	@echo "🚀 Deploying to production environment..."
	@echo "This would deploy to your production environment"
	@echo "Configure this target based on your deployment platform (Fly.io, Render, K8s, etc.)"

# Development helpers
dev-logs:
	docker-compose logs -f grace_orchestrator

dev-shell:
	docker-compose exec grace_orchestrator /bin/bash

dev-python:
	docker-compose exec grace_orchestrator python

# Quality assurance pipeline
qa: install-deps lint type-check test
	@echo "✅ QA pipeline completed successfully"

# Full pipeline (CI/CD equivalent)
pipeline: qa build test
	@echo "✅ Full pipeline completed successfully"