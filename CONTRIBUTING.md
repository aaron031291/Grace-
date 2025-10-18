# Contributing to Grace AI System

Thank you for your interest in contributing to Grace!

## Development Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ (optional, SQLite works for dev)
- Redis 7+ (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/yourorg/grace.git
cd grace

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
make install

# Copy environment config
cp .env.example .env

# Initialize database
python -c "from grace.database import init_db; init_db()"
```

## Validation Before Commit

Always run validations before committing:

```bash
# Complete validation
make validate

# Or individual checks
make imports    # Check imports
make types      # Type checking
make config     # Config validation
make test       # Run tests
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep line length ≤ 100 chars
- Use Black for formatting: `make format`

## Testing

```bash
# Run all tests
make test

# Run specific test
python test_integration_full.py

# Run quick validation
make quick-test
```

## Pull Request Process

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run validations: `make validate`
4. Commit with clear message
5. Push and create PR
6. Ensure CI passes

## Project Structure

```
grace/
├── api/           # FastAPI endpoints
├── auth/          # Authentication
├── config/        # Configuration
├── clarity/       # Clarity Framework
├── mldl/          # ML specialists
├── avn/           # Self-healing
├── swarm/         # Multi-node coordination
└── transcendence/ # Advanced reasoning
```

## Common Issues

### Import Errors
Run: `make imports`

### Type Errors
Run: `make types`

### Configuration Issues
Run: `make config`

## Questions?

- Documentation: `/docs`
- Issues: [GitHub Issues]
- Email: team@grace-ai.example
