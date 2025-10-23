# Grace Development Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Testing](#testing)
5. [Code Standards](#code-standards)
6. [Contributing](#contributing)

## Development Environment Setup

### Prerequisites

- Python 3.11+
- Git
- PostgreSQL 14+ (or Docker)
- Redis 7+ (optional)

### Local Setup

1. **Clone Repository**

```bash
git clone https://github.com/yourorg/grace.git
cd grace
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

4. **Configure Environment**

```bash
cp .env.example .env.local
export $(cat .env.local | xargs)
```

Minimum `.env.local`:

```bash
GRACE_ENV=development
GRACE_DEBUG=true
AUTH_SECRET_KEY=dev-secret-key-minimum-32-characters
DATABASE_URL=postgresql://localhost/grace_dev
REDIS_URL=redis://localhost:6379/0
```

5. **Setup Database**

```bash
# Create database
createdb grace_dev

# Run migrations
python scripts/apply_migrations.py
```

6. **Run Development Server**

```bash
# With hot reload
uvicorn grace.api:create_app --factory --reload --host 0.0.0.0 --port 8000

# Or use the main script
python main.py service
```

7. **Verify Installation**

```bash
curl http://localhost:8000/health
python scripts/verify_install.py
```

### Dev Container (VSCode)

Grace includes a dev container configuration:

1. Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open project in VSCode
3. Press `F1` → "Dev Containers: Reopen in Container"
4. All dependencies installed automatically

## Project Structure

```
grace/
├── grace/                    # Main package
│   ├── api/                 # FastAPI routes
│   │   └── v1/             # API v1 endpoints
│   ├── config/             # Configuration management
│   ├── core/               # Core services (unified_service)
│   ├── events/             # Event system
│   ├── governance/         # Governance engine
│   ├── integration/        # Event bus
│   ├── kernels/            # Kernel implementations
│   │   ├── multi_os.py
│   │   ├── mldl.py
│   │   └── resilience.py
│   ├── mcp/                # Message Control Protocol
│   ├── memory/             # Memory layers
│   ├── observability/      # Metrics, logging, KPIs
│   ├── schemas/            # Data schemas
│   ├── security/           # RBAC, encryption, rate limiting
│   ├── trigger_mesh.py     # Event routing
│   ├── trust/              # Trust system
│   └── watchdog.py         # Global exception handling
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── config/                  # Configuration files
│   ├── trigger_mesh.yaml
│   └── grafana/
├── documentation/          # Documentation
├── docker-compose.yml      # Docker configuration
├── Dockerfile             # Container image
├── main.py                # Entry point
├── setup.py               # Package setup
└── pyproject.toml         # Project metadata
```

## Development Workflow

### Creating a New Feature

1. **Create Feature Branch**

```bash
git checkout -b feature/my-new-feature
```

2. **Implement Feature**

Follow the existing patterns:

```python
# Example: New kernel
from grace.mcp import MCPClient, MCPMessageType

class MyNewKernel:
    def __init__(self, event_bus, event_factory, trigger_mesh=None):
        self.event_bus = event_bus
        self.event_factory = event_factory
        self.mcp_client = MCPClient("my_kernel", event_bus, trigger_mesh)
        self._running = False
    
    async def start(self):
        self._running = True
        # Implementation
    
    async def stop(self):
        self._running = False
    
    def get_health(self) -> dict:
        return {
            "status": "healthy" if self._running else "stopped",
            "running": self._running
        }
```

3. **Write Tests**

```python
# tests/test_my_kernel.py
import pytest

@pytest.mark.asyncio
async def test_my_kernel_start():
    from grace.kernels.my_kernel import MyNewKernel
    from grace.integration.event_bus import EventBus
    
    bus = EventBus()
    kernel = MyNewKernel(bus, None)
    
    await kernel.start()
    
    assert kernel._running is True
    
    await kernel.stop()
```

4. **Run Tests**

```bash
pytest tests/test_my_kernel.py -v
```

5. **Format Code**

```bash
black grace/ tests/
ruff check grace/ tests/
```

6. **Commit Changes**

```bash
git add .
git commit -m "feat: add my new kernel"
```

7. **Push and Create PR**

```bash
git push origin feature/my-new-feature
# Create PR on GitHub
```

### Adding API Endpoints

1. **Create Router**

```python
# grace/api/v1/my_endpoint.py
from fastapi import APIRouter, Depends
from grace.auth.dependencies import get_current_user

router = APIRouter(prefix="/my-endpoint", tags=["MyEndpoint"])

@router.get("/")
async def get_data(current_user = Depends(get_current_user)):
    return {"data": "value"}
```

2. **Register Router**

```python
# grace/api/__init__.py
from grace.api.v1.my_endpoint import router as my_router

def create_app():
    app = FastAPI()
    # ...existing code...
    app.include_router(my_router, prefix=settings.api_prefix)
    return app
```

3. **Test Endpoint**

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/my-endpoint/
```

## Testing

### Test Structure

```
tests/
├── test_event_bus_features.py    # Event bus tests
├── test_trigger_mesh.py           # TriggerMesh tests
├── test_mcp.py                    # MCP protocol tests
├── test_kpis.py                   # KPI validation tests
├── test_security.py               # Security tests
├── test_kernel_management.py      # Kernel tests
└── test_end_to_end.py            # Integration tests
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_event_bus_features.py -v

# With coverage
pytest --cov=grace --cov-report=html

# Specific test
pytest tests/test_kpis.py::test_kpi_target_met -v

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

Follow these patterns:

```python
import pytest
import asyncio

# Sync test
def test_something():
    assert True

# Async test
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None

# Test with fixtures
@pytest.fixture
async def event_bus():
    from grace.integration.event_bus import EventBus
    bus = EventBus()
    yield bus
    await bus.shutdown()

@pytest.mark.asyncio
async def test_with_fixture(event_bus):
    # Use event_bus
    pass
```

### Performance Benchmarks

```bash
pytest tests/test_performance.py --benchmark-only
```

## Code Standards

### Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Maximum line length: 100 characters
- Use descriptive variable names

### Code Formatting

```bash
# Format code
black grace/ tests/ scripts/

# Check formatting
black --check grace/ tests/

# Lint
ruff check grace/ tests/
```

### Type Checking

```bash
mypy grace/ --ignore-missing-imports
```

### Documentation

- Docstrings for all public functions/classes
- Use Google style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg2 is negative
    """
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build/tooling changes

Examples:

```
feat: add MLDL consensus mechanism
fix: resolve memory leak in event bus
docs: update API documentation
test: add tests for rate limiter
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Write tests
4. Implement feature
5. Update documentation
6. Run test suite
7. Submit PR with description

### PR Checklist

- [ ] Tests pass
- [ ] Code formatted (black)
- [ ] Type safety verified
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No security vulnerabilities

### Code Review

All PRs require:

- At least 1 approval
- All CI checks passing
- No merge conflicts
- Up-to-date with main branch

### Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and community support
- Slack: Real-time chat (link in README)