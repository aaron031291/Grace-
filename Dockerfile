# Multi-stage production Dockerfile

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt setup.py setup.cfg pyproject.toml ./
COPY grace/__init__.py grace/__init__.py

# Install dependencies to a local directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -u 1000 grace && \
    mkdir -p /app/logs /app/data && \
    chown -R grace:grace /app

# Copy application
COPY --chown=grace:grace grace/ ./grace/
COPY --chown=grace:grace main.py ./
COPY --chown=grace:grace scripts/ ./scripts/
COPY --chown=grace:grace config/ ./config/

# Switch to non-root user
USER grace

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "grace.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
