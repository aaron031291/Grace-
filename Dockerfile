# Multi-stage build for Grace API

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements
COPY requirements.txt setup.py setup.cfg pyproject.toml ./
COPY grace/ ./grace/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 grace && \
    mkdir -p /app /app/data && \
    chown -R grace:grace /app

WORKDIR /app
USER grace

# Copy from builder
COPY --from=builder --chown=grace:grace /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=grace:grace /usr/local/bin /usr/local/bin

# Copy application
COPY --chown=grace:grace grace/ ./grace/
COPY --chown=grace:grace main.py ./
COPY --chown=grace:grace .env.example ./.env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "grace.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
