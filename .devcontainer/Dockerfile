FROM mcr.microsoft.com/devcontainers/python:3.11

# Install Node.js for frontend
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    build-essential \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dev dependencies
RUN pip install --no-cache-dir \
    black \
    ruff \
    pytest \
    pytest-asyncio \
    pytest-cov \
    ipython \
    jupyter

# Set up user
USER vscode

# Configure git
RUN git config --global user.name "Grace AI" \
    && git config --global user.email "grace@ai.dev"
