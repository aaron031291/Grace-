# Grace Scripts Directory

This directory contains helpful scripts for Grace development:

## Scripts Overview

- **setup.sh**: Automated development environment setup
- **git-workflow.sh**: Git workflow automation (Linux/macOS)
- **git-workflow.ps1**: Git workflow automation (Windows)
- **pre-commit**: Git pre-commit hook for code quality

## Usage

### Quick Setup
```bash
./setup.sh
```

### Git Workflow
```bash
# Create a new feature branch
./git-workflow.sh new-branch feature/my-awesome-feature

# Complete workflow (test, commit, push)
./git-workflow.sh workflow feat core "implement new feature"

# Sync with main branch
./git-workflow.sh sync
```

### Manual Hook Installation
```bash
cp pre-commit ../.git/hooks/
chmod +x ../.git/hooks/pre-commit
```

For detailed usage, run any script with `help` or no arguments.