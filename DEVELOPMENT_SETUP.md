# Grace Development Setup Guide

This guide helps developers set up the Grace Governance Kernel development environment on any operating system (Linux, macOS, Windows) and enable proper Git workflow for push/pull operations.

## 🚀 Quick Start

### Prerequisites

- Git 2.30+ 
- Python 3.11+ 
- Text editor or IDE of choice

### 1. Clone the Repository

```bash
# Using HTTPS (recommended for most users)
git clone https://github.com/aaron031291/Grace-.git
cd Grace-

# OR using SSH (if you have SSH keys configured)
git clone git@github.com:aaron031291/Grace-.git
cd Grace-
```

### 2. Environment Setup

Create a Python virtual environment:

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy the environment template and configure:

```bash
cp .env.template .env
# Edit .env with your preferred text editor
```

### 4. Verify Installation

Test that Grace imports successfully:

```bash
python -c "import grace; print('Grace imports successfully')"
```

Run the test suite:

```bash
python demo_and_tests/tests/smoke_tests.py
```

## 🔧 Development Workflow

### Git Configuration

Set up your Git identity (if not already configured):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Branch Management

Create a new feature branch for your work:

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# OR for bug fixes
git checkout -b fix/issue-description
```

### Making Changes

1. Make your code changes
2. Test your changes locally
3. Add and commit your changes:

```bash
git add .
git commit -m "feat: description of your changes"
```

### Push Your Changes

Push your branch to the remote repository:

```bash
# First time pushing a new branch
git push -u origin feature/your-feature-name

# Subsequent pushes
git push
```

### Pull Latest Changes

Keep your branch up to date with the main branch:

```bash
# Switch to main and pull latest changes
git checkout main
git pull origin main

# Switch back to your branch and merge/rebase
git checkout feature/your-feature-name
git merge main  # or git rebase main
```

## 🖥️ Platform-Specific Setup

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Follow the standard setup above
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python git

# Follow the standard setup above
```

### Windows

1. Install Python from [python.org](https://python.org)
2. Install Git from [git-scm.com](https://git-scm.com)
3. Use PowerShell or Command Prompt
4. Follow the standard setup above (use `python` instead of `python3`)

### Docker (Alternative)

Run Grace in a Docker container:

```bash
# Build the image
docker build -t grace .

# Run the container
docker run -it --rm -v $(pwd):/workspace grace bash
```

## 🔍 Development Tools

### Recommended IDE Extensions

- **VS Code**: Python, GitLens, Black Formatter
- **PyCharm**: Built-in Git integration, Python support
- **Vim/Neovim**: vim-fugitive, python-mode

### Code Formatting

Grace uses automatic code formatting:

```bash
# Install development dependencies (optional)
pip install black isort flake8

# Format code
black .
isort .

# Check code style
flake8 .
```

## 🧪 Testing

### Run Specific Tests

```bash
# Run all tests
python -m pytest demo_and_tests/tests/

# Run specific test file
python demo_and_tests/tests/test_governance_kernel.py

# Run with verbose output
python -m pytest -v demo_and_tests/tests/
```

### Testing Your Changes

Before pushing, always:

1. Run the test suite
2. Test basic functionality
3. Check for import errors
4. Verify configuration loading

```bash
# Quick verification script
python -c "
import grace
from grace.governance.grace_governance_kernel import GraceGovernanceKernel
print('✅ All imports successful')
"
```

## 🚨 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the right directory
pwd  # Should show .../Grace-

# Ensure virtual environment is activated
which python  # Should show venv path
```

**Git push/pull issues:**
```bash
# Check remote configuration
git remote -v

# Check authentication
git config --list | grep user

# For HTTPS authentication issues
git config --global credential.helper store
```

**Permission errors (Linux/macOS):**
```bash
# Fix permissions
chmod +x grace/scripts/*.sh
sudo chown -R $USER:$USER venv/
```

### Getting Help

1. Check existing [Issues](https://github.com/aaron031291/Grace-/issues)
2. Review the [README.md](./README.md)
3. Check component-specific documentation in subdirectories
4. Run diagnostic script: `python system_check.py`

## 🤝 Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Push to your fork
7. Create a Pull Request

### Code Standards

- Follow existing code style
- Add docstrings to new functions/classes
- Include appropriate error handling
- Update documentation for new features
- Ensure constitutional compliance for governance changes

### Commit Message Format

Use conventional commit format:

```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
Scopes: governance, mldl, audit, multi-os, etc.

Examples:
feat(governance): add new constitutional validator
fix(mldl): resolve specialist consensus timeout
docs(setup): update installation instructions
```

## 📚 Next Steps

After setup, explore:

1. [Governance Examples](./governance_examples.py)
2. [Communication Demo](./grace_communication_demo.py) 
3. [System Analysis](./grace_system_analysis.py)
4. [Multi-OS Kernel](./grace/multi_os/README.md)
5. [Architecture Documentation](./GRACE_SYSTEM_ANALYSIS_REPORT.md)

Happy coding! 🎉