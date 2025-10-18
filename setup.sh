#!/bin/bash
# Grace AI System - Complete Setup Script

set -e

echo "🚀 Grace AI System Setup"
echo "========================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "❌ Python 3.10+ required, found $python_version"
    exit 1
fi
echo "✅ Python $python_version"
echo ""

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Install Grace in editable mode
echo "🔧 Installing Grace in editable mode..."
pip install -e .
echo "✅ Grace installed"
echo ""

# Create directories
echo "📁 Creating directory structure..."
mkdir -p grace/demos
mkdir -p tests
mkdir -p logs
mkdir -p data
echo "✅ Directories created"
echo ""

# Copy environment template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.template .env
    echo "✅ .env created (please update with your values)"
else
    echo "✅ .env exists"
fi
echo ""

# Create VS Code settings
if [ ! -d ".vscode" ]; then
    mkdir -p .vscode
    echo "📝 Creating VS Code settings..."
    cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.extraPaths": ["${workspaceFolder}"],
  "python.analysis.typeCheckingMode": "basic"
}
EOF
    echo "✅ VS Code settings created"
fi
echo ""

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v || echo "⚠️  Some tests failed (expected for incomplete features)"
echo ""

# Summary
echo "=" * 70
echo "✨ Grace AI System Setup Complete!"
echo "=" * 70
echo ""
echo "📍 Next steps:"
echo "   1. Update .env with your API keys and configuration"
echo "   2. Activate venv: source .venv/bin/activate"
echo "   3. Run demos: python grace/demos/complete_system_demo.py"
echo "   4. Start coding!"
echo ""
echo "🔍 Quick checks:"
echo "   python -c 'import grace; print(grace.__file__)'"
echo "   python grace/clarity/clarity_demo.py"
echo ""
