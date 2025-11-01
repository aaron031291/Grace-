#!/bin/bash
# Codespaces setup script

echo "🚀 Setting up Grace AI Development Environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -e .
pip install -r requirements.txt

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Setup database
echo "🗄️ Setting up database..."
python database/build_all_tables.py

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs checkpoints data

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "   Backend:  cd backend && python main.py"
echo "   Frontend: cd frontend && npm run dev"
echo "   Grace:    python activate_grace.py"
