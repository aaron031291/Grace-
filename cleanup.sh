#!/bin/bash
"""
Grace AI Repository - Verification & Cleanup Script
====================================================
This script identifies and removes Phase 1-3 deletions safely
"""

echo "Grace AI Repository Cleanup - Phase 1, 2, 3"
echo "==========================================="
echo ""

# PHASE 1: Documentation
echo "PHASE 1: Checking documentation files..."
if [ -f "/workspaces/Grace-/ARCHITECTURE_REFACTORED.md" ]; then
    echo "  ✓ Found: ARCHITECTURE_REFACTORED.md (will delete)"
    rm -f /workspaces/Grace-/ARCHITECTURE_REFACTORED.md
    echo "  ✓ Deleted: ARCHITECTURE_REFACTORED.md"
else
    echo "  ℹ Not found: ARCHITECTURE_REFACTORED.md (already deleted or never existed)"
fi
echo ""

# PHASE 2: Cache & Artifacts
echo "PHASE 2: Removing cache and build artifacts..."

# Count before cleanup
CACHE_COUNT=$(find /workspaces/Grace- -type d -name '__pycache__' 2>/dev/null | wc -l)
if [ "$CACHE_COUNT" -gt 0 ]; then
    echo "  Found $CACHE_COUNT __pycache__ directories"
    find /workspaces/Grace- -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ Removed __pycache__ directories"
fi

PYC_COUNT=$(find /workspaces/Grace- -type f -name '*.pyc' 2>/dev/null | wc -l)
if [ "$PYC_COUNT" -gt 0 ]; then
    echo "  Found $PYC_COUNT .pyc files"
    find /workspaces/Grace- -type f -name '*.pyc' -delete
    echo "  ✓ Removed .pyc files"
fi

PYO_COUNT=$(find /workspaces/Grace- -type f -name '*.pyo' 2>/dev/null | wc -l)
if [ "$PYO_COUNT" -gt 0 ]; then
    echo "  Found $PYO_COUNT .pyo files"
    find /workspaces/Grace- -type f -name '*.pyo' -delete
    echo "  ✓ Removed .pyo files"
fi

if [ -d "/workspaces/Grace-/.pytest_cache" ]; then
    echo "  ✓ Found .pytest_cache (will delete)"
    rm -rf /workspaces/Grace-/.pytest_cache
    echo "  ✓ Deleted .pytest_cache"
fi

if [ -d "/workspaces/Grace-/dist" ]; then
    echo "  ✓ Found dist/ (will delete)"
    rm -rf /workspaces/Grace-/dist
    echo "  ✓ Deleted dist/"
fi

if [ -d "/workspaces/Grace-/build" ]; then
    echo "  ✓ Found build/ (will delete)"
    rm -rf /workspaces/Grace-/build
    echo "  ✓ Deleted build/"
fi

echo ""

# PHASE 3: Redundant Code
echo "PHASE 3: Removing redundant code files..."

if [ -f "/workspaces/Grace-/grace/kernels/resilience_kernel.py" ]; then
    echo "  ✓ Found: resilience_kernel.py (will delete)"
    rm -f /workspaces/Grace-/grace/kernels/resilience_kernel.py
    echo "  ✓ Deleted: resilience_kernel.py"
else
    echo "  ℹ Not found: resilience_kernel.py (already deleted or never existed)"
fi

if [ -f "/workspaces/Grace-/grace/services/observability.py" ]; then
    echo "  ✓ Found: observability.py (will delete)"
    rm -f /workspaces/Grace-/grace/services/observability.py
    echo "  ✓ Deleted: observability.py"
else
    echo "  ℹ Not found: observability.py (already deleted or never existed)"
fi

echo ""
echo "✓ CLEANUP COMPLETE"
echo ""
echo "Verification steps:"
echo "  1. Check git status: git status"
echo "  2. Test syntax: python -m py_compile grace/**/*.py"
echo "  3. Test entry point: python main.py --help"
echo ""
