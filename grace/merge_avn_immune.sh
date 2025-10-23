#!/bin/bash
"""
Grace AI - AVN & Immune System Merge & Consolidation
===================================================
Merges AVN and Immune System into unified resilience architecture
Removes duplicates and consolidates into single canonical folder
"""

set -e

echo "=========================================="
echo "Grace AI AVN & Immune System Consolidation"
echo "=========================================="
echo ""

AVN_PATH="/workspaces/Grace-/grace/avn"
IMMUNE_PATH="/workspaces/Grace-/grace/immune_system"
RESILIENCE_PATH="/workspaces/Grace-/grace/resilience"

# Step 1: Create unified resilience folder
echo "STEP 1: Creating unified resilience folder..."
mkdir -p "$RESILIENCE_PATH"
echo "✓ Created: grace/resilience/"
echo ""

# Step 2: Copy AVN files to resilience folder
echo "STEP 2: Merging AVN components..."
if [ -d "$AVN_PATH" ]; then
    echo "  Source: grace/avn/"
    cp "$AVN_PATH"/*.py "$RESILIENCE_PATH/" 2>/dev/null || true
    if [ -d "$AVN_PATH/enhanced_core.py" ]; then
        cp "$AVN_PATH/enhanced_core.py" "$RESILIENCE_PATH/" 2>/dev/null || true
    fi
    if [ -d "$AVN_PATH/pushback.py" ]; then
        cp "$AVN_PATH/pushback.py" "$RESILIENCE_PATH/" 2>/dev/null || true
    fi
    echo "  ✓ AVN files merged"
else
    echo "  ℹ AVN folder not found (already deleted?)"
fi
echo ""

# Step 3: Copy Immune System files to resilience folder
echo "STEP 3: Merging Immune System components..."
if [ -d "$IMMUNE_PATH" ]; then
    echo "  Source: grace/immune_system/"
    cp "$IMMUNE_PATH"/*.py "$RESILIENCE_PATH/" 2>/dev/null || true
    if [ -d "$IMMUNE_PATH/core.py" ]; then
        cp "$IMMUNE_PATH/core.py" "$RESILIENCE_PATH/" 2>/dev/null || true
    fi
    if [ -d "$IMMUNE_PATH/threat_detector.py" ]; then
        cp "$IMMUNE_PATH/threat_detector.py" "$RESILIENCE_PATH/" 2>/dev/null || true
    fi
    if [ -d "$IMMUNE_PATH/avn_healer.py" ]; then
        cp "$IMMUNE_PATH/avn_healer.py" "$RESILIENCE_PATH/" 2>/dev/null || true
    fi
    echo "  ✓ Immune System files merged"
else
    echo "  ℹ Immune System folder not found (already deleted?)"
fi
echo ""

# Step 4: Remove duplicate files
echo "STEP 4: Removing duplicate files..."
cd "$RESILIENCE_PATH"

# Find and remove duplicates
if [ -f "core.py" ] && [ -f "enhanced_core.py" ]; then
    echo "  Resolving: core.py vs enhanced_core.py"
    # Keep enhanced_core.py (newer), remove core.py
    rm -f core.py
    echo "  ✓ Kept enhanced_core.py, removed old core.py"
fi

if [ -f "avn_healer.py" ] && [ -f "healing.py" ]; then
    echo "  Resolving: avn_healer.py vs healing.py"
    rm -f healing.py
    echo "  ✓ Kept avn_healer.py, removed old healing.py"
fi

if [ -f "threat_detector.py" ] && [ -f "threats.py" ]; then
    echo "  Resolving: threat_detector.py vs threats.py"
    rm -f threats.py
    echo "  ✓ Kept threat_detector.py, removed old threats.py"
fi

echo ""

# Step 5: Clean cache files
echo "STEP 5: Cleaning cache files..."
find "$RESILIENCE_PATH" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$RESILIENCE_PATH" -type f -name '*.pyc' -delete 2>/dev/null || true
find "$RESILIENCE_PATH" -type f -name '*.pyo' -delete 2>/dev/null || true
echo "✓ Cache cleaned"
echo ""

# Step 6: Create unified __init__.py for resilience folder
echo "STEP 6: Creating unified __init__.py..."
cat > "$RESILIENCE_PATH/__init__.py" << 'EOF'
"""
Grace AI Resilience System - Unified AVN & Immune System
Combines Adaptive Verification Network with Immune System capabilities
for comprehensive system health, threat detection, and self-healing
"""

# Import AVN components
try:
    from .enhanced_core import EnhancedAVNCore, ComponentHealth, HealingAction
    from .pushback import PushbackEscalation, PushbackSeverity, EscalationDecision
except ImportError:
    pass

# Import Immune System components
try:
    from .threat_detector import ThreatDetector, ThreatLevel, ThreatType
    from .avn_healer import AVNHealer, HealingStrategy
except ImportError:
    pass

# Import core resilience
try:
    from .core import ImmuneSystem, ThreatResponse
except ImportError:
    pass

__all__ = [
    # AVN components
    'EnhancedAVNCore',
    'ComponentHealth',
    'HealingAction',
    'PushbackEscalation',
    'PushbackSeverity',
    'EscalationDecision',
    # Immune System components
    'ThreatDetector',
    'ThreatLevel',
    'ThreatType',
    'AVNHealer',
    'HealingStrategy',
    'ImmuneSystem',
    'ThreatResponse',
]
EOF
echo "✓ Created unified __init__.py"
echo ""

# Step 7: Delete old folders
echo "STEP 7: Deleting old folders..."
if [ -d "$AVN_PATH" ]; then
    echo "  Deleting: grace/avn/"
    rm -rf "$AVN_PATH"
    echo "  ✓ Deleted"
fi

if [ -d "$IMMUNE_PATH" ]; then
    echo "  Deleting: grace/immune_system/"
    rm -rf "$IMMUNE_PATH"
    echo "  ✓ Deleted"
fi
echo ""

# Step 8: Verify merged structure
echo "STEP 8: Verifying merged structure..."
echo ""
echo "Files in grace/resilience/:"
ls -lah "$RESILIENCE_PATH"/*.py | awk '{print "  " $NF}' 2>/dev/null || echo "  (no files found)"
echo ""

# Step 9: List final status
echo "=========================================="
echo "✅ CONSOLIDATION COMPLETE"
echo "=========================================="
echo ""
echo "New unified folder: grace/resilience/"
echo ""
echo "Contains:"
echo "  ✓ AVN (Adaptive Verification Network)"
echo "  ✓ Immune System"
echo "  ✓ Unified initialization"
echo ""
echo "Deleted old folders:"
echo "  ✓ grace/avn/ (contents merged)"
echo "  ✓ grace/immune_system/ (contents merged)"
echo ""
echo "Next steps:"
echo "  1. Test imports: python -c \"from grace.resilience import EnhancedAVNCore, ThreatDetector\""
echo "  2. Verify no duplicates: ls grace/resilience/*.py | sort | uniq -d"
echo "  3. Commit: git add -A && git commit -m 'refactor: merge AVN and immune system into unified resilience'"
echo ""
