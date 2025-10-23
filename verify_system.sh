#!/bin/bash
"""
Grace AI - System Integration Verification
==========================================
Checks if all components are properly imported and wired
"""

echo "=========================================="
echo "Grace AI System Verification"
echo "=========================================="
echo ""

echo "STEP 1: Checking Python environment..."
python3 --version
echo ""

echo "STEP 2: Verifying imports..."
echo ""

# Test each major component import
echo "Testing core imports:"
python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, '/workspaces/Grace-')

try:
    from grace.core import CoreTruthLayer
    print("  ✓ CoreTruthLayer imports")
except Exception as e:
    print(f"  ✗ CoreTruthLayer error: {e}")

try:
    from grace.memory import UnifiedMemorySystem
    print("  ✓ UnifiedMemorySystem imports")
except Exception as e:
    print(f"  ✗ UnifiedMemorySystem error: {e}")

try:
    from grace.orchestration import TriggerMesh
    print("  ✓ TriggerMesh imports")
except Exception as e:
    print(f"  ✗ TriggerMesh error: {e}")

try:
    from grace.kernels import CognitiveCortex, SentinelKernel, SwarmKernel, MetaLearningKernel
    print("  ✓ All kernels import")
except Exception as e:
    print(f"  ✗ Kernels error: {e}")

try:
    from grace.services import TaskManager, LLMService, PolicyEngine
    print("  ✓ Core services import")
except Exception as e:
    print(f"  ✗ Services error: {e}")

try:
    from grace.immune_system import ImmuneSystem, ThreatDetector
    print("  ✓ Immune System imports")
except Exception as e:
    print(f"  ✗ Immune System error: {e}")

try:
    from grace.avn import EnhancedAVNCore
    print("  ✓ AVN imports")
except Exception as e:
    print(f"  ✗ AVN error: {e}")

try:
    from grace.mcp import MCPManager
    print("  ✓ MCP imports")
except Exception as e:
    print(f"  ✗ MCP error: {e}")

try:
    from grace.consciousness import Consciousness
    print("  ✓ Consciousness imports")
except Exception as e:
    print(f"  ✗ Consciousness error: {e}")

try:
    from grace.api import create_app
    print("  ✓ API layer imports")
except Exception as e:
    print(f"  ✗ API error: {e}")

PYTHON_EOF

echo ""
echo "STEP 3: Running wiring audit..."
python3 /workspaces/Grace-/grace/diagnostics/wiring_audit.py 2>&1 | head -50

echo ""
echo "=========================================="
echo "✅ VERIFICATION COMPLETE"
echo "=========================================="
echo ""
echo "If all imports passed, system is ready!"
echo ""
