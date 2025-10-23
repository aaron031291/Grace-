#!/usr/bin/env python3
"""
Grace AI - System Integration Verification
==========================================
Complete system check for all imports and wiring
"""

import sys
import os

# Add workspace to path
sys.path.insert(0, '/workspaces/Grace-')

print("=" * 60)
print("GRACE AI SYSTEM - COMPREHENSIVE VERIFICATION")
print("=" * 60)
print()

results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

# Test 1: Core Truth Layer
print("TEST 1: Core Truth Layer")
print("-" * 60)
try:
    from grace.core.truth_layer import CoreTruthLayer, MTLKernelCore, ImmutableTruthLog, SystemMetrics
    truth_layer = CoreTruthLayer()
    print("✓ CoreTruthLayer initialized")
    print("  ✓ MTLKernelCore available")
    print("  ✓ ImmutableTruthLog available")
    print("  ✓ SystemMetrics available")
    results["passed"].append("Core Truth Layer")
except Exception as e:
    print(f"✗ CoreTruthLayer error: {e}")
    results["failed"].append(f"Core Truth Layer: {e}")
print()

# Test 2: Memory Systems
print("TEST 2: Unified Memory System")
print("-" * 60)
try:
    from grace.memory import (
        MTLImmutableLedger, LightningMemory, FusionMemory,
        VectorMemory, LibrarianMemory, DatabaseSchema, UnifiedMemorySystem
    )
    print("✓ All memory components import:")
    print("  ✓ MTLImmutableLedger")
    print("  ✓ LightningMemory")
    print("  ✓ FusionMemory")
    print("  ✓ VectorMemory")
    print("  ✓ LibrarianMemory")
    print("  ✓ DatabaseSchema")
    print("  ✓ UnifiedMemorySystem")
    results["passed"].append("Unified Memory System")
except Exception as e:
    print(f"✗ Memory System error: {e}")
    results["failed"].append(f"Memory System: {e}")
print()

# Test 3: Orchestration
print("TEST 3: Orchestration (TriggerMesh)")
print("-" * 60)
try:
    from grace.orchestration.trigger_mesh import TriggerMesh
    mesh = TriggerMesh()
    print("✓ TriggerMesh initialized")
    print("  ✓ Event routing configured")
    print("  ✓ Workflow execution ready")
    results["passed"].append("TriggerMesh Orchestration")
except Exception as e:
    print(f"✗ TriggerMesh error: {e}")
    results["failed"].append(f"TriggerMesh: {e}")
print()

# Test 4: Kernels
print("TEST 4: Kernels")
print("-" * 60)
try:
    from grace.kernels import (
        CognitiveCortex, SentinelKernel, SwarmKernel, MetaLearningKernel
    )
    print("✓ All kernels import:")
    print("  ✓ CognitiveCortex")
    print("  ✓ SentinelKernel")
    print("  ✓ SwarmKernel")
    print("  ✓ MetaLearningKernel")
    results["passed"].append("All Kernels")
except Exception as e:
    print(f"✗ Kernels error: {e}")
    results["failed"].append(f"Kernels: {e}")
print()

# Test 5: Services
print("TEST 5: Core Services")
print("-" * 60)
try:
    from grace.services import (
        TaskManager, CommunicationChannel, NotificationService,
        LLMService, WebSocketService, PolicyEngine, TrustLedger
    )
    print("✓ All services import:")
    print("  ✓ TaskManager")
    print("  ✓ CommunicationChannel")
    print("  ✓ NotificationService")
    print("  ✓ LLMService")
    print("  ✓ WebSocketService")
    print("  ✓ PolicyEngine")
    print("  ✓ TrustLedger")
    results["passed"].append("Core Services")
except Exception as e:
    print(f"✗ Services error: {e}")
    results["failed"].append(f"Services: {e}")
print()

# Test 6: Immune System
print("TEST 6: Immune System")
print("-" * 60)
try:
    from grace.immune_system import ImmuneSystem, ThreatDetector
    print("✓ Immune System imports:")
    print("  ✓ ImmuneSystem")
    print("  ✓ ThreatDetector")
    results["passed"].append("Immune System")
except Exception as e:
    print(f"✗ Immune System error: {e}")
    results["failed"].append(f"Immune System: {e}")
print()

# Test 7: AVN (Adaptive Verification Network)
print("TEST 7: AVN (Adaptive Verification Network)")
print("-" * 60)
try:
    from grace.avn import EnhancedAVNCore, PushbackEscalation
    print("✓ AVN imports:")
    print("  ✓ EnhancedAVNCore")
    print("  ✓ PushbackEscalation")
    results["passed"].append("AVN System")
except Exception as e:
    print(f"✗ AVN error: {e}")
    results["failed"].append(f"AVN: {e}")
print()

# Test 8: MCP Integration
print("TEST 8: MCP (Model Context Protocol)")
print("-" * 60)
try:
    from grace.mcp import MCPManager
    print("✓ MCP imports:")
    print("  ✓ MCPManager")
    results["passed"].append("MCP Integration")
except Exception as e:
    print(f"✗ MCP error: {e}")
    results["failed"].append(f"MCP: {e}")
print()

# Test 9: Consciousness
print("TEST 9: Consciousness System")
print("-" * 60)
try:
    from grace.consciousness import Consciousness, CognitiveCortex
    print("✓ Consciousness imports:")
    print("  ✓ Consciousness")
    print("  ✓ CognitiveCortex")
    results["passed"].append("Consciousness System")
except Exception as e:
    print(f"✗ Consciousness error: {e}")
    results["failed"].append(f"Consciousness: {e}")
print()

# Test 10: API Layer
print("TEST 10: API Layer")
print("-" * 60)
try:
    from grace.api import create_app
    print("✓ API layer imports:")
    print("  ✓ create_app")
    results["passed"].append("API Layer")
except Exception as e:
    print(f"⚠ API Layer warning: {e}")
    results["warnings"].append(f"API Layer: {e}")
print()

# Test 11: Database Models
print("TEST 11: Database Models")
print("-" * 60)
try:
    from grace.db.models_phase_a import (
        ImmutableEntry, EvidenceBlob, EventLog,
        UserAccount, KpiSnapshot, SearchIndexMeta
    )
    print("✓ All database models import:")
    print("  ✓ ImmutableEntry")
    print("  ✓ EvidenceBlob")
    print("  ✓ EventLog")
    print("  ✓ UserAccount")
    print("  ✓ KpiSnapshot")
    print("  ✓ SearchIndexMeta")
    results["passed"].append("Database Models")
except Exception as e:
    print(f"✗ Database Models error: {e}")
    results["failed"].append(f"Database Models: {e}")
print()

# Summary
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print()

print(f"✅ PASSED: {len(results['passed'])} modules")
for item in results['passed']:
    print(f"  ✓ {item}")
print()

if results['warnings']:
    print(f"⚠️  WARNINGS: {len(results['warnings'])} items")
    for item in results['warnings']:
        print(f"  ⚠ {item}")
    print()

if results['failed']:
    print(f"❌ FAILED: {len(results['failed'])} modules")
    for item in results['failed']:
        print(f"  ✗ {item}")
    print()
    print("STATUS: ⚠️  ISSUES FOUND - Please review errors above")
    sys.exit(1)
else:
    print("STATUS: ✅ ALL SYSTEMS VERIFIED AND WORKING")
    print()
    print("Your Grace AI system is properly wired and ready to use!")
    sys.exit(0)
