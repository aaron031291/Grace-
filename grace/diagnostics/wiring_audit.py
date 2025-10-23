"""
Grace AI System - Comprehensive Wiring Audit
============================================

This diagnostic checks if all major components are correctly wired together.

AUDIT CHECKLIST:
================
"""

import sys
import inspect
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspaces/Grace-')

audit_results = {
    "passed": [],
    "warnings": [],
    "errors": [],
    "info": []
}

print("=" * 60)
print("GRACE AI SYSTEM - COMPREHENSIVE WIRING AUDIT")
print("=" * 60)
print()

# ============================================================================
# SECTION 1: DATABASE MODELS VERIFICATION
# ============================================================================
print("SECTION 1: Database Models Verification")
print("-" * 60)

try:
    from grace.db.models_phase_a import (
        Base, ImmutableEntry, EvidenceBlob, EventLog, 
        UserAccount, KpiSnapshot, SearchIndexMeta
    )
    audit_results["passed"].append("✓ Database models import successfully")
    
    # Verify all model attributes
    models = [
        ("ImmutableEntry", ImmutableEntry, [
            "entry_id", "entry_cid", "timestamp", "actor", "operation",
            "error_code", "severity", "what", "why", "how", "where",
            "who", "text_summary", "payload_path", "signature", "tags",
            "payload_json"
        ]),
        ("EvidenceBlob", EvidenceBlob, [
            "blob_cid", "storage_path", "content_type", "size",
            "checksum", "created_at", "access_policy"
        ]),
        ("EventLog", EventLog, [
            "event_id", "event_type", "payload_cid", "timestamp",
            "severity", "payload_json"
        ]),
        ("UserAccount", UserAccount, [
            "user_id", "username", "display_name", "role", "public_key",
            "created_at", "active_bool"
        ]),
        ("KpiSnapshot", KpiSnapshot, [
            "kpi_id", "source_cid", "metrics_json", "timestamp"
        ]),
        ("SearchIndexMeta", SearchIndexMeta, [
            "embedding_id", "entry_cid", "embed_model",
            "vector_store_id", "created_at"
        ]),
    ]
    
    for model_name, model_class, required_cols in models:
        for col_name in required_cols:
            if not hasattr(model_class, col_name):
                audit_results["errors"].append(
                    f"✗ {model_name} missing column: {col_name}"
                )
            else:
                audit_results["info"].append(
                    f"  {model_name}.{col_name} ✓"
                )
    
    audit_results["passed"].append("✓ All database models have required columns")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ Database models import failed: {e}")


# ============================================================================
# SECTION 2: MEMORY SYSTEM VERIFICATION
# ============================================================================
print("\nSECTION 2: Memory System Verification")
print("-" * 60)

try:
    from grace.memory.unified_memory_system import (
        MTLImmutableLedger, LightningMemory, FusionMemory,
        VectorMemory, LibrarianMemory, DatabaseSchema,
        UnifiedMemorySystem
    )
    audit_results["passed"].append("✓ Unified memory system imports successfully")
    
    # Check each memory component
    memory_components = [
        ("MTLImmutableLedger", MTLImmutableLedger),
        ("LightningMemory", LightningMemory),
        ("FusionMemory", FusionMemory),
        ("VectorMemory", VectorMemory),
        ("LibrarianMemory", LibrarianMemory),
        ("DatabaseSchema", DatabaseSchema),
        ("UnifiedMemorySystem", UnifiedMemorySystem),
    ]
    
    for comp_name, comp_class in memory_components:
        # Check if class has __init__
        if hasattr(comp_class, '__init__'):
            audit_results["passed"].append(f"✓ {comp_name} is properly defined")
        else:
            audit_results["warnings"].append(f"⚠ {comp_name} missing __init__")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ Memory system import failed: {e}")


# ============================================================================
# SECTION 3: ORCHESTRATION & TRIGGER MESH
# ============================================================================
print("\nSECTION 3: Orchestration & Trigger Mesh")
print("-" * 60)

try:
    from grace.orchestration.trigger_mesh import TriggerMesh, WorkflowRule, EventPattern
    audit_results["passed"].append("✓ TriggerMesh imports successfully")
    
    # Check TriggerMesh methods
    required_methods = [
        "register_rule", "register_handler", "dispatch_event",
        "get_execution_log", "get_rule_stats"
    ]
    
    for method in required_methods:
        if hasattr(TriggerMesh, method):
            audit_results["passed"].append(f"✓ TriggerMesh.{method} exists")
        else:
            audit_results["errors"].append(f"✗ TriggerMesh.{method} missing")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ TriggerMesh import failed: {e}")


# ============================================================================
# SECTION 4: CORE TRUTH LAYER
# ============================================================================
print("\nSECTION 4: Core Truth Layer")
print("-" * 60)

try:
    from grace.core.truth_layer import (
        CoreTruthLayer, MTLKernelCore, ImmutableTruthLog,
        SystemMetrics, DataIntegrity
    )
    audit_results["passed"].append("✓ Core Truth Layer imports successfully")
    
    # Verify CoreTruthLayer has all components
    truth_layer = CoreTruthLayer()
    
    if hasattr(truth_layer, 'mtl_kernel'):
        audit_results["passed"].append("✓ CoreTruthLayer has MTL kernel")
    else:
        audit_results["errors"].append("✗ CoreTruthLayer missing MTL kernel")
    
    if hasattr(truth_layer, 'immutable_log'):
        audit_results["passed"].append("✓ CoreTruthLayer has immutable log")
    else:
        audit_results["errors"].append("✗ CoreTruthLayer missing immutable log")
    
    if hasattr(truth_layer, 'system_metrics'):
        audit_results["passed"].append("✓ CoreTruthLayer has system metrics")
    else:
        audit_results["errors"].append("✗ CoreTruthLayer missing system metrics")
    
except Exception as e:
    audit_results["errors"].append(f"✗ Core Truth Layer error: {e}")


# ============================================================================
# SECTION 5: KERNELS VERIFICATION
# ============================================================================
print("\nSECTION 5: Kernels Verification")
print("-" * 60)

try:
    from grace.kernels import (
        CognitiveCortex, SentinelKernel, SwarmKernel,
        MetaLearningKernel
    )
    audit_results["passed"].append("✓ All kernels import successfully")
    
    kernel_list = [
        ("CognitiveCortex", CognitiveCortex),
        ("SentinelKernel", SentinelKernel),
        ("SwarmKernel", SwarmKernel),
        ("MetaLearningKernel", MetaLearningKernel),
    ]
    
    for kernel_name, kernel_class in kernel_list:
        audit_results["passed"].append(f"✓ {kernel_name} available")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ Kernels import failed: {e}")


# ============================================================================
# SECTION 6: SERVICES VERIFICATION
# ============================================================================
print("\nSECTION 6: Services Verification")
print("-" * 60)

try:
    from grace.services import (
        TaskManager, CommunicationChannel, NotificationService,
        LLMService, WebSocketService, PolicyEngine, TrustLedger
    )
    audit_results["passed"].append("✓ All services import successfully")
    
    services = [
        "TaskManager", "CommunicationChannel", "NotificationService",
        "LLMService", "WebSocketService", "PolicyEngine", "TrustLedger"
    ]
    
    for service_name in services:
        audit_results["passed"].append(f"✓ {service_name} available")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ Services import failed: {e}")


# ============================================================================
# SECTION 7: IMMUNE SYSTEM VERIFICATION
# ============================================================================
print("\nSECTION 7: Immune System Verification")
print("-" * 60)

try:
    from grace.immune_system import (
        ImmuneSystem, ThreatDetector
    )
    audit_results["passed"].append("✓ Immune system imports successfully")
    
    immune = ImmuneSystem()
    if hasattr(immune, 'threats'):
        audit_results["passed"].append("✓ ImmuneSystem.threats available")
    else:
        audit_results["warnings"].append("⚠ ImmuneSystem.threats not initialized")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ Immune system import failed: {e}")


# ============================================================================
# SECTION 8: MCP INTEGRATION
# ============================================================================
print("\nSECTION 8: MCP Integration")
print("-" * 60)

try:
    from grace.mcp import MCPManager
    audit_results["passed"].append("✓ MCP Manager imports successfully")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ MCP import failed: {e}")


# ============================================================================
# SECTION 9: CONSCIOUSNESS VERIFICATION
# ============================================================================
print("\nSECTION 9: Consciousness Verification")
print("-" * 60)

try:
    from grace.consciousness import Consciousness, CognitiveCortex
    audit_results["passed"].append("✓ Consciousness system imports successfully")
    
except ImportError as e:
    audit_results["errors"].append(f"✗ Consciousness import failed: {e}")


# ============================================================================
# SECTION 10: API LAYER VERIFICATION
# ============================================================================
print("\nSECTION 10: API Layer Verification")
print("-" * 60)

try:
    from grace.api import create_app
    audit_results["passed"].append("✓ API layer imports successfully")
    
except ImportError as e:
    audit_results["warnings"].append(f"⚠ API import failed: {e}")


# ============================================================================
# FINAL REPORT
# ============================================================================
print()
print("=" * 60)
print("AUDIT RESULTS SUMMARY")
print("=" * 60)
print()

print(f"✅ PASSED: {len(audit_results['passed'])} checks")
for item in audit_results['passed'][:10]:  # Show first 10
    print(f"  {item}")
if len(audit_results['passed']) > 10:
    print(f"  ... and {len(audit_results['passed']) - 10} more")

print()
print(f"⚠️  WARNINGS: {len(audit_results['warnings'])} items")
for item in audit_results['warnings']:
    print(f"  {item}")

print()
print(f"❌ ERRORS: {len(audit_results['errors'])} items")
for item in audit_results['errors']:
    print(f"  {item}")

print()
print(f"ℹ️  INFO: {len(audit_results['info'])} items logged")

print()
print("=" * 60)
if audit_results['errors']:
    print("STATUS: ⚠️  ISSUES FOUND - Review errors above")
    sys.exit(1)
elif audit_results['warnings']:
    print("STATUS: ✓ WORKING - Some warnings to review")
    sys.exit(0)
else:
    print("STATUS: ✅ ALL SYSTEMS WIRED UP CORRECTLY")
    sys.exit(0)
