"""
Comprehensive validation script for entire Grace system
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_module(module_name: str) -> Tuple[bool, str]:
    """Check if module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    """Validate all Grace modules"""
    
    print("=" * 80)
    print("Grace System - Comprehensive Validation")
    print("=" * 80)
    
    modules = {
        "Core Configuration": [
            "grace",
            "grace.config",
            "grace.config.settings",
            "grace.database",
        ],
        "Authentication": [
            "grace.auth",
            "grace.auth.models",
            "grace.auth.security",
            "grace.auth.dependencies",
        ],
        "Documents & Embeddings": [
            "grace.documents",
            "grace.embeddings",
            "grace.embeddings.service",
            "grace.embeddings.providers",
            "grace.vectorstore",
            "grace.vectorstore.service",
        ],
        "Governance": [
            "grace.governance",
            "grace.governance.models",
        ],
        "Clarity Framework": [
            "grace.clarity",
            "grace.clarity.memory_bank",
            "grace.clarity.governance_validator",
            "grace.clarity.feedback_integrator",
            "grace.clarity.specialist_consensus",
            "grace.clarity.unified_output",
            "grace.clarity.drift_detector",
            "grace.clarity.quorum_bridge",
        ],
        "MLDL Specialists": [
            "grace.mldl",
            "grace.mldl.quorum_aggregator",
            "grace.mldl.uncertainty",
        ],
        "AVN & Self-Healing": [
            "grace.avn",
            "grace.avn.enhanced_core",
            "grace.avn.pushback",
        ],
        "Orchestration": [
            "grace.orchestration",
            "grace.orchestration.enhanced_scheduler",
            "grace.orchestration.autoscaler",
            "grace.orchestration.heartbeat",
            "grace.orchestration.scheduler_metrics",
        ],
        "Testing": [
            "grace.testing",
            "grace.testing.quality_monitor",
            "grace.testing.pytest_plugin",
        ],
        "Immutable Logs": [
            "grace.mtl",
            "grace.mtl.immutable_logs",
        ],
        "Swarm Intelligence": [
            "grace.swarm",
            "grace.swarm.coordinator",
            "grace.swarm.transport",
            "grace.swarm.consensus",
            "grace.swarm.discovery",
        ],
        "Transcendence Layer": [
            "grace.transcendence",
            "grace.transcendence.quantum_library",
            "grace.transcendence.scientific_discovery",
            "grace.transcendence.societal_impact",
        ],
        "Integration": [
            "grace.integration",
            "grace.integration.event_bus",
            "grace.integration.event_bus_integration",
            "grace.integration.swarm_transcendence_integration",
        ],
        "Observability": [
            "grace.observability",
            "grace.observability.structured_logging",
            "grace.observability.prometheus_metrics",
            "grace.observability.kpi_monitor",
        ],
        "Middleware": [
            "grace.middleware",
            "grace.middleware.logging",
            "grace.middleware.rate_limit",
            "grace.middleware.metrics",
        ],
        "WebSocket": [
            "grace.websocket",
            "grace.websocket.manager",
        ],
        "API": [
            "grace.api",
            "grace.api.v1.auth",
            "grace.api.v1.documents",
            "grace.api.v1.policies",
            "grace.api.v1.sessions",
            "grace.api.v1.tasks",
            "grace.api.v1.websocket",
            "grace.api.v1.logs",
            "grace.api.public",
        ],
    }
    
    total_success = 0
    total_failed = 0
    failed_modules: List[Tuple[str, str, str]] = []
    
    for category, module_list in modules.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        category_success = 0
        for module_name in module_list:
            success, error = check_module(module_name)
            
            if success:
                print(f"  ✅ {module_name}")
                category_success += 1
                total_success += 1
            else:
                print(f"  ❌ {module_name}")
                print(f"     Error: {error[:100]}...")
                failed_modules.append((category, module_name, error))
                total_failed += 1
        
        print(f"  Category: {category_success}/{len(module_list)} passed")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    total = total_success + total_failed
    print(f"\nTotal Modules: {total}")
    print(f"✅ Passed: {total_success}")
    print(f"❌ Failed: {total_failed}")
    print(f"Success Rate: {(total_success/total*100):.1f}%")
    
    if failed_modules:
        print("\n" + "=" * 80)
        print("FAILED MODULES (Details)")
        print("=" * 80)
        
        for category, module_name, error in failed_modules:
            print(f"\n[{category}] {module_name}")
            print(f"  Error: {error[:200]}")
    
    print("\n" + "=" * 80)
    
    if total_failed == 0:
        print("✅ All modules validated successfully!")
        return 0
    else:
        print(f"⚠️  {total_failed} module(s) need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
