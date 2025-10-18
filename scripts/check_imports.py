"""
Check for import errors across the codebase
"""

import sys
import importlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_module(module_name: str) -> bool:
    """Check if module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False


def main():
    """Check all Grace modules"""
    modules = [
        "grace.config",
        "grace.config.settings",
        "grace.auth",
        "grace.auth.models",
        "grace.auth.security",
        "grace.auth.dependencies",
        "grace.database",
        "grace.documents",
        "grace.embeddings",
        "grace.vectorstore",
        "grace.governance",
        "grace.middleware",
        "grace.middleware.logging",
        "grace.middleware.rate_limit",
        "grace.middleware.metrics",
        "grace.observability",
        "grace.observability.structured_logging",
        "grace.observability.prometheus_metrics",
        "grace.observability.kpi_monitor",
        "grace.clarity",
        "grace.clarity.memory_bank",
        "grace.clarity.governance_validator",
        "grace.mldl",
        "grace.mldl.quorum_aggregator",
        "grace.mldl.uncertainty",
        "grace.avn",
        "grace.avn.enhanced_core",
        "grace.orchestration",
        "grace.orchestration.enhanced_scheduler",
        "grace.orchestration.autoscaler",
        "grace.orchestration.heartbeat",
        "grace.testing",
        "grace.testing.quality_monitor",
        "grace.mtl",
        "grace.mtl.immutable_logs",
        "grace.swarm",
        "grace.swarm.coordinator",
        "grace.swarm.transport",
        "grace.swarm.consensus",
        "grace.transcendence",
        "grace.transcendence.quantum_library",
        "grace.transcendence.scientific_discovery",
        "grace.transcendence.societal_impact",
        "grace.integration",
        "grace.integration.event_bus",
        "grace.integration.swarm_transcendence_integration",
        "grace.websocket",
    ]
    
    print("Checking Grace modules...\n")
    
    success_count = 0
    for module in modules:
        if check_module(module):
            success_count += 1
    
    print(f"\n{success_count}/{len(modules)} modules imported successfully")
    
    if success_count == len(modules):
        print("\n✅ All imports successful!")
        return 0
    else:
        print(f"\n❌ {len(modules) - success_count} import errors found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
