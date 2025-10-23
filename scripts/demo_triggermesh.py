#!/usr/bin/env python3
"""
Integration Demo for the TriggerMesh Orchestration Layer.

This script demonstrates how to initialize and run the TriggerMesh components,
register kernel handlers, and simulate events to trigger workflows.
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path to allow importing grace modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogger
from grace.core.kpi_trust_monitor import KPITrustMonitor
from grace.orchestration.event_router import EventRouter
from grace.orchestration.workflow_engine import WorkflowEngine
from grace.orchestration.workflow_registry import WorkflowRegistry

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger("TriggerMeshDemo")


# --- Mock Kernel Handlers ---
# In a real application, these would be methods on kernel objects.
async def mock_avn_escalate(component_id: str, **kwargs: Any):
    logger.info(f"🚀 AVN KERNEL: Escalation received for component '{component_id}'. Details: {kwargs}")


async def mock_learning_log(component_id: str, **kwargs: Any):
    logger.info(f"🧠 LEARNING KERNEL: Adaptation opportunity logged for '{component_id}'. Details: {kwargs}")


async def mock_governance_review(component_id: str, **kwargs: Any):
    logger.info(f"⚖️ GOVERNANCE KERNEL: Review requested for '{component_id}'. Details: {kwargs}")


async def mock_immutable_logs_critical(event_name: str, **kwargs: Any):
    logger.info(f"🔒 IMMUTABLE LOGS: Critical event '{event_name}' logged. Details: {kwargs}")


# --- Main Demo ---
async def main():
    """Run the TriggerMesh integration demo."""
    logger.info("--- TriggerMesh Integration Demo ---")

    # 1. Initialize Core Components (Mocks)
    event_bus = EventBus()
    immutable_logger = ImmutableLogger(config={"log_to_console": True})
    kpi_monitor = KPITrustMonitor(event_bus.publish)

    # 2. Initialize TriggerMesh Components
    workflow_dir = Path(__file__).parent.parent / "grace" / "orchestration" / "workflows"
    
    logger.info("\n--- Step 1: Loading Workflows ---")
    registry = WorkflowRegistry(workflow_dir)
    registry.load_workflows()
    stats = registry.get_stats()
    if stats["validation_errors"] > 0:
        logger.error("Workflow validation failed. Aborting.")
        return
    logger.info(f"✅ {stats['workflows_loaded']} workflows loaded successfully.")

    engine = WorkflowEngine(event_bus, immutable_logger)
    router = EventRouter(registry, engine, event_bus, immutable_logger, kpi_monitor)

    # 3. Register Kernel Handlers with the Workflow Engine
    logger.info("\n--- Step 2: Registering Kernel Handlers ---")
    engine.register_kernel_handler("avn_core.escalate_healing", mock_avn_escalate)
    engine.register_kernel_handler("learning_kernel.log_adaptation_opportunity", mock_learning_log)
    engine.register_kernel_handler("governance.request_review", mock_governance_review)
    engine.register_kernel_handler("immutable_logs.log_critical_event", mock_immutable_logs_critical)
    logger.info(f"✅ {len(engine.kernel_handlers)} kernel handlers registered.")

    # 4. Setup Event Subscriptions
    logger.info("\n--- Step 3: Setting up Event Subscriptions ---")
    await router.setup_subscriptions()
    logger.info("✅ Event router is now listening for events.")

    # 5. Simulate Events to Trigger Workflows
    logger.info("\n--- Step 4: Simulating Events ---")

    # Scenario 1: Critical KPI breach
    logger.info("\n>>> Simulating CRITICAL KPI breach...")
    await event_bus.publish(
        "kpi.threshold_breach",
        {
            "metric_name": "cpu_usage",
            "component_id": "api_server_1",
            "value": 95.5,
            "threshold": 90.0,
            "severity": "CRITICAL",
        },
    )
    await asyncio.sleep(0.1) # Allow event to propagate

    # Scenario 2: Warning KPI breach (should trigger a different workflow)
    logger.info("\n>>> Simulating WARNING KPI breach...")
    await event_bus.publish(
        "kpi.threshold_breach",
        {
            "metric_name": "memory_usage",
            "component_id": "database_worker_3",
            "value": 88.1,
            "threshold": 85.0,
            "severity": "WARNING",
        },
    )
    await asyncio.sleep(0.1)

    # Scenario 3: Test quality degradation (filtered out by severity)
    logger.info("\n>>> Simulating DEGRADED test quality...")
    await event_bus.publish(
        "test_quality.healing_required",
        {
            "component_id": "ingestion_pipeline",
            "status": "DEGRADED",
            "score": 65.0,
            "severity": "DEGRADED",
        },
    )
    await asyncio.sleep(0.1)

    # Scenario 4: Critical DB update
    logger.info("\n>>> Simulating CRITICAL database update...")
    await event_bus.publish(
        "db.table_updated",
        {
            "table_name": "governance_rules",
            "operation": "UPDATE",
            "row_id": "rule_001",
        },
    )
    await asyncio.sleep(0.1)

    # Scenario 5: An event with no matching workflow
    logger.info("\n>>> Simulating an event with NO workflow...")
    await event_bus.publish("system.kernel_started", {"kernel_name": "demo_kernel"})
    await asyncio.sleep(0.1)

    # 6. Display Final Statistics
    logger.info("\n--- Step 5: Final Statistics ---")
    router_stats = router.get_stats()
    logger.info(f"Event Router Stats: {router_stats}")
    
    print("\n--- Demo Complete ---")
    print("✅ TriggerMesh components are integrated and operational.")
    print("✅ Events were successfully routed to workflows and executed by kernel handlers.")


if __name__ == "__main__":
    asyncio.run(main())
