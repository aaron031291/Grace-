#!/usr/bin/env python3
"""
TriggerMesh Integration Example

Demonstrates how to:
1. Initialize TriggerMesh orchestration layer
2. Register kernel handlers
3. Load workflows from YAML
4. Emit events that trigger workflows
5. Monitor workflow execution

This shows the complete integration with Grace's existing architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add Grace root to path
grace_root = Path(__file__).parent.parent
sys.path.insert(0, str(grace_root))

from grace.core.event_bus import EventBus
from grace.core.immutable_logs import ImmutableLogs
from grace.core.kpi_trust_monitor import KPITrustMonitor
from grace.orchestration import (
    WorkflowRegistry,
    EventRouter,
    WorkflowEngine,
)


# ============================================================================
# Mock Kernel Handlers (replace with actual kernel instances)
# ============================================================================


class MockAVNCore:
    """Mock AVN Core for demonstration."""

    async def escalate_healing(self, **kwargs):
        """Handle critical healing escalation."""
        print(f"🔴 AVN Core: Escalating healing for {kwargs.get('component_id')}")
        print(f"   Current score: {kwargs.get('current_score')}")
        print(f"   Threshold: {kwargs.get('threshold')}")
        print(f"   Actions: {kwargs.get('recommended_actions', [])}")
        return {"status": "healing_initiated", "component": kwargs.get("component_id")}


class MockLearningKernel:
    """Mock Learning Kernel for demonstration."""

    async def trigger_adaptive_learning(self, **kwargs):
        """Handle adaptive learning trigger."""
        print(f"🔵 Learning Kernel: Triggering adaptive learning for {kwargs.get('component_id')}")
        print(f"   Focus areas: {kwargs.get('focus_areas', [])}")
        print(f"   Current score: {kwargs.get('current_score')}")
        return {
            "status": "learning_started",
            "component": kwargs.get("component_id"),
            "improvements": ["error_analysis", "coverage_boost"],
        }


class MockGovernanceKernel:
    """Mock Governance Kernel for demonstration."""

    async def initiate_trust_review(self, **kwargs):
        """Handle trust degradation review."""
        print(f"⚖️  Governance: Initiating trust review for {kwargs.get('component_id')}")
        print(f"   Current score: {kwargs.get('current_score')}")
        print(f"   Degradation rate: {kwargs.get('degradation_rate')}")
        return {"status": "review_initiated", "review_id": "REV-001"}


class MockMonitoringKernel:
    """Mock Monitoring Kernel for demonstration."""

    async def track_improvement_opportunity(self, **kwargs):
        """Track improvement opportunity."""
        print(f"📊 Monitoring: Tracking improvement for {kwargs.get('component_id')}")
        print(f"   Current: {kwargs.get('current_score')}%, Target: {kwargs.get('target_score')}%")
        print(f"   Gap: {kwargs.get('gap')}%")
        return {"status": "tracked"}


# ============================================================================
# Main Integration Example
# ============================================================================


async def main():
    print("=" * 80)
    print("TriggerMesh Orchestration Layer - Integration Demo")
    print("=" * 80)

    # Step 1: Initialize core components
    print("\n📦 Step 1: Initializing core components...")
    event_bus = EventBus()
    await event_bus.start()

    immutable_logs = ImmutableLogs()
    await immutable_logs.start()

    kpi_monitor = KPITrustMonitor(event_publisher=event_bus.publish)
    await kpi_monitor.start()

    # Step 2: Load workflows
    print("\n📋 Step 2: Loading workflows from YAML...")
    workflow_dir = grace_root / "grace" / "orchestration" / "workflows"
    registry = WorkflowRegistry()
    registry.load_workflows(str(workflow_dir))

    print(f"   ✅ Loaded {len(registry.workflows)} workflows")
    print(f"   ✅ Monitoring {len(registry.get_trigger_event_types())} event types")

    # Step 3: Initialize workflow engine and register kernels
    print("\n🔧 Step 3: Registering kernel handlers...")
    workflow_engine = WorkflowEngine(event_bus, immutable_logs, kpi_monitor)

    # Register mock kernels
    workflow_engine.register_kernel("avn_core", MockAVNCore())
    workflow_engine.register_kernel("learning_kernel", MockLearningKernel())
    workflow_engine.register_kernel("governance_kernel", MockGovernanceKernel())
    workflow_engine.register_kernel("monitoring_kernel", MockMonitoringKernel())

    print(f"   ✅ Registered {len(workflow_engine.kernel_handlers)} kernels")

    # Step 4: Start event router
    print("\n🚀 Step 4: Starting event router...")
    event_router = EventRouter(registry, event_bus, immutable_logs, kpi_monitor)
    await event_router.start()

    print("   ✅ Event router running")
    print(f"   ✅ Subscribed to {len(registry.get_trigger_event_types())} event types")

    # Step 5: Simulate events that trigger workflows
    print("\n" + "=" * 80)
    print("📡 Step 5: Simulating events that trigger workflows")
    print("=" * 80)

    # Event 1: Critical KPI breach → AVN escalation
    print("\n1️⃣  Simulating CRITICAL KPI breach...")
    await event_bus.publish(
        "kpi.threshold_breach",
        {
            "metric_name": "test_quality_score",
            "component_id": "ingress_kernel",
            "value": 45.0,
            "threshold": 50.0,
            "severity": "CRITICAL",
            "previous_value": 65.0,
            "delta": -20.0,
        },
    )

    await asyncio.sleep(2)  # Allow workflow to execute

    # Event 2: Warning KPI breach → Learning Kernel
    print("\n2️⃣  Simulating WARNING KPI breach...")
    await event_bus.publish(
        "kpi.threshold_breach",
        {
            "metric_name": "test_quality_score",
            "component_id": "learning_kernel",
            "value": 68.0,
            "threshold": 70.0,
            "severity": "WARNING",
            "previous_value": 72.0,
            "delta": -4.0,
        },
    )

    await asyncio.sleep(2)

    # Event 3: Trust degradation → Governance review
    print("\n3️⃣  Simulating trust degradation...")
    await event_bus.publish(
        "trust.degradation_detected",
        {
            "component_id": "orchestration_kernel",
            "current_score": 0.65,
            "degradation_rate": 0.08,
            "affected_metrics": ["reliability", "performance"],
        },
    )

    await asyncio.sleep(2)

    # Event 4: Test quality improvement suggestion
    print("\n4️⃣  Simulating quality improvement suggestion...")
    await event_bus.publish(
        "test_quality.improvement_suggested",
        {
            "component_id": "unknown_component",
            "current_score": 82.6,
            "target_score": 90.0,
            "gap": 7.4,
            "suggestions": [
                "Improve quality by 7.4% to reach threshold",
                "Address high-severity errors first",
                "Review failing test patterns",
            ],
        },
    )

    await asyncio.sleep(2)

    # Step 6: Display statistics
    print("\n" + "=" * 80)
    print("📊 Step 6: System Statistics")
    print("=" * 80)

    router_stats = event_router.get_stats()
    print("\nEvent Router:")
    print(f"  Events received: {router_stats['events_received']}")
    print(f"  Workflows triggered: {router_stats['workflows_triggered']}")
    print(f"  Workflows failed: {router_stats['workflows_failed']}")
    print(f"  Events filtered: {router_stats['events_filtered']}")
    print(f"  Events rate-limited: {router_stats['events_rate_limited']}")

    engine_stats = workflow_engine.get_stats()
    print("\nWorkflow Engine:")
    print(f"  Workflows executed: {engine_stats['workflows_executed']}")
    print(f"  Actions executed: {engine_stats['actions_executed']}")
    print(f"  Actions succeeded: {engine_stats['actions_succeeded']}")
    print(f"  Actions failed: {engine_stats['actions_failed']}")

    registry_stats = registry.get_stats()
    print("\nWorkflow Registry:")
    print(f"  Total workflows: {registry_stats['total_workflows']}")
    print(f"  Enabled workflows: {registry_stats['enabled_workflows']}")
    print(f"  Unique event types: {registry_stats['unique_event_types']}")

    # Step 7: Cleanup
    print("\n" + "=" * 80)
    print("🛑 Step 7: Shutting down...")
    print("=" * 80)

    await event_router.stop()
    await kpi_monitor.stop()
    await immutable_logs.stop()
    await event_bus.stop()

    print("\n✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
