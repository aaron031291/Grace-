#!/usr/bin/env python3
"""
Test Grace Unified Operating Specification Integration

Tests the integration of:
- Loop Orchestrator with tick scheduler + context bus
- Enhanced Memory Explorer (File-Explorer-like brain)
- AVN/RCA/Healing Engine
- Governance Layer with immutable logs
- Observability Fabric
- Trust & KPI Framework
- Voice & Collaboration modes
"""

import asyncio
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

async def test_context_bus():
    """Test Context Bus for unified 'now'."""
    print("\n🔄 Testing Context Bus...")
    
    from grace.interface.unified_spec_integration import ContextBus
    
    bus = ContextBus()
    
    # Set and get context
    await bus.set_context("test_key", "test_value")
    value = bus.get_context("test_key")
    assert value == "test_value", "Context value mismatch"
    print("✅ Context bus set/get working")
    
    # Test subscribers
    received_updates = []
    
    async def subscriber(key, value, old_value):
        received_updates.append({"key": key, "value": value, "old_value": old_value})
    
    bus.subscribe("watched_key", subscriber)
    await bus.set_context("watched_key", "new_value")
    await asyncio.sleep(0.1)  # Give subscriber time to run
    
    assert len(received_updates) > 0, "Subscriber not called"
    print(f"✅ Context bus subscribers working ({len(received_updates)} updates)")
    
    # Get all context
    all_context = bus.get_all_context()
    assert "test_key" in all_context, "All context missing keys"
    print(f"✅ Context bus has {len(all_context)} keys")


async def test_memory_explorer():
    """Test Enhanced Memory Explorer."""
    print("\n🧠 Testing Memory Explorer...")
    
    from grace.interface.unified_spec_integration import ContextBus, MemoryExplorerEnhanced, MemoryItem
    
    bus = ContextBus()
    explorer = MemoryExplorerEnhanced(bus)
    
    # Create memory folder
    manifest = await explorer.create_memory_folder(
        folder_id="api_patterns",
        purpose="API resilience patterns",
        domain="api_resilience",
        policies=["secrets_redaction", "trust_threshold_0.7"]
    )
    assert manifest.folder_id == "api_patterns", "Folder creation failed"
    print(f"✅ Created memory folder: {manifest.folder_id}")
    
    # Store memory items
    item1 = MemoryItem(
        id="mem_001",
        path="api_patterns/timeout_handling",
        tags=["timeout", "resilience"],
        trust=0.85
    )
    await explorer.store_memory_item(item1, folder_id="api_patterns")
    print(f"✅ Stored memory item: {item1.id} (trust={item1.trust})")
    
    item2 = MemoryItem(
        id="mem_002",
        path="api_patterns/retry_logic",
        tags=["retry", "resilience", "api_resilience"],
        trust=0.75
    )
    await explorer.store_memory_item(item2, folder_id="api_patterns")
    print(f"✅ Stored memory item: {item2.id} (trust={item2.trust})")
    
    # Search with filters
    results = await explorer.search_with_context(
        query="resilience",
        filters={"min_trust": 0.7, "limit": 10}
    )
    assert len(results) >= 2, "Search results incomplete"
    print(f"✅ Search found {len(results)} results")
    
    # Update trust feedback
    await explorer.update_trust_feedback(item1.id, success=True, weight=0.1)
    updated_item = explorer.memory_items[item1.id]
    assert updated_item.trust > 0.85, "Trust not increased"
    print(f"✅ Trust feedback updated: {updated_item.trust:.3f}")


async def test_avn_healing_engine():
    """Test AVN/RCA/Healing Engine."""
    print("\n🔧 Testing AVN/RCA/Healing Engine...")
    
    from grace.interface.unified_spec_integration import (
        ContextBus, MemoryExplorerEnhanced, AVNHealingEngine, MemoryItem
    )
    
    bus = ContextBus()
    explorer = MemoryExplorerEnhanced(bus)
    avn = AVNHealingEngine(explorer, bus)
    
    # Add pattern to memory
    pattern = MemoryItem(
        id="pattern_001",
        path="patterns/api_latency_fix",
        tags=["pattern", "fix", "api_latency"],
        trust=0.9,
        metadata={"fix": "Increase timeout to 5s"}
    )
    await explorer.store_memory_item(pattern)
    
    # Detect anomaly
    anomaly = await avn.detect_anomaly(
        metric_name="api_latency",
        current_value=850.0,
        expected_value=200.0,
        threshold=0.2
    )
    assert anomaly is not None, "Anomaly not detected"
    print(f"✅ Anomaly detected: {anomaly.metric_name} (severity={anomaly.severity})")
    
    # Perform RCA
    hypotheses = await avn.perform_rca(anomaly)
    assert len(hypotheses) > 0, "No RCA hypotheses generated"
    print(f"✅ RCA generated {len(hypotheses)} hypotheses")
    
    for h in hypotheses:
        print(f"   - Hypothesis: {h.root_cause} (confidence={h.confidence:.2f})")
        print(f"     Suggested fix: {h.suggested_fix}")
    
    # Execute healing in sandbox
    result = await avn.execute_healing(hypotheses[0], sandbox=True)
    assert result["success"], "Healing execution failed"
    print(f"✅ Healing executed in sandbox: {result['sandbox_output']}")


async def test_loop_orchestrator():
    """Test Loop Orchestrator with tick scheduler."""
    print("\n⏱️  Testing Loop Orchestrator...")
    
    from grace.interface.unified_spec_integration import ContextBus, LoopOrchestratorIntegration
    
    bus = ContextBus()
    orchestrator = LoopOrchestratorIntegration(bus)
    
    # Track loop executions
    executions = []
    
    async def test_loop(loop_id: str, tick: int):
        executions.append({"loop_id": loop_id, "tick": tick})
    
    # Register loops
    await orchestrator.register_loop(
        loop_id="test_loop_1",
        name="Test Loop 1",
        interval_s=1,
        callback=test_loop,
        priority=5
    )
    print("✅ Registered test loop")
    
    # Start orchestrator
    await orchestrator.start(tick_interval_s=0.5)
    print("✅ Orchestrator started")
    
    # Let it run for a bit
    await asyncio.sleep(2.5)
    
    # Check health
    health = orchestrator.get_health_status()
    assert health["running"], "Orchestrator not running"
    assert health["tick_count"] > 0, "No ticks executed"
    print(f"✅ Orchestrator health: {health['tick_count']} ticks, {len(health['loops'])} loops")
    
    # Verify loop executed
    assert len(executions) > 0, "Loop not executed"
    print(f"✅ Loop executed {len(executions)} times")
    
    # Stop orchestrator
    await orchestrator.stop()
    print("✅ Orchestrator stopped")


async def test_trust_kpi_framework():
    """Test Trust & KPI Framework."""
    print("\n📊 Testing Trust & KPI Framework...")
    
    from grace.interface.unified_spec_integration import (
        UnifiedOrbSpecIntegration, TrustMetricType, KPIMetricType
    )
    
    integration = UnifiedOrbSpecIntegration()
    
    # Update trust metric
    trust_metric = await integration.update_trust_metric(
        component_id="test_component",
        metric_type=TrustMetricType.COMPONENT,
        trust_score=0.85,
        confidence=0.9
    )
    assert trust_metric.trust_score == 0.85, "Trust metric not set"
    print(f"✅ Trust metric updated: {trust_metric.component_id} = {trust_metric.trust_score}")
    
    # Update again to test drift
    trust_metric_2 = await integration.update_trust_metric(
        component_id="test_component",
        metric_type=TrustMetricType.COMPONENT,
        trust_score=0.90,
        confidence=0.9
    )
    assert trust_metric_2.trust_drift != 0, "Trust drift not calculated"
    print(f"✅ Trust drift: {trust_metric_2.trust_drift:.3f}")
    
    # Record KPI metrics
    kpi1 = await integration.record_kpi(
        metric_type=KPIMetricType.MTTR,
        value=3.5,
        unit="minutes",
        target=5.0
    )
    assert kpi1.status == "normal", "KPI status incorrect"
    print(f"✅ KPI recorded: MTTR = {kpi1.value}{kpi1.unit} (status={kpi1.status})")
    
    kpi2 = await integration.record_kpi(
        metric_type=KPIMetricType.GOVERNANCE_LATENCY,
        value=75.0,
        unit="seconds",
        target=60.0
    )
    assert kpi2.status in ["warning", "critical"], "KPI status should be warning/critical"
    print(f"✅ KPI recorded: Governance Latency = {kpi2.value}{kpi2.unit} (status={kpi2.status})")


async def test_voice_modes():
    """Test Voice & Collaboration Modes."""
    print("\n🎤 Testing Voice Modes...")
    
    from grace.interface.unified_spec_integration import UnifiedOrbSpecIntegration, VoiceMode
    
    integration = UnifiedOrbSpecIntegration()
    
    # Set voice mode
    await integration.set_voice_mode(VoiceMode.SOLO_VOICE)
    assert integration.voice_mode == VoiceMode.SOLO_VOICE, "Voice mode not set"
    print(f"✅ Voice mode set: {integration.voice_mode.value}")
    
    # Execute voice command
    command = await integration.execute_voice_command(
        intent="analyze market data",
        user_id="test_user"
    )
    assert command.mode == VoiceMode.SOLO_VOICE, "Voice command mode mismatch"
    print(f"✅ Voice command executed: {command.intent} (trace_id={command.trace_id})")
    
    # Change to co-partner mode
    await integration.set_voice_mode(VoiceMode.CO_PARTNER)
    assert integration.voice_mode == VoiceMode.CO_PARTNER, "Voice mode not changed"
    print(f"✅ Voice mode changed to: {integration.voice_mode.value}")


async def test_event_envelopes():
    """Test Unified Event Envelopes."""
    print("\n📨 Testing Event Envelopes...")
    
    from grace.interface.unified_spec_integration import UnifiedOrbSpecIntegration
    
    integration = UnifiedOrbSpecIntegration()
    
    # Publish event
    envelope = await integration.publish_event(
        event_type="grace.event.v1",
        actor="human",
        component="orb",
        payload={"action": "test_action", "data": "test_data"},
        kpi_deltas={"latency_ms": -50}
    )
    
    assert envelope.event_type == "grace.event.v1", "Event type mismatch"
    assert envelope.actor == "human", "Actor mismatch"
    assert len(envelope.trace_id) > 0, "Trace ID missing"
    print(f"✅ Event published: {envelope.event_type} (trace_id={envelope.trace_id})")
    print(f"   KPI deltas: {envelope.kpi_deltas}")


async def test_unified_stats():
    """Test comprehensive unified stats."""
    print("\n📈 Testing Unified Stats...")
    
    from grace.interface.unified_spec_integration import UnifiedOrbSpecIntegration
    
    integration = UnifiedOrbSpecIntegration()
    
    # Start components
    await integration.start()
    await asyncio.sleep(0.5)
    
    # Get stats
    stats = integration.get_unified_stats()
    
    assert "version" in stats, "Version missing from stats"
    assert "orchestrator" in stats, "Orchestrator stats missing"
    assert "context_bus" in stats, "Context bus stats missing"
    assert "memory_explorer" in stats, "Memory explorer stats missing"
    assert "avn_engine" in stats, "AVN engine stats missing"
    assert "trust_metrics" in stats, "Trust metrics missing"
    assert "kpi_metrics" in stats, "KPI metrics missing"
    assert "voice" in stats, "Voice stats missing"
    
    print(f"✅ Unified stats collected:")
    print(f"   Version: {stats['version']}")
    print(f"   Orchestrator ticks: {stats['orchestrator']['tick_count']}")
    print(f"   Context bus keys: {stats['context_bus']['context_keys']}")
    print(f"   Memory items: {stats['memory_explorer']['total_items']}")
    print(f"   Voice mode: {stats['voice']['mode']}")
    
    # Stop components
    await integration.stop()
    print("✅ Components stopped cleanly")


async def main():
    """Run all tests."""
    print("=" * 70)
    print("🚀 Testing Grace Unified Operating Specification Integration")
    print("=" * 70)
    
    try:
        # Run tests in sequence
        await test_context_bus()
        await test_memory_explorer()
        await test_avn_healing_engine()
        await test_loop_orchestrator()
        await test_trust_kpi_framework()
        await test_voice_modes()
        await test_event_envelopes()
        await test_unified_stats()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
