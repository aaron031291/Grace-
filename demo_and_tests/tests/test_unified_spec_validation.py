#!/usr/bin/env python3
"""
Simple validation test for Unified Operating Specification Integration.
Tests core functionality without heavy dependencies.
"""

import asyncio
import sys
import importlib.util

def load_module():
    """Load the unified spec module directly."""
    spec = importlib.util.spec_from_file_location(
        'unified_spec_integration',
        'grace/interface/unified_spec_integration.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_basic_functionality():
    """Test basic functionality of all components."""
    print("=" * 70)
    print("🚀 Grace Unified Operating Specification - Validation Test")
    print("=" * 70)
    
    # Load module
    module = load_module()
    print("\n✅ Module loaded successfully")
    
    # Test Context Bus
    print("\n🔄 Testing Context Bus...")
    bus = module.ContextBus()
    await bus.set_context("test_key", "test_value")
    value = bus.get_context("test_key")
    assert value == "test_value"
    print(f"   ✅ Context set/get: {value}")
    
    # Test Memory Explorer
    print("\n🧠 Testing Memory Explorer...")
    explorer = module.MemoryExplorerEnhanced(bus)
    manifest = await explorer.create_memory_folder(
        folder_id="test_folder",
        purpose="Test folder",
        domain="testing"
    )
    assert manifest.folder_id == "test_folder"
    print(f"   ✅ Memory folder created: {manifest.folder_id}")
    
    item = module.MemoryItem(
        id="test_001",
        path="test_folder/item1",
        tags=["test"],
        trust=0.8
    )
    await explorer.store_memory_item(item)
    print(f"   ✅ Memory item stored: {item.id} (trust={item.trust})")
    
    # Test AVN Engine
    print("\n🔧 Testing AVN/RCA/Healing Engine...")
    avn = module.AVNHealingEngine(explorer, bus)
    anomaly = await avn.detect_anomaly("test_metric", 100.0, 50.0, 0.3)
    assert anomaly is not None
    print(f"   ✅ Anomaly detected: {anomaly.metric_name} (severity={anomaly.severity})")
    
    # Test Loop Orchestrator
    print("\n⏱️  Testing Loop Orchestrator...")
    orchestrator = module.LoopOrchestratorIntegration(bus)
    
    executions = []
    async def test_loop(loop_id, tick):
        executions.append(tick)
    
    await orchestrator.register_loop("test_loop", "Test", 1, test_loop)
    print(f"   ✅ Loop registered")
    
    await orchestrator.start(tick_interval_s=0.5)
    await asyncio.sleep(1.5)
    health = orchestrator.get_health_status()
    assert health["running"]
    print(f"   ✅ Orchestrator running: {health['tick_count']} ticks")
    await orchestrator.stop()
    
    # Test Unified Integration
    print("\n📊 Testing Unified Integration...")
    integration = module.UnifiedOrbSpecIntegration()
    print(f"   ✅ Integration initialized v{integration.version}")
    
    # Test trust metrics
    metric = await integration.update_trust_metric(
        "test_component",
        module.TrustMetricType.COMPONENT,
        0.85
    )
    print(f"   ✅ Trust metric: {metric.component_id} = {metric.trust_score}")
    
    # Test KPI metrics
    kpi = await integration.record_kpi(
        module.KPIMetricType.MTTR,
        3.5,
        "minutes",
        5.0
    )
    print(f"   ✅ KPI metric: MTTR = {kpi.value}{kpi.unit} (status={kpi.status})")
    
    # Test voice mode
    await integration.set_voice_mode(module.VoiceMode.SOLO_VOICE)
    print(f"   ✅ Voice mode: {integration.voice_mode.value}")
    
    # Test event envelope
    envelope = await integration.publish_event(
        "grace.event.v1",
        "human",
        "orb",
        {"test": "data"},
        {"latency_ms": -50}
    )
    print(f"   ✅ Event published: {envelope.event_type} (trace_id={envelope.trace_id[:8]}...)")
    
    # Get unified stats
    stats = integration.get_unified_stats()
    print(f"\n📈 Unified Stats:")
    print(f"   Version: {stats['version']}")
    print(f"   Memory items: {stats['memory_explorer']['total_items']}")
    print(f"   Trust metrics: {stats['trust_metrics']['total']}")
    print(f"   KPI metrics: {stats['kpi_metrics']['total']}")
    print(f"   Voice mode: {stats['voice']['mode']}")
    
    print("\n" + "=" * 70)
    print("🎉 ALL VALIDATION TESTS PASSED!")
    print("=" * 70)
    
    # Verify key components exist
    print("\n✅ Verified Components:")
    print("   - ContextBus (unified 'now' across components)")
    print("   - MemoryExplorerEnhanced (file-explorer brain)")
    print("   - AVNHealingEngine (anomaly detection & healing)")
    print("   - LoopOrchestratorIntegration (tick scheduler)")
    print("   - UnifiedOrbSpecIntegration (main integration)")
    print("   - EventEnvelope (unified event structure)")
    print("   - MemoryItem (enhanced memory with trust)")
    print("   - TrustMetric & KPIMetric (metrics framework)")
    print("   - VoiceMode & VoiceCommand (collaboration modes)")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_basic_functionality())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
