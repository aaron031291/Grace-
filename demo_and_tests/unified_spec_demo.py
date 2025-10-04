#!/usr/bin/env python3
"""
Grace Unified Operating Specification - Interactive Demo

This demo showcases the complete Unified Operating Specification integration
with the Orb Interface, demonstrating:
- Loop Orchestrator with tick scheduler
- Enhanced Memory Explorer
- AVN/RCA/Healing Engine
- Trust & KPI Framework
- Voice & Collaboration modes
"""

import asyncio
import importlib.util
from datetime import datetime


def load_module():
    """Load the unified spec module directly."""
    spec = importlib.util.spec_from_file_location(
        'unified_spec_integration',
        'grace/interface/unified_spec_integration.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def demo_scenario():
    """Run a complete demo scenario."""
    print("=" * 80)
    print(" Grace Unified Operating Specification - Interactive Demo")
    print("=" * 80)
    
    module = load_module()
    
    # Initialize the unified integration
    print("\n📋 Initializing Unified Operating Specification...")
    integration = module.UnifiedOrbSpecIntegration()
    print(f"✅ Initialized v{integration.version}")
    
    # Start the orchestrator
    print("\n⏱️  Starting Loop Orchestrator...")
    await integration.start()
    await asyncio.sleep(0.5)
    print("✅ Orchestrator running")
    
    # Scenario: API Latency Issue Detection and Healing
    print("\n" + "=" * 80)
    print(" SCENARIO: API Latency Degradation - Detection and Automated Healing")
    print("=" * 80)
    
    # Step 1: Set up memory patterns
    print("\n📚 Step 1: Setting up Memory Explorer with API resilience patterns...")
    
    # Create memory folder
    manifest = await integration.memory_explorer.create_memory_folder(
        folder_id="api_resilience",
        purpose="API performance and resilience patterns",
        domain="api_optimization",
        policies=["trust_threshold_0.7", "secrets_redaction"]
    )
    print(f"✅ Created memory folder: {manifest.folder_id}")
    print(f"   Purpose: {manifest.purpose}")
    print(f"   Domain: {manifest.domain}")
    
    # Store historical patterns
    patterns = [
        {
            "id": "pattern_timeout",
            "path": "api_resilience/timeout_handling",
            "tags": ["timeout", "api", "resilience"],
            "trust": 0.92,
            "fix": "Increase connection timeout to 10s and add retry logic"
        },
        {
            "id": "pattern_rate_limit",
            "path": "api_resilience/rate_limiting",
            "tags": ["rate_limit", "api", "performance"],
            "trust": 0.88,
            "fix": "Implement exponential backoff with jitter"
        },
        {
            "id": "pattern_caching",
            "path": "api_resilience/response_caching",
            "tags": ["cache", "api", "latency"],
            "trust": 0.85,
            "fix": "Add Redis caching layer with 5-minute TTL"
        }
    ]
    
    for p in patterns:
        item = module.MemoryItem(
            id=p["id"],
            path=p["path"],
            tags=p["tags"],
            trust=p["trust"],
            metadata={"fix": p["fix"]}
        )
        await integration.memory_explorer.store_memory_item(item, folder_id="api_resilience")
        print(f"✅ Stored pattern: {p['path']} (trust={p['trust']})")
    
    # Step 2: Set voice mode for interaction
    print("\n🎤 Step 2: Setting voice mode to Co-Partner (collaborative)...")
    await integration.set_voice_mode(module.VoiceMode.CO_PARTNER)
    print(f"✅ Voice mode: {integration.voice_mode.value}")
    
    # Step 3: Simulate voice command
    print("\n🗣️  Step 3: User voice command...")
    command = await integration.execute_voice_command(
        intent="Monitor API performance and alert me if latency increases",
        user_id="engineer_123"
    )
    print(f"✅ Voice command executed:")
    print(f"   Intent: {command.intent}")
    print(f"   Mode: {command.mode.value}")
    print(f"   Trace ID: {command.trace_id}")
    
    # Step 4: Update baseline trust
    print("\n📊 Step 4: Establishing baseline trust metrics...")
    trust_metric = await integration.update_trust_metric(
        component_id="api_gateway",
        metric_type=module.TrustMetricType.COMPONENT,
        trust_score=0.92,
        confidence=0.95
    )
    print(f"✅ Baseline trust for api_gateway: {trust_metric.trust_score:.2f}")
    
    # Step 5: Record baseline KPIs
    print("\n📈 Step 5: Recording baseline performance KPIs...")
    baseline_kpi = await integration.record_kpi(
        metric_type=module.KPIMetricType.PERFORMANCE,
        value=180.0,
        unit="ms",
        target=200.0
    )
    print(f"✅ Baseline API latency: {baseline_kpi.value}{baseline_kpi.unit} (status={baseline_kpi.status})")
    
    # Step 6: Simulate latency degradation
    print("\n⚠️  Step 6: SIMULATION - API latency degrading...")
    print("   Current: 180ms → Degraded: 950ms")
    
    # Detect anomaly
    anomaly = await integration.avn_engine.detect_anomaly(
        metric_name="api_latency",
        current_value=950.0,
        expected_value=180.0,
        threshold=0.2
    )
    
    print(f"\n🚨 ANOMALY DETECTED!")
    print(f"   Metric: {anomaly.metric_name}")
    print(f"   Current: {anomaly.current_value}ms")
    print(f"   Expected: {anomaly.expected_value}ms")
    print(f"   Deviation: {anomaly.deviation:.1%}")
    print(f"   Severity: {anomaly.severity}")
    print(f"   Trace ID: {anomaly.trace_id}")
    
    # Step 7: Perform Root Cause Analysis
    print("\n🔍 Step 7: Performing Root Cause Analysis...")
    hypotheses = await integration.avn_engine.perform_rca(anomaly)
    
    print(f"✅ Generated {len(hypotheses)} hypotheses:")
    for i, h in enumerate(hypotheses, 1):
        print(f"\n   Hypothesis {i}:")
        print(f"   └─ Root Cause: {h.root_cause}")
        print(f"   └─ Confidence: {h.confidence:.1%}")
        print(f"   └─ Evidence: {', '.join(h.evidence)}")
        print(f"   └─ Suggested Fix: {h.suggested_fix}")
    
    # Step 8: Execute healing in sandbox
    print("\n🔧 Step 8: Executing healing action in sandbox...")
    if hypotheses:
        best_hypothesis = hypotheses[0]  # Highest confidence
        result = await integration.avn_engine.execute_healing(best_hypothesis, sandbox=True)
        
        print(f"✅ Sandbox execution completed:")
        print(f"   Success: {result['success']}")
        print(f"   Sandbox output: {result.get('sandbox_output', 'N/A')}")
        print(f"   Executed at: {result['executed_at']}")
    
    # Step 9: Update trust based on success
    print("\n📊 Step 9: Updating trust metrics based on healing success...")
    updated_trust = await integration.update_trust_metric(
        component_id="api_gateway",
        metric_type=module.TrustMetricType.COMPONENT,
        trust_score=0.88,  # Slightly decreased due to incident
        confidence=0.90
    )
    print(f"✅ Updated trust for api_gateway: {updated_trust.trust_score:.2f}")
    print(f"   Trust drift: {updated_trust.trust_drift:+.3f}")
    
    # Step 10: Record healing KPI
    print("\n⏱️  Step 10: Recording Mean Time to Repair (MTTR)...")
    mttr_kpi = await integration.record_kpi(
        metric_type=module.KPIMetricType.MTTR,
        value=2.8,
        unit="minutes",
        target=5.0
    )
    print(f"✅ MTTR recorded: {mttr_kpi.value}{mttr_kpi.unit} (status={mttr_kpi.status})")
    print(f"   Target: {mttr_kpi.target} minutes ✅ Under target!")
    
    # Step 11: Publish unified event
    print("\n📨 Step 11: Publishing unified event envelope...")
    envelope = await integration.publish_event(
        event_type="grace.healing.v1",
        actor="grace",
        component="avn",
        payload={
            "anomaly_id": anomaly.anomaly_id,
            "hypothesis_id": best_hypothesis.hypothesis_id if hypotheses else None,
            "action": "healing_executed",
            "success": True
        },
        kpi_deltas={"api_latency_ms": -770}  # Improvement
    )
    print(f"✅ Event published:")
    print(f"   Type: {envelope.event_type}")
    print(f"   Actor: {envelope.actor}")
    print(f"   KPI Deltas: {envelope.kpi_deltas}")
    print(f"   Trace ID: {envelope.trace_id}")
    
    # Step 12: Get comprehensive stats
    print("\n📊 Step 12: Comprehensive Unified Stats...")
    stats = integration.get_unified_stats()
    
    print(f"\n{'='*80}")
    print(" FINAL SYSTEM STATUS")
    print(f"{'='*80}")
    print(f"\n🔄 Orchestrator:")
    print(f"   Running: {stats['orchestrator']['running']}")
    print(f"   Total Ticks: {stats['orchestrator']['tick_count']}")
    print(f"   Active Loops: {stats['orchestrator']['loops_count']}")
    
    print(f"\n🧠 Memory Explorer:")
    print(f"   Total Items: {stats['memory_explorer']['total_items']}")
    print(f"   Folders: {stats['memory_explorer']['folders']}")
    print(f"   Adjacency Graph Nodes: {stats['memory_explorer']['adjacency_graph_nodes']}")
    
    print(f"\n🔧 AVN/Healing Engine:")
    print(f"   Anomalies Detected: {stats['avn_engine']['anomalies_detected']}")
    print(f"   Hypotheses Generated: {stats['avn_engine']['hypotheses_generated']}")
    print(f"   Healing Actions: {stats['avn_engine']['healing_actions']}")
    
    print(f"\n📈 Trust & KPI Metrics:")
    print(f"   Total Trust Metrics: {stats['trust_metrics']['total']}")
    print(f"   Average Trust: {stats['trust_metrics']['avg_trust']:.2f}")
    print(f"   Total KPI Metrics: {stats['kpi_metrics']['total']}")
    print(f"   Critical KPIs: {stats['kpi_metrics']['critical']}")
    print(f"   Warning KPIs: {stats['kpi_metrics']['warning']}")
    
    print(f"\n🎤 Voice & Collaboration:")
    print(f"   Mode: {stats['voice']['mode']}")
    print(f"   Commands Executed: {stats['voice']['commands_executed']}")
    
    print(f"\n📨 Events:")
    print(f"   Total Envelopes: {stats['events']['total_envelopes']}")
    
    print(f"\n🔄 Context Bus:")
    print(f"   Context Keys: {stats['context_bus']['context_keys']}")
    print(f"   History Size: {stats['context_bus']['history_size']}")
    print(f"   Subscribers: {stats['context_bus']['subscribers']}")
    
    # Stop the orchestrator
    print(f"\n{'='*80}")
    await integration.stop()
    print("✅ Orchestrator stopped cleanly")
    
    print(f"\n{'='*80}")
    print(" 🎉 DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print("\nKey Takeaways:")
    print("1. ✅ Loop Orchestrator coordinates all components with tick-based execution")
    print("2. ✅ Memory Explorer stores patterns with trust feedback")
    print("3. ✅ AVN Engine detects anomalies and performs RCA")
    print("4. ✅ Healing actions execute in sandbox before production")
    print("5. ✅ Trust metrics update based on success/failure")
    print("6. ✅ KPIs track performance (MTTR < 5 minutes achieved)")
    print("7. ✅ Voice commands log with unified event envelopes")
    print("8. ✅ Context bus maintains unified 'now' across components")
    
    print("\n📚 Documentation: docs/UNIFIED_SPEC_INTEGRATION.md")
    print("🧪 Tests: demo_and_tests/tests/test_unified_spec_validation.py")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    try:
        asyncio.run(demo_scenario())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
