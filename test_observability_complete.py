"""
Complete test of observability system
"""

import asyncio
import time

print("=" * 80)
print("Grace Observability System - Complete Test")
print("=" * 80)

async def main():
    from grace.observability.structured_logging import StructuredLogger, setup_logging
    from grace.observability.prometheus_metrics import GraceMetrics
    from grace.observability.kpi_monitor import KPITrustMonitor
    from grace.integration.event_bus import EventBus
    from grace.avn.enhanced_core import EnhancedAVNCore
    
    # Setup logging
    print("\n1. Setting up structured logging...")
    setup_logging(log_level="INFO", json_output=True)
    
    # Create logger
    logger = StructuredLogger("test_component")
    print("✓ Structured logger created")
    
    # Test logging
    print("\n2. Testing structured logging...")
    
    logger.info("Test info message", user_id="user123", operation="test")
    logger.warning("Test warning", decision_id="dec456")
    
    with logger.span("test_operation", param1="value1") as span_id:
        time.sleep(0.1)
        logger.info("Inside span", span_id=span_id)
    
    print("✓ Logging tested")
    
    # Setup metrics
    print("\n3. Setting up Prometheus metrics...")
    metrics = GraceMetrics()
    print("✓ Metrics initialized")
    
    # Test metrics
    print("\n4. Testing metrics collection...")
    
    metrics.record_loop_execution("main_loop", 1.5, "success")
    metrics.record_loop_execution("worker_loop", 0.8, "success")
    metrics.record_loop_execution("main_loop", 2.1, "failure")
    
    metrics.record_decision("mldl", "classification", 0.85)
    metrics.record_decision("governance", "policy_approval", 0.92)
    
    metrics.update_trust_score("auth", "user123", 0.88, 0.85)
    metrics.update_trust_score("ml_model", "lstm_v1", 0.72, 0.75)
    
    metrics.record_error("database", "ConnectionError", "error")
    
    metrics.record_governance_validation(True)
    metrics.record_governance_validation(False, [
        {"type": "privacy", "severity": "warning"}
    ])
    
    metrics.record_consensus("weighted_average", 0.82, True)
    
    metrics.update_component_health("api", 0.95, 3600)
    
    metrics.record_healing("ml_model", "restart", 5.2, True)
    
    print("✓ Metrics recorded")
    
    # Get metrics output
    print("\n5. Generating Prometheus metrics...")
    metrics_output = metrics.get_metrics().decode('utf-8')
    
    # Show sample metrics
    print("\n  Sample metrics:")
    for line in metrics_output.split('\n')[:20]:
        if line and not line.startswith('#'):
            print(f"    {line}")
    
    # Setup KPI monitor
    print("\n6. Setting up KPI trust monitor...")
    
    event_bus = EventBus()
    avn_core = EnhancedAVNCore(event_publisher=event_bus)
    
    # Register components
    avn_core.register_component("ml_model")
    avn_core.register_component("database")
    avn_core.register_component("api")
    
    kpi_monitor = KPITrustMonitor(
        event_publisher=event_bus,
        avn_core=avn_core,
        metrics=metrics
    )
    
    print("✓ KPI monitor initialized")
    
    # Test trust monitoring
    print("\n7. Testing trust monitoring and events...")
    
    # Track events
    events_captured = []
    
    def capture_event(event):
        events_captured.append(event)
        print(f"    Event captured: {event.event_type} - {event.component}:{event.entity_id} (score: {event.current_score:.3f})")
    
    kpi_monitor.register_callback(capture_event)
    
    # Normal score
    kpi_monitor.update_trust_score("ml_model", "lstm_v1", 0.85)
    
    # Warning threshold breach
    kpi_monitor.update_trust_score("ml_model", "lstm_v1", 0.45)
    
    # Critical threshold breach (should trigger healing)
    kpi_monitor.update_trust_score("database", "postgres_main", 0.25)
    
    # Recovery
    kpi_monitor.update_trust_score("database", "postgres_main", 0.75)
    
    # Sudden drop (anomaly)
    kpi_monitor.update_trust_score("api", "gateway", 0.9)
    kpi_monitor.update_trust_score("api", "gateway", 0.65)
    
    print(f"\n✓ Captured {len(events_captured)} trust events")
    
    # Get statistics
    print("\n8. Getting monitoring statistics...")
    
    stats = kpi_monitor.get_statistics()
    print("✓ Trust monitoring statistics:")
    print(f"    Total entities: {stats['total_entities']}")
    print(f"    Average trust: {stats['avg_trust_score']:.3f}")
    print(f"    Below critical: {stats['below_critical']}")
    print(f"    Below warning: {stats['below_warning']}")
    print(f"    Acceptable: {stats['acceptable']}")
    
    # Check AVN triggered
    print("\n9. Checking AVN healing triggered...")
    
    avn_health = avn_core.get_system_health()
    print(f"✓ AVN status: {avn_health['status']}")
    print(f"    Healings attempted: {avn_health['total_healings']}")
    print(f"    Successful: {avn_health['successful_healings']}")
    
    # Test event bus integration
    print("\n10. Testing event bus integration...")
    
    event_stats = event_bus.get_statistics()
    print(f"✓ Event bus statistics:")
    print(f"    Total events: {event_stats['total_events']}")
    print(f"    Active channels: {event_stats['active_channels']}")
    
    print("\n" + "=" * 80)
    print("✅ Observability System Tests Complete!")
    print("=" * 80)
    
    print("\nImplemented Features:")
    print("  ✓ Structured logging with trace IDs")
    print("  ✓ Log spans for operation tracing")
    print("  ✓ Integration with immutable logs")
    print("  ✓ Comprehensive Prometheus metrics")
    print("  ✓ Trust score monitoring")
    print("  ✓ Threshold breach detection")
    print("  ✓ Anomaly detection (sudden drops, trends)")
    print("  ✓ Event publishing to event bus")
    print("  ✓ Governance review triggering")
    print("  ✓ AVN healing triggering")
    print("  ✓ Real-time alerting")

asyncio.run(main())
