"""
Automated tests with KPI validation
"""

import pytest
import asyncio

from grace.observability.kpis import KPITracker, KPI


@pytest.mark.asyncio
async def test_kpi_target_met():
    """Assert KPI correctly identifies when target is met"""
    kpi = KPI(
        name="test_metric",
        value=95.0,
        target=90.0,
        unit="%"
    )
    
    assert kpi.is_met() is True
    assert kpi.percentage_of_target() > 100


@pytest.mark.asyncio
async def test_kpi_target_not_met():
    """Assert KPI correctly identifies when target is not met"""
    kpi = KPI(
        name="test_metric",
        value=85.0,
        target=90.0,
        unit="%"
    )
    
    assert kpi.is_met() is False
    assert kpi.percentage_of_target() < 100


@pytest.mark.asyncio
async def test_kpi_tracker_initialization():
    """Assert KPI tracker initializes with all required KPIs"""
    tracker = KPITracker()
    
    required_kpis = [
        "event_throughput",
        "event_success_rate",
        "avg_latency_p95",
        "system_availability",
        "consensus_accuracy"
    ]
    
    for kpi_name in required_kpis:
        assert kpi_name in tracker.kpis


@pytest.mark.asyncio
async def test_kpi_update():
    """Assert KPI values can be updated"""
    tracker = KPITracker()
    
    await tracker.update_kpi("event_throughput", 150.0)
    
    kpi = tracker.kpis["event_throughput"]
    assert kpi.value == 150.0
    assert kpi.is_met() is True  # Target is 100


@pytest.mark.asyncio
async def test_kpi_calculation_from_metrics():
    """Assert KPIs are calculated correctly from metrics"""
    tracker = KPITracker()
    
    metrics = {
        "grace_events_published_total": 1000,
        "grace_events_processed_total": 980,
        "grace_events_failed_total": 20,
        "grace_latency_percentiles": {
            "event_processing": {
                "p95": 85.0
            }
        }
    }
    
    await tracker.calculate_kpis_from_metrics(metrics)
    
    # Check success rate: 980/1000 = 98%
    success_rate = tracker.kpis["event_success_rate"]
    assert success_rate.value == 98.0
    assert success_rate.is_met() is True
    
    # Check latency
    latency = tracker.kpis["avg_latency_p95"]
    assert latency.value == 85.0
    assert latency.is_met() is True  # Target is 100ms


@pytest.mark.asyncio
async def test_kpi_report_generation():
    """Assert KPI report is generated correctly"""
    tracker = KPITracker()
    
    # Set some values
    await tracker.update_kpi("event_success_rate", 98.0)
    await tracker.update_kpi("avg_latency_p95", 85.0)
    
    report = tracker.get_kpi_report()
    
    assert "kpis" in report
    assert "met_kpis" in report
    assert "failed_kpis" in report
    assert "overall_health" in report
    assert "health_percentage" in report


@pytest.mark.asyncio
async def test_system_health_assessment():
    """Assert system health is assessed correctly"""
    tracker = KPITracker()
    
    # Set all KPIs to meet targets
    for name in tracker.kpis.keys():
        kpi = tracker.kpis[name]
        await tracker.update_kpi(name, kpi.target + 1)
    
    report = tracker.get_kpi_report()
    
    assert report["overall_health"] == "healthy"
    assert report["health_percentage"] == 100.0


@pytest.mark.asyncio
async def test_event_bus_meets_throughput_kpi():
    """Integration test: Assert EventBus meets throughput KPI"""
    from grace.integration.event_bus import EventBus
    from grace.schemas.events import GraceEvent
    from grace.observability.metrics import get_metrics_collector
    import time
    
    bus = EventBus()
    tracker = KPITracker()
    
    # Emit events
    start_time = time.time()
    for i in range(100):
        event = GraceEvent(
            event_type="test.throughput",
            source="test",
            payload={"seq": i}
        )
        await bus.emit(event)
    
    elapsed = time.time() - start_time
    
    # Calculate throughput
    throughput = 100 / elapsed if elapsed > 0 else 0
    
    # Verify meets KPI (should be > 100 events/sec)
    assert throughput > 100.0


@pytest.mark.asyncio
async def test_event_success_rate_kpi():
    """Integration test: Assert event success rate meets KPI"""
    from grace.integration.event_bus import EventBus
    from grace.schemas.events import GraceEvent
    
    bus = EventBus()
    tracker = KPITracker()
    
    successful = []
    
    async def handler(event):
        successful.append(event)
    
    bus.subscribe("test.success", handler)
    
    # Emit 100 events
    for i in range(100):
        event = GraceEvent(
            event_type="test.success",
            source="test"
        )
        await bus.emit(event)
    
    await asyncio.sleep(0.3)
    
    # Calculate success rate
    success_rate = (len(successful) / 100) * 100
    
    # Should meet 95% KPI
    assert success_rate >= 95.0
