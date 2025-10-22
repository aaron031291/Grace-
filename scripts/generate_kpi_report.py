"""
Generate KPI report for CI/CD
"""

import sys
import asyncio
import json


async def main():
    """Generate KPI report"""
    from grace.observability.kpis import get_kpi_tracker
    from grace.observability.metrics import get_metrics_collector
    
    tracker = get_kpi_tracker()
    metrics_collector = get_metrics_collector()
    
    # Simulate some metrics
    metrics = {
        "grace_events_published_total": 1000,
        "grace_events_processed_total": 980,
        "grace_events_failed_total": 20,
        "grace_latency_percentiles": {
            "event_processing": {"p95": 85.0}
        }
    }
    
    await tracker.calculate_kpis_from_metrics(metrics)
    
    report = tracker.get_kpi_report()
    
    print(json.dumps(report, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
