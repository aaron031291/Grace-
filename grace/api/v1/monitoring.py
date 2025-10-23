"""
Monitoring and health dashboard API
"""

from fastapi import APIRouter, Depends
from grace.auth.dependencies import get_current_user
from grace.auth.models import User
from grace.observability.kpis import get_kpi_tracker
from grace.observability.metrics import get_metrics_collector

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/kpis")
async def get_kpis(current_user: User = Depends(get_current_user)):
    """
    Get KPI dashboard data
    
    Returns KPI status, targets, and health assessment
    """
    tracker = get_kpi_tracker()
    
    # Update KPIs from latest metrics
    metrics_collector = get_metrics_collector()
    metrics = await metrics_collector.get_metrics()
    await tracker.calculate_kpis_from_metrics(metrics)
    
    # Generate report
    report = tracker.get_kpi_report()
    
    return report


@router.get("/health")
async def get_health_dashboard():
    """
    Get comprehensive health dashboard
    
    Combines metrics, KPIs, and system status
    """
    from grace.integration.event_bus import get_event_bus
    from grace.trigger_mesh import get_trigger_mesh
    
    tracker = get_kpi_tracker()
    metrics_collector = get_metrics_collector()
    event_bus = get_event_bus()
    trigger_mesh = get_trigger_mesh()
    
    # Get all data
    metrics = await metrics_collector.get_metrics()
    await tracker.calculate_kpis_from_metrics(metrics)
    kpi_report = tracker.get_kpi_report()
    
    dashboard = {
        "timestamp": kpi_report["timestamp"],
        "overall_health": kpi_report["overall_health"],
        "health_percentage": kpi_report["health_percentage"],
        "kpis": kpi_report["kpis"],
        "metrics": {
            "events": {
                "published": metrics.get("grace_events_published_total", 0),
                "processed": metrics.get("grace_events_processed_total", 0),
                "failed": metrics.get("grace_events_failed_total", 0),
                "expired": metrics.get("grace_events_expired_total", 0),
            },
            "queues": {
                "pending": metrics.get("grace_pending_queue_size", 0),
                "dlq": metrics.get("grace_dlq_size", 0),
            },
            "latency": metrics.get("grace_latency_percentiles", {}),
        },
        "event_bus": event_bus.get_metrics(),
        "trigger_mesh": trigger_mesh.get_stats() if trigger_mesh else {}
    }
    
    return dashboard


@router.get("/alerts")
async def get_alerts(current_user: User = Depends(get_current_user)):
    """
    Get active alerts based on KPI violations
    
    Returns list of alerts for KPIs not meeting targets
    """
    tracker = get_kpi_tracker()
    metrics_collector = get_metrics_collector()
    
    metrics = await metrics_collector.get_metrics()
    await tracker.calculate_kpis_from_metrics(metrics)
    
    report = tracker.get_kpi_report()
    
    alerts = []
    
    for failed_kpi in report["failed_kpis"]:
        severity = "warning"
        gap_percentage = (failed_kpi["gap"] / failed_kpi["target"]) * 100
        
        if gap_percentage > 20:
            severity = "critical"
        elif gap_percentage > 10:
            severity = "high"
        
        alerts.append({
            "severity": severity,
            "kpi": failed_kpi["name"],
            "message": f"{failed_kpi['name']} is {failed_kpi['value']:.2f}, target is {failed_kpi['target']:.2f}",
            "gap": failed_kpi["gap"],
            "timestamp": report["timestamp"]
        })
    
    return {
        "count": len(alerts),
        "alerts": alerts
    }
