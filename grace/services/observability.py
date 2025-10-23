"""
Grace AI Observability Service - Comprehensive system monitoring and telemetry
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObservabilityService:
    """Provides comprehensive observability into Grace's operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.traces: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
    
    async def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric."""
        self.metrics[metric_name].append(value)
        logger.info(f"Recorded metric: {metric_name} = {value}")
    
    async def record_trace(self, span_name: str, duration_ms: float, attributes: Dict[str, Any] = None):
        """Record a distributed trace span."""
        trace = {
            "span_name": span_name,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        }
        self.traces.append(trace)
        logger.info(f"Recorded trace: {span_name} ({duration_ms}ms)")
    
    async def record_log(self, level: str, message: str, context: Dict[str, Any] = None):
        """Record a structured log entry."""
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        self.logs.append(log_entry)
    
    async def check_and_alert(self, condition_name: str, condition_met: bool, severity: str = "warning"):
        """Check a condition and create an alert if met."""
        if condition_met:
            alert = {
                "condition": condition_name,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"Alert: {condition_name} ({severity})")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
        return summary
    
    def get_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent traces."""
        return self.traces[-limit:]
    
    def get_logs(self, level: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs, optionally filtered by level."""
        logs = self.logs
        if level:
            logs = [l for l in logs if l["level"] == level]
        return logs[-limit:]
    
    def get_alerts(self, severity: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self.alerts
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        return alerts[-limit:]
