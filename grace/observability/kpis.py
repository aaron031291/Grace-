"""
Key Performance Indicators (KPIs) tracking for Grace system
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio


@dataclass
class KPI:
    """Single KPI definition"""
    name: str
    value: float
    target: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_met(self) -> bool:
        """Check if KPI target is met"""
        return self.value >= self.target
    
    def percentage_of_target(self) -> float:
        """Calculate percentage of target achieved"""
        if self.target == 0:
            return 100.0
        return (self.value / self.target) * 100.0


class KPITracker:
    """
    Tracks and evaluates system KPIs
    
    KPIs tracked:
    - Event processing throughput (events/sec)
    - Event success rate (%)
    - Average latency (ms)
    - System availability (%)
    - Consensus accuracy (%)
    - Trust score stability
    """
    
    def __init__(self):
        self.kpis: Dict[str, KPI] = {}
        
        # Define KPI targets
        self._define_kpi_targets()
    
    def _define_kpi_targets(self):
        """Define KPI targets for the system"""
        targets = {
            "event_throughput": {"target": 100.0, "unit": "events/sec"},
            "event_success_rate": {"target": 95.0, "unit": "%"},
            "avg_latency_p95": {"target": 100.0, "unit": "ms"},
            "system_availability": {"target": 99.5, "unit": "%"},
            "consensus_accuracy": {"target": 90.0, "unit": "%"},
            "trust_score_stability": {"target": 0.8, "unit": "score"},
            "memory_write_success": {"target": 99.0, "unit": "%"},
            "dlq_rate": {"target": 1.0, "unit": "%", "inverted": True},  # Lower is better
            "validation_pass_rate": {"target": 95.0, "unit": "%"},
        }
        
        for name, config in targets.items():
            self.kpis[name] = KPI(
                name=name,
                value=0.0,
                target=config["target"],
                unit=config["unit"]
            )
    
    async def update_kpi(self, name: str, value: float):
        """Update KPI value"""
        if name in self.kpis:
            self.kpis[name].value = value
            self.kpis[name].timestamp = datetime.utcnow()
    
    async def calculate_kpis_from_metrics(self, metrics: Dict[str, Any]):
        """
        Calculate KPIs from system metrics
        
        Args:
            metrics: System metrics from EventBus, MemoryCore, etc.
        """
        # Event throughput (events/sec)
        if "grace_events_processed_total" in metrics:
            # Calculate rate (simplified - would use time window in production)
            throughput = metrics.get("grace_events_processed_total", 0) / 60  # per minute / 60
            await self.update_kpi("event_throughput", throughput)
        
        # Event success rate
        published = metrics.get("grace_events_published_total", 0)
        processed = metrics.get("grace_events_processed_total", 0)
        if published > 0:
            success_rate = (processed / published) * 100
            await self.update_kpi("event_success_rate", success_rate)
        
        # Average latency
        latency_percentiles = metrics.get("grace_latency_percentiles", {})
        if "event_processing" in latency_percentiles:
            p95 = latency_percentiles["event_processing"].get("p95", 0)
            await self.update_kpi("avg_latency_p95", p95)
        
        # DLQ rate
        failed = metrics.get("grace_events_failed_total", 0)
        if published > 0:
            dlq_rate = (failed / published) * 100
            await self.update_kpi("dlq_rate", dlq_rate)
    
    def get_kpi_report(self) -> Dict[str, Any]:
        """
        Generate KPI report
        
        Returns:
            Dictionary with KPI status and recommendations
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": {},
            "met_kpis": [],
            "failed_kpis": [],
            "overall_health": "healthy"
        }
        
        total_kpis = len(self.kpis)
        met_count = 0
        
        for name, kpi in self.kpis.items():
            is_met = kpi.is_met()
            percentage = kpi.percentage_of_target()
            
            report["kpis"][name] = {
                "value": kpi.value,
                "target": kpi.target,
                "unit": kpi.unit,
                "percentage_of_target": percentage,
                "met": is_met,
                "timestamp": kpi.timestamp.isoformat()
            }
            
            if is_met:
                met_count += 1
                report["met_kpis"].append(name)
            else:
                report["failed_kpis"].append({
                    "name": name,
                    "value": kpi.value,
                    "target": kpi.target,
                    "gap": kpi.target - kpi.value
                })
        
        # Overall health assessment
        health_percentage = (met_count / total_kpis) * 100
        
        if health_percentage >= 90:
            report["overall_health"] = "healthy"
        elif health_percentage >= 70:
            report["overall_health"] = "degraded"
        else:
            report["overall_health"] = "critical"
        
        report["health_percentage"] = health_percentage
        
        return report


# Global instance
_kpi_tracker: KPITracker = None


def get_kpi_tracker() -> KPITracker:
    """Get global KPI tracker"""
    global _kpi_tracker
    if _kpi_tracker is None:
        _kpi_tracker = KPITracker()
    return _kpi_tracker
