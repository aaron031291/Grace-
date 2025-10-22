"""
Metrics collection for Grace system
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for Grace system
    
    Tracks:
    - Event counts by type
    - Latencies (p50, p95, p99)
    - DLQ size
    - TTL drops
    - Queue depths
    """
    
    def __init__(self):
        # Counters
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.event_published_total = 0
        self.event_processed_total = 0
        self.event_failed_total = 0
        self.event_expired_total = 0
        self.event_deduplicated_total = 0
        self.dlq_total = 0
        
        # Latencies (in milliseconds)
        self.latencies: Dict[str, list] = defaultdict(list)
        self.max_latency_samples = 1000
        
        # Gauges
        self.pending_queue_size = 0
        self.dlq_size = 0
        self.active_subscribers = 0
        
        # Custom metrics
        self.custom_metrics: Dict[str, float] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def increment_event_count(self, event_type: str, count: int = 1):
        """Increment event count by type"""
        async with self._lock:
            self.event_counts[event_type] += count
    
    async def record_event_published(self):
        """Record event published"""
        async with self._lock:
            self.event_published_total += 1
    
    async def record_event_processed(self):
        """Record event processed"""
        async with self._lock:
            self.event_processed_total += 1
    
    async def record_event_failed(self):
        """Record event failed"""
        async with self._lock:
            self.event_failed_total += 1
    
    async def record_event_expired(self):
        """Record event expired due to TTL"""
        async with self._lock:
            self.event_expired_total += 1
    
    async def record_event_deduplicated(self):
        """Record duplicate event blocked"""
        async with self._lock:
            self.event_deduplicated_total += 1
    
    async def record_dlq_event(self):
        """Record event sent to DLQ"""
        async with self._lock:
            self.dlq_total += 1
    
    async def record_latency(self, operation: str, latency_ms: float):
        """
        Record operation latency
        
        Args:
            operation: Operation name (e.g., "event_processing", "consensus_request")
            latency_ms: Latency in milliseconds
        """
        async with self._lock:
            latencies = self.latencies[operation]
            latencies.append(latency_ms)
            
            # Keep only recent samples
            if len(latencies) > self.max_latency_samples:
                self.latencies[operation] = latencies[-self.max_latency_samples:]
    
    async def set_gauge(self, name: str, value: float):
        """Set gauge value"""
        async with self._lock:
            if name == "pending_queue_size":
                self.pending_queue_size = value
            elif name == "dlq_size":
                self.dlq_size = value
            elif name == "active_subscribers":
                self.active_subscribers = value
            else:
                self.custom_metrics[name] = value
    
    def _calculate_percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile from values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as dictionary
        
        Returns Prometheus-style metrics
        """
        async with self._lock:
            metrics = {
                # Counters
                "grace_events_published_total": self.event_published_total,
                "grace_events_processed_total": self.event_processed_total,
                "grace_events_failed_total": self.event_failed_total,
                "grace_events_expired_total": self.event_expired_total,
                "grace_events_deduplicated_total": self.event_deduplicated_total,
                "grace_dlq_events_total": self.dlq_total,
                
                # Gauges
                "grace_pending_queue_size": self.pending_queue_size,
                "grace_dlq_size": self.dlq_size,
                "grace_active_subscribers": self.active_subscribers,
                
                # Event counts by type
                "grace_events_by_type": dict(self.event_counts),
                
                # Latency percentiles
                "grace_latency_percentiles": {}
            }
            
            # Calculate latency percentiles for each operation
            for operation, values in self.latencies.items():
                if values:
                    metrics["grace_latency_percentiles"][operation] = {
                        "p50": self._calculate_percentile(values, 50),
                        "p95": self._calculate_percentile(values, 95),
                        "p99": self._calculate_percentile(values, 99),
                        "max": max(values),
                        "min": min(values),
                        "count": len(values)
                    }
            
            # Custom metrics
            metrics.update(self.custom_metrics)
            
            return metrics
    
    async def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format
        
        Returns:
            String in Prometheus exposition format
        """
        metrics = await self.get_metrics()
        lines = []
        
        # Counters
        lines.append("# HELP grace_events_published_total Total events published")
        lines.append("# TYPE grace_events_published_total counter")
        lines.append(f"grace_events_published_total {metrics['grace_events_published_total']}")
        
        lines.append("# HELP grace_events_processed_total Total events processed")
        lines.append("# TYPE grace_events_processed_total counter")
        lines.append(f"grace_events_processed_total {metrics['grace_events_processed_total']}")
        
        lines.append("# HELP grace_events_failed_total Total events failed")
        lines.append("# TYPE grace_events_failed_total counter")
        lines.append(f"grace_events_failed_total {metrics['grace_events_failed_total']}")
        
        lines.append("# HELP grace_events_expired_total Total events expired (TTL)")
        lines.append("# TYPE grace_events_expired_total counter")
        lines.append(f"grace_events_expired_total {metrics['grace_events_expired_total']}")
        
        lines.append("# HELP grace_events_deduplicated_total Total duplicate events blocked")
        lines.append("# TYPE grace_events_deduplicated_total counter")
        lines.append(f"grace_events_deduplicated_total {metrics['grace_events_deduplicated_total']}")
        
        lines.append("# HELP grace_dlq_events_total Total events sent to DLQ")
        lines.append("# TYPE grace_dlq_events_total counter")
        lines.append(f"grace_dlq_events_total {metrics['grace_dlq_events_total']}")
        
        # Gauges
        lines.append("# HELP grace_pending_queue_size Current pending queue size")
        lines.append("# TYPE grace_pending_queue_size gauge")
        lines.append(f"grace_pending_queue_size {metrics['grace_pending_queue_size']}")
        
        lines.append("# HELP grace_dlq_size Current DLQ size")
        lines.append("# TYPE grace_dlq_size gauge")
        lines.append(f"grace_dlq_size {metrics['grace_dlq_size']}")
        
        # Event counts by type
        lines.append("# HELP grace_events_by_type Events by type")
        lines.append("# TYPE grace_events_by_type counter")
        for event_type, count in metrics['grace_events_by_type'].items():
            safe_type = event_type.replace(".", "_").replace("-", "_")
            lines.append(f'grace_events_by_type{{event_type="{event_type}"}} {count}')
        
        # Latencies
        for operation, percentiles in metrics.get('grace_latency_percentiles', {}).items():
            safe_op = operation.replace(".", "_").replace("-", "_")
            
            lines.append(f'# HELP grace_latency_{safe_op}_ms Latency for {operation}')
            lines.append(f'# TYPE grace_latency_{safe_op}_ms summary')
            
            lines.append(f'grace_latency_{safe_op}_ms{{quantile="0.5"}} {percentiles["p50"]}')
            lines.append(f'grace_latency_{safe_op}_ms{{quantile="0.95"}} {percentiles["p95"]}')
            lines.append(f'grace_latency_{safe_op}_ms{{quantile="0.99"}} {percentiles["p99"]}')
            lines.append(f'grace_latency_{safe_op}_ms_count {percentiles["count"]}')
        
        return "\n".join(lines)


# Global instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
