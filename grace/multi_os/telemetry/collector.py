"""
Multi-OS Telemetry Collector - Metrics, logs, and traces collection.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import json
import uuid
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Collects and manages telemetry data from Multi-OS operations.
    Handles metrics, logs, traces, and events.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Telemetry storage (in-memory for demo, would use proper storage)
        self.metrics = defaultdict(deque)  # metric_name -> deque of measurements
        self.logs = deque(maxlen=10000)    # Recent logs
        self.traces = {}                   # trace_id -> trace data
        self.events = deque(maxlen=5000)   # Recent events
        
        # Aggregation caches
        self.metric_cache = {}
        self.cache_ttl = 60  # seconds
        
        # KPIs being tracked
        self.kpis = {
            "placement_success_rate": 0.0,
            "task_failure_rate": 0.0,
            "mttr_seconds": 0.0,
            "cold_start_ms": 0.0,
            "p95_task_latency": 0.0,
            "gpu_utilization": 0.0,
            "cache_hit_rate": 0.0,
            "sandbox_violations": 0,
            "rollback_count": 0,
            "host_health_slo": 0.0
        }
        
        logger.info("Multi-OS Telemetry Collector initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default telemetry configuration."""
        return {
            "retention": {
                "metrics_days": 7,
                "logs_days": 3,
                "traces_days": 1,
                "events_days": 7
            },
            "aggregation": {
                "metrics_window_seconds": 300,  # 5 minute windows
                "batch_size": 100
            },
            "sampling": {
                "trace_rate": 0.1,  # 10% of traces
                "high_volume_metrics_rate": 0.01  # 1% sampling for high volume
            },
            "alerting": {
                "error_rate_threshold": 0.05,
                "latency_p95_threshold": 5000,  # ms
                "health_score_threshold": 0.7
            }
        }
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, 
                     timestamp: Optional[datetime] = None) -> None:
        """
        Record a metric measurement.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        metric_data = {
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": timestamp.isoformat()
        }
        
        # Store in time series
        self.metrics[name].append(metric_data)
        
        # Maintain size limits
        max_points = 10000
        if len(self.metrics[name]) > max_points:
            self.metrics[name].popleft()
        
        # Update KPIs if relevant
        self._update_kpis(name, value, labels)
    
    def record_log(self, level: str, message: str, host_id: Optional[str] = None,
                  task_id: Optional[str] = None, component: Optional[str] = None,
                  timestamp: Optional[datetime] = None) -> None:
        """
        Record a log entry.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            host_id: Optional host identifier
            task_id: Optional task identifier  
            component: Optional component name
            timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        log_entry = {
            "id": str(uuid.uuid4()),
            "level": level,
            "message": message,
            "host_id": host_id,
            "task_id": task_id,
            "component": component or "multi_os",
            "timestamp": timestamp.isoformat()
        }
        
        self.logs.append(log_entry)
        
        # Also log to Python logger
        python_level = getattr(logging, level, logging.INFO)
        logger.log(python_level, f"[{component or 'multi_os'}] {message}")
    
    def start_trace(self, operation: str, host_id: Optional[str] = None,
                   task_id: Optional[str] = None) -> str:
        """
        Start a distributed trace.
        
        Args:
            operation: Operation being traced
            host_id: Optional host identifier
            task_id: Optional task identifier
            
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        
        trace_data = {
            "trace_id": trace_id,
            "operation": operation,
            "host_id": host_id,
            "task_id": task_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "spans": [],
            "status": "active"
        }
        
        self.traces[trace_id] = trace_data
        return trace_id
    
    def add_span(self, trace_id: str, name: str, duration_ms: Optional[float] = None,
                status: str = "ok", metadata: Optional[Dict] = None) -> None:
        """
        Add a span to a trace.
        
        Args:
            trace_id: Trace identifier
            name: Span name
            duration_ms: Optional span duration
            status: Span status (ok, error)
            metadata: Optional span metadata
        """
        if trace_id not in self.traces:
            return
        
        span = {
            "name": name,
            "duration_ms": duration_ms,
            "status": status,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.traces[trace_id]["spans"].append(span)
    
    def finish_trace(self, trace_id: str, status: str = "completed") -> None:
        """
        Finish a trace.
        
        Args:
            trace_id: Trace identifier
            status: Final trace status
        """
        if trace_id not in self.traces:
            return
        
        trace = self.traces[trace_id]
        trace["status"] = status
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()

        # Calculate total duration
        try:
            start_time = datetime.fromisoformat(trace["started_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            trace["total_duration_ms"] = duration_ms
        except Exception:
            pass
    
    def record_event(self, event_name: str, payload: Dict[str, Any],
                    host_id: Optional[str] = None, timestamp: Optional[datetime] = None) -> str:
        """
        Record a telemetry event.
        
        Args:
            event_name: Event name (e.g., MOS_TASK_COMPLETED)
            payload: Event payload data
            host_id: Optional host identifier
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Event ID
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        event_id = str(uuid.uuid4())

        event = {
            "event_id": event_id,
            "name": event_name,
            "payload": payload,
            "host_id": host_id,
            "timestamp": timestamp.isoformat()
        }

        self.events.append(event)

        # Update metrics based on events
        self._process_event_metrics(event_name, payload, host_id)

        return event_id
    
    def get_metrics_summary(self, metric_name: Optional[str] = None,
                           time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregated metrics summary.
        
        Args:
            metric_name: Optional specific metric name
            time_window: Optional time window in seconds
            
        Returns:
            Metrics summary
        """
        if metric_name and metric_name in self.metrics:
            return self._aggregate_metric(metric_name, time_window)
        
        # Return summary of all metrics
        summary = {
            "total_metrics": len(self.metrics),
            "metrics": {},
            "kpis": self.kpis.copy(),
            "collection_period": {
                "start": None,
                "end": datetime.now(timezone.utc).isoformat()
            }
        }
        
        earliest_time = None
        
        for name, measurements in self.metrics.items():
            if not measurements:
                continue
            
            # Get recent measurements within time window
            recent = self._filter_by_time_window(measurements, time_window)
            
            if not recent:
                continue
            
            values = [m["value"] for m in recent]
            summary["metrics"][name] = {
                "count": len(values),
                "latest": values[-1] if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": sum(values) / len(values) if values else None,
                "latest_timestamp": recent[-1]["timestamp"] if recent else None
            }
            
            # Track earliest time
            if recent and (not earliest_time or recent[0]["timestamp"] < earliest_time):
                earliest_time = recent[0]["timestamp"]
        
        if earliest_time:
            summary["collection_period"]["start"] = earliest_time
        
        return summary
    
    def get_logs(self, level: Optional[str] = None, host_id: Optional[str] = None,
                task_id: Optional[str] = None, component: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get filtered log entries.
        
        Args:
            level: Optional log level filter
            host_id: Optional host filter
            task_id: Optional task filter
            component: Optional component filter
            limit: Maximum number of logs to return
            
        Returns:
            List of matching log entries
        """
        filtered_logs = []
        
        for log_entry in reversed(self.logs):  # Most recent first
            if len(filtered_logs) >= limit:
                break
            
            # Apply filters
            if level and log_entry.get("level") != level:
                continue
            if host_id and log_entry.get("host_id") != host_id:
                continue
            if task_id and log_entry.get("task_id") != task_id:
                continue
            if component and log_entry.get("component") != component:
                continue
            
            filtered_logs.append(log_entry)
        
        return filtered_logs
    
    def get_traces(self, operation: Optional[str] = None, status: Optional[str] = None,
                  limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get filtered traces.
        
        Args:
            operation: Optional operation filter
            status: Optional status filter
            limit: Maximum number of traces to return
            
        Returns:
            List of matching traces
        """
        filtered_traces = []
        
        # Sort traces by start time (most recent first)
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.get("started_at", ""),
            reverse=True
        )
        
        for trace in sorted_traces:
            if len(filtered_traces) >= limit:
                break
            
            # Apply filters
            if operation and trace.get("operation") != operation:
                continue
            if status and trace.get("status") != status:
                continue
            
            filtered_traces.append(trace)
        
        return filtered_traces
    
    def get_events(self, event_name: Optional[str] = None, host_id: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get filtered events.
        
        Args:
            event_name: Optional event name filter
            host_id: Optional host filter
            limit: Maximum number of events to return
            
        Returns:
            List of matching events
        """
        filtered_events = []
        
        for event in reversed(self.events):  # Most recent first
            if len(filtered_events) >= limit:
                break
            
            # Apply filters
            if event_name and event.get("name") != event_name:
                continue
            if host_id and event.get("host_id") != host_id:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def get_kpis(self) -> Dict[str, Any]:
        """Get current KPI values."""
        return {
            "kpis": self.kpis.copy(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "collection_stats": {
                "total_metrics": sum(len(m) for m in self.metrics.values()),
                "total_logs": len(self.logs),
                "total_traces": len(self.traces),
                "total_events": len(self.events)
            }
        }
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """
        Clean up old telemetry data based on retention policy.
        
        Returns:
            Dict with cleanup statistics
        """
        retention = self.config.get("retention", {})
        now = datetime.now(timezone.utc)
        
        stats = {
            "metrics_cleaned": 0,
            "logs_cleaned": 0,
            "traces_cleaned": 0,
            "events_cleaned": 0
        }
        
        # Clean metrics
        metrics_cutoff = now - timedelta(days=retention.get("metrics_days", 7))
        for metric_name, measurements in self.metrics.items():
            original_len = len(measurements)
            # Remove measurements older than cutoff
            while measurements:
                try:
                    mtime = datetime.fromisoformat(measurements[0]["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    # If timestamp malformed, drop it
                    measurements.popleft()
                    continue

                if mtime < metrics_cutoff:
                    measurements.popleft()
                else:
                    break
            stats["metrics_cleaned"] += original_len - len(measurements)
        
        # Clean logs
        logs_cutoff = now - timedelta(days=retention.get("logs_days", 3))
        original_logs = len(self.logs)
        self.logs = deque([
            log for log in self.logs
            if datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc) >= logs_cutoff
        ], maxlen=10000)
        stats["logs_cleaned"] = original_logs - len(self.logs)
        
        # Clean traces
        traces_cutoff = now - timedelta(days=retention.get("traces_days", 1))
        traces_to_remove = []
        for trace_id, trace in self.traces.items():
            try:
                trace_time = datetime.fromisoformat(trace["started_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                continue

            if trace_time < traces_cutoff:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.traces[trace_id]
        stats["traces_cleaned"] = len(traces_to_remove)
        
        # Clean events
        events_cutoff = now - timedelta(days=retention.get("events_days", 7))
        original_events = len(self.events)
        self.events = deque([
            event for event in self.events
            if datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc) >= events_cutoff
        ], maxlen=5000)
        stats["events_cleaned"] = original_events - len(self.events)
        
        logger.info(f"Telemetry cleanup completed: {stats}")
        return stats
    
    def _update_kpis(self, metric_name: str, value: float, labels: Optional[Dict[str, str]]) -> None:
        """Update KPI values based on new metrics."""
        # This is a simplified KPI calculation - in practice would be more sophisticated
        
        if "placement_success" in metric_name:
            self.kpis["placement_success_rate"] = value
        elif "task_failure_rate" in metric_name:
            self.kpis["task_failure_rate"] = value
        elif "cold_start_ms" in metric_name:
            self.kpis["cold_start_ms"] = value
        elif "p95_task_latency" in metric_name:
            self.kpis["p95_task_latency"] = value
        elif "gpu_utilization" in metric_name:
            self.kpis["gpu_utilization"] = value
        elif "cache_hit_rate" in metric_name:
            self.kpis["cache_hit_rate"] = value
        elif "host_health" in metric_name:
            self.kpis["host_health_slo"] = value
    
    def _process_event_metrics(self, event_name: str, payload: Dict[str, Any], host_id: Optional[str]) -> None:
        """Process events to extract metrics."""
        timestamp = datetime.now(timezone.utc)
        
        # Task completion metrics
        if event_name == "MOS_TASK_COMPLETED":
            status = payload.get("status", "unknown")
            duration_ms = payload.get("duration_ms", 0)
            
            # Record task duration
            self.record_metric("task_duration_ms", duration_ms, 
                             {"host_id": host_id, "status": status}, timestamp)
            
            # Record success/failure
            success_value = 1.0 if status == "success" else 0.0
            self.record_metric("task_success", success_value,
                             {"host_id": host_id}, timestamp)
        
        # Host health metrics
        elif event_name == "MOS_HOST_HEALTH":
            health_data = payload
            for metric, value in health_data.items():
                if isinstance(value, (int, float)) and metric != "host_id":
                    self.record_metric(f"host_{metric}", value,
                                     {"host_id": host_id}, timestamp)
        
        # Sandbox violation tracking
        elif "sandbox" in event_name.lower() and "violation" in payload.get("type", "").lower():
            self.kpis["sandbox_violations"] = self.kpis.get("sandbox_violations", 0) + 1
        
        # Rollback tracking
        elif event_name == "ROLLBACK_COMPLETED":
            self.kpis["rollback_count"] = self.kpis.get("rollback_count", 0) + 1
    
    def _aggregate_metric(self, metric_name: str, time_window: Optional[int]) -> Dict[str, Any]:
        """Aggregate a specific metric."""
        measurements = self.metrics.get(metric_name, deque())
        
        if not measurements:
            return {"name": metric_name, "count": 0}
        
        # Filter by time window if specified
        if time_window:
            recent = self._filter_by_time_window(measurements, time_window)
        else:
            recent = list(measurements)
        
        if not recent:
            return {"name": metric_name, "count": 0}
        
        values = [m["value"] for m in recent]
        
        # Calculate percentiles for latency-like metrics
        percentiles = {}
        if "latency" in metric_name.lower() or "duration" in metric_name.lower():
            sorted_values = sorted(values)
            percentiles = {
                "p50": self._percentile(sorted_values, 0.5),
                "p95": self._percentile(sorted_values, 0.95),
                "p99": self._percentile(sorted_values, 0.99)
            }
        
        return {
            "name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "time_window_seconds": time_window,
            **percentiles
        }
    
    def _filter_by_time_window(self, measurements: deque, time_window: Optional[int]) -> List[Dict]:
        """Filter measurements by time window."""
        if not time_window:
            return list(measurements)
        
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=time_window)
        recent = []
        
        for measurement in reversed(measurements):
            try:
                measurement_time = datetime.fromisoformat(
                    measurement["timestamp"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)

                if measurement_time >= cutoff:
                    recent.insert(0, measurement)  # Maintain chronological order
                else:
                    break  # Measurements are ordered, so we can stop
            except (ValueError, KeyError):
                continue
        
        return recent
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        n = len(values)
        index = p * (n - 1)
        
        if index.is_integer():
            return values[int(index)]
        else:
            lower = int(index)
            upper = lower + 1
            weight = index - lower
            return values[lower] * (1 - weight) + values[upper] * weight