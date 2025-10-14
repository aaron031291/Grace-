"""
GTrace Collector - Collects, stores, and manages trace data.
"""

import json
import logging
from datetime import datetime
from ..utils.time import now_utc
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

from .tracer import TraceSpan, SpanStatus, SpanKind

logger = logging.getLogger(__name__)


class GTraceCollector:
    """Collects and manages Grace trace data."""

    def __init__(self, max_traces: int = 10000, retention_hours: int = 24):
        self.max_traces = max_traces
        self.retention_hours = retention_hours

        # Storage
        self._spans: Dict[str, TraceSpan] = {}  # span_id -> span
        self._traces: Dict[str, List[str]] = defaultdict(list)  # trace_id -> [span_ids]
        self._span_queue = deque()  # For FIFO eviction

        # Indices for fast queries
        self._spans_by_operation = defaultdict(list)  # operation -> [span_ids]
        self._spans_by_service = defaultdict(list)  # service -> [span_ids]
        self._spans_by_user = defaultdict(list)  # user_id -> [span_ids]
        self._spans_by_session = defaultdict(list)  # session_id -> [span_ids]

        # Statistics
        self._stats = {
            "spans_collected": 0,
            "traces_collected": 0,
            "spans_evicted": 0,
            "last_collection": None,
        }

    def collect_span(self, span: TraceSpan) -> None:
        """Collect a completed span."""
        span_id = span.context.span_id
        trace_id = span.context.trace_id

        # Store span
        self._spans[span_id] = span
        self._traces[trace_id].append(span_id)
        self._span_queue.append(span_id)

        # Update indices
        self._spans_by_operation[span.operation_name].append(span_id)

        service = span.tags.get("service.name", "unknown")
        self._spans_by_service[service].append(span_id)

        user_id = span.tags.get("user.id")
        if user_id:
            self._spans_by_user[user_id].append(span_id)

        session_id = span.tags.get("session.id")
        if session_id:
            self._spans_by_session[session_id].append(span_id)

        # Update stats
        self._stats["spans_collected"] += 1
        if len(self._traces[trace_id]) == 1:  # First span in trace
            self._stats["traces_collected"] += 1
        self._stats["last_collection"] = now_utc().isoformat()

        # Evict old spans if needed
        self._evict_if_needed()

        logger.debug(f"Collected span {span.operation_name} ({span_id})")

    def _evict_if_needed(self) -> None:
        """Evict old spans if storage limits exceeded."""
        while len(self._spans) > self.max_traces:
            self._evict_oldest_span()

    def _evict_oldest_span(self) -> None:
        """Evict the oldest span."""
        if not self._span_queue:
            return

        span_id = self._span_queue.popleft()
        if span_id not in self._spans:
            return

        span = self._spans[span_id]
        trace_id = span.context.trace_id

        # Remove from storage
        del self._spans[span_id]

        # Remove from trace
        if span_id in self._traces[trace_id]:
            self._traces[trace_id].remove(span_id)

        # Clean up empty traces
        if not self._traces[trace_id]:
            del self._traces[trace_id]

        # Remove from indices
        self._remove_from_indices(span_id, span)

        self._stats["spans_evicted"] += 1

    def _remove_from_indices(self, span_id: str, span: TraceSpan) -> None:
        """Remove span from all indices."""
        # Operation index
        if span_id in self._spans_by_operation[span.operation_name]:
            self._spans_by_operation[span.operation_name].remove(span_id)

        # Service index
        service = span.tags.get("service.name", "unknown")
        if span_id in self._spans_by_service[service]:
            self._spans_by_service[service].remove(span_id)

        # User index
        user_id = span.tags.get("user.id")
        if user_id and span_id in self._spans_by_user[user_id]:
            self._spans_by_user[user_id].remove(span_id)

        # Session index
        session_id = span.tags.get("session.id")
        if session_id and span_id in self._spans_by_session[session_id]:
            self._spans_by_session[session_id].remove(span_id)

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        span_ids = self._traces.get(trace_id, [])
        return [self._spans[sid] for sid in span_ids if sid in self._spans]

    def get_spans(
        self,
        operation_name: Optional[str] = None,
        service_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[SpanStatus] = None,
        kind: Optional[SpanKind] = None,
        time_range: Optional[tuple] = None,
        limit: int = 100,
    ) -> List[TraceSpan]:
        """Query spans with filters."""
        span_ids = set()

        # Apply filters to get candidate span IDs
        if operation_name:
            span_ids.update(self._spans_by_operation.get(operation_name, []))
        elif service_name:
            span_ids.update(self._spans_by_service.get(service_name, []))
        elif user_id:
            span_ids.update(self._spans_by_user.get(user_id, []))
        elif session_id:
            span_ids.update(self._spans_by_session.get(session_id, []))
        else:
            span_ids.update(self._spans.keys())

        # Filter spans by additional criteria
        filtered_spans = []
        for span_id in span_ids:
            if span_id not in self._spans:
                continue

            span = self._spans[span_id]

            # Status filter
            if status and span.status != status:
                continue

            # Kind filter
            if kind and span.kind != kind:
                continue

            # Time range filter
            if time_range:
                start_time, end_time = time_range
                if span.start_time < start_time or span.start_time > end_time:
                    continue

            filtered_spans.append(span)

        # Sort by start time (newest first) and limit
        filtered_spans.sort(key=lambda s: s.start_time, reverse=True)
        return filtered_spans[:limit]

    def get_memory_operations(self, limit: int = 100) -> List[TraceSpan]:
        """Get spans for memory operations."""
        return self.get_spans(kind=SpanKind.MEMORY_OPERATION, limit=limit)

    def get_copilot_operations(self, limit: int = 100) -> List[TraceSpan]:
        """Get spans for copilot operations."""
        return self.get_spans(kind=SpanKind.COPILOT_OPERATION, limit=limit)

    def get_error_spans(self, limit: int = 50) -> List[TraceSpan]:
        """Get spans with errors."""
        return self.get_spans(status=SpanStatus.ERROR, limit=limit)

    def get_slow_operations(
        self, threshold_ms: float = 1000, limit: int = 50
    ) -> List[TraceSpan]:
        """Get slow operations above threshold."""
        all_spans = self.get_spans(limit=1000)  # Get larger sample
        slow_spans = [
            span
            for span in all_spans
            if span.duration_ms and span.duration_ms > threshold_ms
        ]
        slow_spans.sort(key=lambda s: s.duration_ms or 0, reverse=True)
        return slow_spans[:limit]

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a trace."""
        spans = self.get_trace(trace_id)
        if not spans:
            return {"error": "Trace not found"}

        # Sort spans by start time
        spans.sort(key=lambda s: s.start_time)

        # Calculate trace metrics
        trace_start = min(s.start_time for s in spans)
        trace_end = max(s.end_time or s.start_time for s in spans)
        trace_duration = (trace_end - trace_start) * 1000  # ms

        # Count operations and errors
        operations = {}
        error_count = 0
        for span in spans:
            op_name = span.operation_name
            if op_name not in operations:
                operations[op_name] = {"count": 0, "avg_duration": 0.0}
            operations[op_name]["count"] += 1

            if span.status == SpanStatus.ERROR:
                error_count += 1

        # Calculate average durations
        for op_name in operations:
            op_spans = [s for s in spans if s.operation_name == op_name]
            durations = [s.duration_ms for s in op_spans if s.duration_ms]
            if durations:
                operations[op_name]["avg_duration"] = sum(durations) / len(durations)

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "duration_ms": trace_duration,
            "error_count": error_count,
            "success_rate": (len(spans) - error_count) / len(spans) if spans else 0,
            "operations": operations,
            "start_time": trace_start,
            "end_time": trace_end,
            "root_operation": spans[0].operation_name if spans else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        now = now_utc()

        # Calculate storage stats
        total_spans = len(self._spans)
        total_traces = len(self._traces)

        # Calculate recent activity (last hour)
        hour_ago = now.timestamp() - 3600
        recent_spans = [
            span for span in self._spans.values() if span.start_time > hour_ago
        ]

        # Calculate error rates
        error_spans = [s for s in recent_spans if s.status == SpanStatus.ERROR]
        error_rate = len(error_spans) / len(recent_spans) if recent_spans else 0

        # Memory operations stats
        memory_spans = [
            s
            for s in recent_spans
            if s.kind in [SpanKind.MEMORY_OPERATION, SpanKind.COPILOT_OPERATION]
        ]

        # Top operations
        operation_counts = defaultdict(int)
        for span in recent_spans:
            operation_counts[span.operation_name] += 1

        top_operations = sorted(
            operation_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            **self._stats,
            "current_stats": {
                "total_spans": total_spans,
                "total_traces": total_traces,
                "recent_spans_1h": len(recent_spans),
                "recent_errors_1h": len(error_spans),
                "error_rate_1h": error_rate,
                "memory_operations_1h": len(memory_spans),
                "top_operations": top_operations,
                "storage_utilization": total_spans / self.max_traces,
            },
            "configuration": {
                "max_traces": self.max_traces,
                "retention_hours": self.retention_hours,
            },
        }

    def cleanup_old_data(self) -> Dict[str, int]:
        """Clean up old data based on retention policy."""
        cutoff_time = now_utc().timestamp() - (self.retention_hours * 3600)

        old_span_ids = [
            span_id
            for span_id, span in self._spans.items()
            if span.start_time < cutoff_time
        ]

        cleaned_spans = 0
        cleaned_traces = 0

        for span_id in old_span_ids:
            if span_id in self._spans:
                span = self._spans[span_id]
                trace_id = span.context.trace_id

                # Remove span
                del self._spans[span_id]
                self._remove_from_indices(span_id, span)
                cleaned_spans += 1

                # Remove from trace
                if span_id in self._traces[trace_id]:
                    self._traces[trace_id].remove(span_id)

                # Clean up empty trace
                if not self._traces[trace_id]:
                    del self._traces[trace_id]
                    cleaned_traces += 1

                # Remove from queue
                if span_id in self._span_queue:
                    temp_queue = deque()
                    while self._span_queue:
                        item = self._span_queue.popleft()
                        if item != span_id:
                            temp_queue.append(item)
                    self._span_queue = temp_queue

        logger.info(f"Cleaned up {cleaned_spans} spans and {cleaned_traces} traces")

        return {
            "spans_cleaned": cleaned_spans,
            "traces_cleaned": cleaned_traces,
            "cutoff_time": cutoff_time,
        }

    def export_traces(
        self, trace_ids: Optional[List[str]] = None, format: str = "json"
    ) -> str:
        """Export traces in specified format."""
        if trace_ids:
            traces_data = {tid: self.get_trace(tid) for tid in trace_ids}
        else:
            traces_data = {tid: self.get_trace(tid) for tid in self._traces.keys()}

        # Convert spans to dictionaries
        export_data = {}
        for trace_id, spans in traces_data.items():
            export_data[trace_id] = [span.to_dict() for span in spans]

        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
