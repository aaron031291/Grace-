"""
Grace Tracer - Core distributed tracing implementation.
"""
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, AsyncContextManager
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager, contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Types of trace spans."""
    INTERNAL = "internal"
    SERVER = "server"  
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    MEMORY_OPERATION = "memory_operation"
    COPILOT_OPERATION = "copilot_operation"


class SpanStatus(Enum):
    """Span completion status."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class TraceContext:
    """Trace context for correlation across operations."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    flags: int = 0
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure trace_id and span_id are set."""
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        if not self.span_id:
            self.span_id = str(uuid.uuid4())
    
    def create_child_context(self) -> 'TraceContext':
        """Create a child context with same trace ID."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
            flags=self.flags,
            baggage=self.baggage.copy()
        )


@dataclass
class TraceSpan:
    """Individual span in a trace."""
    context: TraceContext
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    kind: SpanKind = SpanKind.INTERNAL
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span."""
        self.tags[key] = value
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set multiple tags on the span."""
        self.tags.update(tags)
    
    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Add a log event to the span."""
        log_entry = {
            "timestamp": time.time(),
            "event": event,
            "payload": payload or {}
        }
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception) -> None:
        """Mark span as error and record exception."""
        self.status = SpanStatus.ERROR
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))
        self.log("error", {
            "error.object": str(error),
            "error.kind": type(error).__name__
        })
    
    def finish(self) -> None:
        """Mark span as finished."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "kind": self.kind.value,
            "tags": self.tags,
            "logs": self.logs
        }


class GTracer:
    """Grace distributed tracer."""
    
    def __init__(self, service_name: str = "grace", 
                 collector_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.collector_endpoint = collector_endpoint
        self._active_spans: Dict[str, TraceSpan] = {}
        self._finished_spans: List[TraceSpan] = []
        self._max_finished_spans = 10000
        
    def start_trace(self, operation_name: str, 
                   parent_context: Optional[TraceContext] = None,
                   kind: SpanKind = SpanKind.INTERNAL,
                   tags: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """Start a new trace span."""
        if parent_context:
            context = parent_context.create_child_context()
        else:
            context = TraceContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())
            )
        
        span = TraceSpan(
            context=context,
            operation_name=operation_name,
            start_time=time.time(),
            kind=kind
        )
        
        if tags:
            span.set_tags(tags)
        
        # Set default service tags
        span.set_tag("service.name", self.service_name)
        span.set_tag("service.version", "1.0.0")
        
        self._active_spans[span.context.span_id] = span
        
        logger.debug(f"Started span {operation_name} ({span.context.span_id})")
        return span
    
    def finish_span(self, span: TraceSpan) -> None:
        """Finish a span and move it to finished spans."""
        span.finish()
        
        # Remove from active spans
        if span.context.span_id in self._active_spans:
            del self._active_spans[span.context.span_id]
        
        # Add to finished spans
        self._finished_spans.append(span)
        
        # Limit finished spans to prevent memory leaks
        if len(self._finished_spans) > self._max_finished_spans:
            self._finished_spans = self._finished_spans[-self._max_finished_spans:]
        
        logger.debug(f"Finished span {span.operation_name} ({span.context.span_id}) "
                    f"in {span.duration_ms:.2f}ms")
    
    @contextmanager
    def span(self, operation_name: str,
             parent_context: Optional[TraceContext] = None,
             kind: SpanKind = SpanKind.INTERNAL,
             tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        span = self.start_trace(operation_name, parent_context, kind, tags)
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
    
    @asynccontextmanager
    async def async_span(self, operation_name: str,
                        parent_context: Optional[TraceContext] = None,
                        kind: SpanKind = SpanKind.INTERNAL,
                        tags: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing operations."""
        span = self.start_trace(operation_name, parent_context, kind, tags)
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
    
    def get_active_spans(self) -> List[TraceSpan]:
        """Get all currently active spans."""
        return list(self._active_spans.values())
    
    def get_finished_spans(self, trace_id: Optional[str] = None,
                          limit: int = 100) -> List[TraceSpan]:
        """Get finished spans, optionally filtered by trace ID."""
        spans = self._finished_spans
        
        if trace_id:
            spans = [s for s in spans if s.context.trace_id == trace_id]
        
        return spans[-limit:] if limit else spans
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a specific trace."""
        # Check active spans
        active = [s for s in self._active_spans.values() 
                 if s.context.trace_id == trace_id]
        
        # Check finished spans  
        finished = [s for s in self._finished_spans
                   if s.context.trace_id == trace_id]
        
        return active + finished
    
    def clear_finished_spans(self) -> int:
        """Clear finished spans and return count cleared."""
        count = len(self._finished_spans)
        self._finished_spans.clear()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        return {
            "service_name": self.service_name,
            "active_spans": len(self._active_spans),
            "finished_spans": len(self._finished_spans),
            "max_finished_spans": self._max_finished_spans,
            "collector_endpoint": self.collector_endpoint
        }