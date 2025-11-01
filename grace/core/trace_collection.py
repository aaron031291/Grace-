"""
Trace Collection System

Instruments all operations to collect complete execution traces.
These traces feed the meta-loop for learning what works and what doesn't.

Every action, tool call, decision, and outcome is captured for analysis.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TraceEventType(Enum):
    """Types of trace events"""
    TASK_START = "task_start"
    TASK_END = "task_end"
    TOOL_CALL = "tool_call"
    MODEL_INFERENCE = "model_inference"
    CONSENSUS = "consensus"
    DECISION = "decision"
    ERROR = "error"
    VERIFICATION = "verification"


@dataclass
class TraceEvent:
    """A single event in an execution trace"""
    event_type: TraceEventType
    timestamp: datetime
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    parent_event_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time()*1000)}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "data": self.data,
            "parent": self.parent_event_id
        }


@dataclass
class TaskTrace:
    """Complete execution trace for a task"""
    trace_id: str
    task_description: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    success: bool = False
    events: List[TraceEvent] = field(default_factory=list)
    final_output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task": self.task_description,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "success": self.success,
            "duration_ms": self.total_duration_ms(),
            "event_count": len(self.events),
            "output": str(self.final_output)[:200] if self.final_output else None,
            "error": self.error,
            "metadata": self.metadata
        }
    
    def total_duration_ms(self) -> Optional[float]:
        """Get total trace duration"""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None
    
    def get_events_by_type(self, event_type: TraceEventType) -> List[TraceEvent]:
        """Filter events by type"""
        return [e for e in self.events if e.event_type == event_type]


class TraceCollector:
    """
    Collects and manages execution traces.
    
    This is critical for learning - we need to know what happened
    in both successful and failed executions.
    """
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend
        self.active_traces: Dict[str, TaskTrace] = {}
        self.completed_traces: List[TaskTrace] = []
        self.max_completed_traces = 1000  # Keep last 1000 traces in memory
        
        logger.info("Trace Collector initialized")
    
    def start_trace(
        self,
        trace_id: str,
        task_description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskTrace:
        """Start a new trace"""
        trace = TaskTrace(
            trace_id=trace_id,
            task_description=task_description,
            started_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.active_traces[trace_id] = trace
        
        # Log start event
        trace.events.append(TraceEvent(
            event_type=TraceEventType.TASK_START,
            timestamp=datetime.utcnow(),
            data={"task": task_description, **trace.metadata}
        ))
        
        logger.debug(f"Started trace: {trace_id}")
        return trace
    
    def log_event(
        self,
        trace_id: str,
        event_type: TraceEventType,
        data: Dict[str, Any],
        duration_ms: Optional[float] = None,
        parent_event_id: Optional[str] = None
    ) -> Optional[TraceEvent]:
        """Log an event to a trace"""
        if trace_id not in self.active_traces:
            logger.warning(f"Trace not found: {trace_id}")
            return None
        
        event = TraceEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            data=data,
            duration_ms=duration_ms,
            parent_event_id=parent_event_id
        )
        
        self.active_traces[trace_id].events.append(event)
        return event
    
    def end_trace(
        self,
        trace_id: str,
        success: bool,
        final_output: Optional[Any] = None,
        error: Optional[str] = None
    ) -> Optional[TaskTrace]:
        """Complete a trace"""
        if trace_id not in self.active_traces:
            logger.warning(f"Trace not found: {trace_id}")
            return None
        
        trace = self.active_traces[trace_id]
        trace.ended_at = datetime.utcnow()
        trace.success = success
        trace.final_output = final_output
        trace.error = error
        
        # Log end event
        trace.events.append(TraceEvent(
            event_type=TraceEventType.TASK_END,
            timestamp=datetime.utcnow(),
            data={
                "success": success,
                "error": error,
                "duration_ms": trace.total_duration_ms()
            }
        ))
        
        # Move to completed
        del self.active_traces[trace_id]
        self.completed_traces.append(trace)
        
        # Trim if too many
        if len(self.completed_traces) > self.max_completed_traces:
            self.completed_traces = self.completed_traces[-self.max_completed_traces:]
        
        # Persist to storage
        if self.storage:
            self.storage.save_trace(trace)
        
        logger.debug(f"Completed trace: {trace_id} (success={success})")
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[TaskTrace]:
        """Get a trace by ID"""
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        for trace in reversed(self.completed_traces):
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_recent_traces(
        self,
        count: int = 10,
        success_only: Optional[bool] = None
    ) -> List[TaskTrace]:
        """Get recent traces"""
        traces = self.completed_traces[:]
        
        if success_only is not None:
            traces = [t for t in traces if t.success == success_only]
        
        return list(reversed(traces))[:count]
    
    def get_failure_traces(self, count: int = 10) -> List[TaskTrace]:
        """Get recent failures for analysis"""
        return self.get_recent_traces(count, success_only=False)
    
    def analyze_traces(
        self,
        traces: Optional[List[TaskTrace]] = None
    ) -> Dict[str, Any]:
        """Analyze a set of traces"""
        if traces is None:
            traces = self.completed_traces
        
        if not traces:
            return {"total": 0}
        
        successful = [t for t in traces if t.success]
        failed = [t for t in traces if not t.success]
        
        # Compute aggregates
        durations = [t.total_duration_ms() for t in traces if t.total_duration_ms()]
        
        analysis = {
            "total_traces": len(traces),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(traces) if traces else 0.0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "max_duration_ms": max(durations) if durations else 0.0,
            "min_duration_ms": min(durations) if durations else 0.0
        }
        
        # Common error patterns
        error_counts = {}
        for trace in failed:
            if trace.error:
                error_type = trace.error.split(":")[0] if ":" in trace.error else trace.error[:50]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        analysis["common_errors"] = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return analysis


# Global trace collector instance
_global_collector: Optional[TraceCollector] = None


def get_trace_collector() -> TraceCollector:
    """Get global trace collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = TraceCollector()
    return _global_collector


# Convenience functions
def start_trace(trace_id: str, task: str, metadata: Dict[str, Any] = None) -> TaskTrace:
    """Start a new trace"""
    return get_trace_collector().start_trace(trace_id, task, metadata)


def log_event(trace_id: str, event_type: TraceEventType, data: Dict[str, Any]):
    """Log an event"""
    return get_trace_collector().log_event(trace_id, event_type, data)


def end_trace(trace_id: str, success: bool, output: Any = None, error: str = None):
    """End a trace"""
    return get_trace_collector().end_trace(trace_id, success, output, error)


if __name__ == "__main__":
    # Demo usage
    print("📝 Trace Collection Demo\n")
    
    collector = TraceCollector()
    
    # Trace a successful task
    trace1 = collector.start_trace(
        trace_id="trace_001",
        task_description="Calculate 7 * 8",
        metadata={"model": "gpt-4"}
    )
    
    collector.log_event(
        "trace_001",
        TraceEventType.MODEL_INFERENCE,
        {"prompt": "What is 7 * 8?", "response": "56"},
        duration_ms=150
    )
    
    collector.end_trace("trace_001", success=True, final_output="56")
    
    # Trace a failed task
    trace2 = collector.start_trace(
        trace_id="trace_002",
        task_description="Solve impossible problem"
    )
    
    collector.log_event(
        "trace_002",
        TraceEventType.ERROR,
        {"error_type": "TimeoutError", "message": "Operation timed out"}
    )
    
    collector.end_trace("trace_002", success=False, error="TimeoutError: Operation timed out")
    
    # Analyze
    analysis = collector.analyze_traces()
    print("Analysis:")
    print(f"  Total traces: {analysis['total_traces']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")
    print(f"  Avg duration: {analysis['avg_duration_ms']:.1f}ms")
    print(f"  Common errors: {analysis['common_errors']}")
