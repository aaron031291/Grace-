"""
Grace Trace (gtrace) - Distributed tracing system for Grace operations.

Provides comprehensive tracing capabilities specifically designed for Grace's
memory operations, copilot integrations, and cross-system observability.
"""

from .tracer import GTracer, TraceContext, TraceSpan
from .memory_tracer import MemoryTracer
from .collector import GTraceCollector

__all__ = ["GTracer", "TraceContext", "TraceSpan", "MemoryTracer", "GTraceCollector"]

# Global tracer instance
_global_tracer = None


def get_tracer() -> GTracer:
    """Get the global Grace tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = GTracer()
    return _global_tracer


def set_tracer(tracer: GTracer) -> None:
    """Set the global Grace tracer instance."""
    global _global_tracer
    _global_tracer = tracer
