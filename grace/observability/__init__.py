"""
Grace Observability - Logging, Metrics, and Monitoring
"""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class LogContext:
    """A dataclass to hold contextual information for logging."""
    request_id: str = ""
    user_id: str = ""
    extra_data: Dict[str, Any] = field(default_factory=dict)

from .structured_logging import StructuredLogger, setup_logging
from .metrics import get_metrics_collector, PerformanceMetrics

__all__ = [
    "StructuredLogger",
    "LogContext",
    "setup_logging",
    "get_metrics_collector",
    "PerformanceMetrics",
]
