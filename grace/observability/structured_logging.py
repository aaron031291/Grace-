"""
Structured logging for Grace system
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import uuid
import logging

# Try to import structlog, fall back to standard logging
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

logger = logging.getLogger(__name__)


@dataclass
class LogContext:
    """Context information for structured logs"""
    timestamp: str
    component: str
    trace_id: str
    user_id: Optional[str] = None
    decision_id: Optional[str] = None
    session_id: Optional[str] = None
    severity: str = "INFO"
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if data['metadata'] is None:
            data['metadata'] = {}
        return data


class StructuredLogger:
    """Structured logger for Grace system"""
    
    def __init__(
        self,
        component: str,
        immutable_logs: Optional[Any] = None,
        log_aggregator: Optional[Any] = None
    ):
        self.component = component
        self.immutable_logs = immutable_logs
        self.log_aggregator = log_aggregator
        self.trace_id = str(uuid.uuid4())
        
        if HAS_STRUCTLOG:
            self.log = structlog.get_logger(component)
        else:
            self.log = logging.getLogger(component)
    
    def info(self, message: str, **kwargs: Any):
        """Log info message"""
        if HAS_STRUCTLOG:
            self.log.info(message, **kwargs)
        else:
            self.log.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs: Any):
        """Log warning message"""
        if HAS_STRUCTLOG:
            self.log.warning(message, **kwargs)
        else:
            self.log.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs: Any):
        """Log error message"""
        if HAS_STRUCTLOG:
            self.log.error(message, **kwargs)
        else:
            self.log.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs: Any):
        """Log debug message"""
        if HAS_STRUCTLOG:
            self.log.debug(message, **kwargs)
        else:
            self.log.debug(message, extra=kwargs)
    
    @contextmanager
    def span(self, operation: str, **kwargs: Any):
        """Create a logging span for tracing operations"""
        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        self.info(f"Starting {operation}", span_id=span_id, **kwargs)
        
        try:
            yield span_id
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.error(
                f"Failed {operation}",
                error=str(e),
                span_id=span_id,
                duration_seconds=duration,
                **kwargs
            )
            raise
        else:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.info(
                f"Completed {operation}",
                span_id=span_id,
                duration_seconds=duration,
                **kwargs
            )
