"""
Structured logging for Grace system
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


class StructuredLogger:
    """Structured logger with correlation IDs"""
    
    def __init__(
        self,
        component: str,
        correlation_id: Optional[str] = None
    ):
        self.component = component
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger(component)
    
    def _format_log(
        self,
        level: str,
        message: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Format log as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "component": self.component,
            "correlation_id": self.correlation_id,
            "message": message,
        }
        
        # Add extra fields
        if kwargs:
            log_entry["extra"] = kwargs
        
        return log_entry
    
    def _log(self, level: str, message: str, **kwargs: Any):
        """Log structured message"""
        log_entry = self._format_log(level, message, **kwargs)
        log_line = json.dumps(log_entry)
        
        # Log at appropriate level
        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, log_line)
    
    def debug(self, message: str, **kwargs: Any):
        """Log debug message"""
        self._log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any):
        """Log info message"""
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any):
        """Log warning message"""
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any):
        """Log error message"""
        self._log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any):
        """Log critical message"""
        self._log("critical", message, **kwargs)
    
    def with_correlation_id(self, correlation_id: str) -> "StructuredLogger":
        """Create new logger with different correlation ID"""
        return StructuredLogger(self.component, correlation_id)


def setup_structured_logging(log_level: str = "INFO"):
    """Setup structured JSON logging globally"""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add JSON formatter
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(handler)
