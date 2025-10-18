"""
Structured logging with consistent format for Grace system
"""

import logging
import json
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import structlog
from contextlib import contextmanager

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
    """
    Structured logger for Grace system
    
    Features:
    - Consistent log format across all components
    - Automatic trace ID generation
    - Integration with immutable logs
    - Export to log aggregators (Loki, ELK)
    """
    
    def __init__(
        self,
        component: str,
        immutable_logs=None,
        log_aggregator=None
    ):
        """
        Initialize structured logger
        
        Args:
            component: Component name (e.g., 'auth', 'governance', 'mldl')
            immutable_logs: ImmutableLogs instance for audit trail
            log_aggregator: Log aggregator client (Loki, Elasticsearch)
        """
        self.component = component
        self.immutable_logs = immutable_logs
        self.log_aggregator = log_aggregator
        self.trace_id = str(uuid.uuid4())
        
        # Setup structlog
        self.log = structlog.get_logger(component)
    
    def _create_context(
        self,
        message: str,
        severity: str,
        user_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> LogContext:
        """Create log context"""
        return LogContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            component=self.component,
            trace_id=self.trace_id,
            user_id=user_id,
            decision_id=decision_id,
            session_id=session_id,
            severity=severity,
            message=message,
            metadata=kwargs
        )
    
    def info(
        self,
        message: str,
        user_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        **kwargs
    ):
        """Log info message"""
        context = self._create_context(
            message, "INFO", user_id, decision_id, **kwargs
        )
        self._emit_log(context)
    
    def warning(
        self,
        message: str,
        user_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        **kwargs
    ):
        """Log warning message"""
        context = self._create_context(
            message, "WARNING", user_id, decision_id, **kwargs
        )
        self._emit_log(context)
    
    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        user_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        **kwargs
    ):
        """Log error message"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        
        context = self._create_context(
            message, "ERROR", user_id, decision_id, **kwargs
        )
        self._emit_log(context)
    
    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        user_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        **kwargs
    ):
        """Log critical message"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        
        context = self._create_context(
            message, "CRITICAL", user_id, decision_id, **kwargs
        )
        self._emit_log(context)
    
    def _emit_log(self, context: LogContext):
        """Emit log to all configured destinations"""
        log_data = context.to_dict()
        
        # 1. Emit to structlog
        log_func = getattr(self.log, context.severity.lower())
        log_func(
            context.message,
            **{k: v for k, v in log_data.items() if k not in ['message', 'severity']}
        )
        
        # 2. Send to immutable logs (async for critical/error)
        if self.immutable_logs and context.severity in ['ERROR', 'CRITICAL']:
            try:
                self.immutable_logs.log_constitutional_operation(
                    operation_type=f"log_{context.severity.lower()}",
                    actor=f"component:{self.component}",
                    action={"message": context.message, "metadata": context.metadata},
                    result={"logged": True},
                    severity=context.severity.lower(),
                    tags=[self.component, context.severity.lower(), "audit"]
                )
            except Exception as e:
                logger.error(f"Failed to write to immutable logs: {e}")
        
        # 3. Send to log aggregator
        if self.log_aggregator:
            try:
                self._send_to_aggregator(log_data)
            except Exception as e:
                logger.error(f"Failed to send to log aggregator: {e}")
    
    def _send_to_aggregator(self, log_data: Dict[str, Any]):
        """Send log to aggregator (Loki, Elasticsearch)"""
        # Loki format
        if hasattr(self.log_aggregator, 'push'):
            self.log_aggregator.push({
                "streams": [{
                    "stream": {
                        "component": self.component,
                        "severity": log_data['severity']
                    },
                    "values": [[
                        str(int(datetime.fromisoformat(log_data['timestamp']).timestamp() * 1e9)),
                        json.dumps(log_data)
                    ]]
                }]
            })
        
        # Elasticsearch format
        elif hasattr(self.log_aggregator, 'index'):
            self.log_aggregator.index(
                index=f"grace-logs-{datetime.now():%Y.%m.%d}",
                body=log_data
            )
    
    @contextmanager
    def span(self, operation: str, **kwargs):
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
                error=e,
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
    
    def set_trace_id(self, trace_id: str):
        """Set custom trace ID"""
        self.trace_id = trace_id
    
    def new_trace(self) -> str:
        """Generate new trace ID"""
        self.trace_id = str(uuid.uuid4())
        return self.trace_id


def setup_logging(
    log_level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None
) -> None:
    """Setup global logging configuration"""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        logging.root.addHandler(file_handler)
    
    logger.info(f"Logging configured: level={log_level}, json={json_output}")
