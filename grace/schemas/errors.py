"""
Grace error hierarchy
"""

from typing import Optional, Any


class GraceError(Exception):
    """Base exception for Grace system"""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EventError(GraceError):
    """Event processing errors"""
    pass


class ValidationError(GraceError):
    """Event validation errors"""
    pass


class TimeoutError(GraceError):
    """Event timeout errors"""
    
    def __init__(self, message: str, timeout_seconds: float, details: Optional[dict] = None):
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds


class DuplicateEventError(GraceError):
    """Duplicate event detected (idempotency violation)"""
    
    def __init__(self, event_id: str, idempotency_key: str, details: Optional[dict] = None):
        message = f"Duplicate event detected: {event_id}"
        super().__init__(message, details)
        self.event_id = event_id
        self.idempotency_key = idempotency_key


class BackpressureError(GraceError):
    """Backpressure threshold exceeded"""
    
    def __init__(self, queue_size: int, max_size: int, details: Optional[dict] = None):
        message = f"Backpressure: queue size {queue_size} exceeds max {max_size}"
        super().__init__(message, details)
        self.queue_size = queue_size
        self.max_size = max_size


class DeadLetterError(GraceError):
    """Event sent to dead letter queue"""
    
    def __init__(self, event_id: str, reason: str, details: Optional[dict] = None):
        message = f"Event {event_id} sent to DLQ: {reason}"
        super().__init__(message, details)
        self.event_id = event_id
        self.reason = reason


class TTLExpiredError(GraceError):
    """Event TTL expired"""
    
    def __init__(self, event_id: str, ttl_seconds: int, details: Optional[dict] = None):
        message = f"Event {event_id} expired after {ttl_seconds}s"
        super().__init__(message, details)
        self.event_id = event_id
        self.ttl_seconds = ttl_seconds
