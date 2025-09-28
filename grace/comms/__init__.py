"""
Grace Communications Module - Implementation of the Grace Message Envelope standard.

This module provides the core implementation for the Grace Communications Schema pack,
offering standardized message envelopes, routing, kernel integration, DLQ, and deduplication.
"""

from .envelope import GraceMessageEnvelope, create_envelope, MessageKind, Priority, QoSClass
from .validator import validate_envelope, validate_payload
from .router import MessageRouter, Route, RouteStrategy, RouteHealth
from .dlq import DeadLetterQueue, DLQReason, DLQEntry
from .dedupe import MessageDeduplicator, DeduplicationEntry

__version__ = "1.0.0"

__all__ = [
    "GraceMessageEnvelope",
    "create_envelope", 
    "MessageKind",
    "Priority", 
    "QoSClass",
    "validate_envelope",
    "validate_payload",
    "MessageRouter",
    "Route",
    "RouteStrategy", 
    "RouteHealth",
    "DeadLetterQueue",
    "DLQReason",
    "DLQEntry",
    "MessageDeduplicator",
    "DeduplicationEntry"
]