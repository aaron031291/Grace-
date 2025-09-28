"""
Grace Event Mesh - Production-ready event routing system.

Provides:
- GraceEventBus with pluggable transports (in-memory, Kafka, NATS, Redis)
- TriggerMesh for sub-millisecond routing
- Grace Message Envelope (GME) with idempotency and tracing
- Dead Letter Queue (DLQ) and retry mechanisms
- Configuration management for different environments
"""

from .grace_event_bus import GraceEventBus, RetryConfig, BackpressureConfig, DeadLetterQueue
from .trigger_mesh import TriggerMesh, RoutingRule, RoutingPriority, RoutingMode
from .transports import EventTransport, InMemoryTransport, KafkaTransport, create_transport
from .config import EventMeshConfig, get_config, DEFAULT_CONFIGS
from ..contracts.message_envelope import GraceMessageEnvelope, GMEHeaders, EventTypes

__all__ = [
    # Event Bus
    'GraceEventBus',
    'RetryConfig',
    'BackpressureConfig', 
    'DeadLetterQueue',
    
    # Trigger Mesh
    'TriggerMesh',
    'RoutingRule',
    'RoutingPriority',
    'RoutingMode',
    
    # Transports
    'EventTransport',
    'InMemoryTransport',
    'KafkaTransport',
    'create_transport',
    
    # Configuration
    'EventMeshConfig',
    'get_config',
    'DEFAULT_CONFIGS',
    
    # Message Envelope
    'GraceMessageEnvelope',
    'GMEHeaders',
    'EventTypes'
]