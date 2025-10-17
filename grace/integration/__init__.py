"""
Integration Layer - Connects various Grace components
"""

from .event_bus import EventBus, Event
from .quorum_integration import QuorumIntegration

__all__ = [
    'EventBus',
    'Event',
    'QuorumIntegration'
]

__version__ = '1.0.0'
