"""
Pytest configuration for Grace tests
"""

import sys
from pathlib import Path

# Add parent directory to path so 'grace' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import asyncio

# Now import Grace modules
from grace.integration.event_bus import EventBus


def pytest_configure(config):
    """Configure pytest with Grace plugins"""
    from grace.testing.pytest_plugin import TestQualityPlugin
    
    # Plugin is registered automatically
    logging.info("Pytest configured with Grace test quality monitoring")


@pytest.fixture
def event_bus():
    """Provide event bus for tests"""
    from grace.integration.event_bus import EventBus
    return EventBus()


@pytest.fixture
def quality_monitor(event_bus):
    """Provide quality monitor for tests"""
    from grace.testing.quality_monitor import TestQualityMonitor
    return TestQualityMonitor(event_publisher=event_bus)


@pytest.fixture
def avn_core(event_bus):
    """Provide AVN core for tests"""
    from grace.avn.enhanced_core import EnhancedAVNCore
    return EnhancedAVNCore(event_publisher=event_bus)
