"""
Pytest configuration for Grace system tests
"""

import pytest
import logging

logging.basicConfig(level=logging.INFO)


def pytest_configure(config):
    """Configure pytest with Grace plugins"""
    from grace.integration.event_bus import EventBus
    from grace.testing.pytest_plugin import TestQualityPlugin
    
    # Create event bus
    event_bus = EventBus()
    
    # Attach to config for plugin access
    config.event_publisher = event_bus
    
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
