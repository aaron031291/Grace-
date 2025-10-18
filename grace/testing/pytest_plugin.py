"""
Pytest plugin for test quality monitoring with event emission
"""

import pytest
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TestQualityPlugin:
    """Pytest plugin that integrates with TestQualityMonitor"""
    
    def __init__(self, quality_monitor=None, event_publisher=None):
        self.quality_monitor = quality_monitor
        self.event_publisher = event_publisher
        
        if not self.quality_monitor:
            try:
                from grace.testing.quality_monitor import TestQualityMonitor
                self.quality_monitor = TestQualityMonitor(event_publisher=event_publisher)
                logger.info("Initialized TestQualityMonitor in pytest plugin")
            except Exception as e:
                logger.warning(f"Could not initialize TestQualityMonitor: {e}")
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Hook to capture test results"""
        outcome = yield
        report = outcome.get_result()
        
        if report.when == "call":
            self._record_test_result(item, report)
    
    def _record_test_result(self, item, report):
        """Record test result to quality monitor"""
        if not self.quality_monitor:
            return
        
        from grace.testing.quality_monitor import TestResult
        
        # Determine status
        if report.passed:
            status = "passed"
            error_message = None
        elif report.failed:
            status = "failed"
            error_message = str(report.longrepr) if hasattr(report, 'longrepr') else "Test failed"
        elif report.skipped:
            status = "skipped"
            error_message = str(report.longrepr) if hasattr(report, 'longrepr') else "Test skipped"
        else:
            status = "error"
            error_message = "Unknown error"
        
        # Create test result
        test_result = TestResult(
            test_id=item.nodeid,
            status=status,
            duration=report.duration,
            error_message=error_message,
            metadata={
                "file": str(item.fspath),
                "function": item.name,
                "markers": [m.name for m in item.iter_markers()]
            }
        )
        
        # Record to monitor
        self.quality_monitor.record_test(test_result)
        
        logger.debug(f"Recorded test: {item.nodeid} - {status}")


def pytest_configure(config):
    """Register the plugin"""
    # Try to get event publisher from config
    event_publisher = getattr(config, 'event_publisher', None)
    
    plugin = TestQualityPlugin(event_publisher=event_publisher)
    config.pluginmanager.register(plugin, "test_quality_plugin")
