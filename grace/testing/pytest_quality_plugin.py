"""
Pytest plugin for Grace Test Quality Monitoring.

This plugin:
1. Hooks into pytest execution to capture test results
2. Feeds results to TestQualityMonitor
3. Reports quality scores instead of raw pass/fail counts
4. Triggers self-healing on quality degradation

Usage:
    Add to conftest.py:
    
    pytest_plugins = ['grace.testing.pytest_quality_plugin']
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .test_quality_monitor import (
    TestQualityMonitor,
    TestResult,
    ErrorSeverity,
    ComponentQualityStatus
)

logger = logging.getLogger(__name__)


class GraceTestQualityPlugin:
    """Pytest plugin for Grace test quality monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.quality_monitor: Optional[TestQualityMonitor] = None
        self.test_results: Dict[str, TestResult] = {}
        
        # Component mapping (test path -> component_id)
        self.component_map = self._build_component_map()
        
        # Stats for final report
        self.session_start = datetime.now()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
    
    def _build_component_map(self) -> Dict[str, str]:
        """Map test file paths to component IDs."""
        return {
            "grace/mcp/tests": "mcp_framework",
            "demo_and_tests/comprehensive_governance_test": "comprehensive_e2e",
            "grace/ingress_kernel": "ingress_kernel",
            "grace/intelligence": "intelligence_kernel",
            "grace/learning_kernel": "learning_kernel",
            "grace/orchestration": "orchestration_kernel",
            "grace/resilience_kernel": "resilience_kernel",
            "grace/governance": "governance_engine",
            "grace/core": "core_systems",
            "grace/gtrace": "tracing_system",
            "test_contract_compliance": "contract_compliance",
            "test_gtrace": "tracing_system",
            "grace/immune": "immune_system",
            "tests/": "general_tests"
        }
    
    def _get_component_id(self, nodeid: str) -> str:
        """Extract component ID from test node ID."""
        for path_prefix, component_id in self.component_map.items():
            if path_prefix in nodeid:
                return component_id
        return "unknown_component"
    
    def _determine_error_severity(self, report) -> ErrorSeverity:
        """Determine error severity from test report."""
        if not hasattr(report, 'longrepr'):
            return ErrorSeverity.MEDIUM
        
        longrepr_str = str(report.longrepr).lower()
        
        # Critical: System crashes, segfaults, OOM
        if any(keyword in longrepr_str for keyword in ['segfault', 'memory error', 'fatal', 'killed']):
            return ErrorSeverity.CRITICAL
        
        # High: Import errors, unexpected exceptions, integration failures
        if any(keyword in longrepr_str for keyword in ['import error', 'module not found', 'connection error']):
            return ErrorSeverity.HIGH
        
        # Low: Deprecation warnings, style issues
        if any(keyword in longrepr_str for keyword in ['deprecat', 'warning', 'futurewarning']):
            return ErrorSeverity.LOW
        
        # Default: Medium (assertion failures, expected exceptions)
        return ErrorSeverity.MEDIUM
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(self, config):
        """Initialize quality monitor on pytest startup."""
        try:
            # Try to import and initialize KPITrustMonitor
            from grace.core.kpi_trust_monitor import KPITrustMonitor
            kpi_monitor = KPITrustMonitor()
            
            # Initialize quality monitor with KPI integration
            self.quality_monitor = TestQualityMonitor(
                kpi_monitor=kpi_monitor,
                event_publisher=None,  # TODO: Connect to EventBus
                enable_self_healing=config.getoption('--enable-self-healing', default=True)
            )
            
            logger.info("Grace Test Quality Monitor initialized with KPI integration")
        except Exception as e:
            # Fallback to basic quality monitor without KPI integration
            logger.warning(f"Could not initialize KPI integration: {e}")
            self.quality_monitor = TestQualityMonitor(
                kpi_monitor=None,
                event_publisher=None,
                enable_self_healing=False
            )
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Capture test results and feed to quality monitor."""
        outcome = yield
        report = outcome.get_result()
        
        # Only process on call phase (not setup/teardown)
        if report.when != 'call':
            return
        
        # Track stats
        self.total_tests += 1
        if report.passed:
            self.passed_tests += 1
        elif report.failed:
            self.failed_tests += 1
        elif report.skipped:
            self.skipped_tests += 1
        
        # Create test result
        component_id = self._get_component_id(item.nodeid)
        
        error_message = None
        error_severity = ErrorSeverity.MEDIUM
        
        if report.failed:
            error_message = str(report.longrepr)[:500]  # Truncate
            error_severity = self._determine_error_severity(report)
        
        # Don't record skipped tests as failures - they shouldn't affect quality score
        if not report.skipped:
            test_result = TestResult(
                test_name=item.nodeid,
                component_id=component_id,
                passed=report.passed,
                execution_time_ms=report.duration * 1000 if hasattr(report, 'duration') else 0.0,
                error_message=error_message,
                error_severity=error_severity,
                timestamp=datetime.now(),
                tags={
                    'markers': [m.name for m in item.iter_markers()],
                    'outcome': report.outcome
                }
            )
            
            # Store for async processing
            self.test_results[item.nodeid] = test_result
    
    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        """Generate quality report at end of session."""
        if not self.quality_monitor:
            return
        
        # Process all test results asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for test_result in self.test_results.values():
                loop.run_until_complete(
                    self.quality_monitor.record_test_result(test_result)
                )
        finally:
            loop.close()
        
        # Generate quality summary
        summary = self.quality_monitor.get_system_quality_summary()
        
        # Print quality report
        self._print_quality_report(summary)
        
        # Save detailed report
        self._save_quality_report(summary)
    
    def _print_quality_report(self, summary: Dict[str, Any]):
        """Print quality report to terminal."""
        print("\n" + "="*80)
        print("🎯 GRACE TEST QUALITY REPORT (90% Threshold Model)")
        print("="*80)
        
        # System-wide metrics
        system_pass_rate = summary['system_pass_rate'] * 100
        overall_quality = summary['overall_quality'] * 100
        
        print(f"\n📊 System-Wide Quality:")
        print(f"  Total Components:     {summary['total_components']}")
        print(f"  Passing Components:   {summary['passing_components']} (≥90% quality)")
        print(f"  System Pass Rate:     {system_pass_rate:.1f}% {'✅' if system_pass_rate >= 90 else '⚠️'}")
        print(f"  Overall Avg Quality:  {overall_quality:.1f}%")
        
        # Status breakdown
        print(f"\n📈 Component Status Breakdown:")
        breakdown = summary['breakdown']
        for status in ['excellent', 'passing', 'acceptable', 'degraded', 'critical']:
            count = breakdown.get(status, 0)
            emoji = {
                'excellent': '🌟',
                'passing': '✅',
                'acceptable': '⚡',
                'degraded': '⚠️',
                'critical': '🔴'
            }[status]
            print(f"  {emoji} {status.upper():12} {count:3} components")
        
        # Raw test counts
        print(f"\n📝 Raw Test Results:")
        print(f"  Total Tests:   {self.total_tests}")
        print(f"  Passed:        {self.passed_tests} ({self.passed_tests/max(1, self.total_tests)*100:.1f}%)")
        print(f"  Failed:        {self.failed_tests} ({self.failed_tests/max(1, self.total_tests)*100:.1f}%)")
        print(f"  Skipped:       {self.skipped_tests} ({self.skipped_tests/max(1, self.total_tests)*100:.1f}%)")
        
        # Components needing attention
        needing_attention = self.quality_monitor.get_components_needing_attention()
        if needing_attention:
            print(f"\n⚠️  Components Needing Attention ({len(needing_attention)}):")
            for comp in needing_attention[:5]:  # Top 5
                print(f"  • {comp['component_id']:25} "
                      f"Score: {comp['score']*100:5.1f}% "
                      f"Gap: {comp['gap_to_passing']*100:5.1f}% "
                      f"Priority: {comp['priority']:.0f}")
        
        # Summary
        print(f"\n{'='*80}")
        if system_pass_rate >= 90:
            print("✅ SYSTEM QUALITY: PASSING - All critical components meet threshold!")
        elif system_pass_rate >= 70:
            print("⚡ SYSTEM QUALITY: ACCEPTABLE - Some components need improvement")
        elif system_pass_rate >= 50:
            print("⚠️  SYSTEM QUALITY: DEGRADED - Multiple components require attention")
        else:
            print("🔴 SYSTEM QUALITY: CRITICAL - Immediate action required")
        print(f"{'='*80}\n")
    
    def _save_quality_report(self, summary: Dict[str, Any]):
        """Save detailed quality report to file."""
        import json
        from pathlib import Path
        
        # Collect detailed component data
        components = {}
        for component_id in self.quality_monitor.components.keys():
            components[component_id] = self.quality_monitor.get_component_details(component_id)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "components": components,
            "needing_attention": self.quality_monitor.get_components_needing_attention(),
            "raw_counts": {
                "total": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "skipped": self.skipped_tests
            }
        }
        
        # Save to file
        report_dir = Path("/workspaces/Grace-/test_reports")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved: {report_file}")


def pytest_configure(config):
    """Register the plugin."""
    config.pluginmanager.register(GraceTestQualityPlugin(config), "grace_quality")


def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption(
        "--enable-self-healing",
        action="store_true",
        default=True,
        help="Enable self-healing triggers on quality degradation"
    )
    parser.addoption(
        "--no-self-healing",
        action="store_false",
        dest="enable_self_healing",
        help="Disable self-healing triggers"
    )
