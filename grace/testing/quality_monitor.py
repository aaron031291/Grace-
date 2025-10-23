"""
Test Quality Monitor with event emission
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TestQualityMonitor:
    """
    Monitors test quality and emits events to event bus
    
    Features:
    - Track passed, failed, and skipped tests
    - Calculate quality metrics
    - Emit events for critical issues
    - Integration with self-healing AVN
    """
    
    def __init__(self, event_publisher=None):
        """
        Initialize test quality monitor
        
        Args:
            event_publisher: Event publisher for event mesh
        """
        self.event_publisher = event_publisher
        self.test_results: List[TestResult] = []
        self.test_counts = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error": 0
        }
        self.quality_thresholds = {
            "critical_failure_rate": 0.1,  # 10% failures triggers critical
            "warning_skip_rate": 0.2,      # 20% skips triggers warning
            "min_pass_rate": 0.8            # 80% pass rate required
        }
    
    def record_test(self, result: TestResult):
        """
        Record a test result and emit events if thresholds exceeded
        
        Args:
            result: Test result to record
        """
        self.test_results.append(result)
        self.test_counts[result.status] += 1
        
        logger.debug(f"Recorded test: {result.test_id} - {result.status}")
        
        # Check thresholds and emit events
        self._check_quality_thresholds()
    
    def _check_quality_thresholds(self):
        """Check if quality thresholds are exceeded and emit events"""
        total = sum(self.test_counts.values())
        if total == 0:
            return
        
        failure_rate = self.test_counts["failed"] / total
        skip_rate = self.test_counts["skipped"] / total
        pass_rate = self.test_counts["passed"] / total
        
        # Critical: High failure rate
        if failure_rate > self.quality_thresholds["critical_failure_rate"]:
            self._emit_event(
                event_type="TEST.QUALITY.CRITICAL",
                severity="critical",
                data={
                    "failure_rate": failure_rate,
                    "threshold": self.quality_thresholds["critical_failure_rate"],
                    "failed_count": self.test_counts["failed"],
                    "total_count": total,
                    "message": f"Test failure rate {failure_rate:.1%} exceeds threshold"
                }
            )
        
        # Warning: High skip rate
        if skip_rate > self.quality_thresholds["warning_skip_rate"]:
            self._emit_event(
                event_type="TEST.QUALITY.WARNING",
                severity="warning",
                data={
                    "skip_rate": skip_rate,
                    "threshold": self.quality_thresholds["warning_skip_rate"],
                    "skipped_count": self.test_counts["skipped"],
                    "total_count": total,
                    "message": f"Test skip rate {skip_rate:.1%} exceeds threshold"
                }
            )
        
        # Warning: Low pass rate
        if pass_rate < self.quality_thresholds["min_pass_rate"]:
            self._emit_event(
                event_type="TEST.QUALITY.LOW_PASS_RATE",
                severity="warning",
                data={
                    "pass_rate": pass_rate,
                    "threshold": self.quality_thresholds["min_pass_rate"],
                    "passed_count": self.test_counts["passed"],
                    "total_count": total,
                    "message": f"Test pass rate {pass_rate:.1%} below threshold"
                }
            )
    
    def _emit_event(self, event_type: str, severity: str, data: Dict[str, Any]):
        """
        Emit event to event mesh
        
        Args:
            event_type: Event type identifier
            severity: Event severity
            data: Event data
        """
        event = {
            "type": event_type,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test_quality_monitor",
            "data": data
        }
        
        if self.event_publisher:
            try:
                self.event_publisher.publish(event)
                logger.info(f"Emitted event: {event_type} ({severity})")
            except Exception as e:
                logger.error(f"Failed to emit event: {e}")
        else:
            logger.warning(f"No event publisher configured, event not emitted: {event_type}")
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics"""
        total = sum(self.test_counts.values())
        
        if total == 0:
            return {
                "total_tests": 0,
                "pass_rate": 0.0,
                "failure_rate": 0.0,
                "skip_rate": 0.0,
                "quality_score": 0.0
            }
        
        pass_rate = self.test_counts["passed"] / total
        failure_rate = self.test_counts["failed"] / total
        skip_rate = self.test_counts["skipped"] / total
        
        # Quality score: weighted combination
        quality_score = (
            pass_rate * 1.0 +
            (1 - failure_rate) * 0.8 +
            (1 - skip_rate) * 0.5
        ) / 2.3
        
        return {
            "total_tests": total,
            "passed": self.test_counts["passed"],
            "failed": self.test_counts["failed"],
            "skipped": self.test_counts["skipped"],
            "errors": self.test_counts["error"],
            "pass_rate": pass_rate,
            "failure_rate": failure_rate,
            "skip_rate": skip_rate,
            "quality_score": quality_score,
            "health_status": self._get_health_status(quality_score)
        }
    
    def _get_health_status(self, quality_score: float) -> str:
        """Determine health status from quality score"""
        if quality_score >= 0.9:
            return "excellent"
        elif quality_score >= 0.75:
            return "good"
        elif quality_score >= 0.6:
            return "fair"
        elif quality_score >= 0.4:
            return "poor"
        else:
            return "critical"
