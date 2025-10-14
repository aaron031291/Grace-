"""
Test Quality Monitor - Adaptive test scoring with KPI integration and self-healing triggers.

This system:
1. Tracks per-component test quality scores (not raw pass/fail counts)
2. Integrates with KPITrustMonitor for component health tracking
3. Triggers TriggerMesh events for adaptive learning and self-healing
4. Uses 90% quality threshold for "passing" components
5. Provides system-wide quality metrics based on component thresholds

Philosophy:
- Components score incrementally (0-100%) based on:
  - Test pass rate
  - Error severity
  - Historical reliability
  - Trust score from KPITrustMonitor
- Only components ≥90% count toward system-wide "passing" percentage
- Components <90% trigger adaptive learning loops
- System-wide progress is clear and predictable
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class ComponentQualityStatus(Enum):
    """Quality status levels for test components."""
    CRITICAL = "critical"      # <50% - immediate attention needed
    DEGRADED = "degraded"      # 50-70% - needs improvement
    ACCEPTABLE = "acceptable"  # 70-90% - functional but not passing
    PASSING = "passing"        # ≥90% - meets quality threshold
    EXCELLENT = "excellent"    # ≥95% - exceptional quality


class ErrorSeverity(Enum):
    """Severity levels for test errors."""
    LOW = 1        # Warnings, deprecated usage
    MEDIUM = 2     # Assertion failures, expected errors
    HIGH = 3       # Unexpected exceptions, integration failures
    CRITICAL = 4   # System crashes, data corruption


@dataclass
class TestResult:
    """Individual test result with quality scoring."""
    test_name: str
    component_id: str
    passed: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    error_severity: ErrorSeverity = ErrorSeverity.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentQualityScore:
    """Quality score for a test component."""
    component_id: str
    raw_score: float  # 0.0 - 1.0 based on test results
    trust_adjusted_score: float  # 0.0 - 1.0 adjusted by KPI trust
    quality_status: ComponentQualityStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_severity_breakdown: Dict[ErrorSeverity, int]
    last_updated: datetime
    historical_scores: List[float] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """Calculate raw pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def is_passing(self) -> bool:
        """Check if component meets the 90% threshold."""
        return self.trust_adjusted_score >= 0.90


class TestQualityMonitor:
    """
    Monitors test quality with KPI integration and self-healing triggers.
    
    Integrates with:
    - KPITrustMonitor: For component trust scores
    - TriggerMesh/EventBus: For triggering adaptive learning
    - Self-Healing Loops: For automatic remediation
    """
    
    # Quality thresholds
    PASSING_THRESHOLD = 0.90      # 90% to be considered "passing"
    EXCELLENT_THRESHOLD = 0.95    # 95% for excellent rating
    DEGRADED_THRESHOLD = 0.70     # 70% triggers warning
    CRITICAL_THRESHOLD = 0.50     # 50% triggers critical alert
    
    # Error severity weights (higher = worse)
    SEVERITY_WEIGHTS = {
        ErrorSeverity.LOW: 0.1,
        ErrorSeverity.MEDIUM: 0.5,
        ErrorSeverity.HIGH: 1.0,
        ErrorSeverity.CRITICAL: 2.0
    }
    
    def __init__(self, 
                 kpi_monitor=None,
                 event_publisher: Optional[Any] = None,
                 enable_self_healing: bool = True):
        """
        Initialize the test quality monitor.
        
        Args:
            kpi_monitor: KPITrustMonitor instance for trust integration
            event_publisher: EventBus/TriggerMesh for event publishing
            enable_self_healing: Whether to trigger self-healing on failures
        """
        self.kpi_monitor = kpi_monitor
        self.event_publisher = event_publisher
        self.enable_self_healing = enable_self_healing
        
        # Component tracking
        self.components: Dict[str, ComponentQualityScore] = {}
        self.test_results: Dict[str, List[TestResult]] = {}
        
        # System-wide metrics
        self.system_quality_history: List[float] = []
        self.last_quality_check: Optional[datetime] = None
        
        logger.info("TestQualityMonitor initialized with 90% passing threshold")
    
    async def record_test_result(self, result: TestResult):
        """
        Record a test result and update component quality score.
        
        This triggers:
        1. Component score recalculation
        2. KPI metric update
        3. Trust score update
        4. Self-healing trigger if needed
        """
        component_id = result.component_id
        
        # Store test result
        if component_id not in self.test_results:
            self.test_results[component_id] = []
        self.test_results[component_id].append(result)
        
        # Limit history
        if len(self.test_results[component_id]) > 100:
            self.test_results[component_id] = self.test_results[component_id][-100:]
        
        # Update component quality score
        await self._update_component_score(component_id)
        
        # Record KPI metric
        if self.kpi_monitor:
            score = self.components[component_id]
            await self.kpi_monitor.record_metric(
                name="test_quality_score",
                value=score.trust_adjusted_score * 100,
                component_id=component_id,
                threshold_warning=self.DEGRADED_THRESHOLD * 100,
                threshold_critical=self.CRITICAL_THRESHOLD * 100,
                tags={
                    "status": score.quality_status.value,
                    "is_passing": str(score.is_passing)
                }
            )
            
            # Update trust score
            await self.kpi_monitor.update_trust_score(
                component_id=component_id,
                performance_score=score.trust_adjusted_score,
                confidence=min(1.0, score.total_tests / 10.0)  # More tests = more confidence
            )
        
        # Publish event
        if self.event_publisher:
            await self.event_publisher(
                "test_result_recorded",
                {
                    "component_id": component_id,
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "quality_score": self.components[component_id].trust_adjusted_score,
                    "is_passing": self.components[component_id].is_passing
                }
            )
        
        # Trigger self-healing if needed
        if self.enable_self_healing:
            await self._check_self_healing_trigger(component_id)
        
        logger.debug(f"Recorded test result for {component_id}: {result.test_name} - {'PASS' if result.passed else 'FAIL'}")
    
    async def _update_component_score(self, component_id: str):
        """Calculate and update component quality score."""
        results = self.test_results.get(component_id, [])
        
        if not results:
            return
        
        # Count results by type
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        
        # Error severity breakdown
        severity_counts = {severity: 0 for severity in ErrorSeverity}
        for r in results:
            if not r.passed and r.error_severity:
                severity_counts[r.error_severity] += 1
        
        # Calculate raw score (0.0 - 1.0)
        # Base score = pass rate
        raw_score = passed / total if total > 0 else 0.0
        
        # Apply error severity penalties
        if failed > 0:
            total_penalty = sum(
                severity_counts[sev] * self.SEVERITY_WEIGHTS[sev]
                for sev in ErrorSeverity
            )
            # Normalize penalty (max penalty = 1.0)
            normalized_penalty = min(1.0, total_penalty / (failed * 2.0))
            raw_score = raw_score * (1.0 - normalized_penalty * 0.3)  # Max 30% penalty
        
        # Get trust-adjusted score from KPI monitor
        trust_adjusted_score = raw_score
        if self.kpi_monitor:
            trust_score_obj = self.kpi_monitor.get_trust_score(component_id)
            if trust_score_obj:
                # Blend raw score with historical trust (80% current, 20% historical)
                trust_adjusted_score = 0.8 * raw_score + 0.2 * trust_score_obj.score
        
        # Determine quality status
        if trust_adjusted_score >= self.EXCELLENT_THRESHOLD:
            status = ComponentQualityStatus.EXCELLENT
        elif trust_adjusted_score >= self.PASSING_THRESHOLD:
            status = ComponentQualityStatus.PASSING
        elif trust_adjusted_score >= self.DEGRADED_THRESHOLD:
            status = ComponentQualityStatus.ACCEPTABLE
        elif trust_adjusted_score >= self.CRITICAL_THRESHOLD:
            status = ComponentQualityStatus.DEGRADED
        else:
            status = ComponentQualityStatus.CRITICAL
        
        # Create/update component score
        if component_id in self.components:
            existing = self.components[component_id]
            existing.historical_scores.append(trust_adjusted_score)
            if len(existing.historical_scores) > 50:
                existing.historical_scores = existing.historical_scores[-50:]
        else:
            existing = None
        
        self.components[component_id] = ComponentQualityScore(
            component_id=component_id,
            raw_score=raw_score,
            trust_adjusted_score=trust_adjusted_score,
            quality_status=status,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=0,  # TODO: track skipped separately
            error_severity_breakdown=severity_counts,
            last_updated=datetime.now(),
            historical_scores=existing.historical_scores if existing else [trust_adjusted_score]
        )
    
    async def _check_self_healing_trigger(self, component_id: str):
        """
        Check if component quality triggers self-healing loop.
        
        Triggers:
        - CRITICAL (<50%): Immediate escalation to AVN
        - DEGRADED (50-70%): Trigger adaptive learning
        - ACCEPTABLE (70-90%): Monitor closely, suggest improvements
        """
        score = self.components.get(component_id)
        if not score:
            return
        
        # Critical: Escalate to AVN for immediate healing
        if score.quality_status == ComponentQualityStatus.CRITICAL:
            await self._trigger_healing_escalation(
                component_id=component_id,
                severity="CRITICAL",
                reason=f"Test quality critically low: {score.trust_adjusted_score:.1%}",
                recommended_actions=[
                    "Review recent code changes",
                    "Check system resources",
                    "Verify dependencies",
                    "Run diagnostic tests"
                ]
            )
        
        # Degraded: Trigger adaptive learning
        elif score.quality_status == ComponentQualityStatus.DEGRADED:
            await self._trigger_adaptive_learning(
                component_id=component_id,
                severity="WARNING",
                reason=f"Test quality degraded: {score.trust_adjusted_score:.1%}",
                focus_areas=self._identify_problem_areas(component_id)
            )
        
        # Acceptable but not passing: Suggest improvements
        elif score.quality_status == ComponentQualityStatus.ACCEPTABLE:
            await self._trigger_improvement_suggestion(
                component_id=component_id,
                current_score=score.trust_adjusted_score,
                target_score=self.PASSING_THRESHOLD,
                suggestions=self._generate_improvement_suggestions(component_id)
            )
    
    async def _trigger_healing_escalation(self, 
                                         component_id: str,
                                         severity: str,
                                         reason: str,
                                         recommended_actions: List[str]):
        """Trigger escalation to AVN/Memory Orchestrator for healing."""
        if self.event_publisher:
            await self.event_publisher(
                "test_quality.healing_required",
                {
                    "component_id": component_id,
                    "severity": severity,
                    "reason": reason,
                    "recommended_actions": recommended_actions,
                    "timestamp": datetime.now().isoformat(),
                    "escalate_to": "avn_core"
                }
            )
        
        logger.warning(f"HEALING ESCALATION: {component_id} - {reason}")
    
    async def _trigger_adaptive_learning(self,
                                        component_id: str,
                                        severity: str,
                                        reason: str,
                                        focus_areas: List[str]):
        """Trigger adaptive learning loop for component improvement."""
        if self.event_publisher:
            await self.event_publisher(
                "test_quality.adaptive_learning_required",
                {
                    "component_id": component_id,
                    "severity": severity,
                    "reason": reason,
                    "focus_areas": focus_areas,
                    "timestamp": datetime.now().isoformat(),
                    "trigger": "learning_kernel"
                }
            )
        
        logger.info(f"ADAPTIVE LEARNING: {component_id} - {reason}")
    
    async def _trigger_improvement_suggestion(self,
                                             component_id: str,
                                             current_score: float,
                                             target_score: float,
                                             suggestions: List[str]):
        """Trigger improvement suggestions for component."""
        if self.event_publisher:
            await self.event_publisher(
                "test_quality.improvement_suggested",
                {
                    "component_id": component_id,
                    "current_score": current_score,
                    "target_score": target_score,
                    "gap": target_score - current_score,
                    "suggestions": suggestions,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        logger.info(f"IMPROVEMENT SUGGESTION: {component_id} - Gap: {(target_score - current_score):.1%}")
    
    def _identify_problem_areas(self, component_id: str) -> List[str]:
        """Identify specific problem areas from test failures."""
        results = self.test_results.get(component_id, [])
        failed_results = [r for r in results if not r.passed]
        
        problem_areas = []
        
        # Analyze error patterns
        error_types = {}
        for r in failed_results:
            if r.error_message:
                # Extract error type (simplified)
                error_type = r.error_message.split(':')[0] if ':' in r.error_message else 'Unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Report top error types
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:3]:
            problem_areas.append(f"{error_type} ({count} occurrences)")
        
        # Check for high-severity errors
        score = self.components.get(component_id)
        if score:
            if score.error_severity_breakdown.get(ErrorSeverity.CRITICAL, 0) > 0:
                problem_areas.append("Critical errors present - requires immediate attention")
            if score.error_severity_breakdown.get(ErrorSeverity.HIGH, 0) > 2:
                problem_areas.append("Multiple high-severity errors")
        
        return problem_areas if problem_areas else ["General test failures - needs investigation"]
    
    def _generate_improvement_suggestions(self, component_id: str) -> List[str]:
        """Generate specific improvement suggestions based on test data."""
        score = self.components.get(component_id)
        if not score:
            return []
        
        suggestions = []
        
        # Based on pass rate
        if score.pass_rate < 0.85:
            suggestions.append("Review and fix failing tests to improve pass rate")
        
        # Based on error severity
        if score.error_severity_breakdown.get(ErrorSeverity.HIGH, 0) > 0:
            suggestions.append("Address high-severity errors first for maximum impact")
        
        # Based on historical trends
        if len(score.historical_scores) >= 3:
            recent_trend = score.historical_scores[-3:]
            if all(recent_trend[i] <= recent_trend[i-1] for i in range(1, len(recent_trend))):
                suggestions.append("Quality declining - investigate recent changes")
        
        # Generic suggestions if no specific ones
        if not suggestions:
            gap = self.PASSING_THRESHOLD - score.trust_adjusted_score
            suggestions.append(f"Improve quality by {gap:.1%} to reach passing threshold")
            suggestions.append("Review test coverage and edge cases")
        
        return suggestions
    
    def get_system_quality_summary(self) -> Dict[str, Any]:
        """
        Get system-wide quality summary using the 90% threshold model.
        
        Returns:
            - total_components: Number of components being tracked
            - passing_components: Components ≥90% quality
            - system_pass_rate: Percentage of components passing
            - overall_quality: Average quality of ALL components (for monitoring)
            - breakdown: Component counts by status
        """
        if not self.components:
            return {
                "total_components": 0,
                "passing_components": 0,
                "system_pass_rate": 0.0,
                "overall_quality": 0.0,
                "breakdown": {}
            }
        
        total = len(self.components)
        passing = sum(1 for c in self.components.values() if c.is_passing)
        
        # System pass rate = percentage of components at ≥90%
        system_pass_rate = passing / total if total > 0 else 0.0
        
        # Overall quality = average of all scores (for trend monitoring)
        overall_quality = statistics.mean(
            c.trust_adjusted_score for c in self.components.values()
        )
        
        # Breakdown by status
        breakdown = {}
        for status in ComponentQualityStatus:
            breakdown[status.value] = sum(
                1 for c in self.components.values()
                if c.quality_status == status
            )
        
        return {
            "total_components": total,
            "passing_components": passing,
            "system_pass_rate": system_pass_rate,
            "overall_quality": overall_quality,
            "breakdown": breakdown,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_component_details(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed quality information for a specific component."""
        score = self.components.get(component_id)
        if not score:
            return None
        
        return {
            "component_id": component_id,
            "raw_score": score.raw_score,
            "trust_adjusted_score": score.trust_adjusted_score,
            "quality_status": score.quality_status.value,
            "is_passing": score.is_passing,
            "total_tests": score.total_tests,
            "passed_tests": score.passed_tests,
            "failed_tests": score.failed_tests,
            "pass_rate": score.pass_rate,
            "error_severity_breakdown": {
                sev.value: count
                for sev, count in score.error_severity_breakdown.items()
            },
            "last_updated": score.last_updated.isoformat(),
            "historical_trend": score.historical_scores[-10:] if score.historical_scores else []
        }
    
    def get_components_needing_attention(self) -> List[Dict[str, Any]]:
        """Get list of components that need attention (not passing threshold)."""
        needing_attention = []
        
        for component_id, score in self.components.items():
            if not score.is_passing:
                needing_attention.append({
                    "component_id": component_id,
                    "score": score.trust_adjusted_score,
                    "status": score.quality_status.value,
                    "gap_to_passing": self.PASSING_THRESHOLD - score.trust_adjusted_score,
                    "priority": self._calculate_priority(score)
                })
        
        # Sort by priority (highest first)
        needing_attention.sort(key=lambda x: x["priority"], reverse=True)
        
        return needing_attention
    
    def _calculate_priority(self, score: ComponentQualityScore) -> float:
        """
        Calculate priority for component attention.
        
        Higher priority for:
        - Lower scores
        - More critical errors
        - Declining trends
        """
        priority = 0.0
        
        # Lower score = higher priority
        priority += (1.0 - score.trust_adjusted_score) * 40
        
        # Critical errors = very high priority
        priority += score.error_severity_breakdown.get(ErrorSeverity.CRITICAL, 0) * 20
        
        # High severity errors = high priority
        priority += score.error_severity_breakdown.get(ErrorSeverity.HIGH, 0) * 10
        
        # Declining trend = moderate priority
        if len(score.historical_scores) >= 3:
            recent = score.historical_scores[-3:]
            if all(recent[i] < recent[i-1] for i in range(1, len(recent))):
                priority += 15
        
        return priority
