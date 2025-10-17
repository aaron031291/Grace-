"""
Class 10: Loop Drift Detection - GraceCognitionLinter and contradiction tracing
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detected"""
    LOGICAL = "logical"
    BEHAVIORAL = "behavioral"
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"
    CONSTITUTIONAL = "constitutional"


class SeverityLevel(Enum):
    """Severity of drift"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    alert_id: str
    drift_type: DriftType
    severity: SeverityLevel
    description: str
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContradictionTrace:
    """Trace of contradictory outputs"""
    trace_id: str
    loop_id: str
    iteration_1: int
    iteration_2: int
    contradiction_type: str
    output_1: Dict[str, Any]
    output_2: Dict[str, Any]
    severity: SeverityLevel
    timestamp: datetime = field(default_factory=datetime.now)


class GraceCognitionLinter:
    """
    Detects cognitive drift and contradictions in loop outputs
    Core implementation of Class 10
    """
    
    def __init__(self):
        self.drift_alerts: List[DriftAlert] = []
        self.contradiction_traces: List[ContradictionTrace] = []
        self.baseline_outputs: Dict[str, Any] = {}
        self.drift_thresholds = {
            'confidence_drop': 0.3,
            'logic_deviation': 0.4,
            'semantic_shift': 0.5
        }
        logger.info("GraceCognitionLinter initialized")
    
    def lint_loop_output(
        self,
        loop_id: str,
        iteration: int,
        output: Dict[str, Any],
        previous_output: Optional[Dict[str, Any]] = None
    ) -> List[DriftAlert]:
        """
        Lint loop output for drift and contradictions
        Core implementation of Class 10
        """
        alerts = []
        
        # Establish baseline if first iteration
        if iteration == 0:
            self.baseline_outputs[loop_id] = output
            return alerts
        
        # Check for logical drift
        logical_alerts = self._check_logical_drift(loop_id, iteration, output)
        alerts.extend(logical_alerts)
        
        # Check for behavioral drift
        if previous_output:
            behavioral_alerts = self._check_behavioral_drift(
                loop_id, iteration, output, previous_output
            )
            alerts.extend(behavioral_alerts)
        
        # Check for semantic drift
        semantic_alerts = self._check_semantic_drift(loop_id, iteration, output)
        alerts.extend(semantic_alerts)
        
        # Check for performance drift
        performance_alerts = self._check_performance_drift(loop_id, iteration, output)
        alerts.extend(performance_alerts)
        
        # Check for contradictions
        if previous_output:
            contradictions = self._detect_contradictions(
                loop_id, iteration - 1, iteration, previous_output, output
            )
            for contradiction in contradictions:
                self.contradiction_traces.append(contradiction)
                
                # Create alert for contradiction
                alerts.append(DriftAlert(
                    alert_id=f"drift_{len(self.drift_alerts)}",
                    drift_type=DriftType.LOGICAL,
                    severity=contradiction.severity,
                    description=f"Contradiction detected: {contradiction.contradiction_type}",
                    evidence={'trace_id': contradiction.trace_id}
                ))
        
        # Store alerts
        self.drift_alerts.extend(alerts)
        
        if alerts:
            logger.warning(f"Detected {len(alerts)} drift alerts in loop {loop_id} iteration {iteration}")
        
        return alerts
    
    def _check_logical_drift(
        self,
        loop_id: str,
        iteration: int,
        output: Dict[str, Any]
    ) -> List[DriftAlert]:
        """Check for logical inconsistencies"""
        alerts = []
        
        # Check confidence consistency
        if 'confidence' in output:
            confidence = output['confidence']
            
            if loop_id in self.baseline_outputs:
                baseline_confidence = self.baseline_outputs[loop_id].get('confidence', 1.0)
                confidence_drop = baseline_confidence - confidence
                
                if confidence_drop > self.drift_thresholds['confidence_drop']:
                    alerts.append(DriftAlert(
                        alert_id=f"drift_{len(self.drift_alerts)}",
                        drift_type=DriftType.LOGICAL,
                        severity=SeverityLevel.WARNING,
                        description=f"Significant confidence drop: {confidence_drop:.2f}",
                        evidence={
                            'baseline': baseline_confidence,
                            'current': confidence,
                            'drop': confidence_drop
                        }
                    ))
        
        # Check for logical inconsistency in result
        if 'result' in output and isinstance(output['result'], dict):
            if 'error' in output['result'] and output.get('status') == 'success':
                alerts.append(DriftAlert(
                    alert_id=f"drift_{len(self.drift_alerts)}",
                    drift_type=DriftType.LOGICAL,
                    severity=SeverityLevel.ERROR,
                    description="Status 'success' but error present in result",
                    evidence={'status': output.get('status'), 'has_error': True}
                ))
        
        return alerts
    
    def _check_behavioral_drift(
        self,
        loop_id: str,
        iteration: int,
        current_output: Dict[str, Any],
        previous_output: Dict[str, Any]
    ) -> List[DriftAlert]:
        """Check for behavioral changes"""
        alerts = []
        
        # Check for execution time drift
        current_time = current_output.get('execution_time', 0)
        previous_time = previous_output.get('execution_time', 0)
        
        if previous_time > 0:
            time_ratio = current_time / previous_time
            
            if time_ratio > 3.0:  # 3x slower
                alerts.append(DriftAlert(
                    alert_id=f"drift_{len(self.drift_alerts)}",
                    drift_type=DriftType.PERFORMANCE,
                    severity=SeverityLevel.WARNING,
                    description=f"Execution time increased significantly: {time_ratio:.1f}x",
                    evidence={
                        'previous': previous_time,
                        'current': current_time,
                        'ratio': time_ratio
                    }
                ))
        
        # Check for status changes
        current_status = current_output.get('status')
        previous_status = previous_output.get('status')
        
        if current_status != previous_status:
            severity = SeverityLevel.INFO
            if previous_status == 'success' and current_status == 'failed':
                severity = SeverityLevel.ERROR
            
            alerts.append(DriftAlert(
                alert_id=f"drift_{len(self.drift_alerts)}",
                drift_type=DriftType.BEHAVIORAL,
                severity=severity,
                description=f"Status changed: {previous_status} -> {current_status}",
                evidence={'previous': previous_status, 'current': current_status}
            ))
        
        return alerts
    
    def _check_semantic_drift(
        self,
        loop_id: str,
        iteration: int,
        output: Dict[str, Any]
    ) -> List[DriftAlert]:
        """Check for semantic meaning drift"""
        alerts = []
        
        if loop_id not in self.baseline_outputs:
            return alerts
        
        baseline = self.baseline_outputs[loop_id]
        
        # Check for key result field changes
        baseline_keys = set(baseline.get('result', {}).keys())
        current_keys = set(output.get('result', {}).keys())
        
        missing_keys = baseline_keys - current_keys
        new_keys = current_keys - baseline_keys
        
        if missing_keys:
            alerts.append(DriftAlert(
                alert_id=f"drift_{len(self.drift_alerts)}",
                drift_type=DriftType.SEMANTIC,
                severity=SeverityLevel.WARNING,
                description=f"Missing result fields: {', '.join(missing_keys)}",
                evidence={'missing_keys': list(missing_keys)}
            ))
        
        if len(new_keys) > 3:  # Significant new keys
            alerts.append(DriftAlert(
                alert_id=f"drift_{len(self.drift_alerts)}",
                drift_type=DriftType.SEMANTIC,
                severity=SeverityLevel.INFO,
                description=f"New result fields appeared: {', '.join(list(new_keys)[:3])}...",
                evidence={'new_keys': list(new_keys)}
            ))
        
        return alerts
    
    def _check_performance_drift(
        self,
        loop_id: str,
        iteration: int,
        output: Dict[str, Any]
    ) -> List[DriftAlert]:
        """Check for performance degradation"""
        alerts = []
        
        # Check memory usage
        memory_used = output.get('memory_used', 0)
        if memory_used > 1_000_000_000:  # 1GB
            alerts.append(DriftAlert(
                alert_id=f"drift_{len(self.drift_alerts)}",
                drift_type=DriftType.PERFORMANCE,
                severity=SeverityLevel.WARNING,
                description=f"High memory usage: {memory_used / 1_000_000:.1f}MB",
                evidence={'memory_used': memory_used}
            ))
        
        # Check clarity score degradation
        clarity = output.get('clarity_score', 1.0)
        if clarity < 0.5:
            alerts.append(DriftAlert(
                alert_id=f"drift_{len(self.drift_alerts)}",
                drift_type=DriftType.PERFORMANCE,
                severity=SeverityLevel.WARNING,
                description=f"Low clarity score: {clarity:.2f}",
                evidence={'clarity_score': clarity}
            ))
        
        return alerts
    
    def _detect_contradictions(
        self,
        loop_id: str,
        iteration_1: int,
        iteration_2: int,
        output_1: Dict[str, Any],
        output_2: Dict[str, Any]
    ) -> List[ContradictionTrace]:
        """Detect contradictions between outputs"""
        contradictions = []
        
        # Check for opposing decisions
        decision_1 = output_1.get('result', {}).get('decision')
        decision_2 = output_2.get('result', {}).get('decision')
        
        if decision_1 is not None and decision_2 is not None:
            if isinstance(decision_1, bool) and isinstance(decision_2, bool):
                if decision_1 != decision_2:
                    contradictions.append(ContradictionTrace(
                        trace_id=f"contra_{len(self.contradiction_traces)}",
                        loop_id=loop_id,
                        iteration_1=iteration_1,
                        iteration_2=iteration_2,
                        contradiction_type="opposing_decisions",
                        output_1=output_1,
                        output_2=output_2,
                        severity=SeverityLevel.ERROR
                    ))
        
        # Check for conflicting values
        result_1 = output_1.get('result', {})
        result_2 = output_2.get('result', {})
        
        for key in set(result_1.keys()) & set(result_2.keys()):
            val_1 = result_1[key]
            val_2 = result_2[key]
            
            # Check numeric contradictions
            if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                if val_1 > 0 and val_2 < 0:  # Sign flip
                    contradictions.append(ContradictionTrace(
                        trace_id=f"contra_{len(self.contradiction_traces)}",
                        loop_id=loop_id,
                        iteration_1=iteration_1,
                        iteration_2=iteration_2,
                        contradiction_type=f"value_sign_flip_{key}",
                        output_1=output_1,
                        output_2=output_2,
                        severity=SeverityLevel.WARNING
                    ))
        
        return contradictions
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate drift detection report"""
        if not self.drift_alerts:
            return {'total_alerts': 0, 'no_drift_detected': True}
        
        by_type = {}
        by_severity = {}
        
        for alert in self.drift_alerts:
            drift_type = alert.drift_type.value
            severity = alert.severity.value
            
            by_type[drift_type] = by_type.get(drift_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            'total_alerts': len(self.drift_alerts),
            'by_type': by_type,
            'by_severity': by_severity,
            'contradictions': len(self.contradiction_traces),
            'unresolved_alerts': sum(1 for a in self.drift_alerts if not a.resolved),
            'critical_alerts': sum(1 for a in self.drift_alerts if a.severity == SeverityLevel.CRITICAL)
        }


class LoopDriftDetector:
    """
    High-level drift detector for loop monitoring
    """
    
    def __init__(self):
        self.linter = GraceCognitionLinter()
        self.loop_history: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("LoopDriftDetector initialized")
    
    def track_loop(self, loop_id: str, iteration: int, output: Dict[str, Any]) -> List[DriftAlert]:
        """Track loop and detect drift"""
        if loop_id not in self.loop_history:
            self.loop_history[loop_id] = []
        
        previous_output = self.loop_history[loop_id][-1] if self.loop_history[loop_id] else None
        
        alerts = self.linter.lint_loop_output(loop_id, iteration, output, previous_output)
        
        self.loop_history[loop_id].append(output)
        
        return alerts
    
    def get_loop_health(self, loop_id: str) -> Dict[str, Any]:
        """Get health status of a loop"""
        if loop_id not in self.loop_history:
            return {'status': 'unknown', 'loop_id': loop_id}
        
        loop_alerts = [a for a in self.linter.drift_alerts if loop_id in a.description or loop_id in str(a.evidence)]
        
        critical_count = sum(1 for a in loop_alerts if a.severity == SeverityLevel.CRITICAL)
        error_count = sum(1 for a in loop_alerts if a.severity == SeverityLevel.ERROR)
        
        if critical_count > 0:
            status = 'critical'
        elif error_count > 2:
            status = 'unhealthy'
        elif error_count > 0:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'loop_id': loop_id,
            'iterations': len(self.loop_history[loop_id]),
            'total_alerts': len(loop_alerts),
            'critical_alerts': critical_count,
            'error_alerts': error_count
        }
