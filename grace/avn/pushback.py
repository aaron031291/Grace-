"""
Pushback escalation system with database-backed error tracking and AVN integration
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import uuid
import logging
import traceback

logger = logging.getLogger(__name__)


class PushbackSeverity(Enum):
    """Severity levels for pushback events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationDecision(Enum):
    """Escalation decision types"""
    IGNORE = "ignore"
    LOG_ONLY = "log_only"
    NOTIFY = "notify"
    ESCALATE_TO_AVN = "escalate_to_avn"
    IMMEDIATE_ACTION = "immediate_action"


class ThresholdRule:
    """Represents an escalation threshold rule"""
    
    def __init__(
        self,
        name: str,
        error_count: int,
        time_window_seconds: int,
        severity: PushbackSeverity,
        decision: EscalationDecision
    ):
        self.name = name
        self.error_count = error_count
        self.time_window_seconds = time_window_seconds
        self.severity = severity
        self.decision = decision
    
    def __repr__(self) -> str:
        return f"<ThresholdRule({self.name}: {self.error_count} errors in {self.time_window_seconds}s)>"


class PushbackEscalation:
    """
    Pushback escalation system with time-windowed threshold monitoring
    
    Note: This is a simplified in-memory version. For production with database,
    import SQLAlchemy models and use async database sessions.
    """
    
    def __init__(self, avn_client: Optional[Any] = None):
        """
        Initialize pushback escalation system
        
        Args:
            avn_client: AVN client for sending escalations
        """
        self.avn_client: Optional[Any] = avn_client
        self.error_history: List[Dict[str, Any]] = []
        self.time_window: timedelta = timedelta(minutes=10)
        
        # Define threshold rules
        self.threshold_rules: List[ThresholdRule] = [
            ThresholdRule("critical_immediate", 1, 60, PushbackSeverity.CRITICAL, EscalationDecision.IMMEDIATE_ACTION),
            ThresholdRule("high_rapid", 2, 300, PushbackSeverity.HIGH, EscalationDecision.ESCALATE_TO_AVN),
            ThresholdRule("medium_burst", 5, 300, PushbackSeverity.MEDIUM, EscalationDecision.ESCALATE_TO_AVN),
            ThresholdRule("low_sustained", 10, 600, PushbackSeverity.LOW, EscalationDecision.NOTIFY),
            ThresholdRule("generic_burst", 15, 300, PushbackSeverity.HIGH, EscalationDecision.ESCALATE_TO_AVN),
        ]
        
        logger.info(f"Initialized PushbackEscalation with {len(self.threshold_rules)} threshold rules")
    
    def record_and_evaluate_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: Optional[PushbackSeverity] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> EscalationDecision:
        """
        Record error and evaluate escalation thresholds
        
        Args:
            error: The error/exception
            context: Context information
            severity: Optional severity override
            user_id: User ID if applicable
            endpoint: API endpoint if applicable
            ip_address: Client IP if applicable
            
        Returns:
            Escalation decision
        """
        # Determine severity if not provided
        if severity is None:
            severity = self._assess_severity(error, context)
        
        # Create error hash for grouping similar errors
        error_hash = self._compute_error_hash(error)
        
        # Record error
        error_record = {
            "id": str(uuid.uuid4()),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_hash": error_hash,
            "severity": severity.value,
            "context": context,
            "stack_trace": traceback.format_exc() if context.get('include_stack_trace') else None,
            "user_id": user_id,
            "endpoint": endpoint,
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.error_history.append(error_record)
        
        # Clean old history
        self._clean_old_history()
        
        logger.info(
            f"Recorded error: {error_record['id']} "
            f"(type={error_record['error_type']}, severity={severity.value})"
        )
        
        # Evaluate thresholds
        decision = self._evaluate_thresholds(error_hash, severity)
        
        if decision in [EscalationDecision.ESCALATE_TO_AVN, EscalationDecision.IMMEDIATE_ACTION]:
            self._trigger_escalation(error_record, decision)
        
        logger.info(f"Escalation decision for {error_record['id']}: {decision.value}")
        
        return decision
    
    def _assess_severity(self, error: Exception, context: Dict[str, Any]) -> PushbackSeverity:
        """Assess error severity based on type and context"""
        error_type = type(error).__name__
        
        # Critical errors
        if any(keyword in error_type.lower() for keyword in ['security', 'auth', 'permission', 'critical']):
            return PushbackSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_type.lower() for keyword in ['database', 'connection', 'timeout']):
            return PushbackSeverity.HIGH
        
        # Medium severity
        if any(keyword in error_type.lower() for keyword in ['validation', 'value', 'key']):
            return PushbackSeverity.MEDIUM
        
        # Default to low
        return PushbackSeverity.LOW
    
    def _compute_error_hash(self, error: Exception) -> str:
        """Compute hash for grouping similar errors"""
        error_signature = f"{type(error).__name__}:{str(error)}"
        return hashlib.md5(error_signature.encode()).hexdigest()
    
    def _evaluate_thresholds(
        self,
        error_hash: str,
        severity: PushbackSeverity
    ) -> EscalationDecision:
        """Evaluate threshold rules against recent error history"""
        recent_errors = self._get_recent_errors()
        now = datetime.now(timezone.utc)
        
        for rule in self.threshold_rules:
            # Skip rules that don't match severity (except generic rules)
            if rule.severity != severity and rule.name != "generic_burst":
                continue
            
            # Calculate time window
            window_start = now - timedelta(seconds=rule.time_window_seconds)
            
            # Count errors in window
            if rule.name == "generic_burst":
                # Generic: count all errors
                error_count = sum(
                    1 for e in recent_errors
                    if e["timestamp"] >= window_start
                )
            else:
                # Specific: count errors with same hash
                error_count = sum(
                    1 for e in recent_errors
                    if e["error_hash"] == error_hash and e["timestamp"] >= window_start
                )
            
            logger.debug(
                f"Rule '{rule.name}': {error_count}/{rule.error_count} errors "
                f"in {rule.time_window_seconds}s window"
            )
            
            if error_count >= rule.error_count:
                logger.warning(
                    f"Threshold exceeded for rule '{rule.name}': "
                    f"{error_count} errors in {rule.time_window_seconds}s"
                )
                return rule.decision
        
        return EscalationDecision.LOG_ONLY
    
    def _trigger_escalation(
        self,
        error_record: Dict[str, Any],
        decision: EscalationDecision
    ) -> None:
        """Trigger escalation actions"""
        logger.warning(
            f"Escalation triggered: {decision.value} for {error_record['error_type']}",
            extra=error_record
        )
        
        # Send to AVN if client
        if self.avn_client:
            try:
                self.avn_client.report_metrics(
                    error_record.get('endpoint', 'unknown'),
                    {
                        "error_rate": 1.0,
                        "latency": 1000,
                        "source": "pushback_escalation",
                        "error_type": error_record['error_type']
                    }
                )
            except Exception as e:
                logger.error(f"Failed to notify AVN: {e}")
    
    def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get errors within the time window"""
        cutoff = datetime.now(timezone.utc) - self.time_window
        return [
            e for e in self.error_history
            if e["timestamp"] > cutoff
        ]
    
    def _clean_old_history(self) -> None:
        """Remove errors outside the time window"""
        cutoff = datetime.now(timezone.utc) - self.time_window
        self.error_history = [
            e for e in self.error_history
            if e["timestamp"] > cutoff
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        recent = self._get_recent_errors()
        
        severity_counts = {
            severity.value: sum(1 for e in recent if e["severity"] == severity.value)
            for severity in PushbackSeverity
        }
        
        error_types: Dict[str, int] = {}
        for error in recent:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "time_window_minutes": self.time_window.total_seconds() / 60,
            "total_recent_errors": len(recent),
            "severity_breakdown": severity_counts,
            "error_types": error_types,
            "top_errors": sorted(
                error_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
