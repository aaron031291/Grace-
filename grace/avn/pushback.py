"""
Pushback escalation system for Adaptive Verification Network (AVN)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging

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


class PushbackEscalation:
    """
    Intelligent pushback escalation system
    
    Decides when errors should be escalated to the AVN based on:
    - Error frequency and patterns
    - Severity and impact
    - Historical context
    - System state
    """
    
    def __init__(self, avn_client=None):
        """
        Initialize pushback escalation system
        
        Args:
            avn_client: AVN client for sending escalations
        """
        self.avn_client = avn_client
        self.error_history: List[Dict[str, Any]] = []
        self.escalation_thresholds = {
            PushbackSeverity.LOW: 10,      # 10 occurrences
            PushbackSeverity.MEDIUM: 5,    # 5 occurrences
            PushbackSeverity.HIGH: 2,      # 2 occurrences
            PushbackSeverity.CRITICAL: 1,  # Immediate
        }
        self.time_window = timedelta(minutes=5)  # Rolling 5-minute window
    
    def evaluate_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: Optional[PushbackSeverity] = None
    ) -> EscalationDecision:
        """
        Evaluate an error and decide on escalation
        
        Args:
            error: The error/exception
            context: Context information (user, operation, etc.)
            severity: Optional severity override
            
        Returns:
            Escalation decision
        """
        # Determine severity if not provided
        if severity is None:
            severity = self._assess_severity(error, context)
        
        # Record error
        error_record = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "context": context,
            "timestamp": datetime.now(timezone.utc),
        }
        self.error_history.append(error_record)
        
        # Clean old entries
        self._clean_old_history()
        
        # Check for patterns
        pattern_severity = self._detect_patterns(error_record)
        if pattern_severity:
            severity = max(severity, pattern_severity, key=lambda s: list(PushbackSeverity).index(s))
        
        # Make escalation decision
        decision = self._make_decision(severity, error_record)
        
        # Execute decision
        if decision == EscalationDecision.ESCALATE_TO_AVN:
            self._escalate_to_avn(error_record)
        elif decision == EscalationDecision.IMMEDIATE_ACTION:
            self._immediate_action(error_record)
        elif decision == EscalationDecision.NOTIFY:
            self._send_notification(error_record)
        
        logger.info(
            f"Pushback evaluation: {severity.value} -> {decision.value}",
            extra={"error_type": type(error).__name__, "context": context}
        )
        
        return decision
    
    def _assess_severity(self, error: Exception, context: Dict[str, Any]) -> PushbackSeverity:
        """
        Assess the severity of an error
        
        Args:
            error: The error
            context: Error context
            
        Returns:
            Severity level
        """
        error_type = type(error).__name__
        
        # Critical errors
        critical_patterns = [
            "SecurityError", "AuthenticationError", "DataCorruption",
            "DatabaseConnectionError", "OutOfMemory"
        ]
        if any(pattern in error_type for pattern in critical_patterns):
            return PushbackSeverity.CRITICAL
        
        # High severity
        high_patterns = [
            "PermissionError", "ValidationError", "IntegrityError",
            "Timeout", "ConnectionError"
        ]
        if any(pattern in error_type for pattern in high_patterns):
            return PushbackSeverity.HIGH
        
        # Check context for severity indicators
        if context.get("affects_multiple_users"):
            return PushbackSeverity.HIGH
        
        if context.get("data_loss_risk"):
            return PushbackSeverity.CRITICAL
        
        if context.get("user_facing"):
            return PushbackSeverity.MEDIUM
        
        # Default to low
        return PushbackSeverity.LOW
    
    def _detect_patterns(self, current_error: Dict[str, Any]) -> Optional[PushbackSeverity]:
        """
        Detect error patterns that may warrant escalation
        
        Args:
            current_error: Current error record
            
        Returns:
            Severity if pattern detected, None otherwise
        """
        recent_errors = self._get_recent_errors()
        
        # Check for error bursts
        if len(recent_errors) > 20:
            logger.warning("Error burst detected (>20 errors in 5 minutes)")
            return PushbackSeverity.HIGH
        
        # Check for repeated errors of same type
        same_type_count = sum(
            1 for e in recent_errors
            if e["error_type"] == current_error["error_type"]
        )
        
        if same_type_count >= 5:
            logger.warning(f"Repeated error pattern: {current_error['error_type']} x{same_type_count}")
            return PushbackSeverity.MEDIUM
        
        # Check for cascading failures
        unique_types = set(e["error_type"] for e in recent_errors[-10:])
        if len(unique_types) >= 5:
            logger.warning("Cascading failures detected (5+ different error types)")
            return PushbackSeverity.HIGH
        
        return None
    
    def _make_decision(
        self,
        severity: PushbackSeverity,
        error_record: Dict[str, Any]
    ) -> EscalationDecision:
        """
        Make escalation decision based on severity and history
        
        Args:
            severity: Error severity
            error_record: Error record
            
        Returns:
            Escalation decision
        """
        # Critical always escalates immediately
        if severity == PushbackSeverity.CRITICAL:
            return EscalationDecision.IMMEDIATE_ACTION
        
        # Check frequency against thresholds
        same_type_count = sum(
            1 for e in self._get_recent_errors()
            if e["error_type"] == error_record["error_type"]
        )
        
        threshold = self.escalation_thresholds[severity]
        
        if same_type_count >= threshold:
            return EscalationDecision.ESCALATE_TO_AVN
        
        # High severity gets notification
        if severity == PushbackSeverity.HIGH:
            return EscalationDecision.NOTIFY
        
        # Medium and low just log
        if severity == PushbackSeverity.MEDIUM:
            return EscalationDecision.LOG_ONLY
        
        return EscalationDecision.IGNORE
    
    def _escalate_to_avn(self, error_record: Dict[str, Any]):
        """
        Escalate error to AVN for verification and healing
        
        Args:
            error_record: Error record to escalate
        """
        if not self.avn_client:
            logger.warning("No AVN client configured, cannot escalate")
            return
        
        try:
            escalation_data = {
                "type": "error_escalation",
                "error_type": error_record["error_type"],
                "error_message": error_record["error_message"],
                "severity": error_record["severity"],
                "context": error_record["context"],
                "timestamp": error_record["timestamp"].isoformat(),
                "recent_errors": len(self._get_recent_errors()),
                "escalation_reason": "Threshold exceeded"
            }
            
            # Send to AVN
            self.avn_client.report_issue(escalation_data)
            
            logger.info(f"Escalated error to AVN: {error_record['error_type']}")
            
        except Exception as e:
            logger.error(f"Failed to escalate to AVN: {e}")
    
    def _immediate_action(self, error_record: Dict[str, Any]):
        """
        Take immediate action for critical errors
        
        Args:
            error_record: Critical error record
        """
        logger.critical(
            f"CRITICAL ERROR: {error_record['error_type']}",
            extra=error_record
        )
        
        # Escalate to AVN
        self._escalate_to_avn(error_record)
        
        # Additional immediate actions could include:
        # - Circuit breaker activation
        # - Failover to backup systems
        # - Emergency notifications
        # - Automated rollback
        
        logger.info("Immediate action protocol activated")
    
    def _send_notification(self, error_record: Dict[str, Any]):
        """
        Send notification for high-severity errors
        
        Args:
            error_record: Error record
        """
        logger.warning(
            f"High severity error notification: {error_record['error_type']}",
            extra=error_record
        )
        
        # In production, send to notification service (email, Slack, PagerDuty, etc.)
    
    def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get errors within the time window"""
        cutoff = datetime.now(timezone.utc) - self.time_window
        return [
            e for e in self.error_history
            if e["timestamp"] > cutoff
        ]
    
    def _clean_old_history(self):
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
        
        error_types = {}
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
