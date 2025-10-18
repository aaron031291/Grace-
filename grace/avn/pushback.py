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

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from grace.avn.models import ErrorAudit, AVNAlert

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
    
    def __repr__(self):
        return f"<ThresholdRule({self.name}: {self.error_count} errors in {self.time_window_seconds}s)>"


class PushbackEscalation:
    """
    Database-backed pushback escalation system with time-windowed threshold monitoring
    """
    
    def __init__(self, avn_client=None):
        """
        Initialize pushback escalation system
        
        Args:
            avn_client: AVN client for sending escalations
        """
        self.avn_client = avn_client
        
        # Define threshold rules (can be loaded from config)
        self.threshold_rules = [
            # Critical errors escalate immediately
            ThresholdRule("critical_immediate", 1, 60, PushbackSeverity.CRITICAL, EscalationDecision.IMMEDIATE_ACTION),
            
            # High severity: 2 errors in 5 minutes
            ThresholdRule("high_rapid", 2, 300, PushbackSeverity.HIGH, EscalationDecision.ESCALATE_TO_AVN),
            
            # Medium severity: 5 errors in 5 minutes
            ThresholdRule("medium_burst", 5, 300, PushbackSeverity.MEDIUM, EscalationDecision.ESCALATE_TO_AVN),
            
            # Low severity: 10 errors in 10 minutes
            ThresholdRule("low_sustained", 10, 600, PushbackSeverity.LOW, EscalationDecision.NOTIFY),
            
            # Generic burst detection: 15 errors in 5 minutes regardless of severity
            ThresholdRule("generic_burst", 15, 300, PushbackSeverity.HIGH, EscalationDecision.ESCALATE_TO_AVN),
        ]
        
        logger.info(f"Initialized PushbackEscalation with {len(self.threshold_rules)} threshold rules")
    
    async def record_and_evaluate_error(
        self,
        db: AsyncSession,
        error: Exception,
        context: Dict[str, Any],
        severity: Optional[PushbackSeverity] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> EscalationDecision:
        """
        Record error in database and evaluate escalation thresholds
        
        Args:
            db: Async database session
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
        
        # Record error in audit table
        error_audit = ErrorAudit(
            id=str(uuid.uuid4()),
            error_type=type(error).__name__,
            error_message=str(error),
            error_hash=error_hash,
            severity=severity.value,
            context_json=context,
            stack_trace=traceback.format_exc() if context.get('include_stack_trace') else None,
            user_id=user_id,
            endpoint=endpoint,
            ip_address=ip_address,
            occurred_at=datetime.now(timezone.utc)
        )
        
        db.add(error_audit)
        await db.flush()  # Ensure ID is available
        
        logger.info(
            f"Recorded error audit: {error_audit.id} "
            f"(type={error_audit.error_type}, severity={severity.value})"
        )
        
        # Evaluate thresholds and determine escalation decision
        decision = await self._evaluate_thresholds(db, error_audit, error_hash, severity)
        
        # Update audit record with escalation decision
        error_audit.escalation_decision = decision.value
        
        if decision in [EscalationDecision.ESCALATE_TO_AVN, EscalationDecision.IMMEDIATE_ACTION]:
            error_audit.escalated = True
            error_audit.escalated_at = datetime.now(timezone.utc)
            
            # Create AVN alert
            await self._create_avn_alert(db, error_audit, error_hash, decision)
        
        await db.commit()
        
        logger.info(f"Escalation decision for {error_audit.id}: {decision.value}")
        
        return decision
    
    async def _evaluate_thresholds(
        self,
        db: AsyncSession,
        current_error: ErrorAudit,
        error_hash: str,
        severity: PushbackSeverity
    ) -> EscalationDecision:
        """
        Evaluate threshold rules against recent error history
        
        Args:
            db: Database session
            current_error: Current error audit record
            error_hash: Error hash for grouping
            severity: Error severity
            
        Returns:
            Escalation decision
        """
        now = datetime.now(timezone.utc)
        
        # Check each threshold rule
        for rule in self.threshold_rules:
            # Skip rules that don't match severity (except generic rules)
            if rule.severity != severity and rule.name != "generic_burst":
                continue
            
            # Calculate time window
            window_start = now - timedelta(seconds=rule.time_window_seconds)
            
            # Query error count in time window
            if rule.name == "generic_burst":
                # Generic burst: count all errors regardless of type
                query = select(func.count(ErrorAudit.id)).where(
                    ErrorAudit.occurred_at >= window_start
                )
            else:
                # Specific pattern: count errors with same hash
                query = select(func.count(ErrorAudit.id)).where(
                    and_(
                        ErrorAudit.error_hash == error_hash,
                        ErrorAudit.occurred_at >= window_start
                    )
                )
            
            result = await db.execute(query)
            error_count = result.scalar()
            
            logger.debug(
                f"Rule '{rule.name}': {error_count}/{rule.error_count} errors "
                f"in {rule.time_window_seconds}s window"
            )
            
            # Check if threshold exceeded
            if error_count >= rule.error_count:
                logger.warning(
                    f"Threshold exceeded for rule '{rule.name}': "
                    f"{error_count} errors in {rule.time_window_seconds}s "
                    f"(threshold: {rule.error_count})"
                )
                return rule.decision
        
        # No threshold exceeded, default to log only
        return EscalationDecision.LOG_ONLY
    
    async def _create_avn_alert(
        self,
        db: AsyncSession,
        error_audit: ErrorAudit,
        error_hash: str,
        decision: EscalationDecision
    ):
        """
        Create AVN alert record and notify AVN if client available
        
        Args:
            db: Database session
            error_audit: Error audit that triggered alert
            error_hash: Error hash for querying related errors
            decision: Escalation decision
        """
        now = datetime.now(timezone.utc)
        
        # Find related errors in last 10 minutes
        window_start = now - timedelta(minutes=10)
        query = select(ErrorAudit.id).where(
            and_(
                ErrorAudit.error_hash == error_hash,
                ErrorAudit.occurred_at >= window_start
            )
        ).limit(50)
        
        result = await db.execute(query)
        related_error_ids = [row[0] for row in result.fetchall()]
        
        # Determine alert severity
        alert_severity = error_audit.severity
        if decision == EscalationDecision.IMMEDIATE_ACTION:
            alert_severity = "critical"
        
        # Create alert
        alert = AVNAlert(
            id=str(uuid.uuid4()),
            alert_type="error_threshold_exceeded",
            severity=alert_severity,
            title=f"Error threshold exceeded: {error_audit.error_type}",
            description=(
                f"Multiple occurrences of {error_audit.error_type} detected. "
                f"Error count: {len(related_error_ids)} in recent window. "
                f"Message: {error_audit.error_message}"
            ),
            error_pattern=error_audit.error_type,
            error_count=len(related_error_ids),
            time_window_seconds=600,  # 10 minutes
            threshold_value=len(related_error_ids),
            related_error_ids=related_error_ids,
            triggered_at=now,
            status="active"
        )
        
       
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
