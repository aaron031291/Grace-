"""
AVN (Adaptive Verification Network) database models
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column

from grace.auth.models import Base


class ErrorAudit(Base):
    """Error audit table for pushback escalation tracking"""
    __tablename__ = 'error_audits'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Error details
    error_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    error_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # For grouping similar errors
    
    # Context
    severity: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    context_json: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    stack_trace: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Actor information
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    endpoint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, index=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    
    # Timestamps
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    
    # Escalation tracking
    escalated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    escalated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    escalation_decision: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    def __repr__(self):
        return f"<ErrorAudit(id={self.id}, type={self.error_type}, severity={self.severity})>"


class AVNAlert(Base):
    """AVN alert records when escalation thresholds are triggered"""
    __tablename__ = 'avn_alerts'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    
    # Alert details
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Trigger information
    error_pattern: Mapped[str] = mapped_column(String(255), nullable=False)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False)
    time_window_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Related errors
    related_error_ids: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # AVN response
    avn_notified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    avn_response: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)
    
    def __repr__(self):
        return f"<AVNAlert(id={self.id}, type={self.alert_type}, status={self.status})>"
