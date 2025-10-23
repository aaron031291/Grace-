"""
AVN (Adaptive Verification Network) API endpoints
"""

from typing import List, Optional
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from grace.auth.models import User
from grace.auth.dependencies import get_current_user, require_role
from grace.database import get_async_db
from grace.avn.pushback import PushbackEscalation, PushbackSeverity
from grace.avn.models import AVNAlert, ErrorAudit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/avn", tags=["AVN"])

# Global pushback escalation instance
_pushback_instance = None


def get_pushback() -> PushbackEscalation:
    """Get or create pushback escalation instance"""
    global _pushback_instance
    if _pushback_instance is None:
        _pushback_instance = PushbackEscalation()
    return _pushback_instance


# Pydantic schemas
class ErrorReportRequest(BaseModel):
    error_type: str
    error_message: str
    context: dict = Field(default_factory=dict)
    severity: Optional[str] = None
    endpoint: Optional[str] = None


class AVNAlertResponse(BaseModel):
    id: str
    alert_type: str
    severity: str
    title: str
    description: str
    error_count: int
    time_window_seconds: int
    triggered_at: datetime
    status: str
    avn_notified: bool
    
    class Config:
        from_attributes = True


class ErrorStatisticsResponse(BaseModel):
    time_window_minutes: int
    total_errors: int
    severity_breakdown: dict
    escalated_errors: int
    active_alerts: int
    escalation_rate: float


@router.post("/report-error")
async def report_error(
    error_report: ErrorReportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Report an error for pushback escalation evaluation
    
    This endpoint:
    1. Records the error in the audit table
    2. Evaluates threshold rules
    3. Escalates to AVN if thresholds are exceeded
    """
    pushback = get_pushback()
    
    try:
        # Create a mock exception for recording
        class ReportedError(Exception):
            pass
        
        error = ReportedError(error_report.error_message)
        error.__class__.__name__ = error_report.error_type
        
        # Parse severity
        severity = None
        if error_report.severity:
            try:
                severity = PushbackSeverity[error_report.severity.upper()]
            except KeyError:
                pass
        
        # Record and evaluate
        decision = await pushback.record_and_evaluate_error(
            db=db,
            error=error,
            context=error_report.context,
            severity=severity,
            user_id=current_user.id,
            endpoint=error_report.endpoint
        )
        
        logger.info(f"Error reported by {current_user.username}: decision={decision.value}")
        
        return {
            "recorded": True,
            "escalation_decision": decision.value,
            "message": "Error recorded and evaluated for escalation"
        }
        
    except Exception as e:
        logger.error(f"Error in report_error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to report error: {str(e)}"
        )


@router.get("/alerts", response_model=List[AVNAlertResponse])
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    status_filter: Optional[str] = Query(None),
    current_user: User = Depends(require_role(["admin", "superuser"])),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get recent AVN alerts (admin only)
    """
    pushback = get_pushback()
    
    try:
        alerts = await pushback.get_recent_alerts(
            db=db,
            limit=limit,
            status_filter=status_filter
        )
        
        return [
            AVNAlertResponse(
                id=alert.id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                error_count=alert.error_count,
                time_window_seconds=alert.time_window_seconds,
                triggered_at=alert.triggered_at,
                status=alert.status,
                avn_notified=alert.avn_notified
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alerts: {str(e)}"
        )


@router.get("/statistics", response_model=ErrorStatisticsResponse)
async def get_error_statistics(
    time_window_minutes: int = Query(60, ge=1, le=1440),
    current_user: User = Depends(require_role(["admin", "superuser"])),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get error statistics for monitoring (admin only)
    """
    pushback = get_pushback()
    
    try:
        stats = await pushback.get_error_statistics(
            db=db,
            time_window_minutes=time_window_minutes
        )
        
        return ErrorStatisticsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch statistics: {str(e)}"
        )


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user: User = Depends(require_role(["admin", "superuser"])),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Mark an alert as resolved (admin only)
    """
    from sqlalchemy import select, update
    
    try:
        # Update alert status
        stmt = (
            update(AVNAlert)
            .where(AVNAlert.id == alert_id)
            .values(
                status="resolved",
                resolved_at=datetime.now(timezone.utc)
            )
        )
        
        result = await db.execute(stmt)
        await db.commit()
        
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        logger.info(f"Alert {alert_id} resolved by {current_user.username}")
        
        return {
            "resolved": True,
            "alert_id": alert_id,
            "resolved_by": current_user.username,
            "resolved_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve alert: {str(e)}"
        )
