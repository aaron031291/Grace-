"""
Governance API routes for Grace Service.
"""
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import structlog

from ..schemas.base import BaseResponse, GovernanceValidateRequest, GovernanceValidateResponse

logger = structlog.get_logger(__name__)

governance_router = APIRouter()


def get_governance_kernel():
    """Dependency to get governance kernel."""
    from ..app import app_state
    kernel = app_state.get("governance_kernel")
    if not kernel:
        raise HTTPException(status_code=503, detail="Governance kernel not initialized")
    return kernel


@governance_router.post("/validate", response_model=GovernanceValidateResponse)
async def validate_governance_request(
    request: GovernanceValidateRequest,
    background_tasks: BackgroundTasks,
    governance_kernel = Depends(get_governance_kernel)
):
    """
    Validate an action against constitutional governance rules.
    
    This is the primary endpoint for governance validation that other
    services and applications should use for decision-making.
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Processing governance validation request", 
            request_id=request_id,
            action=request.action,
            user_id=request.user_id,
            priority=request.priority
        )
        
        # Create governance request context
        governance_context = {
            "action_type": request.action,
            "details": request.context,
            "user_id": request.user_id,
            "priority": request.priority,
            "request_id": request_id
        }
        
        # Call the governance kernel for validation
        decision = await governance_kernel.evaluate_action(
            request.action, 
            request.context,
            {"user_id": request.user_id}
        )
        
        # Schedule background audit logging
        background_tasks.add_task(
            _log_governance_decision,
            request_id=request_id,
            decision=decision,
            context=governance_context
        )
        
        response = GovernanceValidateResponse(
            approved=decision.approved,
            decision_id=decision.decision_id,
            compliance_score=decision.compliance_score,
            violations=[{
                "principle": v.principle,
                "severity": v.severity,
                "description": v.description,
                "recommendation": v.recommendation
            } for v in decision.violations],
            recommendations=decision.recommendations
        )
        
        logger.info(
            "Governance validation completed",
            request_id=request_id,
            approved=decision.approved,
            compliance_score=decision.compliance_score
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Governance validation failed",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Governance validation failed: {str(e)}"
        )


@governance_router.get("/status/{decision_id}")
async def get_decision_status(
    decision_id: str,
    governance_kernel = Depends(get_governance_kernel)
):
    """Get the status of a governance decision."""
    try:
        # This would query the governance kernel for decision status
        # For now, return a placeholder response
        return BaseResponse(
            status="success",
            message="Decision status retrieved",
            data={
                "decision_id": decision_id,
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z"
            }
        )
        
    except Exception as e:
        logger.error("Failed to get decision status", decision_id=decision_id, error=str(e))
        raise HTTPException(
            status_code=404,
            detail="Decision not found"
        )


@governance_router.get("/pending")
async def get_pending_decisions(
    governance_kernel = Depends(get_governance_kernel)
):
    """Get list of pending governance decisions."""
    try:
        # This would query the governance kernel for pending decisions
        return BaseResponse(
            status="success",
            message="Pending decisions retrieved",
            data={
                "pending_count": 0,
                "decisions": []
            }
        )
        
    except Exception as e:
        logger.error("Failed to get pending decisions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve pending decisions"
        )


async def _log_governance_decision(
    request_id: str,
    decision: Any,
    context: Dict[str, Any]
):
    """Background task to log governance decisions for audit trail."""
    try:
        logger.info(
            "Governance decision audit log",
            request_id=request_id,
            decision_id=decision.decision_id,
            approved=decision.approved,
            compliance_score=decision.compliance_score,
            context=context
        )
    except Exception as e:
        logger.error("Failed to log governance decision", error=str(e))