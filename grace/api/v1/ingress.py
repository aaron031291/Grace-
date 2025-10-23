"""
Ingress API endpoints - External request entry point
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from grace.auth.dependencies import get_current_user
from grace.auth.models import User

router = APIRouter(prefix="/ingress", tags=["Ingress"])


class IngressRequest(BaseModel):
    """External ingress request"""
    request_type: str = Field(..., description="Request type (e.g., inference.text)")
    payload: Dict[str, Any] = Field(..., description="Request payload")
    priority: str = Field("normal", pattern="^(low|normal|high|critical)$")


class IngressResponse(BaseModel):
    """Ingress response"""
    status: str
    request_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/request", response_model=IngressResponse)
async def submit_request(
    request: IngressRequest,
    current_user: User = Depends(get_current_user),
    x_source: Optional[str] = Header(None),
    x_correlation_id: Optional[str] = Header(None)
):
    """
    Submit external request through ingress pipeline
    
    Processing:
    1. Rate limiting check
    2. Trust score validation
    3. Governance policy check
    4. Route to internal kernel
    5. Return response
    """
    # Get ingress kernel from app state
    # In production, this would be injected via dependency
    from grace.ingress_kernel.service import IngressKernel
    
    # Placeholder: would get from app state
    # ingress = request.app.state.ingress_kernel
    
    return IngressResponse(
        status="error",
        error="Ingress kernel not wired to HTTP endpoint yet"
    )


@router.get("/health")
async def ingress_health():
    """Get ingress kernel health"""
    # Placeholder: would get from app state
    return {
        "status": "pending_implementation",
        "message": "Ingress kernel health endpoint"
    }


@router.get("/metrics")
async def ingress_metrics(current_user: User = Depends(get_current_user)):
    """Get ingress metrics"""
    # Placeholder: would get from app state
    return {
        "requests_received": 0,
        "requests_accepted": 0,
        "requests_rejected": 0
    }
