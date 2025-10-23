"""
Public-facing API endpoints for external integration
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from grace.auth.dependencies import get_current_user, require_role
from grace.auth.models import User
from grace.config import get_settings

router = APIRouter(prefix="/public", tags=["Public API"])


class SearchRequest(BaseModel):
    """Public API search request"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")


class SearchResult(BaseModel):
    """Public API search result"""
    id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Public API search response"""
    results: List[SearchResult]
    total: int
    query: str


class DecisionRequest(BaseModel):
    """Public API decision request"""
    context: Dict[str, Any] = Field(..., description="Decision context")
    require_consensus: bool = Field(False, description="Require swarm consensus")
    include_reasoning: bool = Field(True, description="Include reasoning chain")


class DecisionResponse(BaseModel):
    """Public API decision response"""
    decision: Dict[str, Any]
    confidence: float
    reasoning: Optional[List[Dict[str, Any]]] = None
    consensus_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Public API health response"""
    status: str
    version: str
    environment: str
    features: Dict[str, bool]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Public health check",
    description="Get system health and feature status (no authentication required)"
)
async def public_health() -> HealthResponse:
    """Public health check endpoint"""
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        environment=settings.environment,
        features=settings.get_deployment_info()["features"]
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    description="Perform semantic search across documents (requires authentication)"
)
async def public_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
) -> SearchResponse:
    """Public semantic search endpoint"""
    # TODO: Implement actual search
    return SearchResponse(
        results=[],
        total=0,
        query=request.query
    )


@router.post(
    "/decision",
    response_model=DecisionResponse,
    summary="Request decision",
    description="Request a decision from the Grace system (requires admin role)"
)
async def public_decision(
    request: DecisionRequest,
    current_user: User = Depends(require_role(["admin"]))
) -> DecisionResponse:
    """Public decision endpoint"""
    # TODO: Implement actual decision logic
    return DecisionResponse(
        decision={"placeholder": True},
        confidence=0.0,
        reasoning=None if not request.include_reasoning else []
    )


@router.get(
    "/capabilities",
    summary="Get system capabilities",
    description="Get available system capabilities and their status"
)
async def get_capabilities(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get system capabilities"""
    settings = get_settings()
    
    return {
        "capabilities": {
            "semantic_search": {
                "available": True,
                "provider": settings.embedding.provider,
                "dimension": settings.embedding.dimension
            },
            "swarm_coordination": {
                "available": settings.swarm.enabled,
                "transport": settings.swarm.transport if settings.swarm.enabled else None
            },
            "quantum_reasoning": {
                "available": settings.transcendence.quantum_enabled
            },
            "scientific_discovery": {
                "available": settings.transcendence.discovery_enabled
            },
            "impact_evaluation": {
                "available": settings.transcendence.impact_enabled
            },
            "governance": {
                "available": True,
                "constitutional_validation": True
            }
        }
    }
