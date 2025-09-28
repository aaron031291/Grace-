"""Governance types and data models."""
from typing import Any, Dict, List, Optional
from enum import Enum

try:
    from pydantic import BaseModel, Field
    from ..contracts.dto_common import BaseDTO
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class PolicyType(str, Enum):
    """Types of governance policies."""
    ACCESS_CONTROL = "access_control"
    CONTENT_FILTER = "content_filter" 
    RISK_ASSESSMENT = "risk_assessment"
    QUORUM_REQUIRED = "quorum_required"
    APPROVAL_WORKFLOW = "approval_workflow"


if PYDANTIC_AVAILABLE:
    class PolicyResult(BaseModel):
        """Result of a policy evaluation."""
        policy_id: str
        policy_type: PolicyType
        passed: bool
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str
        metadata: Optional[Dict[str, Any]] = None


    class VerificationResult(BaseModel):
        """Result of request verification."""
        verified: bool
        confidence: float = Field(ge=0.0, le=1.0)
        verification_method: str
        details: Dict[str, Any] = Field(default_factory=dict)


    class QuorumConsensus(BaseModel):
        """Result of quorum consensus process."""
        consensus_reached: bool
        agreement_level: float = Field(ge=0.0, le=1.0)
        participating_memories: List[str] = Field(default_factory=list)
        consensus_view: str
        dissenting_views: List[str] = Field(default_factory=list)
        confidence: float = Field(ge=0.0, le=1.0)
else:
    # Fallback dict-based types when Pydantic is not available
    PolicyResult = Dict[str, Any]
    VerificationResult = Dict[str, Any] 
    QuorumConsensus = Dict[str, Any]