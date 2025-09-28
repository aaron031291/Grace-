"""Governed decision contracts."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .dto_common import BaseDTO
from .governed_request import GovernedRequest


class GovernedDecision(BaseDTO):
    """The result of governance evaluation."""
    request_id: str
    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    policy_results: Dict[str, Any] = Field(default_factory=dict)
    verification_results: Dict[str, Any] = Field(default_factory=dict)
    quorum_results: Optional[Dict[str, Any]] = None
    
    # Decision metadata
    decision_maker: str = "grace-governance"
    execution_approved: bool = False
    conditions: List[str] = Field(default_factory=list)
    expiry: Optional[str] = None  # ISO format datetime string