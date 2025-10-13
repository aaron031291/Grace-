"""Governed request contracts."""

from typing import Any, Dict, List
from pydantic import Field
from .dto_common import BaseDTO


class GovernedRequest(BaseDTO):
    """A request that requires governance evaluation."""

    request_type: str
    content: str
    requester: str
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)
    tags: List[str] = Field(default_factory=list)

    # Policy-relevant fields
    policy_domains: List[str] = Field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high
    requires_quorum: bool = False
