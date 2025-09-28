"""Data models for BusinessOps Kernel"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Step(BaseModel):
    """Individual step in a plan"""
    name: str
    plugin: str
    args: Dict[str, Any] = Field(default_factory=dict)
    timeout_s: Optional[int] = None
    retries: Optional[int] = None
    idempotency_key: Optional[str] = None


class Plan(BaseModel):
    """Execution plan containing steps"""
    mode: Literal["sequential", "parallel"] = "sequential"
    steps: List[Step]


class GovernedDecisionDTO(BaseModel):
    """Input decision from Governance"""
    decision_id: str
    approved: bool
    plan: Plan
    w5h: Optional[Dict[str, Any]] = None
    policy_tags: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    """Result of executing a single step"""
    step_name: str
    status: Literal["success", "error", "timeout", "skipped"]
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int


class RunReport(BaseModel):
    """Overall execution report"""
    decision_id: str
    approved: bool
    overall_status: Literal["success", "partial_fail", "skipped", "fail"]
    steps: List[StepResult]


class MemoryEntry(BaseModel):
    """MTL memory entry for audit logging"""
    type: str
    content: Dict[str, Any]
    w5h: Dict[str, Any]
    timestamp: Optional[str] = None
    
    
class PluginMetadata(BaseModel):
    """Plugin registry metadata"""
    name: str
    version: str
    capabilities: List[str]
    allowed: bool = True
    hash: Optional[str] = None