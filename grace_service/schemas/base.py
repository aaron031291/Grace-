"""
Base schemas for Grace Service API responses.
"""
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model for all API endpoints."""
    status: str = Field(..., description="Response status: success, error, or warning")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Application-specific error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="Service version")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Optional[Dict[str, Union[int, float]]] = Field(None, description="Service metrics")


class GovernanceValidateRequest(BaseModel):
    """Request schema for governance validation."""
    action: str = Field(..., description="Action to validate")
    context: Dict[str, Any] = Field(..., description="Action context and metadata")
    user_id: Optional[str] = Field(None, description="User identifier")
    priority: str = Field(default="normal", description="Request priority: low, normal, high, critical")


class GovernanceValidateResponse(BaseModel):
    """Response schema for governance validation."""
    approved: bool = Field(..., description="Whether the action was approved")
    decision_id: str = Field(..., description="Unique decision identifier")
    compliance_score: float = Field(..., description="Constitutional compliance score (0-1)")
    violations: list = Field(default_factory=list, description="List of constitutional violations")
    recommendations: list = Field(default_factory=list, description="Recommendations for improvement")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Decision timestamp")


class IngestRequest(BaseModel):
    """Request schema for data ingestion."""
    source_id: str = Field(..., description="Source identifier")
    data: Dict[str, Any] = Field(..., description="Data to ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    priority: str = Field(default="normal", description="Processing priority")


class IngestResponse(BaseModel):
    """Response schema for data ingestion."""
    event_id: str = Field(..., description="Unique event identifier")
    status: str = Field(..., description="Ingestion status")
    trust_score: Optional[float] = Field(None, description="Data trust score")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


class EventStreamMessage(BaseModel):
    """WebSocket event stream message schema."""
    type: str = Field(..., description="Message type")
    event_type: Optional[str] = Field(None, description="Event type for event messages")
    data: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")