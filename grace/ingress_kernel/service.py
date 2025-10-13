"""
Ingress Service - FastAPI facade for the Ingress Kernel.
Provides REST API endpoints for ingress operations.
"""
import asyncio
import logging
from grace.utils.time import now_utc
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .kernel import IngressKernel
from grace.contracts.ingress_contracts import SourceConfig, IngressSnapshot


logger = logging.getLogger(__name__)


# Request/Response models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: now_utc().isoformat())


class SourceRegistrationRequest(BaseModel):
    """Source registration request."""
    kind: str
    uri: str
    auth_mode: str
    secrets_ref: Optional[str] = None
    schedule: str
    parser: str
    parser_opts: Optional[Dict[str, Any]] = None
    target_contract: str
    retention_days: int = 365
    pii_policy: str = "mask"
    governance_label: str = "internal"
    enabled: bool = True


class SourceRegistrationResponse(BaseModel):
    """Source registration response."""
    source_id: str


class CaptureRequest(BaseModel):
    """Capture request for webhook/push data."""
    source_id: str
    payload: Any
    headers: Optional[Dict[str, Any]] = None


class CaptureResponse(BaseModel):
    """Capture response."""
    raw_event_id: str
    status: str = "accepted"


class ReplayRequest(BaseModel):
    """Replay request."""
    snapshot_id: str
    sources: Optional[List[str]] = None


class ReplayResponse(BaseModel):
    """Replay response."""
    job_id: str
    status: str = "started"


class SnapshotResponse(BaseModel):
    """Snapshot export response."""
    snapshot_id: str
    uri: str


class RollbackRequest(BaseModel):
    """Rollback request."""
    to_snapshot: str


class RollbackResponse(BaseModel):
    """Rollback response."""
    status: str = "initiated"
    to_snapshot: str


class MetricsResponse(BaseModel):
    """Metrics response."""
    timeseries: List[Dict[str, Any]]
    summary: Dict[str, Any]


class IngressService:
    """FastAPI service for Ingress Kernel."""
    
    def __init__(self, ingress_kernel: IngressKernel):
        self.kernel = ingress_kernel
        self.app = FastAPI(
            title="Grace Ingress Service",
            version="1.0.0",
            description="Data ingestion service for the Grace system"
        )
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/api/ingress/v1/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                status = self.kernel.get_health_status()
                return HealthResponse(
                    status="healthy" if status["status"] == "running" else "unhealthy"
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unavailable")
        
        @self.app.post("/api/ingress/v1/sources", response_model=SourceRegistrationResponse)
        async def register_source(request: SourceRegistrationRequest):
            """Register a new ingestion source."""
            try:
                # Generate source ID
                from grace.contracts.ingress_contracts import generate_source_id
                source_id = generate_source_id()
                
                # Build configuration
                config = {
                    "source_id": source_id,
                    **request.dict()
                }
                
                # Register source
                registered_id = self.kernel.register_source(config)
                
                return SourceRegistrationResponse(source_id=registered_id)
                
            except Exception as e:
                logger.error(f"Source registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/ingress/v1/sources/{source_id}")
        async def get_source(source_id: str):
            """Get source configuration."""
            source = self.kernel.get_source(source_id)
            if not source:
                raise HTTPException(status_code=404, detail="Source not found")
            return source.dict()
        
        @self.app.post("/api/ingress/v1/sources/{source_id}/enable")
        async def enable_source(source_id: str):
            """Enable a source."""
            source = self.kernel.get_source(source_id)
            if not source:
                raise HTTPException(status_code=404, detail="Source not found")
            
            # Mock implementation - would update source configuration
            return {"enabled": True}
        
        @self.app.post("/api/ingress/v1/capture", response_model=CaptureResponse)
        async def capture_data(request: CaptureRequest, background_tasks: BackgroundTasks):
            """Capture data via webhook/push."""
            try:
                event_id = await self.kernel.capture(
                    source_id=request.source_id,
                    payload=request.payload,
                    headers=request.headers
                )
                
                return CaptureResponse(raw_event_id=event_id)
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Capture failed: {e}")
                raise HTTPException(status_code=500, detail="Capture failed")
        
        @self.app.get("/api/ingress/v1/records/{record_id}")
        async def get_record(record_id: str):
            """Get normalized record."""
            # Mock implementation - would fetch from silver tier
            raise HTTPException(status_code=501, detail="Not implemented")
        
        @self.app.post("/api/ingress/v1/replay", response_model=ReplayResponse)
        async def replay_from_snapshot(request: ReplayRequest):
            """Replay data from snapshot offsets."""
            try:
                job_id = f"replay_{now_utc().strftime('%Y%m%d_%H%M%S')}"
                
                # Mock implementation - would start replay process
                return ReplayResponse(job_id=job_id)
                
            except Exception as e:
                logger.error(f"Replay failed: {e}")
                raise HTTPException(status_code=500, detail="Replay failed")
        
        @self.app.post("/api/ingress/v1/snapshot/export", response_model=SnapshotResponse)
        async def export_snapshot():
            """Export system snapshot."""
            try:
                snapshot = await self.kernel.export_snapshot()
                
                # Mock URI - would be actual storage location
                uri = f"s3://ingress-snapshots/{snapshot.snapshot_id}.json"
                
                return SnapshotResponse(
                    snapshot_id=snapshot.snapshot_id,
                    uri=uri
                )
                
            except Exception as e:
                logger.error(f"Snapshot export failed: {e}")
                raise HTTPException(status_code=500, detail="Snapshot export failed")
        
        @self.app.post("/api/ingress/v1/rollback", response_model=RollbackResponse)
        async def rollback_to_snapshot(request: RollbackRequest):
            """Rollback to previous snapshot."""
            try:
                # Mock implementation - would perform rollback
                return RollbackResponse(to_snapshot=request.to_snapshot)
                
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                raise HTTPException(status_code=500, detail="Rollback failed")
        
        @self.app.get("/api/ingress/v1/metrics", response_model=MetricsResponse)
        async def get_metrics(source_id: Optional[str] = None, since: Optional[str] = None):
            """Get ingress metrics."""
            try:
                # Mock metrics data
                timeseries = [
                    {
                        "timestamp": now_utc().isoformat(),
                        "source_id": source_id or "all",
                        "events_processed": 100,
                        "success_rate": 0.95,
                        "avg_latency_ms": 150
                    }
                ]
                
                summary = {
                    "total_sources": len(self.kernel.sources),
                    "total_events": 1000,
                    "success_rate": 0.95,
                    "avg_latency_ms": 150
                }
                
                return MetricsResponse(
                    timeseries=timeseries,
                    summary=summary
                )
                
            except Exception as e:
                logger.error(f"Metrics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail="Metrics retrieval failed")


def create_ingress_app(ingress_kernel: Optional[IngressKernel] = None) -> FastAPI:
    """Create Ingress FastAPI application."""
    if not ingress_kernel:
        ingress_kernel = IngressKernel()
    
    service = IngressService(ingress_kernel)
    return service.app