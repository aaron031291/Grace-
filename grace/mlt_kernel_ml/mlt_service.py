"""
MLT Service - External FastAPI facade for status, dashboards, and job control.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename

from .contracts import Experience, Insight, AdaptationPlan


logger = logging.getLogger(__name__)


# Request/Response models
class ExperienceRequest(BaseModel):
    source: str
    task: str
    context: Dict[str, Any]
    signals: Dict[str, Any]
    ground_truth_lag_s: int = 86400


class StatusResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    stats: Dict[str, Any]
    timestamp: str


class InsightResponse(BaseModel):
    insights: List[Dict[str, Any]]
    total: int
    generated_at: str


class PlanResponse(BaseModel):
    plans: List[Dict[str, Any]]
    total: int
    status: str


class JobControlRequest(BaseModel):
    action: str  # "start", "stop", "pause", "resume"
    job_id: Optional[str] = None
    job_type: Optional[str] = None


class DashboardData(BaseModel):
    experiences: Dict[str, Any]
    insights: Dict[str, Any]
    plans: Dict[str, Any]
    health: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]


class MLTService:
    """FastAPI service facade for MLT kernel external interface."""
    
    def __init__(self, mlt_kernel=None):
        self.app = FastAPI(
            title="MLT Kernel Service",
            description="Machine Learning Tuning Kernel API",
            version="1.0.0"
        )
        self.mlt_kernel = mlt_kernel
        self.job_registry = {}  # Track active jobs
        
        self._register_routes()
    
    def set_mlt_kernel(self, mlt_kernel):
        """Set the MLT kernel instance."""
        self.mlt_kernel = mlt_kernel
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.get("/api/mlt/status", response_model=StatusResponse)
        async def get_status():
            """Get MLT kernel status and health."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                components = {
                    "experience_collector": hasattr(self.mlt_kernel, 'experience_collector'),
                    "insight_generator": hasattr(self.mlt_kernel, 'insight_generator'),
                    "adaptation_planner": hasattr(self.mlt_kernel, 'adaptation_planner'),
                    "policy_tuner": hasattr(self.mlt_kernel, 'policy_tuner'),
                    "snapshot_manager": hasattr(self.mlt_kernel, 'snapshot_manager')
                }
                
                stats = await self.mlt_kernel.get_comprehensive_stats()
                
                return StatusResponse(
                    status="running" if all(components.values()) else "degraded",
                    components=components,
                    stats=stats,
                    timestamp=iso_format()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
        
        @self.app.post("/api/mlt/experience")
        async def submit_experience(request: ExperienceRequest):
            """Submit a new experience for processing."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                from .contracts import ExperienceSource
                source = ExperienceSource(request.source)
                
                experience = await self.mlt_kernel.experience_collector.collect_experience(
                    source=source,
                    raw_data=request.dict()
                )
                
                return {"experience_id": experience.experience_id, "status": "collected"}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to submit experience: {str(e)}")
        
        @self.app.get("/api/mlt/insights", response_model=InsightResponse)
        async def get_insights(limit: int = 20):
            """Get recent insights."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                insights = self.mlt_kernel.insight_generator.get_recent_insights(limit)
                
                return InsightResponse(
                    insights=[insight.to_dict() for insight in insights],
                    total=len(insights),
                    generated_at=iso_format()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")
        
        @self.app.get("/api/mlt/plans", response_model=PlanResponse)
        async def get_plans(limit: int = 10):
            """Get recent adaptation plans."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                plans = self.mlt_kernel.adaptation_planner.get_recent_plans(limit)
                
                return PlanResponse(
                    plans=[plan.to_dict() for plan in plans],
                    total=len(plans),
                    status="active"
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get plans: {str(e)}")
        
        @self.app.post("/api/mlt/jobs/control")
        async def control_job(request: JobControlRequest):
            """Control MLT jobs (behind governance approval)."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            # Note: This is proposal-only, actual execution requires governance approval
            try:
                proposal = {
                    "action": request.action,
                    "job_id": request.job_id,
                    "job_type": request.job_type,
                    "timestamp": iso_format(),
                    "status": "proposed"
                }
                
                # In a real implementation, this would go through governance
                return {
                    "proposal_id": f"job_{format_for_filename()}",
                    "status": "proposed_to_governance",
                    "proposal": proposal
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to control job: {str(e)}")
        
        @self.app.get("/api/mlt/dashboard", response_model=DashboardData)
        async def get_dashboard():
            """Get dashboard data for read-only monitoring."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                # Collect dashboard metrics
                exp_stats = self.mlt_kernel.experience_collector.get_stats()
                insight_stats = self.mlt_kernel.insight_generator.get_stats()
                plan_stats = self.mlt_kernel.adaptation_planner.get_stats()
                
                # Recent activity
                recent_experiences = self.mlt_kernel.experience_collector.get_recent_experiences(5)
                recent_insights = self.mlt_kernel.insight_generator.get_recent_insights(5)
                
                recent_activity = []
                for exp in recent_experiences:
                    recent_activity.append({
                        "type": "experience",
                        "id": exp.experience_id,
                        "source": exp.source.value,
                        "timestamp": exp.timestamp.isoformat()
                    })
                
                for insight in recent_insights:
                    recent_activity.append({
                        "type": "insight",
                        "id": insight.insight_id,
                        "insight_type": insight.type.value,
                        "timestamp": insight.timestamp.isoformat()
                    })
                
                # Sort by timestamp
                recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
                
                return DashboardData(
                    experiences=exp_stats,
                    insights=insight_stats,
                    plans=plan_stats,
                    health={
                        "status": "healthy",
                        "uptime": "running",
                        "components_ok": 5
                    },
                    recent_activity=recent_activity[:10]
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")
        
        @self.app.get("/api/mlt/snapshots")
        async def get_snapshots(limit: int = 10):
            """Get recent snapshots."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                snapshots = self.mlt_kernel.snapshot_manager.list_snapshots(limit)
                
                return {
                    "snapshots": [snapshot.to_dict() for snapshot in snapshots],
                    "total": len(snapshots)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get snapshots: {str(e)}")
        
        @self.app.post("/api/mlt/snapshots")
        async def create_snapshot():
            """Create a new snapshot (proposal-only)."""
            if not self.mlt_kernel:
                raise HTTPException(status_code=503, detail="MLT kernel not available")
            
            try:
                # This is a proposal - actual snapshot creation requires governance approval
                proposal = {
                    "action": "create_snapshot",
                    "timestamp": iso_format(),
                    "rationale": "Manual snapshot request"
                }
                
                return {
                    "proposal_id": f"snap_{format_for_filename()}",
                    "status": "proposed_to_governance",
                    "proposal": proposal
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to propose snapshot: {str(e)}")
        
        @self.app.get("/api/mlt/health")
        async def health_check():
            """Simple health check endpoint."""
            if not self.mlt_kernel:
                return {"status": "unavailable"}
            
            return {
                "status": "ok",
                "timestamp": iso_format(),
                "version": "1.0.0"
            }
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app