"""
System-level API endpoints
"""

from fastapi import APIRouter, Depends
from grace.auth.dependencies import get_current_user
from grace.auth.models import User

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health")
async def system_health():
    """Get overall system health"""
    from grace.trigger_mesh import get_trigger_mesh
    from grace.integration.event_bus import get_event_bus
    
    trigger_mesh = get_trigger_mesh()
    event_bus = get_event_bus()
    
    return {
        "status": "healthy",
        "trigger_mesh": trigger_mesh.get_stats(),
        "event_bus": event_bus.get_metrics()
    }


@router.get("/routes")
async def get_routes(current_user: User = Depends(get_current_user)):
    """Get TriggerMesh routes"""
    from grace.trigger_mesh import get_trigger_mesh
    
    mesh = get_trigger_mesh()
    
    return {
        "routes": [
            {
                "name": r.name,
                "pattern": r.pattern,
                "targets": r.targets,
                "priority": r.priority,
                "filter_count": len(r.filters),
                "action_count": len(r.actions)
            }
            for r in mesh.routes
        ]
    }
