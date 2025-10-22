"""
Kernel management API endpoints
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from grace.auth.dependencies import get_current_user, require_admin
from grace.auth.models import User

router = APIRouter(prefix="/kernels", tags=["Kernels"])


class KernelStatus(BaseModel):
    """Kernel status response"""
    name: str
    running: bool
    uptime_seconds: float
    events_processed: int
    last_activity: str


class KernelCommand(BaseModel):
    """Kernel command request"""
    action: str  # "start" | "stop" | "restart"


# Global registry of kernel modules
_KERNEL_MODULES = {
    "multi_os": "grace.kernels.multi_os",
    "mldl": "grace.kernels.mldl",
    "resilience": "grace.kernels.resilience",
}


@router.get("/", response_model=List[str])
async def list_kernels(current_user: User = Depends(get_current_user)):
    """List all available kernels"""
    return list(_KERNEL_MODULES.keys())


@router.get("/{kernel_name}/status", response_model=KernelStatus)
async def get_kernel_status(
    kernel_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a specific kernel"""
    if kernel_name not in _KERNEL_MODULES:
        raise HTTPException(status_code=404, detail=f"Kernel '{kernel_name}' not found")
    
    # Import kernel module to check status
    try:
        from importlib import import_module
        from datetime import datetime
        
        mod = import_module(_KERNEL_MODULES[kernel_name])
        
        # Get kernel state if available
        running = getattr(mod, "_running", False)
        start_time = getattr(mod, "_start_time", None)
        events_processed = getattr(mod, "_events_processed", 0)
        
        uptime = 0.0
        if running and start_time:
            uptime = (datetime.utcnow() - start_time).total_seconds()
        
        return KernelStatus(
            name=kernel_name,
            running=running,
            uptime_seconds=uptime,
            events_processed=events_processed,
            last_activity=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get kernel status: {str(e)}")


@router.post("/{kernel_name}/control")
async def control_kernel(
    kernel_name: str,
    command: KernelCommand,
    current_user: User = Depends(require_admin)
):
    """Start, stop, or restart a kernel (admin only)"""
    if kernel_name not in _KERNEL_MODULES:
        raise HTTPException(status_code=404, detail=f"Kernel '{kernel_name}' not found")
    
    try:
        from importlib import import_module
        
        mod = import_module(_KERNEL_MODULES[kernel_name])
        
        if command.action == "start":
            if hasattr(mod, "start"):
                await mod.start()
                return {"status": "started", "kernel": kernel_name}
            raise HTTPException(status_code=400, detail="Kernel does not support start")
        
        elif command.action == "stop":
            if hasattr(mod, "stop"):
                await mod.stop()
                return {"status": "stopped", "kernel": kernel_name}
            raise HTTPException(status_code=400, detail="Kernel does not support stop")
        
        elif command.action == "restart":
            if hasattr(mod, "stop") and hasattr(mod, "start"):
                await mod.stop()
                await mod.start()
                return {"status": "restarted", "kernel": kernel_name}
            raise HTTPException(status_code=400, detail="Kernel does not support restart")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {command.action}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to control kernel: {str(e)}")


@router.get("/{kernel_name}/health")
async def get_kernel_health(
    kernel_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get health metrics for a kernel"""
    if kernel_name not in _KERNEL_MODULES:
        raise HTTPException(status_code=404, detail=f"Kernel '{kernel_name}' not found")
    
    try:
        from importlib import import_module
        
        mod = import_module(_KERNEL_MODULES[kernel_name])
        
        # Get health metrics if available
        health = {
            "status": "healthy" if getattr(mod, "_running", False) else "stopped",
            "errors": getattr(mod, "_error_count", 0),
            "warnings": getattr(mod, "_warning_count", 0),
            "uptime_seconds": 0.0
        }
        
        start_time = getattr(mod, "_start_time", None)
        if start_time:
            from datetime import datetime
            health["uptime_seconds"] = (datetime.utcnow() - start_time).total_seconds()
        
        # Add kernel-specific health checks
        if hasattr(mod, "get_health"):
            health.update(mod.get_health())
        
        return health
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health: {str(e)}")
