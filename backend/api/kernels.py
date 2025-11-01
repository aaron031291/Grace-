"""
Kernel API Endpoints

Exposes ALL Grace kernels via REST API.

Endpoints:
- GET /api/kernels/status - All kernel status
- GET /api/kernels/{name}/health - Specific kernel health
- GET /api/kernels/list - List all kernels
- GET /api/kernels/{name}/info - Kernel information

Makes all kernels visible and accessible!
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any

router = APIRouter(prefix="/kernels", tags=["kernels"])


@router.get("/status")
async def get_all_kernels_status(request: Request):
    """
    Get status of ALL Grace kernels.
    
    Shows which kernels are operational and which failed.
    """
    try:
        if hasattr(request.app.state, 'kernel_manager'):
            kernel_manager = request.app.state.kernel_manager
            return kernel_manager.get_all_status()
        else:
            return {
                "error": "Kernel manager not initialized",
                "message": "Backend started without kernel initialization. This is OK for basic operation.",
                "kernels": {},
                "total_kernels": 0,
                "operational_kernels": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_all_kernels(request: Request):
    """List all available kernels"""
    try:
        if hasattr(request.app.state, 'kernel_manager'):
            kernel_manager = request.app.state.kernel_manager
            
            return {
                "kernels": list(kernel_manager.kernels.keys()),
                "count": len(kernel_manager.kernels)
            }
        else:
            return {"kernels": [], "count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kernel_name}/health")
async def get_kernel_health(kernel_name: str, request: Request):
    """Get health status of specific kernel"""
    try:
        if not hasattr(request.app.state, 'kernel_manager'):
            raise HTTPException(status_code=503, detail="Kernel manager not available")
        
        kernel_manager = request.app.state.kernel_manager
        
        if not kernel_manager.is_kernel_operational(kernel_name):
            return {
                "kernel": kernel_name,
                "status": "not_operational",
                "error": kernel_manager.kernel_status.get(kernel_name, {}).get('error', 'Unknown')
            }
        
        kernel = kernel_manager.get_kernel(kernel_name)
        
        if kernel is None:
            raise HTTPException(status_code=404, detail=f"Kernel '{kernel_name}' not found")
        
        # Try to get health from kernel if it has health_check method
        if hasattr(kernel, 'health_check'):
            health = await kernel.health_check()
            return {"kernel": kernel_name, "status": "operational", "health": health}
        elif hasattr(kernel, 'get_stats'):
            stats = kernel.get_stats()
            return {"kernel": kernel_name, "status": "operational", "stats": stats}
        else:
            return {"kernel": kernel_name, "status": "operational"}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{kernel_name}/info")
async def get_kernel_info(kernel_name: str, request: Request):
    """Get information about specific kernel"""
    try:
        if not hasattr(request.app.state, 'kernel_manager'):
            raise HTTPException(status_code=503, detail="Kernel manager not available")
        
        kernel_manager = request.app.state.kernel_manager
        kernel = kernel_manager.get_kernel(kernel_name)
        
        if kernel is None:
            raise HTTPException(status_code=404, detail=f"Kernel '{kernel_name}' not found")
        
        return {
            "kernel": kernel_name,
            "type": type(kernel).__name__,
            "operational": kernel_manager.is_kernel_operational(kernel_name),
            "status": kernel_manager.kernel_status.get(kernel_name, {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
