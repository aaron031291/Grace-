"""Health check endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import asyncio

from ..database import get_db, DatabaseManager

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, str]:
    """Basic health check."""
    return {"status": "healthy", "version": "1.0.0", "service": "grace-backend"}


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Readiness check with database connectivity."""
    checks = {}

    # Database check
    try:
        db_healthy = await DatabaseManager.health_check()
        checks["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    # Overall status
    all_healthy = all(status == "healthy" for status in checks.values())

    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready"
        )

    return {"status": "ready", "checks": checks, "version": "1.0.0"}
