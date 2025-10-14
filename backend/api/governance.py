"""Governance API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..middleware.auth import require_auth
from ..models import User

router = APIRouter()


@router.get("/policies")
async def list_policies(
    current_user: User = Depends(require_auth), db: AsyncSession = Depends(get_db)
):
    """List governance policies."""
    # TODO: Implement policy listing
    return {"policies": []}
