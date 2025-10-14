"""Collaboration API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..middleware.auth import require_auth
from ..models import User

router = APIRouter()


@router.get("/sessions")
async def list_collab_sessions(
    current_user: User = Depends(require_auth), db: AsyncSession = Depends(get_db)
):
    """List collaboration sessions."""
    # TODO: Implement collab session listing
    return {"sessions": []}
