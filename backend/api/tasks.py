"""Task management API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..middleware.auth import require_auth
from ..models import User

router = APIRouter()


@router.get("/")
async def list_tasks(
    current_user: User = Depends(require_auth), db: AsyncSession = Depends(get_db)
):
    """List user's tasks."""
    # TODO: Implement task listing
    return {"tasks": []}
