"""WebSocket API endpoints."""

from fastapi import APIRouter, WebSocket, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db

router = APIRouter()


@router.websocket("/connect")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection endpoint."""
    await websocket.accept()
    # TODO: Implement WebSocket handling with auth and heartbeat
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception:
        pass
