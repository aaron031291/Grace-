"""
Event streaming routes for Grace Service.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
import structlog

from ..schemas.base import BaseResponse, EventStreamMessage

logger = structlog.get_logger(__name__)

events_router = APIRouter()


def get_websocket_manager():
    """Dependency to get websocket manager."""
    from ..app import app_state
    return app_state.get("websocket_manager")


@events_router.get("/stream/status")
async def get_stream_status(websocket_manager = Depends(get_websocket_manager)):
    """Get the status of the event stream."""
    try:
        connection_count = websocket_manager.get_connection_count() if websocket_manager else 0
        
        return BaseResponse(
            status="success",
            message="Event stream status retrieved",
            data={
                "active_connections": connection_count,
                "stream_enabled": True,
                "supported_events": [
                    "governance.decision",
                    "governance.violation",
                    "ingress.data_received",
                    "ingress.validation_complete",
                    "orchestration.task_started",
                    "orchestration.task_completed",
                    "health.component_status_changed"
                ]
            }
        )
        
    except Exception as e:
        logger.error("Failed to get stream status", error=str(e))
        return BaseResponse(
            status="error",
            message=f"Failed to get stream status: {str(e)}"
        )


@events_router.get("/history")
async def get_recent_events(
    limit: int = 50,
    event_type: str = None
):
    """Get recent events from the system."""
    try:
        # This would query the actual event history
        # For now, return placeholder data
        events = [
            {
                "event_id": "evt_001",
                "event_type": "governance.decision",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {
                    "decision_id": "dec_001",
                    "approved": True,
                    "compliance_score": 0.95
                }
            }
        ]
        
        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        # Limit results
        events = events[:limit]
        
        return BaseResponse(
            status="success",
            message=f"Retrieved {len(events)} recent events",
            data={
                "events": events,
                "total": len(events),
                "filtered_by": event_type
            }
        )
        
    except Exception as e:
        logger.error("Failed to get recent events", error=str(e))
        return BaseResponse(
            status="error",
            message=f"Failed to get recent events: {str(e)}"
        )


@events_router.post("/emit")
async def emit_event(
    event: EventStreamMessage,
    websocket_manager = Depends(get_websocket_manager)
):
    """
    Emit a custom event to all connected WebSocket clients.
    
    This endpoint allows other services to broadcast events
    through the Grace event stream.
    """
    try:
        if websocket_manager:
            await websocket_manager.send_event(
                event_type=event.event_type,
                data=event.data
            )
        
        return BaseResponse(
            status="success",
            message="Event emitted successfully",
            data={
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "broadcast_to": websocket_manager.get_connection_count() if websocket_manager else 0
            }
        )
        
    except Exception as e:
        logger.error("Failed to emit event", error=str(e))
        return BaseResponse(
            status="error",
            message=f"Failed to emit event: {str(e)}"
        )