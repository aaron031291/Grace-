"""
WebSocket connection manager for Grace Service.
"""
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time event streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
            
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients."""
        if not self.active_connections:
            return
            
        message_text = json.dumps(message)
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Failed to broadcast to WebSocket: {e}")
                disconnected_connections.append(connection)
                
        # Remove failed connections
        for connection in disconnected_connections:
            self.disconnect(connection)
            
    async def send_event(self, event_type: str, data: Dict[str, Any]):
        """Send a structured event to all connected clients."""
        event_message = {
            "type": "event",
            "event_type": event_type,
            "timestamp": data.get("timestamp"),
            "data": data
        }
        await self.broadcast(event_message)
        
    def get_connection_count(self) -> int:
        """Get the number of active WebSocket connections."""
        return len(self.active_connections)