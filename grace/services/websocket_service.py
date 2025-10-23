"""
Grace AI WebSocket Service - Real-time bidirectional communication
"""
import logging
from typing import Dict, Any, Set, Callable, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """Represents a WebSocket connection."""
    
    def __init__(self, connection_id: str, send_callback: Callable):
        self.connection_id = connection_id
        self.send_callback = send_callback
        self.created_at = datetime.now().isoformat()
        self.subscriptions: Set[str] = set()
    
    async def send(self, event_type: str, data: Dict[str, Any]):
        """Send a message to this connection."""
        message = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_callback(json.dumps(message))
    
    def subscribe(self, topic: str):
        """Subscribe to a topic."""
        self.subscriptions.add(topic)
        logger.info(f"Connection {self.connection_id} subscribed to {topic}")
    
    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic."""
        self.subscriptions.discard(topic)
        logger.info(f"Connection {self.connection_id} unsubscribed from {topic}")

class WebSocketService:
    """Manages WebSocket connections and real-time communication."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.message_history: List[Dict[str, Any]] = []
    
    def register_connection(self, connection_id: str, send_callback: Callable) -> WebSocketConnection:
        """Register a new WebSocket connection."""
        connection = WebSocketConnection(connection_id, send_callback)
        self.connections[connection_id] = connection
        logger.info(f"WebSocket connection registered: {connection_id}")
        return connection
    
    def unregister_connection(self, connection_id: str):
        """Unregister a WebSocket connection."""
        if connection_id in self.connections:
            del self.connections[connection_id]
            logger.info(f"WebSocket connection unregistered: {connection_id}")
    
    async def broadcast_to_topic(self, topic: str, event_type: str, data: Dict[str, Any]):
        """Broadcast a message to all connections subscribed to a topic."""
        for connection in self.connections.values():
            if topic in connection.subscriptions:
                await connection.send(event_type, data)
        
        self.message_history.append({
            "topic": topic,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_to_connection(self, connection_id: str, event_type: str, data: Dict[str, Any]):
        """Send a message to a specific connection."""
        connection = self.connections.get(connection_id)
        if connection:
            await connection.send(event_type, data)
    
    def get_active_connections(self) -> int:
        """Get the number of active connections."""
        return len(self.connections)
    
    def get_message_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history."""
        return self.message_history[-limit:]
