"""
WebSocket connection manager with ping/pong and heartbeat
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect

from grace.auth.models import User

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MESSAGE = "message"
    NOTIFICATION = "notification"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: str
    data: Any
    channel: Optional[str] = None
    timestamp: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class Connection:
    """Represents a single WebSocket connection"""
    
    def __init__(self, websocket: WebSocket, user: User):
        self.websocket = websocket
        self.user = user
        self.channels: Set[str] = set()
        self.connected_at = datetime.now(timezone.utc)
        self.last_pong = datetime.now(timezone.utc)
        self.is_alive = True
    
    async def send(self, message: WebSocketMessage):
        """Send message to this connection"""
        try:
            await self.websocket.send_text(message.to_json())
        except Exception as e:
            logger.error(f"Error sending to connection: {e}")
            self.is_alive = False
    
    async def ping(self):
        """Send ping to check connection"""
        try:
            ping_msg = WebSocketMessage(
                type=MessageType.PING.value,
                data={"timestamp": datetime.now(timezone.utc).isoformat()}
            )
            await self.send(ping_msg)
        except Exception as e:
            logger.error(f"Error sending ping: {e}")
            self.is_alive = False
    
    def handle_pong(self):
        """Handle pong response"""
        self.last_pong = datetime.now(timezone.utc)
        self.is_alive = True


class ConnectionManager:
    """
    Manages WebSocket connections with authentication, ping/pong, and channels
    """
    
    def __init__(self, ping_interval: int = 30, ping_timeout: int = 60):
        """
        Initialize connection manager
        
        Args:
            ping_interval: Seconds between ping messages
            ping_timeout: Seconds before considering connection dead
        """
        self.active_connections: Dict[str, Connection] = {}  # user_id -> Connection
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self._heartbeat_task: Optional[asyncio.Task] = None
        logger.info(f"ConnectionManager initialized (ping: {ping_interval}s, timeout: {ping_timeout}s)")
    
    async def connect(self, websocket: WebSocket, user: User):
        """Accept and register a new connection"""
        await websocket.accept()
        
        connection = Connection(websocket, user)
        self.active_connections[user.id] = connection
        
        logger.info(f"WebSocket connected: user {user.username} ({user.id})")
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            type=MessageType.SYSTEM.value,
            data={
                "message": "Connected successfully",
                "user_id": user.id,
                "username": user.username
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        await connection.send(welcome_msg)
        
        # Start heartbeat if not running
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    def disconnect(self, user_id: str):
        """Remove connection"""
        if user_id in self.active_connections:
            connection = self.active_connections[user_id]
            logger.info(f"WebSocket disconnected: user {connection.user.username} ({user_id})")
            del self.active_connections[user_id]
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage):
        """Send message to specific user"""
        connection = self.active_connections.get(user_id)
        if connection and connection.is_alive:
            await connection.send(message)
    
    async def broadcast(self, message: WebSocketMessage, exclude_user: Optional[str] = None):
        """Broadcast message to all connected users"""
        disconnected = []
        
        for user_id, connection in self.active_connections.items():
            if user_id == exclude_user:
                continue
            
            if connection.is_alive:
                await connection.send(message)
            else:
                disconnected.append(user_id)
        
        # Clean up disconnected
        for user_id in disconnected:
            self.disconnect(user_id)
    
    async def broadcast_to_channel(self, channel: str, message: WebSocketMessage, exclude_user: Optional[str] = None):
        """Broadcast message to users subscribed to a channel"""
        disconnected = []
        
        for user_id, connection in self.active_connections.items():
            if user_id == exclude_user:
                continue
            
            if channel in connection.channels and connection.is_alive:
                await connection.send(message)
            elif not connection.is_alive:
                disconnected.append(user_id)
        
        # Clean up disconnected
        for user_id in disconnected:
            self.disconnect(user_id)
    
    def subscribe(self, user_id: str, channel: str):
        """Subscribe user to a channel"""
        connection = self.active_connections.get(user_id)
        if connection:
            connection.channels.add(channel)
            logger.info(f"User {user_id} subscribed to channel: {channel}")
            return True
        return False
    
    def unsubscribe(self, user_id: str, channel: str):
        """Unsubscribe user from a channel"""
        connection = self.active_connections.get(user_id)
        if connection:
            connection.channels.discard(channel)
            logger.info(f"User {user_id} unsubscribed from channel: {channel}")
            return True
        return False
    
    async def _heartbeat_loop(self):
        """Background task to send pings and check for dead connections"""
        logger.info("Heartbeat loop started")
        
        try:
            while self.active_connections:
                await asyncio.sleep(self.ping_interval)
                
                now = datetime.now(timezone.utc)
                disconnected = []
                
                for user_id, connection in self.active_connections.items():
                    # Check if connection timed out
                    time_since_pong = (now - connection.last_pong).total_seconds()
                    
                    if time_since_pong > self.ping_timeout:
                        logger.warning(f"Connection timeout for user {user_id}")
                        disconnected.append(user_id)
                    else:
                        # Send ping
                        await connection.ping()
                
                # Clean up timed out connections
                for user_id in disconnected:
                    self.disconnect(user_id)
        
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")
        
        finally:
            logger.info("Heartbeat loop stopped")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_channel_subscribers(self, channel: str) -> Set[str]:
        """Get set of user IDs subscribed to a channel"""
        return {
            user_id for user_id, conn in self.active_connections.items()
            if channel in conn.channels
        }
    
    def get_user_channels(self, user_id: str) -> Set[str]:
        """Get channels a user is subscribed to"""
        connection = self.active_connections.get(user_id)
        return connection.channels if connection else set()
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        channels_count = {}
        for connection in self.active_connections.values():
            for channel in connection.channels:
                channels_count[channel] = channels_count.get(channel, 0) + 1
        
        return {
            "total_connections": len(self.active_connections),
            "channels": channels_count,
            "active_users": [
                {
                    "user_id": conn.user.id,
                    "username": conn.user.username,
                    "channels": list(conn.channels),
                    "connected_at": conn.connected_at.isoformat()
                }
                for conn in self.active_connections.values()
            ]
        }
