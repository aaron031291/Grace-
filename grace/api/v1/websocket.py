"""
WebSocket API endpoints
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from grace.websocket.manager import ConnectionManager, WebSocketMessage, MessageType
from grace.websocket.auth import get_current_user_ws
from grace.websocket.channels import ChannelManager
from grace.auth.models import User
from grace.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])

# Global connection and channel managers
connection_manager = ConnectionManager(ping_interval=30, ping_timeout=60)
channel_manager = ChannelManager()


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    Main WebSocket endpoint with JWT authentication
    
    Connect with: ws://localhost:8000/api/v1/ws/connect?token=<jwt_token>
    
    Message format:
    {
        "type": "message|subscribe|unsubscribe|ping|pong",
        "channel": "channel_name",
        "data": {...}
    }
    """
    
    try:
        # Authenticate user
        user = await get_current_user_ws(websocket, token)
        
        # Connect to manager
        await connection_manager.connect(websocket, user)
        
        # Auto-subscribe to user's private notification channel
        user_channel = channel_manager.get_user_channel(user.id)
        connection_manager.subscribe(user.id, user_channel)
        
        # Auto-subscribe to system channel
        connection_manager.subscribe(user.id, "system")
        
        # Message handling loop
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                
                try:
                    message_data = json.loads(data)
                    message_type = message_data.get("type")
                    
                    # Handle different message types
                    if message_type == MessageType.PONG.value:
                        # Handle pong response
                        connection = connection_manager.active_connections.get(user.id)
                        if connection:
                            connection.handle_pong()
                    
                    elif message_type == MessageType.SUBSCRIBE.value:
                        # Subscribe to channel
                        channel = message_data.get("channel")
                        if channel and channel_manager.can_access(user.id, channel):
                            connection_manager.subscribe(user.id, channel)
                            
                            response = WebSocketMessage(
                                type=MessageType.SYSTEM.value,
                                data={"message": f"Subscribed to {channel}"},
                                channel=channel
                            )
                            await connection_manager.send_to_user(user.id, response)
                        else:
                            error_msg = WebSocketMessage(
                                type=MessageType.ERROR.value,
                                data={"message": "Access denied or invalid channel"}
                            )
                            await connection_manager.send_to_user(user.id, error_msg)
                    
                    elif message_type == MessageType.UNSUBSCRIBE.value:
                        # Unsubscribe from channel
                        channel = message_data.get("channel")
                        if channel:
                            connection_manager.unsubscribe(user.id, channel)
                            
                            response = WebSocketMessage(
                                type=MessageType.SYSTEM.value,
                                data={"message": f"Unsubscribed from {channel}"}
                            )
                            await connection_manager.send_to_user(user.id, response)
                    
                    elif message_type == MessageType.MESSAGE.value:
                        # Publish message to channel
                        channel = message_data.get("channel")
                        content = message_data.get("data")
                        
                        if channel and content:
                            # Check if user can access channel
                            if channel_manager.can_access(user.id, channel):
                                # Broadcast to channel
                                message = WebSocketMessage(
                                    type=MessageType.MESSAGE.value,
                                    data=content,
                                    channel=channel,
                                    user_id=user.id
                                )
                                await connection_manager.broadcast_to_channel(
                                    channel, 
                                    message,
                                    exclude_user=None  # Include sender
                                )
                            else:
                                error_msg = WebSocketMessage(
                                    type=MessageType.ERROR.value,
                                    data={"message": "Access denied to channel"}
                                )
                                await connection_manager.send_to_user(user.id, error_msg)
                    
                    elif message_type == MessageType.PING.value:
                        # Respond to ping
                        pong_msg = WebSocketMessage(
                            type=MessageType.PONG.value,
                            data={"timestamp": message_data.get("data", {}).get("timestamp")}
                        )
                        await connection_manager.send_to_user(user.id, pong_msg)
                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    error_msg = WebSocketMessage(
                        type=MessageType.ERROR.value,
                        data={"message": "Invalid JSON format"}
                    )
                    await connection_manager.send_to_user(user.id, error_msg)
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: user {user.username}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        if 'user' in locals():
            connection_manager.disconnect(user.id)


@router.get("/stats")
async def get_websocket_stats(current_user: User = Depends(get_current_user_ws)):
    """Get WebSocket connection statistics (requires authentication)"""
    return connection_manager.get_stats()


# Helper function to publish messages from REST endpoints
async def publish_to_channel(channel: str, message_type: str, data: dict):
    """
    Publish a message to a WebSocket channel from REST endpoints
    
    Usage:
        await publish_to_channel("collaboration:session-123", "update", {"status": "active"})
    """
    message = WebSocketMessage(
        type=message_type,
        data=data,
        channel=channel
    )
    await connection_manager.broadcast_to_channel(channel, message)


async def notify_user(user_id: str, notification_type: str, data: dict):
    """
    Send notification to a specific user
    
    Usage:
        await notify_user("user-123", "task_assigned", {"task_id": "task-456"})
    """
    user_channel = channel_manager.get_user_channel(user_id)
    message = WebSocketMessage(
        type=MessageType.NOTIFICATION.value,
        data={
            "notification_type": notification_type,
            **data
        },
        channel=user_channel
    )
    await connection_manager.send_to_user(user_id, message)
