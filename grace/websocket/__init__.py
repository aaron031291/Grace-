"""
WebSocket support for real-time communication
"""

from .manager import ConnectionManager, WebSocketMessage
from .auth import get_current_user_ws
from .channels import ChannelManager

__all__ = [
    'ConnectionManager',
    'WebSocketMessage',
    'get_current_user_ws',
    'ChannelManager'
]
