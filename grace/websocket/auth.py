"""
WebSocket authentication using JWT
"""

from typing import Optional
from fastapi import WebSocket, WebSocketException, status
from sqlalchemy.orm import Session
import logging

from grace.auth.models import User
from grace.auth.security import verify_token
from grace.database import SessionLocal

logger = logging.getLogger(__name__)


async def get_current_user_ws(websocket: WebSocket, token: Optional[str] = None) -> User:
    """
    Authenticate WebSocket connection using JWT token
    
    Token can be provided via:
    1. Query parameter: ?token=<jwt>
    2. Cookie: Authorization
    3. First message after connection
    """
    
    # Try to get token from query params
    if not token:
        token = websocket.query_params.get("token")
    
    # Try to get token from cookies
    if not token:
        auth_cookie = websocket.cookies.get("Authorization")
        if auth_cookie and auth_cookie.startswith("Bearer "):
            token = auth_cookie.split(" ")[1]
    
    if not token:
        logger.warning("WebSocket connection attempt without token")
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Missing authentication token"
        )
    
    # Verify token
    payload = verify_token(token, token_type="access")
    if not payload:
        logger.warning("WebSocket connection with invalid token")
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid or expired token"
        )
    
    user_id = payload.get("user_id")
    if not user_id:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid token payload"
        )
    
    # Get user from database
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="User not found"
            )
        
        if not user.is_active:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="User account is inactive"
            )
        
        if user.is_locked:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="User account is locked"
            )
        
        logger.info(f"WebSocket authenticated: user {user.username} ({user.id})")
        return user
        
    finally:
        db.close()
