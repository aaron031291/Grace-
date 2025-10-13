"""
WebSocket authentication for Grace system.

Validates JWT tokens for WebSocket connections.
"""

import logging
from typing import Dict, Any, Optional
from urllib.parse import parse_qs

from fastapi import WebSocket, WebSocketException, status
from starlette.websockets import WebSocketState

from .jwt_auth import JWTManager, get_jwt_manager

logger = logging.getLogger(__name__)


class WebSocketAuth:
    """WebSocket authentication handler."""

    def __init__(self, jwt_manager: Optional[JWTManager] = None):
        self.jwt_manager = jwt_manager or get_jwt_manager()

    async def authenticate_websocket(self, websocket: WebSocket) -> Dict[str, Any]:
        """
        Authenticate a WebSocket connection using JWT token.

        Token can be provided via:
        1. Query parameter: ?token=<jwt_token>
        2. Authorization header (if supported by client)

        Args:
            websocket: WebSocket connection

        Returns:
            User authentication data

        Raises:
            WebSocketException: If authentication fails
        """
        # Try to get token from query parameters first
        token = self._extract_token_from_query(websocket)

        # If not in query, try headers (if available)
        if not token:
            token = self._extract_token_from_headers(websocket)

        if not token:
            logger.warning("WebSocket connection missing authentication token")
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Missing authentication token",
            )

        try:
            # Verify the token
            auth_data = self.jwt_manager.verify_token(token)

            logger.info(f"WebSocket authenticated for user: {auth_data['user_id']}")
            return auth_data

        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=f"Authentication failed: {str(e)}",
            )

    def _extract_token_from_query(self, websocket: WebSocket) -> Optional[str]:
        """Extract JWT token from query parameters."""
        query_params = parse_qs(websocket.url.query)
        token_list = query_params.get("token", [])

        if token_list:
            return token_list[0]

        return None

    def _extract_token_from_headers(self, websocket: WebSocket) -> Optional[str]:
        """Extract JWT token from WebSocket headers."""
        # WebSocket authorization header support varies by client
        auth_header = websocket.headers.get("authorization")

        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        return None

    async def require_scopes(self, auth_data: Dict[str, Any], *required_scopes: str):
        """
        Check if authenticated user has required scopes.

        Args:
            auth_data: Authentication data from authenticate_websocket
            *required_scopes: Required scopes

        Raises:
            WebSocketException: If user lacks required scopes
        """
        user_scopes = auth_data.get("scopes", [])
        missing_scopes = [
            scope for scope in required_scopes if scope not in user_scopes
        ]

        if missing_scopes:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=f"Insufficient permissions. Missing scopes: {missing_scopes}",
            )

    async def require_roles(self, auth_data: Dict[str, Any], *required_roles: str):
        """
        Check if authenticated user has required roles.

        Args:
            auth_data: Authentication data from authenticate_websocket
            *required_roles: Required roles

        Raises:
            WebSocketException: If user lacks required roles
        """
        user_roles = auth_data.get("roles", [])
        has_role = any(role in user_roles for role in required_roles)

        if not has_role:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=f"Insufficient permissions. Required roles: {list(required_roles)}",
            )


class AuthenticatedWebSocket:
    """
    Wrapper for WebSocket with authentication context.

    Usage:
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            auth_ws = AuthenticatedWebSocket(websocket)
            await auth_ws.authenticate()

            # Now you can use auth_ws with authentication context
            user_id = auth_ws.user_id
            await auth_ws.send_text(f"Hello {user_id}")
    """

    def __init__(self, websocket: WebSocket, ws_auth: Optional[WebSocketAuth] = None):
        self.websocket = websocket
        self.ws_auth = ws_auth or WebSocketAuth()
        self.auth_data: Optional[Dict[str, Any]] = None
        self._authenticated = False

    async def authenticate(
        self, required_scopes: list = None, required_roles: list = None
    ):
        """
        Authenticate the WebSocket connection.

        Args:
            required_scopes: Optional list of required scopes
            required_roles: Optional list of required roles
        """
        # Accept the connection first
        if self.websocket.client_state == WebSocketState.CONNECTING:
            await self.websocket.accept()

        # Perform authentication
        self.auth_data = await self.ws_auth.authenticate_websocket(self.websocket)

        # Check scopes if required
        if required_scopes:
            await self.ws_auth.require_scopes(self.auth_data, *required_scopes)

        # Check roles if required
        if required_roles:
            await self.ws_auth.require_roles(self.auth_data, *required_roles)

        self._authenticated = True

        logger.info(
            f"WebSocket authenticated: user={self.user_id}, scopes={self.scopes}"
        )

    @property
    def user_id(self) -> str:
        """Get authenticated user ID."""
        if not self._authenticated or not self.auth_data:
            raise RuntimeError("WebSocket not authenticated")
        return self.auth_data["user_id"]

    @property
    def scopes(self) -> list:
        """Get user scopes."""
        if not self._authenticated or not self.auth_data:
            raise RuntimeError("WebSocket not authenticated")
        return self.auth_data["scopes"]

    @property
    def roles(self) -> list:
        """Get user roles."""
        if not self._authenticated or not self.auth_data:
            raise RuntimeError("WebSocket not authenticated")
        return self.auth_data["roles"]

    async def send_text(self, data: str):
        """Send text data through WebSocket."""
        await self.websocket.send_text(data)

    async def send_json(self, data: dict):
        """Send JSON data through WebSocket."""
        await self.websocket.send_json(data)

    async def receive_text(self) -> str:
        """Receive text data from WebSocket."""
        return await self.websocket.receive_text()

    async def receive_json(self) -> dict:
        """Receive JSON data from WebSocket."""
        return await self.websocket.receive_json()

    async def close(
        self, code: int = status.WS_1000_NORMAL_CLOSURE, reason: str = None
    ):
        """Close the WebSocket connection."""
        await self.websocket.close(code, reason)


# Convenience function for WebSocket authentication
async def authenticate_websocket(
    websocket: WebSocket, required_scopes: list = None, required_roles: list = None
) -> AuthenticatedWebSocket:
    """
    Convenience function to authenticate a WebSocket connection.

    Args:
        websocket: WebSocket connection
        required_scopes: Optional required scopes
        required_roles: Optional required roles

    Returns:
        Authenticated WebSocket wrapper
    """
    auth_ws = AuthenticatedWebSocket(websocket)
    await auth_ws.authenticate(required_scopes, required_roles)
    return auth_ws


# Global WebSocket auth instance
_ws_auth = None


def get_websocket_auth() -> WebSocketAuth:
    """Get global WebSocket auth instance."""
    global _ws_auth
    if _ws_auth is None:
        _ws_auth = WebSocketAuth()
    return _ws_auth
