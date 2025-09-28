"""
JWT-based authentication middleware for Grace API.

Provides JWT token validation and user context extraction.
"""
import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Security scheme for FastAPI
security = HTTPBearer()

# Grace RBAC Scopes
class GraceScopes:
    """Predefined scopes for Grace system."""
    READ_CHAT = "read:chat"
    WRITE_MEMORY = "write:memory"  
    GOVERN_TASKS = "govern:tasks"
    SANDBOX_BUILD = "sandbox:build"
    NETWORK_ACCESS = "network:access"
    ADMIN = "admin"


class JWTManager:
    """JWT token management for Grace authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, user_id: str, scopes: List[str], 
                    roles: List[str] = None, expires_delta: timedelta = None) -> str:
        """
        Create a JWT token for a user.
        
        Args:
            user_id: User identifier
            scopes: List of granted scopes
            roles: User roles
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(hours=24)  # Default 24 hour expiration
        
        expire = datetime.now(timezone.utc) + expires_delta
        
        payload = {
            "sub": user_id,  # Subject (user ID)
            "scopes": scopes,
            "roles": roles or [],
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": "grace-system"  # Issuer
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Created JWT token for user {user_id} with scopes: {scopes}")
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            
            # Validate required fields
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
            scopes = payload.get("scopes", [])
            roles = payload.get("roles", [])
            
            return {
                "user_id": user_id,
                "scopes": scopes,
                "roles": roles,
                "token_data": payload
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    def create_service_token(self, service_name: str, scopes: List[str]) -> str:
        """Create a service token for internal system communication."""
        return self.create_token(
            user_id=f"service:{service_name}",
            scopes=scopes,
            roles=["service"],
            expires_delta=timedelta(days=30)  # Longer expiration for services
        )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.
    
    Returns:
        User authentication data
    """
    settings = get_settings()
    if not settings.jwt_secret_key:
        raise HTTPException(status_code=500, detail="JWT not configured")
    
    jwt_manager = JWTManager(settings.jwt_secret_key)
    
    try:
        auth_data = jwt_manager.verify_token(credentials.credentials)
        return auth_data
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")


def require_scopes(*required_scopes: str):
    """
    Decorator to require specific scopes for an endpoint.
    
    Args:
        *required_scopes: List of required scopes
        
    Returns:
        Decorated function that checks scopes
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: Dict[str, Any] = Depends(get_current_user), **kwargs):
            user_scopes = user.get("scopes", [])
            
            # Check if user has all required scopes
            missing_scopes = [scope for scope in required_scopes if scope not in user_scopes]
            
            if missing_scopes:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Insufficient permissions",
                        "required_scopes": list(required_scopes),
                        "missing_scopes": missing_scopes,
                        "user_scopes": user_scopes
                    }
                )
            
            # Add user to kwargs for the endpoint function
            kwargs["current_user"] = user
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_roles(*required_roles: str):
    """
    Decorator to require specific roles for an endpoint.
    
    Args:
        *required_roles: List of required roles
        
    Returns:
        Decorated function that checks roles
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: Dict[str, Any] = Depends(get_current_user), **kwargs):
            user_roles = user.get("roles", [])
            
            # Check if user has any of the required roles
            has_role = any(role in user_roles for role in required_roles)
            
            if not has_role:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Insufficient permissions",
                        "required_roles": list(required_roles),
                        "user_roles": user_roles
                    }
                )
            
            # Add user to kwargs for the endpoint function
            kwargs["current_user"] = user
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global JWT manager instance
_jwt_manager = None

def get_jwt_manager() -> JWTManager:
    """Get global JWT manager instance."""
    global _jwt_manager
    if _jwt_manager is None:
        settings = get_settings()
        if not settings.jwt_secret_key:
            raise RuntimeError("JWT_SECRET_KEY not configured")
        _jwt_manager = JWTManager(settings.jwt_secret_key)
    return _jwt_manager