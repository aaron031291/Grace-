"""
Authentication and authorization system for Grace.
Provides JWT-based authentication with role-based access control.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """Token payload data."""

    user_id: Optional[str] = None
    username: Optional[str] = None
    scopes: list[str] = []
    token_id: Optional[str] = None


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class UserCreate(BaseModel):
    """User creation schema."""

    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response schema."""

    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class AuthService:
    """Authentication service."""

    def __init__(self):
        self.pwd_context = pwd_context

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)

    def create_access_token(
        self,
        user_id: str,
        username: str,
        token_id: str,
        scopes: Optional[list[str]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token."""
        if scopes is None:
            scopes = ["user"]

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode = {
            "sub": user_id,
            "username": username,
            "token_id": token_id,
            "scopes": scopes,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def create_refresh_token(
        self, user_id: str, token_id: str, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode = {
            "sub": user_id,
            "token_id": token_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Verify token type
            if payload.get("type") != token_type:
                raise credentials_exception

            user_id: str = payload.get("sub")
            if user_id is None:
                raise credentials_exception

            token_data = TokenData(
                user_id=user_id,
                username=payload.get("username"),
                scopes=payload.get("scopes", []),
                token_id=payload.get("token_id"),
            )

            return token_data

        except JWTError:
            raise credentials_exception

    def create_user_tokens(
        self, user_id: str, username: str, scopes: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """Create both access and refresh tokens for a user."""
        token_id = secrets.token_urlsafe(32)

        access_token = self.create_access_token(
            user_id=user_id, username=username, token_id=token_id, scopes=scopes
        )

        refresh_token = self.create_refresh_token(user_id=user_id, token_id=token_id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "token_id": token_id,
        }


class RoleBasedAccessControl:
    """Role-based access control system."""

    # Define role hierarchies and permissions
    ROLES = {"superuser": ["admin", "user"], "admin": ["user"], "user": []}

    PERMISSIONS = {
        # User management
        "users:create": ["superuser"],
        "users:read": ["admin", "user"],  # Users can read their own data
        "users:update": ["admin", "user"],  # Users can update their own data
        "users:delete": ["superuser"],
        "users:list": ["admin"],
        # Memory management
        "memories:create": ["user"],
        "memories:read": ["user"],
        "memories:update": ["user"],
        "memories:delete": ["user"],
        "memories:search": ["user"],
        # System operations
        "system:monitor": ["admin"],
        "system:configure": ["superuser"],
        "system:logs": ["admin"],
        # Background tasks
        "tasks:view": ["admin"],
        "tasks:manage": ["admin"],
    }

    @classmethod
    def get_user_permissions(cls, scopes: list[str]) -> list[str]:
        """Get all permissions for user scopes."""
        permissions = set()

        for scope in scopes:
            if scope == "superuser":
                # Superuser gets all permissions
                for perm_list in cls.PERMISSIONS.values():
                    permissions.update(perm_list)
                break
            else:
                # Add permissions for this scope
                for permission, allowed_scopes in cls.PERMISSIONS.items():
                    if scope in allowed_scopes:
                        permissions.add(permission)

        return list(permissions)

    @classmethod
    def has_permission(cls, user_scopes: list[str], required_permission: str) -> bool:
        """Check if user has required permission."""
        if not user_scopes:
            return False

        # Check if any user scope is allowed for this permission
        allowed_scopes = cls.PERMISSIONS.get(required_permission, [])
        return bool(set(user_scopes) & set(allowed_scopes))

    @classmethod
    def require_permission(cls, permission: str):
        """Decorator to require specific permission."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be implemented in the FastAPI dependency system
                return func(*args, **kwargs)

            return wrapper

        return decorator


# Global auth service instance
auth_service = AuthService()
rbac = RoleBasedAccessControl()
