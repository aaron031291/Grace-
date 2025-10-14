"""
FastAPI dependencies for authentication and authorization.
"""

from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from grace.core.auth import auth_service, TokenData, rbac
from grace.core.database import get_async_session
from grace.core.container import get_repository_container, RepositoryContainer
from grace.core.models import User, Session as UserSession

# Security scheme
security = HTTPBearer()


async def get_current_user_session(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> tuple[User, UserSession]:
    """
    Get current authenticated user and their active session.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Verify the JWT token
    token_data = auth_service.verify_token(credentials.credentials)

    if not token_data.user_id or not token_data.token_id:
        raise credentials_exception

    # Check if session is still active
    user_session = await repos.sessions.get_active_session(token_data.token_id)
    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
        )

    # Get the user
    user = await repos.users.get_by_id(token_data.user_id)
    if not user:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Update session access time
    from datetime import datetime

    await repos.sessions.update(user_session.id, {"last_accessed": datetime.utcnow()})
    await session.commit()

    return user, user_session


async def get_current_user(
    user_session: Annotated[
        tuple[User, UserSession], Depends(get_current_user_session)
    ],
) -> User:
    """Get current authenticated user."""
    user, _ = user_session
    return user


async def get_current_active_user(
    user_session: Annotated[
        tuple[User, UserSession], Depends(get_current_user_session)
    ],
) -> User:
    """Get current active authenticated user."""
    user, _ = user_session

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    return user


def require_permissions(*permissions: str):
    """
    Dependency factory to require specific permissions.
    Usage: deps=[Depends(require_permissions("users:read", "memories:create"))]
    """

    async def permission_checker(
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    ) -> TokenData:
        # Verify token and get user scopes
        token_data = auth_service.verify_token(credentials.credentials)

        # Check permissions
        user_scopes = token_data.scopes or []

        for permission in permissions:
            if not rbac.has_permission(user_scopes, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission}",
                )

        return token_data

    return permission_checker


def require_scopes(*scopes: str):
    """
    Dependency factory to require specific scopes/roles.
    Usage: deps=[Depends(require_scopes("admin", "superuser"))]
    """

    async def scope_checker(
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    ) -> TokenData:
        # Verify token and get user scopes
        token_data = auth_service.verify_token(credentials.credentials)

        # Check if user has any of the required scopes
        user_scopes = set(token_data.scopes or [])
        required_scopes = set(scopes)

        if not (user_scopes & required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges. Required scopes: {list(required_scopes)}",
            )

        return token_data

    return scope_checker


# Common dependency combinations
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[TokenData, Depends(require_scopes("admin", "superuser"))]
SuperUser = Annotated[TokenData, Depends(require_scopes("superuser"))]
