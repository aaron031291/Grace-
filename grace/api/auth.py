"""
Authentication API endpoints for Grace system.
"""
import secrets
from datetime import datetime, timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr

from grace.core.auth import auth_service, UserCreate, UserResponse, Token, TokenData
from grace.core.database import get_async_session
from grace.core.container import get_repository_container, RepositoryContainer
from grace.core.models import User, Session as UserSession
from grace.core.dependencies import get_current_user, CurrentUser, SuperUser

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str
    device_info: str = None
    remember_me: bool = False

class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)]
):
    """Register a new user."""
    
    # Check if user already exists
    existing_user = await repos.users.get_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    existing_email = await repos.users.get_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    hashed_password = auth_service.get_password_hash(user_data.password)
    
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_superuser=False  # Regular users are not superusers by default
    )
    
    created_user = await repos.users.create(user)
    await session.commit()
    
    return UserResponse(
        id=created_user.id,
        username=created_user.username,
        email=created_user.email,
        full_name=created_user.full_name,
        is_active=created_user.is_active,
        is_superuser=created_user.is_superuser,
        created_at=created_user.created_at,
        last_login=created_user.last_login
    )

@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)]
):
    """Authenticate user and return JWT tokens."""
    
    # Find user by username
    user = await repos.users.get_by_username(login_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not auth_service.verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Determine user scopes/roles
    scopes = ["user"]
    if user.is_superuser:
        scopes.append("superuser")
    
    # Create tokens
    token_data = auth_service.create_user_tokens(
        user_id=user.id,
        username=user.username,
        scopes=scopes
    )
    
    # Create session record
    expires_at = datetime.utcnow() + timedelta(days=30 if login_data.remember_me else 1)
    
    user_session = UserSession(
        user_id=user.id,
        token_id=token_data["token_id"],
        device_info=login_data.device_info,
        expires_at=expires_at,
        is_active=True
    )
    
    await repos.sessions.create(user_session)
    
    # Update user's last login
    await repos.users.update_last_login(user.id)
    
    await session.commit()
    
    return Token(
        access_token=token_data["access_token"],
        refresh_token=token_data["refresh_token"],
        token_type="bearer",
        expires_in=token_data["expires_in"]
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)]
):
    """Refresh access token using refresh token."""
    
    try:
        # Verify refresh token
        token_data = auth_service.verify_token(refresh_request.refresh_token, token_type="refresh")
        
        if not token_data.user_id or not token_data.token_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Check if session is still active
        user_session = await repos.sessions.get_active_session(token_data.token_id)
        if not user_session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )
        
        # Get user
        user = await repos.users.get_by_id(token_data.user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Determine user scopes
        scopes = ["user"]
        if user.is_superuser:
            scopes.append("superuser")
        
        # Create new tokens with same token_id
        new_token_data = auth_service.create_user_tokens(
            user_id=user.id,
            username=user.username,
            scopes=scopes
        )
        
        # Update session token_id for the new tokens
        await repos.sessions.update(user_session.id, {
            "token_id": new_token_data["token_id"]
        })
        
        await session.commit()
        
        return Token(
            access_token=new_token_data["access_token"],
            refresh_token=new_token_data["refresh_token"],
            token_type="bearer",
            expires_in=new_token_data["expires_in"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/logout")
async def logout(
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)]
):
    """Logout current user by deactivating session."""
    
    # This would need to be modified to get the current token_id from the request
    # For now, we'll deactivate all sessions for the user
    user_sessions = await repos.sessions.get_user_sessions(current_user.id, active_only=True)
    
    for user_session in user_sessions:
        await repos.sessions.deactivate_session(user_session.token_id)
    
    await session.commit()
    
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: CurrentUser):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )