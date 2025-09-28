"""Authentication API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from typing import Dict, Any

from ..database import get_db
from ..models import User
from ..auth import verify_password, get_password_hash, create_access_token, create_refresh_token
from ..middleware.auth import get_current_user, require_auth
from ..config import get_settings

router = APIRouter()
settings = get_settings()


class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str = ""


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    storage_quota: int
    memory_fragments_quota: int


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserRegistration,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    # Check if user already exists
    result = await db.execute(
        select(User).where(
            (User.email == user_data.email) | (User.username == user_data.username)
        )
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create new user
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        storage_quota=settings.default_storage_quota,
        memory_fragments_quota=settings.default_memory_fragments_quota
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name or "",
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        storage_quota=user.storage_quota,
        memory_fragments_quota=user.memory_fragments_quota
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access tokens."""
    # Find user by username or email
    result = await db.execute(
        select(User).where(
            (User.username == form_data.username) | (User.email == form_data.username)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive"
        )
    
    # Create tokens
    access_token = create_access_token({"sub": user.id, "username": user.username})
    refresh_token = create_refresh_token({"sub": user.id})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(require_auth)):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name or "",
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        storage_quota=current_user.storage_quota,
        memory_fragments_quota=current_user.memory_fragments_quota
    )