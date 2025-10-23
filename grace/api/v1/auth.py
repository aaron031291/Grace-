"""
Authentication API endpoints
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from grace.auth.models import User, Role, RefreshToken
from grace.auth.security import (
    get_password_hash,
    verify_password,
    create_token_pair,
    verify_token
)
from grace.auth.dependencies import get_current_user, require_role
from grace.database import get_db
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    is_superuser: bool
    roles: list[str] = []
    created_at: datetime
    last_login: datetime | None
    
    class Config:
        from_attributes = True


@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """OAuth2 compatible token login - Real implementation with database"""
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is locked until {user.locked_until.isoformat()}"
        )
    
    if not verify_password(form_data.password, user.hashed_password):
        user.failed_login_attempts += 1
        
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            logger.warning(f"Account locked for user {user.username}")
        
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Reset failed login attempts
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.now(timezone.utc)
    
    # Create token with role claims
    token_data = {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "roles": user.role_names,
        "is_superuser": user.is_superuser
    }
    
    tokens = create_token_pair(token_data)
    
    # Store refresh token
    refresh_token_record = RefreshToken(
        id=str(uuid.uuid4()),
        token=tokens["refresh_token"],
        user_id=user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None
    )
    db.add(refresh_token_record)
    db.commit()
    
    logger.info(f"User {user.username} logged in successfully")
    
    return tokens


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        is_superuser=current_user.is_superuser,
        roles=current_user.role_names,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )
