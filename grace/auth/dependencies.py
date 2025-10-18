"""
FastAPI dependencies for authentication and authorization
"""

from typing import List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError
import logging

from grace.auth.models import User
from grace.auth.security import verify_token
from grace.database import get_db

logger = logging.getLogger(__name__)

# OAuth2 scheme for Bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        token: JWT access token
        db: Database session
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify and decode token
        payload = verify_token(token, token_type="access")
        if payload is None:
            raise credentials_exception
        
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
    
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise credentials_exception
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Check if user is locked
    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is locked"
        )
    
    return user


def require_role(required_roles: List[str]):
    """
    Dependency factory for role-based authorization
    
    Args:
        required_roles: List of role names that are allowed
        
    Returns:
        Dependency function that checks user roles
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.is_superuser:
            return current_user  # Superusers bypass role checks
        
        if not current_user.has_any_role(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have required role. Required: {required_roles}"
            )
        
        return current_user
    
    return role_checker


__all__ = [
    'get_current_user',
    'require_role',
]
