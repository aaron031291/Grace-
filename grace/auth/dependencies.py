"""
FastAPI dependencies for authentication
"""

from typing import Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from grace.database import get_db
from grace.auth.models import User
from grace.auth.security import verify_token
from grace.config import get_settings

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    settings = get_settings()
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    payload = verify_token(token, settings.auth.secret_key, settings.auth.algorithm, "access")
    
    if payload is None:
        raise credentials_exception
    
    user_id = payload.get("user_id")
    if user_id is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception
    
    return user


def require_role(allowed_roles: list[str]) -> Callable:
    """Dependency to require specific roles"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_roles = [role.name for role in current_user.roles]
        if not any(role in allowed_roles for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker
