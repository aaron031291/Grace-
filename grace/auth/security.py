"""
Security utilities for authentication
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], secret_key: str, algorithm: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    from grace.config import get_settings
    settings = get_settings()
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.auth.access_token_expire_minutes)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any], secret_key: str, algorithm: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT refresh token"""
    from grace.config import get_settings
    settings = get_settings()
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=settings.auth.refresh_token_expire_days)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def verify_token(token: str, secret_key: str, algorithm: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        if payload.get("type") != token_type:
            return None
        return payload
    except JWTError:
        return None


def create_token_pair(user_id: int, username: str) -> Dict[str, str]:
    """Create access and refresh token pair"""
    from grace.config import get_settings
    settings = get_settings()
    
    data = {"user_id": user_id, "username": username}
    access_token = create_access_token(data, settings.auth.secret_key, settings.auth.algorithm)
    refresh_token = create_refresh_token(data, settings.auth.secret_key, settings.auth.algorithm)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
