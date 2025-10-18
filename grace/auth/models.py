"""
Database models for authentication
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from typing import List

from grace.database import Base

# Association table for many-to-many relationship between users and roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String(36), ForeignKey('users.id'), primary_key=True),
    Column('role_id', String(36), ForeignKey('roles.id'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
    Column('assigned_by', String(36), nullable=True)
)


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships with proper type hints
    roles: List["Role"] = relationship("Role", secondary="user_roles", back_populates="users")
    refresh_tokens: List["RefreshToken"] = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
    
    @property
    def role_names(self) -> List[str]:
        """Get list of role names"""
        return [role.name for role in self.roles]
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role"""
        return role_name in self.role_names
    
    def has_any_role(self, role_names: List[str]) -> bool:
        """Check if user has any of the specified roles"""
        return any(role_name in self.role_names for role_name in role_names)
    
    def has_all_roles(self, role_names: List[str]) -> bool:
        """Check if user has all of the specified roles"""
        return all(role_name in self.role_names for role_name in role_names)
    
    @property
    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until


class Role(Base):
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    
    # Relationships with proper type hints
    users: List["User"] = relationship("User", secondary="user_roles", back_populates="roles")
    
    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name})>"


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships with proper type hints
    user: "User" = relationship("User", back_populates="refresh_tokens")
    
    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"
    
    @property
    def is_valid(self) -> bool:
        """Check if refresh token is still valid"""
        if self.revoked:
            return False
        return datetime.now(timezone.utc) < self.expires_at


# Alias for backward compatibility
UserRole = user_roles

# Create all tables function
def create_tables(engine):
    """Create all authentication tables"""
    Base.metadata.create_all(engine)
