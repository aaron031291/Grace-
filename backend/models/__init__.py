"""Database models for Grace Backend."""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


def generate_uuid():
    """Generate UUID string."""
    return str(uuid.uuid4())


def utc_now():
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    
    # User quotas
    storage_quota = Column(BigInteger, nullable=False)  # bytes
    memory_fragments_quota = Column(Integer, nullable=False)
    
    # Profile
    full_name = Column(String(255))
    avatar_url = Column(String(500))
    
    # Relationships
    sessions = relationship("Session", back_populates="owner", cascade="all, delete-orphan")
    memory_fragments = relationship("MemoryFragment", back_populates="user", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="assignee")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class Session(Base):
    """User session model."""
    __tablename__ = "sessions"
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    expires_at = Column(DateTime(timezone=True))
    
    # Session configuration
    max_panels = Column(Integer, nullable=False, default=20)
    settings = Column(JSON, default=dict)
    
    # Relationships
    owner = relationship("User", back_populates="sessions")
    panels = relationship("Panel", back_populates="session", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    collab_sessions = relationship("CollaborationSession", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_sessions_owner_active", "owner_id", "is_active"),
    )


class Panel(Base):
    """Panel model for session UI panels."""
    __tablename__ = "panels"
    
    id = Column(UUID, primary_key=True, default=generate_uuid)
    session_id = Column(UUID, ForeignKey("sessions.id"), nullable=False)
    panel_type = Column(String(50), nullable=False)  # chat, memory, knowledge, task, etc.
    title = Column(String(255), nullable=False)
    position_x = Column(Integer, default=0)
    position_y = Column(Integer, default=0)
    width = Column(Integer, default=400)
    height = Column(Integer, default=300)
    z_index = Column(Integer, default=0)
    is_minimized = Column(Boolean, default=False)
    is_maximized = Column(Boolean, default=False)
    config = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="panels")
    
    __table_args__ = (
        Index("ix_panels_session_type", "session_id", "panel_type"),
        CheckConstraint("width >= 200 AND height >= 150", name="check_panel_min_size"),
        CheckConstraint("z_index >= 0", name="check_panel_z_index"),
    )


# Add rest of the models from previous message - they're long, so I'll continue in the next part