"""
SQLAlchemy models for Grace system core entities.
Defines the data models for users, sessions, memories, and system operations.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import String, Text, DateTime, Boolean, Integer, LargeBinary, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from grace.core.database import Base

def generate_uuid() -> str:
    """Generate UUID string for primary keys."""
    return str(uuid.uuid4())

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    memories: Mapped[List["Memory"]] = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_users_created_at", "created_at"),
        Index("ix_users_last_login", "last_login"),
    )

class Session(Base):
    """User session model for JWT token management."""
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    token_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    device_info: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 support
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_sessions_user_id_created_at", "user_id", "created_at"),
        Index("ix_sessions_expires_at", "expires_at"),
        Index("ix_sessions_last_accessed", "last_accessed"),
    )

class Memory(Base):
    """Enhanced memory model replacing the old dict-based memory system."""
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)  # Legacy key support
    
    # Content storage
    content: Mapped[Optional[str]] = mapped_column(Text)  # JSON or text content
    content_type: Mapped[str] = mapped_column(String(100), default="application/json")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))  # SHA-256 hash
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Binary content support
    binary_content: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    
    # Metadata and tagging
    memory_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Memory type and classification
    memory_type: Mapped[str] = mapped_column(String(50), default="fusion", index=True)  # lightning, fusion, librarian
    category: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    priority: Mapped[int] = mapped_column(Integer, default=1, index=True)
    
    # Access and lifecycle management
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    ttl_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Compression and optimization
    is_compressed: Mapped[bool] = mapped_column(Boolean, default=False)
    compression_type: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memories")
    embeddings: Mapped[List["MemoryEmbedding"]] = relationship("MemoryEmbedding", back_populates="memory", cascade="all, delete-orphan")
    
    # Indexes for performance  
    __table_args__ = (
        Index("ix_memories_user_id_created_at", "user_id", "created_at"),
        Index("ix_memories_user_id_key", "user_id", "key"),
        Index("ix_memories_expires_at", "expires_at"),
        Index("ix_memories_memory_type", "memory_type"),
        Index("ix_memories_last_accessed", "last_accessed"),
        Index("ix_memories_content_hash", "content_hash"),
    )

class MemoryEmbedding(Base):
    """Vector embeddings for memory content to support semantic search."""
    __tablename__ = "memory_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    memory_id: Mapped[str] = mapped_column(String(36), ForeignKey("memories.id"), nullable=False)
    
    # Embedding data
    embedding: Mapped[List[float]] = mapped_column(JSON, nullable=False)  # Vector data as JSON array
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)  # For multi-chunk content
    chunk_text: Mapped[Optional[str]] = mapped_column(Text)  # Original text that was embedded
    
    # Embedding metadata
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_threshold: Mapped[Optional[float]] = mapped_column()  # Custom threshold for this embedding
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    memory: Mapped["Memory"] = relationship("Memory", back_populates="embeddings")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_memory_embeddings_memory_id", "memory_id"),
        Index("ix_memory_embeddings_model", "embedding_model"),
        Index("ix_memory_embeddings_chunk", "memory_id", "chunk_index"),
    )

class SystemOperation(Base):
    """System operations log for audit and monitoring."""
    __tablename__ = "system_operations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"))
    operation_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    
    # Operation details
    operation_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending, running, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Request context
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")
    
    # Indexes for performance and monitoring
    __table_args__ = (
        Index("ix_operations_user_id_created_at", "user_id", "created_at"),
        Index("ix_operations_type_status", "operation_type", "status"),
        Index("ix_operations_resource", "resource_type", "resource_id"),
        Index("ix_operations_created_at", "created_at"),
    )

class BackgroundTask(Base):
    """Background task queue for async operations."""
    __tablename__ = "background_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    task_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Task data and configuration
    task_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=1, index=True)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Task execution
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(100))
    result_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Indexes for task queue performance
    __table_args__ = (
        Index("ix_tasks_status_priority", "status", "priority"),
        Index("ix_tasks_scheduled_at", "scheduled_at"),
        Index("ix_tasks_task_type", "task_type"),
        Index("ix_tasks_worker_id", "worker_id"),
    )