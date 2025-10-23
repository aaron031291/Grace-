"""
Document models for database
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from grace.auth.models import Base


class Document(Base):
    """Document model with embeddings"""
    __tablename__ = 'documents'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey('users.id'), nullable=False)
    
    # Document content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), default="text/plain")
    
    # Metadata
    source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # List of tags as JSON
    metadata_json: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Embedding info
    vector_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)  # ID in vector store
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Status
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Stats
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    last_accessed: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title[:50]}, user_id={self.user_id})>"
