"""
SQLAlchemy Models for Grace System

Maps existing Pydantic dataclasses to SQLAlchemy models for persistent storage.
"""

import uuid

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Text,
    JSON,
    Float,
    Integer,
    Boolean,
    ForeignKey,
    Table,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

Base = declarative_base()

# Association tables for many-to-many relationships
session_users_table = Table(
    "session_users",
    Base.metadata,
    Column(
        "session_id", UUID(as_uuid=True), ForeignKey("sessions.id"), primary_key=True
    ),
    Column("user_id", String, primary_key=True),
)

knowledge_entry_tags_table = Table(
    "knowledge_entry_tags",
    Base.metadata,
    Column(
        "knowledge_entry_id",
        UUID(as_uuid=True),
        ForeignKey("knowledge_entries.id"),
        primary_key=True,
    ),
    Column("tag", String, primary_key=True),
)


class TimestampMixin:
    """Mixin for timestamp fields."""

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Session(Base, TimestampMixin):
    """User conversation sessions."""

    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    context = Column(JSON, nullable=True)
    status = Column(String, default="active", nullable=False)
    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    # Relationships
    messages = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    panels = relationship(
        "Panel", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_sessions_user_status", "user_id", "status"),
        Index("ix_sessions_created_at", "created_at"),
    )


class Panel(Base, TimestampMixin):
    """UI panels within sessions."""

    __tablename__ = "panels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    panel_type = Column(String, nullable=False)
    title = Column(String, nullable=True)
    content = Column(JSON, nullable=True)
    position = Column(Integer, default=0)
    is_visible = Column(Boolean, default=True)
    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    # Relationships
    session = relationship("Session", back_populates="panels")


class Message(Base, TimestampMixin):
    """Messages in conversations."""

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    content_type = Column(String, default="text/plain")
    message_index = Column(Integer, nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    # W5H indexing
    w5h_who = Column(ARRAY(String), nullable=True)
    w5h_what = Column(ARRAY(String), nullable=True)
    w5h_when = Column(ARRAY(String), nullable=True)
    w5h_where = Column(ARRAY(String), nullable=True)
    w5h_why = Column(ARRAY(String), nullable=True)
    w5h_how = Column(ARRAY(String), nullable=True)

    # Relationships
    session = relationship("Session", back_populates="messages")
    fragments = relationship(
        "Fragment", back_populates="message", cascade="all, delete-orphan"
    )
    children = relationship("Message", backref=backref("parent", remote_side=[id]))

    __table_args__ = (
        Index("ix_messages_session_index", "session_id", "message_index"),
        Index("ix_messages_created_at", "created_at"),
        Index("ix_messages_role", "role"),
    )


class Fragment(Base, TimestampMixin):
    """Text fragments for vector search."""

    __tablename__ = "fragments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False, index=True)
    start_pos = Column(Integer, nullable=False)
    end_pos = Column(Integer, nullable=False)
    embedding = Column(JSON, nullable=True)  # Vector embedding as JSON array
    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    # Relationships
    message = relationship("Message", back_populates="fragments")

    __table_args__ = (
        Index("ix_fragments_hash", "content_hash"),
        Index("ix_fragments_message", "message_id"),
    )


class KnowledgeEntry(Base, TimestampMixin):
    """Knowledge base entries."""

    __tablename__ = "knowledge_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    content_type = Column(String, default="text/plain")
    content_hash = Column(String, nullable=False, index=True)

    # Trust and credibility
    trust_score = Column(Float, default=0.5)
    credibility_score = Column(Float, default=0.5)
    source = Column(String, nullable=True)

    # Vector embedding
    embedding = Column(JSON, nullable=True)

    # W5H indexing
    w5h_who = Column(ARRAY(String), nullable=True)
    w5h_what = Column(ARRAY(String), nullable=True)
    w5h_when = Column(ARRAY(String), nullable=True)
    w5h_where = Column(ARRAY(String), nullable=True)
    w5h_why = Column(ARRAY(String), nullable=True)
    w5h_how = Column(ARRAY(String), nullable=True)

    # Metadata
    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    # Relationships
    session = relationship("Session", backref="knowledge_entries")

    __table_args__ = (
        Index("ix_knowledge_entries_user", "user_id"),
        Index("ix_knowledge_entries_hash", "content_hash"),
        Index("ix_knowledge_entries_trust", "trust_score"),
        Index("ix_knowledge_entries_created", "created_at"),
    )


class Task(Base, TimestampMixin):
    """Background and governance tasks."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=True, index=True)
    task_type = Column(String, nullable=False, index=True)
    status = Column(String, default="pending", nullable=False, index=True)
    priority = Column(String, default="normal", nullable=False)

    # Task details
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    input_data = Column(JSON, nullable=True)
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timing
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Retry handling
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Relationships
    parent_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)
    children = relationship("Task", backref=backref("parent", remote_side=[id]))

    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    __table_args__ = (
        Index("ix_tasks_status_priority", "status", "priority"),
        Index("ix_tasks_type_status", "task_type", "status"),
        Index("ix_tasks_scheduled", "scheduled_at"),
    )


class Notification(Base, TimestampMixin):
    """User notifications."""

    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    notification_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)

    # Status
    is_read = Column(Boolean, default=False, index=True)
    is_actionable = Column(Boolean, default=False)
    action_url = Column(String, nullable=True)

    # Priority and delivery
    priority = Column(String, default="normal", nullable=False)
    delivery_channels = Column(ARRAY(String), nullable=True)

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)

    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    __table_args__ = (
        Index("ix_notifications_user_read", "user_id", "is_read"),
        Index("ix_notifications_created", "created_at"),
        Index("ix_notifications_expires", "expires_at"),
    )


class CollabSession(Base, TimestampMixin):
    """Collaborative editing sessions."""

    __tablename__ = "collab_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    owner_user_id = Column(String, nullable=False, index=True)

    # Session configuration
    document_content = Column(Text, nullable=True)
    document_type = Column(String, default="markdown")
    permissions = Column(JSON, nullable=True)  # Permission matrix

    # Status
    status = Column(String, default="active", nullable=False, index=True)
    is_public = Column(Boolean, default=False)
    max_participants = Column(Integer, default=10)

    # Versioning
    version = Column(Integer, default=1)
    last_edit_by = Column(String, nullable=True)
    last_edit_at = Column(DateTime(timezone=True), nullable=True)

    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    __table_args__ = (
        Index("ix_collab_sessions_owner", "owner_user_id"),
        Index("ix_collab_sessions_status", "status"),
        Index("ix_collab_sessions_public", "is_public"),
    )


# Event log table for audit trail
class EventLog(Base):
    """System event log for audit trail."""

    __tablename__ = "event_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Event identification
    event_type = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    trace_id = Column(String, nullable=True, index=True)

    # Context
    user_id = Column(String, nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Event data
    payload = Column(JSON, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Changed from metadata to extra_data

    __table_args__ = (
        Index("ix_event_logs_timestamp", "timestamp"),
        Index("ix_event_logs_type_source", "event_type", "source"),
        Index("ix_event_logs_user_session", "user_id", "session_id"),
    )
