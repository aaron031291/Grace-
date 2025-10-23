"""
Governance models - Policies, collaboration sessions, and tasks
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship, Mapped, mapped_column
import enum

from grace.auth.models import Base


class PolicyStatus(enum.Enum):
    """Policy status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class PolicyType(enum.Enum):
    """Policy type enumeration"""
    ETHICAL = "ethical"
    SECURITY = "security"
    PRIVACY = "privacy"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    CUSTOM = "custom"


class Policy(Base):
    """Policy model for governance"""
    __tablename__ = 'policies'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey('users.id'), nullable=False)
    
    # Policy details
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    policy_type: Mapped[PolicyType] = mapped_column(SQLEnum(PolicyType), nullable=False)
    status: Mapped[PolicyStatus] = mapped_column(SQLEnum(PolicyStatus), default=PolicyStatus.DRAFT)
    
    # Policy content
    rules: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # List of rules
    constraints: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Constraints
    metadata_json: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Version control
    version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    parent_policy_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('policies.id'), nullable=True)
    
    # Approval workflow
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=True)
    approved_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('users.id'), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
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
    effective_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Policy(id={self.id}, name={self.name}, type={self.policy_type.value})>"


class SessionStatus(enum.Enum):
    """Collaboration session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CollaborationSession(Base):
    """Collaboration session model"""
    __tablename__ = 'collaboration_sessions'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey('users.id'), nullable=False)
    
    # Session details
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    session_type: Mapped[str] = mapped_column(String(50), default="general")
    status: Mapped[SessionStatus] = mapped_column(SQLEnum(SessionStatus), default=SessionStatus.ACTIVE)
    
    # Participants (stored as JSON array of user IDs)
    participants: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Session data
    context: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Session context/state
    messages: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Chat messages
    decisions: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Decisions made
    metadata_json: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
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
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    tasks: Mapped[List["Task"]] = relationship("Task", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<CollaborationSession(id={self.id}, name={self.name}, status={self.status.value})>"


class TaskStatus(enum.Enum):
    """Task status enumeration"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(enum.Enum):
    """Task priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(Base):
    """Task model for collaboration and workflow"""
    __tablename__ = 'tasks'
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(36), ForeignKey('users.id'), nullable=False)
    assigned_to: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('users.id'), nullable=True)
    
    # Task details
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[TaskStatus] = mapped_column(SQLEnum(TaskStatus), default=TaskStatus.TODO)
    priority: Mapped[TaskPriority] = mapped_column(SQLEnum(TaskPriority), default=TaskPriority.MEDIUM)
    
    # Associations
    session_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('collaboration_sessions.id'), nullable=True)
    policy_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey('policies.id'), nullable=True)
    
    # Task data
    tags: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # List of tags
    dependencies: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Task IDs that must complete first
    attachments: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # File references
    metadata_json: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Progress tracking
    progress_percentage: Mapped[int] = mapped_column(Integer, default=0)
    estimated_hours: Mapped[Optional[float]] = mapped_column(Integer, nullable=True)
    actual_hours: Mapped[Optional[float]] = mapped_column(Integer, nullable=True)
    
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
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    session: Mapped[Optional["CollaborationSession"]] = relationship("CollaborationSession", back_populates="tasks")
    
    def __repr__(self):
        return f"<Task(id={self.id}, title={self.title}, status={self.status.value})>"
