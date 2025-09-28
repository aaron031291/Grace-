"""
Repository Pattern Implementation for Grace Models

Provides a clean abstraction layer over database operations with caching.
"""
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload, joinedload

from .models import (
    Session, Panel, Message, Fragment, KnowledgeEntry, 
    Task, Notification, CollabSession, EventLog, Base
)

T = TypeVar('T', bound=Base)

logger = logging.getLogger(__name__)


class BaseRepository(Generic[T], ABC):
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: AsyncSession, model_class: type[T]):
        self.session = session
        self.model_class = model_class
    
    async def get_by_id(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination."""
        result = await self.session.execute(
            select(self.model_class)
            .order_by(self.model_class.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
    
    async def create(self, **kwargs) -> T:
        """Create new entity."""
        entity = self.model_class(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, id: UUID, **kwargs) -> Optional[T]:
        """Update entity by ID."""
        await self.session.execute(
            update(self.model_class)
            .where(self.model_class.id == id)
            .values(**kwargs)
        )
        return await self.get_by_id(id)
    
    async def delete(self, id: UUID) -> bool:
        """Delete entity by ID."""
        result = await self.session.execute(
            delete(self.model_class).where(self.model_class.id == id)
        )
        return result.rowcount > 0


class SessionRepository(BaseRepository[Session]):
    """Repository for user sessions."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Session)
    
    async def get_by_user(self, user_id: str, limit: int = 50) -> List[Session]:
        """Get sessions for a user."""
        result = await self.session.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_with_messages(self, session_id: UUID) -> Optional[Session]:
        """Get session with all messages loaded."""
        result = await self.session.execute(
            select(Session)
            .where(Session.id == session_id)
            .options(selectinload(Session.messages))
        )
        return result.scalar_one_or_none()
    
    async def get_active_by_user(self, user_id: str) -> List[Session]:
        """Get active sessions for a user."""
        result = await self.session.execute(
            select(Session)
            .where(Session.user_id == user_id, Session.status == 'active')
            .order_by(Session.updated_at.desc())
        )
        return list(result.scalars().all())


class MessageRepository(BaseRepository[Message]):
    """Repository for messages."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Message)
    
    async def get_by_session(self, session_id: UUID, limit: int = 100) -> List[Message]:
        """Get messages for a session."""
        result = await self.session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.message_index)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_with_fragments(self, message_id: UUID) -> Optional[Message]:
        """Get message with fragments loaded."""
        result = await self.session.execute(
            select(Message)
            .where(Message.id == message_id)
            .options(selectinload(Message.fragments))
        )
        return result.scalar_one_or_none()
    
    async def search_by_content(self, query: str, limit: int = 50) -> List[Message]:
        """Search messages by content."""
        result = await self.session.execute(
            select(Message)
            .where(Message.content.contains(query))
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class FragmentRepository(BaseRepository[Fragment]):
    """Repository for text fragments."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Fragment)
    
    async def get_by_message(self, message_id: UUID) -> List[Fragment]:
        """Get fragments for a message."""
        result = await self.session.execute(
            select(Fragment)
            .where(Fragment.message_id == message_id)
            .order_by(Fragment.start_pos)
        )
        return list(result.scalars().all())
    
    async def get_by_hash(self, content_hash: str) -> Optional[Fragment]:
        """Get fragment by content hash."""
        result = await self.session.execute(
            select(Fragment).where(Fragment.content_hash == content_hash)
        )
        return result.scalar_one_or_none()


class KnowledgeEntryRepository(BaseRepository[KnowledgeEntry]):
    """Repository for knowledge entries."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, KnowledgeEntry)
    
    async def get_by_user(self, user_id: str, limit: int = 100) -> List[KnowledgeEntry]:
        """Get knowledge entries for a user."""
        result = await self.session.execute(
            select(KnowledgeEntry)
            .where(KnowledgeEntry.user_id == user_id)
            .order_by(KnowledgeEntry.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def search_by_content(self, query: str, trust_threshold: float = 0.5, limit: int = 50) -> List[KnowledgeEntry]:
        """Search knowledge entries by content and trust."""
        result = await self.session.execute(
            select(KnowledgeEntry)
            .where(
                KnowledgeEntry.content.contains(query),
                KnowledgeEntry.trust_score >= trust_threshold
            )
            .order_by(KnowledgeEntry.trust_score.desc(), KnowledgeEntry.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_by_trust_score(self, min_trust: float, limit: int = 100) -> List[KnowledgeEntry]:
        """Get knowledge entries above trust threshold."""
        result = await self.session.execute(
            select(KnowledgeEntry)
            .where(KnowledgeEntry.trust_score >= min_trust)
            .order_by(KnowledgeEntry.trust_score.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class TaskRepository(BaseRepository[Task]):
    """Repository for tasks."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Task)
    
    async def get_by_status(self, status: str, limit: int = 100) -> List[Task]:
        """Get tasks by status."""
        result = await self.session.execute(
            select(Task)
            .where(Task.status == status)
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_pending_tasks(self, limit: int = 100) -> List[Task]:
        """Get pending tasks ordered by priority and creation time."""
        result = await self.session.execute(
            select(Task)
            .where(Task.status == 'pending')
            .order_by(Task.priority.desc(), Task.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_by_user(self, user_id: str, limit: int = 100) -> List[Task]:
        """Get tasks for a user."""
        result = await self.session.execute(
            select(Task)
            .where(Task.user_id == user_id)
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class NotificationRepository(BaseRepository[Notification]):
    """Repository for notifications."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Notification)
    
    async def get_by_user(self, user_id: str, unread_only: bool = False, limit: int = 100) -> List[Notification]:
        """Get notifications for a user."""
        query = select(Notification).where(Notification.user_id == user_id)
        
        if unread_only:
            query = query.where(Notification.is_read == False)
        
        result = await self.session.execute(
            query.order_by(Notification.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
    
    async def mark_as_read(self, notification_id: UUID) -> bool:
        """Mark notification as read."""
        result = await self.session.execute(
            update(Notification)
            .where(Notification.id == notification_id)
            .values(is_read=True)
        )
        return result.rowcount > 0


class CollabSessionRepository(BaseRepository[CollabSession]):
    """Repository for collaborative sessions."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, CollabSession)
    
    async def get_by_owner(self, owner_user_id: str, limit: int = 50) -> List[CollabSession]:
        """Get collaborative sessions owned by user."""
        result = await self.session.execute(
            select(CollabSession)
            .where(CollabSession.owner_user_id == owner_user_id)
            .order_by(CollabSession.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_public_sessions(self, limit: int = 50) -> List[CollabSession]:
        """Get public collaborative sessions."""
        result = await self.session.execute(
            select(CollabSession)
            .where(CollabSession.is_public == True, CollabSession.status == 'active')
            .order_by(CollabSession.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class EventLogRepository(BaseRepository[EventLog]):
    """Repository for event logs."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, EventLog)
    
    async def create_event(self, event_type: str, source: str, payload: Dict[str, Any], 
                          user_id: Optional[str] = None, session_id: Optional[UUID] = None,
                          trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EventLog:
        """Create a new event log entry."""
        return await self.create(
            event_type=event_type,
            source=source,
            payload=payload,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            metadata=metadata
        )
    
    async def get_by_trace_id(self, trace_id: str, limit: int = 100) -> List[EventLog]:
        """Get events by trace ID."""
        result = await self.session.execute(
            select(EventLog)
            .where(EventLog.trace_id == trace_id)
            .order_by(EventLog.timestamp)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_by_user_session(self, user_id: str, session_id: UUID, limit: int = 100) -> List[EventLog]:
        """Get events for a user session."""
        result = await self.session.execute(
            select(EventLog)
            .where(EventLog.user_id == user_id, EventLog.session_id == session_id)
            .order_by(EventLog.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# Cached repository factory with LRU cache for short-lived instances
@lru_cache(maxsize=128)
def _get_repository_class(repo_type: str) -> type:
    """Get repository class by type name."""
    repo_map = {
        'session': SessionRepository,
        'message': MessageRepository,
        'fragment': FragmentRepository,
        'knowledge_entry': KnowledgeEntryRepository,
        'task': TaskRepository,
        'notification': NotificationRepository,
        'collab_session': CollabSessionRepository,
        'event_log': EventLogRepository,
    }
    return repo_map.get(repo_type)


class RepositoryFactory:
    """Factory for creating repository instances with caching."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._repos: Dict[str, Any] = {}
    
    def get_session_repo(self) -> SessionRepository:
        if 'session' not in self._repos:
            self._repos['session'] = SessionRepository(self.session)
        return self._repos['session']
    
    def get_message_repo(self) -> MessageRepository:
        if 'message' not in self._repos:
            self._repos['message'] = MessageRepository(self.session)
        return self._repos['message']
    
    def get_fragment_repo(self) -> FragmentRepository:
        if 'fragment' not in self._repos:
            self._repos['fragment'] = FragmentRepository(self.session)
        return self._repos['fragment']
    
    def get_knowledge_entry_repo(self) -> KnowledgeEntryRepository:
        if 'knowledge_entry' not in self._repos:
            self._repos['knowledge_entry'] = KnowledgeEntryRepository(self.session)
        return self._repos['knowledge_entry']
    
    def get_task_repo(self) -> TaskRepository:
        if 'task' not in self._repos:
            self._repos['task'] = TaskRepository(self.session)
        return self._repos['task']
    
    def get_notification_repo(self) -> NotificationRepository:
        if 'notification' not in self._repos:
            self._repos['notification'] = NotificationRepository(self.session)
        return self._repos['notification']
    
    def get_collab_session_repo(self) -> CollabSessionRepository:
        if 'collab_session' not in self._repos:
            self._repos['collab_session'] = CollabSessionRepository(self.session)
        return self._repos['collab_session']
    
    def get_event_log_repo(self) -> EventLogRepository:
        if 'event_log' not in self._repos:
            self._repos['event_log'] = EventLogRepository(self.session)
        return self._repos['event_log']