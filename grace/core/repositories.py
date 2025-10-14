"""
Repository interfaces and base implementations for Grace system.
Provides the repository pattern implementation to replace dict-based storage.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict, Any, Sequence
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_

from grace.core.models import (
    User,
    Session,
    Memory,
    MemoryEmbedding,
    SystemOperation,
    BackgroundTask,
)

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository interface."""

    @abstractmethod
    async def create(self, obj: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update entity by ID."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Sequence[T]:
        """List entities with pagination and filtering."""
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters."""
        pass


class SQLAlchemyRepository(BaseRepository[T]):
    """Base SQLAlchemy repository implementation."""

    def __init__(self, session: AsyncSession, model_class: type):
        self.session = session
        self.model_class = model_class

    async def create(self, obj: T) -> T:
        """Create a new entity."""
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        stmt = select(self.model_class).where(self.model_class.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update entity by ID."""
        # Remove None values and timestamps that should be auto-managed
        update_data = {k: v for k, v in data.items() if v is not None}
        if "created_at" in update_data:
            del update_data["created_at"]

        # If no data to update after filtering, return current object
        if not update_data:
            return await self.get_by_id(id)

        stmt = (
            update(self.model_class)
            .where(self.model_class.id == id)
            .values(update_data)
        )

        await self.session.execute(stmt)
        return await self.get_by_id(id)

    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        stmt = delete(self.model_class).where(self.model_class.id == id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Sequence[T]:
        """List entities with pagination and filtering."""
        stmt = select(self.model_class)

        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    conditions.append(getattr(self.model_class, key) == value)
            if conditions:
                stmt = stmt.where(and_(*conditions))

        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters."""
        from sqlalchemy import func

        stmt = select(func.count(self.model_class.id))

        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    conditions.append(getattr(self.model_class, key) == value)
            if conditions:
                stmt = stmt.where(and_(*conditions))


class UserRepository(SQLAlchemyRepository[User]):
    """User repository with authentication-specific methods."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        stmt = select(User).where(User.username == username)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp."""
        stmt = (
            update(User).where(User.id == user_id).values(last_login=datetime.utcnow())
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def get_active_users(self, limit: int = 100) -> Sequence[User]:
        """Get active users."""
        stmt = select(User).where(User.is_active == True).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class SessionRepository(SQLAlchemyRepository[Session]):
    """Session repository with token management."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Session)

    async def get_by_token_id(self, token_id: str) -> Optional[Session]:
        """Get session by token ID."""
        stmt = select(Session).where(Session.token_id == token_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_session(self, token_id: str) -> Optional[Session]:
        """Get active session by token ID."""
        stmt = select(Session).where(
            and_(
                Session.token_id == token_id,
                Session.is_active == True,
                Session.expires_at > datetime.utcnow(),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_sessions(
        self, user_id: str, active_only: bool = False
    ) -> Sequence[Session]:
        """Get all sessions for a user."""
        conditions = [Session.user_id == user_id]

        if active_only:
            conditions.extend(
                [Session.is_active == True, Session.expires_at > datetime.utcnow()]
            )

        stmt = select(Session).where(and_(*conditions))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def deactivate_session(self, token_id: str) -> bool:
        """Deactivate a session."""
        stmt = (
            update(Session).where(Session.token_id == token_id).values(is_active=False)
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        stmt = delete(Session).where(Session.expires_at <= datetime.utcnow())
        result = await self.session.execute(stmt)
        return result.rowcount


class MemoryRepository(SQLAlchemyRepository[Memory]):
    """Memory repository implementing the enhanced memory interface."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Memory)

    async def get_by_user_and_key(self, user_id: str, key: str) -> Optional[Memory]:
        """Get memory by user ID and key (legacy support)."""
        stmt = select(Memory).where(and_(Memory.user_id == user_id, Memory.key == key))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Memory]:
        """Get memories for a specific user with filtering."""
        conditions = [Memory.user_id == user_id]

        if memory_type:
            conditions.append(Memory.memory_type == memory_type)
        if category:
            conditions.append(Memory.category == category)

        stmt = select(Memory).where(and_(*conditions)).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def search_memories(
        self,
        user_id: str,
        search_text: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> Sequence[Memory]:
        """Search memories by content (basic text search)."""
        conditions = [Memory.user_id == user_id]

        if memory_type:
            conditions.append(Memory.memory_type == memory_type)

        # Simple text search - can be enhanced with FTS later
        conditions.append(
            or_(
                Memory.content.like(f"%{search_text}%"),
                Memory.key.like(f"%{search_text}%"),
            )
        )

        stmt = select(Memory).where(and_(*conditions)).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_tags(
        self, user_id: str, tags: List[str], limit: int = 50
    ) -> Sequence[Memory]:
        """Get memories by tags."""
        # This is a simplified version - proper JSON querying would be database-specific
        stmt = select(Memory).where(Memory.user_id == user_id).limit(limit)
        result = await self.session.execute(stmt)
        memories = result.scalars().all()

        # Filter in Python for now (can be optimized with proper JSON operators)
        filtered = []
        for memory in memories:
            if memory.tags and any(tag in memory.tags for tag in tags):
                filtered.append(memory)

        return filtered

    async def update_access(self, memory_id: str) -> bool:
        """Update access count and timestamp."""
        stmt = (
            update(Memory)
            .where(Memory.id == memory_id)
            .values(
                access_count=Memory.access_count + 1, last_accessed=datetime.utcnow()
            )
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def cleanup_expired_memories(self) -> int:
        """Remove expired memories."""
        stmt = delete(Memory).where(Memory.expires_at <= datetime.utcnow())
        result = await self.session.execute(stmt)
        return result.rowcount


class MemoryEmbeddingRepository(SQLAlchemyRepository[MemoryEmbedding]):
    """Repository for vector embeddings."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, MemoryEmbedding)

    async def get_by_memory_id(self, memory_id: str) -> Sequence[MemoryEmbedding]:
        """Get all embeddings for a memory."""
        stmt = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_model(
        self, embedding_model: str, limit: int = 1000
    ) -> Sequence[MemoryEmbedding]:
        """Get embeddings by model for batch processing."""
        stmt = (
            select(MemoryEmbedding)
            .where(MemoryEmbedding.embedding_model == embedding_model)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def delete_by_memory_id(self, memory_id: str) -> int:
        """Delete all embeddings for a memory."""
        stmt = delete(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
        result = await self.session.execute(stmt)
        return result.rowcount


class SystemOperationRepository(SQLAlchemyRepository[SystemOperation]):
    """Repository for system operations and audit logs."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, SystemOperation)

    async def log_operation(
        self,
        user_id: Optional[str],
        operation_type: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        operation_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> SystemOperation:
        """Log a system operation."""
        operation = SystemOperation(
            user_id=user_id,
            operation_type=operation_type,
            resource_type=resource_type,
            resource_id=resource_id,
            operation_data=operation_data,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return await self.create(operation)

    async def get_user_operations(
        self, user_id: str, operation_type: Optional[str] = None, limit: int = 100
    ) -> Sequence[SystemOperation]:
        """Get operations for a user."""
        conditions = [SystemOperation.user_id == user_id]

        if operation_type:
            conditions.append(SystemOperation.operation_type == operation_type)

        stmt = select(SystemOperation).where(and_(*conditions)).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class BackgroundTaskRepository(SQLAlchemyRepository[BackgroundTask]):
    """Repository for background task management."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, BackgroundTask)

    async def get_pending_tasks(
        self, task_type: Optional[str] = None, limit: int = 10
    ) -> Sequence[BackgroundTask]:
        """Get pending tasks ordered by priority."""
        conditions = [BackgroundTask.status == "pending"]

        if task_type:
            conditions.append(BackgroundTask.task_type == task_type)

        stmt = (
            select(BackgroundTask)
            .where(and_(*conditions))
            .order_by(BackgroundTask.priority.desc(), BackgroundTask.scheduled_at)
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        worker_id: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Optional[BackgroundTask]:
        """Update task status and execution details."""
        update_data = {"status": status}

        if status == "running" and worker_id:
            update_data["worker_id"] = worker_id
            update_data["started_at"] = datetime.utcnow()
        elif status in ["completed", "failed"]:
            update_data["completed_at"] = datetime.utcnow()
            if result_data:
                update_data["result_data"] = result_data
            if error_message:
                update_data["error_message"] = error_message

        return await self.update(task_id, update_data)
