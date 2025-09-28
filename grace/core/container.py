"""
Dependency injection container for Grace repositories.
Provides repository instances for use in FastAPI dependencies.
"""
from typing import Dict, Any, Type
from sqlalchemy.ext.asyncio import AsyncSession

from grace.core.repositories import (
    UserRepository, 
    SessionRepository, 
    MemoryRepository, 
    MemoryEmbeddingRepository,
    SystemOperationRepository, 
    BackgroundTaskRepository
)

class RepositoryContainer:
    """Container for repository instances."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._repositories: Dict[str, Any] = {}
    
    @property
    def users(self) -> UserRepository:
        """Get or create user repository."""
        if 'users' not in self._repositories:
            self._repositories['users'] = UserRepository(self.session)
        return self._repositories['users']
    
    @property
    def sessions(self) -> SessionRepository:
        """Get or create session repository."""
        if 'sessions' not in self._repositories:
            self._repositories['sessions'] = SessionRepository(self.session)
        return self._repositories['sessions']
    
    @property
    def memories(self) -> MemoryRepository:
        """Get or create memory repository."""
        if 'memories' not in self._repositories:
            self._repositories['memories'] = MemoryRepository(self.session)
        return self._repositories['memories']
    
    @property
    def memory_embeddings(self) -> MemoryEmbeddingRepository:
        """Get or create memory embedding repository."""
        if 'memory_embeddings' not in self._repositories:
            self._repositories['memory_embeddings'] = MemoryEmbeddingRepository(self.session)
        return self._repositories['memory_embeddings']
    
    @property
    def system_operations(self) -> SystemOperationRepository:
        """Get or create system operation repository."""
        if 'system_operations' not in self._repositories:
            self._repositories['system_operations'] = SystemOperationRepository(self.session)
        return self._repositories['system_operations']
    
    @property
    def background_tasks(self) -> BackgroundTaskRepository:
        """Get or create background task repository."""
        if 'background_tasks' not in self._repositories:
            self._repositories['background_tasks'] = BackgroundTaskRepository(self.session)
        return self._repositories['background_tasks']

# FastAPI dependency to provide repository container
async def get_repository_container(session: AsyncSession) -> RepositoryContainer:
    """Dependency that provides repository container."""
    return RepositoryContainer(session)