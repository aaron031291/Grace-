"""
Memory API endpoints for Grace system.
Replaces dict-based memory storage with repository pattern.
"""
from typing import Annotated, Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime

from grace.core.database import get_async_session
from grace.core.container import get_repository_container, RepositoryContainer
from grace.core.models import Memory, MemoryEmbedding, generate_uuid
from grace.core.dependencies import CurrentUser, require_permissions

router = APIRouter(prefix="/api/v1/memories", tags=["Memory"])

class MemoryRequest(BaseModel):
    """Memory creation/update request schema."""
    key: str
    content: Optional[str] = None
    content_type: str = "application/json"
    binary_content: Optional[bytes] = None
    memory_metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    memory_type: str = "fusion"
    category: Optional[str] = None
    priority: int = 1
    ttl_seconds: Optional[int] = None

class MemoryResponse(BaseModel):
    """Memory response schema."""
    id: str
    key: str
    content: Optional[str] = None
    content_type: str
    memory_metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    memory_type: str
    category: Optional[str] = None
    priority: int
    access_count: int
    size_bytes: Optional[int] = None
    is_compressed: bool
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime] = None

class MemoryListResponse(BaseModel):
    """Memory list response with pagination."""
    memories: List[MemoryResponse]
    total: int
    offset: int
    limit: int

@router.post("/", response_model=MemoryResponse)
async def create_memory(
    memory_data: MemoryRequest,
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    _: Annotated[None, Depends(require_permissions("memories:create"))]
):
    """Create a new memory entry."""
    
    # Check if memory with this key already exists for user
    existing = await repos.memories.get_by_user_and_key(current_user.id, memory_data.key)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Memory with key '{memory_data.key}' already exists"
        )
    
    # Calculate content size and hash
    content_size = 0
    content_hash = None
    
    if memory_data.content:
        content_size = len(memory_data.content.encode('utf-8'))
        import hashlib
        content_hash = hashlib.sha256(memory_data.content.encode('utf-8')).hexdigest()
    elif memory_data.binary_content:
        content_size = len(memory_data.binary_content)
        import hashlib
        content_hash = hashlib.sha256(memory_data.binary_content).hexdigest()
    
    # Create memory
    memory = Memory(
        user_id=current_user.id,
        key=memory_data.key,
        content=memory_data.content,
        content_type=memory_data.content_type,
        binary_content=memory_data.binary_content,
        memory_metadata=memory_data.memory_metadata,
        tags=memory_data.tags,
        memory_type=memory_data.memory_type,
        category=memory_data.category,
        priority=memory_data.priority,
        size_bytes=content_size,
        content_hash=content_hash,
        ttl_seconds=memory_data.ttl_seconds
    )
    
    # Set expiration if TTL is provided
    if memory_data.ttl_seconds:
        from datetime import timedelta
        memory.expires_at = datetime.utcnow() + timedelta(seconds=memory_data.ttl_seconds)
    
    created_memory = await repos.memories.create(memory)
    await session.commit()
    
    return MemoryResponse(
        id=created_memory.id,
        key=created_memory.key,
        content=created_memory.content,
        content_type=created_memory.content_type,
        memory_metadata=created_memory.memory_metadata,
        tags=created_memory.tags,
        memory_type=created_memory.memory_type,
        category=created_memory.category,
        priority=created_memory.priority,
        access_count=created_memory.access_count,
        size_bytes=created_memory.size_bytes,
        is_compressed=created_memory.is_compressed,
        created_at=created_memory.created_at,
        updated_at=created_memory.updated_at,
        last_accessed=created_memory.last_accessed,
        expires_at=created_memory.expires_at
    )

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    _: Annotated[None, Depends(require_permissions("memories:read"))]
):
    """Get a memory by ID."""
    
    memory = await repos.memories.get_by_id(memory_id)
    if not memory or memory.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    # Update access count
    await repos.memories.update_access(memory_id)
    await session.commit()
    
    return MemoryResponse(
        id=memory.id,
        key=memory.key,
        content=memory.content,
        content_type=memory.content_type,
        memory_metadata=memory.memory_metadata,
        tags=memory.tags,
        memory_type=memory.memory_type,
        category=memory.category,
        priority=memory.priority,
        access_count=memory.access_count + 1,  # Reflect the update
        size_bytes=memory.size_bytes,
        is_compressed=memory.is_compressed,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        last_accessed=datetime.utcnow(),
        expires_at=memory.expires_at
    )

@router.get("/key/{key}", response_model=MemoryResponse)  
async def get_memory_by_key(
    key: str,
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    _: Annotated[None, Depends(require_permissions("memories:read"))]
):
    """Get a memory by key (legacy support)."""
    
    memory = await repos.memories.get_by_user_and_key(current_user.id, key)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory with key '{key}' not found"
        )
    
    # Update access count
    await repos.memories.update_access(memory.id)
    await session.commit()
    
    return MemoryResponse(
        id=memory.id,
        key=memory.key,
        content=memory.content,
        content_type=memory.content_type,
        memory_metadata=memory.memory_metadata,
        tags=memory.tags,
        memory_type=memory.memory_type,
        category=memory.category,
        priority=memory.priority,
        access_count=memory.access_count + 1,
        size_bytes=memory.size_bytes,
        is_compressed=memory.is_compressed,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        last_accessed=datetime.utcnow(),
        expires_at=memory.expires_at
    )

@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    memory_data: MemoryRequest,
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    _: Annotated[None, Depends(require_permissions("memories:update"))]
):
    """Update a memory entry."""
    
    memory = await repos.memories.get_by_id(memory_id)
    if not memory or memory.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    # Calculate new content size and hash
    content_size = 0
    content_hash = None
    
    if memory_data.content:
        content_size = len(memory_data.content.encode('utf-8'))
        import hashlib
        content_hash = hashlib.sha256(memory_data.content.encode('utf-8')).hexdigest()
    elif memory_data.binary_content:
        content_size = len(memory_data.binary_content)
        import hashlib
        content_hash = hashlib.sha256(memory_data.binary_content).hexdigest()
    
    # Prepare update data
    update_data = {
        "key": memory_data.key,
        "content": memory_data.content,
        "content_type": memory_data.content_type,
        "binary_content": memory_data.binary_content,
        "memory_metadata": memory_data.memory_metadata,
        "tags": memory_data.tags,
        "memory_type": memory_data.memory_type,
        "category": memory_data.category,
        "priority": memory_data.priority,
        "size_bytes": content_size,
        "content_hash": content_hash,
        "ttl_seconds": memory_data.ttl_seconds
    }
    
    # Set expiration if TTL is provided
    if memory_data.ttl_seconds:
        from datetime import timedelta
        update_data["expires_at"] = datetime.utcnow() + timedelta(seconds=memory_data.ttl_seconds)
    
    updated_memory = await repos.memories.update(memory_id, update_data)
    await session.commit()
    
    return MemoryResponse(
        id=updated_memory.id,
        key=updated_memory.key,
        content=updated_memory.content,
        content_type=updated_memory.content_type,
        memory_metadata=updated_memory.memory_metadata,
        tags=updated_memory.tags,
        memory_type=updated_memory.memory_type,
        category=updated_memory.category,
        priority=updated_memory.priority,
        access_count=updated_memory.access_count,
        size_bytes=updated_memory.size_bytes,
        is_compressed=updated_memory.is_compressed,
        created_at=updated_memory.created_at,
        updated_at=updated_memory.updated_at,
        last_accessed=updated_memory.last_accessed,
        expires_at=updated_memory.expires_at
    )

@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    _: Annotated[None, Depends(require_permissions("memories:delete"))]
):
    """Delete a memory entry."""
    
    memory = await repos.memories.get_by_id(memory_id)
    if not memory or memory.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    
    # Delete associated embeddings first
    await repos.memory_embeddings.delete_by_memory_id(memory_id)
    
    # Delete the memory
    await repos.memories.delete(memory_id)
    await session.commit()
    
    return {"message": "Memory deleted successfully"}

@router.get("/", response_model=MemoryListResponse)
async def list_memories(
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    _: Annotated[None, Depends(require_permissions("memories:read"))],
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=100, description="Pagination limit")
):
    """List user's memories with filtering and pagination."""
    
    memories = await repos.memories.get_user_memories(
        user_id=current_user.id,
        memory_type=memory_type,
        category=category,
        offset=offset,
        limit=limit
    )
    
    # Get total count for pagination - handle case where there are no records
    filters = {"user_id": current_user.id}
    if memory_type:
        filters["memory_type"] = memory_type
    if category:
        filters["category"] = category
    
    total = await repos.memories.count(filters) or 0
    
    memory_responses = [
        MemoryResponse(
            id=memory.id,
            key=memory.key,
            content=memory.content,
            content_type=memory.content_type,
            memory_metadata=memory.memory_metadata,
            tags=memory.tags,
            memory_type=memory.memory_type,
            category=memory.category,
            priority=memory.priority,
            access_count=memory.access_count,
            size_bytes=memory.size_bytes,
            is_compressed=memory.is_compressed,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            last_accessed=memory.last_accessed,
            expires_at=memory.expires_at
        )
        for memory in memories
    ]
    
    return MemoryListResponse(
        memories=memory_responses,
        total=total,
        offset=offset,
        limit=limit
    )

@router.get("/search", response_model=MemoryListResponse)
async def search_memories(
    current_user: CurrentUser,
    repos: Annotated[RepositoryContainer, Depends(get_repository_container)],
    _: Annotated[None, Depends(require_permissions("memories:search"))],
    q: str = Query(..., description="Search query"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    limit: int = Query(20, ge=1, le=100, description="Result limit")
):
    """Search user's memories by content."""
    
    memories = await repos.memories.search_memories(
        user_id=current_user.id,
        search_text=q,
        memory_type=memory_type,
        limit=limit
    )
    
    memory_responses = [
        MemoryResponse(
            id=memory.id,
            key=memory.key,
            content=memory.content,
            content_type=memory.content_type,
            memory_metadata=memory.memory_metadata,
            tags=memory.tags,
            memory_type=memory.memory_type,
            category=memory.category,
            priority=memory.priority,
            access_count=memory.access_count,
            size_bytes=memory.size_bytes,
            is_compressed=memory.is_compressed,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            last_accessed=memory.last_accessed,
            expires_at=memory.expires_at
        )
        for memory in memories
    ]
    
    return MemoryListResponse(
        memories=memory_responses,
        total=len(memory_responses),
        offset=0,
        limit=limit
    )