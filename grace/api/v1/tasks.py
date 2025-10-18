"""
Tasks API endpoints - CRUD operations
"""

from datetime import datetime, timezone
from typing import List, Optional
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from grace.auth.models import User
from grace.auth.dependencies import get_current_user
from grace.database import get_db
from grace.governance.models import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


# Pydantic schemas
class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    priority: str = "MEDIUM"
    assigned_to: Optional[str] = None
    session_id: Optional[str] = None
    policy_id: Optional[str] = None
    tags: List[str] = []
    dependencies: List[str] = []
    estimated_hours: Optional[float] = None
    due_date: Optional[datetime] = None
    metadata: Optional[dict] = {}


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_to: Optional[str] = None
    tags: Optional[List[str]] = None
    progress_percentage: Optional[int] = Field(None, ge=0, le=100)
    actual_hours: Optional[float] = None
    due_date: Optional[datetime] = None
    metadata: Optional[dict] = None


class TaskResponse(BaseModel):
    id: str
    created_by: str
    assigned_to: Optional[str]
    title: str
    description: Optional[str]
    status: str
    priority: str
    session_id: Optional[str]
    policy_id: Optional[str]
    tags: Optional[List[str]]
    dependencies: Optional[List[str]]
    progress_percentage: int
    estimated_hours: Optional[float]
    actual_hours: Optional[float]
    metadata: Optional[dict]
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new task"""
    
    # Validate priority
    try:
        priority_enum = TaskPriority[task.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority. Must be one of: {[p.name for p in TaskPriority]}"
        )
    
    db_task = Task(
        id=str(uuid.uuid4()),
        created_by=current_user.id,
        assigned_to=task.assigned_to,
        title=task.title,
        description=task.description,
        status=TaskStatus.TODO,
        priority=priority_enum,
        session_id=task.session_id,
        policy_id=task.policy_id,
        tags=task.tags,
        dependencies=task.dependencies,
        estimated_hours=task.estimated_hours,
        due_date=task.due_date,
        metadata_json=task.metadata
    )
    
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    logger.info(f"Task created: {db_task.id} by user {current_user.id}")
    
    return TaskResponse(
        id=db_task.id,
        created_by=db_task.created_by,
        assigned_to=db_task.assigned_to,
        title=db_task.title,
        description=db_task.description,
        status=db_task.status.value,
        priority=db_task.priority.value,
        session_id=db_task.session_id,
        policy_id=db_task.policy_id,
        tags=db_task.tags,
        dependencies=db_task.dependencies,
        progress_percentage=db_task.progress_percentage,
        estimated_hours=db_task.estimated_hours,
        actual_hours=db_task.actual_hours,
        metadata=db_task.metadata_json,
        created_at=db_task.created_at,
        updated_at=db_task.updated_at,
        due_date=db_task.due_date,
        started_at=db_task.started_at,
        completed_at=db_task.completed_at
    )


@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    assigned_to_me: bool = Query(False),
    session_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List tasks with optional filtering"""
    
    query = db.query(Task)
    
    # Filter by creator or assignee
    if assigned_to_me:
        query = query.filter(Task.assigned_to == current_user.id)
    else:
        query = query.filter(
            (Task.created_by == current_user.id) |
            (Task.assigned_to == current_user.id)
        )
    
    if status_filter:
        try:
            status_enum = TaskStatus[status_filter.upper()]
            query = query.filter(Task.status == status_enum)
        except KeyError:
            pass
    
    if session_id:
        query = query.filter(Task.session_id == session_id)
    
    tasks = query.order_by(Task.created_at.desc()).offset(skip).limit(limit).all()
    
    return [
        TaskResponse(
            id=t.id,
            created_by=t.created_by,
            assigned_to=t.assigned_to,
            title=t.title,
            description=t.description,
            status=t.status.value,
            priority=t.priority.value,
            session_id=t.session_id,
            policy_id=t.policy_id,
            tags=t.tags,
            dependencies=t.dependencies,
            progress_percentage=t.progress_percentage,
            estimated_hours=t.estimated_hours,
            actual_hours=t.actual_hours,
            metadata=t.metadata_json,
            created_at=t.created_at,
            updated_at=t.updated_at,
            due_date=t.due_date,
            started_at=t.started_at,
            completed_at=t.completed_at
        )
        for t in tasks
    ]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific task"""
    
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    # Check access
    if task.created_by != current_user.id and task.assigned_to != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this task"
        )
    
    return TaskResponse(
        id=task.id,
        created_by=task.created_by,
        assigned_to=task.assigned_to,
        title=task.title,
        description=task.description,
        status=task.status.value,
        priority=task.priority.value,
        session_id=task.session_id,
        policy_id=task.policy_id,
        tags=task.tags,
        dependencies=task.dependencies,
        progress_percentage=task.progress_percentage,
        estimated_hours=task.estimated_hours,
        actual_hours=task.actual_hours,
        metadata=task.metadata_json,
        created_at=task.created_at,
        updated_at=task.updated_at,
        due_date=task.due_date,
        started_at=task.started_at,
        completed_at=task.completed_at
    )


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    update: TaskUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a task"""
    
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    # Check permissions
    if task.created_by != current_user.id and task.assigned_to != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this task"
        )
    
    # Update fields
    if update.title is not None:
        task.title = update.title
    if update.description is not None:
        task.description = update.description
    if update.status is not None:
        try:
            new_status = TaskStatus[update.status.upper()]
            task.status = new_status
            
            if new_status == TaskStatus.IN_PROGRESS and not task.started_at:
                task.started_at = datetime.now(timezone.utc)
            elif new_status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now(timezone.utc)
                task.progress_percentage = 100
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.name for s in TaskStatus]}"
            )
    if update.priority is not None:
        try:
            task.priority = TaskPriority[update.priority.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority. Must be one of: {[p.name for p in TaskPriority]}"
            )
    if update.assigned_to is not None:
        task.assigned_to = update.assigned_to
    if update.tags is not None:
        task.tags = update.tags
    if update.progress_percentage is not None:
        task.progress_percentage = update.progress_percentage
    if update.actual_hours is not None:
        task.actual_hours = update.actual_hours
    if update.due_date is not None:
        task.due_date = update.due_date
    if update.metadata is not None:
        task.metadata_json = update.metadata
    
    db.commit()
    db.refresh(task)
    
    logger.info(f"Task updated: {task_id} by user {current_user.id}")
    
    return TaskResponse(
        id=task.id,
        created_by=task.created_by,
        assigned_to=task.assigned_to,
        title=task.title,
        description=task.description,
        status=task.status.value,
        priority=task.priority.value,
        session_id=task.session_id,
        policy_id=task.policy_id,
        tags=task.tags,
        dependencies=task.dependencies,
        progress_percentage=task.progress_percentage,
        estimated_hours=task.estimated_hours,
        actual_hours=task.actual_hours,
        metadata=task.metadata_json,
        created_at=task.created_at,
        updated_at=task.updated_at,
        due_date=task.due_date,
        started_at=task.started_at,
        completed_at=task.completed_at
    )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a task"""
    
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    # Check permissions (only creator can delete)
    if task.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this task"
        )
    
    db.delete(task)
    db.commit()
    
    logger.info(f"Task deleted: {task_id} by user {current_user.id}")
