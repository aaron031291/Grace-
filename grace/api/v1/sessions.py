"""
Collaboration Sessions API endpoints - CRUD operations
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
from grace.governance.models import CollaborationSession, SessionStatus
from grace.api.v1.websocket import publish_to_channel, channel_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Collaboration Sessions"])


# Pydantic schemas
class SessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    session_type: str = "general"
    participants: List[str] = []
    context: Optional[dict] = {}
    metadata: Optional[dict] = {}


class SessionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[str] = None
    participants: Optional[List[str]] = None
    context: Optional[dict] = None
    metadata: Optional[dict] = None


class MessageAdd(BaseModel):
    content: str
    message_type: str = "text"
    metadata: Optional[dict] = {}


class SessionResponse(BaseModel):
    id: str
    created_by: str
    name: str
    description: Optional[str]
    session_type: str
    status: str
    participants: Optional[List[str]]
    context: Optional[dict]
    messages: Optional[List[dict]]
    decisions: Optional[List[dict]]
    metadata: Optional[dict]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new collaboration session"""
    
    # Add creator to participants if not already included
    participants = session.participants or []
    if current_user.id not in participants:
        participants.append(current_user.id)
    
    db_session = CollaborationSession(
        id=str(uuid.uuid4()),
        created_by=current_user.id,
        name=session.name,
        description=session.description,
        session_type=session.session_type,
        status=SessionStatus.ACTIVE,
        participants=participants,
        context=session.context,
        messages=[],
        decisions=[],
        metadata_json=session.metadata,
        started_at=datetime.now(timezone.utc)
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    logger.info(f"Session created: {db_session.id} by user {current_user.id}")
    
    return SessionResponse(
        id=db_session.id,
        created_by=db_session.created_by,
        name=db_session.name,
        description=db_session.description,
        session_type=db_session.session_type,
        status=db_session.status.value,
        participants=db_session.participants,
        context=db_session.context,
        messages=db_session.messages,
        decisions=db_session.decisions,
        metadata=db_session.metadata_json,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        started_at=db_session.started_at,
        completed_at=db_session.completed_at
    )


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List collaboration sessions (user's sessions only)"""
    
    query = db.query(CollaborationSession).filter(
        (CollaborationSession.created_by == current_user.id) |
        (CollaborationSession.participants.contains([current_user.id]))
    )
    
    if status_filter:
        try:
            status_enum = SessionStatus[status_filter.upper()]
            query = query.filter(CollaborationSession.status == status_enum)
        except KeyError:
            pass
    
    sessions = query.order_by(CollaborationSession.updated_at.desc()).offset(skip).limit(limit).all()
    
    return [
        SessionResponse(
            id=s.id,
            created_by=s.created_by,
            name=s.name,
            description=s.description,
            session_type=s.session_type,
            status=s.status.value,
            participants=s.participants,
            context=s.context,
            messages=s.messages,
            decisions=s.decisions,
            metadata=s.metadata_json,
            created_at=s.created_at,
            updated_at=s.updated_at,
            started_at=s.started_at,
            completed_at=s.completed_at
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific collaboration session"""
    
    session = db.query(CollaborationSession).filter(CollaborationSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check access
    if (session.created_by != current_user.id and 
        current_user.id not in (session.participants or [])):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )
    
    return SessionResponse(
        id=session.id,
        created_by=session.created_by,
        name=session.name,
        description=session.description,
        session_type=session.session_type,
        status=session.status.value,
        participants=session.participants,
        context=session.context,
        messages=session.messages,
        decisions=session.decisions,
        metadata=session.metadata_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        started_at=session.started_at,
        completed_at=session.completed_at
    )


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    update: SessionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a collaboration session"""
    
    session = db.query(CollaborationSession).filter(CollaborationSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check permissions
    if session.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this session"
        )
    
    # Update fields
    if update.name is not None:
        session.name = update.name
    if update.description is not None:
        session.description = update.description
    if update.status is not None:
        try:
            session.status = SessionStatus[update.status.upper()]
            if session.status == SessionStatus.COMPLETED:
                session.completed_at = datetime.now(timezone.utc)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.name for s in SessionStatus]}"
            )
    if update.participants is not None:
        session.participants = update.participants
    if update.context is not None:
        session.context = update.context
    if update.metadata is not None:
        session.metadata_json = update.metadata
    
    db.commit()
    db.refresh(session)
    
    logger.info(f"Session updated: {session_id} by user {current_user.id}")
    
    return SessionResponse(
        id=session.id,
        created_by=session.created_by,
        name=session.name,
        description=session.description,
        session_type=session.session_type,
        status=session.status.value,
        participants=session.participants,
        context=session.context,
        messages=session.messages,
        decisions=session.decisions,
        metadata=session.metadata_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        started_at=session.started_at,
        completed_at=session.completed_at
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a collaboration session"""
    
    session = db.query(CollaborationSession).filter(CollaborationSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check permissions
    if session.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this session"
        )
    
    db.delete(session)
    db.commit()
    
    logger.info(f"Session deleted: {session_id} by user {current_user.id}")


@router.post("/{session_id}/messages", response_model=SessionResponse)
async def add_message(
    session_id: str,
    message: MessageAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a message to a collaboration session"""
    
    session = db.query(CollaborationSession).filter(CollaborationSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check access
    if current_user.id not in (session.participants or []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a participant in this session"
        )
    
    # Add message
    messages = session.messages or []
    new_message = {
        "id": str(uuid.uuid4()),
        "user_id": current_user.id,
        "username": current_user.username,  # Add username for display
        "content": message.content,
        "message_type": message.message_type,
        "metadata": message.metadata,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    messages.append(new_message)
    session.messages = messages
    
    db.commit()
    db.refresh(session)
    
    # Publish to WebSocket channel
    channel = channel_manager.get_session_channel(session_id)
    await publish_to_channel(
        channel=channel,
        message_type="message",
        data={
            "action": "new_message",
            "session_id": session_id,
            "message": new_message
        }
    )
    
    return SessionResponse(
        id=session.id,
        created_by=session.created_by,
        name=session.name,
        description=session.description,
        session_type=session.session_type,
        status=session.status.value,
        participants=session.participants,
        context=session.context,
        messages=session.messages,
        decisions=session.decisions,
        metadata=session.metadata_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        started_at=session.started_at,
        completed_at=session.completed_at
    )
