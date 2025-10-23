"""
Policy management API endpoints
"""

from typing import List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid
import logging

from grace.auth.models import User
from grace.auth.dependencies import get_current_user, require_role
from grace.database import get_db
from grace.governance.models import Policy, PolicyStatus, PolicyType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/policies", tags=["Policies"])


# Pydantic schemas
class PolicyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    policy_type: str
    rules: Optional[List[dict]] = []
    constraints: Optional[dict] = {}
    metadata: Optional[dict] = {}
    requires_approval: bool = True
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None


class PolicyUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[str] = None
    rules: Optional[List[dict]] = None
    constraints: Optional[dict] = None
    metadata: Optional[dict] = None
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None


class PolicyResponse(BaseModel):
    id: str
    created_by: str
    name: str
    description: str
    policy_type: str
    status: str
    rules: Optional[List[dict]]
    constraints: Optional[dict]
    metadata: Optional[dict]
    version: str
    requires_approval: bool
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    effective_date: Optional[datetime]
    expiry_date: Optional[datetime]
    
    class Config:
        from_attributes = True


@router.post("", response_model=PolicyResponse, status_code=status.HTTP_201_CREATED)
async def create_policy(
    policy: PolicyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new policy"""
    
    # Validate policy type
    try:
        policy_type_enum = PolicyType[policy.policy_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid policy type. Must be one of: {[t.name for t in PolicyType]}"
        )
    
    db_policy = Policy(
        id=str(uuid.uuid4()),
        created_by=current_user.id,
        name=policy.name,
        description=policy.description,
        policy_type=policy_type_enum,
        status=PolicyStatus.DRAFT,
        rules=policy.rules,
        constraints=policy.constraints,
        metadata_json=policy.metadata,
        requires_approval=policy.requires_approval,
        effective_date=policy.effective_date,
        expiry_date=policy.expiry_date
    )
    
    db.add(db_policy)
    db.commit()
    db.refresh(db_policy)
    
    logger.info(f"Policy created: {db_policy.id} by user {current_user.id}")
    
    return PolicyResponse(
        id=db_policy.id,
        created_by=db_policy.created_by,
        name=db_policy.name,
        description=db_policy.description,
        policy_type=db_policy.policy_type.value,
        status=db_policy.status.value,
        rules=db_policy.rules,
        constraints=db_policy.constraints,
        metadata=db_policy.metadata_json,
        version=db_policy.version,
        requires_approval=db_policy.requires_approval,
        approved_by=db_policy.approved_by,
        approved_at=db_policy.approved_at,
        created_at=db_policy.created_at,
        updated_at=db_policy.updated_at,
        effective_date=db_policy.effective_date,
        expiry_date=db_policy.expiry_date
    )


@router.get("", response_model=List[PolicyResponse])
async def list_policies(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    policy_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List policies with optional filtering"""
    
    query = db.query(Policy)
    
    if policy_type:
        try:
            type_enum = PolicyType[policy_type.upper()]
            query = query.filter(Policy.policy_type == type_enum)
        except KeyError:
            pass
    
    if status:
        try:
            status_enum = PolicyStatus[status.upper()]
            query = query.filter(Policy.status == status_enum)
        except KeyError:
            pass
    
    policies = query.order_by(Policy.created_at.desc()).offset(skip).limit(limit).all()
    
    return [
        PolicyResponse(
            id=p.id,
            created_by=p.created_by,
            name=p.name,
            description=p.description,
            policy_type=p.policy_type.value,
            status=p.status.value,
            rules=p.rules,
            constraints=p.constraints,
            metadata=p.metadata_json,
            version=p.version,
            requires_approval=p.requires_approval,
            approved_by=p.approved_by,
            approved_at=p.approved_at,
            created_at=p.created_at,
            updated_at=p.updated_at,
            effective_date=p.effective_date,
            expiry_date=p.expiry_date
        )
        for p in policies
    ]


@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(
    policy_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific policy"""
    
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Policy not found"
        )
    
    return PolicyResponse(
        id=policy.id,
        created_by=policy.created_by,
        name=policy.name,
        description=policy.description,
        policy_type=policy.policy_type.value,
        status=policy.status.value,
        rules=policy.rules,
        constraints=policy.constraints,
        metadata=policy.metadata_json,
        version=policy.version,
        requires_approval=policy.requires_approval,
        approved_by=policy.approved_by,
        approved_at=policy.approved_at,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
        effective_date=policy.effective_date,
        expiry_date=policy.expiry_date
    )


@router.put("/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str,
    update: PolicyUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a policy"""
    
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Policy not found"
        )
    
    # Check permissions (only creator or admin can update)
    if policy.created_by != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this policy"
        )
    
    # Update fields
    if update.name is not None:
        policy.name = update.name
    if update.description is not None:
        policy.description = update.description
    if update.status is not None:
        try:
            policy.status = PolicyStatus[update.status.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.name for s in PolicyStatus]}"
            )
    if update.rules is not None:
        policy.rules = update.rules
    if update.constraints is not None:
        policy.constraints = update.constraints
    if update.metadata is not None:
        policy.metadata_json = update.metadata
    if update.effective_date is not None:
        policy.effective_date = update.effective_date
    if update.expiry_date is not None:
        policy.expiry_date = update.expiry_date
    
    db.commit()
    db.refresh(policy)
    
    logger.info(f"Policy updated: {policy_id} by user {current_user.id}")
    
    return PolicyResponse(
        id=policy.id,
        created_by=policy.created_by,
        name=policy.name,
        description=policy.description,
        policy_type=policy.policy_type.value,
        status=policy.status.value,
        rules=policy.rules,
        constraints=policy.constraints,
        metadata=policy.metadata_json,
        version=policy.version,
        requires_approval=policy.requires_approval,
        approved_by=policy.approved_by,
        approved_at=policy.approved_at,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
        effective_date=policy.effective_date,
        expiry_date=policy.expiry_date
    )


@router.delete("/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_policy(
    policy_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a policy"""
    
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Policy not found"
        )
    
    # Check permissions
    if policy.created_by != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this policy"
        )
    
    db.delete(policy)
    db.commit()
    
    logger.info(f"Policy deleted: {policy_id} by user {current_user.id}")


@router.post("/{policy_id}/approve", response_model=PolicyResponse)
async def approve_policy(
    policy_id: str,
    current_user: User = Depends(require_role(["admin", "superuser"])),
    db: Session = Depends(get_db)
):
    """Approve a policy (admin only)"""
    
    policy = db.query(Policy).filter(Policy.id == policy_id).first()
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Policy not found"
        )
    
    policy.approved_by = current_user.id
    policy.approved_at = datetime.now(timezone.utc)
    policy.status = PolicyStatus.ACTIVE
    
    db.commit()
    db.refresh(policy)
    
    logger.info(f"Policy approved: {policy_id} by user {current_user.id}")
    
    return PolicyResponse(
        id=policy.id,
        created_by=policy.created_by,
        name=policy.name,
        description=policy.description,
        policy_type=policy.policy_type.value,
        status=policy.status.value,
        rules=policy.rules,
        constraints=policy.constraints,
        metadata=policy.metadata_json,
        version=policy.version,
        requires_approval=policy.requires_approval,
        approved_by=policy.approved_by,
        approved_at=policy.approved_at,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
        effective_date=policy.effective_date,
        expiry_date=policy.expiry_date
    )
