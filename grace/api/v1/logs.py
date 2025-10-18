"""
Immutable Logs API endpoints with semantic search
"""

from datetime import datetime, timezone
from typing import List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from grace.auth.models import User
from grace.auth.dependencies import get_current_user, require_role
from grace.mtl.immutable_logs import ImmutableLogs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["Immutable Logs"])

# Global immutable logs instance (singleton)
_logs_instance = None


def get_logs_instance() -> ImmutableLogs:
    """Get or create immutable logs instance"""
    global _logs_instance
    if _logs_instance is None:
        _logs_instance = ImmutableLogs(storage_path="./data/immutable_logs")
    return _logs_instance


# Pydantic schemas
class LogEntryCreate(BaseModel):
    operation_type: str = Field(..., description="Type of operation")
    action: dict = Field(..., description="Action details")
    result: dict = Field(..., description="Result of the action")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    severity: str = Field("info", description="Log severity: info, warning, error, critical")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class LogEntryResponse(BaseModel):
    cid: str
    operation_type: str
    actor: str
    action: dict
    result: dict
    metadata: dict
    severity: str
    tags: List[str]
    timestamp: str
    signature: str
    search_score: Optional[float] = None
    relevance: Optional[str] = None
    trust_score: Optional[float] = None


class LogSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Semantic search query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    severity_filter: Optional[str] = Field(None, description="Filter by severity")
    operation_type_filter: Optional[str] = Field(None, description="Filter by operation type")
    tag_filter: Optional[List[str]] = Field(None, description="Filter by tags")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")


class TrustSearchRequest(BaseModel):
    min_trust: float = Field(0.0, ge=0.0, le=1.0, description="Minimum trust score")
    max_trust: float = Field(1.0, ge=0.0, le=1.0, description="Maximum trust score")
    k: int = Field(100, ge=1, le=1000, description="Maximum results")


class LogStatistics(BaseModel):
    total_entries: int
    chain_valid: bool
    chain_error: Optional[str]
    severity_breakdown: dict
    operation_breakdown: dict
    indexed_entries: int
    latest_entry: Optional[str]


@router.post("", response_model=LogEntryResponse, status_code=status.HTTP_201_CREATED)
async def create_log_entry(
    log_entry: LogEntryCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new immutable log entry with automatic vectorization
    
    The entry is:
    1. Hashed to create a Content ID (CID)
    2. Chained with previous entry via signature
    3. Vectorized and indexed for semantic search
    4. Persisted to disk
    """
    logs = get_logs_instance()
    
    try:
        cid = logs.log_constitutional_operation(
            operation_type=log_entry.operation_type,
            actor=f"user:{current_user.id}",
            action=log_entry.action,
            result=log_entry.result,
            metadata=log_entry.metadata,
            severity=log_entry.severity,
            tags=log_entry.tags
        )
        
        # Retrieve the created entry
        entry = logs.get_log_entry(cid)
        
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created log entry"
            )
        
        logger.info(f"Created log entry: {cid[:16]}... by user {current_user.username}")
        
        return LogEntryResponse(
            cid=entry["cid"],
            operation_type=entry["operation_type"],
            actor=entry["actor"],
            action=entry["action"],
            result=entry["result"],
            metadata=entry["metadata"],
            severity=entry["severity"],
            tags=entry["tags"],
            timestamp=entry["timestamp"],
            signature=entry["signature"]
        )
        
    except Exception as e:
        logger.error(f"Error creating log entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create log entry: {str(e)}"
        )


@router.get("/{cid}", response_model=LogEntryResponse)
async def get_log_entry(
    cid: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific log entry by its CID"""
    logs = get_logs_instance()
    
    entry = logs.get_log_entry(cid)
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Log entry not found"
        )
    
    # Calculate trust score
    trust_score = logs._calculate_trust_score(entry)
    
    return LogEntryResponse(
        cid=entry["cid"],
        operation_type=entry["operation_type"],
        actor=entry["actor"],
        action=entry["action"],
        result=entry["result"],
        metadata=entry["metadata"],
        severity=entry["severity"],
        tags=entry["tags"],
        timestamp=entry["timestamp"],
        signature=entry["signature"],
        trust_score=trust_score
    )


@router.post("/search/semantic", response_model=List[LogEntryResponse])
async def semantic_search_logs(
    search: LogSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Perform semantic search on immutable logs
    
    Uses vector embeddings to find logs similar to the query text.
    Supports filtering by severity, operation type, and tags.
    """
    logs = get_logs_instance()
    
    try:
        results = logs.semantic_search_logs(
            query=search.query,
            k=search.k,
            severity_filter=search.severity_filter,
            operation_type_filter=search.operation_type_filter,
            tag_filter=search.tag_filter,
            min_similarity=search.min_similarity
        )
        
        logger.info(
            f"Semantic log search by {current_user.username}: "
            f"query='{search.query}', found {len(results)} results"
        )
        
        return [
            LogEntryResponse(
                cid=entry["cid"],
                operation_type=entry["operation_type"],
                actor=entry["actor"],
                action=entry["action"],
                result=entry["result"],
                metadata=entry["metadata"],
                severity=entry["severity"],
                tags=entry["tags"],
                timestamp=entry["timestamp"],
                signature=entry["signature"],
                search_score=entry.get("search_score"),
                relevance=entry.get("relevance")
            )
            for entry in results
        ]
        
    except Exception as e:
        logger.error(f"Error in semantic log search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/search/trust", response_model=List[LogEntryResponse])
async def search_logs_by_trust(
    search: TrustSearchRequest,
    current_user: User = Depends(require_role(["admin", "superuser"]))
):
    """
    Search logs by trust score range (admin only)
    
    Trust score is calculated based on:
    - Chain integrity
    - Entry age
    - Actor verification
    - Severity
    """
    logs = get_logs_instance()
    
    try:
        results = logs.search_by_trust_score(
            min_trust=search.min_trust,
            max_trust=search.max_trust,
            k=search.k
        )
        
        logger.info(
            f"Trust-based log search by {current_user.username}: "
            f"range=[{search.min_trust}, {search.max_trust}], found {len(results)} results"
        )
        
        return [
            LogEntryResponse(
                cid=entry["cid"],
                operation_type=entry["operation_type"],
                actor=entry["actor"],
                action=entry["action"],
                result=entry["result"],
                metadata=entry["metadata"],
                severity=entry["severity"],
                tags=entry["tags"],
                timestamp=entry["timestamp"],
                signature=entry["signature"],
                trust_score=entry.get("trust_score")
            )
            for entry in results
        ]
        
    except Exception as e:
        logger.error(f"Error in trust-based log search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/verify/chain")
async def verify_chain_integrity(
    current_user: User = Depends(require_role(["admin", "superuser"]))
):
    """
    Verify the integrity of the entire log chain (admin only)
    
    Checks that all entries are properly chained and signed.
    """
    logs = get_logs_instance()
    
    is_valid, error = logs.verify_chain_integrity()
    
    logger.info(f"Chain integrity verification by {current_user.username}: valid={is_valid}")
    
    return {
        "valid": is_valid,
        "error": error,
        "total_entries": len(logs.log_entries),
        "verified_by": current_user.username,
        "verified_at": datetime.now(timezone.utc).isoformat()
    }


@router.get("/stats", response_model=LogStatistics)
async def get_log_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get statistics about the immutable logs"""
    logs = get_logs_instance()
    stats = logs.get_statistics()
    
    return LogStatistics(
        total_entries=stats["total_entries"],
        chain_valid=stats["chain_valid"],
        chain_error=stats.get("chain_error"),
        severity_breakdown=stats["severity_breakdown"],
        operation_breakdown=stats["operation_breakdown"],
        indexed_entries=stats["indexed_entries"],
        latest_entry=stats.get("latest_entry")
    )
