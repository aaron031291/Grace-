"""
Grace Meta-Control Protocol (MCP) - Base Classes and Decorators

Provides the foundation for manifest-driven MCP handlers with automatic:
- Authentication & authorization
- Governance validation
- Meta-Loop observation and decision tracking
- Event emission
- Immutable audit logging
- Trust scoring and provenance
- Rate limiting and cost tracking
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Callable, List, Type
from functools import wraps
from dataclasses import dataclass
from enum import Enum

import yaml
from pydantic import BaseModel, Field, ConfigDict
from fastapi import HTTPException, Request, Response

# Grace system imports
from grace.core.event_bus import EventBus
from grace.contracts.governed_decision import GovernedDecision
from grace.contracts.governed_request import GovernedRequest
from grace.governance.governance_engine import GovernanceEngine
from grace.core.memory_core import MemoryCore
from grace.ingress_kernel.db.fusion_db import FusionDB

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity levels for pushback and audit logs"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OwnerType(str, Enum):
    """Who owns and manages the domain"""
    SYSTEM = "system"  # MCP executes full pipeline
    AGENT = "agent"    # MCP routes to agent MTL


@dataclass
class Caller:
    """Authenticated caller identity"""
    id: str
    name: str
    roles: List[str]
    trust_score: float = 0.5
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def has_role(self, role: str) -> bool:
        return role in self.roles
    
    def has_any_role(self, roles: List[str]) -> bool:
        return any(r in self.roles for r in roles)


@dataclass
class MCPContext:
    """Context passed to MCP handlers"""
    caller: Caller
    request: Request
    manifest: Dict[str, Any]
    endpoint_config: Dict[str, Any]
    start_time: float
    request_id: str
    correlation_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    
    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class MCPResponse(BaseModel):
    """Standard MCP response wrapper"""
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    audit_id: Optional[str] = None
    trust_score: Optional[float] = None
    provenance: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"id": "obs_123", "created_at": "2025-10-14T08:00:00Z"},
                "audit_id": "audit_456",
                "trust_score": 0.85,
                "metadata": {"latency_ms": 45}
            }
        }
    )


class MCPException(Exception):
    """Base exception for MCP operations"""
    def __init__(self, 
                 message: str,
                 error_code: str,
                 http_status: int = 500,
                 severity: Severity = Severity.MEDIUM,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.severity = severity
        self.metadata = metadata or {}


class GovernanceRejection(MCPException):
    """Raised when governance validation fails"""
    def __init__(self, reason: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Governance rejected: {reason}",
            error_code="GOVERNANCE_REJECTED",
            http_status=403,
            severity=Severity.HIGH,
            metadata=metadata
        )


class ServiceUnavailable(MCPException):
    """Raised when dependent service is down"""
    def __init__(self, service_name: str, retry_after: int = 30):
        super().__init__(
            message=f"Service {service_name} is unavailable",
            error_code="SERVICE_UNAVAILABLE",
            http_status=503,
            severity=Severity.MEDIUM,
            metadata={"service": service_name, "retry_after": retry_after}
        )


class BaseMCP:
    """
    Base class for all MCP handlers.
    
    Provides:
    - Manifest loading and validation
    - Client connections (DB, Vector, Events, Governance)
    - Automatic observation recording (O-Loop)
    - Decision tracking (D-Loop)
    - Evaluation recording (E-Loop)
    - Immutable audit logging
    - Event emission
    - Trust scoring
    """
    
    domain: str = None  # Override in subclass
    manifest_path: str = None  # Override or auto-derive
    owner: OwnerType = OwnerType.SYSTEM
    
    def __init__(self, embedding_service=None, vector_store=None):
        """
        Initialize MCP with embedding service and vector store
        
        Args:
            embedding_service: EmbeddingService instance from grace.embeddings
            vector_store: VectorStore instance from grace.vectorstore
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
        # Initialize services if not provided
        if not self.embedding_service:
            try:
                from grace.embeddings.service import EmbeddingService
                self.embedding_service = EmbeddingService()
                logger.info("Initialized default embedding service")
            except Exception as e:
                logger.error(f"Could not initialize embedding service: {e}")
                self.embedding_service = None
        
        if not self.vector_store:
            try:
                from grace.vectorstore.service import VectorStoreService
                dimension = self.embedding_service.dimension if self.embedding_service else 384
                self.vector_store = VectorStoreService(
                    dimension=dimension,
                    index_path="./data/mcp_vectors.bin"
                )
                logger.info("Initialized default vector store")
            except Exception as e:
                logger.error(f"Could not initialize vector store: {e}")
                self.vector_store = None
    
    def vectorize(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector using production embedding service
        
        Args:
            text: Text to vectorize
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.embedding_service:
            logger.warning("No embedding service available, returning zero vector")
            return np.zeros(384, dtype=np.float32)
        
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.embedding_service.dimension, dtype=np.float32)
        
        try:
            embedding = self.embedding_service.embed_text(text)
            logger.debug(f"Vectorized text (length={len(text)}, dim={len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error vectorizing text: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_service.dimension, dtype=np.float32)
    
    def upsert_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Insert or update a vector in the vector store with retries
        
        Args:
            vector_id: Unique identifier for the vector
            vector: Embedding vector
            metadata: Associated metadata
            
        Returns:
            Success status
        """
        if not self.vector_store:
            logger.error("No vector store available")
            return False
        
        if vector is None or len(vector) == 0:
            logger.error("Invalid vector provided")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add timestamp to metadata
                metadata = metadata.copy()
                metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()
                metadata["vector_id"] = vector_id
                
                # Check if vector already exists
                existing = self.vector_store.get_store().get_by_id(vector_id)
                
                if existing:
                    # Delete old vector
                    self.vector_store.get_store().delete([vector_id])
                    logger.debug(f"Deleted existing vector: {vector_id}")
                
                # Insert new/updated vector
                self.vector_store.get_store().add_vectors(
                    vectors=[vector],
                    metadata=[metadata],
                    ids=[vector_id]
                )
                
                logger.info(f"Upserted vector: {vector_id} (attempt {attempt + 1})")
                return True
                
            except Exception as e:
                logger.warning(f"Upsert attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to upsert vector {vector_id} after {max_retries} attempts")
                    return False
                import time
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return False
    
    def semantic_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity with error handling
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of search results with scores and metadata
        """
        if not self.embedding_service or not self.vector_store:
            logger.error("Embedding service or vector store not available")
            return []
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Vectorize query
                query_vector = self.vectorize(query)
                
                if np.all(query_vector == 0):
                    logger.warning("Query vectorization produced zero vector")
                    return []
                
                # Search vector store
                results = self.vector_store.get_store().search(
                    query_vector=query_vector,
                    k=k,
                    filter=filter
                )
                
                # Format results
                formatted_results = []
                for vector_id, score, metadata in results:
                    formatted_results.append({
                        "id": vector_id,
                        "score": float(score),
                        "metadata": metadata,
                        "relevance": self._calculate_relevance(score)
                    })
                
                logger.info(f"Semantic search completed: {len(formatted_results)} results")
                return formatted_results
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Search failed after {max_retries} attempts")
                    return []
                import time
                time.sleep(0.5 * (attempt + 1))
        
        return []
    
    def _calculate_relevance(self, score: float) -> str:
        """Calculate relevance category from similarity score"""
        if score > 0.9:
            return "very_high"
        elif score > 0.75:
            return "high"
        elif score > 0.5:
            return "medium"
        elif score > 0.25:
            return "low"
        else:
            return "very_low"
    
    # --- Event emission ---
    
    async def emit_event(self, 
                        event_type: str,
                        payload: Dict[str, Any],
                        context: MCPContext):
        """
        Emit event to TriggerMesh/EventBus.
        Automatically adds context and metadata.
        """
        event = {
            "event_type": event_type,
            "domain": self.domain,
            "timestamp": time.time(),
            "caller": context.caller.id,
            "request_id": context.request_id,
            "correlation_id": context.correlation_id,
            "payload": payload
        }
        
        await self.events.publish(event_type, event)
    
    # --- Audit logging ---
    
    async def audit_log(self,
                       action: str,
                       payload: Dict[str, Any],
                       context: MCPContext,
                       severity: Severity = Severity.MEDIUM) -> str:
        """
        Append to immutable audit log.
        Returns audit_id.
        """
        # Convert Pydantic models to dict if needed
        if hasattr(payload, 'model_dump'):
            payload = payload.model_dump()
        elif hasattr(payload, 'dict'):
            payload = payload.dict()
        
        record = {
            "action": action,
            "domain": self.domain,
            "caller": context.caller.id,
            "request_id": context.request_id,
            "severity": severity.value,
            "payload": payload,
            "timestamp": time.time()
        }
        
        # Compute hash chain (blockchain-style)
        prev_hash = await self._get_last_audit_hash()
        record_str = json.dumps(record, sort_keys=True)
        record['hash'] = hashlib.sha256(f"{prev_hash}{record_str}".encode()).hexdigest()
        record['prev_hash'] = prev_hash
        
        audit_id = await self.db.insert('audit_logs', record)
        return audit_id
    
    async def _get_last_audit_hash(self) -> str:
        """Get hash of last audit log entry for chaining"""
        result = await self.db.query_one(
            "SELECT hash FROM audit_logs ORDER BY timestamp DESC LIMIT 1"
        )
        return result['hash'] if result else "genesis"
    
    # --- Governance integration ---
    
    async def validate_governance(self,
                                 operation: str,
                                 payload: Dict[str, Any],
                                 context: MCPContext,
                                 endpoint_config: Dict[str, Any]) -> bool:
        """
        Validate operation through governance kernel.
        Raises GovernanceRejection if rejected.
        """
        gov_config = endpoint_config.get('governance', {})
        if not gov_config.get('require_validation', False):
            return True  # Skip validation if not required
        
        # Build governance request
        request = GovernedRequest(
            operation=operation,
            domain=self.domain,
            payload=payload,
            caller_id=context.caller.id,
            caller_trust_score=context.caller.trust_score,
            sensitivity=gov_config.get('sensitivity', 'medium'),
            request_id=context.request_id
        )
        
        # Validate through governance engine
        decision: GovernedDecision = await self.governance.validate(request)
        
        if not decision.approved:
            # Record rejection in D-Loop and T-Loop
            await self.record_decision(
                orientation_id=None,
                decision_type="governance_check",
                selected_option={"approved": False},
                rationale=decision.reason,
                context=context,
                governance_approved=False
            )
            
            # Log ethics violation if critical
            if decision.severity == "critical":
                await self.db.insert('ethics_violations', {
                    "violation_type": decision.violation_type,
                    "severity": "critical",
                    "component": f"mcp.{self.domain}",
                    "action_id": None,
                    "violation_details": {"reason": decision.reason},
                    "detected_at": time.time()
                })
            
            raise GovernanceRejection(decision.reason, metadata=decision.metadata)
        
        return True


def mcp_endpoint(manifest: str, endpoint: str):
    """
    Decorator for MCP endpoint handlers.
    
    Automatically handles:
    - Authentication
    - Governance validation
    - Observation recording (O-Loop)
    - Event emission
    - Audit logging
    - Error handling with pushback
    - Response wrapping
    
    Usage:
        @mcp_endpoint(manifest="patterns.yaml", endpoint="create")
        async def create_pattern(self, request: PatternCreateRequest, context: MCPContext):
            # Your business logic here
            return {"id": "pattern_123"}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self: BaseMCP, request: BaseModel, context: MCPContext):
            start_time = time.time()
            obs_id = None
            audit_id = None
            
            try:
                # 1. Record observation (O-Loop)
                obs_id = await self.observe(
                    operation=func.__name__,
                    data=request.model_dump() if hasattr(request, 'model_dump') else request.dict(),
                    context=context
                )
                
                # 2. Governance validation
                endpoint_config = self._get_endpoint_config(endpoint)
                await self.validate_governance(
                    operation=func.__name__,
                    payload=request.model_dump() if hasattr(request, 'model_dump') else request.dict(),
                    context=context,
                    endpoint_config=endpoint_config
                )
                
                # 3. Execute handler
                result = await func(self, request, context)
                
                # Convert result to dict if it's a Pydantic model
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else (result.dict() if hasattr(result, 'dict') else result)
                
                # 4. Emit success event
                event_type = endpoint_config.get('side_effects', {}).get('emit_event')
                if event_type:
                    await self.emit_event(
                        event_type=f"{self.domain.upper()}.{event_type}",
                        payload={"result": result_dict, "observation_id": obs_id},
                        context=context
                    )
                
                # 5. Audit log
                if endpoint_config.get('side_effects', {}).get('log_immutable'):
                    audit_id = await self.audit_log(
                        action=func.__name__,
                        payload={"request": request.model_dump() if hasattr(request, 'model_dump') else request.dict(), "result": result_dict},
                        context=context,
                        severity=Severity.LOW
                    )
                
                # 6. Evaluate outcome (E-Loop)
                elapsed_ms = (time.time() - start_time) * 1000
                await self.evaluate_outcome(
                    action_id=obs_id,
                    intended={"success": True},
                    actual={"success": True, "result": result_dict},
                    success=True,
                    metrics={"latency_ms": elapsed_ms},
                    context=context
                )
                
                # 7. Wrap response
                return MCPResponse(
                    success=True,
                    data=result,
                    audit_id=audit_id,
                    trust_score=context.caller.trust_score,
                    metadata={"latency_ms": elapsed_ms, "observation_id": obs_id}
                )
                
            except GovernanceRejection as e:
                # Already logged in validate_governance
                return MCPResponse(
                    success=False,
                    error=e.message,
                    error_code=e.error_code,
                    metadata=e.metadata
                )
            
            except Exception as e:
                # Log failure
                elapsed_ms = (time.time() - start_time) * 1000
                audit_id = await self.audit_log(
                    action=f"{func.__name__}_error",
                    payload={"error": str(e), "request": request.model_dump() if hasattr(request, 'model_dump') else request.dict()},
                    context=context,
                    severity=Severity.HIGH
                )
                
                # Record failure evaluation
                if obs_id:
                    await self.evaluate_outcome(
                        action_id=obs_id,
                        intended={"success": True},
                        actual={"success": False, "error": str(e)},
                        success=False,
                        metrics={"latency_ms": elapsed_ms},
                        context=context
                    )
                
                # Emit error event
                await self.emit_event(
                    event_type=f"{self.domain.upper()}.ERROR",
                    payload={"error": str(e), "audit_id": audit_id},
                    context=context
                )
                
                return MCPResponse(
                    success=False,
                    error=str(e),
                    error_code="INTERNAL_ERROR",
                    audit_id=audit_id,
                    metadata={"latency_ms": elapsed_ms}
                )
        
        return wrapper
    return decorator
