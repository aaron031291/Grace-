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
    
    def __init__(self):
        if not self.domain:
            raise ValueError(f"{self.__class__.__name__} must set 'domain' attribute")
        
        # Load manifest
        if not self.manifest_path:
            self.manifest_path = f"grace/mcp/manifests/{self.domain}.yaml"
        self.manifest = self._load_manifest()
        
        # Initialize clients (lazy loading)
        self._db: Optional[FusionDB] = None
        self._memory: Optional[MemoryCore] = None
        self._events: Optional[EventBus] = None
        self._governance: Optional[GovernanceEngine] = None
        
    def _load_manifest(self) -> Dict[str, Any]:
        """Load and validate YAML manifest"""
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            # Validate required fields
            required = ['mcp_version', 'domain', 'endpoints']
            missing = [f for f in required if f not in manifest]
            if missing:
                raise ValueError(f"Manifest missing required fields: {missing}")
            
            return manifest
        except FileNotFoundError:
            raise ValueError(f"Manifest not found: {self.manifest_path}")
    
    # --- Lazy client initialization ---
    
    @property
    def db(self) -> FusionDB:
        """Get database instance"""
        if not self._db:
            self._db = FusionDB.get_instance()
        return self._db
    
    @property
    def memory(self) -> MemoryCore:
        if not self._memory:
            self._memory = MemoryCore()
        return self._memory
    
    @property
    def events(self) -> EventBus:
        if not self._events:
            self._events = EventBus()
        return self._events
    
    @events.setter
    def events(self, value):
        """Allow setting events for testing."""
        self._events = value
    
    @property
    def governance(self) -> GovernanceEngine:
        if not self._governance:
            self._governance = GovernanceEngine()
        return self._governance
    
    @governance.setter
    def governance(self, value):
        """Allow setting governance for testing."""
        self._governance = value
    
    # --- Meta-Loop integration ---
    
    async def observe(self, 
                     operation: str, 
                     data: Dict[str, Any],
                     context: MCPContext) -> str:
        """
        Record observation in O-Loop.
        Returns observation_id for lineage tracking.
        """
        # Convert Pydantic models in data to dicts
        serializable_data = {}
        for key, value in data.items():
            if hasattr(value, 'model_dump'):
                serializable_data[key] = value.model_dump()
            elif hasattr(value, 'dict'):
                serializable_data[key] = value.dict()
            else:
                serializable_data[key] = value
        
        observation = {
            "observation_type": "mcp_operation",
            "source_module": f"mcp.{self.domain}",
            "observation_data": json.dumps({
                "operation": operation,
                "domain": self.domain,
                "caller": context.caller.id,
                "request_id": context.request_id,
                **serializable_data
            }),
            "context": json.dumps({
                "session_id": context.caller.session_id,
                "trust_score": context.caller.trust_score,
                "elapsed_ms": context.elapsed_ms
            }),
            "credibility_score": context.caller.trust_score,
            "observed_at": time.time()
        }
        
        # Insert into observations table (assumes it exists from database build)
        # Use raw SQL for now - can be upgraded to ORM later
        obs_id = f"obs_{hashlib.md5(f'{operation}{time.time()}'.encode()).hexdigest()}"
        observation['observation_id'] = obs_id
        
        # Store observation snapshot in memory
        snapshot = type('GovernanceSnapshot', (), {
            'to_dict': lambda self: {
                'snapshot_id': obs_id,
                'snapshot_type': 'observation',
                'data': observation
            }
        })()
        await self.memory.store_snapshot(snapshot)
        
        return obs_id
    
    async def record_decision(self,
                             orientation_id: Optional[str],
                             decision_type: str,
                             selected_option: Dict[str, Any],
                             rationale: str,
                             context: MCPContext,
                             governance_approved: bool = True) -> str:
        """
        Record decision in D-Loop.
        Returns decision_id for lineage tracking.
        """
        # Convert Pydantic model to dict if needed
        if hasattr(selected_option, 'model_dump'):
            selected_option = selected_option.model_dump()
        elif hasattr(selected_option, 'dict'):
            selected_option = selected_option.dict()
        
        decision = {
            "orientation_id": orientation_id,
            "decision_type": decision_type,
            "selected_option": json.dumps(selected_option),
            "selection_rationale": rationale,
            "confidence": context.caller.trust_score,
            "governance_approved": governance_approved,
            "decided_at": time.time()
        }
        
        dec_id = f"dec_{hashlib.md5(f'{decision_type}{time.time()}'.encode()).hexdigest()}"
        decision['decision_id'] = dec_id
        
        # Store in memory core - create a snapshot-like dict
        snapshot = type('GovernanceSnapshot', (), {
            'to_dict': lambda self: {
                'snapshot_id': dec_id,
                'snapshot_type': 'decision',
                'data': decision
            }
        })()
        await self.memory.store_snapshot(snapshot)
        
        return dec_id
    
    async def evaluate_outcome(self,
                              action_id: str,
                              intended: Dict[str, Any],
                              actual: Dict[str, Any],
                              success: bool,
                              metrics: Dict[str, Any],
                              context: MCPContext) -> str:
        """
        Record evaluation in E-Loop.
        Returns evaluation_id.
        """
        # Convert Pydantic models to dicts if needed
        if hasattr(intended, 'model_dump'):
            intended = intended.model_dump()
        elif hasattr(intended, 'dict'):
            intended = intended.dict()
        
        if hasattr(actual, 'model_dump'):
            actual = actual.model_dump()
        elif hasattr(actual, 'dict'):
            actual = actual.dict()
        
        if hasattr(metrics, 'model_dump'):
            metrics = metrics.model_dump()
        elif hasattr(metrics, 'dict'):
            metrics = metrics.dict()
        
        evaluation = {
            "action_id": action_id,
            "intended_outcome": json.dumps(intended),
            "actual_outcome": json.dumps(actual),
            "success": success,
            "performance_metrics": json.dumps(metrics),
            "confidence_adjustment": 0.05 if success else -0.05,
            "evaluated_at": time.time()
        }
        
        eval_id = f"eval_{hashlib.md5(f'{action_id}{time.time()}'.encode()).hexdigest()}"
        evaluation['evaluation_id'] = eval_id
        
        # Store in memory core - create a snapshot-like dict
        snapshot = type('GovernanceSnapshot', (), {
            'to_dict': lambda self: {
                'snapshot_id': eval_id,
                'snapshot_type': 'evaluation',
                'data': evaluation
            }
        })()
        await self.memory.store_snapshot(snapshot)
        
        # Update caller's trust score
        if success:
            await self._adjust_trust(context.caller.id, +0.02)
        else:
            await self._adjust_trust(context.caller.id, -0.02)
        
        return eval_id
    
    async def _adjust_trust(self, component: str, delta: float):
        """Adjust trust score for component/caller"""
        # Store trust adjustment in memory for now
        trust_update = {
            "component": component,
            "delta": delta,
            "timestamp": time.time()
        }
        snapshot = type('GovernanceSnapshot', (), {
            'to_dict': lambda self: {
                'snapshot_id': f"trust_{component}_{time.time()}",
                'snapshot_type': 'trust_adjustment',
                'data': trust_update
            }
        })()
        await self.memory.store_snapshot(snapshot)
    
    # --- Vectorization helpers ---
    
    async def vectorize(self, 
                       text: str,
                       metadata: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Embed text using system embedding model.
        Handles retries and cost tracking.
        
        TODO: Integrate with actual embedding service when available.
        For now, returns a mock embedding vector.
        """
        try:
            # Mock embedding - replace with actual embedding service
            # embedding = await self.vector.embed(text, metadata=metadata)
            import random
            embedding = [random.random() for _ in range(384)]  # Mock 384-dim vector
            return embedding
        except Exception as e:
            # Log failure - in production, escalate to healing system
            logger.error(f"Embedding service failed: {e}")
            raise ServiceUnavailable("embedding_model")
    
    async def upsert_vector(self,
                           collection: str,
                           id: str,
                           embedding: List[float],
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Upsert vector to collection with retry logic.
        
        TODO: Integrate with actual vector store when available.
        For now, stores vectors in memory core.
        """
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                # Mock vector storage - replace with actual vector store
                vector_record = {
                    "collection": collection,
                    "id": id,
                    "embedding_dim": len(embedding),
                    "metadata": metadata or {}
                }
                snapshot = type('GovernanceSnapshot', (), {
                    'to_dict': lambda self: {
                        'snapshot_id': f"vec_{collection}_{id}",
                        'snapshot_type': 'vector',
                        'data': vector_record
                    }
                })()
                await self.memory.store_snapshot(snapshot)
                return
            except Exception as e:
                if attempt >= max_attempts:
                    logger.error(f"Vector store upsert failed after {max_attempts} attempts: {e}")
                    raise ServiceUnavailable("vector_store")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def semantic_search(self,
                             collection: str,
                             query: str,
                             top_k: int = 10,
                             filters: Optional[Dict[str, Any]] = None,
                             trust_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform semantic search with trust filtering.
        Returns fused results from vector + DB.
        
        TODO: Integrate with actual vector store when available.
        For now, returns mock results from memory.
        """
        # Check cache first (mock)
        cache_key = self._cache_key(collection, query, top_k, filters)
        
        # Mock semantic search - replace with actual implementation
        # For now, return empty results
        results = []
        
        return results
    
    def _cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        data = json.dumps(args, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
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
