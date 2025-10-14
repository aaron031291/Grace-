"""
Grace MCP - Unified Pushback System

Canonical error handling, immutable logging, event emission, AVN escalation,
and forensic investigation coordination.

Fully integrated with Meta-Loop system for learning from failures.
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict

from grace.core.event_bus import EventBusClient
from grace.governance.governance_engine import GovernanceEngine
from grace.immune.avn_core import AVNClient
from grace.mlt_kernel_ml.memory_orchestrator import MemoryOrchestrator


class PushbackSeverity(str, Enum):
    """Severity levels for pushback events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PushbackCategory(str, Enum):
    """Categories of pushback for routing and learning"""
    GOVERNANCE = "governance"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SERVICE_DEGRADATION = "service_degradation"
    DATA_QUALITY = "data_quality"
    RATE_LIMIT = "rate_limit"
    VECTOR_FAILURE = "vector_failure"
    INDEX_FAILURE = "index_failure"
    VALIDATION = "validation"
    INTERNAL = "internal"
    FORENSIC = "forensic"


@dataclass
class PushbackPayload:
    """Structured pushback information"""
    error_code: str
    category: PushbackCategory
    severity: PushbackSeverity
    message: str
    domain: str
    timestamp: float
    caller_id: Optional[str] = None
    request_id: Optional[str] = None
    request_snapshot_hash: Optional[str] = None
    retry_after_seconds: Optional[int] = None
    remediation_steps: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PushbackHandler:
    """
    Centralized pushback handling with Meta-Loop integration.
    
    Responsibilities:
    - Log pushback to immutable audit log
    - Emit events to TriggerMesh for routing
    - Record observations in O-Loop
    - Record failures in E-Loop for learning
    - Escalate to AVN for healing
    - Create forensic investigations for critical issues
    - Track patterns in F-Loop for systemic improvements
    """
    
    _instance = None
    
    def __init__(self):
        # Lazy-loaded clients
        self._events: Optional[EventBusClient] = None
        self._governance: Optional[GovernanceEngine] = None
        self._avn: Optional[AVNClient] = None
        self._memory: Optional[MemoryOrchestrator] = None
        self._db = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def events(self) -> EventBusClient:
        if not self._events:
            self._events = EventBusClient.get_instance()
        return self._events
    
    @property
    def governance(self) -> GovernanceEngine:
        if not self._governance:
            self._governance = GovernanceEngine.get_instance()
        return self._governance
    
    @property
    def avn(self) -> AVNClient:
        if not self._avn:
            self._avn = AVNClient.get_instance()
        return self._avn
    
    @property
    def memory(self) -> MemoryOrchestrator:
        if not self._memory:
            self._memory = MemoryOrchestrator.get_instance()
        return self._memory
    
    @property
    def db(self):
        if not self._db:
            from grace.ingress_kernel.db.fusion_db import FusionDB
            self._db = FusionDB.get_instance()
        return self._db
    
    async def handle_pushback(self, 
                             payload: PushbackPayload,
                             request_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for handling pushback.
        
        Returns dict with:
        - audit_id: Immutable log ID
        - observation_id: O-Loop observation ID
        - evaluation_id: E-Loop evaluation ID (if applicable)
        - event_types: List of events emitted
        - avn_ticket_id: AVN healing ticket ID (if escalated)
        - forensic_case_id: Forensic investigation ID (if critical)
        """
        result = {
            "audit_id": None,
            "observation_id": None,
            "evaluation_id": None,
            "event_types": [],
            "avn_ticket_id": None,
            "forensic_case_id": None
        }
        
        # 1. Append to immutable audit log
        result["audit_id"] = await self._log_immutable(payload, request_snapshot)
        
        # 2. Record observation in O-Loop
        result["observation_id"] = await self._observe_pushback(payload)
        
        # 3. Record failure evaluation in E-Loop (for learning)
        result["evaluation_id"] = await self._evaluate_failure(payload)
        
        # 4. Emit events based on category and severity
        events_emitted = await self._emit_events(payload, result["audit_id"])
        result["event_types"] = events_emitted
        
        # 5. Escalate to AVN if service degradation or repeated failures
        if await self._should_escalate_to_avn(payload):
            result["avn_ticket_id"] = await self._escalate_to_avn(payload, result)
        
        # 6. Create forensic investigation if critical
        if payload.severity == PushbackSeverity.CRITICAL:
            result["forensic_case_id"] = await self._create_forensic_case(payload, result)
        
        # 7. Update failure patterns (for F-Loop learning)
        await self._update_failure_patterns(payload)
        
        return result
    
    async def _log_immutable(self, 
                            payload: PushbackPayload,
                            request_snapshot: Optional[Dict[str, Any]]) -> str:
        """Append pushback to immutable audit log with hash chain"""
        # Hash request snapshot if provided
        if request_snapshot:
            snapshot_str = json.dumps(request_snapshot, sort_keys=True)
            payload.request_snapshot_hash = hashlib.sha256(snapshot_str.encode()).hexdigest()
        
        # Get previous hash for blockchain-style chaining
        prev_hash = await self._get_last_audit_hash()
        
        audit_record = {
            "action": "pushback",
            "category": payload.category.value,
            "severity": payload.severity.value,
            "error_code": payload.error_code,
            "message": payload.message,
            "domain": payload.domain,
            "caller_id": payload.caller_id,
            "request_id": payload.request_id,
            "request_snapshot_hash": payload.request_snapshot_hash,
            "metadata": payload.metadata or {},
            "timestamp": payload.timestamp,
            "prev_hash": prev_hash
        }
        
        # Compute hash of this record
        record_str = json.dumps(audit_record, sort_keys=True)
        audit_record["hash"] = hashlib.sha256(f"{prev_hash}{record_str}".encode()).hexdigest()
        
        audit_id = await self.db.insert('audit_logs', audit_record)
        return audit_id
    
    async def _get_last_audit_hash(self) -> str:
        """Get hash of last audit log for chaining"""
        result = await self.db.query_one(
            "SELECT hash FROM audit_logs ORDER BY created_at DESC LIMIT 1"
        )
        return result['hash'] if result else "genesis"
    
    async def _observe_pushback(self, payload: PushbackPayload) -> str:
        """Record pushback as observation in O-Loop"""
        observation = {
            "observation_type": "error",
            "source_module": f"mcp.{payload.domain}",
            "observation_data": {
                "error_code": payload.error_code,
                "category": payload.category.value,
                "message": payload.message,
                "metadata": payload.metadata or {}
            },
            "context": {
                "caller_id": payload.caller_id,
                "request_id": payload.request_id,
                "severity": payload.severity.value
            },
            "credibility_score": 1.0,  # Errors are always credible
            "novelty_score": await self._calculate_novelty(payload),
            "observed_at": payload.timestamp,
            "processed": False
        }
        
        obs_id = await self.db.insert('observations', observation)
        return obs_id
    
    async def _calculate_novelty(self, payload: PushbackPayload) -> float:
        """Calculate how novel/unexpected this error is"""
        # Check if we've seen this error_code recently
        recent_count = await self.db.query_scalar("""
            SELECT COUNT(*) 
            FROM observations 
            WHERE observation_type = 'error'
            AND json_extract(observation_data, '$.error_code') = ?
            AND observed_at >= ?
        """, (payload.error_code, time.time() - 3600))  # Last hour
        
        # Novel if first occurrence or rare
        if recent_count == 0:
            return 1.0
        elif recent_count < 3:
            return 0.7
        else:
            return 0.3  # Common error
    
    async def _evaluate_failure(self, payload: PushbackPayload) -> Optional[str]:
        """
        Record failure evaluation in E-Loop for learning.
        This helps F-Loop detect patterns and V-Loop propose fixes.
        """
        # Only create evaluation if we have context about the intended action
        if not payload.request_id:
            return None
        
        evaluation = {
            "action_id": payload.request_id,  # Link to original request
            "intended_outcome": {"success": True},
            "actual_outcome": {
                "success": False,
                "error_code": payload.error_code,
                "error_message": payload.message
            },
            "success": False,
            "performance_metrics": {
                "category": payload.category.value,
                "severity": payload.severity.value
            },
            "side_effects_identified": payload.metadata or {},
            "error_analysis": {
                "error_code": payload.error_code,
                "category": payload.category.value,
                "root_cause_hypothesis": self._hypothesize_root_cause(payload)
            },
            "lessons_learned": self._extract_lessons(payload),
            "confidence_adjustment": -0.05,  # Reduce trust on failures
            "evaluated_at": time.time()
        }
        
        eval_id = await self.db.insert('evaluations', evaluation)
        
        # Adjust trust score for the component
        await self._adjust_trust(payload.domain, -0.02)
        
        return eval_id
    
    def _hypothesize_root_cause(self, payload: PushbackPayload) -> str:
        """Generate root cause hypothesis based on error category"""
        hypotheses = {
            PushbackCategory.GOVERNANCE: "Policy violation or insufficient trust score",
            PushbackCategory.SERVICE_DEGRADATION: "Dependent service down or degraded performance",
            PushbackCategory.VECTOR_FAILURE: "Embedding model unavailable or vector DB error",
            PushbackCategory.INDEX_FAILURE: "Vector upsert failed or collection not found",
            PushbackCategory.DATA_QUALITY: "Invalid input data or schema violation",
            PushbackCategory.RATE_LIMIT: "Request rate exceeded configured limits",
            PushbackCategory.AUTHENTICATION: "Invalid or expired credentials",
            PushbackCategory.AUTHORIZATION: "Insufficient permissions for operation"
        }
        return hypotheses.get(payload.category, "Unknown root cause - requires investigation")
    
    def _extract_lessons(self, payload: PushbackPayload) -> Dict[str, Any]:
        """Extract actionable lessons from failure"""
        lessons = {
            "error_code": payload.error_code,
            "category": payload.category.value,
            "recommendations": []
        }
        
        # Add category-specific recommendations
        if payload.category == PushbackCategory.SERVICE_DEGRADATION:
            lessons["recommendations"].extend([
                "Implement circuit breaker pattern",
                "Add fallback/degraded mode",
                "Increase health check frequency"
            ])
        elif payload.category == PushbackCategory.VECTOR_FAILURE:
            lessons["recommendations"].extend([
                "Add embedding model redundancy",
                "Implement request queuing and batch processing",
                "Consider on-device embedding fallback"
            ])
        elif payload.category == PushbackCategory.GOVERNANCE:
            lessons["recommendations"].extend([
                "Review and clarify policy",
                "Improve trust score through validated actions",
                "Request quorum review if policy is incorrect"
            ])
        
        return lessons
    
    async def _adjust_trust(self, component: str, delta: float):
        """Adjust trust score for component"""
        await self.db.execute("""
            INSERT INTO trust_scores (component, trust_score, previous_score, change_reason, updated_at)
            VALUES (?, 0.5, 0.5, 'initial', ?)
            ON CONFLICT(component) DO UPDATE SET
                previous_score = trust_score,
                trust_score = LEAST(1.0, GREATEST(0.0, trust_score + ?)),
                change_reason = 'failure_adjustment',
                updated_at = ?
        """, (component, time.time(), delta, time.time()))
    
    async def _emit_events(self, payload: PushbackPayload, audit_id: str) -> List[str]:
        """Emit events based on category and severity"""
        events_emitted = []
        
        # Map category to event types
        event_mappings = {
            PushbackCategory.GOVERNANCE: "GOVERNANCE.REJECTED",
            PushbackCategory.SERVICE_DEGRADATION: "SERVICE.DEGRADED",
            PushbackCategory.AUTHENTICATION: "AUTH.FAILURE",
            PushbackCategory.AUTHORIZATION: "AUTH.FORBIDDEN",
            PushbackCategory.VECTOR_FAILURE: "VECTOR.FAILURE",
            PushbackCategory.INDEX_FAILURE: "INDEX.FAILURE",
            PushbackCategory.RATE_LIMIT: "RATE_LIMIT.EXCEEDED",
            PushbackCategory.DATA_QUALITY: "DATA.INVALID",
            PushbackCategory.FORENSIC: "FORENSIC.INVESTIGATION_REQUEST"
        }
        
        event_type = event_mappings.get(payload.category, "PUSHBACK.GENERIC")
        
        # Emit primary event
        await self.events.publish(event_type, {
            "audit_id": audit_id,
            "error_code": payload.error_code,
            "message": payload.message,
            "domain": payload.domain,
            "severity": payload.severity.value,
            "timestamp": payload.timestamp
        })
        events_emitted.append(event_type)
        
        # Emit critical alert if severity is critical
        if payload.severity == PushbackSeverity.CRITICAL:
            await self.events.publish("ALERT.CRITICAL", {
                "audit_id": audit_id,
                "error_code": payload.error_code,
                "domain": payload.domain,
                "message": payload.message
            })
            events_emitted.append("ALERT.CRITICAL")
        
        return events_emitted
    
    async def _should_escalate_to_avn(self, payload: PushbackPayload) -> bool:
        """Determine if error should be escalated to AVN for healing"""
        # Escalate service degradations and repeated failures
        if payload.category in [
            PushbackCategory.SERVICE_DEGRADATION,
            PushbackCategory.VECTOR_FAILURE,
            PushbackCategory.INDEX_FAILURE
        ]:
            return True
        
        # Escalate if error has occurred multiple times recently
        recent_count = await self.db.query_scalar("""
            SELECT COUNT(*)
            FROM audit_logs
            WHERE action = 'pushback'
            AND error_code = ?
            AND timestamp >= ?
        """, (payload.error_code, time.time() - 300))  # Last 5 minutes
        
        return recent_count >= 3  # 3+ occurrences in 5 minutes
    
    async def _escalate_to_avn(self, 
                               payload: PushbackPayload,
                               context: Dict[str, Any]) -> str:
        """Escalate to AVN (Auto-Verification Node) for healing"""
        healing_request = {
            "service": payload.domain,
            "error_code": payload.error_code,
            "category": payload.category.value,
            "severity": payload.severity.value,
            "audit_id": context["audit_id"],
            "observation_id": context["observation_id"],
            "metadata": payload.metadata or {}
        }
        
        ticket_id = await self.avn.request_healing(payload.domain, healing_request)
        
        # Log escalation
        await self.db.insert('meta_loop_escalations', {
            "loop_type": "healing",
            "trigger_observation_id": context["observation_id"],
            "escalation_reason": payload.message,
            "escalation_target": "avn",
            "escalation_data": healing_request,
            "escalated_at": time.time()
        })
        
        return ticket_id
    
    async def _create_forensic_case(self,
                                   payload: PushbackPayload,
                                   context: Dict[str, Any]) -> str:
        """Create forensic investigation case for critical errors"""
        case = {
            "case_type": "critical_error",
            "error_code": payload.error_code,
            "domain": payload.domain,
            "audit_id": context["audit_id"],
            "observation_id": context["observation_id"],
            "evaluation_id": context["evaluation_id"],
            "severity": payload.severity.value,
            "description": payload.message,
            "evidence": {
                "request_snapshot_hash": payload.request_snapshot_hash,
                "metadata": payload.metadata or {}
            },
            "status": "open",
            "created_at": time.time()
        }
        
        # Insert into forensic cases table (assumes it exists)
        try:
            case_id = await self.db.insert('forensic_cases', case)
        except:
            # Fallback: emit event for forensic system
            await self.events.publish("FORENSIC.CASE_CREATED", case)
            case_id = f"forensic_{payload.error_code}_{int(time.time())}"
        
        return case_id
    
    async def _update_failure_patterns(self, payload: PushbackPayload):
        """
        Update or create failure pattern for F-Loop learning.
        Helps system learn from repeated failures.
        """
        # Check if pattern exists
        pattern = await self.db.query_one("""
            SELECT * FROM outcome_patterns
            WHERE pattern_type = 'recurring_failure'
            AND action_type = ?
            AND json_extract(conditions, '$.error_code') = ?
        """, (payload.domain, payload.error_code))
        
        if pattern:
            # Update existing pattern
            await self.db.execute("""
                UPDATE outcome_patterns
                SET frequency = frequency + 1,
                    last_occurrence = ?,
                    confidence = LEAST(1.0, confidence + 0.05)
                WHERE pattern_id = ?
            """, (time.time(), pattern['pattern_id']))
        else:
            # Create new pattern
            await self.db.insert('outcome_patterns', {
                "pattern_type": "recurring_failure",
                "action_type": payload.domain,
                "conditions": {
                    "error_code": payload.error_code,
                    "category": payload.category.value
                },
                "outcome": {
                    "success": False,
                    "typical_message": payload.message
                },
                "frequency": 1,
                "confidence": 0.3,
                "actionable_insight": self._extract_lessons(payload)["recommendations"],
                "first_occurrence": time.time(),
                "last_occurrence": time.time()
            })


# --- Convenience functions for common pushback scenarios ---

async def handle_governance_rejection(
    domain: str,
    reason: str,
    caller_id: str,
    request_id: str,
    request_snapshot: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Handle governance rejection pushback"""
    handler = PushbackHandler.get_instance()
    
    payload = PushbackPayload(
        error_code="GOVERNANCE_REJECTED",
        category=PushbackCategory.GOVERNANCE,
        severity=PushbackSeverity.HIGH,
        message=f"Governance rejected: {reason}",
        domain=domain,
        timestamp=time.time(),
        caller_id=caller_id,
        request_id=request_id,
        remediation_steps=[
            "Review policy requirements",
            "Improve trust score through validated actions",
            "Contact governance team if policy is incorrect"
        ]
    )
    
    result = await handler.handle_pushback(payload, request_snapshot)
    
    return {
        "http_status": 403,
        "body": {
            "error": "governance_rejected",
            "message": reason,
            "audit_id": result["audit_id"],
            "remediation": payload.remediation_steps
        }
    }


async def handle_service_unavailable(
    domain: str,
    service_name: str,
    caller_id: str,
    request_id: str,
    retry_after: int = 30
) -> Dict[str, Any]:
    """Handle service unavailable pushback"""
    handler = PushbackHandler.get_instance()
    
    payload = PushbackPayload(
        error_code="SERVICE_UNAVAILABLE",
        category=PushbackCategory.SERVICE_DEGRADATION,
        severity=PushbackSeverity.ERROR,
        message=f"Service {service_name} is unavailable",
        domain=domain,
        timestamp=time.time(),
        caller_id=caller_id,
        request_id=request_id,
        retry_after_seconds=retry_after,
        metadata={"service": service_name}
    )
    
    result = await handler.handle_pushback(payload)
    
    return {
        "http_status": 503,
        "body": {
            "error": "service_unavailable",
            "service": service_name,
            "retry_after": retry_after,
            "audit_id": result["audit_id"],
            "avn_ticket": result.get("avn_ticket_id")
        }
    }


async def handle_index_failure(
    domain: str,
    record_id: str,
    error: Exception,
    caller_id: str,
    request_id: str,
    attempt: int = 1
) -> Dict[str, Any]:
    """Handle vector index failure pushback"""
    handler = PushbackHandler.get_instance()
    
    payload = PushbackPayload(
        error_code="INDEX_FAILURE",
        category=PushbackCategory.INDEX_FAILURE,
        severity=PushbackSeverity.HIGH if attempt >= 3 else PushbackSeverity.ERROR,
        message=f"Index failure for {record_id}: {str(error)}",
        domain=domain,
        timestamp=time.time(),
        caller_id=caller_id,
        request_id=request_id,
        metadata={
            "record_id": record_id,
            "attempt": attempt,
            "error_type": type(error).__name__
        }
    )
    
    result = await handler.handle_pushback(payload)
    
    return {
        "http_status": 500,
        "body": {
            "error": "index_failure",
            "record_id": record_id,
            "attempt": attempt,
            "audit_id": result["audit_id"],
            "avn_ticket": result.get("avn_ticket_id"),
            "forensic_case": result.get("forensic_case_id")
        }
    }


async def retry_with_backoff(
    fn: callable,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    *args,
    **kwargs
):
    """
    Retry function with exponential backoff.
    Logs failures to pushback system.
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            if attempt >= max_attempts:
                # Final attempt failed - log to pushback
                handler = PushbackHandler.get_instance()
                await handler.handle_pushback(PushbackPayload(
                    error_code="RETRY_EXHAUSTED",
                    category=PushbackCategory.INTERNAL,
                    severity=PushbackSeverity.ERROR,
                    message=f"Retry exhausted after {max_attempts} attempts: {str(e)}",
                    domain="retry_handler",
                    timestamp=time.time(),
                    metadata={
                        "function": fn.__name__,
                        "attempts": max_attempts,
                        "final_error": str(e)
                    }
                ))
                raise
            
            # Wait before next attempt
            await asyncio.sleep(delay)
            delay *= 2
    
    raise last_error
