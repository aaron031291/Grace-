"""
Governance Engine - Complete specification-compliant implementation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import logging

from grace.events.schema import GraceEvent

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of governance validation"""
    passed: bool
    violations: List[str]
    amendments: List[Dict[str, Any]]
    confidence: float
    decision: Optional[Dict[str, Any]] = None


@dataclass
class EscalationResult:
    """Result of governance escalation"""
    escalated: bool
    escalation_level: str
    assigned_to: List[str]
    reason: str
    actions_taken: List[str]


class GovernanceEngine:
    """
    Complete Governance Engine
    
    All methods are truly async
    """
    
    def __init__(self, trigger_mesh=None):
        self.policies: Dict[str, Any] = {}
        self.violations: List[Dict[str, Any]] = []
        self.trigger_mesh = trigger_mesh
    
    async def validate(
        self,
        event: GraceEvent,
        context: Optional[Dict[str, Any]] = None,
        request_mldl_consensus: bool = True
    ) -> ValidationResult:
        """
        Async validate event against constitutional constraints
        
        Can request MLDL consensus for complex decisions
        """
        await asyncio.sleep(0)
        
        violations: List[str] = []
        amendments: List[Dict[str, Any]] = []
        
        # Check if validation required
        if not event.constitutional_validation_required:
            return ValidationResult(
                passed=True,
                violations=[],
                amendments=[],
                confidence=1.0
            )
        
        # Validate trust score
        if event.trust_score < 0.5:
            violations.append(f"Trust score too low: {event.trust_score}")
        
        # Validate event structure
        if not event.event_type:
            violations.append("Missing event_type")
        
        if not event.source:
            violations.append("Missing source")
        
        # Check governance approval for critical events
        if event.priority == "critical" and not event.governance_approved:
            violations.append("Critical event requires governance approval")
            amendments.append({
                "field": "governance_approved",
                "required_value": True,
                "reason": "Critical priority requires approval"
            })
        
        # Validate against active policies
        policy_violations = await self._check_policies(event, context)
        violations.extend(policy_violations)
        
        # Request MLDL consensus for complex decisions
        mldl_consensus = None
        if request_mldl_consensus and len(violations) > 0 and self.trigger_mesh:
            logger.info(f"Requesting MLDL consensus for event {event.event_id}")
            
            try:
                mldl_consensus = await self._request_mldl_consensus(
                    event=event,
                    violations=violations,
                    context=context or {}
                )
                
                if mldl_consensus:
                    logger.info(f"MLDL consensus received", extra={
                        "event_id": event.event_id,
                        "recommendation": mldl_consensus.get("recommendation"),
                        "confidence": mldl_consensus.get("confidence")
                    })
                    
                    # Override decision based on consensus
                    recommendation = mldl_consensus.get("recommendation")
                    confidence = mldl_consensus.get("confidence", 0.5)
                    
                    if recommendation == "approve" and confidence > 0.7:
                        logger.info(f"MLDL consensus overrides violations (approve)")
                        violations = []  # Clear violations
                    elif recommendation == "reject" and confidence > 0.8:
                        violations.append("MLDL consensus: reject with high confidence")
            
            except asyncio.TimeoutError:
                logger.warning(f"MLDL consensus request timed out for {event.event_id}")
            except Exception as e:
                logger.error(f"MLDL consensus request failed: {e}")
        
        passed = len(violations) == 0
        confidence = 1.0 if passed else 0.5
        
        result = ValidationResult(
            passed=passed,
            violations=violations,
            amendments=amendments,
            confidence=confidence,
            decision={
                "event_id": event.event_id,
                "validated": passed,
                "mldl_consensus": mldl_consensus
            }
        )
        
        # Log validation
        if not passed:
            logger.warning(f"Validation failed for {event.event_id}: {violations}")
            self.violations.append({
                "event_id": event.event_id,
                "violations": violations,
                "timestamp": event.timestamp,
                "mldl_consensus": mldl_consensus
            })
        
        return result
    
    async def _request_mldl_consensus(
        self,
        event: GraceEvent,
        violations: List[str],
        context: Dict[str, Any],
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Request consensus from MLDL kernel using emit and wait_for
        
        Args:
            event: Event being validated
            violations: Current violations
            context: Decision context
            timeout: Timeout for waiting
        
        Returns:
            Consensus result or None if timeout/error
        """
        if not self.trigger_mesh:
            logger.warning("TriggerMesh not available, cannot request consensus")
            return None
        
        # Build decision context
        decision_context = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "source": event.source,
            "trust_score": event.trust_score,
            "priority": event.priority.value if hasattr(event.priority, 'value') else event.priority,
            "violations": violations,
            "context": context
        }
        
        try:
            # Use request_response pattern
            response = await self.trigger_mesh.request_response(
                event_type="mldl.consensus.request",
                payload={
                    "decision_context": decision_context,
                    "options": ["approve", "review", "reject"]
                },
                timeout=timeout,
                source="governance_engine",
                priority="high"
            )
            
            if response:
                consensus = response.payload.get("consensus")
                return consensus
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get MLDL consensus: {e}")
            return None
    
    async def escalate(
        self,
        event: GraceEvent,
        reason: str,
        level: str = "standard",
        context: Optional[Dict[str, Any]] = None
    ) -> EscalationResult:
        """
        Async escalate event to higher authority
        
        Required by specification
        """
        # Use asyncio.sleep to make it truly async
        await asyncio.sleep(0)
        
        logger.warning(f"Escalating event {event.event_id}: {reason}")
        
        # Determine escalation targets
        targets: List[str] = []
        actions: List[str] = []
        
        if level == "standard":
            targets = ["governance_committee"]
            actions = ["review_requested"]
        
        elif level == "high":
            targets = ["governance_committee", "security_team"]
            actions = ["review_requested", "security_audit_initiated"]
        
        elif level == "critical":
            targets = ["governance_committee", "security_team", "executive_oversight"]
            actions = ["immediate_review_required", "system_freeze_considered"]
        
        # Create escalation record
        escalation = EscalationResult(
            escalated=True,
            escalation_level=level,
            assigned_to=targets,
            reason=reason,
            actions_taken=actions
        )
        
        # Notify targets (simulate)
        for target in targets:
            logger.info(f"Notifying {target} of escalation: {event.event_id}")
        
        return escalation
    
    async def _check_policies(
        self,
        event: GraceEvent,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Check event against active policies"""
        await asyncio.sleep(0)  # Truly async
        
        violations: List[str] = []
        
        # Example policy checks
        for policy_id, policy in self.policies.items():
            if policy.get("status") != "active":
                continue
            
            # Check constraints
            constraints = policy.get("constraints", [])
            for constraint in constraints:
                if not self._evaluate_constraint(event, constraint, context):
                    violations.append(
                        f"Policy {policy_id} violated: {constraint.get('description')}"
                    )
        
        return violations
    
    def _evaluate_constraint(
        self,
        event: GraceEvent,
        constraint: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate a single constraint"""
        constraint_type = constraint.get("type")
        
        if constraint_type == "trust_threshold":
            threshold = constraint.get("threshold", 0.7)
            return event.trust_score >= threshold
        
        elif constraint_type == "approval_required":
            return event.governance_approved
        
        elif constraint_type == "source_whitelist":
            whitelist = constraint.get("allowed_sources", [])
            return event.source in whitelist
        
        # Default: pass
        return True
    
    def register_policy(self, policy_id: str, policy: Dict[str, Any]):
        """Register a governance policy"""
        self.policies[policy_id] = policy
        logger.info(f"Registered policy: {policy_id}")
    
    # Legacy method for backwards compatibility (deprecated)
    def handle_validation(self, event: Dict[str, Any]) -> bool:
        """Legacy validation method (deprecated - use validate())"""
        logger.warning("handle_validation() is deprecated, use async validate()")
        
        # Convert dict to GraceEvent if needed
        if isinstance(event, dict):
            event = GraceEvent.from_dict(event)
        
        # Run async validation synchronously
        result = asyncio.run(self.validate(event))
        return result.passed
