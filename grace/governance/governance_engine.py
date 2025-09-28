"""
Governance Engine - Main orchestrator for Grace governance kernel.
Binds Verification + Unified Logic, manages policies and thresholds, handles events.
"""
import asyncio
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from ..core.contracts import (
    UnifiedDecision, GovernanceSnapshot, Experience, EventType,
    generate_snapshot_id, generate_correlation_id
)
from .verification_engine import VerificationEngine
from .unified_logic import UnifiedLogic


logger = logging.getLogger(__name__)


class GovernanceEngine:
    """
    Main orchestrator for governance decisions. Coordinates verification,
    unified logic, policy enforcement, and event management.
    """
    
    def __init__(self, event_bus, memory_core, verifier: VerificationEngine, 
                 unifier: UnifiedLogic, avn_core=None):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.verifier = verifier
        self.unifier = unifier
        self.avn_core = avn_core
        
        self.instance_id = "governance-primary"
        self.version = "1.0.0"
        self.shadow_mode = False
        self.shadow_instance = None
        
        self.policies = self._load_default_policies()
        self.governance_thresholds = self._load_default_thresholds()
        
        # Subscribe to governance validation events
        asyncio.create_task(self._setup_event_subscriptions())
    
    def _load_default_policies(self) -> Dict[str, Any]:
        """Load default governance policies."""
        return {
            "constitutional_principles": {
                "transparency": True,
                "fairness": True,
                "accountability": True,
                "consistency": True,
                "harm_prevention": True
            },
            "approval_gates": {
                "require_verification": True,
                "require_trust_validation": True,
                "require_constitutional_check": True,
                "allow_shadow_mode": True
            },
            "escalation_rules": {
                "high_risk_decisions": ["deployment", "policy"],
                "require_parliament_review": ["constitutional_change"],
                "auto_escalate_threshold": 0.3  # Confidence below this triggers review
            },
            "audit_settings": {
                "log_all_decisions": True,
                "transparency_level": "democratic_oversight",
                "retention_period_days": 2555  # ~7 years
            }
        }
    
    def _load_default_thresholds(self) -> Dict[str, float]:
        """Load default governance thresholds."""
        return {
            "min_confidence": 0.78,
            "min_trust": 0.72,
            "constitutional_compliance_min": 0.85,
            "shadow_switchover_accuracy_delta": 0.02,
            "rollback_compliance_threshold": 0.98,
            "anomaly_tolerance": 0.1
        }
    
    async def _setup_event_subscriptions(self):
        """Set up event subscriptions for governance."""
        await self.event_bus.subscribe(
            EventType.GOVERNANCE_VALIDATION.value, 
            self._handle_governance_validation
        )
        
        # Subscribe to learning experiences for threshold tuning
        await self.event_bus.subscribe(
            "LEARNING_EXPERIENCE",
            self._handle_learning_feedback
        )
        
        # Subscribe to anomaly alerts from AVN
        if self.avn_core:
            await self.event_bus.subscribe(
                "ANOMALY_DETECTED",
                self._handle_anomaly_alert
            )
    
    async def handle_validation(self, unified_decision: UnifiedDecision, 
                              claims: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main governance validation handler. Applies policies and generates verdict.
        
        Args:
            unified_decision: Decision from unified logic
            claims: List of claims to validate
            context: Additional context for decision
            
        Returns:
            Governance verdict dictionary
        """
        validation_start = datetime.now()
        correlation_id = generate_correlation_id()
        
        try:
            logger.info(f"Processing governance validation {correlation_id} for decision {unified_decision.decision_id}")
            
            # Step 1: Apply constitutional policies
            constitutional_check = await self._apply_constitutional_policies(
                unified_decision, claims, context
            )
            
            # Step 2: Validate against thresholds
            threshold_check = await self._validate_thresholds(unified_decision)
            
            # Step 3: Check for escalation requirements
            escalation_needed = await self._check_escalation_requirements(
                unified_decision, context
            )
            
            # Step 4: Generate final verdict
            verdict = await self._generate_verdict(
                unified_decision, constitutional_check, threshold_check, 
                escalation_needed, correlation_id
            )
            
            # Step 5: Audit and publish result
            await self._audit_and_publish(verdict, validation_start)
            
            # Step 6: Record governance experience
            await self._record_governance_experience(
                unified_decision, verdict, 
                (datetime.now() - validation_start).total_seconds()
            )
            
            return verdict
            
        except Exception as e:
            logger.error(f"Error in governance validation {correlation_id}: {e}")
            error_verdict = {
                "correlation_id": correlation_id,
                "decision_id": unified_decision.decision_id,
                "outcome": "GOVERNANCE_REJECTED",
                "rationale": f"Governance processing error: {str(e)}",
                "confidence": 0.0,
                "trust_score": 0.0,
                "timestamp": datetime.now().isoformat(),
                "requires_manual_review": True
            }
            
            await self._audit_and_publish(error_verdict, validation_start)
            return error_verdict
    
    async def _handle_governance_validation(self, event: Dict[str, Any]):
        """Handle incoming governance validation events."""
        payload = event.get("payload", {})
        correlation_id = event.get("correlation_id")
        
        try:
            # Extract event data
            decision_subject = payload.get("decision_subject")
            inputs = payload.get("inputs", {})
            claims = inputs.get("claims", [])
            context = inputs.get("context", {})
            thresholds = payload.get("thresholds", {})
            
            # Get MLDL consensus if available
            mldl_consensus_id = inputs.get("mldl_consensus_id")
            
            # Create unified decision through the pipeline
            unified_decision = await self.unifier.synthesize_decision(
                topic=f"{decision_subject}_validation",
                inputs=inputs,
                mldl_consensus_id=mldl_consensus_id
            )
            
            # Override thresholds if provided in event
            if thresholds:
                original_thresholds = self.governance_thresholds.copy()
                self.governance_thresholds.update(thresholds)
            
            # Process governance validation
            verdict = await self.handle_validation(unified_decision, claims, context)
            
            # Restore original thresholds if they were overridden
            if thresholds:
                self.governance_thresholds = original_thresholds
            
            # Update correlation in verdict
            verdict["correlation_id"] = correlation_id
            
        except Exception as e:
            logger.error(f"Error handling governance validation event: {e}")
    
    async def _apply_constitutional_policies(self, decision: UnifiedDecision,
                                           claims: List[Any], 
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply constitutional policies to the decision."""
        constitutional_results = {
            "compliant": True,
            "violations": [],
            "compliance_score": 1.0
        }
        
        # Check each constitutional principle
        principles = self.policies["constitutional_principles"]
        
        if principles.get("transparency", False):
            transparency_score = await self._check_transparency(decision, context)
            if transparency_score < 0.8:
                constitutional_results["violations"].append("transparency_insufficient")
                constitutional_results["compliant"] = False
        
        if principles.get("fairness", False):
            fairness_score = await self._check_fairness(decision, claims)
            if fairness_score < 0.8:
                constitutional_results["violations"].append("fairness_concerns")
                constitutional_results["compliant"] = False
        
        if principles.get("accountability", False):
            accountability_score = await self._check_accountability(decision, context)
            if accountability_score < 0.7:
                constitutional_results["violations"].append("accountability_unclear")
                constitutional_results["compliant"] = False
        
        # Calculate overall compliance score
        scores = [
            await self._check_transparency(decision, context),
            await self._check_fairness(decision, claims),
            await self._check_accountability(decision, context),
            await self._check_consistency(decision),
            await self._check_harm_prevention(decision, context)
        ]
        
        constitutional_results["compliance_score"] = sum(scores) / len(scores)
        
        return constitutional_results
    
    async def _check_transparency(self, decision: UnifiedDecision, 
                                context: Dict[str, Any]) -> float:
        """Check transparency requirements."""
        score = 0.8  # Base score
        
        # Check if rationale is provided and sufficiently detailed
        if len(decision.rationale) > 50:
            score += 0.1
        
        # Check if audit trail is available
        if "audit_trail" in context:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _check_fairness(self, decision: UnifiedDecision, claims: List[Any]) -> float:
        """Check fairness and bias concerns."""
        score = 0.7  # Base score
        
        # Look for fairness signals in decision inputs
        fairness_signals = [
            signal for signal in decision.inputs.get("component_signals", [])
            if isinstance(signal, dict) and "fairness" in signal.get("component", "")
        ]
        
        if fairness_signals:
            avg_fairness = sum(signal.get("weight", 0) for signal in fairness_signals) / len(fairness_signals)
            score = max(score, avg_fairness)
        
        return min(score, 1.0)
    
    async def _check_accountability(self, decision: UnifiedDecision, 
                                  context: Dict[str, Any]) -> float:
        """Check accountability requirements."""
        score = 0.6  # Base score
        
        # Check if responsible parties are identified
        if "responsible_party" in context:
            score += 0.2
        
        # Check if review process is defined
        if "review_process" in context:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _check_consistency(self, decision: UnifiedDecision) -> float:
        """Check consistency with precedents."""
        # Look for similar past decisions
        similar_decisions = self.memory_core.get_similar_decisions(decision.topic, limit=5)
        
        if not similar_decisions:
            return 0.7  # Neutral score for novel decisions
        
        # Simple consistency check
        consistent_count = 0
        for past_decision in similar_decisions:
            if past_decision.get("outcome") == decision.recommendation:
                consistent_count += 1
        
        consistency_ratio = consistent_count / len(similar_decisions)
        return max(0.3, consistency_ratio)
    
    async def _check_harm_prevention(self, decision: UnifiedDecision, 
                                   context: Dict[str, Any]) -> float:
        """Check harm prevention measures."""
        score = 0.8  # Base score
        
        # Check for risk assessment
        if "risk_assessment" in context:
            risk_level = context["risk_assessment"].get("level", "medium")
            if risk_level == "low":
                score += 0.1
            elif risk_level == "high":
                score -= 0.2
        
        # Check for mitigation strategies
        if "mitigation_strategies" in context:
            score += 0.1
        
        return max(0.0, min(score, 1.0))
    
    async def _validate_thresholds(self, decision: UnifiedDecision) -> Dict[str, bool]:
        """Validate decision against governance thresholds."""
        return {
            "confidence_sufficient": decision.confidence >= self.governance_thresholds["min_confidence"],
            "trust_sufficient": decision.trust_score >= self.governance_thresholds["min_trust"],
            "meets_all_thresholds": (
                decision.confidence >= self.governance_thresholds["min_confidence"] and
                decision.trust_score >= self.governance_thresholds["min_trust"]
            )
        }
    
    async def _check_escalation_requirements(self, decision: UnifiedDecision,
                                           context: Dict[str, Any]) -> bool:
        """Check if decision requires escalation to parliament or human review."""
        escalation_rules = self.policies["escalation_rules"]
        
        # Check if decision type requires parliament review
        decision_type = context.get("decision_type", "")
        if decision_type in escalation_rules["require_parliament_review"]:
            return True
        
        # Check confidence threshold for auto-escalation
        if decision.confidence < escalation_rules["auto_escalate_threshold"]:
            return True
        
        # Check for high-risk decisions
        if decision_type in escalation_rules["high_risk_decisions"]:
            return True
        
        return False
    
    async def _generate_verdict(self, decision: UnifiedDecision,
                              constitutional_check: Dict[str, Any],
                              threshold_check: Dict[str, bool],
                              escalation_needed: bool,
                              correlation_id: str) -> Dict[str, Any]:
        """Generate final governance verdict."""
        
        # Determine outcome
        if not constitutional_check["compliant"]:
            outcome = "GOVERNANCE_REJECTED"
            rationale = f"Constitutional violations: {', '.join(constitutional_check['violations'])}"
        elif not threshold_check["meets_all_thresholds"]:
            outcome = "GOVERNANCE_REJECTED"
            rationale = f"Insufficient confidence ({decision.confidence:.3f}) or trust ({decision.trust_score:.3f})"
        elif escalation_needed:
            outcome = "GOVERNANCE_NEEDS_REVIEW"
            rationale = "Decision requires human review due to escalation criteria"
        elif decision.recommendation == "approve":
            outcome = "GOVERNANCE_APPROVED"
            rationale = f"Decision approved: {decision.rationale}"
        elif decision.recommendation == "reject":
            outcome = "GOVERNANCE_REJECTED"
            rationale = f"Decision rejected: {decision.rationale}"
        else:  # review
            outcome = "GOVERNANCE_NEEDS_REVIEW"
            rationale = f"Decision requires review: {decision.rationale}"
        
        return {
            "correlation_id": correlation_id,
            "decision_id": decision.decision_id,
            "outcome": outcome,
            "rationale": rationale,
            "confidence": decision.confidence,
            "trust_score": decision.trust_score,
            "constitutional_compliance": constitutional_check["compliance_score"],
            "timestamp": datetime.now().isoformat(),
            "instance_id": self.instance_id,
            "version": self.version,
            "requires_manual_review": escalation_needed or outcome == "GOVERNANCE_NEEDS_REVIEW"
        }
    
    async def _audit_and_publish(self, verdict: Dict[str, Any], 
                               processing_start: datetime):
        """Audit decision and publish governance event."""
        # Store decision in memory
        decision_id = verdict["decision_id"]
        self.memory_core.store_decision(
            # Create a minimal UnifiedDecision for storage
            type('UnifiedDecision', (), {
                'decision_id': decision_id,
                'topic': verdict.get('rationale', ''),
                'inputs': {},
                'recommendation': verdict['outcome'].split('_')[-1].lower(),
                'rationale': verdict['rationale'],
                'confidence': verdict['confidence'],
                'trust_score': verdict['trust_score'],
                'timestamp': datetime.fromisoformat(verdict['timestamp'].replace('Z', '+00:00'))
            })(),
            outcome=verdict['outcome'],
            instance_id=self.instance_id,
            version=self.version
        )
        
        # Publish governance event
        event_type = verdict["outcome"]
        await self.event_bus.publish(event_type, verdict, verdict["correlation_id"])
        
        # Log processing time
        processing_time = (datetime.now() - processing_start).total_seconds()
        logger.info(f"Governance decision {decision_id} processed in {processing_time:.3f}s: {verdict['outcome']}")
    
    async def _record_governance_experience(self, decision: UnifiedDecision,
                                          verdict: Dict[str, Any],
                                          processing_time: float):
        """Record governance experience for meta-learning."""
        # Determine success score
        success_score = verdict["confidence"]
        if verdict["outcome"] == "GOVERNANCE_REJECTED":
            success_score *= 0.5  # Penalize rejections
        elif verdict["outcome"] == "GOVERNANCE_NEEDS_REVIEW":
            success_score *= 0.7  # Partially penalize reviews
        
        experience = Experience(
            type="CONSTITUTIONAL_COMPLIANCE",
            component_id="governance_engine",
            context={
                "principles_triggered": len(self.policies["constitutional_principles"]),
                "processing_time": processing_time,
                "escalation_needed": verdict.get("requires_manual_review", False)
            },
            outcome={
                "compliance_score": verdict.get("constitutional_compliance", 0.0),
                "final_verdict": verdict["outcome"]
            },
            success_score=success_score,
            timestamp=datetime.now()
        )
        
        self.memory_core.store_experience(experience)
        await self.event_bus.publish("LEARNING_EXPERIENCE", experience.to_dict())
    
    async def _handle_learning_feedback(self, event: Dict[str, Any]):
        """Handle learning feedback to update thresholds."""
        payload = event.get("payload", {})
        experience_type = payload.get("type", "")
        
        # Auto-tune thresholds based on experiences
        if experience_type == "CONSTITUTIONAL_COMPLIANCE":
            compliance_score = payload.get("outcome", {}).get("compliance_score", 0.0)
            
            # If compliance is consistently low, lower the threshold slightly
            if compliance_score < 0.6:
                current_min = self.governance_thresholds["constitutional_compliance_min"]
                new_min = max(0.7, current_min - 0.01)  # Don't go below 0.7
                self.governance_thresholds["constitutional_compliance_min"] = new_min
        
        elif experience_type == "VERIFICATION_RESULT":
            confidence = payload.get("outcome", {}).get("confidence", 0.0)
            
            # Adjust confidence thresholds based on verification quality
            if confidence > 0.9:
                # High verification confidence suggests we can be slightly more stringent
                current_min = self.governance_thresholds["min_confidence"]
                new_min = min(0.85, current_min + 0.005)
                self.governance_thresholds["min_confidence"] = new_min
    
    async def _handle_anomaly_alert(self, event: Dict[str, Any]):
        """Handle anomaly alerts from AVN system."""
        payload = event.get("payload", {})
        anomaly_type = payload.get("type", "")
        severity = payload.get("severity", "medium")
        
        if severity == "critical":
            logger.warning(f"Critical anomaly detected: {anomaly_type}")
            # Temporarily lower thresholds to be more conservative
            self.governance_thresholds["min_confidence"] = min(
                0.9, self.governance_thresholds["min_confidence"] + 0.1
            )
            self.governance_thresholds["min_trust"] = min(
                0.85, self.governance_thresholds["min_trust"] + 0.1
            )
    
    # Snapshot and rollback functionality
    async def create_snapshot(self) -> GovernanceSnapshot:
        """Create a snapshot of current governance state."""
        snapshot_id = generate_snapshot_id()
        
        # Calculate state hash
        state_data = {
            "policies": self.policies,
            "thresholds": self.governance_thresholds,
            "version": self.version,
            "instance_id": self.instance_id
        }
        state_hash = hashlib.sha256(
            json.dumps(state_data, sort_keys=True).encode()
        ).hexdigest()
        
        snapshot = GovernanceSnapshot(
            snapshot_id=snapshot_id,
            instance_id=self.instance_id,
            version=self.version,
            policies=self.policies.copy(),
            thresholds=self.governance_thresholds.copy(),
            model_weights=self.unifier.get_current_weights(),
            state_hash=state_hash,
            created_at=datetime.now()
        )
        
        # Store snapshot
        self.memory_core.store_snapshot(snapshot)
        
        # Emit snapshot event
        await self.event_bus.publish(
            EventType.GOVERNANCE_SNAPSHOT_CREATED.value,
            snapshot.to_dict()
        )
        
        logger.info(f"Created governance snapshot {snapshot_id}")
        return snapshot
    
    async def load_snapshot(self, snapshot_id: str) -> bool:
        """Load a governance snapshot."""
        try:
            snapshot_data = self.memory_core.get_latest_snapshot(self.instance_id)
            if not snapshot_data or snapshot_data.get("snapshot_id") != snapshot_id:
                logger.error(f"Snapshot {snapshot_id} not found")
                return False
            
            # Restore state from snapshot
            self.policies = json.loads(snapshot_data["policies_json"])
            self.governance_thresholds = json.loads(snapshot_data["thresholds_json"])
            model_weights = json.loads(snapshot_data["model_weights_json"])
            
            # Update unifier weights
            await self.unifier.update_weights(model_weights)
            
            logger.info(f"Loaded snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            return False
    
    async def start_shadow_mode(self, new_instance_id: str = "governance-shadow"):
        """Start shadow mode with a new governance instance."""
        self.shadow_mode = True
        self.shadow_instance = new_instance_id
        logger.info(f"Started shadow mode with instance {new_instance_id}")
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get current governance status."""
        return {
            "instance_id": self.instance_id,
            "version": self.version,
            "shadow_mode": self.shadow_mode,
            "shadow_instance": self.shadow_instance,
            "policies": self.policies,
            "thresholds": self.governance_thresholds,
            "timestamp": datetime.now().isoformat()
        }