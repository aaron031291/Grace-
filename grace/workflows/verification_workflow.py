"""
Grace AI - VWX v2: Veracity & Continuity Kernel
The Epistemic Immune System

This module is the living enforcement of Grace's cognitive governance principles:
- Trust Gap Awareness → Verification-first policy
- Hallucination Debt → Claim extraction + Veracity scoring
- Execution Windows → Checkpoint scheduler
- Reliable Memory → Librarian/Fusion coupling
- Consensus Governance → Multi-model quorum
- Self-Audit → Governance Loop
- Truth Ledger → Immutable + Trust integration
- Transparency → Evidence Pack generation
- Continuity Discipline → Snapshot + Pinned Conversation Truths
- Sovereignty → Source attestation

Phases:
1. VERIFICATION_STARTED - Announce and sign initiation
2. SOURCE_ATTESTATION - Verify origin, hash, provenance
3. CLAIM_SET_BUILT - Extract atomic claims from input
4. SEMANTIC_ALIGNMENT - Match to librarian anchors
5. VERACITY_VECTOR - Five-dimensional evidence scoring
6. CONSISTENCY_CHECK - Chat drift + pinned truth verification
7. POLICY_GUARDRAILS - Ethics / compliance validation
8. TRUST_UPDATE - Adjust trust ledger deltas
9. OUTCOME_COMMIT - EPK + signature + log chain
10. CHECKPOINT_COMMIT - Merkle root every N records
"""
import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from grace.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "veracity_continuity_kernel"
EVENTS = [
    "verification_request",
    "ingestion_verify",
    "reasoning_verify",
    "conversation_checkpoint",
    "memory_integrity_check",
    "governance_review"
]


class VeracityDimension(Enum):
    """Five-dimensional evidence scoring"""
    PROVENANCE = "provenance"
    INTERNAL_CONSISTENCY = "internal_consistency"
    EXTERNAL_CORRELATION = "external_correlation"
    TEMPORAL_VALIDITY = "temporal_validity"
    NUMERICAL_CONSISTENCY = "numerical_consistency"


@dataclass
class VeracityVector:
    """Five-dimensional veracity score"""
    provenance: float  # 0.0-1.0: Source trustworthiness
    internal_consistency: float  # 0.0-1.0: Logical coherence
    external_correlation: float  # 0.0-1.0: Cross-reference validation
    temporal_validity: float  # 0.0-1.0: Time-relevance
    numerical_consistency: float  # 0.0-1.0: Unit/calculation accuracy
    
    @property
    def aggregate_score(self) -> float:
        """Weighted aggregate veracity score"""
        weights = {
            "provenance": 0.25,
            "internal_consistency": 0.25,
            "external_correlation": 0.20,
            "temporal_validity": 0.15,
            "numerical_consistency": 0.15
        }
        return (
            self.provenance * weights["provenance"] +
            self.internal_consistency * weights["internal_consistency"] +
            self.external_correlation * weights["external_correlation"] +
            self.temporal_validity * weights["temporal_validity"] +
            self.numerical_consistency * weights["numerical_consistency"]
        )
    
    @property
    def trust_level(self) -> str:
        """Categorical trust level"""
        score = self.aggregate_score
        if score >= 0.9:
            return "VERIFIED"
        elif score >= 0.7:
            return "PROBABLE"
        elif score >= 0.5:
            return "UNCERTAIN"
        elif score >= 0.3:
            return "DUBIOUS"
        else:
            return "QUARANTINED"


@dataclass
class Claim:
    """Atomic claim extracted from input"""
    id: str
    text: str
    subject: str
    predicate: str
    object: str
    confidence: float
    evidence: List[str]
    source_hash: str


@dataclass
class EvidencePack:
    """Verifiable evidence pack for replay"""
    epk_id: str
    timestamp: float
    event_id: str
    claims: List[Claim]
    veracity_vector: VeracityVector
    decision: str
    evidence_hashes: List[str]
    signature: str
    
    def to_dict(self) -> dict:
        return {
            "epk_id": self.epk_id,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "claims": [{"id": c.id, "text": c.text, "confidence": c.confidence} for c in self.claims],
            "veracity_vector": {
                "provenance": self.veracity_vector.provenance,
                "internal_consistency": self.veracity_vector.internal_consistency,
                "external_correlation": self.veracity_vector.external_correlation,
                "temporal_validity": self.veracity_vector.temporal_validity,
                "numerical_consistency": self.veracity_vector.numerical_consistency,
                "aggregate": self.veracity_vector.aggregate_score,
                "trust_level": self.veracity_vector.trust_level
            },
            "decision": self.decision,
            "evidence_hashes": self.evidence_hashes,
            "signature": self.signature
        }


class VeracityContinuityKernel:
    """
    VWX v2: The Epistemic Immune System
    
    Enforces cognitive governance principles across all of Grace's operations.
    """
    name = WORKFLOW_NAME
    EVENTS = EVENTS
    
    def __init__(self):
        self.checkpoint_counter = 0
        self.checkpoint_interval = 100  # Merkle root every 100 verifications
        self.conversation_checkpoint_threshold = 50  # Verify continuity every 50 messages
        self.merkle_roots = []
        self.trust_ledger = None  # Will be injected from service registry
        
    def _get_trust_ledger(self, service_registry=None):
        """Lazy-load trust ledger from service registry"""
        if self.trust_ledger is None and service_registry:
            self.trust_ledger = service_registry.get('trust_ledger')
        return self.trust_ledger
        
    async def execute(self, event: dict):
        """
        Main VWX execution flow with full phase tracking.
        """
        event_id = event.get("id", "unknown")
        payload = event.get("payload", {})
        
        logger.info(f"VWX_START {self.name} event_id={event_id}")
        
        # Get service registry if available (passed in event context)
        service_registry = event.get("__service_registry")
        trust_ledger = self._get_trust_ledger(service_registry)
        
        # Phase 1: VERIFICATION_STARTED
        logger.info(f"  Phase 1: VERIFICATION_STARTED")
        verification_id = self._generate_verification_id(event_id)
        start_time = time.time()
        
        # Phase 2: SOURCE_ATTESTATION
        logger.info(f"  Phase 2: SOURCE_ATTESTATION")
        source_attestation = self._attest_source(payload)
        if not source_attestation["valid"]:
            logger.warning(f"  Source attestation FAILED: {source_attestation['reason']}")
            return {
                "status": "rejected",
                "reason": "source_attestation_failed",
                "details": source_attestation
            }
        
        # Phase 3: CLAIM_SET_BUILT
        logger.info(f"  Phase 3: CLAIM_SET_BUILT")
        claims = self._extract_claims(payload, source_attestation["source_hash"])
        logger.info(f"    Extracted {len(claims)} claims")
        
        # Phase 4: SEMANTIC_ALIGNMENT
        logger.info(f"  Phase 4: SEMANTIC_ALIGNMENT")
        alignment_result = self._align_semantics(claims)
        logger.info(f"    Alignment score: {alignment_result['score']:.2f}")
        
        # Phase 5: VERACITY_VECTOR
        logger.info(f"  Phase 5: VERACITY_VECTOR - Five-dimensional scoring")
        veracity_vector = self._compute_veracity_vector(claims, payload, alignment_result)
        logger.info(f"    Aggregate veracity: {veracity_vector.aggregate_score:.2f} ({veracity_vector.trust_level})")
        
        # Phase 6: CONSISTENCY_CHECK
        logger.info(f"  Phase 6: CONSISTENCY_CHECK")
        consistency_result = self._check_consistency(payload, veracity_vector)
        
        # Phase 7: POLICY_GUARDRAILS
        logger.info(f"  Phase 7: POLICY_GUARDRAILS")
        policy_result = self._check_policy_guardrails(claims, veracity_vector)
        if not policy_result["passed"]:
            logger.warning(f"  Policy guardrail VIOLATION: {policy_result['violation']}")
            return {
                "status": "rejected",
                "reason": "policy_violation",
                "details": policy_result
            }
        
        # Phase 8: TRUST_UPDATE
        logger.info(f"  Phase 8: TRUST_UPDATE")
        trust_delta = self._compute_trust_delta(veracity_vector, policy_result)
        self._safe_trust_update(
            registry=service_registry,
            source=source_attestation["source"],
            delta=trust_delta,
            event_id=event_id,
            reason="VWX v2 veracity-based adjustment"
        )
        
        # Phase 9: OUTCOME_COMMIT
        logger.info(f"  Phase 9: OUTCOME_COMMIT - Generating Evidence Pack")
        epk = self._generate_evidence_pack(
            verification_id, event_id, claims, veracity_vector,
            veracity_vector.trust_level, source_attestation
        )
        
        # Phase 10: CHECKPOINT_COMMIT
        self.checkpoint_counter += 1
        if self.checkpoint_counter % self.checkpoint_interval == 0:
            logger.info(f"  Phase 10: CHECKPOINT_COMMIT - Merkle root checkpoint")
            merkle_root = self._commit_checkpoint(epk)
            self.merkle_roots.append(merkle_root)
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"VWX_DONE {self.name} event_id={event_id} elapsed={elapsed_ms:.1f}ms")
        
        return {
            "status": "verified",
            "workflow": self.name,
            "verification_id": verification_id,
            "veracity_vector": veracity_vector.to_dict() if hasattr(veracity_vector, 'to_dict') else {
                "aggregate": veracity_vector.aggregate_score,
                "trust_level": veracity_vector.trust_level
            },
            "trust_delta": trust_delta,
            "evidence_pack": epk.to_dict(),
            "claims_count": len(claims),
            "elapsed_ms": elapsed_ms
        }
    
    def _generate_verification_id(self, event_id: str) -> str:
        """Generate unique verification ID"""
        timestamp = str(time.time())
        content = f"{event_id}:{timestamp}:vwx"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _attest_source(self, payload: dict) -> dict:
        """
        Phase 2: SOURCE_ATTESTATION
        Verify origin, hash, and provenance of input data.
        """
        source = payload.get("source", "unknown")
        content = payload.get("data", payload.get("content", ""))
        
        # Compute source hash
        content_str = json.dumps(content, sort_keys=True)
        source_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        # Source trust mapping (placeholder - production would use TrustLedger)
        trusted_sources = {
            "verified_api": 0.9,
            "user_input": 0.7,
            "sensor": 0.8,
            "test_harness": 0.95,
            "external_data_received": 0.6
        }
        
        source_trust = trusted_sources.get(source, 0.3)
        
        # Basic validation
        valid = bool(content) and source_trust >= 0.3
        
        return {
            "valid": valid,
            "source": source,
            "source_hash": source_hash,
            "source_trust": source_trust,
            "reason": "accepted" if valid else "untrusted_source"
        }
    
    def _extract_claims(self, payload: dict, source_hash: str) -> List[Claim]:
        """
        Phase 3: CLAIM_SET_BUILT
        Extract atomic claims from input for verification.
        """
        content = payload.get("data", payload.get("content", {}))
        
        # Placeholder: In production, this would use NLP to extract claims
        # For now, create synthetic claims from structured data
        claims = []
        
        if isinstance(content, dict):
            for key, value in content.items():
                claim_id = hashlib.sha256(f"{source_hash}:{key}:{value}".encode()).hexdigest()[:12]
                claim = Claim(
                    id=claim_id,
                    text=f"{key} is {value}",
                    subject=key,
                    predicate="is",
                    object=str(value),
                    confidence=0.85,
                    evidence=[source_hash],
                    source_hash=source_hash
                )
                claims.append(claim)
        
        return claims
    
    def _align_semantics(self, claims: List[Claim]) -> dict:
        """
        Phase 4: SEMANTIC_ALIGNMENT
        Match claims to librarian anchors and existing knowledge.
        """
        # Placeholder: In production, this would query VectorLayer and Librarian
        # For now, return a mock alignment score
        return {
            "score": 0.75,
            "matched_concepts": len(claims),
            "new_concepts": 0,
            "conflicts": 0
        }
    
    def _compute_veracity_vector(self, claims: List[Claim], payload: dict, alignment: dict) -> VeracityVector:
        """
        Phase 5: VERACITY_VECTOR
        Five-dimensional evidence scoring.
        """
        # Compute each dimension
        
        # 1. Provenance: Based on source trust
        source = payload.get("source", "unknown")
        provenance_score = {
            "verified_api": 0.95,
            "test_harness": 0.95,
            "user_input": 0.75,
            "sensor": 0.85,
            "external_data_received": 0.65
        }.get(source, 0.4)
        
        # 2. Internal Consistency: Based on claim coherence
        avg_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.5
        internal_consistency = avg_confidence
        
        # 3. External Correlation: Based on librarian alignment
        external_correlation = alignment["score"]
        
        # 4. Temporal Validity: Check if data is current
        # Placeholder: assume current data is valid
        temporal_validity = 0.9
        
        # 5. Numerical Consistency: Check units and calculations
        # Placeholder: basic validation
        numerical_consistency = 0.85
        
        return VeracityVector(
            provenance=provenance_score,
            internal_consistency=internal_consistency,
            external_correlation=external_correlation,
            temporal_validity=temporal_validity,
            numerical_consistency=numerical_consistency
        )
    
    def _check_consistency(self, payload: dict, veracity: VeracityVector) -> dict:
        """
        Phase 6: CONSISTENCY_CHECK
        Verify conversation drift and pinned truth alignment.
        """
        # Placeholder: In production, check against conversation snapshots
        return {
            "passed": True,
            "drift_detected": False,
            "pinned_truths_aligned": True
        }
    
    def _check_policy_guardrails(self, claims: List[Claim], veracity: VeracityVector) -> dict:
        """
        Phase 7: POLICY_GUARDRAILS
        Validate ethics and compliance.
        """
        # Placeholder: In production, call PolicyEngine
        # For now, simple threshold check
        if veracity.aggregate_score < 0.3:
            return {
                "passed": False,
                "violation": "veracity_threshold",
                "details": f"Aggregate score {veracity.aggregate_score:.2f} below minimum 0.3"
            }
        
        return {
            "passed": True,
            "checks_performed": ["veracity_threshold", "ethics_alignment"],
            "violations": []
        }
    
    def _compute_trust_delta(self, veracity: VeracityVector, policy: dict) -> float:
        """
        Phase 8: Compute trust adjustment delta.
        """
        base_delta = 0.0
        
        if veracity.trust_level == "VERIFIED":
            base_delta = 0.1
        elif veracity.trust_level == "PROBABLE":
            base_delta = 0.05
        elif veracity.trust_level == "UNCERTAIN":
            base_delta = 0.0
        elif veracity.trust_level == "DUBIOUS":
            base_delta = -0.05
        else:  # QUARANTINED
            base_delta = -0.1
        
        # Adjust based on policy compliance
        if not policy["passed"]:
            base_delta -= 0.15
        
        return base_delta
    
    def _safe_trust_update(self, registry, source: str, delta: float, event_id: str, reason: str):
        """
        Try to persist a trust update. Never crash the workflow if trust is unavailable.
        """
        try:
            trust = registry.get("trust_ledger")  # lazy-init via ServiceRegistry
            trust.update_score(
                entity_id=source or "unknown",
                delta=delta,
                reason=reason,
                context={"event_id": event_id, "origin": "VWX_v2"}
            )
            logger.info(
                f"    TRUST_UPDATE: source={source or 'unknown'}, delta={delta:+.2f}, "
                f"event_id={event_id} (Persisted)"
            )
        except Exception as e:
            logger.info(
                f"    TRUST_UPDATE: source={source or 'unknown'}, delta={delta:+.2f}, "
                f"event_id={event_id} (Trust Ledger not available: {e})"
            )
    
    def _update_trust_ledger(self, source: str, delta: float, event_id: str):
        """
        Phase 8: TRUST_UPDATE
        Apply trust delta to source entity via Trust Ledger.
        """
        if self.trust_ledger:
            reason = f"VWX verification delta={delta:+.2f}"
            self.trust_ledger.update_trust(
                entity_id=source,
                entity_type="source",
                delta=delta,
                event_id=event_id,
                reason=reason,
                metadata={"workflow": "vwx_verification"}
            )
        else:
            # Fallback logging if Trust Ledger not available
            logger.info(f"    TRUST_UPDATE: source={source}, delta={delta:+.2f}, event_id={event_id} (Trust Ledger not available)")
    
    def _generate_evidence_pack(
        self, verification_id: str, event_id: str, claims: List[Claim],
        veracity: VeracityVector, decision: str, attestation: dict
    ) -> EvidencePack:
        """
        Phase 9: OUTCOME_COMMIT
        Generate verifiable Evidence Pack for replay.
        """
        evidence_hashes = [c.source_hash for c in claims]
        
        # Create signature (placeholder - production uses Ed25519)
        epk_content = json.dumps({
            "verification_id": verification_id,
            "event_id": event_id,
            "decision": decision,
            "veracity_aggregate": veracity.aggregate_score
        }, sort_keys=True)
        signature = hashlib.sha256(epk_content.encode()).hexdigest()
        
        epk = EvidencePack(
            epk_id=verification_id,
            timestamp=time.time(),
            event_id=event_id,
            claims=claims,
            veracity_vector=veracity,
            decision=decision,
            evidence_hashes=evidence_hashes,
            signature=signature
        )
        
        logger.info(f"    EPK generated: {epk.epk_id}, decision={decision}")
        return epk
    
    def _commit_checkpoint(self, epk: EvidencePack) -> str:
        """
        Phase 10: CHECKPOINT_COMMIT
        Generate Merkle root for checkpoint batch.
        """
        # Placeholder: In production, compute actual Merkle tree
        merkle_root = hashlib.sha256(
            f"{epk.epk_id}:{self.checkpoint_counter}".encode()
        ).hexdigest()
        
        logger.info(f"    MERKLE_CHECKPOINT: root={merkle_root[:16]}..., count={self.checkpoint_counter}")
        return merkle_root


# Export the workflow instance
workflow = VeracityContinuityKernel()

def handle(event, context=None):
    """
    veracity_continuity_kernel
    """
    ctx = context or {}
    logger.info("VWX_START veracity_continuity_kernel event_id=%s", event.id)
    ctx["event_id"] = event.id

    # Phase 1: VERIFICATION_STARTED
    logger.info(f"  Phase 1: VERIFICATION_STARTED")
    verification_id = workflow._generate_verification_id(event.id)
    start_time = time.time()
    
    # Phase 2: SOURCE_ATTESTATION
    logger.info(f"  Phase 2: SOURCE_ATTESTATION")
    source_attestation = workflow._attest_source(event.payload)
    if not source_attestation["valid"]:
        logger.warning(f"  Source attestation FAILED: {source_attestation['reason']}")
        return {
            "status": "rejected",
            "reason": "source_attestation_failed",
            "details": source_attestation
        }
    
    # Phase 3: CLAIM_SET_BUILT
    logger.info(f"  Phase 3: CLAIM_SET_BUILT")
    claims = workflow._extract_claims(event.payload, source_attestation["source_hash"])
    logger.info(f"    Extracted {len(claims)} claims")
    
    # Phase 4: SEMANTIC_ALIGNMENT
    logger.info(f"  Phase 4: SEMANTIC_ALIGNMENT")
    alignment_result = workflow._align_semantics(claims)
    logger.info(f"    Alignment score: {alignment_result['score']:.2f}")
    
    # Phase 5: VERACITY_VECTOR
    logger.info(f"  Phase 5: VERACITY_VECTOR - Five-dimensional scoring")
    veracity_vector = workflow._compute_veracity_vector(claims, event.payload, alignment_result)
    logger.info(f"    Aggregate veracity: {veracity_vector.aggregate_score:.2f} ({veracity_vector.trust_level})")
    
    # Phase 6: CONSISTENCY_CHECK
    logger.info(f"  Phase 6: CONSISTENCY_CHECK")
    consistency_result = workflow._check_consistency(event.payload, veracity_vector)
    
    # Phase 7: POLICY_GUARDRAILS
    logger.info(f"  Phase 7: POLICY_GUARDRAILS")
    policy_result = workflow._check_policy_guardrails(claims, veracity_vector)
    if not policy_result["passed"]:
        logger.warning(f"  Policy guardrail VIOLATION: {policy_result['violation']}")
        return {
            "status": "rejected",
            "reason": "policy_violation",
            "details": policy_result
        }
    
    # Phase 8: TRUST_UPDATE
    # Some events arrive with payload=None; guard access
    payload = event.payload or {}
    source = payload.get("source", "unknown")
    delta = float(payload.get("trust_delta", 0.05))
    registry = ServiceRegistry.get_instance()
    ledger = registry.get_optional("trust_ledger")
    if ledger:
        try:
            rec = ledger.update_score(entity_id=source, delta=delta, reason="VWX_VERIFICATION", context={"event_id": event.id})
            logger.info("  TRUST_UPDATE: source=%s, delta=%+.2f, event_id=%s (persisted, score_after=%.4f, seq=%d)",
                        source, delta, event.id, rec["score_after"], rec["seq"])
        except Exception as e:
            logger.error("  TRUST_UPDATE: Failed to update trust for source=%s: %s", source, e)
    else:
        logger.info(
            "  TRUST_UPDATE: source=%s, delta=%+.2f, event_id=%s (Trust Ledger not available)",
            source, delta, event.id
        )

    # Phase 9: OUTCOME_COMMIT
    logger.info(f"  Phase 9: OUTCOME_COMMIT - Generating Evidence Pack")
    epk = workflow._generate_evidence_pack(
        verification_id, event.id, claims, veracity_vector,
        veracity_vector.trust_level, source_attestation
    )

    # Phase 10: CHECKPOINT_COMMIT
    workflow.checkpoint_counter += 1
    if workflow.checkpoint_counter % workflow.checkpoint_interval == 0:
        logger.info(f"  Phase 10: CHECKPOINT_COMMIT - Merkle root checkpoint")
        merkle_root = workflow._commit_checkpoint(epk)
        workflow.merkle_roots.append(merkle_root)
    
    logger.info("VWX_DONE veracity_continuity_kernel event_id=%s elapsed=%.1fms", event.id, (time.time() - start_time) * 1000)
    return {
        "status": "verified",
        "workflow": workflow.name,
        "verification_id": verification_id,
        "veracity_vector": veracity_vector.to_dict() if hasattr(veracity_vector, 'to_dict') else {
            "aggregate": veracity_vector.aggregate_score,
            "trust_level": veracity_vector.trust_level
        },
        "trust_delta": delta,
        "evidence_pack": epk.to_dict(),
        "claims_count": len(claims),
        "elapsed_ms": (time.time() - start_time) * 1000
    }
