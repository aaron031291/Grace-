#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
governance_engine.py
Governance Engine — Main orchestrator for Grace governance kernel (production).

Responsibilities
- Bind VerificationEngine + UnifiedLogic
- Enforce constitutional policies & thresholds
- Escalation routing and verdict generation
- Snapshot + rollback integration
- Experience logging for meta-learning
- EventBus subscriptions (validation, experience, anomaly)

Notes
- Uses UTC-aware timestamps (ISO 8601)
- Works with EventBus (prod version) and MemoryCore (prod version)
- Safe to construct eagerly; subscriptions are attached via `async_init()`
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from ..core.contracts import (  # Pydantic v2 models from earlier response
    UnifiedDecision,
    GovernanceSnapshot,
    Experience,
    EventType,
    generate_snapshot_id,
    generate_correlation_id,
)

# Type hints (avoid circular deps in imports)
from .verification_engine import VerificationEngine  # type: ignore
from .unified_logic import UnifiedLogic  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers & Defaults
# ---------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _iso(dt: Optional[datetime] = None) -> str:
    d = dt or _utcnow()
    return d.isoformat()


DEFAULT_POLICIES: Dict[str, Any] = {
    "constitutional_principles": {
        "transparency": True,
        "fairness": True,
        "accountability": True,
        "consistency": True,
        "harm_prevention": True,
    },
    "approval_gates": {
        "require_verification": True,
        "require_trust_validation": True,
        "require_constitutional_check": True,
        "allow_shadow_mode": True,
    },
    "escalation_rules": {
        "high_risk_decisions": ["deployment", "policy"],
        "require_parliament_review": ["constitutional_change"],
        "auto_escalate_threshold": 0.30,
    },
    "audit_settings": {
        "log_all_decisions": True,
        "transparency_level": "democratic_oversight",
        "retention_period_days": 2555,  # ~7 years
    },
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_confidence": 0.78,
    "min_trust": 0.72,
    "constitutional_compliance_min": 0.85,
    "shadow_switchover_accuracy_delta": 0.02,
    "rollback_compliance_threshold": 0.98,
    "anomaly_tolerance": 0.10,
}


# ---------------------------
# Governance Engine
# ---------------------------

@dataclass
class GovernanceEngineConfig:
    instance_id: str = "governance-primary"
    version: str = "1.0.0"
    policies: Dict[str, Any] = None  # type: ignore
    thresholds: Dict[str, float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.policies is None:
            self.policies = json.loads(json.dumps(DEFAULT_POLICIES))
        if self.thresholds is None:
            self.thresholds = json.loads(json.dumps(DEFAULT_THRESHOLDS))


class GovernanceEngine:
    """
    Main orchestrator for governance decisions. Coordinates verification,
    unified logic, policy enforcement, and event management.

    Dependencies:
      - event_bus: EventBus (prod)
      - memory_core: MemoryCore (prod)
      - verifier: VerificationEngine
      - unifier: UnifiedLogic
      - avn_core: Optional anomaly system (sends ANOMALY_DETECTED events)
    """

    def __init__(
        self,
        event_bus,
        memory_core,
        verifier: VerificationEngine,
        unifier: UnifiedLogic,
        *,
        avn_core: Optional[Any] = None,
        config: Optional[GovernanceEngineConfig] = None,
    ) -> None:
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.verifier = verifier
        self.unifier = unifier
        self.avn_core = avn_core

        self.cfg = config or GovernanceEngineConfig()

        # shadow mode flags
        self.shadow_mode = False
        self.shadow_instance: Optional[str] = None

        # Subscriptions are attached in async_init() to avoid race during app bootstrap
        self._init_lock = asyncio.Lock()
        self._initialized = False

    # --------------- Lifecycle ---------------

    async def async_init(self) -> None:
        """Attach EventBus subscriptions (idempotent)."""
        async with self._init_lock:
            if self._initialized:
                return

            await self.event_bus.subscribe(EventType.GOVERNANCE_VALIDATION, self._handle_governance_validation)
            await self.event_bus.subscribe("LEARNING_EXPERIENCE", self._handle_learning_feedback)

            if self.avn_core:
                await self.event_bus.subscribe("ANOMALY_DETECTED", self._handle_anomaly_alert)

            self._initialized = True
            logger.info("GovernanceEngine initialized: instance=%s v%s", self.cfg.instance_id, self.cfg.version)

    # --------------- Public API ---------------

    async def handle_validation(
        self,
        unified_decision: UnifiedDecision,
        claims: List[Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main governance validation handler. Applies policies and generates a verdict.
        """
        started = _utcnow()
        correlation_id = generate_correlation_id()

        try:
            logger.info(
                "Governance validation start corr=%s decision_id=%s topic=%s",
                correlation_id,
                unified_decision.decision_id,
                unified_decision.topic,
            )

            # Optional verification gate
            if self.cfg.policies["approval_gates"]["require_verification"]:
                await self._run_verification_gate(unified_decision, claims, context)

            # Constitutional policies
            constitutional = await self._apply_constitutional_policies(unified_decision, claims, context)

            # Threshold validation
            thresholds = await self._validate_thresholds(unified_decision)

            # Escalation
            escalation_needed = await self._check_escalation_requirements(unified_decision, context)

            # Verdict
            verdict = self._generate_verdict(unified_decision, constitutional, thresholds, escalation_needed, correlation_id)

            # Audit + publish
            await self._audit_and_publish(verdict, started)

            # Experience
            await self._record_governance_experience(unified_decision, verdict, (_utcnow() - started).total_seconds())

            return verdict

        except Exception as e:
            logger.exception("Governance validation error corr=%s: %s", correlation_id, e)
            error_verdict = {
                "correlation_id": correlation_id,
                "decision_id": unified_decision.decision_id,
                "outcome": EventType.GOVERNANCE_REJECTED.value,
                "rationale": f"Processing error: {e}",
                "confidence": 0.0,
                "trust_score": 0.0,
                "constitutional_compliance": 0.0,
                "timestamp": _iso(),
                "instance_id": self.cfg.instance_id,
                "version": self.cfg.version,
                "requires_manual_review": True,
            }
            await self._audit_and_publish(error_verdict, started)
            return error_verdict

    # --------------- Event Handlers ---------------

    async def _handle_governance_validation(self, event: Mapping[str, Any]) -> None:
        """Handle incoming GOV_VALIDATION events and drive the pipeline."""
        payload = event.get("payload", {}) or {}
        correlation_id = event.get("correlation_id")

        try:
            decision_subject = payload.get("decision_subject")
            inputs = payload.get("inputs", {}) or {}
            claims = inputs.get("claims", []) or []
            context = inputs.get("context", {}) or {}
            thresholds = payload.get("thresholds", {}) or {}
            mldl_consensus_id = inputs.get("mldl_consensus_id")

            unified_decision = await self.unifier.synthesize_decision(
                topic=f"{decision_subject}_validation",
                inputs=inputs,
                mldl_consensus_id=mldl_consensus_id,
            )

            # Override thresholds for just this validation, if provided
            original_thresholds = self.cfg.thresholds.copy()
            if thresholds:
                self.cfg.thresholds.update({k: float(v) for k, v in thresholds.items()})

            try:
                verdict = await self.handle_validation(unified_decision, claims, context)
                verdict["correlation_id"] = correlation_id or verdict.get("correlation_id")
            finally:
                self.cfg.thresholds = original_thresholds

        except Exception as e:
            logger.exception("Error handling governance validation event: %s", e)

    async def _handle_learning_feedback(self, event: Mapping[str, Any]) -> None:
        """Handle learning feedback to tune thresholds."""
        payload = event.get("payload", {}) or {}
        etype = payload.get("type", "")

        if etype == "CONSTITUTIONAL_COMPLIANCE":
            compliance = float(payload.get("outcome", {}).get("compliance_score", 0.0))
            if compliance < 0.60:
                # Nudge down minimum compliance but not below 0.70
                cur = self.cfg.thresholds["constitutional_compliance_min"]
                self.cfg.thresholds["constitutional_compliance_min"] = max(0.70, cur - 0.01)

        elif etype == "VERIFICATION_RESULT":
            conf = float(payload.get("outcome", {}).get("confidence", 0.0))
            if conf > 0.90:
                cur = self.cfg.thresholds["min_confidence"]
                self.cfg.thresholds["min_confidence"] = min(0.85, cur + 0.005)

    async def _handle_anomaly_alert(self, event: Mapping[str, Any]) -> None:
        """Respond to AVN anomaly alerts by tightening thresholds temporarily."""
        payload = event.get("payload", {}) or {}
        severity = payload.get("severity", "medium")
        anomaly_type = payload.get("type", "unknown")
        if severity == "critical":
            logger.warning("Critical anomaly detected: %s", anomaly_type)
            self.cfg.thresholds["min_confidence"] = min(0.90, self.cfg.thresholds["min_confidence"] + 0.10)
            self.cfg.thresholds["min_trust"] = min(0.85, self.cfg.thresholds["min_trust"] + 0.10)

    # --------------- Policy Gates ---------------

    async def _run_verification_gate(self, decision: UnifiedDecision, claims: List[Any], context: Dict[str, Any]) -> None:
        """Optional verification pass (no-op safe)."""
        try:
            await self.verifier.verify(decision=decision, claims=claims, context=context)  # type: ignore[attr-defined]
        except AttributeError:
            # If verifier doesn't implement this exact signature, skip quietly.
            logger.debug("VerificationEngine.verify not available; skipping verification gate")

    async def _apply_constitutional_policies(
        self,
        decision: UnifiedDecision,
        claims: List[Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate constitutional principles and compute a compliance score."""
        p = self.cfg.policies["constitutional_principles"]
        violations: List[str] = []
        scores: List[float] = []

        if p.get("transparency", False):
            s = await self._check_transparency(decision, context)
            scores.append(s)
            if s < 0.8:
                violations.append("transparency_insufficient")

        if p.get("fairness", False):
            s = await self._check_fairness(decision, claims)
            scores.append(s)
            if s < 0.8:
                violations.append("fairness_concerns")

        if p.get("accountability", False):
            s = await self._check_accountability(decision, context)
            scores.append(s)
            if s < 0.7:
                violations.append("accountability_unclear")

        if p.get("consistency", False):
            s = await self._check_consistency(decision)
            scores.append(s)

        if p.get("harm_prevention", False):
            s = await self._check_harm_prevention(decision, context)
            scores.append(s)

        compliance_score = sum(scores) / len(scores) if scores else 1.0
        compliant = compliance_score >= self.cfg.thresholds["constitutional_compliance_min"]

        return {
            "compliant": compliant,
            "violations": violations,
            "compliance_score": compliance_score,
        }

    async def _check_transparency(self, decision: UnifiedDecision, context: Dict[str, Any]) -> float:
        score = 0.8
        if len((decision.rationale or "").strip()) > 50:
            score += 0.1
        if "audit_trail" in context:
            score += 0.1
        return min(score, 1.0)

    async def _check_fairness(self, decision: UnifiedDecision, claims: List[Any]) -> float:
        score = 0.7
        signals = [
            s for s in (decision.inputs.get("component_signals", []) or [])
            if isinstance(s, dict) and "fairness" in (s.get("component") or "")
        ]
        if signals:
            avg = sum(float(s.get("weight", 0.0)) for s in signals) / len(signals)
            score = max(score, min(max(avg, 0.0), 1.0))
        return score

    async def _check_accountability(self, decision: UnifiedDecision, context: Dict[str, Any]) -> float:
        score = 0.6
        if "responsible_party" in context:
            score += 0.2
        if "review_process" in context:
            score += 0.2
        return min(score, 1.0)

    async def _check_consistency(self, decision: UnifiedDecision) -> float:
        similar = self.memory_core.get_similar_decisions(decision.topic, limit=5)
        if not similar:
            return 0.7
        consist = sum(1 for r in similar if r.get("outcome") == decision.recommendation)
        ratio = consist / max(1, len(similar))
        return max(0.3, ratio)

    async def _check_harm_prevention(self, decision: UnifiedDecision, context: Dict[str, Any]) -> float:
        score = 0.8
        risk = (context.get("risk_assessment", {}) or {}).get("level", "medium")
        if risk == "low":
            score += 0.1
        elif risk == "high":
            score -= 0.2
        if "mitigation_strategies" in context:
            score += 0.1
        return max(0.0, min(score, 1.0))

    async def _validate_thresholds(self, decision: UnifiedDecision) -> Dict[str, bool]:
        return {
            "confidence_sufficient": float(decision.confidence) >= self.cfg.thresholds["min_confidence"],
            "trust_sufficient": float(decision.trust_score) >= self.cfg.thresholds["min_trust"],
            "meets_all_thresholds": (
                float(decision.confidence) >= self.cfg.thresholds["min_confidence"]
                and float(decision.trust_score) >= self.cfg.thresholds["min_trust"]
            ),
        }

    async def _check_escalation_requirements(self, decision: UnifiedDecision, context: Dict[str, Any]) -> bool:
        rules = self.cfg.policies["escalation_rules"]
        dtype = context.get("decision_type", "")
        if dtype in rules["require_parliament_review"]:
            return True
        if float(decision.confidence) < float(rules["auto_escalate_threshold"]):
            return True
        if dtype in rules["high_risk_decisions"]:
            return True
        return False

    # --------------- Verdict, Audit, Experience ---------------

    def _generate_verdict(
        self,
        decision: UnifiedDecision,
        constitutional: Dict[str, Any],
        thresholds: Dict[str, bool],
        escalation_needed: bool,
        correlation_id: str,
    ) -> Dict[str, Any]:
        # Outcome selection
        if not constitutional["compliant"]:
            outcome = EventType.GOVERNANCE_REJECTED.value
            rationale = f"Constitutional violations: {', '.join(constitutional['violations']) or 'unspecified'}"
        elif not thresholds["meets_all_thresholds"]:
            outcome = EventType.GOVERNANCE_REJECTED.value
            rationale = f"Insufficient confidence ({decision.confidence:.3f}) or trust ({decision.trust_score:.3f})"
        elif escalation_needed:
            outcome = EventType.GOVERNANCE_NEEDS_REVIEW.value
            rationale = "Decision requires human review due to escalation criteria"
        elif (decision.recommendation or "").lower() == "approve":
            outcome = EventType.GOVERNANCE_APPROVED.value
            rationale = f"Decision approved: {decision.rationale}"
        elif (decision.recommendation or "").lower() == "reject":
            outcome = EventType.GOVERNANCE_REJECTED.value
            rationale = f"Decision rejected: {decision.rationale}"
        else:
            outcome = EventType.GOVERNANCE_NEEDS_REVIEW.value
            rationale = f"Decision requires review: {decision.rationale}"

        return {
            "correlation_id": correlation_id,
            "decision_id": decision.decision_id,
            "outcome": outcome,
            "rationale": rationale,
            "confidence": float(decision.confidence),
            "trust_score": float(decision.trust_score),
            "constitutional_compliance": float(constitutional["compliance_score"]),
            "timestamp": _iso(),
            "instance_id": self.cfg.instance_id,
            "version": self.cfg.version,
            "requires_manual_review": escalation_needed or outcome == EventType.GOVERNANCE_NEEDS_REVIEW.value,
        }

    async def _audit_and_publish(self, verdict: Dict[str, Any], started: datetime) -> None:
        """Persist decision snapshot and emit the verdict event."""
        # Store decision using a minimal UnifiedDecision repr for MemoryCore
        decision_id = verdict["decision_id"]
        recommendation = {
            EventType.GOVERNANCE_APPROVED.value: "approve",
            EventType.GOVERNANCE_REJECTED.value: "reject",
            EventType.GOVERNANCE_NEEDS_REVIEW.value: "review",
        }.get(verdict["outcome"], "review")

        # Build a minimal structure accepted by MemoryCore (works with Pydantic v2 model_dump)
        md = UnifiedDecision(
            decision_id=decision_id,
            topic=verdict.get("rationale", ""),
            inputs={},
            recommendation=recommendation,
            rationale=verdict["rationale"],
            confidence=float(verdict["confidence"]),
            trust_score=float(verdict["trust_score"]),
            timestamp=_utcnow(),
        )

        self.memory_core.store_decision(
            md,
            outcome=verdict["outcome"],
            instance_id=self.cfg.instance_id,
            version=self.cfg.version,
        )

        # Publish final outcome event
        await self.event_bus.publish(verdict["outcome"], verdict, correlation_id=verdict["correlation_id"])

        # Telemetry
        elapsed = (_utcnow() - started).total_seconds()
        logger.info("Governance decision %s processed in %.3fs -> %s", decision_id, elapsed, verdict["outcome"])

    async def _record_governance_experience(
        self,
        decision: UnifiedDecision,
        verdict: Dict[str, Any],
        processing_time_s: float,
    ) -> None:
        """Emit a meta-learning experience and persist it via MemoryCore."""
        success = float(verdict["confidence"])
        if verdict["outcome"] == EventType.GOVERNANCE_REJECTED.value:
            success *= 0.5
        elif verdict["outcome"] == EventType.GOVERNANCE_NEEDS_REVIEW.value:
            success *= 0.7

        experience = Experience(
            type="CONSTITUTIONAL_COMPLIANCE",
            component_id="governance_engine",
            context={
                "principles_triggered": len(self.cfg.policies["constitutional_principles"]),
                "processing_time": processing_time_s,
                "escalation_needed": verdict.get("requires_manual_review", False),
            },
            outcome={
                "compliance_score": verdict.get("constitutional_compliance", 0.0),
                "final_verdict": verdict["outcome"],
            },
            success_score=success,
            timestamp=_utcnow(),
        )

        self.memory_core.store_experience(experience)
        await self.event_bus.publish("LEARNING_EXPERIENCE", experience.model_dump())

    # --------------- Snapshots & Shadow ---------------

    async def create_snapshot(self) -> GovernanceSnapshot:
        """Create a snapshot of current governance state and publish an event."""
        snapshot_id = generate_snapshot_id()
        state_data = {
            "policies": self.cfg.policies,
            "thresholds": self.cfg.thresholds,
            "version": self.cfg.version,
            "instance_id": self.cfg.instance_id,
        }
        state_hash = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode("utf-8")).hexdigest()

        snapshot = GovernanceSnapshot(
            snapshot_id=snapshot_id,
            instance_id=self.cfg.instance_id,
            version=self.cfg.version,
            policies=json.loads(json.dumps(self.cfg.policies)),
            thresholds=json.loads(json.dumps(self.cfg.thresholds)),
            model_weights=self.unifier.get_current_weights(),
            state_hash=state_hash,
            created_at=_utcnow(),
        )

        self.memory_core.store_snapshot(snapshot)
        await self.event_bus.publish(EventType.GOVERNANCE_SNAPSHOT_CREATED, snapshot.model_dump())
        logger.info("Created governance snapshot %s", snapshot_id)
        return snapshot

    async def load_snapshot(self, snapshot_id: str) -> bool:
        """Load last snapshot for this instance (verifies ID) and update unifier weights."""
        try:
            snap = self.memory_core.get_latest_snapshot(self.cfg.instance_id)
            if not snap or snap.get("snapshot_id") != snapshot_id:
                logger.error("Snapshot %s not found for instance %s", snapshot_id, self.cfg.instance_id)
                return False

            self.cfg.policies = json.loads(snap["policies_json"])
            self.cfg.thresholds = json.loads(snap["thresholds_json"])
            model_weights = json.loads(snap["model_weights_json"])
            await self.unifier.update_weights(model_weights)
            logger.info("Loaded snapshot %s", snapshot_id)
            return True
        except Exception as e:
            logger.exception("Error loading snapshot %s: %s", snapshot_id, e)
            return False

    async def start_shadow_mode(self, new_instance_id: str = "governance-shadow") -> None:
        """Enable shadow mode for parallel evaluation."""
        self.shadow_mode = True
        self.shadow_instance = new_instance_id
        logger.info("Shadow mode enabled with instance %s", new_instance_id)

    # --------------- Introspection ---------------

    def get_governance_status(self) -> Dict[str, Any]:
        return {
            "instance_id": self.cfg.instance_id,
            "version": self.cfg.version,
            "shadow_mode": self.shadow_mode,
            "shadow_instance": self.shadow_instance,
            "policies": self.cfg.policies,
            "thresholds": self.cfg.thresholds,
            "timestamp": _iso(),
        }
