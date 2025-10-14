"""
Verification Engine - Truth validation and claim analysis for Grace governance kernel.
"""

from typing import List, Dict, Any, Optional
import re
from datetime import datetime
import logging
from dataclasses import dataclass

from ..core.contracts import (
    Claim,
    VerifiedClaims,
    LogicReport,
    Experience,
)


logger = logging.getLogger(__name__)


@dataclass
class ContradictionReport:
    claim_ids: List[str]
    contradiction_type: str
    description: str
    confidence: float


class VerificationEngine:
    """
    Handles claim analysis, multi-source verification, constitutional reasoning,
    confidence scoring, and contradiction detection.
    """

    def __init__(self, event_bus, memory_core, vault_engine=None):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.vault_engine = (
            vault_engine  # Optional vault engine for enhanced verification
        )
        self.constitutional_principles = self._load_constitutional_principles()

    def _load_constitutional_principles(self) -> Dict[str, Dict[str, Any]]:
        """Load constitutional principles for reasoning."""
        return {
            "transparency": {
                "description": "All decisions must be transparent and auditable",
                "weight": 1.0,
                "checks": ["audit_trail_present", "reasoning_provided"],
            },
            "fairness": {
                "description": "Decisions must be fair and unbiased",
                "weight": 1.0,
                "checks": ["bias_analysis", "equal_treatment"],
            },
            "accountability": {
                "description": "Decision makers must be accountable",
                "weight": 0.9,
                "checks": ["responsible_party_identified", "review_process"],
            },
            "consistency": {
                "description": "Similar cases should have similar outcomes",
                "weight": 0.8,
                "checks": ["precedent_analysis", "rule_compliance"],
            },
            "harm_prevention": {
                "description": "Decisions must not cause unnecessary harm",
                "weight": 1.0,
                "checks": ["risk_assessment", "mitigation_strategies"],
            },
        }

    async def verify_claims_realtime(
        self, claims: List[Claim], level: str = "standard"
    ) -> VerifiedClaims:
        """
        Perform real-time verification of claims with multi-source analysis.

        Args:
            claims: List of claims to verify
            level: Verification level ("basic", "standard", "comprehensive")
        """
        verification_start = datetime.now()

        try:
            verified_claims = []
            contradictions = []
            overall_confidence_scores = []

            for claim in claims:
                # Verify individual claim
                verified_claim = await self._verify_single_claim(claim, level)
                verified_claims.append(verified_claim)
                overall_confidence_scores.append(verified_claim.confidence)

                # Check for contradictions with other claims
                claim_contradictions = await self._check_contradictions(
                    verified_claim, verified_claims[:-1]
                )
                contradictions.extend(claim_contradictions)

            # Calculate overall confidence
            overall_confidence = (
                sum(overall_confidence_scores) / len(overall_confidence_scores)
                if overall_confidence_scores
                else 0.0
            )

            # Determine verification status
            verification_status = self._determine_verification_status(
                verified_claims, contradictions, overall_confidence
            )

            result = VerifiedClaims(
                claims=verified_claims,
                overall_confidence=overall_confidence,
                verification_status=verification_status,
                contradictions=[c.description for c in contradictions],
            )

            # Record experience for learning
            await self._record_verification_experience(
                len(claims),
                verification_status,
                overall_confidence,
                (datetime.now() - verification_start).total_seconds(),
            )

            return result

        except Exception as e:
            logger.error(f"Error in claim verification: {e}")
            return VerifiedClaims(
                claims=claims,
                overall_confidence=0.0,
                verification_status="error",
                contradictions=[f"Verification error: {str(e)}"],
            )

    async def _verify_single_claim(self, claim: Claim, level: str) -> Claim:
        """Verify a single claim through multiple validation methods."""
        confidence_factors = []

        # Source credibility analysis
        source_score = await self._analyze_source_credibility(claim.sources)
        confidence_factors.append(source_score)

        # Evidence quality assessment
        evidence_score = await self._assess_evidence_quality(claim.evidence)
        confidence_factors.append(evidence_score)

        # Logical chain validation
        logic_score = await self._validate_logical_chain(claim.logical_chain)
        confidence_factors.append(logic_score)

        # Constitutional compliance check
        if level in ["standard", "comprehensive"]:
            constitutional_score = await self._check_constitutional_compliance(claim)
            confidence_factors.append(constitutional_score)

        # Cross-reference with known facts
        if level == "comprehensive":
            fact_check_score = await self._cross_reference_facts(claim)
            confidence_factors.append(fact_check_score)

        # Calculate weighted confidence
        final_confidence = sum(confidence_factors) / len(confidence_factors)

        # Create updated claim with new confidence
        verified_claim = Claim(
            id=claim.id,
            statement=claim.statement,
            sources=claim.sources,
            evidence=claim.evidence,
            confidence=min(final_confidence, 1.0),  # Cap at 1.0
            logical_chain=claim.logical_chain,
        )

        return verified_claim

    async def _analyze_source_credibility(self, sources: List) -> float:
        """Analyze the credibility of claim sources."""
        if not sources:
            return 0.1  # Very low confidence for claims without sources

        credibility_scores = []
        for source in sources:
            # Use provided credibility or calculate based on URI patterns
            if hasattr(source, "credibility"):
                credibility_scores.append(source.credibility)
            else:
                # Simple heuristic based on URI patterns
                uri = getattr(source, "uri", "")
                credibility = self._estimate_uri_credibility(uri)
                credibility_scores.append(credibility)

        return sum(credibility_scores) / len(credibility_scores)

    def _estimate_uri_credibility(self, uri: str) -> float:
        """Estimate credibility based on URI patterns."""
        # Academic and government sources
        if any(domain in uri.lower() for domain in [".edu", ".gov", ".org"]):
            return 0.8
        # Known reliable news sources (simplified)
        elif any(domain in uri.lower() for domain in ["reuters", "ap.org", "bbc"]):
            return 0.7
        # HTTPS vs HTTP
        elif uri.startswith("https://"):
            return 0.6
        elif uri.startswith("http://"):
            return 0.4
        else:
            return 0.3

    async def _assess_evidence_quality(self, evidence: List) -> float:
        """Assess the quality and reliability of evidence."""
        if not evidence:
            return 0.2

        quality_scores = []
        for item in evidence:
            evidence_type = getattr(item, "type", "unknown")
            pointer = getattr(item, "pointer", "")

            # Score based on evidence type and accessibility
            if evidence_type == "doc":
                quality_scores.append(0.7 if pointer else 0.5)
            elif evidence_type == "db":
                quality_scores.append(0.8 if pointer else 0.6)
            elif evidence_type == "api":
                quality_scores.append(0.6 if pointer else 0.4)
            else:
                quality_scores.append(0.3)

        return sum(quality_scores) / len(quality_scores)

    async def _validate_logical_chain(self, logical_chain: List) -> float:
        """Validate the logical reasoning chain."""
        if not logical_chain:
            return 0.3  # Some confidence for claims without explicit reasoning

        # Simple validation - check for logical consistency
        chain_quality = min(
            1.0, len(logical_chain) * 0.2
        )  # More steps = better reasoning

        # Look for logical fallacies or inconsistencies
        fallacy_penalty = 0.0
        for step in logical_chain:
            step_text = getattr(step, "step", "").lower()
            if any(
                fallacy in step_text
                for fallacy in ["because i said so", "everyone knows", "obviously"]
            ):
                fallacy_penalty += 0.1

        return max(0.1, chain_quality - fallacy_penalty)

    async def _check_constitutional_compliance(self, claim: Claim) -> float:
        """Check claim against constitutional principles."""
        compliance_scores = []

        for principle_name, principle in self.constitutional_principles.items():
            # Simple keyword-based compliance check
            statement = claim.statement.lower()
            compliance_keywords = {
                "transparency": ["transparent", "open", "public", "documented"],
                "fairness": ["fair", "equal", "unbiased", "just"],
                "accountability": ["responsible", "accountable", "oversight"],
                "consistency": ["consistent", "standard", "uniform"],
                "harm_prevention": ["safe", "harmless", "protect", "prevent"],
            }

            if principle_name in compliance_keywords:
                keywords = compliance_keywords[principle_name]
                matches = sum(1 for keyword in keywords if keyword in statement)
                score = min(1.0, matches * 0.25) * principle["weight"]
                compliance_scores.append(score)

        return (
            sum(compliance_scores) / len(compliance_scores)
            if compliance_scores
            else 0.5
        )

    async def _cross_reference_facts(self, claim: Claim) -> float:
        """Cross-reference claim with known facts from memory."""
        # Look for similar claims in memory
        similar_decisions = self.memory_core.get_similar_decisions(
            claim.statement, limit=5
        )

        if not similar_decisions:
            return 0.5  # Neutral score for novel claims

        # Calculate consistency with past decisions
        consistency_score = 0.0
        for decision in similar_decisions:
            # Simple text similarity check
            similarity = self._calculate_text_similarity(
                claim.statement, decision.get("topic", "")
            )
            if similarity > 0.7:
                consistency_score += 0.2

        return min(1.0, consistency_score)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def _check_contradictions(
        self, claim: Claim, other_claims: List[Claim]
    ) -> List[ContradictionReport]:
        """Check for contradictions between claims."""
        contradictions = []

        for other_claim in other_claims:
            contradiction = await self._detect_contradiction(claim, other_claim)
            if contradiction:
                contradictions.append(contradiction)

        return contradictions

    async def _detect_contradiction(
        self, claim1: Claim, claim2: Claim
    ) -> Optional[ContradictionReport]:
        """Detect if two claims contradict each other."""
        statement1 = claim1.statement.lower()
        statement2 = claim2.statement.lower()

        # Simple contradiction patterns - check for positive/negative pairs
        positive_patterns = [r"is (.+)", r"will (.+)", r"can (.+)", r"should (.+)"]

        negative_patterns = [
            r"is not (.+)",
            r"will not (.+)",
            r"cannot (.+)",
            r"should not (.+)",
        ]

        # Check each positive pattern against corresponding negative pattern
        for i, (pos_pattern, neg_pattern) in enumerate(
            zip(positive_patterns, negative_patterns)
        ):
            pos_match1 = re.search(pos_pattern, statement1)
            neg_match2 = re.search(neg_pattern, statement2)

            if pos_match1 and neg_match2:
                # Extract the subject and see if they're similar
                subject1 = pos_match1.group(1)
                subject2 = neg_match2.group(1)

                # Simple similarity check
                if (
                    subject1 == subject2
                    or self._calculate_text_similarity(subject1, subject2) > 0.7
                ):
                    return ContradictionReport(
                        claim_ids=[claim1.id, claim2.id],
                        contradiction_type="direct_negation",
                        description=f"Claim {claim1.id} contradicts claim {claim2.id} through direct negation",
                        confidence=0.8,
                    )

            # Check the reverse
            neg_match1 = re.search(neg_pattern, statement1)
            pos_match2 = re.search(pos_pattern, statement2)

            if neg_match1 and pos_match2:
                subject1 = neg_match1.group(1)
                subject2 = pos_match2.group(1)

                if (
                    subject1 == subject2
                    or self._calculate_text_similarity(subject1, subject2) > 0.7
                ):
                    return ContradictionReport(
                        claim_ids=[claim1.id, claim2.id],
                        contradiction_type="direct_negation",
                        description=f"Claim {claim1.id} contradicts claim {claim2.id} through direct negation",
                        confidence=0.8,
                    )

        return None

    def _determine_verification_status(
        self,
        claims: List[Claim],
        contradictions: List[ContradictionReport],
        overall_confidence: float,
    ) -> str:
        """Determine overall verification status."""
        if contradictions:
            return "refuted"
        elif overall_confidence >= 0.8:
            return "verified"
        elif overall_confidence >= 0.5:
            return "probable"
        else:
            return "inconclusive"

    async def analyze_reasoning_chain(self, argument: Dict[str, Any]) -> LogicReport:
        """Analyze a reasoning chain for logical validity."""
        try:
            premises = argument.get("premises", [])
            conclusion = argument.get("conclusion", "")
            reasoning_steps = argument.get("steps", [])

            logical_errors = []
            validity = True
            confidence = 1.0

            # Check for common logical fallacies
            if not premises:
                logical_errors.append("No premises provided")
                validity = False
                confidence -= 0.3

            if not conclusion:
                logical_errors.append("No conclusion provided")
                validity = False
                confidence -= 0.3

            # Check reasoning steps
            if reasoning_steps:
                for i, step in enumerate(reasoning_steps):
                    step_text = str(step).lower()

                    # Check for fallacies
                    if any(
                        fallacy in step_text
                        for fallacy in [
                            "ad hominem",
                            "straw man",
                            "false dichotomy",
                            "circular reasoning",
                            "appeal to authority",
                        ]
                    ):
                        logical_errors.append(
                            f"Step {i + 1}: Potential logical fallacy detected"
                        )
                        confidence -= 0.2

                    # Check for unsupported jumps
                    if "therefore" in step_text and i == 0:
                        logical_errors.append(
                            f"Step {i + 1}: Conclusion without sufficient premises"
                        )
                        confidence -= 0.1

            confidence = max(0.0, min(1.0, confidence))
            validity = validity and confidence > 0.5

            return LogicReport(
                argument=argument,
                validity=validity,
                confidence=confidence,
                logical_errors=logical_errors,
            )

        except Exception as e:
            logger.error(f"Error analyzing reasoning chain: {e}")
            return LogicReport(
                argument=argument,
                validity=False,
                confidence=0.0,
                logical_errors=[f"Analysis error: {str(e)}"],
            )

    async def _record_verification_experience(
        self, claims_count: int, status: str, confidence: float, processing_time: float
    ):
        """Record verification experience for meta-learning."""
        experience = Experience(
            type="VERIFICATION_RESULT",
            component_id="verification_engine",
            context={"claims_count": claims_count, "processing_time": processing_time},
            outcome={"status": status, "confidence": confidence},
            success_score=confidence,  # Use confidence as success metric
            timestamp=datetime.now(),
        )

        self.memory_core.store_experience(experience)

        # Emit learning event
        await self.event_bus.publish("LEARNING_EXPERIENCE", experience.to_dict())
