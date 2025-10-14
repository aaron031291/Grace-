"""
Cross-Persona Consensus Engine - Enhanced MLDL Quorum integration for sandbox governance.

This module extends the existing MLDL quorum system to provide multi-specialist
consensus for sandbox experiments, ensuring all "senior dev roles" reach agreement
before approving sandbox merges.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, asdict

from ..mldl.quorum import MLDLQuorum, QuorumConsensus
from ..core.contracts import generate_correlation_id

logger = logging.getLogger(__name__)


class PersonaRole(Enum):
    """Senior development persona roles for consensus."""

    SENIOR_ARCHITECT = "senior_architect"
    DEVOPS_ENGINEER = "devops_engineer"
    AI_ENGINEER = "ai_engineer"
    SECURITY_ENGINEER = "security_engineer"
    FULLSTACK_ENGINEER = "fullstack_engineer"
    WEB3_SPECIALIST = "web3_specialist"
    DATA_ENGINEER = "data_engineer"
    UX_DESIGNER = "ux_designer"
    SYSTEM_DESIGNER = "system_designer"
    PERFORMANCE_ENGINEER = "performance_engineer"
    COMPLIANCE_OFFICER = "compliance_officer"


@dataclass
class PersonaOpinion:
    """Individual persona's opinion on a sandbox proposal."""

    persona_role: PersonaRole
    specialist_ids: List[str]  # Which MLDL specialists represent this persona
    vote: str  # "approve", "reject", "abstain"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    concerns: List[str]
    requirements: List[str]
    timestamp: datetime


@dataclass
class ConsensusRequest:
    """Request for cross-persona consensus on sandbox merge."""

    request_id: str
    sandbox_id: str
    proposal_type: str  # "merge", "architecture_change", "policy_update"
    description: str
    validation_results: Dict[str, Any]
    impact_assessment: Dict[str, Any]
    required_personas: Set[PersonaRole]
    minimum_approval_threshold: float = 0.7  # 70% approval needed
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ConsensusResult:
    """Result of cross-persona consensus process."""

    request_id: str
    consensus_reached: bool
    overall_approval: bool
    approval_percentage: float
    persona_opinions: List[PersonaOpinion]
    final_decision: str  # "approved", "rejected", "requires_revision"
    decision_rationale: str
    requirements_for_approval: List[str]
    completion_time: datetime
    processing_duration_seconds: float


class CrossPersonaConsensus:
    """
    Orchestrates consensus among multiple senior development personas
    using the MLDL quorum system as the underlying intelligence.
    """

    def __init__(self, mldl_quorum: MLDLQuorum, trust_threshold: float = 0.6):
        self.mldl_quorum = mldl_quorum
        self.trust_threshold = trust_threshold

        # Map personas to specialist combinations
        self.persona_specialist_mapping = self._initialize_persona_mappings()

        # Historical consensus data for learning
        self.consensus_history: List[ConsensusResult] = []
        self.persona_reliability_scores: Dict[PersonaRole, float] = {}

        # Initialize persona reliability scores
        for role in PersonaRole:
            self.persona_reliability_scores[role] = 0.8  # Start with good baseline

    def _initialize_persona_mappings(self) -> Dict[PersonaRole, List[str]]:
        """Map personas to their representative MLDL specialists."""
        return {
            PersonaRole.SENIOR_ARCHITECT: [
                "system_architect",
                "software_architect",
                "solution_architect",
            ],
            PersonaRole.DEVOPS_ENGINEER: [
                "devops_specialist",
                "infrastructure_specialist",
                "deployment_specialist",
            ],
            PersonaRole.AI_ENGINEER: [
                "ml_engineer",
                "deep_learning_specialist",
                "ai_researcher",
            ],
            PersonaRole.SECURITY_ENGINEER: [
                "security_specialist",
                "cryptography_specialist",
                "compliance_specialist",
            ],
            PersonaRole.FULLSTACK_ENGINEER: [
                "frontend_specialist",
                "backend_specialist",
                "api_specialist",
            ],
            PersonaRole.WEB3_SPECIALIST: [
                "blockchain_specialist",
                "smart_contract_specialist",
                "defi_specialist",
            ],
            PersonaRole.DATA_ENGINEER: [
                "data_pipeline_specialist",
                "database_specialist",
                "analytics_specialist",
            ],
            PersonaRole.UX_DESIGNER: [
                "ui_specialist",
                "ux_specialist",
                "design_system_specialist",
            ],
            PersonaRole.SYSTEM_DESIGNER: [
                "system_designer",
                "integration_specialist",
                "api_designer",
            ],
            PersonaRole.PERFORMANCE_ENGINEER: [
                "performance_specialist",
                "optimization_specialist",
                "scalability_specialist",
            ],
            PersonaRole.COMPLIANCE_OFFICER: [
                "compliance_specialist",
                "governance_specialist",
                "audit_specialist",
            ],
        }

    async def request_consensus(
        self,
        sandbox_id: str,
        proposal_type: str,
        description: str,
        validation_results: Dict[str, Any],
        impact_assessment: Dict[str, Any],
        required_personas: Optional[Set[PersonaRole]] = None,
    ) -> ConsensusResult:
        """Request cross-persona consensus on a sandbox proposal."""

        start_time = datetime.now()
        request_id = f"consensus_{generate_correlation_id()}"

        # Determine required personas if not specified
        if required_personas is None:
            required_personas = self._determine_required_personas(
                proposal_type, validation_results, impact_assessment
            )

        consensus_request = ConsensusRequest(
            request_id=request_id,
            sandbox_id=sandbox_id,
            proposal_type=proposal_type,
            description=description,
            validation_results=validation_results,
            impact_assessment=impact_assessment,
            required_personas=required_personas,
            created_at=start_time,
        )

        logger.info(
            f"Requesting consensus from {len(required_personas)} personas for sandbox {sandbox_id}"
        )

        # Collect opinions from each required persona
        persona_opinions = []
        for persona_role in required_personas:
            opinion = await self._get_persona_opinion(consensus_request, persona_role)
            persona_opinions.append(opinion)

        # Analyze consensus
        consensus_result = self._analyze_consensus(
            consensus_request, persona_opinions, start_time
        )

        # Store for learning
        self.consensus_history.append(consensus_result)
        self._update_persona_reliability(consensus_result)

        logger.info(
            f"Consensus {'reached' if consensus_result.consensus_reached else 'not reached'} "
            f"for sandbox {sandbox_id}: {consensus_result.final_decision}"
        )

        return consensus_result

    def _determine_required_personas(
        self,
        proposal_type: str,
        validation_results: Dict[str, Any],
        impact_assessment: Dict[str, Any],
    ) -> Set[PersonaRole]:
        """Determine which personas need to participate based on the proposal."""

        required = {PersonaRole.SENIOR_ARCHITECT}  # Always include architect

        # Add personas based on proposal type
        if proposal_type == "merge":
            required.update(
                [PersonaRole.DEVOPS_ENGINEER, PersonaRole.SECURITY_ENGINEER]
            )
        elif proposal_type == "architecture_change":
            required.update(
                [
                    PersonaRole.SYSTEM_DESIGNER,
                    PersonaRole.PERFORMANCE_ENGINEER,
                    PersonaRole.SECURITY_ENGINEER,
                ]
            )
        elif proposal_type == "policy_update":
            required.update(
                [PersonaRole.COMPLIANCE_OFFICER, PersonaRole.SECURITY_ENGINEER]
            )

        # Add personas based on validation results
        validation_categories = validation_results.get("test_results", [])
        for test_result in validation_categories:
            category = test_result.get("category")

            if category == "security_scan" and test_result.get("status") != "passed":
                required.add(PersonaRole.SECURITY_ENGINEER)
            elif category == "performance_test":
                required.add(PersonaRole.PERFORMANCE_ENGINEER)
            elif category == "compliance_check":
                required.add(PersonaRole.COMPLIANCE_OFFICER)

        # Add personas based on impact assessment
        impact_areas = impact_assessment.get("affected_systems", [])

        if "ai_models" in impact_areas:
            required.add(PersonaRole.AI_ENGINEER)
        if "blockchain" in impact_areas or "web3" in impact_areas:
            required.add(PersonaRole.WEB3_SPECIALIST)
        if "data_pipeline" in impact_areas:
            required.add(PersonaRole.DATA_ENGINEER)
        if "user_interface" in impact_areas:
            required.add(PersonaRole.UX_DESIGNER)
        if "api" in impact_areas or "integration" in impact_areas:
            required.add(PersonaRole.FULLSTACK_ENGINEER)

        return required

    async def _get_persona_opinion(
        self, consensus_request: ConsensusRequest, persona_role: PersonaRole
    ) -> PersonaOpinion:
        """Get opinion from a specific persona using MLDL specialists."""

        specialist_ids = self.persona_specialist_mapping.get(persona_role, [])

        # Prepare task context for MLDL quorum
        task_context = {
            "task_type": "sandbox_review",
            "proposal_type": consensus_request.proposal_type,
            "sandbox_id": consensus_request.sandbox_id,
            "description": consensus_request.description,
            "validation_results": consensus_request.validation_results,
            "impact_assessment": consensus_request.impact_assessment,
            "persona_role": persona_role.value,
            "review_criteria": self._get_persona_review_criteria(persona_role),
        }

        # Get consensus from relevant specialists
        quorum_consensus = await self.mldl_quorum.request_consensus(
            task_type="sandbox_review",
            inputs=task_context,
            required_specialists=specialist_ids,
        )

        # Interpret quorum result as persona opinion
        opinion = self._interpret_quorum_as_persona_opinion(
            persona_role, quorum_consensus, specialist_ids
        )

        return opinion

    def _get_persona_review_criteria(self, persona_role: PersonaRole) -> List[str]:
        """Get review criteria specific to each persona role."""

        criteria_map = {
            PersonaRole.SENIOR_ARCHITECT: [
                "System design coherence",
                "Architectural patterns compliance",
                "Scalability considerations",
                "Integration impact",
            ],
            PersonaRole.DEVOPS_ENGINEER: [
                "Deployment complexity",
                "Infrastructure requirements",
                "Monitoring and observability",
                "Rollback strategy",
            ],
            PersonaRole.AI_ENGINEER: [
                "Model performance impact",
                "Data quality requirements",
                "Inference latency",
                "ML pipeline integration",
            ],
            PersonaRole.SECURITY_ENGINEER: [
                "Security vulnerabilities",
                "Access control changes",
                "Data privacy compliance",
                "Threat model impact",
            ],
            PersonaRole.FULLSTACK_ENGINEER: [
                "API compatibility",
                "Frontend-backend integration",
                "User experience impact",
                "Code maintainability",
            ],
            PersonaRole.WEB3_SPECIALIST: [
                "Smart contract security",
                "Gas optimization",
                "Blockchain integration",
                "DeFi protocol compliance",
            ],
            PersonaRole.DATA_ENGINEER: [
                "Data pipeline integrity",
                "ETL process impact",
                "Data quality validation",
                "Storage optimization",
            ],
            PersonaRole.UX_DESIGNER: [
                "User experience consistency",
                "Accessibility compliance",
                "Design system adherence",
                "Usability impact",
            ],
            PersonaRole.SYSTEM_DESIGNER: [
                "System integration points",
                "API design consistency",
                "Service boundaries",
                "Data flow optimization",
            ],
            PersonaRole.PERFORMANCE_ENGINEER: [
                "Performance benchmarks",
                "Resource utilization",
                "Latency impact",
                "Scalability bottlenecks",
            ],
            PersonaRole.COMPLIANCE_OFFICER: [
                "Regulatory compliance",
                "Audit trail completeness",
                "Policy adherence",
                "Risk assessment",
            ],
        }

        return criteria_map.get(persona_role, ["General code quality"])

    def _interpret_quorum_as_persona_opinion(
        self,
        persona_role: PersonaRole,
        quorum_consensus: QuorumConsensus,
        specialist_ids: List[str],
    ) -> PersonaOpinion:
        """Convert MLDL quorum consensus into persona opinion."""

        # Map consensus confidence to vote
        if quorum_consensus.confidence >= 0.8:
            vote = "approve"
        elif quorum_consensus.confidence >= 0.4:
            vote = "abstain"
        else:
            vote = "reject"

        # Extract reasoning and concerns from consensus details
        reasoning = (
            quorum_consensus.reasoning
            or f"Analysis from {persona_role.value} perspective"
        )

        concerns = []
        requirements = []

        if quorum_consensus.confidence < 0.8:
            concerns.append("Confidence level below optimal threshold")

        if quorum_consensus.disagreement_level > 0.3:
            concerns.append("High disagreement among specialists")

        # Add role-specific concerns based on consensus details
        if (
            persona_role == PersonaRole.SECURITY_ENGINEER
            and quorum_consensus.confidence < 0.9
        ):
            concerns.append("Security implications need deeper analysis")
            requirements.append("Additional security validation required")

        return PersonaOpinion(
            persona_role=persona_role,
            specialist_ids=specialist_ids,
            vote=vote,
            confidence=quorum_consensus.confidence,
            reasoning=reasoning,
            concerns=concerns,
            requirements=requirements,
            timestamp=datetime.now(),
        )

    def _analyze_consensus(
        self,
        consensus_request: ConsensusRequest,
        persona_opinions: List[PersonaOpinion],
        start_time: datetime,
    ) -> ConsensusResult:
        """Analyze persona opinions to determine overall consensus."""

        completion_time = datetime.now()
        processing_duration = (completion_time - start_time).total_seconds()

        # Count votes
        approval_votes = len([op for op in persona_opinions if op.vote == "approve"])
        rejection_votes = len([op for op in persona_opinions if op.vote == "reject"])
        abstain_votes = len([op for op in persona_opinions if op.vote == "abstain"])

        total_votes = len(persona_opinions)
        approval_percentage = approval_votes / total_votes if total_votes > 0 else 0.0

        # Determine if consensus is reached
        consensus_reached = (
            approval_percentage >= consensus_request.minimum_approval_threshold
        )

        # Determine final decision
        if consensus_reached and rejection_votes == 0:
            final_decision = "approved"
            decision_rationale = (
                f"Strong consensus achieved ({approval_percentage:.1%} approval)"
            )
        elif consensus_reached:
            final_decision = "approved"
            decision_rationale = f"Consensus achieved ({approval_percentage:.1%} approval) with some reservations"
        elif approval_percentage >= 0.5:
            final_decision = "requires_revision"
            decision_rationale = (
                "Mixed opinions - revision recommended to address concerns"
            )
        else:
            final_decision = "rejected"
            decision_rationale = f"Insufficient approval ({approval_percentage:.1%})"

        # Collect requirements for approval
        requirements_for_approval = []
        for opinion in persona_opinions:
            requirements_for_approval.extend(opinion.requirements)

        # Remove duplicates
        requirements_for_approval = list(set(requirements_for_approval))

        return ConsensusResult(
            request_id=consensus_request.request_id,
            consensus_reached=consensus_reached,
            overall_approval=final_decision == "approved",
            approval_percentage=approval_percentage,
            persona_opinions=persona_opinions,
            final_decision=final_decision,
            decision_rationale=decision_rationale,
            requirements_for_approval=requirements_for_approval,
            completion_time=completion_time,
            processing_duration_seconds=processing_duration,
        )

    def _update_persona_reliability(self, consensus_result: ConsensusResult):
        """Update persona reliability scores based on consensus outcome."""

        # This would typically be updated based on how well the consensus
        # prediction aligns with actual outcomes after deployment
        # For now, we'll use a simplified heuristic

        for opinion in consensus_result.persona_opinions:
            current_score = self.persona_reliability_scores.get(
                opinion.persona_role, 0.8
            )

            # Adjust based on confidence calibration
            if opinion.confidence >= 0.8 and consensus_result.overall_approval:
                # High confidence and good outcome
                new_score = min(1.0, current_score + 0.05)
            elif opinion.confidence < 0.5 and not consensus_result.overall_approval:
                # Low confidence and correctly cautious
                new_score = min(1.0, current_score + 0.02)
            elif opinion.confidence >= 0.8 and not consensus_result.overall_approval:
                # High confidence but poor outcome
                new_score = max(0.3, current_score - 0.1)
            else:
                # No significant change
                new_score = current_score

            self.persona_reliability_scores[opinion.persona_role] = new_score

    async def get_consensus_history(
        self, sandbox_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get consensus history, optionally filtered by sandbox ID."""

        history = self.consensus_history

        if sandbox_id:
            history = [c for c in history if c.sandbox_id == sandbox_id]

        # Sort by completion time, most recent first
        history.sort(key=lambda x: x.completion_time, reverse=True)

        return [asdict(result) for result in history[:limit]]

    def get_persona_reliability_scores(self) -> Dict[str, float]:
        """Get current reliability scores for all personas."""
        return {
            role.value: score for role, score in self.persona_reliability_scores.items()
        }

    async def simulate_consensus(
        self, consensus_request: ConsensusRequest
    ) -> Dict[str, Any]:
        """Simulate consensus without actually running it (for testing/preview)."""

        simulation_result = {
            "request_id": consensus_request.request_id,
            "required_personas": [
                role.value for role in consensus_request.required_personas
            ],
            "estimated_processing_time_seconds": len(
                consensus_request.required_personas
            )
            * 2.5,
            "likely_outcome": "unknown",
            "confidence_prediction": 0.5,
        }

        # Simple heuristic for likely outcome
        if consensus_request.proposal_type == "merge":
            # Check validation results
            validation_score = consensus_request.validation_results.get(
                "overall_score", 0.5
            )
            if validation_score >= 0.8:
                simulation_result["likely_outcome"] = "approved"
                simulation_result["confidence_prediction"] = validation_score
            elif validation_score >= 0.6:
                simulation_result["likely_outcome"] = "requires_revision"
                simulation_result["confidence_prediction"] = validation_score * 0.8
            else:
                simulation_result["likely_outcome"] = "rejected"
                simulation_result["confidence_prediction"] = 0.3

        return simulation_result
