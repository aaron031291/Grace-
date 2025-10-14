"""
Sovereign Sandbox Orchestrator - Main integration layer for Grace's sandbox system.

This module integrates all sandbox components into a unified system that provides:
- Sandbox lifecycle management
- Validation orchestration
- Cross-persona consensus
- Human feedback integration
- Orb interface support
- Meta-learning and governance evolution
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

from .sandbox_manager import (
    SandboxManager,
    GraceSandbox,
    ExperimentType,
    ResourceQuota,
)
from .validation_harness import ValidationHarness, ValidationLevel
from .consensus_engine import CrossPersonaConsensus, ConsensusResult
from .human_feedback import HumanFeedbackInterface, FeedbackType
from ..mldl.quorum import MLDLQuorum
from ..layer_04_audit_logs.immutable_logs import ImmutableLogs
from ..core.kpi_trust_monitor import KPITrustMonitor
from ..resilience.chaos.runner import ChaosRunner

logger = logging.getLogger(__name__)


class SovereignAction(Enum):
    """High-level sovereign actions Grace can take."""

    CREATE_EXPERIMENT = "create_experiment"
    VALIDATE_SANDBOX = "validate_sandbox"
    REQUEST_CONSENSUS = "request_consensus"
    SUBMIT_FOR_REVIEW = "submit_for_review"
    MERGE_APPROVED = "merge_approved"
    LEARN_FROM_FEEDBACK = "learn_from_feedback"
    EVOLVE_GOVERNANCE = "evolve_governance"


@dataclass
class SovereignDecision:
    """A high-level decision made by Grace in her sovereign capacity."""

    decision_id: str
    action: SovereignAction
    context: Dict[str, Any]
    reasoning: str
    confidence: float
    expected_outcome: str
    risk_assessment: Dict[str, float]
    timestamp: datetime
    requires_human_approval: bool = True


@dataclass
class OrbitDashboardData:
    """Data structure for the Orb interface dashboard."""

    sandbox_overview: Dict[str, Any]
    active_experiments: List[Dict[str, Any]]
    validation_status: Dict[str, Any]
    consensus_status: Dict[str, Any]
    human_feedback_status: Dict[str, Any]
    grace_autonomy_metrics: Dict[str, float]
    recent_sovereign_decisions: List[Dict[str, Any]]
    system_health: Dict[str, Any]


class SovereignSandboxOrchestrator:
    """
    Main orchestrator for Grace's sovereign sandbox system.
    Coordinates all components to provide a unified sandbox experience.
    """

    def __init__(
        self,
        event_bus=None,
        mldl_quorum: Optional[MLDLQuorum] = None,
        immutable_logs: Optional[ImmutableLogs] = None,
        trust_monitor: Optional[KPITrustMonitor] = None,
        chaos_runner: Optional[ChaosRunner] = None,
        governance_engine=None,
    ):
        self.event_bus = event_bus
        self.governance_engine = governance_engine

        # Initialize core components
        self.immutable_logs = immutable_logs or ImmutableLogs("sovereign_sandbox.db")
        self.trust_monitor = trust_monitor

        # Initialize sandbox system components
        self.sandbox_manager = SandboxManager(
            event_bus=event_bus,
            immutable_logs=self.immutable_logs,
            trust_monitor=trust_monitor,
            governance_engine=governance_engine,
        )

        self.validation_harness = ValidationHarness(
            chaos_runner=chaos_runner,
            trust_monitor=trust_monitor,
            immutable_logs=self.immutable_logs,
        )

        self.consensus_engine = CrossPersonaConsensus(
            mldl_quorum=mldl_quorum or MLDLQuorum(event_bus), trust_threshold=0.7
        )

        self.human_feedback = HumanFeedbackInterface(
            immutable_logs=self.immutable_logs,
            feedback_callback=self._handle_feedback_callback,
        )

        # Grace's sovereign decision tracking
        self.sovereign_decisions: List[SovereignDecision] = []
        self.autonomy_level = 0.6  # Start with moderate autonomy
        self.learning_enabled = True

        # Meta-learning and governance evolution
        self.governance_learning_data = {
            "successful_patterns": [],
            "failed_patterns": [],
            "policy_adaptations": [],
            "trust_evolution": [],
        }

        # Start background processes
        self._start_sovereign_processes()

    async def grace_experiment_cycle(
        self,
        experiment_description: str,
        experiment_type: ExperimentType,
        custom_quota: Optional[ResourceQuota] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> Dict[str, Any]:
        """
        Full Grace experiment cycle: create sandbox, experiment, validate, seek consensus, merge.
        This is Grace's main sovereign workflow.
        """

        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        logger.info(
            f"Starting Grace experiment cycle {cycle_id}: {experiment_description}"
        )

        try:
            # Step 1: Grace makes sovereign decision to create experiment
            create_decision = await self._make_sovereign_decision(
                action=SovereignAction.CREATE_EXPERIMENT,
                context={
                    "description": experiment_description,
                    "type": experiment_type.value,
                    "validation_level": validation_level.value,
                },
                reasoning="Autonomous curiosity-driven experimentation",
            )

            # Step 2: Create sandbox
            sandbox_id = await self.sandbox_manager.create_sandbox(
                experiment_type=experiment_type,
                description=experiment_description,
                custom_quota=custom_quota,
            )

            # Step 3: Run experiment
            sandbox = await self.sandbox_manager.get_sandbox(sandbox_id)
            experiment_id = await sandbox.start_experiment(
                experiment_type=experiment_type,
                description=experiment_description,
                hypothesis=f"This experiment will improve Grace's capabilities in {experiment_type.value}",
            )

            # Simulate experimental work (in a real implementation, this would be actual experimentation)
            await self._simulate_experiment_work(sandbox, experiment_id)

            # Step 4: Validate sandbox
            validation_decision = await self._make_sovereign_decision(
                action=SovereignAction.VALIDATE_SANDBOX,
                context={"sandbox_id": sandbox_id, "experiment_id": experiment_id},
                reasoning="Ensuring experiment meets quality and safety standards",
            )

            validation_results = await self.validation_harness.validate_sandbox(
                sandbox_id=sandbox_id,
                sandbox_data=sandbox.to_dict(),
                validation_level=validation_level,
            )

            # Step 5: Check if validation passed
            if validation_results["overall_status"] not in ["passed", "warning"]:
                logger.warning(f"Sandbox {sandbox_id} failed validation")
                await self.sandbox_manager.destroy_sandbox(
                    sandbox_id, reason="validation_failed"
                )
                return {
                    "cycle_id": cycle_id,
                    "status": "failed",
                    "reason": "validation_failed",
                    "validation_results": validation_results,
                }

            # Step 6: Request cross-persona consensus
            consensus_decision = await self._make_sovereign_decision(
                action=SovereignAction.REQUEST_CONSENSUS,
                context={
                    "sandbox_id": sandbox_id,
                    "validation_results": validation_results,
                },
                reasoning="Seeking multi-disciplinary approval for sandbox merge",
            )

            consensus_result = await self.consensus_engine.request_consensus(
                sandbox_id=sandbox_id,
                proposal_type="merge",
                description=experiment_description,
                validation_results=validation_results,
                impact_assessment=self._assess_impact(sandbox, validation_results),
            )

            # Step 7: Handle consensus result
            if not consensus_result.overall_approval:
                # Request human feedback on rejection
                await self._request_human_feedback_on_rejection(
                    sandbox_id, consensus_result
                )
                return {
                    "cycle_id": cycle_id,
                    "status": "consensus_rejected",
                    "sandbox_id": sandbox_id,
                    "consensus_result": asdict(consensus_result),
                }

            # Step 8: Submit for human review (if required)
            human_approval_required = self._determine_human_approval_requirement(
                validation_results, consensus_result
            )

            if human_approval_required:
                review_decision = await self._make_sovereign_decision(
                    action=SovereignAction.SUBMIT_FOR_REVIEW,
                    context={
                        "sandbox_id": sandbox_id,
                        "consensus_result": asdict(consensus_result),
                    },
                    reasoning="Human oversight required for final approval",
                    requires_human_approval=True,
                )

                feedback_request_id = await self.human_feedback.request_feedback(
                    feedback_type=FeedbackType.MERGE_APPROVAL,
                    target_id=sandbox_id,
                    title=f"Approve sandbox merge: {experiment_description}",
                    description=f"Grace has completed experiment {experiment_id} and requests approval to merge.",
                    context_data={
                        "sandbox_data": sandbox.to_dict(),
                        "validation_results": validation_results,
                        "consensus_result": asdict(consensus_result),
                    },
                    priority="normal",
                )

                return {
                    "cycle_id": cycle_id,
                    "status": "awaiting_human_approval",
                    "sandbox_id": sandbox_id,
                    "feedback_request_id": feedback_request_id,
                    "validation_results": validation_results,
                    "consensus_result": asdict(consensus_result),
                }

            # Step 9: Auto-approve merge (high autonomy scenario)
            merge_result = await self._auto_approve_merge(sandbox_id, consensus_result)

            end_time = datetime.now()
            cycle_duration = (end_time - start_time).total_seconds()

            # Log successful completion
            await self.immutable_logs.log_governance_action(
                action_type="sovereign_experiment_cycle_completed",
                data={
                    "cycle_id": cycle_id,
                    "sandbox_id": sandbox_id,
                    "experiment_id": experiment_id,
                    "duration_seconds": cycle_duration,
                    "validation_score": validation_results.get("overall_score", 0.0),
                    "consensus_approval": consensus_result.approval_percentage,
                    "final_status": "merged",
                },
                transparency_level="democratic_oversight",
            )

            return {
                "cycle_id": cycle_id,
                "status": "completed_successfully",
                "sandbox_id": sandbox_id,
                "experiment_id": experiment_id,
                "duration_seconds": cycle_duration,
                "merge_result": merge_result,
                "validation_results": validation_results,
                "consensus_result": asdict(consensus_result),
            }

        except Exception as e:
            logger.error(f"Error in experiment cycle {cycle_id}: {e}")

            # Clean up sandbox if it was created
            if "sandbox_id" in locals():
                await self.sandbox_manager.destroy_sandbox(
                    sandbox_id, reason="cycle_error"
                )

            return {"cycle_id": cycle_id, "status": "error", "error": str(e)}

    async def _make_sovereign_decision(
        self,
        action: SovereignAction,
        context: Dict[str, Any],
        reasoning: str,
        confidence: float = None,
        requires_human_approval: bool = None,
    ) -> SovereignDecision:
        """Grace makes a sovereign decision within her sandbox domain."""

        if confidence is None:
            confidence = min(1.0, self.autonomy_level + 0.1)  # Slight confidence boost

        if requires_human_approval is None:
            requires_human_approval = (
                self.autonomy_level < 0.8
            )  # High autonomy = less human approval needed

        decision = SovereignDecision(
            decision_id=f"sovereign_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            action=action,
            context=context,
            reasoning=reasoning,
            confidence=confidence,
            expected_outcome="Positive contribution to Grace's capabilities",
            risk_assessment=self._assess_decision_risk(action, context),
            timestamp=datetime.now(),
            requires_human_approval=requires_human_approval,
        )

        self.sovereign_decisions.append(decision)

        # Log sovereign decision
        await self.immutable_logs.log_governance_action(
            action_type="sovereign_decision",
            data={
                **asdict(decision),
                "action": decision.action.value,  # Convert enum to string
                "risk_assessment": decision.risk_assessment,
                "timestamp": decision.timestamp.isoformat(),  # Convert datetime to string
            },
            transparency_level="democratic_oversight",
        )

        logger.info(
            f"Grace made sovereign decision: {action.value} (confidence: {confidence:.2f})"
        )
        return decision

    def _assess_decision_risk(
        self, action: SovereignAction, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess risk levels for a sovereign decision."""

        base_risks = {
            "technical_risk": 0.2,
            "security_risk": 0.1,
            "operational_risk": 0.15,
            "governance_risk": 0.1,
        }

        # Adjust risks based on action type
        if action == SovereignAction.CREATE_EXPERIMENT:
            base_risks["technical_risk"] += 0.1
        elif action == SovereignAction.MERGE_APPROVED:
            base_risks["operational_risk"] += 0.2
            base_risks["governance_risk"] += 0.15
        elif action == SovereignAction.EVOLVE_GOVERNANCE:
            base_risks["governance_risk"] += 0.3

        # Adjust based on context
        if context.get("validation_level") == "critical":
            for risk_type in base_risks:
                base_risks[risk_type] *= 0.8  # Lower risk if thoroughly validated

        return base_risks

    async def _simulate_experiment_work(
        self, sandbox: GraceSandbox, experiment_id: str
    ):
        """Simulate Grace doing experimental work in her sandbox."""

        # This would be replaced with actual experimental work
        await asyncio.sleep(2.0)  # Simulate some processing time

        # Complete the experiment with simulated results
        results = {
            "hypothesis_confirmed": True,
            "improvements_identified": [
                "Better error handling",
                "Performance optimization",
            ],
            "metrics": {
                "execution_time_improvement": 0.15,
                "memory_usage_reduction": 0.08,
                "reliability_increase": 0.12,
            },
            "new_capabilities": [
                "Enhanced pattern recognition",
                "Improved decision confidence",
            ],
            "lessons_learned": [
                "Iterative validation is crucial",
                "Multi-perspective consensus adds value",
            ],
        }

        await sandbox.complete_experiment(experiment_id, results, success=True)

        # Update sandbox trust score based on results
        sandbox.metrics.trust_score = min(1.0, sandbox.metrics.trust_score + 0.1)

    def _assess_impact(
        self, sandbox: GraceSandbox, validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the potential impact of merging a sandbox."""

        return {
            "affected_systems": ["governance", "decision_making", "learning"],
            "impact_magnitude": "medium",
            "reversibility": "high",
            "confidence_in_assessment": validation_results.get("overall_score", 0.5),
            "benefits": sandbox.experiments[0].results.get(
                "improvements_identified", []
            )
            if sandbox.experiments
            else [],
            "risks": ["Potential integration issues", "Learning curve adjustment"],
        }

    def _determine_human_approval_requirement(
        self, validation_results: Dict[str, Any], consensus_result: ConsensusResult
    ) -> bool:
        """Determine if human approval is required for merge."""

        # Always require human approval if autonomy is low
        if self.autonomy_level < 0.7:
            return True

        # Require if validation had warnings or failures
        if validation_results.get("overall_status") != "passed":
            return True

        # Require if consensus was not strong
        if consensus_result.approval_percentage < 0.8:
            return True

        # Require if there are unresolved requirements
        if consensus_result.requirements_for_approval:
            return True

        # High autonomy and strong results = auto-approve
        return False

    async def _auto_approve_merge(
        self, sandbox_id: str, consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Auto-approve merge when Grace has high autonomy."""

        merge_decision = await self._make_sovereign_decision(
            action=SovereignAction.MERGE_APPROVED,
            context={
                "sandbox_id": sandbox_id,
                "consensus_result": asdict(consensus_result),
            },
            reasoning="High autonomy level enables automatic merge approval",
            confidence=self.autonomy_level,
        )

        # Perform the actual merge
        merge_result = await self.sandbox_manager.approve_merge(
            sandbox_id=sandbox_id,
            human_approval=False,  # Grace is acting autonomously
        )

        # Learn from this successful autonomous action
        await self._learn_from_autonomous_action(merge_decision, merge_result)

        return merge_result

    async def _request_human_feedback_on_rejection(
        self, sandbox_id: str, consensus_result: ConsensusResult
    ):
        """Request human feedback when sandbox is rejected by consensus."""

        await self.human_feedback.request_feedback(
            feedback_type=FeedbackType.GENERAL_GUIDANCE,
            target_id=sandbox_id,
            title="Sandbox rejected by consensus - guidance needed",
            description=f"Sandbox {sandbox_id} was rejected. Grace seeks guidance on improvement.",
            context_data={
                "consensus_result": asdict(consensus_result),
                "rejection_reasons": consensus_result.requirements_for_approval,
            },
            priority="high",
        )

    async def _handle_feedback_callback(self, event_type: str, data: Any):
        """Handle callbacks from the human feedback system."""

        if event_type == "feedback_submitted":
            await self._process_human_feedback(data)
        elif event_type == "feedback_requested":
            logger.info(f"Feedback requested: {data.title}")

    async def _process_human_feedback(self, feedback):
        """Process human feedback and learn from it."""

        learning_decision = await self._make_sovereign_decision(
            action=SovereignAction.LEARN_FROM_FEEDBACK,
            context={
                "feedback_id": feedback.feedback_id,
                "target_id": feedback.target_id,
            },
            reasoning="Integrating human feedback into learning processes",
            requires_human_approval=False,
        )

        # Update autonomy based on feedback
        if feedback.score.value >= 4:  # Good feedback
            self.autonomy_level = min(1.0, self.autonomy_level + 0.02)
        elif feedback.score.value <= 2:  # Poor feedback
            self.autonomy_level = max(0.3, self.autonomy_level - 0.05)

        # Store learning data
        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "feedback_score": feedback.score.value,
            "feedback_type": feedback.feedback_type.value,
            "target_id": feedback.target_id,
            "learning_applied": True,
            "autonomy_adjustment": self.autonomy_level,
        }

        if feedback.score.value >= 3:
            self.governance_learning_data["successful_patterns"].append(learning_entry)
        else:
            self.governance_learning_data["failed_patterns"].append(learning_entry)

        logger.info(
            f"Processed feedback {feedback.feedback_id}, autonomy now: {self.autonomy_level:.2f}"
        )

    async def _learn_from_autonomous_action(
        self, decision: SovereignDecision, result: Dict[str, Any]
    ):
        """Learn from successful autonomous actions to increase confidence."""

        if result.get("status") == "merged":
            # Successful autonomous action
            self.autonomy_level = min(1.0, self.autonomy_level + 0.01)

            learning_entry = {
                "timestamp": datetime.now().isoformat(),
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "success": True,
                "confidence_at_decision": decision.confidence,
                "autonomy_increase": 0.01,
            }

            self.governance_learning_data["successful_patterns"].append(learning_entry)

    def _start_sovereign_processes(self):
        """Start background processes for sovereign operation."""

        # Start meta-learning process
        asyncio.create_task(self._meta_learning_loop())

        # Start autonomy calibration
        asyncio.create_task(self._autonomy_calibration_loop())

    async def _meta_learning_loop(self):
        """Continuously learn and adapt governance based on experience."""

        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                if self.learning_enabled:
                    await self._perform_meta_learning()

            except Exception as e:
                logger.error(f"Error in meta-learning loop: {e}")

    async def _perform_meta_learning(self):
        """Perform meta-learning analysis and governance evolution."""

        # Analyze patterns
        successful_patterns = self.governance_learning_data["successful_patterns"]
        failed_patterns = self.governance_learning_data["failed_patterns"]

        if len(successful_patterns) + len(failed_patterns) < 10:
            return  # Need more data

        evolution_decision = await self._make_sovereign_decision(
            action=SovereignAction.EVOLVE_GOVERNANCE,
            context={"pattern_count": len(successful_patterns) + len(failed_patterns)},
            reasoning="Evolving governance based on accumulated experience",
            confidence=0.8,
        )

        # Identify successful patterns
        success_rate = len(successful_patterns) / (
            len(successful_patterns) + len(failed_patterns)
        )

        if success_rate > 0.7:
            # Good performance, can increase autonomy slightly
            self.autonomy_level = min(0.95, self.autonomy_level + 0.05)
        elif success_rate < 0.5:
            # Poor performance, decrease autonomy
            self.autonomy_level = max(0.4, self.autonomy_level - 0.1)

        # Log governance evolution
        await self.immutable_logs.log_governance_action(
            action_type="governance_evolved",
            data={
                "success_rate": success_rate,
                "new_autonomy_level": self.autonomy_level,
                "patterns_analyzed": len(successful_patterns) + len(failed_patterns),
            },
            transparency_level="democratic_oversight",
        )

        logger.info(f"Governance evolved: autonomy level now {self.autonomy_level:.2f}")

    async def _autonomy_calibration_loop(self):
        """Calibrate autonomy level based on recent performance."""

        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes

                # Get recent feedback
                recent_feedback = self.human_feedback.get_feedback_history(limit=20)

                if len(recent_feedback) >= 5:
                    avg_score = sum(f["score"] for f in recent_feedback) / len(
                        recent_feedback
                    )

                    if avg_score >= 4.0:  # Consistently good feedback
                        self.autonomy_level = min(0.95, self.autonomy_level + 0.01)
                    elif avg_score <= 2.5:  # Consistently poor feedback
                        self.autonomy_level = max(0.3, self.autonomy_level - 0.02)

            except Exception as e:
                logger.error(f"Error in autonomy calibration: {e}")

    async def get_orb_dashboard_data(self) -> OrbitDashboardData:
        """Get comprehensive dashboard data for the Orb interface."""

        # Get data from all components
        sandbox_data = await self.sandbox_manager.get_sandbox_dashboard_data()
        validation_history = await self.validation_harness.get_validation_history(
            limit=10
        )
        consensus_history = await self.consensus_engine.get_consensus_history(limit=10)
        feedback_metrics = self.human_feedback.get_system_feedback_metrics()

        # Grace's autonomy metrics
        autonomy_metrics = {
            "current_autonomy_level": self.autonomy_level,
            "recent_sovereign_decisions": len(
                [
                    d
                    for d in self.sovereign_decisions
                    if (datetime.now() - d.timestamp).days < 7
                ]
            ),
            "successful_autonomous_actions": len(
                self.governance_learning_data["successful_patterns"]
            ),
            "learning_enabled": self.learning_enabled,
            "trust_calibration": self.trust_monitor.get_trust_score(
                "sovereign_orchestrator"
            ).score
            if self.trust_monitor
            else 0.5,
        }

        # System health overview
        active_sandboxes = len(await self.sandbox_manager.list_sandboxes())
        pending_validations = len(
            [v for v in validation_history if v.get("overall_status") == "pending"]
        )
        pending_consensus = len(consensus_history)  # Simplified
        pending_feedback = len(self.human_feedback.get_pending_feedback_requests())

        system_health = {
            "overall_status": "healthy"
            if active_sandboxes < 10 and pending_feedback < 5
            else "busy",
            "active_components": 4,  # sandbox_manager, validation_harness, consensus_engine, human_feedback
            "process_queue_size": pending_validations
            + pending_consensus
            + pending_feedback,
            "learning_active": self.learning_enabled,
            "last_meta_learning": "1 hour ago",  # Would be calculated from actual data
        }

        return OrbitDashboardData(
            sandbox_overview=sandbox_data["dashboard"],
            active_experiments=sandbox_data["dashboard"]["recent_experiments"],
            validation_status={
                "recent_validations": validation_history,
                "pending_validations": pending_validations,
            },
            consensus_status={
                "recent_consensus": consensus_history,
                "persona_reliability": self.consensus_engine.get_persona_reliability_scores(),
            },
            human_feedback_status=feedback_metrics,
            grace_autonomy_metrics=autonomy_metrics,
            recent_sovereign_decisions=[
                {
                    **asdict(d),
                    "action": d.action.value,  # Convert enum to string
                    "timestamp": d.timestamp.isoformat(),  # Convert datetime to string
                }
                for d in self.sovereign_decisions[-10:]
            ],
            system_health=system_health,
        )

    async def get_sovereignty_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on Grace's sovereign sandbox operations."""

        # Collect comprehensive metrics
        orb_data = await self.get_orb_dashboard_data()
        feedback_insights = await self.human_feedback.generate_feedback_insights()

        # Calculate sovereignty metrics
        recent_decisions = [
            d
            for d in self.sovereign_decisions
            if (datetime.now() - d.timestamp).days < 30
        ]

        autonomous_decisions = [
            d for d in recent_decisions if not d.requires_human_approval
        ]

        sovereignty_metrics = {
            "autonomy_level": self.autonomy_level,
            "total_sovereign_decisions": len(self.sovereign_decisions),
            "recent_decisions_30_days": len(recent_decisions),
            "autonomous_decisions_ratio": len(autonomous_decisions)
            / len(recent_decisions)
            if recent_decisions
            else 0.0,
            "avg_decision_confidence": sum(d.confidence for d in recent_decisions)
            / len(recent_decisions)
            if recent_decisions
            else 0.0,
            "learning_data_points": len(
                self.governance_learning_data["successful_patterns"]
            )
            + len(self.governance_learning_data["failed_patterns"]),
        }

        return {
            "report_timestamp": datetime.now().isoformat(),
            "sovereignty_metrics": sovereignty_metrics,
            "orb_dashboard_data": asdict(orb_data),
            "feedback_insights": feedback_insights,
            "governance_evolution": self.governance_learning_data,
            "system_status": "operational",
            "recommendations": self._generate_sovereignty_recommendations(
                sovereignty_metrics
            ),
        }

    def _generate_sovereignty_recommendations(
        self, metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving Grace's sovereignty."""

        recommendations = []

        if metrics["autonomy_level"] < 0.5:
            recommendations.append(
                "Consider increasing validation automation to build confidence"
            )

        if metrics["autonomous_decisions_ratio"] < 0.3:
            recommendations.append(
                "Gradually increase autonomous decision-making in low-risk scenarios"
            )

        if metrics["avg_decision_confidence"] < 0.6:
            recommendations.append(
                "Focus on improving decision-making calibration through more feedback"
            )

        if metrics["learning_data_points"] < 50:
            recommendations.append(
                "Collect more operational data to improve meta-learning effectiveness"
            )

        return recommendations

    async def shutdown(self):
        """Gracefully shutdown the sovereign sandbox orchestrator."""

        logger.info("Shutting down Sovereign Sandbox Orchestrator...")

        # Shutdown all components
        await self.sandbox_manager.shutdown()

        # Export final learning data
        final_report = await self.get_sovereignty_report()

        await self.immutable_logs.log_governance_action(
            action_type="sovereign_orchestrator_shutdown",
            data={
                "final_autonomy_level": self.autonomy_level,
                "total_decisions_made": len(self.sovereign_decisions),
                "shutdown_report": final_report,
            },
            transparency_level="democratic_oversight",
        )

        logger.info("Sovereign Sandbox Orchestrator shutdown complete")
