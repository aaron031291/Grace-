"""
Grace Sandbox Manager - Version-replicating sandbox system for safe experimentation.

This module implements Grace's sovereign sandbox environment where she can:
- Create isolated experimental branches
- Run multi-disciplinary experiments
- Validate changes through continuous testing
- Maintain provenance of all experiments
- Self-destruct expired or failed experiments
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

from ..layer_04_audit_logs.immutable_logs import ImmutableLogs
from ..core.kpi_trust_monitor import KPITrustMonitor

logger = logging.getLogger(__name__)


class SandboxState(Enum):
    """Sandbox lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    VALIDATING = "validating"
    READY_FOR_MERGE = "ready_for_merge"
    EXPIRED = "expired"
    FAILED = "failed"
    MERGED = "merged"
    DESTROYED = "destroyed"


class ExperimentType(Enum):
    """Types of sandbox experiments."""

    CURIOSITY_DRIVEN = "curiosity_driven"
    PROBLEM_SOLVING = "problem_solving"
    CAPABILITY_ENHANCEMENT = "capability_enhancement"
    ARCHITECTURAL_IMPROVEMENT = "architectural_improvement"
    INTEGRATION_TEST = "integration_test"


@dataclass
class ResourceQuota:
    """Resource quotas for sandbox experiments."""

    max_compute_units: int = 100
    max_memory_mb: int = 2048
    max_storage_mb: int = 1024
    max_api_calls_per_hour: int = 1000
    max_duration_hours: int = 24
    max_concurrent_experiments: int = 5


@dataclass
class SandboxMetrics:
    """Metrics tracking for sandbox performance."""

    creation_time: datetime
    last_activity: datetime
    resource_usage: Dict[str, float]
    trust_score: float
    validation_results: List[Dict[str, Any]]
    human_feedback_score: Optional[float] = None
    experiment_count: int = 0
    success_rate: float = 0.0


@dataclass
class SandboxExperiment:
    """Individual experiment within a sandbox."""

    experiment_id: str
    experiment_type: ExperimentType
    description: str
    hypothesis: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    validation_status: str = "pending"
    specialist_consensus: Optional[Dict[str, Any]] = None


class GraceSandbox:
    """
    Individual sandbox instance for Grace experiments.
    Represents a version-replicating branch with governance oversight.
    """

    def __init__(
        self,
        sandbox_id: str,
        parent_version: str,
        experiment_type: ExperimentType,
        description: str,
        resource_quota: ResourceQuota,
        governance_engine=None,
    ):
        self.sandbox_id = sandbox_id
        self.parent_version = parent_version
        self.experiment_type = experiment_type
        self.description = description
        self.resource_quota = resource_quota
        self.governance_engine = governance_engine

        self.state = SandboxState.PENDING
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(
            hours=resource_quota.max_duration_hours
        )

        self.experiments: List[SandboxExperiment] = []
        self.metrics = SandboxMetrics(
            creation_time=self.created_at,
            last_activity=self.created_at,
            resource_usage={},
            trust_score=0.5,  # Start with neutral trust
            validation_results=[],
        )

        self.auto_cleanup_enabled = True
        self.merge_approved = False
        self.human_review_required = True

    async def start_experiment(
        self, experiment_type: ExperimentType, description: str, hypothesis: str
    ) -> str:
        """Start a new experiment within this sandbox."""
        if self.state not in [SandboxState.ACTIVE, SandboxState.PENDING]:
            raise ValueError(f"Cannot start experiment in sandbox state: {self.state}")

        experiment = SandboxExperiment(
            experiment_id=str(uuid.uuid4()),
            experiment_type=experiment_type,
            description=description,
            hypothesis=hypothesis,
            created_at=datetime.now(),
        )

        self.experiments.append(experiment)
        self.metrics.experiment_count += 1
        self.metrics.last_activity = datetime.now()

        if self.state == SandboxState.PENDING:
            self.state = SandboxState.ACTIVE

        logger.info(
            f"Started experiment {experiment.experiment_id} in sandbox {self.sandbox_id}"
        )
        return experiment.experiment_id

    async def complete_experiment(
        self, experiment_id: str, results: Dict[str, Any], success: bool = True
    ):
        """Complete an experiment with results."""
        experiment = self._get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.completed_at = datetime.now()
        experiment.results = results
        experiment.validation_status = "completed" if success else "failed"

        # Update success rate
        completed_experiments = [
            e for e in self.experiments if e.completed_at is not None
        ]
        successful_experiments = [
            e for e in completed_experiments if e.validation_status == "completed"
        ]
        self.metrics.success_rate = (
            len(successful_experiments) / len(completed_experiments)
            if completed_experiments
            else 0.0
        )

        self.metrics.last_activity = datetime.now()

        logger.info(f"Completed experiment {experiment_id} with success: {success}")

    def _get_experiment(self, experiment_id: str) -> Optional[SandboxExperiment]:
        """Get experiment by ID."""
        return next(
            (e for e in self.experiments if e.experiment_id == experiment_id), None
        )

    async def validate(self) -> Dict[str, Any]:
        """Run validation tests on the sandbox."""
        self.state = SandboxState.VALIDATING

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "resource_compliance": self._check_resource_compliance(),
            "trust_score": self.metrics.trust_score,
            "experiment_results": [asdict(exp) for exp in self.experiments],
            "overall_status": "pending",
        }

        # Simple validation logic
        if (
            self.metrics.trust_score >= 0.6
            and validation_results["resource_compliance"]["compliant"]
            and self.metrics.success_rate >= 0.5
        ):
            validation_results["overall_status"] = "approved"
            self.state = SandboxState.READY_FOR_MERGE
        else:
            validation_results["overall_status"] = "rejected"
            self.state = SandboxState.FAILED

        self.metrics.validation_results.append(validation_results)
        return validation_results

    def _check_resource_compliance(self) -> Dict[str, Any]:
        """Check if sandbox is within resource quotas."""
        current_usage = self.metrics.resource_usage

        return {
            "compliant": True,  # Simplified for now
            "usage": current_usage,
            "limits": asdict(self.resource_quota),
            "violations": [],
        }

    async def destroy(self, reason: str = "expired"):
        """Destroy the sandbox and clean up resources."""
        logger.info(f"Destroying sandbox {self.sandbox_id}, reason: {reason}")
        self.state = SandboxState.DESTROYED
        # Here we would clean up actual resources, files, etc.

    def should_auto_cleanup(self) -> bool:
        """Check if sandbox should be automatically cleaned up."""
        if not self.auto_cleanup_enabled:
            return False

        # Check expiration
        if datetime.now() > self.expires_at:
            return True

        # Check trust score
        if self.metrics.trust_score < 0.3:
            return True

        # Check if failed for too long
        if (
            self.state == SandboxState.FAILED
            and datetime.now() - self.metrics.last_activity > timedelta(hours=1)
        ):
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert sandbox to dictionary for serialization."""
        return {
            "sandbox_id": self.sandbox_id,
            "parent_version": self.parent_version,
            "experiment_type": self.experiment_type.value,
            "description": self.description,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "resource_quota": asdict(self.resource_quota),
            "metrics": asdict(self.metrics),
            "experiments": [asdict(exp) for exp in self.experiments],
            "merge_approved": self.merge_approved,
        }


class SandboxManager:
    """
    Main orchestrator for Grace's sandbox system.
    Manages the lifecycle of sandbox environments and experiments.
    """

    def __init__(
        self,
        event_bus=None,
        immutable_logs: Optional[ImmutableLogs] = None,
        trust_monitor: Optional[KPITrustMonitor] = None,
        governance_engine=None,
    ):
        self.event_bus = event_bus
        self.immutable_logs = immutable_logs or ImmutableLogs("sandbox_audit.db")
        self.trust_monitor = trust_monitor
        self.governance_engine = governance_engine

        self.active_sandboxes: Dict[str, GraceSandbox] = {}
        self.sandbox_history: List[Dict[str, Any]] = []
        self.default_quota = ResourceQuota()

        # Meta-learning for sandbox governance
        self.governance_learning = {
            "success_patterns": [],
            "failure_patterns": [],
            "resource_optimization": {},
        }

        self._cleanup_task = None
        self._start_cleanup_monitor()

    async def create_sandbox(
        self,
        experiment_type: ExperimentType,
        description: str,
        parent_version: str = "main",
        custom_quota: Optional[ResourceQuota] = None,
    ) -> str:
        """Create a new sandbox for experimentation."""

        # Check if we're within limits
        if len(self.active_sandboxes) >= self.default_quota.max_concurrent_experiments:
            raise ValueError(
                f"Maximum concurrent sandboxes ({self.default_quota.max_concurrent_experiments}) reached"
            )

        sandbox_id = f"sandbox_{str(uuid.uuid4())[:8]}"
        quota = custom_quota or self.default_quota

        sandbox = GraceSandbox(
            sandbox_id=sandbox_id,
            parent_version=parent_version,
            experiment_type=experiment_type,
            description=description,
            resource_quota=quota,
            governance_engine=self.governance_engine,
        )

        self.active_sandboxes[sandbox_id] = sandbox

        # Log creation
        await self.immutable_logs.log_governance_action(
            action_type="sandbox_created",
            data={
                "sandbox_id": sandbox_id,
                "experiment_type": experiment_type.value,
                "description": description,
                "parent_version": parent_version,
                "quota": asdict(quota),
            },
            transparency_level="democratic_oversight",
        )

        logger.info(f"Created sandbox {sandbox_id} for {experiment_type.value}")
        return sandbox_id

    async def get_sandbox(self, sandbox_id: str) -> Optional[GraceSandbox]:
        """Get sandbox by ID."""
        return self.active_sandboxes.get(sandbox_id)

    async def list_sandboxes(
        self, state_filter: Optional[SandboxState] = None
    ) -> List[Dict[str, Any]]:
        """List all sandboxes, optionally filtered by state."""
        sandboxes = []
        for sandbox in self.active_sandboxes.values():
            if state_filter is None or sandbox.state == state_filter:
                sandboxes.append(sandbox.to_dict())
        return sandboxes

    async def validate_sandbox(self, sandbox_id: str) -> Dict[str, Any]:
        """Run validation on a specific sandbox."""
        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        validation_results = await sandbox.validate()

        # Log validation
        await self.immutable_logs.log_governance_action(
            action_type="sandbox_validated",
            data={"sandbox_id": sandbox_id, "validation_results": validation_results},
            transparency_level="democratic_oversight",
        )

        return validation_results

    async def approve_merge(
        self, sandbox_id: str, human_approval: bool = True
    ) -> Dict[str, Any]:
        """Approve sandbox for merging into main version."""
        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        if sandbox.state != SandboxState.READY_FOR_MERGE:
            raise ValueError(
                f"Sandbox {sandbox_id} not ready for merge (state: {sandbox.state})"
            )

        if sandbox.human_review_required and not human_approval:
            raise ValueError(f"Sandbox {sandbox_id} requires human approval")

        sandbox.merge_approved = True
        sandbox.state = SandboxState.MERGED

        # Move to history
        self.sandbox_history.append(sandbox.to_dict())
        del self.active_sandboxes[sandbox_id]

        # Log merge
        await self.immutable_logs.log_governance_action(
            action_type="sandbox_merged",
            data={
                "sandbox_id": sandbox_id,
                "human_approval": human_approval,
                "experiments_merged": len(sandbox.experiments),
            },
            transparency_level="democratic_oversight",
        )

        logger.info(f"Approved merge for sandbox {sandbox_id}")
        return {"status": "merged", "sandbox_id": sandbox_id}

    async def destroy_sandbox(
        self, sandbox_id: str, reason: str = "manual"
    ) -> Dict[str, Any]:
        """Destroy a sandbox."""
        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")

        await sandbox.destroy(reason=reason)

        # Move to history
        sandbox_data = sandbox.to_dict()
        sandbox_data["destruction_reason"] = reason
        self.sandbox_history.append(sandbox_data)
        del self.active_sandboxes[sandbox_id]

        # Log destruction
        await self.immutable_logs.log_governance_action(
            action_type="sandbox_destroyed",
            data={
                "sandbox_id": sandbox_id,
                "reason": reason,
                "experiments_lost": len(sandbox.experiments),
            },
            transparency_level="democratic_oversight",
        )

        logger.info(f"Destroyed sandbox {sandbox_id}, reason: {reason}")
        return {"status": "destroyed", "sandbox_id": sandbox_id, "reason": reason}

    def _start_cleanup_monitor(self):
        """Start the automatic cleanup monitoring task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_monitor_loop())

    async def _cleanup_monitor_loop(self):
        """Monitor and cleanup expired or failed sandboxes."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                sandboxes_to_cleanup = []
                for sandbox_id, sandbox in self.active_sandboxes.items():
                    if sandbox.should_auto_cleanup():
                        sandboxes_to_cleanup.append(sandbox_id)

                for sandbox_id in sandboxes_to_cleanup:
                    await self.destroy_sandbox(sandbox_id, reason="auto_cleanup")

                if sandboxes_to_cleanup:
                    logger.info(
                        f"Auto-cleaned up {len(sandboxes_to_cleanup)} sandboxes"
                    )

            except Exception as e:
                logger.error(f"Error in cleanup monitor: {e}")

    async def get_sandbox_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for the Orb interface."""
        active_count = len(self.active_sandboxes)

        states_summary = {}
        for state in SandboxState:
            count = sum(1 for s in self.active_sandboxes.values() if s.state == state)
            states_summary[state.value] = count

        resource_usage = {
            "total_active": active_count,
            "resource_utilization": 0.0,  # Would calculate based on actual usage
            "avg_trust_score": 0.0,
        }

        if active_count > 0:
            total_trust = sum(
                s.metrics.trust_score for s in self.active_sandboxes.values()
            )
            resource_usage["avg_trust_score"] = total_trust / active_count

        return {
            "dashboard": {
                "active_sandboxes": active_count,
                "states_summary": states_summary,
                "resource_usage": resource_usage,
                "recent_experiments": self._get_recent_experiments(),
                "success_metrics": self._get_success_metrics(),
            }
        }

    def _get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent experiments across all sandboxes."""
        all_experiments = []
        for sandbox in self.active_sandboxes.values():
            for exp in sandbox.experiments:
                exp_data = asdict(exp)
                exp_data["sandbox_id"] = sandbox.sandbox_id
                all_experiments.append(exp_data)

        # Sort by creation time, most recent first
        all_experiments.sort(key=lambda x: x["created_at"], reverse=True)
        return all_experiments[:limit]

    def _get_success_metrics(self) -> Dict[str, Any]:
        """Get overall success metrics."""
        total_experiments = sum(
            len(s.experiments) for s in self.active_sandboxes.values()
        )
        total_completed = sum(
            len([e for e in s.experiments if e.completed_at is not None])
            for s in self.active_sandboxes.values()
        )
        total_successful = sum(
            len([e for e in s.experiments if e.validation_status == "completed"])
            for s in self.active_sandboxes.values()
        )

        return {
            "total_experiments": total_experiments,
            "completion_rate": total_completed / total_experiments
            if total_experiments > 0
            else 0.0,
            "success_rate": total_successful / total_completed
            if total_completed > 0
            else 0.0,
            "active_sandbox_health": sum(
                1
                for s in self.active_sandboxes.values()
                if s.metrics.trust_score >= 0.6
            ),
        }

    async def shutdown(self):
        """Shutdown the sandbox manager and cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        # Archive active sandboxes
        for sandbox_id in list(self.active_sandboxes.keys()):
            await self.destroy_sandbox(sandbox_id, reason="shutdown")

        logger.info("Sandbox manager shutdown complete")
