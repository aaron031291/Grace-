"""
Multi-OS Kernel Scheduler - Task placement and orchestration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid


logger = logging.getLogger(__name__)


class Scheduler:
    """
    Multi-OS task scheduler with placement optimization and health management.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.placement_weights = self.config.get("placement", {}).get("weights", {})
        self.host_scores = {}  # Cache for host scoring
        self.placement_history = []  # Track placement decisions for learning

        logger.info("Multi-OS Scheduler initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default scheduler configuration."""
        return {
            "placement": {
                "weights": {
                    "capability_fit": 0.4,
                    "latency": 0.25,
                    "success": 0.25,
                    "gpu": 0.1,
                }
            },
            "timeouts": {"placement_timeout": 30, "health_check_interval": 60},
            "fallback": {
                "retry_same_os": True,
                "diversify_on_failure": True,
                "max_retries": 3,
            },
        }

    async def place(
        self, exec_task: Dict[str, Any], hosts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find the optimal host for task placement.

        Args:
            exec_task: ExecTask specification
            hosts: List of available HostDescriptor objects

        Returns:
            Placement decision with selected host and reasoning
        """
        try:
            placement_id = str(uuid.uuid4())

            # Filter hosts by basic constraints
            eligible_hosts = self._filter_eligible_hosts(exec_task, hosts)

            if not eligible_hosts:
                return {
                    "placement_id": placement_id,
                    "success": False,
                    "reason": "No eligible hosts found",
                    "host_id": None,
                    "score": 0,
                    "constraints_checked": self._extract_constraints(exec_task),
                }

            # Score eligible hosts
            scored_hosts = []
            for host in eligible_hosts:
                score = await self._score_host(host, exec_task)
                scored_hosts.append((host, score))

            # Select best host
            scored_hosts.sort(key=lambda x: x[1], reverse=True)
            selected_host, best_score = scored_hosts[0]

            # Record placement decision
            placement_decision = {
                "placement_id": placement_id,
                "success": True,
                "host_id": selected_host["host_id"],
                "score": best_score,
                "task_id": exec_task.get("task_id"),
                "constraints": self._extract_constraints(exec_task),
                "alternatives": [
                    (h["host_id"], s) for h, s in scored_hosts[1:3]
                ],  # Top 2 alternatives
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning": self._explain_placement(
                    selected_host, exec_task, best_score
                ),
            }

            self.placement_history.append(placement_decision)

            logger.info(
                f"Task {exec_task.get('task_id')} placed on {selected_host['host_id']} with score {best_score:.3f}"
            )

            return placement_decision

        except Exception as e:
            logger.error(f"Placement error: {e}")
            return {
                "placement_id": str(uuid.uuid4()),
                "success": False,
                "reason": f"Placement error: {str(e)}",
                "host_id": None,
                "score": 0,
            }

    async def submit(self, host_id: str, exec_task: Dict[str, Any]) -> str:
        """
        Submit task to a specific host.

        Args:
            host_id: Target host identifier
            exec_task: ExecTask specification

        Returns:
            Task submission ID
        """
        submission_id = str(uuid.uuid4())

        # In a real implementation, this would:
        # 1. Validate host availability
        # 2. Submit task via host agent API
        # 3. Set up monitoring
        # 4. Return tracking ID

        logger.info(f"Task {exec_task.get('task_id')} submitted to host {host_id}")

        return submission_id

    def _filter_eligible_hosts(
        self, exec_task: Dict[str, Any], hosts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter hosts that can satisfy task constraints."""
        constraints = exec_task.get("constraints", {})
        eligible = []

        for host in hosts:
            # Check OS constraint
            if constraints.get("os") and host["os"] not in constraints["os"]:
                continue

            # Check architecture constraint
            if constraints.get("arch") and host["arch"] not in constraints["arch"]:
                continue

            # Check status
            if host["status"] != "online":
                continue

            # Check capability requirements
            required_capabilities = self._infer_capabilities(exec_task, constraints)
            host_capabilities = set(host.get("capabilities", []))

            if not required_capabilities.issubset(host_capabilities):
                continue

            # Check GPU requirement
            if constraints.get("gpu_required", False):
                if "gpu" not in host_capabilities:
                    continue

                # Check for GPU in labels
                gpu_labels = [
                    label
                    for label in host.get("labels", [])
                    if label.startswith("gpu:")
                ]
                if not gpu_labels or any("none" in label for label in gpu_labels):
                    continue

            eligible.append(host)

        return eligible

    async def _score_host(
        self, host: Dict[str, Any], exec_task: Dict[str, Any]
    ) -> float:
        """
        Score a host for task placement based on multiple factors.

        Returns score between 0.0 and 1.0 (higher is better).
        """
        weights = self.placement_weights
        score = 0.0

        # 1. Capability fit (how well host capabilities match task needs)
        capability_score = self._score_capability_fit(host, exec_task)
        score += weights.get("capability_fit", 0.4) * capability_score

        # 2. Latency/performance score (mock - would use real metrics)
        latency_score = self._score_latency(host, exec_task)
        score += weights.get("latency", 0.25) * latency_score

        # 3. Success rate score (historical success on similar tasks)
        success_score = self._score_success_rate(host, exec_task)
        score += weights.get("success", 0.25) * success_score

        # 4. GPU availability score
        gpu_score = self._score_gpu(host, exec_task)
        score += weights.get("gpu", 0.1) * gpu_score

        # Apply penalties
        penalties = self._calculate_penalties(host, exec_task)
        score = max(0.0, score - penalties)

        return min(1.0, score)

    def _score_capability_fit(
        self, host: Dict[str, Any], exec_task: Dict[str, Any]
    ) -> float:
        """Score how well host capabilities match task requirements."""
        constraints = exec_task.get("constraints", {})
        required_capabilities = self._infer_capabilities(exec_task, constraints)
        host_capabilities = set(host.get("capabilities", []))

        if not required_capabilities:
            return 1.0

        match_count = len(required_capabilities.intersection(host_capabilities))
        return match_count / len(required_capabilities)

    def _score_latency(self, host: Dict[str, Any], exec_task: Dict[str, Any]) -> float:
        """Score host based on expected latency/performance."""
        # Mock implementation - would use real metrics
        # Consider factors like:
        # - Geographic distance
        # - Network latency
        # - Host load
        # - Historical performance

        labels = host.get("labels", [])

        # Prefer local region (mock)
        region_bonus = 0.0
        for label in labels:
            if label.startswith("region:"):
                # Mock: prefer us-west region
                if "us-west" in label:
                    region_bonus = 0.2
                elif "us-" in label:
                    region_bonus = 0.1

        base_score = 0.7  # Default performance assumption
        return min(1.0, base_score + region_bonus)

    def _score_success_rate(
        self, host: Dict[str, Any], exec_task: Dict[str, Any]
    ) -> float:
        """Score host based on historical success rate."""
        # Mock implementation - would use real metrics from database
        # Factors to consider:
        # - Task completion rate
        # - Error frequency
        # - Resource constraint violations
        # - Similar task performance

        host_id = host["host_id"]

        # Mock: assign success scores based on host patterns
        if "linux" in host_id:
            return 0.85  # Linux hosts tend to be more stable
        elif "mac" in host_id:
            return 0.80  # macOS hosts are generally stable
        elif "win" in host_id:
            return 0.75  # Windows hosts may have more variability

        return 0.70  # Default assumption

    def _score_gpu(self, host: Dict[str, Any], exec_task: Dict[str, Any]) -> float:
        """Score GPU availability and suitability."""
        constraints = exec_task.get("constraints", {})

        if not constraints.get("gpu_required", False):
            return 1.0  # GPU not needed, perfect score

        labels = host.get("labels", [])
        gpu_labels = [label for label in labels if label.startswith("gpu:")]

        if not gpu_labels:
            return 0.0  # No GPU info available

        # Score different GPU types
        for label in gpu_labels:
            if "a100" in label.lower():
                return 1.0  # Best GPU
            elif "nvidia" in label.lower():
                return 0.8  # Good GPU
            elif "apple" in label.lower():
                return 0.6  # Decent for macOS
            elif "none" in label.lower():
                return 0.0  # No GPU

        return 0.5  # Unknown GPU type

    def _calculate_penalties(
        self, host: Dict[str, Any], exec_task: Dict[str, Any]
    ) -> float:
        """Calculate placement penalties based on various factors."""
        penalties = 0.0

        # Penalty for degraded hosts
        if host.get("status") == "degraded":
            penalties += 0.2

        # Penalty for very old agent versions (mock)
        agent_version = host.get("agent_version", "0.0.0")
        try:
            version_parts = [int(x) for x in agent_version.split(".")[:2]]
            if version_parts < [2, 0]:  # Versions older than 2.0
                penalties += 0.3
        except (ValueError, IndexError):
            penalties += 0.1  # Unknown version format

        return penalties

    def _infer_capabilities(
        self, exec_task: Dict[str, Any], constraints: Dict[str, Any]
    ) -> set:
        """Infer required capabilities from task specification."""
        required = {"process"}  # All tasks need process capability

        # Check if filesystem operations are needed
        io_spec = exec_task.get("io", {})
        if io_spec.get("files_in") or io_spec.get("files_out"):
            required.add("fs")

        # Check for network requirements
        if constraints.get("network_policy") != "none":
            required.add("net")

        # Check for package management needs
        runtime = exec_task.get("runtime", {})
        if runtime.get("packages"):
            required.add("pkg")

        # Check for sandboxing
        if constraints.get("sandbox", "none") != "none":
            required.add("sandbox")

        # Check for GPU
        if constraints.get("gpu_required", False):
            required.add("gpu")

        return required

    def _extract_constraints(self, exec_task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract constraints from task for analysis."""
        return exec_task.get("constraints", {})

    def _explain_placement(
        self, host: Dict[str, Any], exec_task: Dict[str, Any], score: float
    ) -> Dict[str, Any]:
        """Generate explanation for placement decision."""
        return {
            "selected_host": {
                "host_id": host["host_id"],
                "os": host["os"],
                "arch": host["arch"],
                "capabilities": host.get("capabilities", []),
                "status": host["status"],
            },
            "score_breakdown": {
                "total_score": score,
                "capability_fit": self._score_capability_fit(host, exec_task),
                "latency_score": self._score_latency(host, exec_task),
                "success_rate": self._score_success_rate(host, exec_task),
                "gpu_score": self._score_gpu(host, exec_task),
            },
            "requirements_met": {
                "os": exec_task.get("constraints", {}).get("os", ["any"]),
                "arch": exec_task.get("constraints", {}).get("arch", ["any"]),
                "gpu_required": exec_task.get("constraints", {}).get(
                    "gpu_required", False
                ),
                "capabilities_needed": list(
                    self._infer_capabilities(
                        exec_task, exec_task.get("constraints", {})
                    )
                ),
            },
        }

    def get_placement_stats(self) -> Dict[str, Any]:
        """Get placement statistics for monitoring."""
        if not self.placement_history:
            return {
                "total_placements": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "host_distribution": {},
            }

        total = len(self.placement_history)
        successful = sum(1 for p in self.placement_history if p["success"])
        scores = [p["score"] for p in self.placement_history if p["success"]]

        host_counts = {}
        for p in self.placement_history:
            if p["success"]:
                host_id = p["host_id"]
                host_counts[host_id] = host_counts.get(host_id, 0) + 1

        return {
            "total_placements": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "host_distribution": host_counts,
            "recent_failures": [
                p for p in self.placement_history[-10:] if not p["success"]
            ],
        }
