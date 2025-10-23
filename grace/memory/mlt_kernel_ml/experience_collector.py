"""
Experience Collector - Subscribes to events and normalizes experiences from various sources.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .contracts import Experience, ExperienceSource, TaskType, generate_experience_id


logger = logging.getLogger(__name__)


class ExperienceCollector:
    """Collects and normalizes experiences from training, inference, governance, and ops."""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.experiences: List[Experience] = []
        self.source_normalizers = {
            ExperienceSource.TRAINING: self._normalize_training_experience,
            ExperienceSource.INFERENCE: self._normalize_inference_experience,
            ExperienceSource.GOVERNANCE: self._normalize_governance_experience,
            ExperienceSource.OPS: self._normalize_ops_experience,
        }

    async def collect_experience(
        self, source: ExperienceSource, raw_data: Dict[str, Any]
    ) -> Experience:
        """Collect and normalize a raw experience."""
        try:
            # Normalize based on source
            normalizer = self.source_normalizers.get(source)
            if not normalizer:
                raise ValueError(f"No normalizer for source: {source}")

            experience = normalizer(raw_data)

            # Store experience
            self.experiences.append(experience)

            # Emit event if event bus available
            if self.event_bus:
                await self.event_bus.publish(
                    "EXPERIENCE_INGESTED", experience.to_dict()
                )

            logger.info(
                f"Collected experience {experience.experience_id} from {source.value}"
            )
            return experience

        except Exception as e:
            logger.error(f"Failed to collect experience from {source.value}: {e}")
            raise

    def _normalize_training_experience(self, raw_data: Dict[str, Any]) -> Experience:
        """Normalize training pipeline experience."""
        return Experience(
            experience_id=generate_experience_id(),
            source=ExperienceSource.TRAINING,
            task=TaskType(raw_data.get("task", "classification")),
            context={
                "dataset_id": raw_data.get("dataset_id", "unknown"),
                "model_key": raw_data.get("model_key", "unknown"),
                "version": raw_data.get("version", "1.0.0"),
                "conditions": raw_data.get("conditions", {}),
            },
            signals={
                "metrics": raw_data.get("metrics", {}),
                "drift": raw_data.get("drift", {"psi": 0.0, "feature_alerts": []}),
                "fairness": raw_data.get("fairness", {"delta": 0.0, "groups": []}),
                "latency": raw_data.get("latency", {"p95_ms": 0}),
                "compliance": raw_data.get("compliance", {"constitutional": 1.0}),
            },
            ground_truth_lag_s=raw_data.get("ground_truth_lag_s", 86400),
            timestamp=datetime.now(),
        )

    def _normalize_inference_experience(self, raw_data: Dict[str, Any]) -> Experience:
        """Normalize inference experience."""
        return Experience(
            experience_id=generate_experience_id(),
            source=ExperienceSource.INFERENCE,
            task=TaskType(raw_data.get("task", "classification")),
            context={
                "dataset_id": raw_data.get("dataset_id", "inference_batch"),
                "model_key": raw_data.get("model_key", "unknown"),
                "version": raw_data.get("version", "1.0.0"),
                "conditions": {
                    "segment": raw_data.get("segment", "default"),
                    "time": raw_data.get("time", datetime.now().isoformat()),
                },
            },
            signals={
                "metrics": {
                    "predictions_made": raw_data.get("predictions_made", 0),
                    "avg_confidence": raw_data.get("avg_confidence", 0.0),
                    "calibration": raw_data.get("calibration", 0.0),
                },
                "drift": raw_data.get("drift", {"psi": 0.0, "feature_alerts": []}),
                "fairness": raw_data.get("fairness", {"delta": 0.0, "groups": []}),
                "latency": raw_data.get("latency", {"p95_ms": 0}),
                "compliance": raw_data.get("compliance", {"constitutional": 1.0}),
            },
            ground_truth_lag_s=raw_data.get(
                "ground_truth_lag_s", 3600
            ),  # Shorter for inference
            timestamp=datetime.now(),
        )

    def _normalize_governance_experience(self, raw_data: Dict[str, Any]) -> Experience:
        """Normalize governance decision experience."""
        return Experience(
            experience_id=generate_experience_id(),
            source=ExperienceSource.GOVERNANCE,
            task=TaskType.CLASSIFICATION,  # Governance decisions are classification tasks
            context={
                "decision_id": raw_data.get("decision_id", "unknown"),
                "model_key": "governance",
                "version": raw_data.get("version", "1.0.0"),
                "conditions": {
                    "decision_type": raw_data.get("decision_type", "unknown"),
                    "confidence_threshold": raw_data.get("confidence_threshold", 0.5),
                },
            },
            signals={
                "metrics": {
                    "decision_confidence": raw_data.get("decision_confidence", 0.0),
                    "consensus_score": raw_data.get("consensus_score", 0.0),
                    "approval_rate": raw_data.get("approval_rate", 0.0),
                },
                "drift": {"psi": 0.0, "feature_alerts": []},
                "fairness": raw_data.get("fairness", {"delta": 0.0, "groups": []}),
                "latency": raw_data.get("latency", {"p95_ms": 0}),
                "compliance": raw_data.get("compliance", {"constitutional": 1.0}),
            },
            ground_truth_lag_s=raw_data.get("ground_truth_lag_s", 86400),
            timestamp=datetime.now(),
        )

    def _normalize_ops_experience(self, raw_data: Dict[str, Any]) -> Experience:
        """Normalize operational experience."""
        return Experience(
            experience_id=generate_experience_id(),
            source=ExperienceSource.OPS,
            task=TaskType(raw_data.get("task", "classification")),
            context={
                "system_component": raw_data.get("system_component", "unknown"),
                "model_key": raw_data.get("model_key", "ops"),
                "version": raw_data.get("version", "1.0.0"),
                "conditions": {
                    "resource_usage": raw_data.get("resource_usage", {}),
                    "error_rate": raw_data.get("error_rate", 0.0),
                },
            },
            signals={
                "metrics": {
                    "uptime": raw_data.get("uptime", 1.0),
                    "throughput": raw_data.get("throughput", 0.0),
                    "error_rate": raw_data.get("error_rate", 0.0),
                },
                "drift": {"psi": 0.0, "feature_alerts": []},
                "fairness": {"delta": 0.0, "groups": []},
                "latency": raw_data.get("latency", {"p95_ms": 0}),
                "compliance": {"constitutional": 1.0},
            },
            ground_truth_lag_s=raw_data.get("ground_truth_lag_s", 300),  # Short for ops
            timestamp=datetime.now(),
        )

    def get_recent_experiences(
        self, limit: int = 100, source: Optional[ExperienceSource] = None
    ) -> List[Experience]:
        """Get recent experiences, optionally filtered by source."""
        experiences = self.experiences

        if source:
            experiences = [exp for exp in experiences if exp.source == source]

        # Sort by timestamp descending
        experiences.sort(key=lambda x: x.timestamp, reverse=True)

        return experiences[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        by_source = {}
        for source in ExperienceSource:
            by_source[source.value] = len(
                [exp for exp in self.experiences if exp.source == source]
            )

        return {
            "total_experiences": len(self.experiences),
            "by_source": by_source,
            "latest_timestamp": max([exp.timestamp for exp in self.experiences])
            if self.experiences
            else None,
        }
