"""
Insight Generator - Analyzes experiences and generates insights using statistical tests and analysis.
"""

import logging
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime

from .contracts import (
    Experience,
    Insight,
    InsightType,
    RecommendationType,
    generate_insight_id,
)


logger = logging.getLogger(__name__)


class InsightGenerator:
    """Analyzes aggregated experiences to generate actionable insights."""

    def __init__(self):
        self.insights: List[Insight] = []
        self.analysis_methods = {
            InsightType.PERFORMANCE: self._analyze_performance,
            InsightType.DRIFT: self._analyze_drift,
            InsightType.FAIRNESS: self._analyze_fairness,
            InsightType.CALIBRATION: self._analyze_calibration,
            InsightType.STABILITY: self._analyze_stability,
            InsightType.GOVERNANCE_ALIGNMENT: self._analyze_governance_alignment,
        }

    async def generate_insights(self, experiences: List[Experience]) -> List[Insight]:
        """Generate insights from a collection of experiences."""
        if not experiences:
            return []

        insights = []

        try:
            # Analyze different aspects
            for insight_type in InsightType:
                analyzer = self.analysis_methods.get(insight_type)
                if analyzer:
                    insight = await analyzer(experiences)
                    if insight:
                        insights.append(insight)
                        self.insights.append(insight)

            logger.info(
                f"Generated {len(insights)} insights from {len(experiences)} experiences"
            )
            return insights

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []

    async def _analyze_performance(
        self, experiences: List[Experience]
    ) -> Optional[Insight]:
        """Analyze performance trends across experiences."""
        performance_metrics = []

        for exp in experiences:
            metrics = exp.signals.get("metrics", {})
            if "f1" in metrics:
                performance_metrics.append(metrics["f1"])
            elif "auroc" in metrics:
                performance_metrics.append(metrics["auroc"])
            elif "rmse" in metrics:
                # Convert RMSE to a "higher is better" metric
                performance_metrics.append(1.0 / (1.0 + metrics["rmse"]))

        if len(performance_metrics) < 2:
            return None

        # Simple trend analysis
        recent_half = performance_metrics[-len(performance_metrics) // 2 :]
        earlier_half = performance_metrics[: len(performance_metrics) // 2]

        if recent_half and earlier_half:
            recent_avg = statistics.mean(recent_half)
            earlier_avg = statistics.mean(earlier_half)

            confidence = min(0.95, 0.5 + abs(recent_avg - earlier_avg) * 2)

            if recent_avg < earlier_avg - 0.05:  # Performance degradation
                return Insight(
                    insight_id=generate_insight_id(),
                    type=InsightType.PERFORMANCE,
                    scope="model",
                    evidence={
                        "before": {"avg_performance": earlier_avg},
                        "after": {"avg_performance": recent_avg},
                        "tests": ["trend_analysis"],
                        "degradation": earlier_avg - recent_avg,
                    },
                    confidence=confidence,
                    recommendation=RecommendationType.RETRAIN,
                    timestamp=datetime.now(),
                )

        return None

    async def _analyze_drift(self, experiences: List[Experience]) -> Optional[Insight]:
        """Analyze data drift patterns."""
        psi_values = []
        feature_alerts = set()

        for exp in experiences:
            drift = exp.signals.get("drift", {})
            if "psi" in drift:
                psi_values.append(drift["psi"])
            if "feature_alerts" in drift:
                feature_alerts.update(drift["feature_alerts"])

        if not psi_values:
            return None

        avg_psi = statistics.mean(psi_values)
        max_psi = max(psi_values)

        if avg_psi > 0.1 or max_psi > 0.25 or len(feature_alerts) > 0:
            confidence = min(0.95, 0.6 + (avg_psi * 2) + (len(feature_alerts) * 0.1))

            return Insight(
                insight_id=generate_insight_id(),
                type=InsightType.DRIFT,
                scope="dataset",
                evidence={
                    "before": None,
                    "after": {"avg_psi": avg_psi, "max_psi": max_psi},
                    "tests": ["psi"],
                    "alerts": list(feature_alerts),
                },
                confidence=confidence,
                recommendation=RecommendationType.RETRAIN
                if avg_psi > 0.2
                else RecommendationType.RECALIBRATE,
                timestamp=datetime.now(),
            )

        return None

    async def _analyze_fairness(
        self, experiences: List[Experience]
    ) -> Optional[Insight]:
        """Analyze fairness metrics across experiences."""
        fairness_deltas = []
        affected_groups = set()

        for exp in experiences:
            fairness = exp.signals.get("fairness", {})
            if "delta" in fairness:
                fairness_deltas.append(fairness["delta"])
            if "groups" in fairness:
                affected_groups.update(fairness["groups"])

        if not fairness_deltas:
            return None

        max_delta = max(fairness_deltas)
        avg_delta = statistics.mean(fairness_deltas)

        if max_delta > 0.05 or avg_delta > 0.02:
            confidence = min(0.95, 0.7 + max_delta * 5)

            return Insight(
                insight_id=generate_insight_id(),
                type=InsightType.FAIRNESS,
                scope="model",
                evidence={
                    "before": None,
                    "after": {"max_delta": max_delta, "avg_delta": avg_delta},
                    "tests": ["fairness_delta"],
                    "affected_groups": list(affected_groups),
                },
                confidence=confidence,
                recommendation=RecommendationType.REWEIGHT,
                timestamp=datetime.now(),
            )

        return None

    async def _analyze_calibration(
        self, experiences: List[Experience]
    ) -> Optional[Insight]:
        """Analyze model calibration quality."""
        calibration_scores = []

        for exp in experiences:
            metrics = exp.signals.get("metrics", {})
            if "calibration" in metrics:
                calibration_scores.append(metrics["calibration"])

        if len(calibration_scores) < 2:
            return None

        avg_calibration = statistics.mean(calibration_scores)
        min_calibration = min(calibration_scores)

        if avg_calibration < 0.85 or min_calibration < 0.7:
            confidence = min(0.95, 0.6 + (1.0 - avg_calibration))

            return Insight(
                insight_id=generate_insight_id(),
                type=InsightType.CALIBRATION,
                scope="model",
                evidence={
                    "before": None,
                    "after": {
                        "avg_calibration": avg_calibration,
                        "min_calibration": min_calibration,
                    },
                    "tests": ["calibration_analysis"],
                },
                confidence=confidence,
                recommendation=RecommendationType.RECALIBRATE,
                timestamp=datetime.now(),
            )

        return None

    async def _analyze_stability(
        self, experiences: List[Experience]
    ) -> Optional[Insight]:
        """Analyze system stability and variance."""
        latencies = []
        error_rates = []

        for exp in experiences:
            latency = exp.signals.get("latency", {})
            if "p95_ms" in latency:
                latencies.append(latency["p95_ms"])

            metrics = exp.signals.get("metrics", {})
            if "error_rate" in metrics:
                error_rates.append(metrics["error_rate"])

        stability_issues = []

        if len(latencies) > 1:
            latency_var = statistics.variance(latencies) if len(latencies) > 1 else 0
            if latency_var > 100:  # High latency variance
                stability_issues.append(f"High latency variance: {latency_var:.2f}")

        if error_rates:
            max_error_rate = max(error_rates)
            if max_error_rate > 0.05:
                stability_issues.append(f"High error rate: {max_error_rate:.3f}")

        if stability_issues:
            confidence = min(0.95, 0.6 + len(stability_issues) * 0.15)

            return Insight(
                insight_id=generate_insight_id(),
                type=InsightType.STABILITY,
                scope="specialist",
                evidence={
                    "before": None,
                    "after": {"issues": stability_issues},
                    "tests": ["variance_analysis"],
                },
                confidence=confidence,
                recommendation=RecommendationType.HPO,
                timestamp=datetime.now(),
            )

        return None

    async def _analyze_governance_alignment(
        self, experiences: List[Experience]
    ) -> Optional[Insight]:
        """Analyze alignment with governance policies."""
        compliance_scores = []
        governance_experiences = []

        for exp in experiences:
            compliance = exp.signals.get("compliance", {})
            if "constitutional" in compliance:
                compliance_scores.append(compliance["constitutional"])

            if exp.source.value == "governance":
                governance_experiences.append(exp)

        if not compliance_scores:
            return None

        avg_compliance = statistics.mean(compliance_scores)
        min_compliance = min(compliance_scores)

        if avg_compliance < 0.9 or min_compliance < 0.8:
            confidence = min(0.95, 0.8 + (1.0 - avg_compliance))

            return Insight(
                insight_id=generate_insight_id(),
                type=InsightType.GOVERNANCE_ALIGNMENT,
                scope="policy",
                evidence={
                    "before": None,
                    "after": {
                        "avg_compliance": avg_compliance,
                        "min_compliance": min_compliance,
                    },
                    "tests": ["compliance_analysis"],
                    "governance_decisions": len(governance_experiences),
                },
                confidence=confidence,
                recommendation=RecommendationType.POLICY_TUNE,
                timestamp=datetime.now(),
            )

        return None

    def get_recent_insights(self, limit: int = 50) -> List[Insight]:
        """Get recent insights."""
        return sorted(self.insights, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get insight generation statistics."""
        by_type = {}
        for insight_type in InsightType:
            by_type[insight_type.value] = len(
                [ins for ins in self.insights if ins.type == insight_type]
            )

        return {
            "total_insights": len(self.insights),
            "by_type": by_type,
            "avg_confidence": statistics.mean([ins.confidence for ins in self.insights])
            if self.insights
            else 0.0,
        }
