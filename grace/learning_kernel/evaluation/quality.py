"""Data quality evaluation including bias, fairness, leakage detection, and drift monitoring."""

import json
import sqlite3
import random
from datetime import datetime
from typing import Dict, List, Optional, Any


class QualityEvaluator:
    """Evaluates dataset quality including bias, fairness, leakage, and drift."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def generate_report(
        self, dataset_id: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        report_id = f"qr_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get dataset info
        dataset_info = self._get_dataset_info(dataset_id, version)
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Run all quality checks
        leakage_flags = self._detect_leakage(dataset_id, version)
        bias_metrics = self._analyze_bias_fairness(dataset_id, version)
        coverage_metrics = self._analyze_coverage_balance(dataset_id, version)
        noise_metrics = self._analyze_label_noise(dataset_id, version)
        drift_metrics = self._analyze_data_drift(dataset_id, version)

        # Aggregate overall quality score
        quality_score = self._calculate_overall_quality_score(
            {
                "leakage_flags": leakage_flags,
                "bias_score": bias_metrics.get("overall_bias_score", 0.5),
                "coverage_score": coverage_metrics.get("coverage_score", 0.5),
                "noise_score": noise_metrics.get("noise_score", 0.5),
                "drift_score": drift_metrics.get("drift_score", 0.5),
            }
        )

        report = {
            "report_id": report_id,
            "dataset_id": dataset_id,
            "version": version,
            "generated_at": datetime.now().isoformat(),
            "overall_quality_score": quality_score,
            "metrics": {
                "leakage": {
                    "flags_count": leakage_flags,
                    "severity": "critical" if leakage_flags > 0 else "none",
                },
                "bias_fairness": bias_metrics,
                "coverage_balance": coverage_metrics,
                "label_noise": noise_metrics,
                "data_drift": drift_metrics,
            },
            "recommendations": self._generate_recommendations(
                leakage_flags,
                bias_metrics,
                coverage_metrics,
                noise_metrics,
                drift_metrics,
            ),
        }

        # Store report in database
        self._store_report(report)

        return report

    def _get_dataset_info(
        self, dataset_id: str, version: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get basic dataset information."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            if version:
                cursor.execute(
                    """
                    SELECT d.*, dv.row_count, dv.byte_size, dv.stats_json
                    FROM datasets d
                    JOIN dataset_versions dv ON d.dataset_id = dv.dataset_id
                    WHERE d.dataset_id = ? AND dv.version = ?
                """,
                    (dataset_id, version),
                )
            else:
                cursor.execute(
                    """
                    SELECT d.*, dv.row_count, dv.byte_size, dv.stats_json
                    FROM datasets d
                    LEFT JOIN dataset_versions dv ON d.dataset_id = dv.dataset_id
                    WHERE d.dataset_id = ?
                    ORDER BY dv.created_at DESC LIMIT 1
                """,
                    (dataset_id,),
                )

            row = cursor.fetchone()
            if not row:
                return None

            return {
                "dataset_id": row["dataset_id"],
                "task": row["task"],
                "modality": row["modality"],
                "row_count": row["row_count"] or 0,
                "byte_size": row["byte_size"] or 0,
                "stats": json.loads(row["stats_json"] or "{}"),
            }
        finally:
            conn.close()

    def _detect_leakage(self, dataset_id: str, version: Optional[str]) -> int:
        """Detect various types of data leakage."""
        leakage_flags = 0

        # Mock leakage detection - in practice would implement sophisticated detection
        # 1. Train/test overlap detection
        train_test_overlap = self._detect_train_test_overlap(dataset_id, version)
        if train_test_overlap:
            leakage_flags += 1

        # 2. Target leakage detection
        target_leakage = self._detect_target_leakage(dataset_id, version)
        if target_leakage:
            leakage_flags += 1

        # 3. Temporal leakage detection
        temporal_leakage = self._detect_temporal_leakage(dataset_id, version)
        if temporal_leakage:
            leakage_flags += 1

        return leakage_flags

    def _detect_train_test_overlap(
        self, dataset_id: str, version: Optional[str]
    ) -> bool:
        """Detect overlap between train and test sets."""
        # Mock implementation - would check for identical or near-identical samples
        return random.random() < 0.05  # 5% chance of detecting overlap

    def _detect_target_leakage(self, dataset_id: str, version: Optional[str]) -> bool:
        """Detect features that leak information about the target."""
        # Mock implementation - would analyze feature-target correlations
        return random.random() < 0.03  # 3% chance of detecting target leakage

    def _detect_temporal_leakage(self, dataset_id: str, version: Optional[str]) -> bool:
        """Detect temporal leakage (future information in features)."""
        # Mock implementation - would check timestamps and causality
        return random.random() < 0.02  # 2% chance of detecting temporal leakage

    def _analyze_bias_fairness(
        self, dataset_id: str, version: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze bias and fairness across sensitive attributes."""
        # Mock sensitive attributes - in practice would be configurable
        sensitive_attrs = ["gender", "age_group", "race", "geography"]

        bias_metrics = {}
        overall_bias_scores = []

        for attr in sensitive_attrs:
            # Mock bias analysis
            group_performance = {
                "group_a": {
                    "accuracy": random.uniform(0.7, 0.9),
                    "precision": random.uniform(0.7, 0.9),
                },
                "group_b": {
                    "accuracy": random.uniform(0.6, 0.85),
                    "precision": random.uniform(0.6, 0.85),
                },
            }

            # Calculate bias metrics
            accuracy_diff = abs(
                group_performance["group_a"]["accuracy"]
                - group_performance["group_b"]["accuracy"]
            )
            precision_diff = abs(
                group_performance["group_a"]["precision"]
                - group_performance["group_b"]["precision"]
            )

            bias_score = max(accuracy_diff, precision_diff)
            overall_bias_scores.append(bias_score)

            bias_metrics[attr] = {
                "accuracy_difference": round(accuracy_diff, 3),
                "precision_difference": round(precision_diff, 3),
                "bias_score": round(bias_score, 3),
                "group_performance": group_performance,
                "severity": "high"
                if bias_score > 0.15
                else "medium"
                if bias_score > 0.08
                else "low",
            }

        overall_bias_score = (
            sum(overall_bias_scores) / len(overall_bias_scores)
            if overall_bias_scores
            else 0.0
        )

        return {
            "overall_bias_score": round(overall_bias_score, 3),
            "sensitive_attributes": bias_metrics,
            "demographic_parity_violations": sum(
                1 for score in overall_bias_scores if score > 0.1
            ),
            "equalized_odds_violations": sum(
                1 for score in overall_bias_scores if score > 0.08
            ),
        }

    def _analyze_coverage_balance(
        self, dataset_id: str, version: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze label distribution, coverage, and class balance."""
        # Mock label distribution analysis
        mock_labels = ["positive", "negative", "neutral"]
        mock_distribution = {
            "positive": random.randint(100, 1000),
            "negative": random.randint(80, 800),
            "neutral": random.randint(20, 200),
        }

        total_samples = sum(mock_distribution.values())
        label_ratios = {
            label: count / total_samples for label, count in mock_distribution.items()
        }

        # Calculate balance metrics
        max_ratio = max(label_ratios.values())
        min_ratio = min(label_ratios.values())
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float("inf")

        # Coverage analysis
        coverage_ratio = random.uniform(0.7, 0.95)  # Mock coverage
        rare_class_support = min(mock_distribution.values())

        coverage_score = min(coverage_ratio, 1.0 / max(1.0, imbalance_ratio / 10.0))

        return {
            "label_distribution": mock_distribution,
            "label_ratios": {k: round(v, 3) for k, v in label_ratios.items()},
            "imbalance_ratio": round(imbalance_ratio, 2),
            "coverage_ratio": round(coverage_ratio, 3),
            "rare_class_support": rare_class_support,
            "coverage_score": round(coverage_score, 3),
            "balance_severity": "high"
            if imbalance_ratio > 20
            else "medium"
            if imbalance_ratio > 5
            else "low",
        }

    def _analyze_label_noise(
        self, dataset_id: str, version: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze label noise and inter-annotator agreement."""
        # Get label statistics from database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT AVG(agreement) as avg_agreement,
                       COUNT(CASE WHEN gold_correct = 1 THEN 1 END) as gold_correct,
                       COUNT(CASE WHEN gold_correct IS NOT NULL THEN 1 END) as gold_total,
                       AVG(confidence) as avg_confidence
                FROM labels l
                JOIN label_tasks lt ON l.task_id = lt.task_id
                WHERE lt.dataset_id = ?
            """,
                (dataset_id,),
            )

            row = cursor.fetchone()

            if row and row["gold_total"] > 0:
                avg_agreement = row["avg_agreement"] or 0.8
                gold_accuracy = row["gold_correct"] / row["gold_total"]
                avg_confidence = row["avg_confidence"] or 0.75
            else:
                # Mock values if no data
                avg_agreement = random.uniform(0.75, 0.95)
                gold_accuracy = random.uniform(0.8, 0.95)
                avg_confidence = random.uniform(0.7, 0.9)

            # Calculate noise score (inverse of quality indicators)
            noise_score = 1.0 - ((avg_agreement + gold_accuracy + avg_confidence) / 3.0)

            return {
                "avg_inter_annotator_agreement": round(avg_agreement, 3),
                "gold_standard_accuracy": round(gold_accuracy, 3),
                "avg_label_confidence": round(avg_confidence, 3),
                "estimated_noise_rate": round(1.0 - gold_accuracy, 3),
                "noise_score": round(noise_score, 3),
                "quality_level": "high"
                if noise_score < 0.1
                else "medium"
                if noise_score < 0.2
                else "low",
            }

        finally:
            conn.close()

    def _analyze_data_drift(
        self, dataset_id: str, version: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze data drift using Population Stability Index (PSI) and other metrics."""
        # Mock drift analysis - in practice would compare distributions over time

        # Calculate mock PSI for different features
        feature_psi = {}
        features = ["feature_1", "feature_2", "feature_3", "text_length", "num_tokens"]

        overall_psi_scores = []
        for feature in features:
            psi_score = random.uniform(0.0, 0.3)  # PSI typically ranges 0-1+
            overall_psi_scores.append(psi_score)

            feature_psi[feature] = {
                "psi_score": round(psi_score, 3),
                "drift_severity": "high"
                if psi_score > 0.25
                else "medium"
                if psi_score > 0.1
                else "low",
            }

        overall_psi = sum(overall_psi_scores) / len(overall_psi_scores)

        # KL divergence mock
        kl_divergence = random.uniform(0.0, 0.5)

        drift_score = 1.0 - min(1.0, (overall_psi + kl_divergence) / 2.0)

        return {
            "overall_psi": round(overall_psi, 3),
            "kl_divergence": round(kl_divergence, 3),
            "feature_drift": feature_psi,
            "drift_score": round(drift_score, 3),
            "drift_severity": "high"
            if overall_psi > 0.25
            else "medium"
            if overall_psi > 0.1
            else "low",
            "temporal_drift_detected": overall_psi > 0.15,
        }

    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metric scores."""
        # Weight different aspects of quality
        weights = {
            "leakage_penalty": 0.3,  # Heavy penalty for leakage
            "bias_score": 0.25,  # Bias and fairness
            "coverage_score": 0.2,  # Coverage and balance
            "noise_score": 0.15,  # Label noise
            "drift_score": 0.1,  # Data drift
        }

        # Calculate weighted score (higher is better)
        leakage_penalty = metrics["leakage_flags"] * 0.2  # Each leak reduces score
        quality_score = (
            max(0.0, 1.0 - leakage_penalty) * weights["leakage_penalty"]
            + (1.0 - metrics["bias_score"]) * weights["bias_score"]
            + metrics["coverage_score"] * weights["coverage_score"]
            + (1.0 - metrics["noise_score"]) * weights["noise_score"]
            + metrics["drift_score"] * weights["drift_score"]
        )

        return round(quality_score, 3)

    def _generate_recommendations(
        self,
        leakage_flags: int,
        bias_metrics: Dict,
        coverage_metrics: Dict,
        noise_metrics: Dict,
        drift_metrics: Dict,
    ) -> List[str]:
        """Generate actionable recommendations based on quality analysis."""
        recommendations = []

        # Leakage recommendations
        if leakage_flags > 0:
            recommendations.append(
                "CRITICAL: Data leakage detected. Review train/test splits and feature engineering."
            )
            if leakage_flags > 2:
                recommendations.append(
                    "Multiple leakage issues found. Consider complete data pipeline review."
                )

        # Bias recommendations
        if bias_metrics.get("overall_bias_score", 0) > 0.15:
            recommendations.append(
                "High bias detected across sensitive attributes. Consider bias mitigation techniques."
            )

        violations = bias_metrics.get("demographic_parity_violations", 0)
        if violations > 0:
            recommendations.append(
                f"Demographic parity violations found in {violations} attributes. Review fairness constraints."
            )

        # Coverage recommendations
        imbalance_ratio = coverage_metrics.get("imbalance_ratio", 1)
        if imbalance_ratio > 10:
            recommendations.append(
                "Severe class imbalance detected. Consider resampling or stratified approaches."
            )

        if coverage_metrics.get("coverage_ratio", 1.0) < 0.8:
            recommendations.append(
                "Low data coverage. Increase labeling effort or active learning strategies."
            )

        # Noise recommendations
        if noise_metrics.get("noise_score", 0) > 0.2:
            recommendations.append(
                "High label noise detected. Review labeling guidelines and annotator training."
            )

        if noise_metrics.get("avg_inter_annotator_agreement", 1.0) < 0.7:
            recommendations.append(
                "Low inter-annotator agreement. Clarify labeling instructions and provide examples."
            )

        # Drift recommendations
        if drift_metrics.get("overall_psi", 0) > 0.25:
            recommendations.append(
                "Significant data drift detected. Monitor model performance and consider retraining."
            )

        if drift_metrics.get("temporal_drift_detected", False):
            recommendations.append(
                "Temporal drift identified. Implement continuous monitoring and periodic model updates."
            )

        if not recommendations:
            recommendations.append(
                "Data quality looks good! Continue monitoring for any changes."
            )

        return recommendations

    def _store_report(self, report: Dict[str, Any]):
        """Store quality report in database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO quality_reports (
                    report_id, dataset_id, version, leakage_flags, bias_metrics_json,
                    coverage_ratio, label_agreement, drift_psi, noise_estimate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    report["report_id"],
                    report["dataset_id"],
                    report["version"],
                    report["metrics"]["leakage"]["flags_count"],
                    json.dumps(report["metrics"]["bias_fairness"]),
                    report["metrics"]["coverage_balance"]["coverage_ratio"],
                    report["metrics"]["label_noise"]["avg_inter_annotator_agreement"],
                    report["metrics"]["data_drift"]["overall_psi"],
                    report["metrics"]["label_noise"]["noise_score"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_quality_history(
        self, dataset_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get quality report history for a dataset."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT report_id, version, leakage_flags, coverage_ratio,
                       label_agreement, drift_psi, noise_estimate, created_at
                FROM quality_reports
                WHERE dataset_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (dataset_id, limit),
            )

            history = []
            for row in cursor.fetchall():
                history.append(
                    {
                        "report_id": row["report_id"],
                        "version": row["version"],
                        "leakage_flags": row["leakage_flags"],
                        "coverage_ratio": row["coverage_ratio"],
                        "label_agreement": row["label_agreement"],
                        "drift_psi": row["drift_psi"],
                        "noise_estimate": row["noise_estimate"],
                        "created_at": row["created_at"],
                    }
                )

            return history
        finally:
            conn.close()
