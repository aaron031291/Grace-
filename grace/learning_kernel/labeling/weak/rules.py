"""Weak supervision with label models, heuristics, and Snorkel-like rules."""

import json
import sqlite3
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


def weak_vote(rules_outputs: List[Dict], threshold: float) -> Dict:
    """Aggregate weak supervision outputs using majority voting with confidence threshold."""
    if not rules_outputs:
        return {"prediction": None, "confidence": 0.0, "abstain": True}

    # Collect all predictions with their confidence scores
    predictions = []
    total_confidence = 0.0

    for output in rules_outputs:
        if "prediction" in output and "confidence" in output:
            if output["confidence"] >= threshold:
                predictions.append(output["prediction"])
                total_confidence += output["confidence"]

    if not predictions:
        return {"prediction": None, "confidence": 0.0, "abstain": True}

    # Simple majority voting
    from collections import Counter

    vote_counts = Counter(predictions)

    if not vote_counts:
        return {"prediction": None, "confidence": 0.0, "abstain": True}

    # Get most common prediction
    most_common = vote_counts.most_common(1)[0]
    final_prediction = most_common[0]
    vote_strength = most_common[1] / len(predictions)

    # Calculate aggregate confidence
    avg_confidence = total_confidence / len(predictions)
    final_confidence = min(vote_strength * avg_confidence, 1.0)

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "abstain": final_confidence < threshold,
        "vote_details": {
            "total_votes": len(predictions),
            "winning_votes": most_common[1],
            "vote_strength": vote_strength,
            "avg_source_confidence": avg_confidence,
        },
    }


class WeakSupervision:
    """Manages weak supervision rules, label models, and heuristics."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def create_labeler(
        self,
        name: str,
        labeler_type: str,
        rules: Dict[str, Any],
        threshold: float = 0.65,
    ) -> str:
        """Create a new weak labeler."""
        valid_types = ["rule", "model", "heuristic"]
        if labeler_type not in valid_types:
            raise ValueError(f"Invalid labeler_type. Must be one of: {valid_types}")

        # Generate labeler ID
        labeler_id = f"{labeler_type}_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO weak_labelers (
                    labeler_id, name, labeler_type, threshold, rules_json, active
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (labeler_id, name, labeler_type, threshold, json.dumps(rules), True),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Labeler {labeler_id} already exists")
        finally:
            conn.close()

        return labeler_id

    def apply_labeler(
        self, labeler_id: str, item_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a weak labeler to an item and return prediction."""
        labeler = self.get_labeler(labeler_id)
        if not labeler or not labeler["active"]:
            return {"prediction": None, "confidence": 0.0, "abstain": True}

        # Get labeler rules/config
        rules = labeler["rules"]
        threshold = labeler["threshold"]
        labeler_type = labeler["labeler_type"]

        # Apply labeler based on type
        if labeler_type == "rule":
            result = self._apply_rule_labeler(rules, item_data, threshold)
        elif labeler_type == "model":
            result = self._apply_model_labeler(rules, item_data, threshold)
        elif labeler_type == "heuristic":
            result = self._apply_heuristic_labeler(rules, item_data, threshold)
        else:
            result = {"prediction": None, "confidence": 0.0, "abstain": True}

        # Store prediction
        if "item_id" in item_data:
            self._store_prediction(labeler_id, item_data["item_id"], result)

        return result

    def _apply_rule_labeler(
        self, rules: Dict[str, Any], item_data: Dict[str, Any], threshold: float
    ) -> Dict[str, Any]:
        """Apply rule-based labeler (simplified Snorkel-like functionality)."""
        content = item_data.get("content", "").lower()

        # Example rule patterns
        positive_patterns = rules.get("positive_patterns", [])
        negative_patterns = rules.get("negative_patterns", [])

        pos_matches = sum(
            1 for pattern in positive_patterns if pattern.lower() in content
        )
        neg_matches = sum(
            1 for pattern in negative_patterns if pattern.lower() in content
        )

        if pos_matches > 0 and neg_matches == 0:
            confidence = min(0.9, 0.6 + pos_matches * 0.1)
            return {
                "prediction": "positive",
                "confidence": confidence,
                "abstain": confidence < threshold,
                "rule_matches": {"positive": pos_matches, "negative": neg_matches},
            }
        elif neg_matches > 0 and pos_matches == 0:
            confidence = min(0.9, 0.6 + neg_matches * 0.1)
            return {
                "prediction": "negative",
                "confidence": confidence,
                "abstain": confidence < threshold,
                "rule_matches": {"positive": pos_matches, "negative": neg_matches},
            }
        else:
            # Conflicting or no matches
            return {
                "prediction": None,
                "confidence": 0.0,
                "abstain": True,
                "rule_matches": {"positive": pos_matches, "negative": neg_matches},
            }

    def _apply_model_labeler(
        self, rules: Dict[str, Any], item_data: Dict[str, Any], threshold: float
    ) -> Dict[str, Any]:
        """Apply model-based weak labeler."""
        # Mock model prediction - in practice would call actual ML model
        mock_confidence = random.uniform(0.3, 0.95)
        mock_prediction = "positive" if mock_confidence > 0.6 else "negative"

        return {
            "prediction": mock_prediction,
            "confidence": mock_confidence,
            "abstain": mock_confidence < threshold,
            "model_info": rules.get("model_info", {}),
        }

    def _apply_heuristic_labeler(
        self, rules: Dict[str, Any], item_data: Dict[str, Any], threshold: float
    ) -> Dict[str, Any]:
        """Apply heuristic-based labeler."""
        content = item_data.get("content", "")

        # Example heuristics
        word_count = len(content.split())
        has_numbers = any(c.isdigit() for c in content)
        has_caps = any(c.isupper() for c in content)

        # Simple heuristic scoring
        score = 0.0
        if word_count > rules.get("min_words", 5):
            score += 0.3
        if has_numbers and rules.get("require_numbers", False):
            score += 0.2
        if has_caps and rules.get("require_caps", False):
            score += 0.2

        prediction = "positive" if score >= 0.5 else "negative"
        confidence = min(score + 0.3, 1.0)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "abstain": confidence < threshold,
            "heuristic_scores": {
                "word_count": word_count,
                "has_numbers": has_numbers,
                "has_caps": has_caps,
                "total_score": score,
            },
        }

    def _store_prediction(
        self, labeler_id: str, item_id: str, prediction: Dict[str, Any]
    ):
        """Store weak labeler prediction in database."""
        prediction_id = (
            f"pred_{labeler_id}_{item_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO weak_predictions (
                    prediction_id, labeler_id, item_id, prediction, confidence
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    prediction_id,
                    labeler_id,
                    item_id,
                    json.dumps(prediction),
                    prediction["confidence"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_labeler(self, labeler_id: str) -> Optional[Dict[str, Any]]:
        """Get labeler details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM weak_labelers WHERE labeler_id = ?
            """,
                (labeler_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return {
                "labeler_id": row["labeler_id"],
                "name": row["name"],
                "labeler_type": row["labeler_type"],
                "threshold": row["threshold"],
                "rules": json.loads(row["rules_json"]),
                "active": bool(row["active"]),
                "precision": row["precision"],
                "recall": row["recall"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def ensemble_predict(
        self,
        labelers: List[str],
        item_data: Dict[str, Any],
        ensemble_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Apply ensemble of weak labelers and aggregate results."""
        if not labelers:
            return {"prediction": None, "confidence": 0.0, "abstain": True}

        # Apply each labeler
        labeler_outputs = []
        for labeler_id in labelers:
            result = self.apply_labeler(labeler_id, item_data)
            if not result.get("abstain", True):
                labeler_outputs.append(result)

        # Aggregate using weak voting
        ensemble_result = weak_vote(labeler_outputs, ensemble_threshold)
        ensemble_result["ensemble_info"] = {
            "total_labelers": len(labelers),
            "voting_labelers": len(labeler_outputs),
            "ensemble_threshold": ensemble_threshold,
        }

        return ensemble_result

    def evaluate_labeler_performance(
        self, labeler_id: str, gold_labels: List[Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate weak labeler performance against gold standard labels."""
        if not gold_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "coverage": 0.0}

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        predictions_made = 0

        for item_id, true_label in gold_labels:
            # Get labeler prediction for this item
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT prediction, confidence FROM weak_predictions
                WHERE labeler_id = ? AND item_id = ?
                ORDER BY created_at DESC LIMIT 1
            """,
                (labeler_id, item_id),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                pred_data = json.loads(row[0])
                if not pred_data.get("abstain", True):
                    predictions_made += 1
                    pred_label = pred_data["prediction"]

                    if pred_label == true_label and true_label == "positive":
                        true_positives += 1
                    elif pred_label == "positive" and true_label != "positive":
                        false_positives += 1
                    elif pred_label != "positive" and true_label == "positive":
                        false_negatives += 1

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        coverage = predictions_made / len(gold_labels)

        # Update labeler performance metrics
        self.update_labeler_performance(labeler_id, precision, recall)

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "coverage": round(coverage, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "predictions_made": predictions_made,
            "total_samples": len(gold_labels),
        }

    def update_labeler_performance(
        self, labeler_id: str, precision: float, recall: float
    ):
        """Update labeler performance metrics in database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE weak_labelers
                SET precision = ?, recall = ?, updated_at = CURRENT_TIMESTAMP
                WHERE labeler_id = ?
            """,
                (precision, recall, labeler_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_labelers(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all weak labelers."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            query = "SELECT * FROM weak_labelers"
            params = []

            if active_only:
                query += " WHERE active = ?"
                params.append(True)

            query += " ORDER BY updated_at DESC"

            cursor.execute(query, params)

            labelers = []
            for row in cursor.fetchall():
                labelers.append(
                    {
                        "labeler_id": row["labeler_id"],
                        "name": row["name"],
                        "labeler_type": row["labeler_type"],
                        "threshold": row["threshold"],
                        "active": bool(row["active"]),
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "created_at": row["created_at"],
                    }
                )

            return labelers
        finally:
            conn.close()

    def deactivate_labeler(self, labeler_id: str) -> bool:
        """Deactivate a weak labeler."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE weak_labelers 
                SET active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE labeler_id = ?
            """,
                (labeler_id,),
            )

            return cursor.rowcount > 0
        finally:
            conn.close()
