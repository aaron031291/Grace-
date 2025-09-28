"""Labeling policies and quality gates service."""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any


class PolicyService:
    """Manages labeling policies, rubrics, and quality assurance gates."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_default_policies()
    
    def _ensure_default_policies(self):
        """Create default labeling policies if they don't exist."""
        default_policies = [
            {
                "policy_id": "default_text_classification",
                "rubric": {
                    "task_description": "Classify text into positive or negative sentiment",
                    "labels": {
                        "positive": "Text expresses positive sentiment, satisfaction, or approval",
                        "negative": "Text expresses negative sentiment, dissatisfaction, or criticism"
                    },
                    "edge_cases": [
                        "Neutral text should be classified based on overall tone",
                        "Sarcastic text should be classified based on literal meaning",
                        "Mixed sentiment should favor the stronger emotion"
                    ]
                },
                "qa": {
                    "gold_ratio": 0.05,
                    "dual_label_ratio": 0.1, 
                    "min_agreement": 0.8
                },
                "pii": {
                    "allow": False,
                    "mask": True
                }
            },
            {
                "policy_id": "strict_quality_policy",
                "rubric": {
                    "task_description": "High-quality labeling with strict quality requirements",
                    "quality_requirements": [
                        "All labels must include evidence/rationale",
                        "Uncertain cases must be flagged for expert review",
                        "Inter-annotator agreement must exceed 90%"
                    ]
                },
                "qa": {
                    "gold_ratio": 0.1,
                    "dual_label_ratio": 0.2,
                    "min_agreement": 0.9
                },
                "pii": {
                    "allow": False,
                    "mask": True
                }
            }
        ]
        
        for policy in default_policies:
            try:
                self.create_policy(policy, overwrite=False)
            except ValueError:
                # Policy already exists, skip
                pass
    
    def create_policy(self, policy: Dict[str, Any], overwrite: bool = False) -> str:
        """Create a new labeling policy."""
        # Validate required fields
        required_fields = ["policy_id", "rubric", "qa", "pii"]
        for field in required_fields:
            if field not in policy:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate QA configuration
        qa = policy["qa"]
        required_qa_fields = ["gold_ratio", "dual_label_ratio", "min_agreement"]
        for field in required_qa_fields:
            if field not in qa:
                raise ValueError(f"Missing required QA field: {field}")
        
        # Validate ratios are between 0 and 1
        if not (0.0 <= qa["gold_ratio"] <= 1.0):
            raise ValueError("gold_ratio must be between 0.0 and 1.0")
        if not (0.0 <= qa["dual_label_ratio"] <= 1.0):
            raise ValueError("dual_label_ratio must be between 0.0 and 1.0") 
        if not (0.0 <= qa["min_agreement"] <= 1.0):
            raise ValueError("min_agreement must be between 0.0 and 1.0")
        
        # Validate PII configuration
        pii = policy["pii"]
        if "allow" not in pii or "mask" not in pii:
            raise ValueError("PII configuration must include 'allow' and 'mask' fields")
        
        policy_id = policy["policy_id"]
        
        # Check if policy exists
        if not overwrite and self.get_policy(policy_id):
            raise ValueError(f"Policy {policy_id} already exists")
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        try:
            if overwrite:
                conn.execute("DELETE FROM label_policies WHERE policy_id = ?", (policy_id,))
            
            conn.execute("""
                INSERT INTO label_policies (
                    policy_id, rubric_json, gold_ratio, dual_label_ratio,
                    min_agreement, allow_pii, mask_pii
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                policy_id,
                json.dumps(policy["rubric"]),
                qa["gold_ratio"],
                qa["dual_label_ratio"],
                qa["min_agreement"],
                pii["allow"],
                pii["mask"]
            ))
            conn.commit()
        except sqlite3.IntegrityError as e:
            if not overwrite:
                raise ValueError(f"Policy {policy_id} already exists") from e
        finally:
            conn.close()
        
        return policy_id
    
    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get labeling policy by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM label_policies WHERE policy_id = ?
            """, (policy_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "policy_id": row["policy_id"],
                "rubric": json.loads(row["rubric_json"]),
                "qa": {
                    "gold_ratio": row["gold_ratio"],
                    "dual_label_ratio": row["dual_label_ratio"],
                    "min_agreement": row["min_agreement"]
                },
                "pii": {
                    "allow": bool(row["allow_pii"]),
                    "mask": bool(row["mask_pii"])
                },
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        finally:
            conn.close()
    
    def validate_label_quality(self, label: Dict[str, Any], policy_id: str) -> Dict[str, Any]:
        """Validate a label against policy quality requirements."""
        policy = self.get_policy(policy_id)
        if not policy:
            return {"valid": False, "errors": [f"Policy {policy_id} not found"]}
        
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["annotator_id", "y", "quality"]
        for field in required_fields:
            if field not in label:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check quality metrics
        quality = label["quality"]
        min_agreement = policy["qa"]["min_agreement"]
        
        if "agreement" in quality:
            agreement = quality["agreement"]
            if agreement < min_agreement:
                errors.append(f"Agreement {agreement} below minimum {min_agreement}")
        
        # Check for PII if policy doesn't allow it
        if not policy["pii"]["allow"]:
            content = str(label.get("y", ""))
            evidence = str(label.get("evidence", ""))
            
            # Simple PII detection (in practice would use more sophisticated methods)
            pii_indicators = ["@", "phone", "ssn", "social security", "credit card"]
            for indicator in pii_indicators:
                if indicator.lower() in content.lower() or indicator.lower() in evidence.lower():
                    if policy["pii"]["mask"]:
                        warnings.append(f"Potential PII detected: {indicator} (should be masked)")
                    else:
                        errors.append(f"PII not allowed: {indicator}")
        
        # Check for evidence if required by rubric
        rubric = policy["rubric"]
        if "require_evidence" in rubric and rubric["require_evidence"]:
            if "evidence" not in label or not label["evidence"]:
                errors.append("Evidence/rationale required by policy")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "policy_id": policy_id
        }
    
    def should_use_gold_standard(self, policy_id: str) -> bool:
        """Determine if a gold standard item should be used based on policy."""
        policy = self.get_policy(policy_id)
        if not policy:
            return False
        
        import random
        return random.random() < policy["qa"]["gold_ratio"]
    
    def should_dual_label(self, policy_id: str) -> bool:
        """Determine if an item should be dual-labeled based on policy."""
        policy = self.get_policy(policy_id)
        if not policy:
            return False
        
        import random
        return random.random() < policy["qa"]["dual_label_ratio"]
    
    def calculate_task_quality_metrics(self, task_id: str, policy_id: str) -> Dict[str, Any]:
        """Calculate quality metrics for a labeling task."""
        policy = self.get_policy(policy_id)
        if not policy:
            return {"error": f"Policy {policy_id} not found"}
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get all labels for this task
            cursor = conn.cursor()
            cursor.execute("""
                SELECT item_id, annotator_id, agreement, gold_correct, confidence
                FROM labels
                WHERE task_id = ?
            """, (task_id,))
            
            labels = cursor.fetchall()
            
            if not labels:
                return {"error": "No labels found for task"}
            
            # Calculate metrics
            total_labels = len(labels)
            agreements = [row["agreement"] for row in labels if row["agreement"] is not None]
            confidences = [row["confidence"] for row in labels if row["confidence"] is not None]
            gold_labels = [row for row in labels if row["gold_correct"] is not None]
            
            avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            gold_accuracy = 0.0
            if gold_labels:
                correct_gold = sum(1 for row in gold_labels if row["gold_correct"])
                gold_accuracy = correct_gold / len(gold_labels)
            
            # Quality gates
            min_agreement = policy["qa"]["min_agreement"]
            agreement_passed = avg_agreement >= min_agreement
            
            # Inter-annotator agreement for dual-labeled items
            inter_annotator_agreement = self._calculate_inter_annotator_agreement(task_id)
            
            return {
                "task_id": task_id,
                "policy_id": policy_id,
                "total_labels": total_labels,
                "avg_agreement": round(avg_agreement, 3),
                "avg_confidence": round(avg_confidence, 3),
                "gold_accuracy": round(gold_accuracy, 3),
                "inter_annotator_agreement": round(inter_annotator_agreement, 3),
                "quality_gates": {
                    "min_agreement_required": min_agreement,
                    "agreement_passed": agreement_passed,
                    "gold_labels_count": len(gold_labels)
                },
                "quality_score": round((avg_agreement + gold_accuracy + inter_annotator_agreement) / 3, 3)
            }
            
        finally:
            conn.close()
    
    def _calculate_inter_annotator_agreement(self, task_id: str) -> float:
        """Calculate inter-annotator agreement for dual-labeled items."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.cursor()
            # Find items with multiple labels
            cursor.execute("""
                SELECT item_id, COUNT(*) as label_count
                FROM labels
                WHERE task_id = ?
                GROUP BY item_id
                HAVING COUNT(*) > 1
            """, (task_id,))
            
            multi_labeled_items = cursor.fetchall()
            
            if not multi_labeled_items:
                return 0.0
            
            total_agreements = 0
            agreement_count = 0
            
            for item_id, label_count in multi_labeled_items:
                # Get all labels for this item
                cursor.execute("""
                    SELECT y_value FROM labels
                    WHERE task_id = ? AND item_id = ?
                """, (task_id, item_id))
                
                labels = [json.loads(row[0]) for row in cursor.fetchall()]
                
                # Calculate pairwise agreement
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        # Simple equality check (in practice would use more sophisticated agreement metrics)
                        if labels[i] == labels[j]:
                            total_agreements += 1
                        agreement_count += 1
            
            return total_agreements / agreement_count if agreement_count > 0 else 0.0
            
        finally:
            conn.close()
    
    def list_policies(self) -> List[Dict[str, Any]]:
        """List all labeling policies."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT policy_id, gold_ratio, dual_label_ratio, min_agreement, created_at
                FROM label_policies
                ORDER BY created_at DESC
            """)
            
            policies = []
            for row in cursor.fetchall():
                policies.append({
                    "policy_id": row["policy_id"],
                    "qa": {
                        "gold_ratio": row["gold_ratio"],
                        "dual_label_ratio": row["dual_label_ratio"],
                        "min_agreement": row["min_agreement"]
                    },
                    "created_at": row["created_at"]
                })
            
            return policies
        finally:
            conn.close()
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing labeling policy."""
        existing_policy = self.get_policy(policy_id)
        if not existing_policy:
            return False
        
        # Merge updates
        updated_policy = existing_policy.copy()
        for key, value in updates.items():
            if key in updated_policy:
                if isinstance(updated_policy[key], dict) and isinstance(value, dict):
                    updated_policy[key].update(value)
                else:
                    updated_policy[key] = value
        
        # Recreate the policy
        self.create_policy(updated_policy, overwrite=True)
        return True