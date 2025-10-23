"""
Ethical Framework - Policy-based ethical evaluation (OLD Cortex enhanced)
Production implementation with timezone-aware timestamps
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from enum import Enum, auto
import json
import logging
import uuid
import os

logger = logging.getLogger(__name__)


class EthicalCategory(Enum):
    """Categories for ethical policies"""
    PRIVACY = auto()
    FAIRNESS = auto()
    TRANSPARENCY = auto()
    SAFETY = auto()
    AUTONOMY = auto()
    BENEFICENCE = auto()
    JUSTICE = auto()
    ACCOUNTABILITY = auto()
    SUSTAINABILITY = auto()
    DIGNITY = auto()
    SECURITY = auto()


class EthicalFramework:
    """
    Manages and evaluates ethical policies for system actions
    Original Cortex implementation with timezone fixes
    """
    
    def __init__(self, policies_path: Optional[str] = None):
        self.logger = logging.getLogger("grace.cortex.ethics")
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.policy_lock = Lock()
        
        # Setup storage
        if policies_path:
            self.policies_path = Path(policies_path)
        else:
            self.policies_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "cortex_policies"
        
        self.policies_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Ethical framework initialized at {self.policies_path}")
        
        self._load_policies()
    
    def _load_policies(self) -> None:
        """Load ethical policies from storage"""
        try:
            policy_files = list(self.policies_path.glob("*.json"))
            self.logger.info(f"Loading {len(policy_files)} ethical policies")
            
            for policy_file in policy_files:
                try:
                    with open(policy_file, "r") as f:
                        policy_data = json.load(f)
                    
                    policy_id = policy_data.get("id")
                    if not policy_id:
                        continue
                    
                    # Convert string categories to enum
                    if "categories" in policy_data and isinstance(policy_data["categories"], list):
                        categories = []
                        for category in policy_data["categories"]:
                            if isinstance(category, str):
                                try:
                                    categories.append(EthicalCategory[category])
                                except KeyError:
                                    self.logger.warning(f"Unknown category: {category}")
                        policy_data["categories"] = categories
                    
                    with self.policy_lock:
                        self.policies[policy_id] = policy_data
                
                except Exception as e:
                    self.logger.error(f"Error loading policy from {policy_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error loading policies: {e}")
    
    def _save_policy(self, policy_id: str) -> bool:
        """Save policy to storage"""
        try:
            with self.policy_lock:
                if policy_id not in self.policies:
                    return False
                
                policy_data = self.policies[policy_id].copy()
                
                # Convert enum categories to strings
                if "categories" in policy_data and isinstance(policy_data["categories"], list):
                    categories = []
                    for category in policy_data["categories"]:
                        if isinstance(category, EthicalCategory):
                            categories.append(category.name)
                        else:
                            categories.append(str(category))
                    policy_data["categories"] = categories
                
                file_path = self.policies_path / f"{policy_id}.json"
                with open(file_path, "w") as f:
                    json.dump(policy_data, f, indent=2)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error saving policy {policy_id}: {e}")
            return False
    
    def add_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new ethical policy"""
        try:
            policy_id = policy_data.get("id", str(uuid.uuid4()))
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Convert string categories to enum
            categories = []
            if "categories" in policy_data and isinstance(policy_data["categories"], list):
                for category in policy_data["categories"]:
                    if isinstance(category, str):
                        try:
                            categories.append(EthicalCategory[category.upper()])
                        except KeyError:
                            self.logger.warning(f"Unknown category: {category}")
                    elif isinstance(category, EthicalCategory):
                        categories.append(category)
            
            policy_record = {
                "id": policy_id,
                "name": policy_data.get("name", "Unnamed Policy"),
                "description": policy_data.get("description", ""),
                "categories": categories,
                "rules": policy_data.get("rules", []),
                "created_at": timestamp,
                "updated_at": timestamp,
                "version": policy_data.get("version", "1.0.0"),
                "metadata": policy_data.get("metadata", {})
            }
            
            with self.policy_lock:
                self.policies[policy_id] = policy_record
            
            self._save_policy(policy_id)
            self.logger.info(f"Added ethical policy: {policy_id}")
            
            return policy_record
        
        except Exception as e:
            self.logger.error(f"Error adding ethical policy: {e}")
            raise ValueError(f"Failed to add ethical policy: {e}")
    
    def get_policy(self, policy_id: str) -> Dict[str, Any]:
        """Get policy by ID"""
        with self.policy_lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            return self.policies[policy_id].copy()
    
    def get_all_policies(self) -> List[Dict[str, Any]]:
        """Get all ethical policies"""
        with self.policy_lock:
            return [policy.copy() for policy in self.policies.values()]
    
    def get_policies_by_category(self, category: EthicalCategory) -> List[Dict[str, Any]]:
        """Get policies by category"""
        with self.policy_lock:
            return [
                policy.copy() for policy in self.policies.values()
                if "categories" in policy and category in policy["categories"]
            ]
    
    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action against ethical policies"""
        try:
            results = {
                "action": action,
                "compliant": True,
                "policy_evaluations": [],
                "overall_score": 1.0,
                "concerns": []
            }
            
            with self.policy_lock:
                if not self.policies:
                    return results
                
                # Evaluate each policy
                for policy_id, policy in self.policies.items():
                    policy_result = self._evaluate_policy(policy, action)
                    results["policy_evaluations"].append(policy_result)
                    
                    if not policy_result["compliant"]:
                        results["compliant"] = False
                        results["concerns"].extend(policy_result["concerns"])
                
                # Calculate overall score
                if results["policy_evaluations"]:
                    total_score = sum(
                        eval_result["score"] 
                        for eval_result in results["policy_evaluations"]
                    )
                    results["overall_score"] = total_score / len(results["policy_evaluations"])
                
                # Remove duplicate concerns
                results["concerns"] = list({
                    concern["description"]: concern 
                    for concern in results["concerns"]
                }.values())
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error evaluating action: {e}")
            return {
                "action": action,
                "compliant": False,
                "error": str(e),
                "policy_evaluations": [],
                "overall_score": 0.0,
                "concerns": [{"description": f"Evaluation error: {e}", "severity": "high"}]
            }
    
    def _evaluate_policy(self, policy: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action against specific policy"""
        policy_name = policy.get("name", "Unnamed Policy")
        policy_id = policy.get("id", "unknown")
        rules = policy.get("rules", [])
        
        result = {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "compliant": True,
            "score": 1.0,
            "concerns": []
        }
        
        if not rules:
            return result
        
        # Evaluate each rule
        rule_scores = []
        for rule in rules:
            rule_result = self._evaluate_rule(rule, action)
            rule_scores.append(rule_result["score"])
            
            if not rule_result["compliant"]:
                result["compliant"] = False
                result["concerns"].append({
                    "rule": rule.get("name", "Unnamed Rule"),
                    "description": rule_result.get("reason", "Rule violated"),
                    "severity": rule.get("severity", "medium")
                })
        
        # Calculate policy score
        if rule_scores:
            result["score"] = sum(rule_scores) / len(rule_scores)
        
        return result
    
    def _evaluate_rule(self, rule: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action against specific rule"""
        rule_type = rule.get("type", "")
        rule_condition = rule.get("condition", {})
        rule_name = rule.get("name", "Unnamed Rule")
        
        result = {
            "rule_name": rule_name,
            "compliant": True,
            "score": 1.0,
            "reason": ""
        }
        
        try:
            if rule_type == "parameter_constraint":
                self._evaluate_parameter_constraint(rule_condition, action, result)
            elif rule_type == "action_type_constraint":
                self._evaluate_action_type_constraint(rule_condition, action, result)
            elif rule_type == "context_constraint":
                self._evaluate_context_constraint(rule_condition, action, result)
        
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule_name}: {e}")
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Rule evaluation error: {e}"
            })
        
        return result
    
    def _evaluate_parameter_constraint(
        self,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Evaluate parameter constraint"""
        param_name = condition.get("parameter", "")
        constraint = condition.get("constraint", "")
        value = condition.get("value")
        
        if not param_name or not constraint or not action.get("parameters"):
            return
        
        param_value = action["parameters"].get(param_name)
        
        if constraint == "equals" and param_value != value:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Parameter '{param_name}' must equal '{value}'"
            })
        elif constraint == "not_equals" and param_value == value:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Parameter '{param_name}' must not equal '{value}'"
            })
        elif constraint == "contains" and (not param_value or value not in str(param_value)):
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Parameter '{param_name}' must contain '{value}'"
            })
        elif constraint == "not_contains" and param_value and value in str(param_value):
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Parameter '{param_name}' must not contain '{value}'"
            })
        elif constraint == "greater_than" and (param_value is None or param_value <= value):
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Parameter '{param_name}' must be greater than {value}"
            })
        elif constraint == "less_than" and (param_value is None or param_value >= value):
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Parameter '{param_name}' must be less than {value}"
            })
    
    def _evaluate_action_type_constraint(
        self,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Evaluate action type constraint"""
        allowed_types = condition.get("allowed_types", [])
        disallowed_types = condition.get("disallowed_types", [])
        action_type = action.get("type", "")
        
        if allowed_types and action_type not in allowed_types:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Action type '{action_type}' not in allowed types"
            })
        
        if disallowed_types and action_type in disallowed_types:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Action type '{action_type}' is disallowed"
            })
    
    def _evaluate_context_constraint(
        self,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Evaluate context constraint"""
        context_key = condition.get("key", "")
        constraint = condition.get("constraint", "")
        value = condition.get("value")
        
        if not context_key or not constraint or not action.get("context"):
            return
        
        context_value = action["context"].get(context_key)
        
        if constraint == "equals" and context_value != value:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Context '{context_key}' must equal '{value}'"
            })
        elif constraint == "not_equals" and context_value == value:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Context '{context_key}' must not equal '{value}'"
            })
        elif constraint == "exists" and context_value is None:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Context must include '{context_key}'"
            })
        elif constraint == "not_exists" and context_value is not None:
            result.update({
                "compliant": False,
                "score": 0.0,
                "reason": f"Context must not include '{context_key}'"
            })
