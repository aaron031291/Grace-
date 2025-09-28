"""
Plan Builder - Creates candidate graph plans with search space and constraints.

Handles:
1. Building execution plans from routes
2. Constraint validation and optimization
3. Search space exploration
4. Pre-flight checks for data/model availability
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
import uuid

logger = logging.getLogger(__name__)


class PlanBuilder:
    """Builds executable plans from routing decisions."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.active_plans: Dict[str, Dict] = {}
        
        logger.info("Plan Builder initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default plan builder configuration."""
        return {
            "max_concurrent_plans": 100,
            "plan_timeout_seconds": 300,
            "enable_preflight_checks": True,
            "default_log_level": "summary"
        }
    
    def build_plan(self, task_req: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build execution plan from task request and route.
        Returns Plan object as defined in contracts.
        """
        try:
            # Generate plan ID
            plan_id = f"plan_{format_for_filename()}_{uuid.uuid4().hex[:8]}"
            
            # Extract request info
            req_id = task_req.get("req_id", "unknown")
            context = task_req.get("context", {})
            constraints = task_req.get("constraints", {})
            
            # Build policy from constraints and context
            policy = self._build_policy(context, constraints)
            
            # Validate and optimize route
            optimized_route = self._optimize_route(route, task_req)
            
            # Pre-flight checks
            if self.config.get("enable_preflight_checks", True):
                preflight_result = self._preflight_checks(optimized_route, task_req)
                if not preflight_result["passed"]:
                    raise ValueError(f"Pre-flight check failed: {preflight_result['reason']}")
            
            # Construct plan
            plan = {
                "plan_id": plan_id,
                "req_id": req_id,
                "route": optimized_route,
                "policy": policy,
                "created_at": iso_format(),
                "status": "ready"
            }
            
            # Store active plan
            self.active_plans[plan_id] = plan
            
            logger.info(f"Built plan {plan_id} for request {req_id}")
            return plan
            
        except Exception as e:
            logger.error(f"Plan building failed: {e}")
            raise
    
    def _build_policy(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Build policy configuration from context and constraints."""
        # Start with defaults
        policy = {
            "min_confidence": 0.75,
            "min_calibration": 0.96,
            "fairness_delta_max": 0.02,
            "log_level": self.config.get("default_log_level", "summary")
        }
        
        # Override with constraints
        if "min_calibration" in constraints:
            policy["min_calibration"] = constraints["min_calibration"]
        
        if "fairness_delta_max" in constraints:
            policy["fairness_delta_max"] = constraints["fairness_delta_max"]
        
        # Adjust based on context
        env = context.get("env", "dev")
        if env == "prod":
            # Stricter policies in production
            policy["min_confidence"] = max(policy["min_confidence"], 0.8)
            policy["min_calibration"] = max(policy["min_calibration"], 0.97)
        
        # Explanation requirements
        if context.get("explanation", False):
            policy["log_level"] = "full"
            policy["require_explanations"] = True
        
        return policy
    
    def _optimize_route(self, route: Dict[str, Any], task_req: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize route based on search space exploration."""
        optimized_route = route.copy()
        
        context = task_req.get("context", {})
        
        # Optimize model selection based on latency budget
        latency_budget = context.get("latency_budget_ms")
        if latency_budget:
            optimized_route = self._optimize_for_latency(optimized_route, latency_budget)
        
        # Optimize for cost budget
        cost_budget = context.get("cost_budget_units")
        if cost_budget:
            optimized_route = self._optimize_for_cost(optimized_route, cost_budget)
        
        # Ensemble optimization
        optimized_route = self._optimize_ensemble(optimized_route, context)
        
        return optimized_route
    
    def _optimize_for_latency(self, route: Dict[str, Any], latency_budget: int) -> Dict[str, Any]:
        """Optimize route for latency constraints."""
        models = route.get("models", [])
        
        # Model latency estimates (ms)
        latency_estimates = {
            "xgb@1.3.2": 50,
            "random_forest@2.0.1": 30,
            "neural_net@2.1.0": 200,
            "bert-base@1.2.0": 400,
            "resnet50@1.1.0": 300,
            "lstm_model@1.0.5": 150
        }
        
        # Filter models that fit budget
        fast_models = []
        for model in models:
            estimated_latency = latency_estimates.get(model, 100)
            if estimated_latency <= latency_budget * 0.8:  # Leave 20% buffer
                fast_models.append(model)
        
        if fast_models:
            route["models"] = fast_models[:3]  # Limit to top 3 fast models
            
            # Use faster ensemble method
            if latency_budget < 200:
                route["ensemble"] = "vote"
            elif latency_budget < 500:
                route["ensemble"] = "blend"
        
        return route
    
    def _optimize_for_cost(self, route: Dict[str, Any], cost_budget: float) -> Dict[str, Any]:
        """Optimize route for cost constraints."""
        # Cost estimates per prediction (arbitrary units)
        cost_estimates = {
            "xgb@1.3.2": 0.01,
            "random_forest@2.0.1": 0.005,
            "neural_net@2.1.0": 0.05,
            "bert-base@1.2.0": 0.2,
            "resnet50@1.1.0": 0.15,
            "lstm_model@1.0.5": 0.08
        }
        
        models = route.get("models", [])
        affordable_models = []
        
        for model in models:
            estimated_cost = cost_estimates.get(model, 0.02)
            if estimated_cost <= cost_budget:
                affordable_models.append(model)
        
        if affordable_models:
            route["models"] = affordable_models
            
            # Reduce canary percentage to save costs
            route["canary_pct"] = min(route.get("canary_pct", 10), 5)
        
        return route
    
    def _optimize_ensemble(self, route: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize ensemble strategy based on context."""
        models = route.get("models", [])
        num_models = len(models)
        
        if num_models <= 1:
            route["ensemble"] = "none"
        elif num_models == 2:
            route["ensemble"] = "blend"
        elif context.get("latency_budget_ms", 1000) > 500:
            route["ensemble"] = "stack"  # Most accurate but slowest
        else:
            route["ensemble"] = "vote"   # Fast majority voting
        
        return route
    
    def _preflight_checks(self, route: Dict[str, Any], task_req: Dict[str, Any]) -> Dict[str, bool]:
        """Perform pre-flight checks before execution."""
        checks = {
            "passed": True,
            "reason": None,
            "details": []
        }
        
        try:
            # Check 1: Model availability
            models = route.get("models", [])
            if not models:
                checks["passed"] = False
                checks["reason"] = "No models specified in route"
                return checks
            
            # Check 2: Data schema compatibility
            input_data = task_req.get("input", {})
            modality = input_data.get("modality", "tabular")
            
            # Validate model-modality compatibility
            incompatible_models = []
            for model in models:
                if "bert" in model and modality != "text":
                    incompatible_models.append(f"{model} incompatible with {modality}")
                elif "resnet" in model and modality != "image":
                    incompatible_models.append(f"{model} incompatible with {modality}")
                elif "lstm" in model and modality not in ["text", "timeseries"]:
                    incompatible_models.append(f"{model} incompatible with {modality}")
            
            if incompatible_models:
                checks["passed"] = False
                checks["reason"] = f"Model incompatibilities: {'; '.join(incompatible_models)}"
                return checks
            
            # Check 3: Feature view availability (mock check)
            # In real implementation, this would check Memory kernel
            checks["details"].append("Feature view check: PASSED")
            
            # Check 4: Resource availability
            ensemble_method = route.get("ensemble", "none")
            if ensemble_method == "stack" and len(models) > 5:
                checks["details"].append("WARNING: Large ensemble may be resource intensive")
            
            logger.info(f"Pre-flight checks passed: {checks['details']}")
            
        except Exception as e:
            checks["passed"] = False
            checks["reason"] = f"Pre-flight check error: {str(e)}"
            logger.error(f"Pre-flight check failed: {e}")
        
        return checks
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve plan by ID."""
        return self.active_plans.get(plan_id)
    
    def update_plan_status(self, plan_id: str, status: str, details: Optional[Dict] = None):
        """Update plan execution status."""
        if plan_id in self.active_plans:
            self.active_plans[plan_id]["status"] = status
            self.active_plans[plan_id]["updated_at"] = iso_format()
            
            if details:
                self.active_plans[plan_id]["execution_details"] = details
            
            logger.info(f"Plan {plan_id} status updated to: {status}")
    
    def cleanup_completed_plans(self, max_age_hours: int = 24):
        """Clean up old completed plans."""
        current_time = utc_now()
        plans_to_remove = []
        
        for plan_id, plan in self.active_plans.items():
            created_at = datetime.fromisoformat(plan["created_at"])
            age_hours = (current_time - created_at).total_seconds() / 3600
            
            if age_hours > max_age_hours and plan.get("status") in ["completed", "failed", "cancelled"]:
                plans_to_remove.append(plan_id)
        
        for plan_id in plans_to_remove:
            del self.active_plans[plan_id]
        
        if plans_to_remove:
            logger.info(f"Cleaned up {len(plans_to_remove)} old plans")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plan builder statistics."""
        status_counts = {}
        for plan in self.active_plans.values():
            status = plan.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_active_plans": len(self.active_plans),
            "status_distribution": status_counts,
            "config": self.config
        }