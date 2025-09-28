"""
Task Router - Detects tasks and performs policy-aware routing.

Handles:
1. Task detection from TaskRequest.task or auto-detect via heuristics
2. Policy screening (governance thresholds, blocklists, PII labels) 
3. Specialist selection based on task type and constraints
4. Route optimization considering latency/cost/accuracy trade-offs
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class TaskRouter:
    """Routes tasks to appropriate specialists and models with policy awareness."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Task detection patterns
        self.task_patterns = {
            "classification": [
                r"\b(classify|categorize|predict class|binary|multiclass)\b",
                r"\b(sentiment|spam|fraud detection)\b"
            ],
            "regression": [
                r"\b(predict|estimate|forecast)\b.*\b(price|value|amount|score)\b",
                r"\b(regression|continuous|numerical prediction)\b"
            ],
            "clustering": [
                r"\b(cluster|segment|group|unsupervised)\b",
                r"\b(k-means|hierarchical|dbscan)\b"
            ],
            "nlp": [
                r"\b(text|natural language|nlp|language model)\b",
                r"\b(tokenize|parse|translate|summarize)\b"
            ],
            "vision": [
                r"\b(image|vision|computer vision|cv)\b",
                r"\b(object detection|segmentation|recognition)\b"
            ],
            "timeseries": [
                r"\b(time series|temporal|sequential|forecasting)\b",
                r"\b(trend|seasonality|auto-regressive)\b"
            ],
            "rl": [
                r"\b(reinforcement learning|rl|agent|policy)\b",
                r"\b(reward|action|environment|mdp)\b"
            ]
        }
        
        # Specialist mappings by task type
        self.specialist_mapping = {
            "classification": [
                "tabular_classification", "deep_classification", "ensemble_classifier",
                "bayesian_classifier", "meta_classifier"
            ],
            "regression": [
                "linear_regression", "tree_regression", "neural_regression", 
                "gaussian_process", "bayesian_regression"
            ],
            "clustering": [
                "kmeans_specialist", "hierarchical_specialist", "density_specialist",
                "spectral_specialist", "mixture_model_specialist"
            ],
            "nlp": [
                "transformer_nlp", "bert_specialist", "gpt_specialist",
                "sentiment_specialist", "named_entity_specialist"
            ],
            "vision": [
                "cnn_vision", "resnet_specialist", "vit_specialist",
                "object_detection_specialist", "segmentation_specialist"
            ],
            "timeseries": [
                "lstm_timeseries", "prophet_specialist", "arima_specialist",
                "transformer_timeseries", "statistical_forecaster"
            ],
            "rl": [
                "dqn_specialist", "policy_gradient_specialist", "actor_critic_specialist",
                "multi_agent_specialist", "model_based_rl"
            ]
        }
        
        logger.info("Task Router initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default router configuration."""
        return {
            "hybrid_weights": {
                "latency": 0.3,
                "quality": 0.7
            },
            "allow_shadow": True,
            "canary_pct_default": 10,
            "max_specialists_per_task": 5,
            "auto_detect_confidence_threshold": 0.8
        }
    
    def route(self, task_req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main routing function.
        Returns route specification with selected specialists and models.
        """
        try:
            # Step 1: Task detection
            detected_task = self._detect_task(task_req)
            
            # Step 2: Policy screening
            policy_result = self._screen_policies(task_req, detected_task)
            if not policy_result["allowed"]:
                raise ValueError(f"Policy violation: {policy_result['reason']}")
            
            # Step 3: Specialist selection
            specialists = self._select_specialists(detected_task, task_req)
            
            # Step 4: Model selection 
            models = self._select_models(specialists, task_req)
            
            # Step 5: Route optimization
            optimized_route = self._optimize_route(specialists, models, task_req)
            
            return optimized_route
            
        except Exception as e:
            logger.error(f"Routing error: {e}")
            raise
    
    def _detect_task(self, task_req: Dict[str, Any]) -> str:
        """Detect task type from request or auto-detect from input/context."""
        # Explicit task type
        if "task" in task_req and task_req["task"]:
            return task_req["task"]
        
        # Auto-detection from content
        content = ""
        if "input" in task_req:
            input_data = task_req["input"]
            if isinstance(input_data.get("X"), str):
                content += input_data["X"]
            if "modality" in input_data:
                content += f" {input_data['modality']}"
        
        if "context" in task_req:
            context = task_req["context"]
            if "user_ctx" in context and isinstance(context["user_ctx"], dict):
                content += " " + str(context["user_ctx"])
        
        # Pattern matching for task detection
        task_scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content.lower()))
                score += matches
            task_scores[task_type] = score
        
        # Return highest scoring task if above threshold
        if task_scores:
            best_task = max(task_scores.items(), key=lambda x: x[1])
            if best_task[1] > 0:
                logger.info(f"Auto-detected task: {best_task[0]} (score: {best_task[1]})")
                return best_task[0]
        
        # Default fallback
        modality = task_req.get("input", {}).get("modality", "tabular")
        if modality == "text":
            return "nlp"
        elif modality == "image":
            return "vision"
        else:
            return "classification"  # Safe default
    
    def _screen_policies(self, task_req: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Screen request against policies and constraints."""
        context = task_req.get("context", {})
        constraints = task_req.get("constraints", {})
        
        # Check environment restrictions
        env = context.get("env", "dev")
        if env == "prod" and task_type == "rl":
            return {
                "allowed": False,
                "reason": "RL tasks not allowed in production environment"
            }
        
        # Check model allowlist/denylist
        models_allowlist = constraints.get("models_allowlist", [])
        models_denylist = constraints.get("models_denylist", [])
        
        if models_denylist:
            # Check if any specialists for this task are in denylist
            specialists = self.specialist_mapping.get(task_type, [])
            for specialist in specialists[:3]:  # Check top 3
                if any(blocked in specialist for blocked in models_denylist):
                    logger.warning(f"Specialist {specialist} blocked by denylist")
        
        # Check PII patterns (basic)
        input_data = task_req.get("input", {})
        if isinstance(input_data.get("X"), str):
            pii_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
            ]
            content = input_data["X"]
            for pattern in pii_patterns:
                if re.search(pattern, content):
                    return {
                        "allowed": False,
                        "reason": "PII detected in input data"
                    }
        
        # Check latency budget
        latency_budget = context.get("latency_budget_ms")
        if latency_budget and latency_budget < 100:
            if task_type in ["vision", "nlp"] and "transformer" in str(models_allowlist):
                return {
                    "allowed": False,
                    "reason": "Latency budget too low for transformer models"
                }
        
        return {"allowed": True, "reason": None}
    
    def _select_specialists(self, task_type: str, task_req: Dict[str, Any]) -> List[str]:
        """Select appropriate specialists for the task."""
        available_specialists = self.specialist_mapping.get(task_type, [])
        
        if not available_specialists:
            logger.warning(f"No specialists found for task type: {task_type}")
            return ["general_ml_specialist"]
        
        # Filter by constraints
        constraints = task_req.get("constraints", {})
        models_allowlist = constraints.get("models_allowlist", [])
        models_denylist = constraints.get("models_denylist", [])
        
        filtered_specialists = []
        for specialist in available_specialists:
            # Apply denylist
            if models_denylist and any(blocked in specialist for blocked in models_denylist):
                continue
            
            # Apply allowlist if specified
            if models_allowlist and not any(allowed in specialist for allowed in models_allowlist):
                continue
                
            filtered_specialists.append(specialist)
        
        # Limit number of specialists
        max_specialists = self.config.get("max_specialists_per_task", 5)
        selected = filtered_specialists[:max_specialists]
        
        # Ensure we have at least one specialist
        if not selected and available_specialists:
            selected = [available_specialists[0]]
        
        logger.info(f"Selected specialists for {task_type}: {selected}")
        return selected
    
    def _select_models(self, specialists: List[str], task_req: Dict[str, Any]) -> List[str]:
        """Select concrete models for the specialists."""
        models = []
        
        # Map specialists to concrete model versions
        model_mapping = {
            "tabular_classification": "xgb@1.3.2",
            "deep_classification": "neural_net@2.1.0", 
            "linear_regression": "sklearn_lr@1.0.1",
            "tree_regression": "random_forest@2.0.1",
            "transformer_nlp": "bert-base@1.2.0",
            "cnn_vision": "resnet50@1.1.0",
            "lstm_timeseries": "lstm_model@1.0.5",
            "dqn_specialist": "dqn_agent@0.9.2"
        }
        
        for specialist in specialists:
            if specialist in model_mapping:
                models.append(model_mapping[specialist])
            else:
                # Generate generic model name
                models.append(f"{specialist}_model@1.0.0")
        
        return models
    
    def _optimize_route(self, specialists: List[str], models: List[str], task_req: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize route based on latency/quality trade-offs."""
        context = task_req.get("context", {})
        
        # Determine ensemble strategy
        ensemble_strategy = "stack"  # Default
        if len(models) == 1:
            ensemble_strategy = "none"
        elif context.get("latency_budget_ms", 1000) < 200:
            ensemble_strategy = "vote"  # Faster than stacking
        
        # Determine canary percentage
        canary_pct = self.config.get("canary_pct_default", 10)
        if context.get("env") == "prod":
            canary_pct = min(canary_pct, 5)  # More conservative in prod
        
        # Shadow deployment
        allow_shadow = (
            self.config.get("allow_shadow", True) and 
            context.get("canary_allowed", True)
        )
        
        route = {
            "specialists": specialists,
            "models": models,
            "ensemble": ensemble_strategy,
            "canary_pct": canary_pct,
            "shadow": allow_shadow
        }
        
        logger.info(f"Optimized route: {route}")
        return route
    
    def get_state(self) -> Dict[str, Any]:
        """Get current router state for snapshots."""
        return {
            "config": self.config,
            "version": "1.0.0",
            "specialist_mapping": self.specialist_mapping
        }
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """Load router state from snapshot."""
        try:
            self.config = state.get("config", self.config)
            logger.info("Router state loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load router state: {e}")
            return False