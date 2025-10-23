"""
Grace AI Meta-Learning Kernel - Top-level intelligence optimization
Observes all kernel actions and learns to optimize the system
Applies intelligence to improve trigger rules, handlers, and strategies
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class KernelInsight:
    """Insight extracted from kernel action patterns."""
    
    def __init__(self, insight_type: str, kernel: str, pattern: str, confidence: float):
        self.insight_id = str(__import__('uuid').uuid4())[:8]
        self.insight_type = insight_type
        self.kernel = kernel
        self.pattern = pattern
        self.confidence = confidence
        self.created_at = datetime.now().isoformat()
        self.actionable = False

class MetaLearningKernel:
    """
    Meta-Learning Kernel - Learns from system-wide patterns
    Observes all kernel executions and optimizes the system
    Applies learned intelligence to improve decision-making
    """
    
    def __init__(self, truth_layer, trigger_mesh):
        self.truth_layer = truth_layer
        self.trigger_mesh = trigger_mesh
        self.insights: Dict[str, KernelInsight] = {}
        self.optimization_log: List[Dict[str, Any]] = []
        self.performance_model: Dict[str, float] = {}
    
    async def analyze_kernel_patterns(self) -> List[KernelInsight]:
        """Analyze patterns in kernel executions."""
        insights = []
        
        # Get execution log from trigger mesh
        execution_history = self.trigger_mesh.get_execution_log(limit=1000)
        
        # Analyze patterns
        kernel_success_rates = {}
        handler_latencies = {}
        event_frequencies = {}
        
        for execution in execution_history:
            if execution["status"] == "completed":
                for action in execution.get("actions", []):
                    handler = action.get("handler", "unknown")
                    status = action.get("status", "unknown")
                    
                    if handler not in kernel_success_rates:
                        kernel_success_rates[handler] = {"success": 0, "total": 0}
                    
                    kernel_success_rates[handler]["total"] += 1
                    if status == "completed":
                        kernel_success_rates[handler]["success"] += 1
        
        # Generate insights from patterns
        for handler, stats in kernel_success_rates.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                
                if success_rate < 0.8:
                    insight = KernelInsight(
                        insight_type="low_reliability",
                        kernel=handler,
                        pattern=f"Success rate: {success_rate:.1%}",
                        confidence=0.9
                    )
                    self.insights[insight.insight_id] = insight
                    insights.append(insight)
                    
                    logger.warning(f"Meta-Learning detected low reliability in {handler}: {success_rate:.1%}")
        
        return insights
    
    async def recommend_optimizations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze insights
        for insight in self.insights.values():
            if insight.insight_type == "low_reliability":
                recommendations.append({
                    "type": "increase_monitoring",
                    "target": insight.kernel,
                    "action": f"Add health monitoring and recovery for {insight.kernel}",
                    "confidence": insight.confidence
                })
                
                logger.info(f"Recommending optimization: {insight.kernel}")
        
        return recommendations
    
    async def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply an optimization recommendation."""
        opt_type = optimization.get("type")
        target = optimization.get("target")
        
        logger.info(f"Applying optimization: {opt_type} to {target}")
        
        # Record in truth layer
        await self.truth_layer.immutable_log.record_event(
            event_type="meta_learning.optimization_applied",
            component="meta_learning_kernel",
            data=optimization
        )
        
        # Record in optimization log
        self.optimization_log.append({
            "optimization": optimization,
            "applied_at": datetime.now().isoformat(),
            "status": "applied"
        })
        
        return True
    
    async def evaluate_kernel_effectiveness(self) -> Dict[str, float]:
        """Evaluate the effectiveness of each kernel."""
        effectiveness_scores = {}
        
        rule_stats = self.trigger_mesh.get_rule_stats()
        
        for handler_id in self.trigger_mesh.handlers.keys():
            # Simple effectiveness metric: success rate * execution frequency
            match_count = rule_stats.get("rule_matches", {}).get(handler_id, 0)
            
            if match_count > 0:
                effectiveness_scores[handler_id] = min(1.0, match_count / 100.0)
        
        self.performance_model = effectiveness_scores
        return effectiveness_scores
    
    async def predict_system_improvement(self) -> Dict[str, Any]:
        """Predict potential system improvements."""
        predictions = {
            "potential_kpi_gains": {},
            "risk_factors": [],
            "recommended_actions": []
        }
        
        # Analyze current health
        health = await self.truth_layer.system_metrics.get_overall_health()
        
        if health["average_kpi"] < 95:
            predictions["potential_kpi_gains"]["stability"] = 5.0
            predictions["recommended_actions"].append("Increase system monitoring")
        
        if health["average_trust"] < 70:
            predictions["risk_factors"].append("Low trust score")
            predictions["recommended_actions"].append("Run diagnostic on critical kernels")
        
        return predictions
    
    def get_meta_insights(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all meta-learning insights."""
        return [
            {
                "insight_id": i.insight_id,
                "type": i.insight_type,
                "kernel": i.kernel,
                "pattern": i.pattern,
                "confidence": i.confidence,
                "created_at": i.created_at
            }
            for i in list(self.insights.values())[-limit:]
        ]
