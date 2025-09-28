"""
Policy Tuner - Recommends changes to governance policies and thresholds based on insights.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .contracts import Insight, InsightType


logger = logging.getLogger(__name__)


class PolicyRecommendation:
    """A single policy recommendation."""
    
    def __init__(self, path: str, current_value: Any, recommended_value: Any, 
                 rationale: str, confidence: float):
        self.path = path
        self.current_value = current_value
        self.recommended_value = recommended_value
        self.rationale = rationale
        self.confidence = confidence
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class PolicyTuner:
    """Derives recommended changes to governance thresholds and policies."""
    
    def __init__(self):
        self.recommendations: List[PolicyRecommendation] = []
        self.policy_schema = {
            "governance.thresholds.min_confidence": {"type": float, "range": [0.5, 1.0]},
            "governance.thresholds.min_calibration": {"type": float, "range": [0.7, 1.0]},
            "governance.thresholds.min_trust_score": {"type": float, "range": [0.6, 1.0]},
            "governance.fairness.max_delta": {"type": float, "range": [0.01, 0.1]},
            "governance.canary.steps": {"type": list, "default": [5, 25, 50, 100]},
            "governance.rollback.drift_threshold": {"type": float, "range": [1.0, 5.0]},
            "governance.rollback.performance_drop_threshold": {"type": float, "range": [0.02, 0.1]}
        }
    
    async def generate_policy_recommendations(self, insights: List[Insight], 
                                            current_policies: Dict[str, Any]) -> List[PolicyRecommendation]:
        """Generate policy recommendations based on insights."""
        recommendations = []
        
        try:
            for insight in insights:
                if insight.type == InsightType.GOVERNANCE_ALIGNMENT:
                    recs = await self._tune_governance_thresholds(insight, current_policies)
                    recommendations.extend(recs)
                    
                elif insight.type == InsightType.FAIRNESS:
                    recs = await self._tune_fairness_policies(insight, current_policies)
                    recommendations.extend(recs)
                    
                elif insight.type == InsightType.DRIFT:
                    recs = await self._tune_drift_policies(insight, current_policies)
                    recommendations.extend(recs)
                    
                elif insight.type == InsightType.STABILITY:
                    recs = await self._tune_stability_policies(insight, current_policies)
                    recommendations.extend(recs)
                    
                elif insight.type == InsightType.CALIBRATION:
                    recs = await self._tune_calibration_policies(insight, current_policies)
                    recommendations.extend(recs)
            
            # Store recommendations
            self.recommendations.extend(recommendations)
            
            logger.info(f"Generated {len(recommendations)} policy recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate policy recommendations: {e}")
            return []
    
    async def _tune_governance_thresholds(self, insight: Insight, 
                                        current_policies: Dict[str, Any]) -> List[PolicyRecommendation]:
        """Tune general governance thresholds."""
        recommendations = []
        
        evidence = insight.evidence.get("after", {})
        compliance = evidence.get("avg_compliance", 1.0)
        
        # If compliance is low, consider lowering confidence threshold
        if compliance < 0.85:
            current_confidence = current_policies.get("governance.thresholds.min_confidence", 0.8)
            
            if current_confidence > 0.6:
                new_confidence = max(0.6, current_confidence - 0.05)
                
                recommendations.append(PolicyRecommendation(
                    path="governance.thresholds.min_confidence",
                    current_value=current_confidence,
                    recommended_value=new_confidence,
                    rationale=f"Lower confidence threshold to improve compliance rate ({compliance:.3f})",
                    confidence=insight.confidence * 0.9
                ))
        
        # If compliance is very high, consider raising thresholds
        elif compliance > 0.98:
            current_confidence = current_policies.get("governance.thresholds.min_confidence", 0.8)
            
            if current_confidence < 0.9:
                new_confidence = min(0.9, current_confidence + 0.02)
                
                recommendations.append(PolicyRecommendation(
                    path="governance.thresholds.min_confidence",
                    current_value=current_confidence,
                    recommended_value=new_confidence,
                    rationale=f"Raise confidence threshold to maintain high standards ({compliance:.3f})",
                    confidence=insight.confidence * 0.8
                ))
        
        return recommendations
    
    async def _tune_fairness_policies(self, insight: Insight, 
                                    current_policies: Dict[str, Any]) -> List[PolicyRecommendation]:
        """Tune fairness-related policies."""
        recommendations = []
        
        evidence = insight.evidence.get("after", {})
        max_delta = evidence.get("max_delta", 0.0)
        
        if max_delta > 0.05:
            current_threshold = current_policies.get("governance.fairness.max_delta", 0.05)
            
            # Tighten fairness threshold
            new_threshold = max(0.02, current_threshold * 0.8)
            
            recommendations.append(PolicyRecommendation(
                path="governance.fairness.max_delta",
                current_value=current_threshold,
                recommended_value=new_threshold,
                rationale=f"Tighten fairness threshold due to high delta ({max_delta:.3f})",
                confidence=insight.confidence
            ))
        
        return recommendations
    
    async def _tune_drift_policies(self, insight: Insight, 
                                 current_policies: Dict[str, Any]) -> List[PolicyRecommendation]:
        """Tune drift detection policies."""
        recommendations = []
        
        evidence = insight.evidence.get("after", {})
        max_psi = evidence.get("max_psi", 0.0)
        
        if max_psi > 0.2:
            current_threshold = current_policies.get("governance.rollback.drift_threshold", 3.0)
            
            # Lower drift threshold for faster rollback
            new_threshold = max(2.0, current_threshold - 0.5)
            
            recommendations.append(PolicyRecommendation(
                path="governance.rollback.drift_threshold",
                current_value=current_threshold,
                recommended_value=new_threshold,
                rationale=f"Lower drift threshold for faster rollback (PSI: {max_psi:.3f})",
                confidence=insight.confidence
            ))
        
        return recommendations
    
    async def _tune_stability_policies(self, insight: Insight, 
                                     current_policies: Dict[str, Any]) -> List[PolicyRecommendation]:
        """Tune stability-related policies."""
        recommendations = []
        
        evidence = insight.evidence.get("after", {})
        issues = evidence.get("issues", [])
        
        if len(issues) > 1:
            # Adjust canary steps for more gradual rollout
            current_steps = current_policies.get("governance.canary.steps", [5, 25, 50, 100])
            
            # More conservative canary steps
            new_steps = [2, 5, 15, 35, 70, 100]
            
            recommendations.append(PolicyRecommendation(
                path="governance.canary.steps",
                current_value=current_steps,
                recommended_value=new_steps,
                rationale=f"More gradual canary rollout due to stability issues ({len(issues)} detected)",
                confidence=insight.confidence * 0.85
            ))
        
        return recommendations
    
    async def _tune_calibration_policies(self, insight: Insight, 
                                       current_policies: Dict[str, Any]) -> List[PolicyRecommendation]:
        """Tune calibration-related policies."""
        recommendations = []
        
        evidence = insight.evidence.get("after", {})
        avg_calibration = evidence.get("avg_calibration", 1.0)
        
        if avg_calibration < 0.85:
            current_threshold = current_policies.get("governance.thresholds.min_calibration", 0.95)
            
            # Temporarily lower calibration threshold
            new_threshold = max(0.8, avg_calibration + 0.05)
            
            recommendations.append(PolicyRecommendation(
                path="governance.thresholds.min_calibration",
                current_value=current_threshold,
                recommended_value=new_threshold,
                rationale=f"Temporarily lower calibration threshold ({avg_calibration:.3f})",
                confidence=insight.confidence * 0.9
            ))
        
        return recommendations
    
    def validate_recommendation(self, recommendation: PolicyRecommendation) -> bool:
        """Validate a policy recommendation against schema."""
        schema = self.policy_schema.get(recommendation.path)
        if not schema:
            return False
        
        value = recommendation.recommended_value
        
        if schema["type"] == float:
            if not isinstance(value, (int, float)):
                return False
            if "range" in schema:
                min_val, max_val = schema["range"]
                return min_val <= value <= max_val
                
        elif schema["type"] == list:
            if not isinstance(value, list):
                return False
                
        return True
    
    def get_recommendations_by_confidence(self, min_confidence: float = 0.8) -> List[PolicyRecommendation]:
        """Get recommendations above a confidence threshold."""
        return [rec for rec in self.recommendations if rec.confidence >= min_confidence]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy tuner statistics."""
        by_path = {}
        for rec in self.recommendations:
            path = rec.path
            if path not in by_path:
                by_path[path] = 0
            by_path[path] += 1
        
        return {
            "total_recommendations": len(self.recommendations),
            "by_policy_path": by_path,
            "avg_confidence": sum(rec.confidence for rec in self.recommendations) / len(self.recommendations) if self.recommendations else 0.0,
            "high_confidence_count": len(self.get_recommendations_by_confidence(0.8))
        }