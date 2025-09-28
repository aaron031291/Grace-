"""Learning kernel - outcome recording and system adaptation."""
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class LearningOutcome(BaseModel):
    """Learning outcome record."""
    outcome_type: str
    context: Dict
    result: Dict
    success: bool
    confidence: float
    timestamp: datetime


class LearningKernel:
    """Handles outcome recording and system adaptation."""
    
    def __init__(self):
        self.outcomes = []
        self.adaptation_rules = {}
    
    def record_outcome(self, outcome_type: str, context: Dict, result: Dict, success: bool, confidence: float = 1.0):
        """Record a learning outcome."""
        outcome = LearningOutcome(
            outcome_type=outcome_type,
            context=context,
            result=result,
            success=success,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        
        self.outcomes.append(outcome)
        
        # Trigger adaptation if needed
        self._trigger_adaptation(outcome)
    
    def adapt(self) -> Dict:
        """Analyze outcomes and adapt system behavior."""
        if not self.outcomes:
            return {"adaptations": 0, "message": "No outcomes to analyze"}
        
        recent_outcomes = self.outcomes[-100:]  # Last 100 outcomes
        
        # Simple adaptation logic
        adaptations = {}
        
        # Analyze success rates by outcome type
        success_rates = {}
        for outcome in recent_outcomes:
            outcome_type = outcome.outcome_type
            if outcome_type not in success_rates:
                success_rates[outcome_type] = {"total": 0, "success": 0}
            
            success_rates[outcome_type]["total"] += 1
            if outcome.success:
                success_rates[outcome_type]["success"] += 1
        
        # Generate adaptations for low success rates
        for outcome_type, stats in success_rates.items():
            success_rate = stats["success"] / stats["total"]
            if success_rate < 0.7 and stats["total"] >= 5:  # Low success with enough samples
                adaptations[outcome_type] = f"Consider adjustment - success rate: {success_rate:.2%}"
        
        return {
            "adaptations": len(adaptations),
            "recommendations": adaptations,
            "analyzed_outcomes": len(recent_outcomes)
        }
    
    def _trigger_adaptation(self, outcome: LearningOutcome):
        """Trigger adaptation based on outcome."""
        # Simple rule-based adaptation trigger
        if not outcome.success and outcome.confidence > 0.8:
            # High confidence failure - may need adaptation
            outcome_type = outcome.outcome_type
            if outcome_type not in self.adaptation_rules:
                self.adaptation_rules[outcome_type] = {"failure_count": 0}
            
            self.adaptation_rules[outcome_type]["failure_count"] += 1
    
    def get_stats(self) -> Dict:
        """Get learning kernel statistics."""
        total_outcomes = len(self.outcomes)
        if total_outcomes == 0:
            return {"total_outcomes": 0, "success_rate": 0.0}
        
        successful_outcomes = sum(1 for o in self.outcomes if o.success)
        success_rate = successful_outcomes / total_outcomes
        
        return {
            "total_outcomes": total_outcomes,
            "successful_outcomes": successful_outcomes,
            "success_rate": success_rate,
            "adaptation_rules": len(self.adaptation_rules),
            "recent_outcomes": len(self.outcomes[-10:])  # Last 10
        }