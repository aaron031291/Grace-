"""
Feedback Integration - Captures and applies loop feedback (Class 7)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Individual feedback record"""
    feedback_id: str
    loop_id: str
    decision_id: str
    feedback_type: str
    rating: float
    corrections: Optional[Dict[str, Any]]
    reviewer_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackIntegrator:
    """
    Captures and integrates feedback from previous loops
    
    Features:
    - Human corrections
    - Reviewer voting
    - Automatic evaluation
    - Weight/trust score adjustments
    - Feedback aggregation
    """
    
    def __init__(self, trust_manager=None, quorum_aggregator=None):
        self.trust_manager = trust_manager
        self.quorum_aggregator = quorum_aggregator
        
        self.feedback_records: List[FeedbackRecord] = []
        self.component_adjustments: Dict[str, float] = {}
        
        logger.info("FeedbackIntegrator initialized")
    
    def record_feedback(
        self,
        loop_id: str,
        decision_id: str,
        feedback_type: str,
        rating: float,
        corrections: Optional[Dict[str, Any]] = None,
        reviewer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackRecord:
        """Record feedback for a loop decision"""
        feedback = FeedbackRecord(
            feedback_id=f"fb_{len(self.feedback_records)}",
            loop_id=loop_id,
            decision_id=decision_id,
            feedback_type=feedback_type,
            rating=max(-1.0, min(1.0, rating)),  # Clamp to [-1, 1]
            corrections=corrections,
            reviewer_id=reviewer_id,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.feedback_records.append(feedback)
        
        # Apply adjustments
        self._apply_feedback(feedback)
        
        logger.info(f"Recorded {feedback_type} feedback: {rating:.2f} for {decision_id}")
        return feedback
    
    def _apply_feedback(self, feedback: FeedbackRecord):
        """Apply feedback to adjust weights and trust scores"""
        # Extract affected components from metadata
        affected_components = feedback.metadata.get("components", [])
        
        for component_id in affected_components:
            # Update trust manager if available
            if self.trust_manager:
                if feedback.rating > 0:
                    self.trust_manager.record_success(
                        component_id,
                        weight=abs(feedback.rating),
                        context={"feedback_type": feedback.feedback_type}
                    )
                else:
                    self.trust_manager.record_failure(
                        component_id,
                        severity=abs(feedback.rating),
                        context={"feedback_type": feedback.feedback_type}
                    )
            
            # Track cumulative adjustments
            if component_id not in self.component_adjustments:
                self.component_adjustments[component_id] = 0.0
            
            self.component_adjustments[component_id] += feedback.rating
            
            # Update quorum aggregator weights if available
            if self.quorum_aggregator:
                try:
                    # Convert rating to performance score (0-1)
                    performance = (feedback.rating + 1) / 2
                    self.quorum_aggregator.update_specialist_weight(
                        component_id, performance
                    )
                except:
                    pass
    
    def get_aggregated_feedback(
        self,
        loop_id: Optional[str] = None,
        decision_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregated feedback statistics"""
        filtered = self.feedback_records
        
        if loop_id:
            filtered = [f for f in filtered if f.loop_id == loop_id]
        
        if decision_id:
            filtered = [f for f in filtered if f.decision_id == decision_id]
        
        if not filtered:
            return {"total": 0}
        
        ratings = [f.rating for f in filtered]
        
        return {
            "total": len(filtered),
            "avg_rating": sum(ratings) / len(ratings),
            "positive": sum(1 for r in ratings if r > 0),
            "negative": sum(1 for r in ratings if r < 0),
            "by_type": self._aggregate_by_type(filtered),
            "component_adjustments": self.component_adjustments.copy()
        }
    
    def _aggregate_by_type(self, records: List[FeedbackRecord]) -> Dict[str, Dict[str, Any]]:
        """Aggregate feedback by type"""
        by_type = {}
        
        for record in records:
            if record.feedback_type not in by_type:
                by_type[record.feedback_type] = {"count": 0, "ratings": []}
            
            by_type[record.feedback_type]["count"] += 1
            by_type[record.feedback_type]["ratings"].append(record.rating)
        
        # Calculate averages
        for feedback_type, data in by_type.items():
            data["avg_rating"] = sum(data["ratings"]) / len(data["ratings"])
        
        return by_type
    
    def apply_corrections(
        self,
        decision: Dict[str, Any],
        decision_id: str
    ) -> Dict[str, Any]:
        """Apply accumulated corrections to a decision"""
        # Find corrections for this decision
        corrections_list = [
            f.corrections for f in self.feedback_records
            if f.decision_id == decision_id and f.corrections
        ]
        
        if not corrections_list:
            return decision
        
        corrected = decision.copy()
        
        # Apply corrections in order
        for corrections in corrections_list:
            for key, value in corrections.items():
                corrected[key] = value
        
        logger.info(f"Applied {len(corrections_list)} corrections to decision {decision_id}")
        return corrected
