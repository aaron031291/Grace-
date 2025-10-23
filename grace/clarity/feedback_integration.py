"""
Class 7: Feedback Integration - Links loop output to memory
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTIVE = "corrective"
    INFORMATIVE = "informative"


@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    feedback_id: str
    feedback_type: FeedbackType
    loop_id: str
    content: Dict[str, Any]
    impact_score: float = 0.5
    applied: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackIntegrator:
    """
    Integrates loop output feedback into memory system
    Core implementation of Class 7
    """
    
    def __init__(self, memory_bank=None):
        self.memory_bank = memory_bank
        self.feedback_queue: List[FeedbackEntry] = []
        self.feedback_history: Dict[str, FeedbackEntry] = {}
        self.loop_feedback_map: Dict[str, List[str]] = {}
        logger.info("FeedbackIntegrator initialized")
    
    def loop_output_to_memory(
        self,
        loop_id: str,
        loop_output: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.INFORMATIVE
    ) -> FeedbackEntry:
        """
        Link loop output to memory system
        Core implementation of Class 7
        """
        # Create feedback entry
        feedback = FeedbackEntry(
            feedback_id=f"fb_{loop_id}_{len(self.feedback_history)}",
            feedback_type=feedback_type,
            loop_id=loop_id,
            content=loop_output,
            impact_score=self._calculate_impact(loop_output)
        )
        
        # Store feedback
        self.feedback_history[feedback.feedback_id] = feedback
        self.feedback_queue.append(feedback)
        
        # Map to loop
        if loop_id not in self.loop_feedback_map:
            self.loop_feedback_map[loop_id] = []
        self.loop_feedback_map[loop_id].append(feedback.feedback_id)
        
        # Apply to memory bank if available
        if self.memory_bank:
            self._apply_to_memory(feedback)
        
        logger.info(f"Created feedback: {feedback.feedback_id} for loop {loop_id}")
        
        return feedback
    
    def _calculate_impact(self, output: Dict[str, Any]) -> float:
        """Calculate impact score of feedback"""
        impact = 0.5  # Base impact
        
        # Confidence contribution
        if 'confidence' in output:
            impact += output['confidence'] * 0.3
        
        # Success indication
        if output.get('success', False):
            impact += 0.2
        
        # Error indication (negative impact)
        if 'error' in output or output.get('failed', False):
            impact -= 0.3
        
        # Quality metrics
        if 'quality_score' in output:
            impact += output['quality_score'] * 0.2
        
        return max(0.0, min(1.0, impact))
    
    def _apply_to_memory(self, feedback: FeedbackEntry):
        """Apply feedback to memory bank"""
        try:
            # Store as memory
            from .memory_scoring import MemoryType
            
            memory_id = f"mem_{feedback.feedback_id}"
            
            self.memory_bank.store(
                memory_id=memory_id,
                memory_type=MemoryType.EPISODIC,
                content={
                    'loop_id': feedback.loop_id,
                    'feedback_type': feedback.feedback_type.value,
                    'output': feedback.content,
                    'impact': feedback.impact_score
                },
                source="feedback_loop",
                metadata={'feedback_id': feedback.feedback_id}
            )
            
            feedback.applied = True
            logger.debug(f"Applied feedback {feedback.feedback_id} to memory")
            
        except Exception as e:
            logger.error(f"Failed to apply feedback to memory: {e}")
    
    def process_feedback_queue(self) -> int:
        """Process queued feedback"""
        processed = 0
        
        for feedback in self.feedback_queue[:]:
            if not feedback.applied and self.memory_bank:
                self._apply_to_memory(feedback)
                processed += 1
        
        # Clear processed feedback
        self.feedback_queue = [f for f in self.feedback_queue if not f.applied]
        
        logger.info(f"Processed {processed} feedback entries")
        return processed
    
    def get_loop_feedback(self, loop_id: str) -> List[FeedbackEntry]:
        """Get all feedback for a specific loop"""
        feedback_ids = self.loop_feedback_map.get(loop_id, [])
        return [
            self.feedback_history[fid]
            for fid in feedback_ids
            if fid in self.feedback_history
        ]
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns"""
        if not self.feedback_history:
            return {'total_feedback': 0}
        
        feedback_list = list(self.feedback_history.values())
        
        by_type = {}
        for fb in feedback_list:
            fb_type = fb.feedback_type.value
            by_type[fb_type] = by_type.get(fb_type, 0) + 1
        
        avg_impact = sum(fb.impact_score for fb in feedback_list) / len(feedback_list)
        
        return {
            'total_feedback': len(feedback_list),
            'by_type': by_type,
            'avg_impact': avg_impact,
            'applied_count': sum(1 for fb in feedback_list if fb.applied),
            'queued_count': len(self.feedback_queue)
        }
