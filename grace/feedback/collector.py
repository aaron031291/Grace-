"""
Feedback collection system
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    IMPROVEMENT = "improvement"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    GENERAL = "general"


class FeedbackPriority(Enum):
    """Feedback priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackCollector:
    """
    Collects and stores user feedback
    """
    
    def __init__(self, storage=None, event_bus=None):
        self.storage = storage or []  # In-memory storage, use DB in production
        self.event_bus = event_bus
        
        self.feedback_count = 0
        self.feedback_by_type = {}
    
    async def submit_feedback(
        self,
        user_id: str,
        feedback_type: FeedbackType,
        title: str,
        description: str,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        component: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit user feedback
        
        Args:
            user_id: User submitting feedback
            feedback_type: Type of feedback
            title: Brief title
            description: Detailed description
            priority: Priority level
            component: System component affected
            version: System version
            metadata: Additional context
        
        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        
        feedback = {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "type": feedback_type.value,
            "title": title,
            "description": description,
            "priority": priority.value,
            "component": component,
            "version": version,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "status": "open",
            "votes": 0,
            "comments": []
        }
        
        # Store feedback
        self.storage.append(feedback)
        self.feedback_count += 1
        
        # Track by type
        type_key = feedback_type.value
        self.feedback_by_type[type_key] = self.feedback_by_type.get(type_key, 0) + 1
        
        # Emit event
        if self.event_bus:
            from grace.schemas.events import GraceEvent
            
            event = GraceEvent(
                event_type="feedback.submitted",
                source="feedback_collector",
                payload={
                    "feedback_id": feedback_id,
                    "type": feedback_type.value,
                    "priority": priority.value,
                    "user_id": user_id
                }
            )
            await self.event_bus.emit(event)
        
        logger.info(f"Feedback submitted: {feedback_id}", extra={
            "feedback_id": feedback_id,
            "type": feedback_type.value,
            "user_id": user_id
        })
        
        return feedback_id
    
    async def upvote_feedback(self, feedback_id: str, user_id: str) -> bool:
        """Upvote feedback item"""
        for feedback in self.storage:
            if feedback["feedback_id"] == feedback_id:
                feedback["votes"] += 1
                
                logger.info(f"Feedback upvoted: {feedback_id}", extra={
                    "feedback_id": feedback_id,
                    "votes": feedback["votes"]
                })
                
                return True
        
        return False
    
    async def add_comment(
        self,
        feedback_id: str,
        user_id: str,
        comment: str
    ) -> bool:
        """Add comment to feedback"""
        for feedback in self.storage:
            if feedback["feedback_id"] == feedback_id:
                feedback["comments"].append({
                    "user_id": user_id,
                    "comment": comment,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                return True
        
        return False
    
    async def update_status(
        self,
        feedback_id: str,
        new_status: str,
        resolution: Optional[str] = None
    ) -> bool:
        """Update feedback status"""
        for feedback in self.storage:
            if feedback["feedback_id"] == feedback_id:
                feedback["status"] = new_status
                feedback["updated_at"] = datetime.utcnow().isoformat()
                
                if resolution:
                    feedback["resolution"] = resolution
                
                logger.info(f"Feedback status updated: {feedback_id} -> {new_status}")
                
                return True
        
        return False
    
    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """Get specific feedback item"""
        for feedback in self.storage:
            if feedback["feedback_id"] == feedback_id:
                return feedback
        
        return None
    
    def get_all_feedback(
        self,
        status: Optional[str] = None,
        feedback_type: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get feedback items with filters"""
        results = self.storage.copy()
        
        if status:
            results = [f for f in results if f["status"] == status]
        
        if feedback_type:
            results = [f for f in results if f["type"] == feedback_type]
        
        if priority:
            results = [f for f in results if f["priority"] == priority]
        
        # Sort by votes and timestamp
        results.sort(key=lambda x: (x["votes"], x["timestamp"]), reverse=True)
        
        return results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        open_count = sum(1 for f in self.storage if f["status"] == "open")
        
        return {
            "total_feedback": self.feedback_count,
            "open_feedback": open_count,
            "closed_feedback": self.feedback_count - open_count,
            "by_type": self.feedback_by_type,
            "top_voted": sorted(
                self.storage,
                key=lambda x: x["votes"],
                reverse=True
            )[:5]
        }
