"""
Feedback API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from grace.auth.dependencies import get_current_user
from grace.auth.models import User
from grace.feedback import FeedbackCollector, FeedbackAnalyzer, FeedbackType, FeedbackPriority

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# Global collector (in production, use dependency injection)
_collector = None


def get_collector():
    global _collector
    if _collector is None:
        from grace.integration.event_bus import get_event_bus
        _collector = FeedbackCollector(event_bus=get_event_bus())
    return _collector


class FeedbackSubmission(BaseModel):
    """Feedback submission request"""
    type: str = Field(..., description="Feedback type")
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    priority: str = Field(default="medium")
    component: Optional[str] = None
    version: Optional[str] = None


class FeedbackComment(BaseModel):
    """Add comment to feedback"""
    comment: str = Field(..., min_length=1, max_length=500)


@router.post("/submit")
async def submit_feedback(
    feedback: FeedbackSubmission,
    current_user: User = Depends(get_current_user)
):
    """Submit user feedback"""
    collector = get_collector()
    
    try:
        feedback_type = FeedbackType(feedback.type)
        priority = FeedbackPriority(feedback.priority)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid type or priority: {e}")
    
    feedback_id = await collector.submit_feedback(
        user_id=current_user.user_id,
        feedback_type=feedback_type,
        title=feedback.title,
        description=feedback.description,
        priority=priority,
        component=feedback.component,
        version=feedback.version
    )
    
    return {
        "success": True,
        "feedback_id": feedback_id,
        "message": "Thank you for your feedback!"
    }


@router.post("/{feedback_id}/upvote")
async def upvote_feedback(
    feedback_id: str,
    current_user: User = Depends(get_current_user)
):
    """Upvote feedback item"""
    collector = get_collector()
    
    success = await collector.upvote_feedback(feedback_id, current_user.user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    return {"success": True, "message": "Feedback upvoted"}


@router.post("/{feedback_id}/comment")
async def add_comment(
    feedback_id: str,
    comment: FeedbackComment,
    current_user: User = Depends(get_current_user)
):
    """Add comment to feedback"""
    collector = get_collector()
    
    success = await collector.add_comment(
        feedback_id,
        current_user.user_id,
        comment.comment
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    return {"success": True, "message": "Comment added"}


@router.get("/")
async def list_feedback(
    status: Optional[str] = None,
    type: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """List feedback items"""
    collector = get_collector()
    
    feedback_list = collector.get_all_feedback(
        status=status,
        feedback_type=type,
        priority=priority,
        limit=limit
    )
    
    return {
        "feedback": feedback_list,
        "total": len(feedback_list)
    }


@router.get("/stats")
async def get_feedback_stats(current_user: User = Depends(get_current_user)):
    """Get feedback statistics"""
    collector = get_collector()
    return collector.get_stats()


@router.get("/analysis")
async def get_feedback_analysis(current_user: User = Depends(get_current_user)):
    """Get feedback analysis and trends"""
    collector = get_collector()
    analyzer = FeedbackAnalyzer(collector)
    
    trends = analyzer.analyze_trends()
    recommendations = analyzer.generate_recommendations()
    
    return {
        "trends": trends,
        "recommendations": recommendations
    }
