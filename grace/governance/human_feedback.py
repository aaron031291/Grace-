"""
Human UX Feedback Loop - Interface for human review and grading of Grace's sandbox outputs.

This module provides the human-in-the-loop feedback system that allows humans to:
- Review and grade sandbox experiments
- Provide feedback on Grace's reasoning and decisions
- Approve or reject sandbox merges
- Guide Grace's learning and evolution
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict

from ..layer_04_audit_logs.immutable_logs import ImmutableLogs
from ..core.contracts import Experience, generate_correlation_id

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of human feedback."""
    SANDBOX_REVIEW = "sandbox_review"
    EXPERIMENT_EVALUATION = "experiment_evaluation"
    MERGE_APPROVAL = "merge_approval"
    POLICY_FEEDBACK = "policy_feedback"
    GENERAL_GUIDANCE = "general_guidance"


class FeedbackScore(Enum):
    """Standardized feedback scores."""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    NEEDS_IMPROVEMENT = 2
    POOR = 1


@dataclass
class HumanFeedback:
    """Human feedback on Grace's sandbox outputs."""
    feedback_id: str
    feedback_type: FeedbackType
    target_id: str  # sandbox_id, experiment_id, etc.
    reviewer_id: str
    score: FeedbackScore
    approval_status: str  # "approved", "rejected", "requires_revision"
    detailed_feedback: str
    specific_comments: List[str]
    suggestions: List[str]
    areas_of_concern: List[str]
    positive_aspects: List[str]
    confidence_in_feedback: float  # 0.0 to 1.0
    timestamp: datetime
    processing_time_seconds: float = 0.0


@dataclass
class FeedbackRequest:
    """Request for human feedback on a sandbox or experiment."""
    request_id: str
    feedback_type: FeedbackType
    target_id: str
    title: str
    description: str
    context_data: Dict[str, Any]
    priority: str = "normal"  # "low", "normal", "high", "critical"
    deadline: Optional[datetime] = None
    requested_reviewers: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.deadline is None:
            # Default deadline is 24 hours for normal priority
            hours_map = {"low": 72, "normal": 24, "high": 4, "critical": 1}
            self.deadline = self.created_at + timedelta(hours=hours_map.get(self.priority, 24))


@dataclass
class ReviewSession:
    """Human review session data."""
    session_id: str
    reviewer_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    feedback_requests_reviewed: List[str] = None
    total_items_reviewed: int = 0
    average_review_time: float = 0.0
    
    def __post_init__(self):
        if self.feedback_requests_reviewed is None:
            self.feedback_requests_reviewed = []


class HumanFeedbackInterface:
    """
    Interface for collecting and managing human feedback on Grace's sandbox outputs.
    Provides both programmatic API and web interface integration points.
    """
    
    def __init__(
        self,
        immutable_logs: Optional[ImmutableLogs] = None,
        feedback_callback: Optional[Callable] = None
    ):
        self.immutable_logs = immutable_logs or ImmutableLogs("feedback_audit.db")
        self.feedback_callback = feedback_callback
        
        # Active feedback requests awaiting human review
        self.pending_feedback_requests: Dict[str, FeedbackRequest] = {}
        
        # Completed feedback
        self.feedback_history: List[HumanFeedback] = []
        
        # Active review sessions
        self.active_sessions: Dict[str, ReviewSession] = {}
        
        # Reviewer performance tracking
        self.reviewer_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Learning from feedback patterns
        self.feedback_patterns = {
            "common_issues": {},
            "reviewer_preferences": {},
            "improvement_trends": []
        }
    
    async def request_feedback(
        self,
        feedback_type: FeedbackType,
        target_id: str,
        title: str,
        description: str,
        context_data: Dict[str, Any],
        priority: str = "normal",
        requested_reviewers: Optional[List[str]] = None
    ) -> str:
        """Request human feedback on a sandbox output."""
        
        request_id = f"feedback_{generate_correlation_id()}"
        
        feedback_request = FeedbackRequest(
            request_id=request_id,
            feedback_type=feedback_type,
            target_id=target_id,
            title=title,
            description=description,
            context_data=context_data,
            priority=priority,
            requested_reviewers=requested_reviewers or []
        )
        
        self.pending_feedback_requests[request_id] = feedback_request
        
        # Log the feedback request
        await self.immutable_logs.log_governance_action(
            action_type="feedback_requested",
            data={
                "request_id": request_id,
                "feedback_type": feedback_type.value,
                "target_id": target_id,
                "priority": priority,
                "requested_reviewers": requested_reviewers
            },
            transparency_level="democratic_oversight"
        )
        
        # Notify via callback if available
        if self.feedback_callback:
            await self.feedback_callback("feedback_requested", feedback_request)
        
        logger.info(f"Requested {feedback_type.value} feedback for {target_id} (priority: {priority})")
        return request_id
    
    async def submit_feedback(
        self,
        request_id: str,
        reviewer_id: str,
        score: FeedbackScore,
        approval_status: str,
        detailed_feedback: str,
        specific_comments: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        areas_of_concern: Optional[List[str]] = None,
        positive_aspects: Optional[List[str]] = None,
        confidence: float = 0.8
    ) -> str:
        """Submit human feedback for a pending request."""
        
        if request_id not in self.pending_feedback_requests:
            raise ValueError(f"Feedback request {request_id} not found")
        
        request = self.pending_feedback_requests[request_id]
        feedback_id = f"feedback_{generate_correlation_id()}"
        
        feedback = HumanFeedback(
            feedback_id=feedback_id,
            feedback_type=request.feedback_type,
            target_id=request.target_id,
            reviewer_id=reviewer_id,
            score=score,
            approval_status=approval_status,
            detailed_feedback=detailed_feedback,
            specific_comments=specific_comments or [],
            suggestions=suggestions or [],
            areas_of_concern=areas_of_concern or [],
            positive_aspects=positive_aspects or [],
            confidence_in_feedback=confidence,
            timestamp=datetime.now(),
            processing_time_seconds=(datetime.now() - request.created_at).total_seconds()
        )
        
        # Move from pending to completed
        self.feedback_history.append(feedback)
        del self.pending_feedback_requests[request_id]
        
        # Update reviewer profile
        self._update_reviewer_profile(reviewer_id, feedback)
        
        # Learn from feedback patterns
        self._analyze_feedback_patterns(feedback)
        
        # Log the feedback submission
        await self.immutable_logs.log_governance_action(
            action_type="feedback_submitted",
            data={
                "feedback_id": feedback_id,
                "request_id": request_id,
                "reviewer_id": reviewer_id,
                "score": score.value,
                "approval_status": approval_status,
                "target_id": request.target_id
            },
            transparency_level="democratic_oversight"
        )
        
        # Notify via callback
        if self.feedback_callback:
            await self.feedback_callback("feedback_submitted", feedback)
        
        logger.info(f"Received feedback {feedback_id} from {reviewer_id}: {approval_status} (score: {score.value})")
        return feedback_id
    
    async def start_review_session(self, reviewer_id: str) -> str:
        """Start a human review session."""
        session_id = f"session_{generate_correlation_id()}"
        
        session = ReviewSession(
            session_id=session_id,
            reviewer_id=reviewer_id,
            started_at=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Started review session {session_id} for reviewer {reviewer_id}")
        return session_id
    
    async def end_review_session(self, session_id: str) -> Dict[str, Any]:
        """End a review session and return summary."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Review session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.completed_at = datetime.now()
        
        # Calculate session metrics
        session_duration = (session.completed_at - session.started_at).total_seconds()
        session.average_review_time = (
            session_duration / session.total_items_reviewed 
            if session.total_items_reviewed > 0 else 0.0
        )
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        session_summary = {
            "session_id": session_id,
            "reviewer_id": session.reviewer_id,
            "duration_seconds": session_duration,
            "items_reviewed": session.total_items_reviewed,
            "average_review_time": session.average_review_time,
            "feedback_requests_completed": len(session.feedback_requests_reviewed)
        }
        
        logger.info(f"Completed review session {session_id}: {session.total_items_reviewed} items reviewed")
        return session_summary
    
    def get_pending_feedback_requests(
        self,
        reviewer_id: Optional[str] = None,
        priority: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None
    ) -> List[Dict[str, Any]]:
        """Get pending feedback requests, optionally filtered."""
        
        requests = list(self.pending_feedback_requests.values())
        
        # Apply filters
        if reviewer_id:
            requests = [r for r in requests if not r.requested_reviewers or reviewer_id in r.requested_reviewers]
        
        if priority:
            requests = [r for r in requests if r.priority == priority]
        
        if feedback_type:
            requests = [r for r in requests if r.feedback_type == feedback_type]
        
        # Sort by priority and deadline
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        requests.sort(key=lambda x: (priority_order.get(x.priority, 2), x.deadline))
        
        return [asdict(request) for request in requests]
    
    def get_feedback_history(
        self,
        target_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get feedback history, optionally filtered."""
        
        history = self.feedback_history
        
        if target_id:
            history = [f for f in history if f.target_id == target_id]
        
        if reviewer_id:
            history = [f for f in history if f.reviewer_id == reviewer_id]
        
        if feedback_type:
            history = [f for f in history if f.feedback_type == feedback_type]
        
        # Sort by timestamp, most recent first
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [asdict(feedback) for feedback in history[:limit]]
    
    def get_reviewer_dashboard(self, reviewer_id: str) -> Dict[str, Any]:
        """Get dashboard data for a specific reviewer."""
        
        reviewer_feedback = [f for f in self.feedback_history if f.reviewer_id == reviewer_id]
        pending_requests = self.get_pending_feedback_requests(reviewer_id=reviewer_id)
        
        # Calculate reviewer metrics
        total_reviews = len(reviewer_feedback)
        avg_score = (
            sum(f.score.value for f in reviewer_feedback) / total_reviews 
            if total_reviews > 0 else 0.0
        )
        
        approval_rate = (
            len([f for f in reviewer_feedback if f.approval_status == "approved"]) / total_reviews
            if total_reviews > 0 else 0.0
        )
        
        avg_confidence = (
            sum(f.confidence_in_feedback for f in reviewer_feedback) / total_reviews
            if total_reviews > 0 else 0.0
        )
        
        return {
            "reviewer_id": reviewer_id,
            "total_reviews_completed": total_reviews,
            "pending_reviews": len(pending_requests),
            "average_score_given": round(avg_score, 2),
            "approval_rate": round(approval_rate, 2),
            "average_confidence": round(avg_confidence, 2),
            "recent_feedback": [asdict(f) for f in reviewer_feedback[-5:]],
            "pending_requests": pending_requests[:10],  # Show top 10 pending
            "reviewer_profile": self.reviewer_profiles.get(reviewer_id, {})
        }
    
    def get_system_feedback_metrics(self) -> Dict[str, Any]:
        """Get system-wide feedback metrics."""
        
        total_requests = len(self.feedback_history) + len(self.pending_feedback_requests)
        completed_requests = len(self.feedback_history)
        
        if completed_requests > 0:
            avg_processing_time = sum(f.processing_time_seconds for f in self.feedback_history) / completed_requests
            
            score_distribution = {}
            for score in FeedbackScore:
                count = len([f for f in self.feedback_history if f.score == score])
                score_distribution[score.name] = count
            
            approval_stats = {
                "approved": len([f for f in self.feedback_history if f.approval_status == "approved"]),
                "rejected": len([f for f in self.feedback_history if f.approval_status == "rejected"]),
                "requires_revision": len([f for f in self.feedback_history if f.approval_status == "requires_revision"])
            }
        else:
            avg_processing_time = 0.0
            score_distribution = {}
            approval_stats = {}
        
        return {
            "total_feedback_requests": total_requests,
            "completed_reviews": completed_requests,
            "pending_reviews": len(self.pending_feedback_requests),
            "completion_rate": completed_requests / total_requests if total_requests > 0 else 0.0,
            "average_processing_time_seconds": avg_processing_time,
            "score_distribution": score_distribution,
            "approval_statistics": approval_stats,
            "active_reviewers": len(set(f.reviewer_id for f in self.feedback_history)),
            "common_feedback_patterns": self.feedback_patterns["common_issues"],
            "improvement_trends": self.feedback_patterns["improvement_trends"][-10:]  # Last 10 trends
        }
    
    def _update_reviewer_profile(self, reviewer_id: str, feedback: HumanFeedback):
        """Update reviewer profile based on submitted feedback."""
        
        if reviewer_id not in self.reviewer_profiles:
            self.reviewer_profiles[reviewer_id] = {
                "first_review_date": feedback.timestamp.isoformat(),
                "total_reviews": 0,
                "expertise_areas": set(),
                "average_confidence": 0.0,
                "preferred_feedback_types": {},
                "quality_indicators": {
                    "detailed_feedback_ratio": 0.0,
                    "suggestions_provided_ratio": 0.0,
                    "consistency_score": 0.0
                }
            }
        
        profile = self.reviewer_profiles[reviewer_id]
        profile["total_reviews"] += 1
        profile["last_review_date"] = feedback.timestamp.isoformat()
        
        # Update expertise areas based on feedback type
        if hasattr(profile["expertise_areas"], "add"):
            profile["expertise_areas"].add(feedback.feedback_type.value)
        else:
            profile["expertise_areas"] = set([feedback.feedback_type.value])
        
        # Update preferred feedback types
        feedback_type = feedback.feedback_type.value
        if feedback_type not in profile["preferred_feedback_types"]:
            profile["preferred_feedback_types"][feedback_type] = 0
        profile["preferred_feedback_types"][feedback_type] += 1
        
        # Recalculate quality indicators
        reviewer_feedback = [f for f in self.feedback_history if f.reviewer_id == reviewer_id]
        total_reviews = len(reviewer_feedback)
        
        if total_reviews > 0:
            profile["average_confidence"] = sum(f.confidence_in_feedback for f in reviewer_feedback) / total_reviews
            
            detailed_count = len([f for f in reviewer_feedback if len(f.detailed_feedback) > 50])
            profile["quality_indicators"]["detailed_feedback_ratio"] = detailed_count / total_reviews
            
            suggestions_count = len([f for f in reviewer_feedback if f.suggestions])
            profile["quality_indicators"]["suggestions_provided_ratio"] = suggestions_count / total_reviews
    
    def _analyze_feedback_patterns(self, feedback: HumanFeedback):
        """Analyze feedback to identify patterns and learning opportunities."""
        
        # Track common issues mentioned in areas of concern
        for concern in feedback.areas_of_concern:
            concern_lower = concern.lower()
            if concern_lower not in self.feedback_patterns["common_issues"]:
                self.feedback_patterns["common_issues"][concern_lower] = 0
            self.feedback_patterns["common_issues"][concern_lower] += 1
        
        # Track reviewer preferences
        reviewer_id = feedback.reviewer_id
        if reviewer_id not in self.feedback_patterns["reviewer_preferences"]:
            self.feedback_patterns["reviewer_preferences"][reviewer_id] = {
                "strict_scoring": False,
                "detailed_feedback": False,
                "focus_areas": []
            }
        
        # Update reviewer preferences based on this feedback
        prefs = self.feedback_patterns["reviewer_preferences"][reviewer_id]
        
        if feedback.score.value <= 2:  # Strict scoring if giving low scores
            prefs["strict_scoring"] = True
        
        if len(feedback.detailed_feedback) > 100:  # Detailed feedback if long descriptions
            prefs["detailed_feedback"] = True
        
        # Add improvement trend data
        trend_data = {
            "timestamp": feedback.timestamp.isoformat(),
            "feedback_type": feedback.feedback_type.value,
            "score": feedback.score.value,
            "approval_status": feedback.approval_status,
            "has_suggestions": len(feedback.suggestions) > 0
        }
        
        self.feedback_patterns["improvement_trends"].append(trend_data)
        
        # Keep only recent trends (last 100)
        if len(self.feedback_patterns["improvement_trends"]) > 100:
            self.feedback_patterns["improvement_trends"] = self.feedback_patterns["improvement_trends"][-100:]
    
    async def generate_feedback_insights(self) -> Dict[str, Any]:
        """Generate insights from feedback patterns to guide Grace's improvement."""
        
        insights = {
            "most_common_issues": [],
            "reviewer_consensus_areas": [],
            "improvement_recommendations": [],
            "learning_priorities": []
        }
        
        # Analyze most common issues
        common_issues = self.feedback_patterns["common_issues"]
        if common_issues:
            sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)
            insights["most_common_issues"] = [
                {"issue": issue, "frequency": count} 
                for issue, count in sorted_issues[:10]
            ]
        
        # Analyze areas where reviewers consistently agree
        feedback_by_target = {}
        for feedback in self.feedback_history:
            target = feedback.target_id
            if target not in feedback_by_target:
                feedback_by_target[target] = []
            feedback_by_target[target].append(feedback)
        
        consensus_areas = []
        for target_id, target_feedback in feedback_by_target.items():
            if len(target_feedback) >= 2:  # Multiple reviewers
                scores = [f.score.value for f in target_feedback]
                if max(scores) - min(scores) <= 1:  # Close agreement
                    avg_score = sum(scores) / len(scores)
                    consensus_areas.append({
                        "target_id": target_id,
                        "reviewer_count": len(target_feedback),
                        "average_score": avg_score,
                        "consensus_level": "high"
                    })
        
        insights["reviewer_consensus_areas"] = consensus_areas[:5]
        
        # Generate improvement recommendations based on patterns
        recommendations = []
        
        if "security" in [issue for issue, _ in common_issues.items()][:5]:
            recommendations.append("Focus on security validation and testing procedures")
        
        if "performance" in [issue for issue, _ in common_issues.items()][:5]:
            recommendations.append("Enhance performance benchmarking and optimization")
        
        low_score_feedback = [f for f in self.feedback_history if f.score.value <= 2]
        if len(low_score_feedback) > len(self.feedback_history) * 0.2:  # More than 20% low scores
            recommendations.append("Review and improve fundamental processes based on consistent low scores")
        
        insights["improvement_recommendations"] = recommendations
        
        # Determine learning priorities
        priorities = []
        
        # Priority based on feedback frequency and severity
        for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:3]:
            if count >= 3:  # Mentioned at least 3 times
                priorities.append(f"Address recurring issue: {issue}")
        
        insights["learning_priorities"] = priorities
        
        return insights
    
    async def export_feedback_data(self, format: str = "json") -> str:
        """Export feedback data for external analysis."""
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_feedback_entries": len(self.feedback_history),
            "pending_requests": len(self.pending_feedback_requests),
            "feedback_history": [asdict(f) for f in self.feedback_history],
            "pending_requests": [asdict(r) for r in self.pending_feedback_requests.values()],
            "reviewer_profiles": {
                reviewer_id: {
                    **profile,
                    "expertise_areas": list(profile.get("expertise_areas", set()))
                }
                for reviewer_id, profile in self.reviewer_profiles.items()
            },
            "feedback_patterns": self.feedback_patterns,
            "system_metrics": self.get_system_feedback_metrics()
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old feedback data beyond retention period."""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Archive old feedback
        old_feedback = [f for f in self.feedback_history if f.timestamp < cutoff_date]
        self.feedback_history = [f for f in self.feedback_history if f.timestamp >= cutoff_date]
        
        # Clean up old pending requests (these should be escalated or cancelled)
        old_requests = [r for r in self.pending_feedback_requests.values() if r.created_at < cutoff_date]
        for old_request in old_requests:
            del self.pending_feedback_requests[old_request.request_id]
        
        logger.info(
            f"Cleaned up {len(old_feedback)} old feedback entries and "
            f"{len(old_requests)} expired requests (retention: {retention_days} days)"
        )
        
        return {
            "archived_feedback": len(old_feedback),
            "expired_requests": len(old_requests),
            "retention_days": retention_days
        }