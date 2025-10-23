"""
Feedback analysis and insights
"""

from typing import Dict, Any, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """
    Analyzes feedback to identify trends and priorities
    """
    
    def __init__(self, feedback_collector):
        self.collector = feedback_collector
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze feedback trends"""
        all_feedback = self.collector.get_all_feedback()
        
        # Analyze by type
        type_counts = Counter(f["type"] for f in all_feedback)
        
        # Analyze by component
        component_counts = Counter(
            f.get("component", "unknown")
            for f in all_feedback
            if f.get("component")
        )
        
        # Analyze by priority
        priority_counts = Counter(f["priority"] for f in all_feedback)
        
        # Most voted items
        top_voted = sorted(all_feedback, key=lambda x: x["votes"], reverse=True)[:10]
        
        # Identify common themes
        themes = self._extract_themes(all_feedback)
        
        return {
            "total_feedback": len(all_feedback),
            "by_type": dict(type_counts),
            "by_component": dict(component_counts),
            "by_priority": dict(priority_counts),
            "top_voted": [
                {
                    "feedback_id": f["feedback_id"],
                    "title": f["title"],
                    "votes": f["votes"],
                    "type": f["type"]
                }
                for f in top_voted
            ],
            "themes": themes
        }
    
    def _extract_themes(self, feedback_list: List[Dict]) -> List[Dict[str, Any]]:
        """Extract common themes from feedback"""
        # Simple keyword-based theme extraction
        # In production, use NLP/ML for better analysis
        
        keywords = {
            "performance": ["slow", "latency", "performance", "speed"],
            "usability": ["difficult", "confusing", "unclear", "hard to use"],
            "documentation": ["docs", "documentation", "guide", "tutorial"],
            "features": ["feature", "add", "want", "need"],
            "bugs": ["bug", "error", "crash", "broken"]
        }
        
        theme_counts = Counter()
        
        for feedback in feedback_list:
            text = f"{feedback['title']} {feedback['description']}".lower()
            
            for theme, words in keywords.items():
                if any(word in text for word in words):
                    theme_counts[theme] += 1
        
        return [
            {"theme": theme, "count": count}
            for theme, count in theme_counts.most_common(5)
        ]
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on feedback"""
        trends = self.analyze_trends()
        recommendations = []
        
        # High-priority items
        high_priority = self.collector.get_all_feedback(priority="high")
        if len(high_priority) > 5:
            recommendations.append({
                "priority": "high",
                "recommendation": f"Address {len(high_priority)} high-priority items",
                "action": "Review high-priority feedback queue"
            })
        
        # Popular requests
        top_voted = trends["top_voted"]
        if top_voted:
            recommendations.append({
                "priority": "medium",
                "recommendation": f"Implement most-voted feature: {top_voted[0]['title']}",
                "action": f"Review feedback #{top_voted[0]['feedback_id']}"
            })
        
        # Component issues
        components = trends.get("by_component", {})
        if components:
            top_component = max(components, key=components.get)
            recommendations.append({
                "priority": "medium",
                "recommendation": f"Focus on {top_component} component ({components[top_component]} issues)",
                "action": f"Review {top_component} component stability"
            })
        
        return recommendations
