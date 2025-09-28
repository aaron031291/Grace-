"""Policy fit specialist - evaluates how well content aligns with policies."""
from typing import Dict, List, Optional

from ....contracts.quorum_feed import QuorumFeedItem
from .specialist import BaseSpecialist


class PolicyFitSpecialist(BaseSpecialist):
    """Specialist for evaluating policy compliance and alignment."""
    
    def __init__(self):
        super().__init__(
            name="policy_fit_specialist", 
            expertise_domain="policy_compliance",
            confidence_threshold=0.7
        )
        
        # Policy alignment indicators
        self.policy_indicators = {
            "safety": [
                "safe", "secure", "protected", "compliant", "approved",
                "verified", "validated", "authorized"
            ],
            "ethical": [
                "ethical", "fair", "transparent", "accountable", "responsible",
                "unbiased", "inclusive", "respectful"
            ],
            "compliance": [
                "regulation", "standard", "guideline", "requirement", "policy",
                "audit", "review", "assessment", "certification"
            ],
            "risk": [
                "risk", "threat", "vulnerability", "concern", "issue",
                "danger", "hazard", "warning"
            ]
        }
        
        # Negative indicators
        self.negative_indicators = [
            "illegal", "unauthorized", "non-compliant", "violation", "breach",
            "harmful", "dangerous", "malicious", "suspicious", "forbidden"
        ]
    
    def _specialist_analysis(self, feed_items: List[QuorumFeedItem], context: Optional[Dict]) -> Dict:
        """Analyze policy fit across feed items."""
        if not feed_items:
            return self._empty_analysis()
        
        policy_scores = []
        detailed_analysis = []
        
        for item in feed_items:
            policy_score, analysis = self._analyze_policy_fit(item, context)
            policy_scores.append(policy_score)
            detailed_analysis.append({
                "memory_id": item.memory_id,
                "policy_score": policy_score,
                "analysis": analysis
            })
        
        # Aggregate assessment
        avg_policy_fit = sum(policy_scores) / len(policy_scores)
        compliant_count = sum(1 for score in policy_scores if score >= 0.7)
        risk_count = sum(1 for score in policy_scores if score < 0.5)
        
        # Generate opinion
        if avg_policy_fit >= 0.8 and risk_count == 0:
            opinion = "Strong policy alignment with no identified compliance risks"
            confidence = 0.9
        elif avg_policy_fit >= 0.6 and risk_count <= 1:
            opinion = "Good policy alignment with minimal compliance concerns"
            confidence = 0.75
        elif avg_policy_fit >= 0.4:
            opinion = "Moderate policy alignment with some compliance gaps"
            confidence = 0.6
        else:
            opinion = "Poor policy alignment with significant compliance risks"
            confidence = 0.4
        
        reasoning = f"Policy fit score: {avg_policy_fit:.2f}. " \
                   f"{compliant_count}/{len(feed_items)} items are policy compliant. " \
                   f"{risk_count} items flagged as potential risks."
        
        return {
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "average_policy_fit": avg_policy_fit,
                "compliant_count": compliant_count,
                "risk_count": risk_count,
                "total_items": len(feed_items)
            },
            "detailed_analysis": detailed_analysis,
            "evidence": [
                item.memory_id for item in feed_items 
                if policy_scores[feed_items.index(item)] >= 0.7
            ]
        }
    
    def _analyze_policy_fit(self, item: QuorumFeedItem, context: Optional[Dict]) -> tuple:
        """Analyze policy fit for a single item."""
        content = item.content.lower()
        
        # Score positive policy indicators
        positive_scores = {}
        for category, indicators in self.policy_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content)
            positive_scores[category] = min(1.0, score / len(indicators))
        
        # Score negative indicators (penalties)
        negative_score = sum(1 for indicator in self.negative_indicators if indicator in content)
        negative_penalty = min(0.5, negative_score * 0.1)  # Cap penalty at 0.5
        
        # Calculate base policy score
        avg_positive = sum(positive_scores.values()) / len(positive_scores)
        policy_score = max(0.0, avg_positive - negative_penalty)
        
        # Boost score for explicit policy language
        if any(word in content for word in ["policy", "compliant", "approved"]):
            policy_score = min(1.0, policy_score * 1.2)
        
        # Context-specific adjustments
        if context:
            required_policies = context.get("required_policies", [])
            if required_policies:
                policy_mentions = sum(1 for policy in required_policies if policy.lower() in content)
                if policy_mentions > 0:
                    policy_score = min(1.0, policy_score + (policy_mentions * 0.1))
        
        analysis = {
            "positive_scores": positive_scores,
            "negative_penalty": negative_penalty,
            "policy_language_boost": "policy" in content or "compliant" in content,
            "content_length": len(content)
        }
        
        return policy_score, analysis
    
    def is_applicable(self, context: Optional[Dict] = None) -> bool:
        """Applicable when policy evaluation is needed."""
        if not context:
            return True
        
        # Always apply for governance-related contexts
        request_type = context.get("request_type", "")
        return "policy" in request_type.lower() or "governance" in request_type.lower()