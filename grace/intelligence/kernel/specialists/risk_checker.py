"""Risk checker specialist - identifies and assesses risks in content."""
from typing import Dict, List, Optional
import re

from ....contracts.quorum_feed import QuorumFeedItem
from .specialist import BaseSpecialist


class RiskCheckerSpecialist(BaseSpecialist):
    """Specialist for identifying and assessing risks."""
    
    def __init__(self):
        super().__init__(
            name="risk_checker_specialist",
            expertise_domain="risk_assessment",
            confidence_threshold=0.8
        )
        
        # Risk patterns by category
        self.risk_patterns = {
            "security": [
                r"\b(password|credential|token|key|secret)\b",
                r"\b(hack|exploit|vulnerability|breach|attack)\b",
                r"\b(malware|virus|trojan|spyware)\b"
            ],
            "privacy": [
                r"\b(ssn|social security|\d{3}-\d{2}-\d{4})\b",
                r"\b(credit card|\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",
                r"\b(email|phone|address|personal data)\b"
            ],
            "financial": [
                r"\b(fraud|scam|money laundering|embezzle)\b",
                r"\b(unauthorized|illegal transaction|financial crime)\b",
                r"\$\d{4,}",  # Large monetary amounts
            ],
            "operational": [
                r"\b(failure|error|crash|down|outage)\b",
                r"\b(critical|emergency|urgent|severe)\b",
                r"\b(loss|damage|corruption|deleted)\b"
            ],
            "compliance": [
                r"\b(violation|breach|non-compliant|illegal)\b",
                r"\b(audit failure|regulatory|fine|penalty)\b",
                r"\b(unauthorized access|data breach)\b"
            ]
        }
        
        # Risk severity keywords
        self.severity_keywords = {
            "critical": ["critical", "severe", "catastrophic", "disaster"],
            "high": ["high risk", "dangerous", "threat", "major"],
            "medium": ["moderate", "concern", "potential", "possible"],
            "low": ["minor", "slight", "negligible", "minimal"]
        }
    
    def _specialist_analysis(self, feed_items: List[QuorumFeedItem], context: Optional[Dict]) -> Dict:
        """Analyze risks across feed items."""
        if not feed_items:
            return self._empty_analysis()
        
        risk_assessments = []
        total_risk_score = 0
        risk_categories = {category: 0 for category in self.risk_patterns.keys()}
        
        for item in feed_items:
            risk_assessment = self._analyze_item_risks(item)
            risk_assessments.append(risk_assessment)
            total_risk_score += risk_assessment["overall_risk_score"]
            
            # Aggregate category risks
            for category, score in risk_assessment["category_risks"].items():
                risk_categories[category] += score
        
        # Calculate aggregate metrics
        avg_risk_score = total_risk_score / len(feed_items)
        high_risk_count = sum(1 for assessment in risk_assessments if assessment["overall_risk_score"] >= 0.7)
        
        # Determine overall risk level
        if avg_risk_score >= 0.8:
            risk_level = "CRITICAL"
            confidence = 0.95
            opinion = "Critical risks identified requiring immediate attention"
        elif avg_risk_score >= 0.6:
            risk_level = "HIGH" 
            confidence = 0.85
            opinion = "High risk level detected, review and mitigation recommended"
        elif avg_risk_score >= 0.4:
            risk_level = "MEDIUM"
            confidence = 0.75
            opinion = "Moderate risk level, monitoring and controls advised"
        elif avg_risk_score >= 0.2:
            risk_level = "LOW"
            confidence = 0.7
            opinion = "Low risk level with minor concerns identified"
        else:
            risk_level = "MINIMAL"
            confidence = 0.8
            opinion = "Minimal risks detected, acceptable risk profile"
        
        # Find top risk category
        top_risk_category = max(risk_categories.items(), key=lambda x: x[1])[0]
        
        reasoning = f"Average risk score: {avg_risk_score:.2f} ({risk_level}). " \
                   f"{high_risk_count}/{len(feed_items)} items flagged as high risk. " \
                   f"Primary risk category: {top_risk_category}."
        
        return {
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "overall_risk_level": risk_level,
                "average_risk_score": avg_risk_score,
                "high_risk_count": high_risk_count,
                "risk_categories": risk_categories,
                "top_risk_category": top_risk_category
            },
            "detailed_assessments": risk_assessments,
            "evidence": [
                assessment["memory_id"] for assessment in risk_assessments 
                if assessment["overall_risk_score"] >= 0.6
            ]
        }
    
    def _analyze_item_risks(self, item: QuorumFeedItem) -> Dict:
        """Analyze risks in a single feed item."""
        content = item.content.lower()
        
        # Check each risk category
        category_risks = {}
        category_matches = {}
        
        for category, patterns in self.risk_patterns.items():
            matches = []
            total_matches = 0
            
            for pattern in patterns:
                pattern_matches = re.findall(pattern, content, re.IGNORECASE)
                if pattern_matches:
                    matches.extend(pattern_matches)
                    total_matches += len(pattern_matches)
            
            # Score based on number of matches (normalized)
            risk_score = min(1.0, total_matches / 3)  # Cap at 3 matches
            category_risks[category] = risk_score
            category_matches[category] = matches
        
        # Calculate overall risk score (weighted by category importance)
        category_weights = {
            "security": 0.3,
            "privacy": 0.25,
            "compliance": 0.2,
            "financial": 0.15,
            "operational": 0.1
        }
        
        overall_risk_score = sum(
            category_risks[cat] * category_weights[cat] 
            for cat in category_risks
        )
        
        # Adjust for severity keywords
        severity_multiplier = self._detect_severity(content)
        overall_risk_score = min(1.0, overall_risk_score * severity_multiplier)
        
        return {
            "memory_id": item.memory_id,
            "overall_risk_score": overall_risk_score,
            "category_risks": category_risks,
            "category_matches": category_matches,
            "severity_multiplier": severity_multiplier
        }
    
    def _detect_severity(self, content: str) -> float:
        """Detect severity indicators and return multiplier."""
        for severity, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    if severity == "critical":
                        return 1.5
                    elif severity == "high":
                        return 1.3
                    elif severity == "medium":
                        return 1.1
                    elif severity == "low":
                        return 0.8
        
        return 1.0  # Default multiplier
    
    def is_applicable(self, context: Optional[Dict] = None) -> bool:
        """Always applicable for risk assessment."""
        return True