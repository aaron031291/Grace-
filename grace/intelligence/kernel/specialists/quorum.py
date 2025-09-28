"""Quorum specialist - facilitates consensus among other specialists."""
from typing import Dict, List, Optional
import statistics

from ....contracts.quorum_feed import QuorumFeedItem, QuorumResult
from .specialist import BaseSpecialist


class QuorumSpecialist(BaseSpecialist):
    """Meta-specialist that facilitates consensus among other specialists."""
    
    def __init__(self):
        super().__init__(
            name="quorum_specialist",
            expertise_domain="consensus_facilitation",
            confidence_threshold=0.6
        )
        
        self.consensus_threshold = 0.7  # Agreement threshold for consensus
    
    def facilitate_consensus(self, specialist_analyses: List[Dict], feed_items: List[QuorumFeedItem]) -> QuorumResult:
        """Facilitate consensus among specialist analyses."""
        if not specialist_analyses:
            return QuorumResult(
                consensus="No specialist analyses available",
                confidence=0.0,
                participant_count=0,
                agreement_level=0.0,
                evidence=feed_items
            )
        
        # Analyze specialist opinions
        consensus_analysis = self._analyze_specialist_consensus(specialist_analyses)
        
        # Generate consensus result
        return QuorumResult(
            consensus=consensus_analysis["consensus_opinion"],
            confidence=consensus_analysis["confidence"],
            participant_count=len(specialist_analyses),
            agreement_level=consensus_analysis["agreement_level"],
            dissenting_views=consensus_analysis["dissenting_views"],
            evidence=feed_items[:5]  # Limit evidence items
        )
    
    def _analyze_specialist_consensus(self, analyses: List[Dict]) -> Dict:
        """Analyze consensus among specialist analyses."""
        # Extract confidence scores and opinions
        confidences = [analysis.get("confidence", 0.0) for analysis in analyses]
        opinions = [analysis.get("opinion", "") for analysis in analyses]
        specialists = [analysis.get("specialist", "unknown") for analysis in analyses]
        
        # Calculate agreement metrics
        if len(confidences) > 1:
            confidence_stdev = statistics.stdev(confidences)
            agreement_level = max(0.0, 1.0 - confidence_stdev)  # Lower stdev = higher agreement
        else:
            agreement_level = 1.0
        
        avg_confidence = statistics.mean(confidences)
        
        # Identify consensus opinion
        high_confidence_analyses = [
            analysis for analysis in analyses 
            if analysis.get("confidence", 0.0) >= self.consensus_threshold
        ]
        
        if len(high_confidence_analyses) >= len(analyses) * 0.6:
            # Majority high confidence - build consensus opinion
            consensus_opinion = self._build_consensus_opinion(high_confidence_analyses)
            consensus_confidence = min(0.9, avg_confidence * agreement_level)
            dissenting_views = self._identify_dissenting_views(analyses, high_confidence_analyses)
        else:
            # No clear consensus
            consensus_opinion = "No clear consensus among specialists"
            consensus_confidence = max(0.3, avg_confidence * 0.5)
            dissenting_views = [
                f"{analysis.get('specialist', 'Unknown')}: {analysis.get('opinion', 'No opinion')}"
                for analysis in analyses
            ]
        
        return {
            "consensus_opinion": consensus_opinion,
            "confidence": consensus_confidence,
            "agreement_level": agreement_level,
            "dissenting_views": dissenting_views,
            "specialist_count": len(analyses),
            "high_confidence_count": len(high_confidence_analyses)
        }
    
    def _build_consensus_opinion(self, high_confidence_analyses: List[Dict]) -> str:
        """Build consensus opinion from high-confidence analyses."""
        # Extract key themes from opinions
        all_opinions = " ".join([
            analysis.get("opinion", "") for analysis in high_confidence_analyses
        ])
        
        # Simple keyword analysis for consensus themes
        positive_keywords = ["good", "strong", "high", "compliant", "safe", "quality"]
        negative_keywords = ["poor", "low", "risk", "concern", "issue", "problem"]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in all_opinions.lower())
        negative_count = sum(1 for keyword in negative_keywords if keyword in all_opinions.lower())
        
        if positive_count > negative_count * 1.5:
            return "Specialist consensus indicates positive assessment with good quality and compliance"
        elif negative_count > positive_count * 1.5:
            return "Specialist consensus identifies concerns requiring attention and review"
        else:
            return "Mixed specialist opinions with both positive aspects and areas for improvement"
    
    def _identify_dissenting_views(self, all_analyses: List[Dict], consensus_analyses: List[Dict]) -> List[str]:
        """Identify dissenting views from non-consensus analyses."""
        consensus_specialists = {analysis.get("specialist") for analysis in consensus_analyses}
        
        dissenting_views = []
        for analysis in all_analyses:
            specialist = analysis.get("specialist", "unknown")
            if specialist not in consensus_specialists:
                opinion = analysis.get("opinion", "No opinion provided")
                dissenting_views.append(f"{specialist}: {opinion}")
        
        return dissenting_views
    
    def _specialist_analysis(self, feed_items: List[QuorumFeedItem], context: Optional[Dict]) -> Dict:
        """Quorum specialist's own analysis (meta-analysis)."""
        # This specialist doesn't do direct content analysis
        # It facilitates consensus among other specialists
        
        metrics = self.calculate_consensus_metrics(feed_items)
        
        return {
            "opinion": f"Feed contains {len(feed_items)} items with {metrics['agreement']:.1%} score agreement",
            "confidence": 0.8,
            "reasoning": "Meta-analysis of feed structure and consistency",
            "metrics": metrics,
            "evidence": [item.memory_id for item in feed_items[:3]]
        }
    
    def is_applicable(self, context: Optional[Dict] = None) -> bool:
        """Always applicable for consensus facilitation."""
        return True