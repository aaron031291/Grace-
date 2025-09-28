"""Base specialist class and common utilities."""
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import statistics
from datetime import datetime

from ....contracts.quorum_feed import QuorumFeedItem


class BaseSpecialist:
    """Base implementation for intelligence specialists."""
    
    def __init__(self, name: str, expertise_domain: str, confidence_threshold: float = 0.5):
        self.name = name
        self.expertise_domain = expertise_domain
        self.confidence_threshold = confidence_threshold
        self.analysis_count = 0
        self.last_analysis = None
    
    def analyze(self, feed_items: List[QuorumFeedItem], context: Optional[Dict] = None) -> Dict:
        """Base analysis with common preprocessing."""
        self.analysis_count += 1
        self.last_analysis = datetime.utcnow()
        
        if not feed_items:
            return self._empty_analysis()
        
        # Common preprocessing
        preprocessed = self._preprocess_feed(feed_items)
        
        # Delegate to specialist implementation
        result = self._specialist_analysis(preprocessed, context)
        
        # Add metadata
        result["specialist"] = self.name
        result["analysis_timestamp"] = self.last_analysis.isoformat()
        result["feed_size"] = len(feed_items)
        
        return result
    
    def _preprocess_feed(self, feed_items: List[QuorumFeedItem]) -> List[QuorumFeedItem]:
        """Common preprocessing of feed items."""
        # Filter by relevance threshold
        filtered = [
            item for item in feed_items 
            if item.relevance_score >= 0.3
        ]
        
        # Sort by combined relevance and trust scores
        filtered.sort(
            key=lambda x: (x.relevance_score + x.trust_score) / 2,
            reverse=True
        )
        
        return filtered
    
    def _empty_analysis(self) -> Dict:
        """Default response for empty feed."""
        return {
            "opinion": "No data available for analysis",
            "confidence": 0.0,
            "reasoning": "Empty or insufficient feed data",
            "evidence": []
        }
    
    @abstractmethod
    def _specialist_analysis(self, feed_items: List[QuorumFeedItem], context: Optional[Dict]) -> Dict:
        """Specialist-specific analysis implementation."""
        pass
    
    def get_confidence(self, analysis_result: Dict) -> float:
        """Extract confidence from analysis result."""
        return analysis_result.get("confidence", 0.0)
    
    def is_applicable(self, context: Optional[Dict] = None) -> bool:
        """Check if specialist should participate."""
        if not context:
            return True
        
        # Check if context matches expertise domain
        relevant_domains = context.get("domains", [])
        if relevant_domains:
            return self.expertise_domain in relevant_domains
        
        return True
    
    def calculate_consensus_metrics(self, feed_items: List[QuorumFeedItem]) -> Dict:
        """Calculate common consensus metrics from feed."""
        if not feed_items:
            return {"agreement": 0.0, "diversity": 0.0}
        
        # Calculate agreement (similarity in scores)
        relevance_scores = [item.relevance_score for item in feed_items]
        trust_scores = [item.trust_score for item in feed_items]
        
        relevance_stdev = statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0
        trust_stdev = statistics.stdev(trust_scores) if len(trust_scores) > 1 else 0
        
        # Agreement is inverse of standard deviation (normalized)
        agreement = 1.0 - min(1.0, (relevance_stdev + trust_stdev) / 2)
        
        # Diversity based on unique content patterns
        unique_words = set()
        for item in feed_items:
            words = item.content.lower().split()
            unique_words.update(words)
        
        total_words = sum(len(item.content.split()) for item in feed_items)
        diversity = len(unique_words) / max(1, total_words)
        
        return {
            "agreement": agreement,
            "diversity": min(1.0, diversity * 2),  # Scale diversity
            "relevance_avg": statistics.mean(relevance_scores),
            "trust_avg": statistics.mean(trust_scores)
        }
    
    def get_stats(self) -> Dict:
        """Get specialist statistics."""
        return {
            "name": self.name,
            "domain": self.expertise_domain,
            "analysis_count": self.analysis_count,
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
            "confidence_threshold": self.confidence_threshold
        }