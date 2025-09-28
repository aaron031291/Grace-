"""Document quality specialist - evaluates document quality and reliability."""
from typing import Dict, List, Optional
import re

from ...contracts.quorum_feed import QuorumFeedItem
from .specialist import BaseSpecialist


class DocQualitySpecialist(BaseSpecialist):
    """Specialist for evaluating document quality and reliability."""
    
    def __init__(self):
        super().__init__(
            name="doc_quality_specialist",
            expertise_domain="document_analysis",
            confidence_threshold=0.6
        )
        
        # Quality indicators
        self.quality_patterns = {
            "structure": [
                r"\n\s*\n",  # Proper paragraph spacing
                r"^#+ ",     # Headers
                r"\d+\.",    # Numbered lists
                r"[-*+] ",   # Bullet points
            ],
            "language_quality": [
                r"\b[A-Z][a-z]+\b",  # Proper capitalization
                r"[.!?]",            # Punctuation
                r"\b(however|therefore|furthermore|moreover|consequently)\b",  # Transitions
            ],
            "content_depth": [
                r"\b(because|since|due to|reason|cause|effect|result)\b",  # Causal language
                r"\b(example|specifically|particularly|namely)\b",         # Elaboration
                r"\b(data|evidence|study|research|analysis)\b",            # Evidence references
            ]
        }
    
    def _specialist_analysis(self, feed_items: List[QuorumFeedItem], context: Optional[Dict]) -> Dict:
        """Analyze document quality across feed items."""
        if not feed_items:
            return self._empty_analysis()
        
        quality_scores = []
        detailed_analysis = []
        
        for item in feed_items:
            quality_score, analysis = self._analyze_document_quality(item)
            quality_scores.append(quality_score)
            detailed_analysis.append({
                "memory_id": item.memory_id,
                "quality_score": quality_score,
                "analysis": analysis
            })
        
        # Aggregate quality assessment
        avg_quality = sum(quality_scores) / len(quality_scores)
        high_quality_count = sum(1 for score in quality_scores if score >= 0.7)
        
        # Generate opinion based on quality distribution
        if avg_quality >= 0.8:
            opinion = "High quality documentation with strong structure and depth"
            confidence = 0.9
        elif avg_quality >= 0.6:
            opinion = "Moderate quality documentation with some structural elements"
            confidence = 0.7
        elif avg_quality >= 0.4:
            opinion = "Basic quality documentation, may lack depth or structure"
            confidence = 0.5
        else:
            opinion = "Low quality documentation with significant deficiencies"
            confidence = 0.3
        
        reasoning = f"Average quality score: {avg_quality:.2f}. " \
                   f"{high_quality_count}/{len(feed_items)} items meet high quality threshold."
        
        return {
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "average_quality": avg_quality,
                "high_quality_count": high_quality_count,
                "total_documents": len(feed_items)
            },
            "detailed_analysis": detailed_analysis,
            "evidence": [item.memory_id for item in feed_items if quality_scores[feed_items.index(item)] >= 0.7]
        }
    
    def _analyze_document_quality(self, item: QuorumFeedItem) -> tuple:
        """Analyze quality of a single document."""
        content = item.content
        
        # Structural analysis
        structure_score = self._score_patterns(content, self.quality_patterns["structure"])
        
        # Language quality analysis  
        language_score = self._score_patterns(content, self.quality_patterns["language_quality"])
        
        # Content depth analysis
        depth_score = self._score_patterns(content, self.quality_patterns["content_depth"])
        
        # Length and completeness
        length_score = min(1.0, len(content) / 500)  # Normalize around 500 characters
        
        # Overall quality score (weighted average)
        quality_score = (
            structure_score * 0.3 +
            language_score * 0.3 + 
            depth_score * 0.25 +
            length_score * 0.15
        )
        
        analysis = {
            "structure_score": structure_score,
            "language_score": language_score,
            "depth_score": depth_score,
            "length_score": length_score,
            "content_length": len(content)
        }
        
        return quality_score, analysis
    
    def _score_patterns(self, content: str, patterns: List[str]) -> float:
        """Score content based on pattern matches."""
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                matches += 1
        
        return matches / max(1, total_patterns)
    
    def is_applicable(self, context: Optional[Dict] = None) -> bool:
        """Always applicable for document quality assessment."""
        return True