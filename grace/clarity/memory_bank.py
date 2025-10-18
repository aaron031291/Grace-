"""
Loop Memory Bank - Memory scoring and trust-based filtering (Class 5)
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryFragment:
    """Individual memory fragment with metadata"""
    fragment_id: str
    content: Any
    source: str
    created_at: datetime
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    confidence: float = 0.5
    relevance: float = 0.5
    trust_score: float = 0.5
    consensus_votes: int = 0
    contradiction_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class LoopMemoryBank:
    """
    Memory scoring and trust-based filtering system
    
    Rates memory fragments based on:
    - Source credibility (trust scores)
    - Recency (temporal decay)
    - Consensus (multiple sources agreeing)
    - Access patterns (frequently accessed = more relevant)
    - Contradiction detection
    """
    
    def __init__(
        self,
        trust_manager=None,
        min_confidence: float = 0.6,
        min_relevance: float = 0.5,
        recency_weight: float = 0.3,
        trust_weight: float = 0.4,
        consensus_weight: float = 0.3
    ):
        """
        Initialize memory bank
        
        Args:
            trust_manager: TrustScoreManager for source credibility
            min_confidence: Minimum confidence threshold
            min_relevance: Minimum relevance threshold
            recency_weight: Weight for recency scoring
            trust_weight: Weight for trust scoring
            consensus_weight: Weight for consensus scoring
        """
        self.trust_manager = trust_manager
        self.min_confidence = min_confidence
        self.min_relevance = min_relevance
        self.recency_weight = recency_weight
        self.trust_weight = trust_weight
        self.consensus_weight = consensus_weight
        
        self.fragments: Dict[str, MemoryFragment] = {}
        self.source_credibility: Dict[str, float] = {}
        
        logger.info("LoopMemoryBank initialized")
    
    def add_fragment(
        self,
        fragment_id: str,
        content: Any,
        source: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryFragment:
        """
        Add a memory fragment to the bank
        
        Args:
            fragment_id: Unique identifier
            content: Memory content
            source: Source of the memory
            confidence: Initial confidence (auto-calculated if None)
            metadata: Additional metadata
            
        Returns:
            Created memory fragment
        """
        # Calculate initial confidence from source credibility
        if confidence is None:
            confidence = self._get_source_credibility(source)
        
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            content=content,
            source=source,
            created_at=datetime.now(timezone.utc),
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.fragments[fragment_id] = fragment
        
        logger.debug(f"Added memory fragment: {fragment_id} from {source}")
        return fragment
    
    def score_fragment(
        self,
        fragment: MemoryFragment,
        query_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Score a memory fragment based on multiple factors
        
        Returns composite score (0-1)
        """
        # Recency score (exponential decay)
        recency_score = self._calculate_recency_score(fragment)
        
        # Trust score from source
        trust_score = self._calculate_trust_score(fragment)
        
        # Consensus score
        consensus_score = self._calculate_consensus_score(fragment)
        
        # Access pattern score
        access_score = self._calculate_access_score(fragment)
        
        # Relevance score (if query context provided)
        relevance_score = fragment.relevance
        if query_context:
            relevance_score = self._calculate_relevance_score(fragment, query_context)
            fragment.relevance = relevance_score
        
        # Weighted composite score
        composite_score = (
            self.recency_weight * recency_score +
            self.trust_weight * trust_score +
            self.consensus_weight * consensus_score +
            0.1 * access_score +  # Minor weight for access patterns
            0.2 * relevance_score  # Relevance boost
        )
        
        # Normalize to 0-1
        composite_score = min(1.0, composite_score)
        
        # Update fragment scores
        fragment.trust_score = composite_score
        
        return composite_score
    
    def retrieve_memories(
        self,
        query_context: Dict[str, Any],
        max_results: int = 10,
        min_score: Optional[float] = None
    ) -> List[Tuple[MemoryFragment, float]]:
        """
        Retrieve and filter memories based on scoring
        
        Args:
            query_context: Context for relevance scoring
            max_results: Maximum number of results
            min_score: Minimum score threshold
            
        Returns:
            List of (fragment, score) tuples sorted by score
        """
        if min_score is None:
            min_score = min(self.min_confidence, self.min_relevance)
        
        scored_fragments = []
        
        for fragment in self.fragments.values():
            # Score the fragment
            score = self.score_fragment(fragment, query_context)
            
            # Update access patterns
            fragment.accessed_count += 1
            fragment.last_accessed = datetime.now(timezone.utc)
            
            # Apply filters
            if score < min_score:
                continue
            
            if fragment.confidence < self.min_confidence:
                continue
            
            if fragment.relevance < self.min_relevance:
                continue
            
            scored_fragments.append((fragment, score))
        
        # Sort by score descending
        scored_fragments.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Retrieved {len(scored_fragments)} memories "
            f"(filtered from {len(self.fragments)})"
        )
        
        return scored_fragments[:max_results]
    
    def update_consensus(self, fragment_id: str, agreement: bool):
        """Update consensus voting for a fragment"""
        if fragment_id not in self.fragments:
            return
        
        fragment = self.fragments[fragment_id]
        
        if agreement:
            fragment.consensus_votes += 1
        else:
            fragment.contradiction_count += 1
        
        # Recalculate confidence based on consensus
        total_votes = fragment.consensus_votes + fragment.contradiction_count
        if total_votes > 0:
            consensus_ratio = fragment.consensus_votes / total_votes
            # Adjust confidence
            fragment.confidence = (fragment.confidence + consensus_ratio) / 2
    
    def _calculate_recency_score(self, fragment: MemoryFragment) -> float:
        """Calculate recency score with exponential decay"""
        age = (datetime.now(timezone.utc) - fragment.created_at).total_seconds()
        
        # Half-life of 30 days (2,592,000 seconds)
        half_life = 30 * 24 * 3600
        decay_factor = 0.5 ** (age / half_life)
        
        return decay_factor
    
    def _calculate_trust_score(self, fragment: MemoryFragment) -> float:
        """Calculate trust score from source credibility"""
        source_cred = self._get_source_credibility(fragment.source)
        
        # Combine with fragment's own confidence
        return (source_cred + fragment.confidence) / 2
    
    def _calculate_consensus_score(self, fragment: MemoryFragment) -> float:
        """Calculate consensus score"""
        total_votes = fragment.consensus_votes + fragment.contradiction_count
        
        if total_votes == 0:
            return 0.5  # Neutral
        
        consensus_ratio = fragment.consensus_votes / total_votes
        
        # Boost if many votes
        vote_boost = min(1.0, total_votes / 10)
        
        return consensus_ratio * (0.7 + 0.3 * vote_boost)
    
    def _calculate_access_score(self, fragment: MemoryFragment) -> float:
        """Calculate access pattern score"""
        # Frequently accessed = more relevant
        access_score = min(1.0, fragment.accessed_count / 100)
        
        # Boost if recently accessed
        if fragment.last_accessed:
            age = (datetime.now(timezone.utc) - fragment.last_accessed).total_seconds()
            if age < 3600:  # Last hour
                access_score *= 1.2
        
        return min(1.0, access_score)
    
    def _calculate_relevance_score(
        self,
        fragment: MemoryFragment,
        query_context: Dict[str, Any]
    ) -> float:
        """Calculate relevance to query context"""
        # Simple keyword matching (in production, use embeddings)
        query_text = str(query_context).lower()
        fragment_text = str(fragment.content).lower()
        
        # Count matching keywords
        query_words = set(query_text.split())
        fragment_words = set(fragment_text.split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words & fragment_words)
        relevance = overlap / len(query_words)
        
        return min(1.0, relevance)
    
    def _get_source_credibility(self, source: str) -> float:
        """Get credibility score for a source"""
        if source in self.source_credibility:
            return self.source_credibility[source]
        
        # Check trust manager
        if self.trust_manager:
            try:
                trust_data = self.trust_manager.get_trust_score(source)
                if trust_data:
                    cred = trust_data.get('current_score', 0.5)
                    self.source_credibility[source] = cred
                    return cred
            except:
                pass
        
        # Default credibility
        self.source_credibility[source] = 0.7
        return 0.7
    
    def prune_low_quality(self, threshold: float = 0.3):
        """Remove low-quality fragments"""
        to_remove = []
        
        for frag_id, fragment in self.fragments.items():
            score = self.score_fragment(fragment)
            
            if score < threshold:
                to_remove.append(frag_id)
        
        for frag_id in to_remove:
            del self.fragments[frag_id]
        
        logger.info(f"Pruned {len(to_remove)} low-quality fragments")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        if not self.fragments:
            return {"total_fragments": 0}
        
        fragments = list(self.fragments.values())
        
        return {
            "total_fragments": len(fragments),
            "avg_confidence": np.mean([f.confidence for f in fragments]),
            "avg_relevance": np.mean([f.relevance for f in fragments]),
            "avg_trust_score": np.mean([f.trust_score for f in fragments]),
            "total_consensus_votes": sum(f.consensus_votes for f in fragments),
            "total_contradictions": sum(f.contradiction_count for f in fragments),
            "unique_sources": len(set(f.source for f in fragments))
        }
