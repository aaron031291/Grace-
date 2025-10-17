"""
Class 5: Memory Scoring Ambiguity - LoopMemoryBank with scoring
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory entries"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    META = "meta"


@dataclass
class MemoryEntry:
    """Individual memory entry"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    clarity_score: float = 1.0
    relevance_score: float = 1.0
    ambiguity_score: float = 0.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoopMemoryBank:
    """
    Memory bank with ambiguity scoring
    Tracks, scores, and manages memories across loop iterations
    """
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
        self.access_history: List[Tuple[str, datetime]] = []
        self.ambiguity_threshold = 0.7
        self.decay_rate = 0.95
        logger.info("LoopMemoryBank initialized")
    
    def store(
        self,
        memory_id: str,
        memory_type: MemoryType,
        content: Dict[str, Any],
        source: str = "loop",
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """Store a new memory with initial scoring"""
        clarity_score = self._calculate_clarity(content)
        ambiguity_score = self._calculate_ambiguity(content)
        
        memory = MemoryEntry(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            clarity_score=clarity_score,
            ambiguity_score=ambiguity_score,
            source=source,
            metadata=metadata or {}
        )
        
        self.memories[memory_id] = memory
        logger.debug(f"Stored memory: {memory_id} (clarity: {clarity_score:.2f}, ambiguity: {ambiguity_score:.2f})")
        
        return memory
    
    def score(
        self,
        memory_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Score a memory for clarity, relevance, and ambiguity
        Core implementation of Class 5
        """
        if memory_id not in self.memories:
            logger.warning(f"Memory not found for scoring: {memory_id}")
            return {'clarity': 0.0, 'relevance': 0.0, 'ambiguity': 1.0, 'composite': 0.0}
        
        memory = self.memories[memory_id]
        
        # Update access count
        memory.access_count += 1
        self.access_history.append((memory_id, datetime.now()))
        
        # Calculate scores
        clarity_score = self._calculate_clarity(memory.content)
        relevance_score = self._calculate_relevance(memory, context or {})
        ambiguity_score = self._calculate_ambiguity(memory.content)
        
        # Time decay factor
        age_seconds = (datetime.now() - memory.timestamp).total_seconds()
        decay_factor = self.decay_rate ** (age_seconds / 3600)  # Hourly decay
        
        # Composite score
        composite_score = (
            clarity_score * 0.4 +
            relevance_score * 0.3 +
            (1 - ambiguity_score) * 0.2 +
            decay_factor * 0.1
        )
        
        # Update memory scores
        memory.clarity_score = clarity_score
        memory.relevance_score = relevance_score
        memory.ambiguity_score = ambiguity_score
        
        scores = {
            'clarity': clarity_score,
            'relevance': relevance_score,
            'ambiguity': ambiguity_score,
            'decay': decay_factor,
            'composite': composite_score
        }
        
        logger.debug(f"Scored memory {memory_id}: composite={composite_score:.3f}")
        
        return scores
    
    def _calculate_clarity(self, content: Dict[str, Any]) -> float:
        """Calculate clarity score based on content completeness and structure"""
        score = 0.5  # Base score
        
        # Completeness check
        if 'description' in content and len(str(content['description'])) > 10:
            score += 0.2
        
        if 'confidence' in content:
            score += 0.1
        
        # Structure check
        if isinstance(content, dict) and len(content) >= 3:
            score += 0.1
        
        # Specificity check (presence of specific data types)
        if any(isinstance(v, (int, float)) for v in content.values()):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_relevance(self, memory: MemoryEntry, context: Dict[str, Any]) -> float:
        """Calculate relevance score based on context"""
        if not context:
            return 0.5
        
        score = 0.0
        
        # Recency bonus
        age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
        recency_score = max(0, 1.0 - (age_hours / 24))  # Decay over 24 hours
        score += recency_score * 0.3
        
        # Access frequency bonus
        if memory.access_count > 0:
            frequency_score = min(memory.access_count / 10, 1.0)
            score += frequency_score * 0.2
        
        # Context matching
        context_matches = sum(
            1 for key in context.keys()
            if key in memory.content or key in memory.metadata
        )
        if context:
            context_score = context_matches / len(context)
            score += context_score * 0.5
        
        return min(score, 1.0)
    
    def _calculate_ambiguity(self, content: Dict[str, Any]) -> float:
        """Calculate ambiguity score (higher = more ambiguous)"""
        ambiguity = 0.0
        
        # Check for vague language indicators
        vague_terms = {'maybe', 'possibly', 'unclear', 'unknown', 'ambiguous', 'uncertain'}
        content_str = str(content).lower()
        
        vague_count = sum(1 for term in vague_terms if term in content_str)
        ambiguity += min(vague_count * 0.2, 0.6)
        
        # Check for missing critical fields
        critical_fields = {'action', 'result', 'confidence', 'description'}
        missing_fields = critical_fields - set(content.keys())
        ambiguity += len(missing_fields) * 0.1
        
        # Check for conflicting information (simplified)
        if 'confidence' in content and content['confidence'] < 0.5:
            ambiguity += 0.2
        
        return min(ambiguity, 1.0)
    
    def get_high_ambiguity_memories(self, threshold: Optional[float] = None) -> List[MemoryEntry]:
        """Get memories with high ambiguity scores"""
        threshold = threshold or self.ambiguity_threshold
        
        return [
            memory for memory in self.memories.values()
            if memory.ambiguity_score >= threshold
        ]
    
    def consolidate_similar(self, similarity_threshold: float = 0.8) -> int:
        """Consolidate similar memories to reduce ambiguity"""
        consolidated = 0
        processed = set()
        
        memory_list = list(self.memories.values())
        
        for i, mem1 in enumerate(memory_list):
            if mem1.memory_id in processed:
                continue
            
            for mem2 in memory_list[i+1:]:
                if mem2.memory_id in processed:
                    continue
                
                similarity = self._calculate_similarity(mem1, mem2)
                
                if similarity >= similarity_threshold:
                    # Merge memories
                    self._merge_memories(mem1, mem2)
                    processed.add(mem2.memory_id)
                    del self.memories[mem2.memory_id]
                    consolidated += 1
        
        logger.info(f"Consolidated {consolidated} similar memories")
        return consolidated
    
    def _calculate_similarity(self, mem1: MemoryEntry, mem2: MemoryEntry) -> float:
        """Calculate similarity between two memories"""
        if mem1.memory_type != mem2.memory_type:
            return 0.0
        
        # Simple key overlap similarity
        keys1 = set(mem1.content.keys())
        keys2 = set(mem2.content.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        overlap = len(keys1 & keys2) / len(keys1 | keys2)
        
        return overlap
    
    def _merge_memories(self, target: MemoryEntry, source: MemoryEntry):
        """Merge source memory into target"""
        # Merge content
        target.content.update({
            k: v for k, v in source.content.items()
            if k not in target.content
        })
        
        # Update scores (weighted average)
        weight_target = target.access_count + 1
        weight_source = source.access_count + 1
        total_weight = weight_target + weight_source
        
        target.clarity_score = (
            target.clarity_score * weight_target +
            source.clarity_score * weight_source
        ) / total_weight
        
        target.ambiguity_score = (
            target.ambiguity_score * weight_target +
            source.ambiguity_score * weight_source
        ) / total_weight
        
        target.access_count += source.access_count
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        if not self.memories:
            return {'total_memories': 0}
        
        scores = [self.score(mid) for mid in self.memories.keys()]
        
        return {
            'total_memories': len(self.memories),
            'high_ambiguity_count': len(self.get_high_ambiguity_memories()),
            'avg_clarity': np.mean([s['clarity'] for s in scores]),
            'avg_ambiguity': np.mean([s['ambiguity'] for s in scores]),
            'avg_composite': np.mean([s['composite'] for s in scores]),
            'memory_types': {
                mt.value: sum(1 for m in self.memories.values() if m.memory_type == mt)
                for mt in MemoryType
            }
        }
