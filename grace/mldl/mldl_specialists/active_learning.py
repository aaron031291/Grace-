"""
Active Learning & Human-in-the-Loop (HITL)

Implements continuous improvement through:
- Uncertainty-based sample selection
- Human review queue management
- Priority scoring for labeling
- Feedback loop to retrain models
- Diversity-aware sampling
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict
import logging

from .uncertainty_ood import (
    OODDetectionResult,
    uncertainty_sampling_priority
)

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of human review"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_INFO = "needs_more_info"
    ESCALATED = "escalated"


class SamplingStrategy(Enum):
    """Active learning sampling strategies"""
    UNCERTAINTY = "uncertainty"  # Lowest confidence
    MARGIN = "margin"  # Smallest margin between top 2 classes
    ENTROPY = "entropy"  # Highest entropy
    COMMITTEE = "committee"  # Ensemble disagreement
    DIVERSITY = "diversity"  # Maximize coverage
    HYBRID = "hybrid"  # Weighted combination


@dataclass
class ReviewQueueItem:
    """Item in human review queue"""
    item_id: str
    trace_id: str
    
    # Input data
    input_data: Dict[str, Any]
    input_hash: str
    
    # Model prediction
    model_id: str
    model_version: str
    prediction: Any
    confidence: float
    
    # Uncertainty & OOD
    ood_result: Optional[OODDetectionResult] = None
    uncertainty_score: float = 0.0
    
    # Priority
    priority_score: float = 0.0
    diversity_score: float = 0.0
    
    # Review
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_to: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewer_label: Optional[Any] = None
    reviewer_feedback: Optional[str] = None
    
    # Metadata
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class LabeledSample:
    """Labeled sample for retraining"""
    sample_id: str
    input_data: Dict[str, Any]
    ground_truth_label: Any
    
    # Provenance
    labeled_by: str
    labeled_at: datetime
    labeling_confidence: float  # Human confidence
    
    # Original prediction (for comparison)
    original_prediction: Optional[Any] = None
    original_confidence: Optional[float] = None
    model_version: Optional[str] = None
    
    # Sample characteristics
    was_ood: bool = False
    was_uncertain: bool = False
    priority_score: float = 0.0


class ReviewQueue:
    """
    Manages human review queue with priority ordering.
    
    Samples are prioritized by:
    1. Uncertainty score
    2. OOD score
    3. Diversity
    4. Business impact (from context)
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        auto_approval_threshold: float = 0.95
    ):
        self.max_queue_size = max_queue_size
        self.auto_approval_threshold = auto_approval_threshold
        
        self.queue: List[ReviewQueueItem] = []
        self.reviewed_items: List[ReviewQueueItem] = []
        
        # Statistics
        self.stats = {
            'total_enqueued': 0,
            'total_reviewed': 0,
            'total_approved': 0,
            'total_rejected': 0,
            'avg_review_time_minutes': 0.0
        }
    
    def enqueue(
        self,
        item: ReviewQueueItem,
        force: bool = False
    ) -> bool:
        """
        Add item to review queue.
        
        Args:
            item: Item to review
            force: Force enqueue even if queue is full
        
        Returns:
            True if enqueued successfully
        """
        # Auto-approve high confidence predictions
        if not force and item.confidence >= self.auto_approval_threshold:
            item.status = ReviewStatus.APPROVED
            item.reviewer_feedback = "Auto-approved (high confidence)"
            self.reviewed_items.append(item)
            self.stats['total_approved'] += 1
            logger.info(f"Auto-approved {item.item_id} (confidence={item.confidence:.3f})")
            return True
        
        # Check queue capacity
        if len(self.queue) >= self.max_queue_size and not force:
            # Remove lowest priority item
            self.queue.sort(key=lambda x: x.priority_score)
            removed = self.queue.pop(0)
            logger.warning(f"Queue full, removed lowest priority item: {removed.item_id}")
        
        # Calculate priority if not set
        if item.priority_score == 0.0:
            item.priority_score = self._calculate_priority(item)
        
        self.queue.append(item)
        self.stats['total_enqueued'] += 1
        
        # Sort by priority (highest first)
        self.queue.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(
            f"Enqueued {item.item_id} for review "
            f"(priority={item.priority_score:.3f}, queue_size={len(self.queue)})"
        )
        return True
    
    def _calculate_priority(self, item: ReviewQueueItem) -> float:
        """Calculate item priority score"""
        ood_score = item.ood_result.ood_score if item.ood_result else 0.0
        
        return uncertainty_sampling_priority(
            confidence=item.confidence,
            ood_score=ood_score,
            diversity_score=item.diversity_score
        )
    
    def get_next_item(
        self,
        reviewer_id: Optional[str] = None
    ) -> Optional[ReviewQueueItem]:
        """
        Get next item for review.
        
        Args:
            reviewer_id: Assign to this reviewer
        
        Returns:
            Next item or None if queue empty
        """
        if not self.queue:
            return None
        
        item = self.queue.pop(0)
        item.status = ReviewStatus.IN_REVIEW
        item.assigned_to = reviewer_id
        item.updated_at = datetime.now()
        
        return item
    
    def submit_review(
        self,
        item_id: str,
        status: ReviewStatus,
        label: Optional[Any] = None,
        feedback: Optional[str] = None,
        reviewer_id: Optional[str] = None
    ) -> bool:
        """
        Submit review result.
        
        Args:
            item_id: Item being reviewed
            status: Review decision
            label: Ground truth label (if applicable)
            feedback: Reviewer comments
            reviewer_id: Reviewer identifier
        
        Returns:
            True if successful
        """
        # Find item in queue
        item = None
        for i, queue_item in enumerate(self.queue):
            if queue_item.item_id == item_id:
                item = self.queue.pop(i)
                break
        
        if item is None:
            logger.error(f"Item not found in queue: {item_id}")
            return False
        
        # Update item
        item.status = status
        item.reviewer_label = label
        item.reviewer_feedback = feedback
        item.reviewed_at = datetime.now()
        
        if reviewer_id:
            item.assigned_to = reviewer_id
        
        # Move to reviewed items
        self.reviewed_items.append(item)
        
        # Update statistics
        self.stats['total_reviewed'] += 1
        if status == ReviewStatus.APPROVED:
            self.stats['total_approved'] += 1
        elif status == ReviewStatus.REJECTED:
            self.stats['total_rejected'] += 1
        
        # Update avg review time
        if item.reviewed_at and item.created_at:
            review_time = (item.reviewed_at - item.created_at).total_seconds() / 60
            self.stats['avg_review_time_minutes'] = (
                (self.stats['avg_review_time_minutes'] * (self.stats['total_reviewed'] - 1) + review_time)
                / self.stats['total_reviewed']
            )
        
        logger.info(
            f"Review submitted for {item_id}: {status.value} "
            f"(label={label}, reviewer={reviewer_id})"
        )
        return True
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics"""
        status_counts = defaultdict(int)
        for item in self.queue:
            status_counts[item.status.value] += 1
        
        return {
            'queue_size': len(self.queue),
            'reviewed_count': len(self.reviewed_items),
            'status_breakdown': dict(status_counts),
            'statistics': self.stats,
            'top_priority_items': [
                {
                    'item_id': item.item_id,
                    'priority': item.priority_score,
                    'confidence': item.confidence,
                    'ood_score': item.ood_result.ood_score if item.ood_result else None
                }
                for item in self.queue[:5]  # Top 5
            ]
        }


class ActiveLearner:
    """
    Active learning system for continuous model improvement.
    
    Workflow:
    1. Select high-value samples for labeling
    2. Queue for human review
    3. Collect labeled samples
    4. Trigger retraining when threshold reached
    """
    
    def __init__(
        self,
        review_queue: ReviewQueue,
        retrain_threshold: int = 100,  # Retrain after N new labels
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID
    ):
        self.review_queue = review_queue
        self.retrain_threshold = retrain_threshold
        self.sampling_strategy = sampling_strategy
        
        self.labeled_samples: List[LabeledSample] = []
        self.retrain_queue: List[LabeledSample] = []
        
        # Diversity tracking (embeddings of sampled data)
        self.sampled_embeddings: List[np.ndarray] = []
    
    def select_samples_for_labeling(
        self,
        candidates: List[Dict[str, Any]],
        n_samples: int = 10,
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select samples for human labeling using active learning.
        
        Args:
            candidates: List of candidate samples with predictions
            n_samples: Number to select
            embeddings: Sample embeddings for diversity calculation
        
        Returns:
            Selected samples
        """
        if len(candidates) <= n_samples:
            return candidates
        
        if self.sampling_strategy == SamplingStrategy.UNCERTAINTY:
            return self._select_by_uncertainty(candidates, n_samples)
        elif self.sampling_strategy == SamplingStrategy.DIVERSITY:
            return self._select_by_diversity(candidates, n_samples, embeddings)
        elif self.sampling_strategy == SamplingStrategy.HYBRID:
            return self._select_hybrid(candidates, n_samples, embeddings)
        else:
            # Default: uncertainty
            return self._select_by_uncertainty(candidates, n_samples)
    
    def _select_by_uncertainty(
        self,
        candidates: List[Dict[str, Any]],
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """Select samples with highest uncertainty"""
        # Sort by confidence (ascending - lowest confidence first)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('confidence', 1.0)
        )
        return sorted_candidates[:n_samples]
    
    def _select_by_diversity(
        self,
        candidates: List[Dict[str, Any]],
        n_samples: int,
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """Select diverse samples using k-center greedy algorithm"""
        if embeddings is None or len(embeddings) != len(candidates):
            # Fallback to uncertainty
            return self._select_by_uncertainty(candidates, n_samples)
        
        selected_indices = []
        embeddings_array = np.array(embeddings)
        
        # Start with most uncertain sample
        uncertainties = [1.0 - c.get('confidence', 1.0) for c in candidates]
        selected_indices.append(int(np.argmax(uncertainties)))
        
        # Greedily select samples farthest from selected set
        for _ in range(n_samples - 1):
            max_min_distance = -1
            best_idx = -1
            
            for idx in range(len(candidates)):
                if idx in selected_indices:
                    continue
                
                # Minimum distance to any selected sample
                distances = [
                    np.linalg.norm(embeddings_array[idx] - embeddings_array[sel_idx])
                    for sel_idx in selected_indices
                ]
                min_distance = min(distances)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
        
        return [candidates[idx] for idx in selected_indices]
    
    def _select_hybrid(
        self,
        candidates: List[Dict[str, Any]],
        n_samples: int,
        embeddings: Optional[List[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid selection: 70% uncertainty + 30% diversity"""
        n_uncertain = int(n_samples * 0.7)
        n_diverse = n_samples - n_uncertain
        
        # Select uncertain samples
        uncertain_samples = self._select_by_uncertainty(candidates, n_uncertain)
        uncertain_indices = [candidates.index(s) for s in uncertain_samples]
        
        # Select diverse samples from remaining
        remaining_candidates = [
            c for i, c in enumerate(candidates)
            if i not in uncertain_indices
        ]
        remaining_embeddings = (
            [embeddings[i] for i in range(len(candidates)) if i not in uncertain_indices]
            if embeddings else None
        )
        
        diverse_samples = self._select_by_diversity(
            remaining_candidates,
            n_diverse,
            remaining_embeddings
        )
        
        return uncertain_samples + diverse_samples
    
    def add_labeled_sample(
        self,
        sample: LabeledSample
    ):
        """Add newly labeled sample to dataset"""
        self.labeled_samples.append(sample)
        self.retrain_queue.append(sample)
        
        logger.info(
            f"Added labeled sample: {sample.sample_id} "
            f"(retrain_queue={len(self.retrain_queue)})"
        )
        
        # Check if retrain threshold reached
        if len(self.retrain_queue) >= self.retrain_threshold:
            logger.info(
                f"Retrain threshold reached ({self.retrain_threshold}), "
                f"triggering retraining workflow"
            )
            # In production, this would trigger TriggerMesh workflow
    
    def get_retrain_batch(
        self,
        clear_queue: bool = True
    ) -> List[LabeledSample]:
        """
        Get batch of labeled samples for retraining.
        
        Args:
            clear_queue: Clear retrain queue after retrieval
        
        Returns:
            List of labeled samples
        """
        batch = list(self.retrain_queue)
        
        if clear_queue:
            self.retrain_queue.clear()
        
        logger.info(f"Retrieved retrain batch: {len(batch)} samples")
        return batch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics"""
        total_labels = len(self.labeled_samples)
        
        # Count samples by characteristics
        ood_count = sum(1 for s in self.labeled_samples if s.was_ood)
        uncertain_count = sum(1 for s in self.labeled_samples if s.was_uncertain)
        
        # Calculate agreement rate (model vs human)
        agreement_count = sum(
            1 for s in self.labeled_samples
            if s.original_prediction == s.ground_truth_label
        )
        agreement_rate = agreement_count / total_labels if total_labels > 0 else 0.0
        
        return {
            'total_labeled_samples': total_labels,
            'retrain_queue_size': len(self.retrain_queue),
            'retrain_threshold': self.retrain_threshold,
            'ood_samples': ood_count,
            'uncertain_samples': uncertain_count,
            'model_human_agreement_rate': agreement_rate,
            'sampling_strategy': self.sampling_strategy.value,
            'review_queue_status': self.review_queue.get_queue_status()
        }
