"""Quorum bridge - interfaces with intelligence kernel for consensus."""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class QuorumBridge:
    """Bridge to intelligence kernel for quorum consensus operations."""

    def __init__(self, intelligence_kernel=None):
        self.intelligence_kernel = intelligence_kernel

    def consensus(
        self, feed_ids: List[str], context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get consensus from intelligence kernel quorum process.

        Returns:
            Dict with consensus results compatible with both contract systems
        """
        if not feed_ids:
            return {
                "consensus_reached": False,
                "agreement_level": 0.0,
                "consensus_view": "No feed data available",
                "confidence": 0.0,
                "participating_memories": [],
                "dissenting_views": [],
            }

        # If intelligence kernel is available, delegate to it
        if self.intelligence_kernel:
            try:
                result = self.intelligence_kernel.consensus(feed_ids)
                return self._convert_to_governance_consensus(result)
            except Exception as e:
                logger.warning(
                    f"Intelligence kernel consensus failed: {e}, falling back to mock"
                )
                pass

        # Mock consensus for development
        return self._mock_consensus(feed_ids, context)

    def _mock_consensus(
        self, feed_ids: List[str], context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Mock consensus implementation for development."""
        # Simple mock logic based on feed size
        agreement_level = min(
            1.0, len(feed_ids) / 5.0
        )  # Higher agreement with more feed items
        consensus_reached = agreement_level >= 0.6

        if consensus_reached:
            consensus_view = (
                f"Consensus reached based on {len(feed_ids)} memory entries"
            )
            confidence = agreement_level * 0.9  # Slightly lower than agreement level
            dissenting_views = []
        else:
            consensus_view = (
                f"Insufficient consensus from {len(feed_ids)} memory entries"
            )
            confidence = agreement_level * 0.5
            dissenting_views = [
                "Insufficient historical data",
                "Low agreement threshold",
            ]

        return {
            "consensus_reached": consensus_reached,
            "agreement_level": agreement_level,
            "participating_memories": feed_ids,
            "consensus_view": consensus_view,
            "dissenting_views": dissenting_views,
            "confidence": confidence,
        }

    def _convert_to_governance_consensus(self, intelligence_result) -> Dict[str, Any]:
        """Convert intelligence kernel result to governance consensus."""
        # This would convert the intelligence kernel's QuorumResult
        # to our governance consensus format
        return {
            "consensus_reached": getattr(
                intelligence_result, "consensus_reached", False
            ),
            "agreement_level": getattr(intelligence_result, "agreement_level", 0.0),
            "participating_memories": getattr(intelligence_result, "evidence", []),
            "consensus_view": getattr(intelligence_result, "consensus", ""),
            "dissenting_views": getattr(intelligence_result, "dissenting_views", []),
            "confidence": getattr(intelligence_result, "confidence", 0.0),
        }

    def set_intelligence_kernel(self, intelligence_kernel):
        """Set the intelligence kernel for actual consensus operations."""
        self.intelligence_kernel = intelligence_kernel
