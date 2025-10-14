"""Intelligence kernel - main orchestration for specialist-based consensus."""

from typing import Dict, List, Optional

from ...contracts.quorum_feed import QuorumResult, QuorumFeedItem
from .specialists.registry import SpecialistRegistry
from .specialists.doc_quality import DocQualitySpecialist
from .specialists.policy_fit import PolicyFitSpecialist
from .specialists.risk_checker import RiskCheckerSpecialist
from .specialists.quorum import QuorumSpecialist


class IntelligenceKernel:
    """Main intelligence kernel orchestrating specialist-based consensus."""

    def __init__(self, mtl_kernel=None):
        self.mtl_kernel = mtl_kernel
        self.registry = SpecialistRegistry()
        self.quorum_specialist = QuorumSpecialist()

        # Register default specialists
        self._register_default_specialists()

    def _register_default_specialists(self):
        """Register the default set of specialists."""
        specialists = [
            DocQualitySpecialist(),
            PolicyFitSpecialist(),
            RiskCheckerSpecialist(),
            self.quorum_specialist,
        ]

        for specialist in specialists:
            self.registry.register(specialist)

    def consensus(
        self, feed_ids: List[str], context: Optional[Dict] = None
    ) -> QuorumResult:
        """
        Main consensus function called by governance kernel.
        Coordinates specialists to analyze memory feed and reach consensus.
        """
        if not feed_ids:
            return QuorumResult(
                consensus="No feed data provided",
                confidence=0.0,
                participant_count=0,
                agreement_level=0.0,
                evidence=[],
            )

        # Convert feed IDs to QuorumFeedItem objects
        feed_items = self._convert_feed_ids_to_items(feed_ids)

        if not feed_items:
            return QuorumResult(
                consensus="Unable to retrieve feed data",
                confidence=0.0,
                participant_count=0,
                agreement_level=0.0,
                evidence=[],
            )

        # Get applicable specialists (excluding quorum specialist for analysis)
        specialists = [
            s
            for s in self.registry.get_applicable_specialists(context)
            if s.name != "quorum_specialist"
        ]

        if not specialists:
            return QuorumResult(
                consensus="No applicable specialists available",
                confidence=0.0,
                participant_count=0,
                agreement_level=0.0,
                evidence=feed_items,
            )

        # Run specialist analyses
        specialist_analyses = []
        for specialist in specialists:
            try:
                analysis = specialist.analyze(feed_items, context)
                specialist_analyses.append(analysis)
            except Exception:
                # Log error but continue with other specialists
                continue

        # Facilitate consensus using quorum specialist
        consensus_result = self.quorum_specialist.facilitate_consensus(
            specialist_analyses, feed_items
        )

        return consensus_result

    def _convert_feed_ids_to_items(self, feed_ids: List[str]) -> List[QuorumFeedItem]:
        """Convert memory IDs to QuorumFeedItem objects."""
        feed_items = []

        if not self.mtl_kernel:
            # Create mock items for development
            for i, memory_id in enumerate(feed_ids[:5]):  # Limit to 5 items
                feed_items.append(
                    QuorumFeedItem(
                        memory_id=memory_id,
                        content=f"Mock content for memory {memory_id}",
                        relevance_score=0.7,
                        trust_score=0.6,
                        context={"source": "mock"},
                    )
                )
            return feed_items

        # Retrieve actual memory entries from MTL kernel
        for memory_id in feed_ids[:10]:  # Limit to 10 items for performance
            try:
                memory_entry = self.mtl_kernel.memory_service.retrieve(memory_id)
                if memory_entry:
                    # Get trust score
                    trust_score = self.mtl_kernel.trust_service.get_trust_score(
                        memory_id
                    )

                    feed_item = QuorumFeedItem(
                        memory_id=memory_id,
                        content=memory_entry.content,
                        relevance_score=0.8,  # Default relevance
                        trust_score=trust_score,
                        context=memory_entry.metadata or {},
                    )
                    feed_items.append(feed_item)
            except Exception:
                # Skip problematic items
                continue

        return feed_items

    def register_specialist(self, specialist):
        """Register a new specialist."""
        self.registry.register(specialist)

    def get_specialist_analysis(
        self, specialist_name: str, feed_ids: List[str], context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get analysis from a specific specialist."""
        specialist = self.registry.get_specialist(specialist_name)
        if not specialist:
            return None

        feed_items = self._convert_feed_ids_to_items(feed_ids)
        return specialist.analyze(feed_items, context)

    def get_stats(self) -> Dict:
        """Get intelligence kernel statistics."""
        registry_stats = self.registry.get_stats()

        return {
            "total_specialists": registry_stats["total_specialists"],
            "specialist_domains": registry_stats["domains"],
            "mtl_connected": bool(self.mtl_kernel),
            "quorum_specialist_active": bool(self.quorum_specialist),
            "registry_stats": registry_stats,
        }

    def set_mtl_kernel(self, mtl_kernel):
        """Set the MTL kernel for memory retrieval."""
        self.mtl_kernel = mtl_kernel
