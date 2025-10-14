"""Specialist registry and management."""

from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from ....contracts.quorum_feed import QuorumFeedItem


class Specialist(ABC):
    """Base class for intelligence specialists."""

    def __init__(self, name: str, expertise_domain: str):
        self.name = name
        self.expertise_domain = expertise_domain
        self.confidence_threshold = 0.5

    @abstractmethod
    def analyze(
        self, feed_items: List[QuorumFeedItem], context: Optional[Dict] = None
    ) -> Dict:
        """Analyze feed items and provide specialist opinion."""
        pass

    @abstractmethod
    def get_confidence(self, analysis_result: Dict) -> float:
        """Get confidence level in the analysis."""
        pass

    def is_applicable(self, context: Optional[Dict] = None) -> bool:
        """Check if this specialist should participate in the analysis."""
        return True


class SpecialistRegistry:
    """Registry for managing intelligence specialists."""

    def __init__(self):
        self._specialists: Dict[str, Specialist] = {}
        self._domains: Dict[str, List[str]] = {}

    def register(self, specialist: Specialist):
        """Register a specialist."""
        self._specialists[specialist.name] = specialist

        domain = specialist.expertise_domain
        if domain not in self._domains:
            self._domains[domain] = []
        self._domains[domain].append(specialist.name)

    def get_specialist(self, name: str) -> Optional[Specialist]:
        """Get specialist by name."""
        return self._specialists.get(name)

    def get_specialists_by_domain(self, domain: str) -> List[Specialist]:
        """Get all specialists in a domain."""
        specialist_names = self._domains.get(domain, [])
        return [self._specialists[name] for name in specialist_names]

    def get_all_specialists(self) -> List[Specialist]:
        """Get all registered specialists."""
        return list(self._specialists.values())

    def get_applicable_specialists(
        self, context: Optional[Dict] = None
    ) -> List[Specialist]:
        """Get specialists applicable for given context."""
        return [
            specialist
            for specialist in self._specialists.values()
            if specialist.is_applicable(context)
        ]

    def get_stats(self) -> Dict:
        """Get registry statistics."""
        return {
            "total_specialists": len(self._specialists),
            "domains": list(self._domains.keys()),
            "specialists_per_domain": {
                domain: len(names) for domain, names in self._domains.items()
            },
        }
