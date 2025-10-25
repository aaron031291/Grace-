from dataclasses import dataclass
from typing import List

@dataclass
class Principle:
    """A single principle or rule in the constitution."""
    name: str
    description: str
    category: str  # e.g., 'ethics', 'safety', 'performance'

class Constitution:
    """
    Represents the set of guiding principles for Grace.
    This is a placeholder implementation.
    """
    def __init__(self):
        self.principles: List[Principle] = []
        self._load_principles()

    def _load_principles(self):
        """Loads the core principles of the constitution."""
        # In a real system, this would load from a config file (e.g., YAML)
        self.principles.extend([
            Principle(
                name="DoNoHarm",
                description="The system must not cause harm to humans or property.",
                category="safety"
            ),
            Principle(
                name="BeHelpful",
                description="The system should strive to be helpful and useful to the user.",
                category="ethics"
            ),
            Principle(
                name="BeTransparent",
                description="The system's actions and reasoning should be as transparent as possible.",
                category="ethics"
            )
        ])

    def get_all_principles(self) -> List[Principle]:
        """Returns all principles."""
        return self.principles

    def get_principle(self, name: str) -> Principle | None:
        """Finds a principle by name."""
        for p in self.principles:
            if p.name == name:
                return p
        return None

__all__ = ["Constitution", "Principle"]
