"""Grace Immune System - Health monitoring and predictive analytics."""

from typing import TYPE_CHECKING

# Type checking imports
if TYPE_CHECKING:
    try:
        from grace.immune_system.enhanced_avn_core import (
            EnhancedAVNCore,
            HealthStatus,
            PredictiveAlert,
        )
    except ImportError:
        pass

# Runtime imports with fallbacks
try:
    from grace.immune_system.enhanced_avn_core import (
        EnhancedAVNCore,
        HealthStatus,
        PredictiveAlert,
    )
except ImportError:
    # Provide stub classes if module doesn't exist
    class EnhancedAVNCore:  # type: ignore
        """Stub for EnhancedAVNCore."""
        pass
    
    class HealthStatus:  # type: ignore
        """Stub for HealthStatus."""
        pass
    
    class PredictiveAlert:  # type: ignore
        """Stub for PredictiveAlert."""
        pass

__all__ = ["EnhancedAVNCore", "HealthStatus", "PredictiveAlert"]
