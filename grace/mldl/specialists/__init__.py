"""
Enhanced ML/DL Specialists package - Next-generation specialists with cross-domain validation.
"""

# Enhanced specialists
from .enhanced_specialists import (
    EnhancedMLSpecialist,
    SpecialistPrediction,
    CrossDomainValidation,
    CrossDomainValidator,
    GraphNeuralNetworkSpecialist,
    MultimodalAISpecialist,
    UncertaintyQuantificationSpecialist,
    create_enhanced_specialists,
    create_cross_domain_validators,
)

# Legacy governance liaison (if available)
try:
    from .governance_liaison import GovernanceLiaisonSpecialist

    LEGACY_LIAISON_AVAILABLE = True
except ImportError:
    LEGACY_LIAISON_AVAILABLE = False

__all__ = [
    "EnhancedMLSpecialist",
    "SpecialistPrediction",
    "CrossDomainValidation",
    "CrossDomainValidator",
    "GraphNeuralNetworkSpecialist",
    "MultimodalAISpecialist",
    "UncertaintyQuantificationSpecialist",
    "create_enhanced_specialists",
    "create_cross_domain_validators",
]

if LEGACY_LIAISON_AVAILABLE:
    __all__.append("GovernanceLiaisonSpecialist")
