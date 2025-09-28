"""
Grace AI System - Distributed Governance & Intelligence

Grace is a comprehensive AI-powered system featuring a multi-kernel architecture
for distributed governance, intelligence gathering, and memory management.

This package provides:
- Multi-kernel distributed architecture
- AI-powered governance and decision making
- Comprehensive memory management (Lightning/Fusion/Vector)
- Trust ledgers with decay mechanisms
- Immutable logging with Merkle proofs
- Quorum-based consensus intelligence
- Self-healing and adaptation capabilities
- Cross-platform runtime management
"""

__version__ = "0.1.0"
__author__ = "Grace Team"
__email__ = "team@grace.ai"

from .mtl_kernel.kernel import MTLKernel
from .governance_kernel.kernel import GovernanceKernel
from .intelligence_kernel.kernel import IntelligenceKernel

__all__ = [
    "MTLKernel",
    "GovernanceKernel", 
    "IntelligenceKernel",
]