"""
Intelligence Kernel Package

The brain that routes tasks → specialists → models, fuses outputs, reasons about uncertainty, 
and collaborates with Governance, MLT, Memory, and Ingress.

Purpose:
- Plan → Select → Execute → Explain across the 21-specialist quorum
- Guarantee policy safety, traceability, and adaptability
- Serve online inference and batch jobs with canary/shadow + rollback
"""

# Import new Grace Intelligence components
from .grace_intelligence import (
    GraceIntelligence, 
    ReasoningContext, 
    ReasoningResult, 
    ReasoningStage,
    TaskSubtask,
    ActionPlan,
    VerificationResult
)

__all__ = [
    'GraceIntelligence',
    'ReasoningContext', 
    'ReasoningResult',
    'ReasoningStage',
    'TaskSubtask',
    'ActionPlan', 
    'VerificationResult'
]