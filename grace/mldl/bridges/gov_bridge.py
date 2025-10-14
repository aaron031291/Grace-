"""
MLDL-Governance Bridge - Connects MLDL Kernel to Governance Engine.
"""

import logging

logger = logging.getLogger(__name__)


class MLDLGovernanceBridge:
    """Bridge between MLDL Kernel and Governance Engine."""

    def __init__(self, governance_engine=None, event_bus=None):
        self.governance_engine = governance_engine
        self.event_bus = event_bus

        # Governance checkpoints for MLDL
        self.governance_checkpoints = {
            "model_training": {
                "required": False,
                "approval_types": ["data_usage", "resource_usage"],
            },
            "model_registration": {
                "required": True,
                "approval_types": ["model_quality", "fairness_check", "security_scan"],
            },
            "staging_deployment": {
                "required": False,
                "approval_types": ["deployment_readiness"],
            },
            "production_deployment": {
                "required": True,
                "approval_types": [
                    "production_readiness",
                    "compliance_check",
                    "risk_assessment",
                ],
            },
            "canary_promotion": {
                "required": False,
                "approval_types": ["performance_validation"],
            },
        }

        logger.info("MLDL Governance Bridge initialized")
