"""
Grace Vault Compliance Checker - Validates requests against all vault requirements.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .vault_engine import VaultEngine, ComplianceReport
from .vault_specifications import VaultSpecifications


logger = logging.getLogger(__name__)


class VaultComplianceChecker:
    """
    High-level compliance checker that orchestrates vault validation
    and provides easy-to-use interfaces for the governance system.
    """
    
    def __init__(self, vault_engine: VaultEngine = None):
        self.vault_engine = vault_engine or VaultEngine()
        self.vault_specs = VaultSpecifications()
        
    async def check_request_compliance(self, request: Dict[str, Any]) -> ComplianceReport:
        """
        Check a request for complete vault compliance.
        
        Args:
            request: The request to validate
            
        Returns:
            ComplianceReport with validation results
        """
        logger.info(f"Checking vault compliance for request: {request.get('request_id', 'unknown')}")
        
        # Perform comprehensive validation
        report = await self.vault_engine.validate_comprehensive(request, priority_only=False)
        
        return report
    
    async def check_priority_compliance(self, request: Dict[str, Any]) -> ComplianceReport:
        """
        Check a request for priority vault compliance (vaults 2,3,6,12,15).
        
        Args:
            request: The request to validate
            
        Returns:
            ComplianceReport with priority validation results
        """
        logger.info(f"Checking priority vault compliance for request: {request.get('request_id', 'unknown')}")
        
        # Perform priority validation only
        report = await self.vault_engine.validate_comprehensive(request, priority_only=True)
        
        return report
    
    def is_compliant(self, report: ComplianceReport) -> bool:
        """Check if a compliance report indicates overall compliance."""
        return report.overall_compliance and not report.resolution_required
    
    def get_critical_violations(self, report: ComplianceReport) -> List[str]:
        """Extract critical violation descriptions from a compliance report."""
        return [violation.description for violation in report.critical_violations]
    
    def generate_compliance_summary(self, report: ComplianceReport) -> str:
        """Generate a human-readable compliance summary."""
        summary = f"Vault Compliance Report for {report.request_id}\n"
        summary += f"Timestamp: {report.timestamp}\n"
        summary += f"Overall Compliance: {'✅ PASS' if report.overall_compliance else '❌ FAIL'}\n"
        summary += f"Compliance Score: {report.compliance_score:.3f}\n"
        summary += f"Critical Violations: {len(report.critical_violations)}\n"
        
        if report.critical_violations:
            summary += "\nCritical Issues:\n"
            for violation in report.critical_violations:
                summary += f"  • Vault {violation.vault_id}: {violation.description}\n"
        
        return summary
    
    async def validate_governance_decision(self, decision_data: Dict[str, Any]) -> ComplianceReport:
        """
        Validate a governance decision against vault requirements.
        
        Args:
            decision_data: Decision data including claims, reasoning, etc.
            
        Returns:
            ComplianceReport with validation results
        """
        # Add governance-specific context
        request = {
            **decision_data,
            'request_type': 'governance_decision',
            'constitutional_compliance': True,  # Assume compliance unless proven otherwise
            'reasoning_chain': decision_data.get('reasoning_chain', []),
            'claims': decision_data.get('claims', [])
        }
        
        return await self.check_request_compliance(request)
    
    async def validate_code_change(self, code_data: Dict[str, Any]) -> ComplianceReport:
        """
        Validate a code change against vault requirements.
        
        Args:
            code_data: Code change data including changes, verification status, etc.
            
        Returns:
            ComplianceReport with validation results  
        """
        # Add code-specific context
        request = {
            **code_data,
            'request_type': 'code_change',
            'code_execution': True,
            'verification_status': code_data.get('verification_status', 'unverified'),
            'trust_level': code_data.get('trust_level', 0.0),
            'sandbox_enabled': code_data.get('sandbox_enabled', False),
            'code_changes': code_data.get('changes', [])
        }
        
        return await self.check_priority_compliance(request)  # Focus on priority vaults for code
    
    def get_vault_requirements_summary(self) -> str:
        """Get a summary of all vault requirements."""
        vaults = self.vault_specs.get_all_vaults()
        
        summary = "Grace Vault Requirements (Constitutional Trust Framework)\n"
        summary += "=" * 60 + "\n\n"
        
        for vault_id, vault in vaults.items():
            summary += f"**Vault {vault_id}: {vault.name}**\n"
            summary += f"Severity: {vault.severity.value.upper()}\n"
            summary += f"Description: {vault.description}\n"
            summary += f"Compliance Checks: {', '.join(vault.compliance_checks)}\n"
            summary += f"Validation Logic: {vault.validation_logic}\n"
            summary += f"Watermark Required: {'Yes' if vault.watermark_required else 'No'}\n"
            summary += f"Explainable: {'Yes' if vault.explainable else 'No'}\n"
            summary += "-" * 40 + "\n\n"
        
        return summary