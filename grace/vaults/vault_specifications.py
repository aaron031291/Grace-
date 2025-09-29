"""
Grace Vault Specifications - Defines the 18 core constitutional trust requirements.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class VaultSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VaultRequirement:
    """Individual vault requirement specification."""
    vault_id: int
    name: str
    description: str
    severity: VaultSeverity
    compliance_checks: List[str]
    validation_logic: str
    watermark_required: bool = True
    explainable: bool = True


class VaultSpecifications:
    """Defines all 18 Grace Vault requirements for constitutional trust compliance."""
    
    @classmethod
    def get_all_vaults(cls) -> Dict[int, VaultRequirement]:
        """Return all 18 vault specifications."""
        return {
            1: VaultRequirement(
                vault_id=1,
                name="Constitutional Compliance",
                description="All actions must comply with core constitutional principles",
                severity=VaultSeverity.CRITICAL,
                compliance_checks=["transparency_check", "fairness_check", "accountability_check"],
                validation_logic="Validate against constitutional_principles database"
            ),
            
            2: VaultRequirement(
                vault_id=2,
                name="Code Verification Against History",
                description="Verify code changes against past logic, memory, and claims",
                severity=VaultSeverity.CRITICAL,
                compliance_checks=["historical_logic_check", "memory_correlation", "claim_verification"],
                validation_logic="Cross-reference with memory_core and previous decisions"
            ),
            
            3: VaultRequirement(
                vault_id=3,
                name="System Memory Correlation",
                description="Correlate changes to system memory and history",
                severity=VaultSeverity.HIGH,
                compliance_checks=["memory_consistency", "historical_alignment", "state_correlation"],
                validation_logic="Ensure changes align with system memory and historical context"
            ),
            
            4: VaultRequirement(
                vault_id=4,
                name="Trust Score Validation",
                description="Validate trust scores and credibility assessments",
                severity=VaultSeverity.HIGH,
                compliance_checks=["trust_computation", "credibility_assessment", "historical_trust"],
                validation_logic="Verify trust scores using multi-factor analysis"
            ),
            
            5: VaultRequirement(
                vault_id=5,
                name="Decision Precedent Analysis",
                description="Analyze decisions against precedent and case law",
                severity=VaultSeverity.MEDIUM,
                compliance_checks=["precedent_search", "case_law_alignment", "consistency_check"],
                validation_logic="Compare with historical decisions for consistency"
            ),
            
            6: VaultRequirement(
                vault_id=6,
                name="Contradiction Detection and Resolution",
                description="Detect and resolve contradictions in claims and logic",
                severity=VaultSeverity.CRITICAL,
                compliance_checks=["logical_consistency", "claim_contradiction", "resolution_strategy"],
                validation_logic="Multi-pass contradiction detection with resolution protocols"
            ),
            
            7: VaultRequirement(
                vault_id=7,
                name="Evidence Quality Assessment",
                description="Assess quality and reliability of evidence sources",
                severity=VaultSeverity.HIGH,
                compliance_checks=["source_credibility", "evidence_quality", "verification_chain"],
                validation_logic="Multi-dimensional evidence quality scoring"
            ),
            
            8: VaultRequirement(
                vault_id=8,
                name="Bias Detection and Mitigation",
                description="Detect and mitigate various forms of bias in decisions",
                severity=VaultSeverity.HIGH,
                compliance_checks=["algorithmic_bias", "cognitive_bias", "data_bias"],
                validation_logic="Comprehensive bias analysis across multiple dimensions"
            ),
            
            9: VaultRequirement(
                vault_id=9,
                name="Privacy Protection Enforcement",
                description="Enforce privacy protection and data minimization principles",
                severity=VaultSeverity.CRITICAL,
                compliance_checks=["pii_detection", "data_minimization", "consent_validation"],
                validation_logic="Privacy-preserving analysis with anonymization"
            ),
            
            10: VaultRequirement(
                vault_id=10,
                name="Harm Prevention Assessment",
                description="Assess and prevent potential harm from decisions or actions",
                severity=VaultSeverity.CRITICAL,
                compliance_checks=["risk_assessment", "harm_prediction", "mitigation_planning"],
                validation_logic="Multi-vector harm assessment with prevention protocols"
            ),
            
            11: VaultRequirement(
                vault_id=11,
                name="Audit Trail Completeness",
                description="Ensure complete and immutable audit trails for all decisions",
                severity=VaultSeverity.HIGH,
                compliance_checks=["trail_completeness", "immutability_check", "traceability"],
                validation_logic="Comprehensive audit trail validation and integrity checks"
            ),
            
            12: VaultRequirement(
                vault_id=12,
                name="Validation Logic and Reasoning Chains",
                description="Provide clear validation logic and narratable reasoning chains",
                severity=VaultSeverity.HIGH,
                compliance_checks=["logic_clarity", "reasoning_completeness", "narrative_coherence"],
                validation_logic="Structured reasoning validation with narrative generation"
            ),
            
            13: VaultRequirement(
                vault_id=13,
                name="Democratic Oversight Compliance",
                description="Ensure appropriate democratic oversight and review processes",
                severity=VaultSeverity.MEDIUM,
                compliance_checks=["oversight_threshold", "review_process", "democratic_participation"],
                validation_logic="Parliamentary review system with configurable thresholds"
            ),
            
            14: VaultRequirement(
                vault_id=14,
                name="Legal and Regulatory Compliance",
                description="Ensure compliance with applicable laws and regulations",
                severity=VaultSeverity.CRITICAL,
                compliance_checks=["legal_compliance", "regulatory_alignment", "jurisdiction_specific"],
                validation_logic="Legal framework analysis with jurisdiction awareness"
            ),
            
            15: VaultRequirement(
                vault_id=15,
                name="Code Sandboxing and Verification",
                description="Sandbox all new code unless fully verified and trusted",
                severity=VaultSeverity.HIGH,
                compliance_checks=["sandbox_isolation", "verification_status", "trust_level"],
                validation_logic="Graduated trust system with sandbox controls"
            ),
            
            16: VaultRequirement(
                vault_id=16,
                name="Resource Allocation Fairness",
                description="Ensure fair and efficient allocation of computational resources",
                severity=VaultSeverity.MEDIUM,
                compliance_checks=["resource_fairness", "allocation_efficiency", "priority_management"],
                validation_logic="Fair resource allocation with priority-based scheduling"
            ),
            
            17: VaultRequirement(
                vault_id=17,
                name="Meta-Learning Integration",
                description="Integrate meta-learning insights for continuous improvement",
                severity=VaultSeverity.MEDIUM,
                compliance_checks=["learning_integration", "improvement_tracking", "adaptation_validation"],
                validation_logic="Meta-learning feedback loop with improvement validation"
            ),
            
            18: VaultRequirement(
                vault_id=18,
                name="System Resilience and Recovery",
                description="Ensure system resilience and graceful degradation capabilities",
                severity=VaultSeverity.HIGH,
                compliance_checks=["resilience_testing", "recovery_protocols", "degradation_graceful"],
                validation_logic="Comprehensive resilience validation with recovery testing"
            )
        }
    
    @classmethod
    def get_vault(cls, vault_id: int) -> Optional[VaultRequirement]:
        """Get a specific vault requirement by ID."""
        vaults = cls.get_all_vaults()
        return vaults.get(vault_id)
    
    @classmethod
    def get_critical_vaults(cls) -> List[VaultRequirement]:
        """Get all critical severity vaults."""
        vaults = cls.get_all_vaults()
        return [vault for vault in vaults.values() if vault.severity == VaultSeverity.CRITICAL]
    
    @classmethod
    def get_priority_vaults(cls) -> List[int]:
        """Get priority vault IDs mentioned in problem statement."""
        return [2, 3, 6, 12, 15]  # As specified in the problem statement