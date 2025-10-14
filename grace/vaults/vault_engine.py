"""
Grace Vault Engine - Core implementation of the constitutional trust framework.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass

from .vault_specifications import VaultSpecifications, VaultRequirement, VaultSeverity
from ..core.contracts import generate_decision_id


logger = logging.getLogger(__name__)


@dataclass
class VaultViolation:
    """Represents a violation of a vault requirement."""

    vault_id: int
    vault_name: str
    severity: VaultSeverity
    description: str
    evidence: List[str]
    resolution_required: bool = True
    watermark: str = ""


@dataclass
class VaultValidationResult:
    """Result of vault validation process."""

    request_id: str
    vault_id: int
    passed: bool
    confidence: float
    violations: List[VaultViolation]
    reasoning_chain: List[str]
    watermark: str
    explainable_narrative: str


@dataclass
class ComplianceReport:
    """Comprehensive compliance report for all vaults."""

    request_id: str
    timestamp: datetime
    overall_compliance: bool
    compliance_score: float
    vault_results: Dict[int, VaultValidationResult]
    critical_violations: List[VaultViolation]
    resolution_required: bool
    explainable_summary: str


class VaultEngine:
    """
    Core engine for Grace Vault validation and constitutional trust enforcement.

    This engine orchestrates validation across all 18 vaults and ensures
    constitutional compliance for all system operations.
    """

    def __init__(self, event_bus=None, memory_core=None, governance_kernel=None):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.governance_kernel = governance_kernel
        self.vault_specs = VaultSpecifications()
        self.validation_history = []

    async def validate_comprehensive(
        self, request: Dict[str, Any], priority_only: bool = False
    ) -> ComplianceReport:
        """
        Perform comprehensive vault validation across all or priority vaults.

        Args:
            request: The request/action to validate
            priority_only: If True, only validate priority vaults (2,3,6,12,15)
        """
        request_id = request.get("request_id", generate_decision_id())
        logger.info(f"Starting comprehensive vault validation for {request_id}")

        # Determine which vaults to validate
        if priority_only:
            vault_ids = self.vault_specs.get_priority_vaults()
        else:
            vault_ids = list(self.vault_specs.get_all_vaults().keys())

        vault_results = {}
        critical_violations = []

        # Validate each vault
        for vault_id in vault_ids:
            result = await self._validate_single_vault(request, vault_id)
            vault_results[vault_id] = result

            # Collect critical violations
            critical_violations.extend(
                [v for v in result.violations if v.severity == VaultSeverity.CRITICAL]
            )

        # Calculate overall compliance
        overall_compliance = len(critical_violations) == 0
        compliance_score = self._calculate_compliance_score(vault_results)

        # Generate explainable summary
        explainable_summary = self._generate_compliance_narrative(
            vault_results, critical_violations, compliance_score
        )

        report = ComplianceReport(
            request_id=request_id,
            timestamp=datetime.now(),
            overall_compliance=overall_compliance,
            compliance_score=compliance_score,
            vault_results=vault_results,
            critical_violations=critical_violations,
            resolution_required=len(critical_violations) > 0,
            explainable_summary=explainable_summary,
        )

        # Store in validation history
        self.validation_history.append(report)

        return report

    async def _validate_single_vault(
        self, request: Dict[str, Any], vault_id: int
    ) -> VaultValidationResult:
        """Validate a single vault requirement."""
        vault_spec = self.vault_specs.get_vault(vault_id)
        if not vault_spec:
            raise ValueError(f"Unknown vault ID: {vault_id}")

        request_id = request.get("request_id", generate_decision_id())

        # Dispatch to specific vault validation logic
        if vault_id == 2:
            return await self._validate_vault_2(request, vault_spec)
        elif vault_id == 3:
            return await self._validate_vault_3(request, vault_spec)
        elif vault_id == 6:
            return await self._validate_vault_6(request, vault_spec)
        elif vault_id == 12:
            return await self._validate_vault_12(request, vault_spec)
        elif vault_id == 15:
            return await self._validate_vault_15(request, vault_spec)
        else:
            return await self._validate_generic_vault(request, vault_spec)

    async def _validate_vault_2(
        self, request: Dict[str, Any], vault_spec: VaultRequirement
    ) -> VaultValidationResult:
        """Validate Vault 2: Code Verification Against History."""
        request_id = request.get("request_id", generate_decision_id())
        violations = []
        reasoning_chain = []

        # Check if we have memory core for historical verification
        if not self.memory_core:
            violations.append(
                VaultViolation(
                    vault_id=2,
                    vault_name=vault_spec.name,
                    severity=vault_spec.severity,
                    description="Memory core not available for historical verification",
                    evidence=["memory_core_missing"],
                )
            )

        # Check for code changes in request
        code_changes = request.get("code_changes", [])
        if code_changes:
            reasoning_chain.append("Analyzing code changes against historical logic")

            # Simulate historical logic check
            historical_conflicts = await self._check_historical_logic(code_changes)
            if historical_conflicts:
                violations.append(
                    VaultViolation(
                        vault_id=2,
                        vault_name=vault_spec.name,
                        severity=vault_spec.severity,
                        description=f"Code conflicts with {len(historical_conflicts)} historical decisions",
                        evidence=historical_conflicts,
                    )
                )

        # Check memory correlation
        if self.memory_core:
            reasoning_chain.append("Correlating with system memory")
            memory_issues = await self._check_memory_correlation(request)
            if memory_issues:
                violations.append(
                    VaultViolation(
                        vault_id=2,
                        vault_name=vault_spec.name,
                        severity=vault_spec.severity,
                        description="Memory correlation issues detected",
                        evidence=memory_issues,
                    )
                )

        passed = len(violations) == 0
        confidence = 0.9 if passed else max(0.1, 0.9 - (len(violations) * 0.2))

        watermark = f"VAULT2_VERIFIED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        narrative = self._generate_vault_narrative(
            vault_spec, violations, reasoning_chain, passed
        )

        return VaultValidationResult(
            request_id=request_id,
            vault_id=2,
            passed=passed,
            confidence=confidence,
            violations=violations,
            reasoning_chain=reasoning_chain,
            watermark=watermark,
            explainable_narrative=narrative,
        )

    async def _validate_vault_3(
        self, request: Dict[str, Any], vault_spec: VaultRequirement
    ) -> VaultValidationResult:
        """Validate Vault 3: System Memory Correlation."""
        request_id = request.get("request_id", generate_decision_id())
        violations = []
        reasoning_chain = []

        reasoning_chain.append("Analyzing system memory correlation")

        # Check memory consistency
        if self.memory_core:
            memory_state = await self._get_memory_state()
            if not self._validate_memory_consistency(request, memory_state):
                violations.append(
                    VaultViolation(
                        vault_id=3,
                        vault_name=vault_spec.name,
                        severity=vault_spec.severity,
                        description="Request inconsistent with current memory state",
                        evidence=["memory_state_mismatch"],
                    )
                )

        # Check historical alignment
        historical_alignment = await self._check_historical_alignment(request)
        if not historical_alignment:
            violations.append(
                VaultViolation(
                    vault_id=3,
                    vault_name=vault_spec.name,
                    severity=vault_spec.severity,
                    description="Request not aligned with historical patterns",
                    evidence=["historical_misalignment"],
                )
            )

        passed = len(violations) == 0
        confidence = 0.85 if passed else max(0.2, 0.85 - (len(violations) * 0.3))

        watermark = f"VAULT3_MEMORY_CORR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        narrative = self._generate_vault_narrative(
            vault_spec, violations, reasoning_chain, passed
        )

        return VaultValidationResult(
            request_id=request_id,
            vault_id=3,
            passed=passed,
            confidence=confidence,
            violations=violations,
            reasoning_chain=reasoning_chain,
            watermark=watermark,
            explainable_narrative=narrative,
        )

    async def _validate_vault_6(
        self, request: Dict[str, Any], vault_spec: VaultRequirement
    ) -> VaultValidationResult:
        """Validate Vault 6: Contradiction Detection and Resolution."""
        request_id = request.get("request_id", generate_decision_id())
        violations = []
        reasoning_chain = []

        reasoning_chain.append("Performing contradiction detection analysis")

        # Extract claims from request
        claims = request.get("claims", [])
        if claims:
            # Use existing verification engine for contradiction detection
            if hasattr(self.governance_kernel, "verification_engine"):
                verification_engine = self.governance_kernel.verification_engine
                contradictions = []

                # Check for internal contradictions
                for i, claim in enumerate(claims):
                    for j, other_claim in enumerate(claims[i + 1 :], i + 1):
                        contradiction = await verification_engine._detect_contradiction(
                            claim, other_claim
                        )
                        if contradiction:
                            contradictions.append(contradiction)

                if contradictions:
                    violations.append(
                        VaultViolation(
                            vault_id=6,
                            vault_name=vault_spec.name,
                            severity=vault_spec.severity,
                            description=f"Detected {len(contradictions)} contradictions in claims",
                            evidence=[
                                f"Contradiction: {c.description}"
                                for c in contradictions
                            ],
                        )
                    )

        # Check logical consistency
        logical_issues = await self._check_logical_consistency(request)
        if logical_issues:
            violations.append(
                VaultViolation(
                    vault_id=6,
                    vault_name=vault_spec.name,
                    severity=vault_spec.severity,
                    description="Logical consistency issues detected",
                    evidence=logical_issues,
                )
            )

        passed = len(violations) == 0
        confidence = 0.95 if passed else max(0.1, 0.95 - (len(violations) * 0.4))

        watermark = f"VAULT6_CONTRADICTION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        narrative = self._generate_vault_narrative(
            vault_spec, violations, reasoning_chain, passed
        )

        return VaultValidationResult(
            request_id=request_id,
            vault_id=6,
            passed=passed,
            confidence=confidence,
            violations=violations,
            reasoning_chain=reasoning_chain,
            watermark=watermark,
            explainable_narrative=narrative,
        )

    async def _validate_vault_12(
        self, request: Dict[str, Any], vault_spec: VaultRequirement
    ) -> VaultValidationResult:
        """Validate Vault 12: Validation Logic and Reasoning Chains."""
        request_id = request.get("request_id", generate_decision_id())
        violations = []
        reasoning_chain = []

        reasoning_chain.append("Validating reasoning chain completeness and clarity")

        # Check for reasoning chain presence
        if "reasoning_chain" not in request and "logical_chain" not in request:
            violations.append(
                VaultViolation(
                    vault_id=12,
                    vault_name=vault_spec.name,
                    severity=vault_spec.severity,
                    description="No reasoning chain provided with request",
                    evidence=["missing_reasoning_chain"],
                )
            )

        # Validate reasoning clarity
        reasoning = request.get("reasoning_chain", request.get("logical_chain", []))
        if reasoning:
            clarity_score = self._assess_reasoning_clarity(reasoning)
            if clarity_score < 0.7:
                violations.append(
                    VaultViolation(
                        vault_id=12,
                        vault_name=vault_spec.name,
                        severity=vault_spec.severity,
                        description=f"Reasoning clarity insufficient (score: {clarity_score:.2f})",
                        evidence=["low_clarity_score"],
                    )
                )

        # Check narrative coherence
        narrative_coherence = self._check_narrative_coherence(request)
        if not narrative_coherence:
            violations.append(
                VaultViolation(
                    vault_id=12,
                    vault_name=vault_spec.name,
                    severity=vault_spec.severity,
                    description="Narrative coherence issues detected",
                    evidence=["incoherent_narrative"],
                )
            )

        passed = len(violations) == 0
        confidence = 0.9 if passed else max(0.3, 0.9 - (len(violations) * 0.2))

        watermark = f"VAULT12_REASONING_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        narrative = self._generate_vault_narrative(
            vault_spec, violations, reasoning_chain, passed
        )

        return VaultValidationResult(
            request_id=request_id,
            vault_id=12,
            passed=passed,
            confidence=confidence,
            violations=violations,
            reasoning_chain=reasoning_chain,
            watermark=watermark,
            explainable_narrative=narrative,
        )

    async def _validate_vault_15(
        self, request: Dict[str, Any], vault_spec: VaultRequirement
    ) -> VaultValidationResult:
        """Validate Vault 15: Code Sandboxing and Verification."""
        request_id = request.get("request_id", generate_decision_id())
        violations = []
        reasoning_chain = []

        reasoning_chain.append(
            "Checking sandbox isolation and code verification status"
        )

        # Check for code execution requests
        code_execution = request.get("code_execution", False)
        new_code = request.get("code_changes", [])

        if code_execution or new_code:
            # Check verification status
            verification_status = request.get("verification_status", "unverified")
            trust_level = request.get("trust_level", 0.0)

            if verification_status != "verified" and trust_level < 0.8:
                sandbox_required = True
                sandbox_enabled = request.get("sandbox_enabled", False)

                if not sandbox_enabled:
                    violations.append(
                        VaultViolation(
                            vault_id=15,
                            vault_name=vault_spec.name,
                            severity=vault_spec.severity,
                            description="Unverified code must run in sandbox",
                            evidence=[
                                f"verification_status: {verification_status}",
                                f"trust_level: {trust_level}",
                            ],
                        )
                    )

        # Check sandbox isolation (only if sandbox is required for unverified code)
        if request.get("sandbox_enabled", False):
            verification_status = request.get("verification_status", "unverified")
            trust_level = request.get("trust_level", 0.0)

            # Only check isolation quality if code is unverified or low trust
            if verification_status != "verified" or trust_level < 0.8:
                isolation_score = self._assess_sandbox_isolation(request)
                if isolation_score < 0.8:
                    violations.append(
                        VaultViolation(
                            vault_id=15,
                            vault_name=vault_spec.name,
                            severity=vault_spec.severity,
                            description=f"Sandbox isolation insufficient (score: {isolation_score:.2f})",
                            evidence=["weak_sandbox_isolation"],
                        )
                    )

        passed = len(violations) == 0
        confidence = 0.95 if passed else max(0.2, 0.95 - (len(violations) * 0.3))

        watermark = f"VAULT15_SANDBOX_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        narrative = self._generate_vault_narrative(
            vault_spec, violations, reasoning_chain, passed
        )

        return VaultValidationResult(
            request_id=request_id,
            vault_id=15,
            passed=passed,
            confidence=confidence,
            violations=violations,
            reasoning_chain=reasoning_chain,
            watermark=watermark,
            explainable_narrative=narrative,
        )

    async def _validate_generic_vault(
        self, request: Dict[str, Any], vault_spec: VaultRequirement
    ) -> VaultValidationResult:
        """Generic validation for non-priority vaults."""
        request_id = request.get("request_id", generate_decision_id())

        # Basic validation - check for constitutional compliance
        violations = []
        reasoning_chain = [f"Performing basic validation for {vault_spec.name}"]

        # Placeholder logic for generic vaults
        basic_compliance = request.get("constitutional_compliance", True)
        if not basic_compliance:
            violations.append(
                VaultViolation(
                    vault_id=vault_spec.vault_id,
                    vault_name=vault_spec.name,
                    severity=vault_spec.severity,
                    description="Basic constitutional compliance check failed",
                    evidence=["constitutional_compliance_false"],
                )
            )

        passed = len(violations) == 0
        confidence = 0.7 if passed else 0.3

        watermark = f"VAULT{vault_spec.vault_id}_BASIC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        narrative = self._generate_vault_narrative(
            vault_spec, violations, reasoning_chain, passed
        )

        return VaultValidationResult(
            request_id=request_id,
            vault_id=vault_spec.vault_id,
            passed=passed,
            confidence=confidence,
            violations=violations,
            reasoning_chain=reasoning_chain,
            watermark=watermark,
            explainable_narrative=narrative,
        )

    # Helper methods
    async def _check_historical_logic(self, code_changes: List[str]) -> List[str]:
        """Check code changes against historical logic."""
        # Placeholder implementation
        return []  # No conflicts found

    async def _check_memory_correlation(self, request: Dict[str, Any]) -> List[str]:
        """Check memory correlation issues."""
        # Placeholder implementation
        return []  # No issues found

    async def _get_memory_state(self) -> Dict[str, Any]:
        """Get current memory state."""
        if self.memory_core:
            # Return simplified memory state
            return {"status": "healthy", "entries": 100}
        return {}

    def _validate_memory_consistency(
        self, request: Dict[str, Any], memory_state: Dict[str, Any]
    ) -> bool:
        """Validate memory consistency."""
        return True  # Placeholder

    async def _check_historical_alignment(self, request: Dict[str, Any]) -> bool:
        """Check historical alignment."""
        return True  # Placeholder

    async def _check_logical_consistency(self, request: Dict[str, Any]) -> List[str]:
        """Check logical consistency."""
        return []  # No issues found

    def _assess_reasoning_clarity(self, reasoning: List[str]) -> float:
        """Assess clarity of reasoning chain."""
        if not reasoning:
            return 0.0

        # Simple clarity assessment based on length and content
        total_length = sum(len(step) for step in reasoning)
        avg_length = total_length / len(reasoning)

        # Good reasoning steps should be neither too short nor too long
        if 20 <= avg_length <= 200:
            return 0.9
        elif 10 <= avg_length <= 300:
            return 0.7
        else:
            return 0.5

    def _check_narrative_coherence(self, request: Dict[str, Any]) -> bool:
        """Check narrative coherence."""
        return True  # Placeholder

    def _assess_sandbox_isolation(self, request: Dict[str, Any]) -> float:
        """Assess sandbox isolation quality."""
        isolation_features = request.get("sandbox_features", [])
        if not isolation_features:
            return 0.3

        # Score based on isolation features
        feature_scores = {
            "process_isolation": 0.3,
            "network_isolation": 0.2,
            "filesystem_isolation": 0.2,
            "memory_isolation": 0.2,
            "resource_limits": 0.1,
        }

        score = sum(feature_scores.get(feature, 0) for feature in isolation_features)
        return min(1.0, score)

    def _calculate_compliance_score(
        self, vault_results: Dict[int, VaultValidationResult]
    ) -> float:
        """Calculate overall compliance score."""
        if not vault_results:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for vault_id, result in vault_results.items():
            vault_spec = self.vault_specs.get_vault(vault_id)
            weight = 1.0

            if vault_spec.severity == VaultSeverity.CRITICAL:
                weight = 3.0
            elif vault_spec.severity == VaultSeverity.HIGH:
                weight = 2.0
            elif vault_spec.severity == VaultSeverity.MEDIUM:
                weight = 1.5

            score = result.confidence if result.passed else 0.0
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_vault_narrative(
        self,
        vault_spec: VaultRequirement,
        violations: List[VaultViolation],
        reasoning_chain: List[str],
        passed: bool,
    ) -> str:
        """Generate explainable narrative for vault validation."""
        narrative = f"**{vault_spec.name} (Vault {vault_spec.vault_id})**: "

        if passed:
            narrative += f"✅ PASSED - {vault_spec.description}\n"
            narrative += f"Validation process: {' → '.join(reasoning_chain)}\n"
            narrative += "All compliance checks satisfied."
        else:
            narrative += f"❌ FAILED - {len(violations)} violation(s) detected\n"
            narrative += f"Validation process: {' → '.join(reasoning_chain)}\n"
            narrative += "Violations:\n"
            for violation in violations:
                narrative += f"  • {violation.description}\n"
                narrative += f"    Evidence: {', '.join(violation.evidence)}\n"

        return narrative

    def _generate_compliance_narrative(
        self,
        vault_results: Dict[int, VaultValidationResult],
        critical_violations: List[VaultViolation],
        compliance_score: float,
    ) -> str:
        """Generate overall compliance narrative."""
        narrative = "**Grace Vault Compliance Report**\n"
        narrative += f"Overall Compliance Score: {compliance_score:.2f}/1.00\n\n"

        if not critical_violations:
            narrative += (
                "✅ **COMPLIANCE ACHIEVED** - All critical requirements met\n\n"
            )
        else:
            narrative += f"❌ **COMPLIANCE FAILED** - {len(critical_violations)} critical violations\n\n"

        # Summary by severity
        passed_count = sum(1 for r in vault_results.values() if r.passed)
        total_count = len(vault_results)

        narrative += f"Vault Results: {passed_count}/{total_count} passed\n"

        # Critical violations detail
        if critical_violations:
            narrative += "\n**Critical Violations Requiring Resolution:**\n"
            for violation in critical_violations:
                narrative += f"• Vault {violation.vault_id}: {violation.description}\n"

        narrative += "\n**Constitutional Trust Framework**: "
        if compliance_score >= 0.9:
            narrative += "Fully aligned with Grace constitutional principles"
        elif compliance_score >= 0.7:
            narrative += "Substantially compliant with constitutional principles"
        else:
            narrative += "Significant constitutional compliance gaps require resolution"

        return narrative
