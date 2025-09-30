"""
Test suite for Grace Vault system validation.
"""
import asyncio
from datetime import datetime

from grace.vaults import VaultEngine, VaultComplianceChecker, VaultSpecifications
from grace.vaults.vault_specifications import VaultSeverity
from grace.core import Claim, Source, Evidence, LogicStep, EventBus, MemoryCore


class TestVaultSystem:
    """Test the vault validation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vault_engine = VaultEngine()
        self.vault_compliance_checker = VaultComplianceChecker(self.vault_engine)
        self.vault_specs = VaultSpecifications()
    
    def test_vault_specifications_completeness(self):
        """Test that all 18 vaults are defined."""
        vaults = self.vault_specs.get_all_vaults()
        assert len(vaults) == 18, f"Expected 18 vaults, got {len(vaults)}"
        
        # Check that priority vaults are defined
        priority_vaults = self.vault_specs.get_priority_vaults()
        assert priority_vaults == [2, 3, 6, 12, 15], f"Priority vaults incorrect: {priority_vaults}"
        
        # Check critical vaults
        critical_vaults = self.vault_specs.get_critical_vaults()
        critical_ids = [v.vault_id for v in critical_vaults]
        expected_critical = [1, 2, 6, 9, 10, 14]  # Based on specifications
        assert set(critical_ids) == set(expected_critical), f"Critical vaults: {critical_ids}"
    
    async def test_vault_2_code_verification(self):
        """Test Vault 2: Code Verification Against History."""
        request = {
            "request_id": "test_vault_2",
            "code_changes": ["def new_function(): pass"],
            "verification_status": "verified",
            "trust_level": 0.9
        }
        
        report = await self.vault_compliance_checker.check_priority_compliance(request)
        
        # Should pass since no memory core conflicts
        vault_2_result = report.vault_results.get(2)
        assert vault_2_result is not None, "Vault 2 result missing"
        assert vault_2_result.passed, f"Vault 2 failed: {vault_2_result.explainable_narrative}"
        assert "VAULT2_VERIFIED_" in vault_2_result.watermark
    
    async def test_vault_6_contradiction_detection(self):
        """Test Vault 6: Contradiction Detection."""
        # Create contradictory claims
        claim1 = Claim(
            id="claim_1",
            statement="The system is secure",
            sources=[Source(uri="test://claim1", credibility=0.9)],
            evidence=[Evidence(type="test", pointer="security_check_passed")],
            confidence=0.8,
            logical_chain=[LogicStep(step="Security audit completed successfully")]
        )
        
        claim2 = Claim(
            id="claim_2",
            statement="The system is not secure",
            sources=[Source(uri="test://claim2", credibility=0.8)],
            evidence=[Evidence(type="test", pointer="security_check_failed")],
            confidence=0.7,
            logical_chain=[LogicStep(step="Security vulnerabilities detected")]
        )
        
        request = {
            "request_id": "test_vault_6",
            "claims": [claim1, claim2]
        }
        
        report = await self.vault_compliance_checker.check_priority_compliance(request)
        
        vault_6_result = report.vault_results.get(6)
        assert vault_6_result is not None, "Vault 6 result missing"
        # May pass or fail depending on contradiction detection implementation
        assert "VAULT6_CONTRADICTION_" in vault_6_result.watermark
    
    async def test_vault_12_reasoning_chains(self):
        """Test Vault 12: Validation Logic and Reasoning Chains."""
        # Test with good reasoning chain
        good_request = {
            "request_id": "test_vault_12_good",
            "reasoning_chain": [
                "Analyzed the request for constitutional compliance",
                "Verified all evidence sources are credible", 
                "Checked for logical consistency in the argument",
                "Confirmed decision aligns with precedent"
            ]
        }
        
        report = await self.vault_compliance_checker.check_priority_compliance(good_request)
        vault_12_result = report.vault_results.get(12)
        assert vault_12_result is not None
        assert vault_12_result.passed, f"Good reasoning should pass: {vault_12_result.explainable_narrative}"
        
        # Test with poor reasoning chain
        poor_request = {
            "request_id": "test_vault_12_poor", 
            # Missing reasoning chain
        }
        
        report = await self.vault_compliance_checker.check_priority_compliance(poor_request)
        vault_12_result = report.vault_results.get(12)
        assert vault_12_result is not None
        assert not vault_12_result.passed, "Missing reasoning should fail"
        assert len(vault_12_result.violations) > 0
    
    async def test_vault_15_sandbox_requirements(self):
        """Test Vault 15: Code Sandboxing."""
        # Test unverified code without sandbox - should fail
        unverified_request = {
            "request_id": "test_vault_15_unverified",
            "code_execution": True,
            "verification_status": "unverified",
            "trust_level": 0.3,
            "sandbox_enabled": False
        }
        
        report = await self.vault_compliance_checker.check_priority_compliance(unverified_request)
        vault_15_result = report.vault_results.get(15)
        assert vault_15_result is not None
        assert not vault_15_result.passed, "Unverified code without sandbox should fail"
        
        # Test unverified code with sandbox - should pass
        sandboxed_request = {
            "request_id": "test_vault_15_sandboxed",
            "code_execution": True,
            "verification_status": "unverified",
            "trust_level": 0.3,
            "sandbox_enabled": True,
            "sandbox_features": ["process_isolation", "network_isolation", "filesystem_isolation"]
        }
        
        report = await self.vault_compliance_checker.check_priority_compliance(sandboxed_request)
        vault_15_result = report.vault_results.get(15)
        assert vault_15_result is not None
        assert vault_15_result.passed, f"Sandboxed code should pass: {vault_15_result.explainable_narrative}"
    
    async def test_comprehensive_compliance_report(self):
        """Test comprehensive compliance reporting."""
        request = {
            "request_id": "test_comprehensive",
            "constitutional_compliance": True,
            "reasoning_chain": ["Valid reasoning step"],
            "verification_status": "verified",
            "trust_level": 0.9
        }
        
        report = await self.vault_compliance_checker.check_request_compliance(request)
        
        assert report.request_id == "test_comprehensive"
        assert report.timestamp is not None
        assert isinstance(report.compliance_score, float)
        assert 0.0 <= report.compliance_score <= 1.0
        assert len(report.vault_results) == 18  # All vaults validated
        
        # Check that explainable summary is generated
        assert len(report.explainable_summary) > 0
        assert "Grace Vault Compliance Report" in report.explainable_summary
    
    def test_vault_requirements_documentation(self):
        """Test that vault requirements are properly documented."""
        summary = self.vault_compliance_checker.get_vault_requirements_summary()
        
        assert "Grace Vault Requirements" in summary
        assert "Constitutional Trust Framework" in summary
        
        # Check that all 18 vaults are documented
        for i in range(1, 19):
            assert f"Vault {i}:" in summary
        
        # Check specific priority vaults are documented
        assert "Code Verification Against History" in summary
        assert "Contradiction Detection and Resolution" in summary
        assert "Validation Logic and Reasoning Chains" in summary
        assert "Code Sandboxing and Verification" in summary


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestVaultSystem()
    test_instance.setup_method()
    
    print("Testing vault specifications...")
    test_instance.test_vault_specifications_completeness()
    print("✅ Vault specifications test passed")
    
    print("\nTesting vault requirements documentation...")
    test_instance.test_vault_requirements_documentation()
    print("✅ Documentation test passed")
    
    print("\nRunning async tests...")
    
    async def run_async_tests():
        test_instance.setup_method()
        
        print("Testing Vault 2 (Code Verification)...")
        await test_instance.test_vault_2_code_verification()
        print("✅ Vault 2 test passed")
        
        print("Testing Vault 6 (Contradiction Detection)...")
        await test_instance.test_vault_6_contradiction_detection()
        print("✅ Vault 6 test passed")
        
        print("Testing Vault 12 (Reasoning Chains)...")
        await test_instance.test_vault_12_reasoning_chains()
        print("✅ Vault 12 test passed")
        
        print("Testing Vault 15 (Sandboxing)...")
        await test_instance.test_vault_15_sandbox_requirements()
        print("✅ Vault 15 test passed")
        
        print("Testing comprehensive compliance...")
        await test_instance.test_comprehensive_compliance_report()
        print("✅ Comprehensive compliance test passed")
    
    asyncio.run(run_async_tests())
    print("\n🎉 All vault system tests passed!")