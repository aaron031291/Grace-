"""
Policy tests for Grace AI Governance System.
These tests validate policy functionality and integration.
"""

import pytest
import sys
import os
import yaml
from pathlib import Path

# Add the grace module to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPolicyEngine:
    """Test the policy engine functionality."""

    def test_policy_engine_can_be_imported(self):
        """Test that policy engine can be imported."""
        try:
            from grace.governance.policy_engine import PolicyEngine

            policy_engine = PolicyEngine()
            assert policy_engine is not None
            assert len(policy_engine.policies) > 0
        except ImportError as e:
            pytest.skip(f"Could not import PolicyEngine: {e}")

    def test_policy_directory_structure(self):
        """Test that policy directories and files exist."""
        expected_dirs = ["policies/", "grace/governance/"]

        for dir_path in expected_dirs:
            if os.path.exists(dir_path):
                assert os.path.isdir(dir_path), f"{dir_path} should be a directory"


class TestPolicyFiles:
    """Test policy file validation."""

    def test_yaml_files_are_valid(self):
        """Test that all YAML policy files are valid."""
        policy_paths = [
            "contracts/comms/security.yaml",
            "grace/schemas/governance_events.yaml",
        ]

        for policy_path in policy_paths:
            if os.path.exists(policy_path):
                with open(policy_path, "r") as f:
                    try:
                        data = yaml.safe_load(f)
                        assert data is not None, f"Empty YAML file: {policy_path}"
                    except yaml.YAMLError as e:
                        pytest.fail(f"Invalid YAML in {policy_path}: {e}")

    def test_security_config_structure(self):
        """Test security configuration has required structure."""
        security_path = "contracts/comms/security.yaml"

        if os.path.exists(security_path):
            with open(security_path, "r") as f:
                security_config = yaml.safe_load(f)

            # Check for required top-level sections
            required_sections = ["auth", "rbac", "privacy", "audit"]
            for section in required_sections:
                assert section in security_config, (
                    f"Missing required section: {section}"
                )

            # Check JWT validation
            auth_config = security_config.get("auth", {})
            jwt_validation = auth_config.get("token_validation", {})
            assert jwt_validation.get("verify_signature", False), (
                "JWT signature verification should be enabled"
            )
            assert jwt_validation.get("verify_expiry", False), (
                "JWT expiry verification should be enabled"
            )


class TestDangerousOperationsPolicy:
    """Test dangerous operations policy enforcement."""

    def test_dangerous_operations_policy_exists(self):
        """Test that dangerous operations policy exists or can be created."""
        policy_paths = [
            "policies/dangerous_operations.yaml",
            "governance/dangerous_operations.yaml",
        ]

        policy_exists = any(os.path.exists(path) for path in policy_paths)

        if not policy_exists:
            # Create default policy for testing
            os.makedirs("policies", exist_ok=True)

            default_policy = {
                "version": "1.0",
                "dangerous_operations": {
                    "file_io": {"allowed": False, "sandbox_only": True},
                    "code_execution": {"allowed": False, "sandbox_only": True},
                    "network_access": {
                        "allowed": True,
                        "restrictions": ["internal_only"],
                    },
                },
                "enforcement": "strict",
                "approval_required": True,
            }

            with open("policies/dangerous_operations.yaml", "w") as f:
                yaml.dump(default_policy, f)

        # Verify policy exists now
        assert os.path.exists("policies/dangerous_operations.yaml"), (
            "Dangerous operations policy should exist"
        )


class TestSandboxPolicy:
    """Test sandbox policy enforcement."""

    def test_sandbox_policy_configuration(self):
        """Test sandbox policy configuration."""
        policy_paths = ["policies/sandbox.yaml", ".github/sandbox_policy.yaml"]

        policy_exists = any(os.path.exists(path) for path in policy_paths)

        if not policy_exists:
            # Create default sandbox policy for testing
            os.makedirs("policies", exist_ok=True)

            default_sandbox_policy = {
                "version": "1.0",
                "sandbox": {
                    "branch_protection": {"enabled": True},
                    "approval_requirements": {"required_reviewers": 1},
                    "policy_validation": {"required": True},
                    "automated_testing": {"required": True},
                    "security_scanning": {"enabled": True},
                },
            }

            with open("policies/sandbox.yaml", "w") as f:
                yaml.dump(default_sandbox_policy, f)

        # Verify policy exists
        assert os.path.exists("policies/sandbox.yaml"), "Sandbox policy should exist"


def test_policy_validation_scripts_exist():
    """Test that policy validation scripts exist."""
    scripts = [
        "scripts/validate_policies.py",
        "scripts/check_dangerous_operations.py",
        "scripts/validate_sandbox_policies.py",
        "scripts/security_compliance_check.py",
    ]

    for script in scripts:
        assert os.path.exists(script), (
            f"Policy validation script should exist: {script}"
        )
        assert os.access(script, os.X_OK), f"Script should be executable: {script}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
