#!/usr/bin/env python3
"""
Check dangerous operations policy compliance for Grace AI Governance System.
Validates that dangerous operations (file I/O, code execution, network access) are properly governed.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any


class DangerousOperationsChecker:
    """Checker for dangerous operations policies."""

    def __init__(self):
        self.dangerous_operations = [
            "file_io",
            "code_execution",
            "network_access",
            "system_commands",
            "database_operations",
            "external_api_calls",
        ]

        self.required_policies = [
            "authentication_required",
            "authorization_required",
            "audit_logging",
            "approval_workflow",
            "sandbox_enforcement",
        ]

    def check_policy_file(self, policy_path: Path) -> Dict[str, Any]:
        """Check a policy file for dangerous operations coverage."""
        result = {
            "file": str(policy_path),
            "compliant": False,
            "covered_operations": [],
            "missing_operations": [],
            "policy_violations": [],
            "warnings": [],
        }

        try:
            with open(policy_path, "r") as f:
                policy_data = yaml.safe_load(f)

            if not policy_data:
                result["warnings"].append("Empty policy file")
                return result

            # Check for dangerous operations coverage
            covered_ops = []

            # Look for operations in various policy structure formats
            if "operations" in policy_data:
                covered_ops.extend(policy_data["operations"].keys())
            elif "rules" in policy_data:
                for rule in policy_data["rules"]:
                    if isinstance(rule, dict) and "operation" in rule:
                        covered_ops.append(rule["operation"])
            elif "dangerous_operations" in policy_data:
                covered_ops.extend(policy_data["dangerous_operations"].keys())

            result["covered_operations"] = covered_ops
            result["missing_operations"] = [
                op for op in self.dangerous_operations if op not in covered_ops
            ]

            # Check policy enforcement requirements
            violations = []
            if "enforcement" not in policy_data:
                violations.append("Missing enforcement configuration")
            if "approval_required" not in policy_data:
                violations.append("Missing approval requirements")
            if "audit" not in policy_data:
                violations.append("Missing audit requirements")

            result["policy_violations"] = violations
            result["compliant"] = (
                len(result["missing_operations"]) == 0 and len(violations) == 0
            )

        except Exception as e:
            result["warnings"].append(f"Failed to parse policy file: {e}")

        return result

    def create_default_policy(self) -> Dict[str, Any]:
        """Create a default dangerous operations policy."""
        return {
            "version": "1.0",
            "name": "Grace Dangerous Operations Policy",
            "description": "Policy governing dangerous operations in Grace AI system",
            "enforcement": "strict",
            "approval_required": True,
            "audit": {"enabled": True, "retention_days": 365, "include_payload": False},
            "dangerous_operations": {
                "file_io": {
                    "allowed": False,
                    "exceptions": ["config_read", "log_write"],
                    "approval_level": "admin",
                    "sandbox_only": True,
                },
                "code_execution": {
                    "allowed": False,
                    "exceptions": ["approved_scripts"],
                    "approval_level": "security_admin",
                    "sandbox_only": True,
                },
                "network_access": {
                    "allowed": True,
                    "restrictions": ["internal_only"],
                    "approval_level": "operator",
                    "audit_required": True,
                },
                "system_commands": {
                    "allowed": False,
                    "exceptions": [],
                    "approval_level": "system_admin",
                    "sandbox_only": True,
                },
                "database_operations": {
                    "allowed": True,
                    "restrictions": ["read_only", "approved_tables"],
                    "approval_level": "data_admin",
                    "audit_required": True,
                },
                "external_api_calls": {
                    "allowed": True,
                    "restrictions": ["approved_endpoints"],
                    "approval_level": "operator",
                    "rate_limit": "100/hour",
                },
            },
        }


def main():
    """Main dangerous operations policy checker."""
    print("🚨 Grace Dangerous Operations Policy Checker")
    print("=" * 50)

    checker = DangerousOperationsChecker()

    # Look for existing dangerous operations policy files
    policy_files = []

    search_paths = [
        "policies/dangerous_operations.yaml",
        "policies/dangerous_operations.yml",
        "governance/dangerous_operations.yaml",
        "grace/governance/dangerous_operations.yaml",
    ]

    for policy_path in search_paths:
        if os.path.exists(policy_path):
            policy_files.append(Path(policy_path))

    if not policy_files:
        print("⚠️  No dangerous operations policy files found!")
        print("   Creating default policy at: policies/dangerous_operations.yaml")

        # Create policies directory if it doesn't exist
        os.makedirs("policies", exist_ok=True)

        # Create default policy
        default_policy = checker.create_default_policy()
        with open("policies/dangerous_operations.yaml", "w") as f:
            yaml.dump(default_policy, f, default_flow_style=False, sort_keys=False)

        print("✅ Default dangerous operations policy created!")
        return 0

    print(f"Found {len(policy_files)} dangerous operations policy files:")

    all_compliant = True

    for policy_file in policy_files:
        print(f"\n📄 Checking: {policy_file}")
        result = checker.check_policy_file(policy_file)

        if result["compliant"]:
            print("  ✅ Compliant with dangerous operations policy")
        else:
            print("  ❌ Non-compliant with dangerous operations policy")
            all_compliant = False

        if result["covered_operations"]:
            print(f"  🛡️  Covered operations: {', '.join(result['covered_operations'])}")

        if result["missing_operations"]:
            print(f"  🔴 Missing operations: {', '.join(result['missing_operations'])}")

        for violation in result["policy_violations"]:
            print(f"  🔴 Policy violation: {violation}")

        for warning in result["warnings"]:
            print(f"  🟡 Warning: {warning}")

    print("\n" + "=" * 50)

    if all_compliant:
        print("🎉 All dangerous operations policies are compliant!")
        return 0
    else:
        print("💥 Some dangerous operations policies are non-compliant!")
        print("   Please review and update your policy files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
