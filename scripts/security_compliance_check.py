#!/usr/bin/env python3
"""
Security policy compliance checker for Grace AI Governance System.
Validates security configurations and compliance with governance policies.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any


class SecurityComplianceChecker:
    """Checker for security policy compliance."""

    def __init__(self):
        self.required_security_controls = [
            "authentication",
            "authorization",
            "encryption",
            "audit_logging",
            "access_control",
            "data_protection",
        ]

        self.critical_settings = [
            "jwt_validation",
            "rbac_enforcement",
            "pii_protection",
            "audit_retention",
            "encryption_at_rest",
            "transport_security",
        ]

    def check_security_config(self, config_path: Path) -> Dict[str, Any]:
        """Check security configuration compliance."""
        result = {
            "file": str(config_path),
            "compliant": False,
            "security_score": 0.0,
            "missing_controls": [],
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                result["critical_issues"].append("Empty security configuration")
                return result

            # Check required security controls
            found_controls = 0
            missing_controls = []

            for control in self.required_security_controls:
                if control in config_data or self._find_control_variant(
                    control, config_data
                ):
                    found_controls += 1
                else:
                    missing_controls.append(control)

            result["missing_controls"] = missing_controls
            result["security_score"] = found_controls / len(
                self.required_security_controls
            )

            # Check critical security settings
            critical_issues = []

            # JWT validation
            auth_config = config_data.get("auth", {})
            jwt_validation = auth_config.get("token_validation", {})
            if not jwt_validation.get("verify_signature", False):
                critical_issues.append("JWT signature verification disabled")
            if not jwt_validation.get("verify_expiry", False):
                critical_issues.append("JWT expiry verification disabled")

            # RBAC enforcement
            rbac_config = config_data.get("rbac", {})
            if rbac_config.get("enforcement") != "deny-by-default":
                critical_issues.append("RBAC not set to deny-by-default")

            # PII protection
            privacy_config = config_data.get("privacy", {})
            if privacy_config.get("on_pii_without_consent") != "block":
                critical_issues.append("PII access without consent not blocked")

            # Audit logging
            audit_config = config_data.get("audit", {})
            if not audit_config.get("log_all_governance", False):
                critical_issues.append("Governance actions not fully audited")

            # Encryption
            encryption_config = config_data.get("encryption", {})
            if not encryption_config.get("payload_sensitive", False):
                result["warnings"].append("Sensitive payload encryption not enforced")

            result["critical_issues"] = critical_issues

            # Generate recommendations
            recommendations = []
            if result["security_score"] < 0.8:
                recommendations.append("Implement missing security controls")
            if critical_issues:
                recommendations.append("Address critical security issues immediately")
            if result["security_score"] < 1.0:
                recommendations.append("Consider additional security hardening")

            result["recommendations"] = recommendations
            result["compliant"] = (
                len(critical_issues) == 0 and result["security_score"] >= 0.8
            )

        except Exception as e:
            result["critical_issues"].append(f"Failed to parse security config: {e}")

        return result

    def _find_control_variant(self, control: str, config: Dict[str, Any]) -> bool:
        """Find control variants in configuration."""
        variants = {
            "authentication": ["auth", "authentication", "login"],
            "authorization": ["authz", "authorization", "rbac"],
            "encryption": ["crypto", "encryption", "security"],
            "audit_logging": ["audit", "logging", "logs"],
            "access_control": ["access", "permissions", "rbac"],
            "data_protection": ["privacy", "protection", "pii"],
        }

        for variant in variants.get(control, []):
            if variant in config:
                return True
        return False

    def create_security_recommendations(self) -> Dict[str, Any]:
        """Create security recommendations document."""
        return {
            "version": "1.0",
            "name": "Grace Security Compliance Recommendations",
            "recommendations": {
                "authentication": {
                    "priority": "critical",
                    "description": "Implement strong JWT authentication",
                    "requirements": [
                        "Enable signature verification",
                        "Enable expiry verification",
                        "Set reasonable clock skew",
                        "Use strong signing algorithms",
                    ],
                },
                "authorization": {
                    "priority": "critical",
                    "description": "Implement deny-by-default RBAC",
                    "requirements": [
                        "Set enforcement to deny-by-default",
                        "Define clear role hierarchies",
                        "Implement principle of least privilege",
                        "Regular access reviews",
                    ],
                },
                "encryption": {
                    "priority": "high",
                    "description": "Encrypt sensitive data",
                    "requirements": [
                        "Enable payload encryption for sensitive data",
                        "Use strong encryption algorithms",
                        "Implement key rotation",
                        "Secure key management",
                    ],
                },
                "audit_logging": {
                    "priority": "high",
                    "description": "Comprehensive audit logging",
                    "requirements": [
                        "Log all governance actions",
                        "Log PII access attempts",
                        "Log security failures",
                        "Adequate retention periods",
                    ],
                },
            },
        }


def main():
    """Main security compliance checker."""
    print("🔐 Grace Security Policy Compliance Checker")
    print("=" * 50)

    checker = SecurityComplianceChecker()

    # Look for security configuration files
    security_files = []

    search_paths = [
        "policies/security.yaml",
        "contracts/comms/security.yaml",
        "governance/security_policy.yaml",
        "grace/governance/security.yaml",
        "config/security.yaml",
    ]

    for security_path in search_paths:
        if os.path.exists(security_path):
            security_files.append(Path(security_path))

    if not security_files:
        print("⚠️  No security configuration files found!")
        print("   Please ensure security.yaml exists in one of the expected locations.")

        # Create recommendations file
        os.makedirs("policies", exist_ok=True)
        recommendations = checker.create_security_recommendations()
        with open("policies/security_recommendations.yaml", "w") as f:
            yaml.dump(recommendations, f, default_flow_style=False, sort_keys=False)

        print(
            "📋 Security recommendations created at: policies/security_recommendations.yaml"
        )
        return 1

    print(f"Found {len(security_files)} security configuration files:")

    all_compliant = True
    overall_score = 0.0

    for security_file in security_files:
        print(f"\n📄 Checking: {security_file}")
        result = checker.check_security_config(security_file)

        if result["compliant"]:
            print("  ✅ Security compliant")
        else:
            print("  ❌ Security non-compliant")
            all_compliant = False

        print(f"  📊 Security score: {result['security_score']:.2%}")
        overall_score += result["security_score"]

        if result["missing_controls"]:
            print(f"  🔴 Missing controls: {', '.join(result['missing_controls'])}")

        for issue in result["critical_issues"]:
            print(f"  🚨 Critical issue: {issue}")

        for warning in result["warnings"]:
            print(f"  🟡 Warning: {warning}")

        for rec in result["recommendations"]:
            print(f"  💡 Recommendation: {rec}")

    if security_files:
        overall_score /= len(security_files)

    print("\n" + "=" * 50)
    print(f"📊 Overall security score: {overall_score:.2%}")

    if all_compliant:
        print("🎉 All security configurations are compliant!")
        return 0
    else:
        print("💥 Some security configurations are non-compliant!")
        print("   Please review and address the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
