#!/usr/bin/env python3
"""
Validate sandbox branch policies for Grace AI Governance System.
Ensures sandbox branch enforcement with required approvals and policy:pass labels.
"""
import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

class SandboxPolicyValidator:
    """Validator for sandbox branch policies."""
    
    def __init__(self):
        self.required_sandbox_policies = [
            "branch_protection",
            "approval_requirements", 
            "policy_validation",
            "automated_testing",
            "security_scanning"
        ]
        
        self.policy_labels = [
            "policy:pass",
            "security:reviewed", 
            "governance:approved",
            "sandbox:safe"
        ]
    
    def validate_sandbox_config(self, config_path: Path) -> Dict[str, Any]:
        """Validate sandbox configuration file."""
        result = {
            "file": str(config_path),
            "valid": False,
            "errors": [],
            "warnings": [],
            "missing_policies": [],
            "compliance_score": 0.0
        }
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            if not config_data:
                result["errors"].append("Empty configuration file")
                return result
            
            # Check for required sandbox policies
            missing_policies = []
            found_policies = 0
            
            sandbox_config = config_data.get('sandbox', {})
            
            for policy in self.required_sandbox_policies:
                if policy in sandbox_config:
                    found_policies += 1
                else:
                    missing_policies.append(policy)
            
            result["missing_policies"] = missing_policies
            result["compliance_score"] = found_policies / len(self.required_sandbox_policies)
            
            # Check branch protection settings
            branch_protection = sandbox_config.get('branch_protection', {})
            if not branch_protection.get('enabled', False):
                result["warnings"].append("Branch protection not enabled")
            
            # Check approval requirements
            approvals = sandbox_config.get('approval_requirements', {})
            required_approvals = approvals.get('required_reviewers', 0)
            if required_approvals < 1:
                result["warnings"].append("No required reviewers configured")
            
            # Check policy validation requirements
            policy_validation = sandbox_config.get('policy_validation', {})
            if not policy_validation.get('required', False):
                result["errors"].append("Policy validation not required for sandbox branches")
            
            result["valid"] = len(result["errors"]) == 0 and result["compliance_score"] >= 0.8
            
        except Exception as e:
            result["errors"].append(f"Failed to parse config: {e}")
            
        return result
    
    def create_default_sandbox_config(self) -> Dict[str, Any]:
        """Create default sandbox configuration."""
        return {
            "version": "1.0",
            "name": "Grace Sandbox Branch Policy",
            "description": "Sandbox branch enforcement with required approvals and policy validation",
            "sandbox": {
                "branch_protection": {
                    "enabled": True,
                    "enforce_admins": True,
                    "required_status_checks": [
                        "policy-check",
                        "security-scan",
                        "governance-validation"
                    ],
                    "strict": True
                },
                "approval_requirements": {
                    "required_reviewers": 2,
                    "dismiss_stale_reviews": True,
                    "require_code_owner_reviews": True,
                    "required_approving_review_count": 1
                },
                "policy_validation": {
                    "required": True,
                    "blocking": True,
                    "checks": [
                        "dangerous_operations",
                        "security_compliance", 
                        "governance_alignment"
                    ]
                },
                "automated_testing": {
                    "required": True,
                    "test_suites": [
                        "unit_tests",
                        "integration_tests",
                        "policy_tests"
                    ],
                    "coverage_threshold": 0.8
                },
                "security_scanning": {
                    "enabled": True,
                    "tools": ["bandit", "semgrep", "safety"],
                    "fail_on": ["high", "critical"]
                },
                "labels": {
                    "required_for_merge": [
                        "policy:pass",
                        "security:reviewed",
                        "governance:approved"
                    ],
                    "auto_apply": [
                        "sandbox:active"
                    ]
                },
                "experiment_limits": {
                    "max_duration_hours": 24,
                    "max_resource_usage": 0.5,
                    "auto_cleanup": True
                }
            }
        }

def main():
    """Main sandbox policy validation function."""
    print("🔒 Grace Sandbox Branch Policy Validator") 
    print("=" * 50)
    
    validator = SandboxPolicyValidator()
    
    # Look for sandbox configuration files
    config_files = []
    
    search_paths = [
        "policies/sandbox.yaml",
        "policies/sandbox.yml",
        "governance/sandbox_policy.yaml",
        "grace/governance/sandbox_config.yaml",
        ".github/sandbox_policy.yaml"
    ]
    
    for config_path in search_paths:
        if os.path.exists(config_path):
            config_files.append(Path(config_path))
    
    if not config_files:
        print("⚠️  No sandbox policy configuration found!")
        print("   Creating default sandbox policy at: policies/sandbox.yaml")
        
        # Create policies directory
        os.makedirs("policies", exist_ok=True)
        
        # Create default sandbox config
        default_config = validator.create_default_sandbox_config()
        with open("policies/sandbox.yaml", 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Default sandbox policy created!")
        return 0
    
    print(f"Found {len(config_files)} sandbox policy files:")
    
    all_valid = True
    
    for config_file in config_files:
        print(f"\n📄 Validating: {config_file}")
        result = validator.validate_sandbox_config(config_file)
        
        if result["valid"]:
            print("  ✅ Valid sandbox policy configuration")
        else:
            print("  ❌ Invalid sandbox policy configuration") 
            all_valid = False
        
        print(f"  📊 Compliance score: {result['compliance_score']:.2%}")
        
        if result["missing_policies"]:
            print(f"  🔴 Missing policies: {', '.join(result['missing_policies'])}")
        
        for error in result["errors"]:
            print(f"  🔴 Error: {error}")
            
        for warning in result["warnings"]:
            print(f"  🟡 Warning: {warning}")
    
    print("\n" + "=" * 50)
    
    if all_valid:
        print("🎉 All sandbox policies are valid!")
        return 0
    else:
        print("💥 Some sandbox policies are invalid!")
        print("   Please review and update your sandbox policy files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())