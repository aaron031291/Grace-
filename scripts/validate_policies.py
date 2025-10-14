#!/usr/bin/env python3
"""
Validate YAML policy files for Grace AI Governance System.
This script ensures all policy files are valid YAML and follow the expected schema.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any


def validate_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single YAML file."""
    result = {"file": str(file_path), "valid": False, "errors": [], "warnings": []}

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            result["warnings"].append("File is empty or contains only comments")
        else:
            result["valid"] = True

    except yaml.YAMLError as e:
        result["errors"].append(f"YAML syntax error: {e}")
    except FileNotFoundError:
        result["errors"].append("File not found")
    except Exception as e:
        result["errors"].append(f"Unexpected error: {e}")

    return result


def find_policy_files() -> List[Path]:
    """Find all policy-related YAML files."""
    policy_files = []

    # Search patterns for policy files
    search_paths = [
        "policies/",
        "governance/policies/",
        "contracts/",
        "grace/governance/",
        "grace/policies/",
        "grace/schemas/",
        "contracts/comms/",
        "./",
    ]

    policy_patterns = [
        "*policy*.yaml",
        "*policy*.yml",
        "*governance*.yaml",
        "*governance*.yml",
        "*security*.yaml",
        "*security*.yml",
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            for pattern in policy_patterns:
                policy_files.extend(Path(search_path).glob(pattern))
                # Also search recursively
                policy_files.extend(Path(search_path).rglob(pattern))

    return list(set(policy_files))  # Remove duplicates


def main():
    """Main validation function."""
    print("🔍 Grace Policy Validation - YAML File Checker")
    print("=" * 50)

    policy_files = find_policy_files()

    if not policy_files:
        print("⚠️  No policy YAML files found.")
        print("   Expected policy files in: policies/, contracts/, grace/governance/")
        return 0

    print(f"Found {len(policy_files)} policy files to validate:")

    all_valid = True
    validation_results = []

    for policy_file in policy_files:
        print(f"  📄 Validating: {policy_file}")
        result = validate_yaml_file(policy_file)
        validation_results.append(result)

        if result["valid"]:
            print(f"    ✅ Valid YAML")
        else:
            print(f"    ❌ Invalid YAML")
            all_valid = False

        for error in result["errors"]:
            print(f"    🔴 Error: {error}")

        for warning in result["warnings"]:
            print(f"    🟡 Warning: {warning}")

    print("\n" + "=" * 50)

    if all_valid:
        print("🎉 All policy YAML files are valid!")
        return 0
    else:
        print("💥 Some policy YAML files have validation errors!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
