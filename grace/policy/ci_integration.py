"""
CI integration for Grace policy validation.

Runs policy checks in CI pipeline and enforces policy compliance,
including constitutional validation, envelope schema checks, and governance enforcement.
"""

import sys
import argparse
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from .rules import get_policy_engine
from ..governance.constitutional_validator import ConstitutionalValidator

logger = logging.getLogger(__name__)


def analyze_code_changes(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze code changes for policy violations.

    Args:
        file_paths: List of changed file paths

    Returns:
        List of operations representing the changes
    """
    operations = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            # Deleted file
            operations.append(
                {
                    "type": "file_delete",
                    "file_path": str(path),
                    "user_id": "ci",
                    "user_roles": ["developer"],
                    "user_scopes": [],
                }
            )
            continue

        # Read file content for analysis
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")

            # Check for dangerous patterns
            if path.suffix in [".py", ".js", ".sh", ".bat"]:
                operations.append(
                    {
                        "type": "code_modification",
                        "file_path": str(path),
                        "content": content,
                        "user_id": "ci",
                        "user_roles": ["developer"],
                        "user_scopes": [],
                    }
                )

            # Check for file operations
            if path.suffix in [".sh", ".bat", ".ps1"]:
                operations.append(
                    {
                        "type": "shell_script",
                        "file_path": str(path),
                        "content": content,
                        "user_id": "ci",
                        "user_roles": ["developer"],
                        "user_scopes": [],
                    }
                )

            # Check for configuration files
            if path.name in ["Dockerfile", "docker-compose.yml", "requirements.txt"]:
                operations.append(
                    {
                        "type": "config_change",
                        "file_path": str(path),
                        "content": content,
                        "user_id": "ci",
                        "user_roles": ["developer"],
                        "user_scopes": [],
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    return operations


def check_policies(file_paths: List[str]) -> Dict[str, Any]:
    """
    Check policies against code changes.

    Args:
        file_paths: List of changed file paths

    Returns:
        Policy check results
    """
    policy_engine = get_policy_engine()
    operations = analyze_code_changes(file_paths)

    all_violations = []
    blocked_operations = []
    constitutional_violations = []
    governance_warnings = []
    highest_severity = None

    # Initialize governance validators
    try:
        constitutional_validator = ConstitutionalValidator()
    except Exception as e:
        logger.warning(f"Could not initialize constitutional validator: {e}")
        constitutional_validator = None

    for operation in operations:
        # Standard policy check
        result = policy_engine.evaluate_operation(operation)

        if result["violations"]:
            all_violations.extend(result["violations"])

            if not result["allowed"]:
                blocked_operations.append(operation)

            # Track highest severity
            if result["severity"]:
                if highest_severity is None or policy_engine._severity_level(
                    result["severity"]
                ) > policy_engine._severity_level(highest_severity):
                    highest_severity = result["severity"]

        # Constitutional validation for sensitive operations
        if constitutional_validator and operation.get("type") in [
            "code_modification",
            "config_change",
            "file_delete",
        ]:
            try:
                # Create a synchronous version for CI
                validation_result = asyncio.run(
                    constitutional_validator.validate_against_constitution(
                        operation, {"transparency_level": "democratic_oversight"}
                    )
                )

                if not validation_result.is_valid:
                    constitutional_violations.extend(
                        [vars(v) for v in validation_result.violations]
                    )

            except Exception as e:
                logger.warning(
                    f"Constitutional validation failed for {operation.get('file_path')}: {e}"
                )

        # Check for governance integration in API files
        if operation.get("type") == "code_modification":
            file_path = operation.get("file_path", "")
            content = operation.get("content", "")

            if ("grace/api" in file_path or "grace/interface" in file_path) and content:
                if "FastAPI" in content or "@app." in content:
                    if (
                        "PolicyEnforcementMiddleware" not in content
                        and "constitutional_check" not in content
                    ):
                        governance_warnings.append(
                            {
                                "file": file_path,
                                "warning": "API endpoint may lack governance enforcement",
                            }
                        )

            # Check for missing constitutional decorators on sensitive functions
            if any(
                pattern in content.lower()
                for pattern in ["def delete", "def remove", "def admin", "def execute"]
            ):
                if "@constitutional_check" not in content:
                    governance_warnings.append(
                        {
                            "file": file_path,
                            "warning": "Sensitive function may lack constitutional decorator",
                        }
                    )

    return {
        "passed": len(blocked_operations) == 0 and len(constitutional_violations) == 0,
        "total_operations": len(operations),
        "violations": all_violations,
        "constitutional_violations": constitutional_violations,
        "governance_warnings": governance_warnings,
        "blocked_operations": blocked_operations,
        "highest_severity": highest_severity,
        "summary": {
            "total_violations": len(all_violations),
            "constitutional_violations": len(constitutional_violations),
            "governance_warnings": len(governance_warnings),
            "blocked_count": len(blocked_operations),
            "policy_file": policy_engine.policy_file,
        },
    }


def create_policy_report(results: Dict[str, Any], output_file: str = None) -> str:
    """Create a formatted policy report."""
    report_lines = []

    report_lines.append("# Grace Policy Validation Report")
    report_lines.append("")

    if results["passed"]:
        report_lines.append("✅ **PASSED** - No policy violations found")
    else:
        report_lines.append("❌ **FAILED** - Policy violations detected")

    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- Total operations analyzed: {results['total_operations']}")
    report_lines.append(
        f"- Policy violations: {results['summary']['total_violations']}"
    )
    report_lines.append(
        f"- Constitutional violations: {results['summary'].get('constitutional_violations', 0)}"
    )
    report_lines.append(
        f"- Governance warnings: {results['summary'].get('governance_warnings', 0)}"
    )
    report_lines.append(f"- Blocked operations: {results['summary']['blocked_count']}")
    report_lines.append(f"- Highest severity: {results['highest_severity'] or 'None'}")

    if results["violations"]:
        report_lines.append("")
        report_lines.append("## Policy Violations")

        for violation in results["violations"]:
            report_lines.append(
                f"### {violation['rule_name']} ({violation['severity']})"
            )
            report_lines.append(f"**Description:** {violation['description']}")
            report_lines.append(f"**Actions:** {', '.join(violation['actions'])}")
            report_lines.append("")

    if results.get("constitutional_violations"):
        report_lines.append("")
        report_lines.append("## 🏛️ Constitutional Violations")

        for violation in results["constitutional_violations"]:
            report_lines.append(
                f"### {violation.get('principle', 'Unknown')} ({violation.get('severity', 'unknown')})"
            )
            report_lines.append(
                f"**Description:** {violation.get('description', 'No description')}"
            )
            if violation.get("recommendation"):
                report_lines.append(
                    f"**Recommendation:** {violation['recommendation']}"
                )
            report_lines.append("")

    if results.get("governance_warnings"):
        report_lines.append("")
        report_lines.append("## ⚠️ Governance Warnings")

        for warning in results["governance_warnings"]:
            report_lines.append(f"- **{warning['file']}**: {warning['warning']}")
        report_lines.append("")

    if results["blocked_operations"]:
        report_lines.append("")
        report_lines.append("## Blocked Operations")

        for operation in results["blocked_operations"]:
            report_lines.append(
                f"- **{operation['type']}**: {operation.get('file_path', 'N/A')}"
            )

    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"Policy file: `{results['summary']['policy_file']}`")
    report_lines.append("")
    report_lines.append(
        "This report includes constitutional compliance checks and governance enforcement validation."
    )

    report = "\n".join(report_lines)

    if output_file:
        Path(output_file).write_text(report)
        logger.info(f"Policy report written to {output_file}")

    return report


def main():
    """Main CLI entry point for policy validation."""
    parser = argparse.ArgumentParser(description="Grace Policy Validation")
    parser.add_argument("files", nargs="*", help="Files to check (or read from stdin)")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Get file list
    if args.files:
        file_paths = args.files
    else:
        # Read from stdin (useful for git hooks)
        file_paths = []
        for line in sys.stdin:
            file_path = line.strip()
            if file_path:
                file_paths.append(file_path)

    if not file_paths:
        logger.error("No files to check")
        sys.exit(1)

    logger.info(f"Checking {len(file_paths)} files for policy violations")

    # Run policy checks
    results = check_policies(file_paths)

    # Output results
    if args.json:
        output = json.dumps(results, indent=2)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)
    else:
        report = create_policy_report(results, args.output)
        if not args.output:
            print(report)

    # Exit with appropriate code
    if results["passed"]:
        logger.info("Policy validation PASSED")
        sys.exit(0)
    else:
        logger.error("Policy validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
