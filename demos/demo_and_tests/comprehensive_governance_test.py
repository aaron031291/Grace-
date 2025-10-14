"""
Comprehensive governance integration tests (pytest-friendly)

This file runs a set of lightweight integration checks under pytest.
Asynchronous helpers are executed via asyncio.run() so pytest does not need
an async plugin for these tests.
"""

import os
import asyncio


def test_path_drift_fix():
    """Ensure immutable audit symlink exists and imports work."""
    audit_symlink = "grace/audit/immutable_logs.py"
    assert os.path.islink(audit_symlink), "Symlink grace/audit/immutable_logs.py not found"

    target = os.readlink(audit_symlink)
    print(f"Symlink points to: {target}")

    try:
        from grace.audit import ImmutableLogs, CoreImmutableLogs, ImmutableLogService  # noqa: F401
    except ImportError as e:
        assert False, f"Import from grace.audit failed: {e}"


def test_enforcement_hooks():
    """Check governance enforcement hooks and consensus schema exist."""
    try:
        from grace.governance.constitutional_decorator import (
            constitutional_check,
            trust_middleware,
            ContradictionService,
            uniform_envelope_builder,
        )  # noqa: F401

        from grace.governance.quorum_consensus_schema import QuorumConsensusEngine, ConsensusProposal  # noqa: F401
    except ImportError as e:
        assert False, f"Enforcement hooks or quorum schema missing: {e}"


def test_ci_integration():
    """Verify CI integration mentions constitutional validation."""
    try:
        from grace.policy.ci_integration import check_policies  # noqa: F401
    except ImportError as e:
        assert False, f"CI integration module missing: {e}"

    with open("grace/policy/ci_integration.py", "r") as f:
        ci_content = f.read()

    assert "constitutional_validator" in ci_content, "Constitutional validation not found in CI"


def test_golden_path_audit():
    """Run async golden-path audit helpers synchronously via asyncio.run()."""

    async def _run():
        try:
            from grace.audit.golden_path_auditor import append_audit, verify_audit  # noqa: F401

            audit_id = await append_audit(
                operation_type="test_golden_path",
                operation_data={"test": "integration"},
                user_id="test_user",
            )

            verification = await verify_audit(audit_id)
            assert verification.get("verified"), "Audit verification failed"

        except Exception:
            # Re-raise so pytest records the failure
            raise

    asyncio.run(_run())


def test_api_integration():
    """Ensure API service integrates golden-path auditor."""
    with open("grace/api/api_service.py", "r") as f:
        api_content = f.read()

    assert "golden_path_auditor" in api_content, "API service missing golden_path_auditor import"
    assert "append_audit" in api_content, "API endpoints lack append_audit() calls"


def test_documentation_updates():
    """Check README contains key governance documentation entries."""
    with open("README.md", "r") as f:
        readme_content = f.read()

    checks = [
        ("grace/audit/" in readme_content, "Audit path mapping documented"),
        ("constitutional_check" in readme_content, "Constitutional decorator documented"),
        ("11-kernel structure" in readme_content, "Kernel structure documented"),
        ("append_audit" in readme_content, "Golden path audit documented"),
        ("⚠️ Deprecation Notice" in readme_content, "Layer deprecation notice added"),
    ]

    missing = [desc for ok, desc in checks if not ok]
    assert not missing, f"Documentation missing: {missing}"
