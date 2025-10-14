"""
Basic test to validate Grace Governance Kernel integration.
"""

import asyncio
import sys
import os

# Add the project root to Python path (go up 2 directories from tests to root)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from grace.governance.grace_governance_kernel import GraceGovernanceKernel


async def test_governance_kernel():
    """Test basic governance kernel functionality."""
    print("Testing Grace Governance Kernel...")

    try:
        # Initialize kernel
        kernel = GraceGovernanceKernel()

        # Test initialization
        print("1. Initializing kernel...")
        await kernel.initialize()
        print("   ✓ Initialization successful")

        # Test startup
        print("2. Starting kernel...")
        await kernel.start()
        print("   ✓ Startup successful")

        # Test status
        print("3. Checking system status...")
        status = kernel.get_system_status()
        print(f"   ✓ Status: {status['status']}")
        print(f"   ✓ Components: {len(status['components'])} initialized")

        # Test metrics
        print("4. Getting governance metrics...")
        metrics = kernel.get_governance_metrics()
        print(f"   ✓ Metrics collected from {len(metrics)} systems")

        # Test governance request
        print("5. Processing governance request...")
        result = await kernel.process_governance_request(
            "claim",
            {
                "claims": [
                    {
                        "id": "test_claim_001",
                        "statement": "This is a test claim for validation",
                        "sources": [
                            {"uri": "https://test.example.com", "credibility": 0.7}
                        ],
                        "evidence": [{"type": "doc", "pointer": "test_evidence.pdf"}],
                        "confidence": 0.8,
                        "logical_chain": [{"step": "Test logical reasoning"}],
                    }
                ],
                "context": {"decision_type": "claim", "test_mode": True},
            },
        )
        print(f"   ✓ Governance decision: {result.get('outcome', 'unknown')}")

        # Test shutdown
        print("6. Shutting down kernel...")
        await kernel.shutdown()
        print("   ✓ Shutdown successful")

        print("\nAll tests passed! 🎉")
        print("Grace Governance Kernel is working correctly.")

        return

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Test failed with error: {e}"


if __name__ == "__main__":
    success = asyncio.run(test_governance_kernel())
    sys.exit(0 if success else 1)
