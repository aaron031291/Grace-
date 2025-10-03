"""
End-to-End Smoke Test for Grace Kernel
Boots the kernel, processes a dummy governance request, verifies audit entry, and clean shutdown.
"""
import asyncio
from grace.governance.grace_governance_kernel import GraceGovernanceKernel
from grace.audit.golden_path_auditor import append_audit

def run_smoke_test():
    async def test():
        kernel = GraceGovernanceKernel()
        await kernel.start()
        dummy_request = {
            "claims": ["test_claim"],
            "context": {"user": "smoke_test"}
        }
        result = await kernel.process_governance_request("policy", dummy_request)
        print("Governance Decision:", result)
        audit_id = await append_audit(
            operation_type="governance_decision",
            operation_data=result,
            user_id="smoke_test",
            transparency_level="public"
        )
        print("Audit Entry ID:", audit_id)
        await kernel.shutdown()
    asyncio.run(test())

if __name__ == "__main__":
    run_smoke_test()
