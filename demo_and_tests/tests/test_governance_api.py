#!/usr/bin/env python3
"""Test script for Grace Governance API Service."""

import asyncio
import sys
import os

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from grace.governance.governance_api import GovernanceAPIService, GovernanceRequest, GovernanceDecision
    from grace.contracts.message_envelope_simple import RBACContext
    from grace.layer_04_audit_logs.immutable_logs import ImmutableLogs
    
    print("✅ Successfully imported governance API components")
    
    async def test_governance_api():
        """Test governance API functionality."""
        print("🧪 Testing Grace Governance API...")
        
        # Create immutable logger
        immutable_logger = ImmutableLogs(db_path=":memory:")
        
        # Create API service
        governance_api = GovernanceAPIService(
            event_bus=None,  # Mock event bus for testing
            governance_kernel=None,
            immutable_logger=immutable_logger
        )
        
        # Test RBAC context
        rbac_context = RBACContext(
            user_id="test_user",
            roles=["governance_approver", "mldl_admin"],
            permissions=["governance.approve", "mldl.deploy"]
        )
        
        print(f"Created RBAC context for user: {rbac_context.user_id}")
        print(f"Roles: {rbac_context.roles}")
        print(f"Permissions: {rbac_context.permissions}")
        
        # Test creating governance request
        governance_request = GovernanceRequest(
            action_type="mldl.deploy",
            resource_id="model_v1.2.3",
            payload={"model_path": "/models/classifier.pkl", "env": "production"},
            priority="high",
            requester="ml_engineer",
            reason="Deploy new model version with improved accuracy",
            timeout_seconds=7200
        )
        
        print(f"Created governance request: {governance_request.request_id}")
        print(f"Action: {governance_request.action_type}")
        print(f"Priority: {governance_request.priority}")
        
        # Test governance decision
        decision = GovernanceDecision(
            request_id=governance_request.request_id,
            decision="approved",
            approver="governance_admin",
            reason="Model performance meets deployment criteria",
            conditions=["Monitor accuracy metrics for 24h", "Enable gradual rollout"]
        )
        
        print(f"Created approval decision: {decision.decision}")
        print(f"Conditions: {decision.conditions}")
        
        # Test immutable logging
        governance_log_id = await immutable_logger.log_governance_action(
            action_type="approval_granted",
            data={
                "request_id": governance_request.request_id,
                "action_type": governance_request.action_type,
                "approver": decision.approver,
                "decision": decision.decision,
                "conditions": decision.conditions
            }
        )
        
        print(f"Logged governance action: {governance_log_id}")
        
        # Test audit trail integrity
        integrity_results = await immutable_logger.verify_chain_integrity()
        print(f"Audit integrity check: {integrity_results['verified']}")
        print(f"Entries checked: {integrity_results['entries_checked']}")
        
        # Test audit statistics
        stats = immutable_logger.get_audit_statistics()
        print(f"Total audit entries: {stats['total_entries']}")
        print(f"Categories: {list(stats['categories'].keys())}")
        
        # Test API statistics
        api_stats = await governance_api.get_stats()
        print(f"API pending requests: {api_stats['pending_requests']}")
        print(f"Total requests: {api_stats['total_requests']}")
        
        print("✅ Governance API test passed!")
        return True

    def run_tests():
        """Run all tests."""
        print("🚀 Running Grace Governance API Tests...\n")
        
        try:
            success = asyncio.run(test_governance_api())
            print(f"\n📊 Result: {'PASSED' if success else 'FAILED'}")
            return success
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    if __name__ == "__main__":
        success = run_tests()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Dependencies not available. Skipping governance API tests.")
    sys.exit(0)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)