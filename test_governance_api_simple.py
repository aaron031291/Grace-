#!/usr/bin/env python3
"""Simple test script for Grace Governance API Service."""

import asyncio
import sys
import os

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_governance_api_basic():
    """Test basic governance API functionality."""
    print("🧪 Testing Grace Governance API (Basic)...")
    
    try:
        from grace.contracts.message_envelope_simple import RBACContext
        
        # Test RBAC context
        rbac_context = RBACContext(
            user_id="test_user",
            roles=["governance_approver", "mldl_admin"],
            permissions=["governance.approve", "mldl.deploy"]
        )
        
        print(f"✅ Created RBAC context for user: {rbac_context.user_id}")
        print(f"   Roles: {rbac_context.roles}")
        print(f"   Permissions: {rbac_context.permissions}")
        
        # Test RBAC check methods
        from grace.governance.governance_api import RBACCheck
        
        has_permission = RBACCheck.check_permission(rbac_context, "governance.approve")
        print(f"✅ Has governance.approve permission: {has_permission}")
        
        # Test governance API service creation
        from grace.governance.governance_api import GovernanceAPIService
        
        api_service = GovernanceAPIService()
        print(f"✅ Created governance API service")
        
        # Test RBAC policies
        can_submit = api_service._can_submit_request("mldl.deploy", rbac_context)
        print(f"✅ Can submit MLDL deploy request: {can_submit}")
        
        can_approve = api_service._can_approve_request("mldl.deploy", rbac_context)
        print(f"✅ Can approve MLDL deploy request: {can_approve}")
        
        # Test required approvals calculation
        required_approvals = api_service._get_required_approvals("mldl.deploy")
        print(f"✅ Required approvals for MLDL deploy: {required_approvals}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_immutable_logs():
    """Test immutable logs functionality."""
    print("\n🧪 Testing Immutable Logs...")
    
    try:
        from grace.layer_04_audit_logs.immutable_logs import ImmutableLogs
        
        # Create immutable logger
        logger = ImmutableLogs(db_path=":memory:")
        print("✅ Created immutable logger")
        
        # Test statistics
        stats = logger.get_audit_statistics()
        print(f"✅ Initial statistics: {stats['total_entries']} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("🚀 Running Grace Governance API Tests...\n")
    
    tests = [
        test_governance_api_basic,
        test_immutable_logs
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
    
    print(f"\n📊 Result: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)