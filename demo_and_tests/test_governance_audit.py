"""
Simple governance test - demonstrates working audit integration.
"""
import asyncio
import sys
import os

# Add Grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.audit.golden_path_auditor import append_audit, verify_audit


async def test_audit_only():
    """Test just the audit functionality without constitutional checks."""
    print("🔍 Testing Grace Audit System...")
    
    # Test basic audit logging
    audit_id = await append_audit(
        operation_type="test_operation",
        operation_data={
            "test": "data",
            "operation": "simple_test",
            "description": "Testing audit functionality"
        },
        user_id="test_user",
        transparency_level="public"
    )
    
    print(f"✅ Audit logged with ID: {audit_id}")
    
    # Test audit verification
    verification = await verify_audit(audit_id)
    print(f"✅ Audit verification: {verification['verified']}")
    
    if verification['verified']:
        print("✅ Audit system working correctly!")
        return True
    else:
        print("❌ Audit verification failed")
        return False


async def test_memory_operations():
    """Test audit logging for memory operations."""
    print("\n📝 Testing Memory Operation Auditing...")
    
    from grace.audit.golden_path_auditor import get_golden_path_auditor
    
    auditor = get_golden_path_auditor()
    
    # Test audit session
    async with auditor.audit_session("test_session_123") as session:
        print("📖 Logging memory read...")
        read_id = await session.log_memory_read({
            "query": "test query",
            "filters": {},
            "results_count": 5
        }, user_id="test_user")
        
        print("✏️ Logging memory write...")
        write_id = await session.log_memory_write({
            "content": "test content",
            "metadata": {"source": "test"},
            "trust_score": 0.8
        }, user_id="test_user")
        
        print("📤 Logging API response...")
        response_id = await session.log_api_response({
            "endpoint": "/api/test",
            "status": "success",
            "response_size": 1024
        }, user_id="test_user")
    
    print(f"✅ Session audit complete: read={read_id}, write={write_id}, response={response_id}")
    return True


async def main():
    """Run all tests."""
    print("🏛️ Grace Governance System - Audit Integration Test")
    print("=" * 60)
    
    try:
        # Test basic audit functionality
        audit_test = await test_audit_only()
        
        # Test memory operation auditing
        memory_test = await test_memory_operations()
        
        if audit_test and memory_test:
            print("\n🎉 All tests passed! Governance audit system is operational.")
            return 0
        else:
            print("\n💥 Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)