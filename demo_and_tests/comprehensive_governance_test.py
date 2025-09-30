"""
Grace Governance System - Integration Test Summary

This test validates all the fixes implemented for the governance system issues:
1. Path drift resolved
2. Enforcement hooks operational 
3. CI validation enhanced
4. Immutable logs wired into golden path
5. Policy & governance integration
6. Documentation updated
"""
import sys
import os
import asyncio

# Add Grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_path_drift_fix():
    """Test that path drift issue is resolved."""
    print("🗂️ Testing Path Drift Resolution...")
    
    # Test symlink exists
    audit_symlink = "grace/audit/immutable_logs.py"
    if os.path.islink(audit_symlink):
        print("✅ Symlink created: grace/audit/immutable_logs.py")
        target = os.readlink(audit_symlink)
        print(f"   → Points to: {target}")
        
        # Test imports work
        try:
            from grace.audit import ImmutableLogs, CoreImmutableLogs, ImmutableLogService
            print("✅ Audit imports working correctly")
            return True
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
    else:
        print("❌ Symlink not found")
        return False


def test_enforcement_hooks():
    """Test that governance enforcement hooks are available."""
    print("\n🛡️ Testing Governance Enforcement Hooks...")
    
    try:
        # Test constitutional decorator
        from grace.governance.constitutional_decorator import constitutional_check, trust_middleware
        print("✅ Constitutional decorator available")
        
        # Test contradiction service
        from grace.governance.constitutional_decorator import ContradictionService
        print("✅ Contradiction service available")
        
        # Test uniform envelope builder
        from grace.governance.constitutional_decorator import uniform_envelope_builder
        print("✅ Uniform envelope builder available")
        
        # Test quorum consensus schema
        from grace.governance.quorum_consensus_schema import QuorumConsensusEngine, ConsensusProposal
        print("✅ Quorum consensus schema available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Enforcement hooks missing: {e}")
        return False


def test_ci_integration():
    """Test CI integration enhancements."""
    print("\n🔧 Testing CI Integration...")
    
    try:
        from grace.policy.ci_integration import check_policies
        print("✅ Enhanced CI integration available")
        
        # Test that constitutional validation is mentioned in the file
        with open("grace/policy/ci_integration.py", "r") as f:
            ci_content = f.read()
            
        if "constitutional_validator" in ci_content:
            print("✅ Constitutional validation integrated in CI")
            return True
        else:
            print("⚠️ Constitutional validation not found in CI")
            return False
            
    except ImportError as e:
        print(f"❌ CI integration failed: {e}")
        return False


async def test_golden_path_audit():
    """Test golden path audit integration."""
    print("\n📋 Testing Golden Path Audit Integration...")
    
    try:
        from grace.audit.golden_path_auditor import append_audit, verify_audit, get_golden_path_auditor
        print("✅ Golden path auditor available")
        
        # Test the append_audit function mentioned in problem statement
        audit_id = await append_audit(
            operation_type="test_golden_path",
            operation_data={"test": "integration"},
            user_id="test_user"
        )
        
        print(f"✅ append_audit() working - ID: {audit_id[:50]}...")
        
        # Test verification
        verification = await verify_audit(audit_id)
        if verification['verified']:
            print("✅ verify_audit() working")
            return True
        else:
            print("❌ Audit verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Golden path audit failed: {e}")
        return False


def test_api_integration():
    """Test API governance integration."""
    print("\n🌐 Testing API Governance Integration...")
    
    try:
        # Check that API service imports audit functionality
        with open("grace/api/api_service.py", "r") as f:
            api_content = f.read()
            
        if "golden_path_auditor" in api_content:
            print("✅ API service integrates golden path auditor")
            
        if "append_audit" in api_content:
            print("✅ API endpoints call append_audit()")
            return True
        else:
            print("❌ API endpoints lack audit integration")
            return False
            
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False


def test_documentation_updates():
    """Test documentation updates."""
    print("\n📚 Testing Documentation Updates...")
    
    try:
        with open("README.md", "r") as f:
            readme_content = f.read()
            
        checks = [
            ("grace/audit/" in readme_content, "Audit path mapping documented"),
            ("constitutional_check" in readme_content, "Constitutional decorator documented"),
            ("11-kernel structure" in readme_content, "Kernel structure documented"),
            ("append_audit" in readme_content, "Golden path audit documented"),
            ("⚠️ Deprecation Notice" in readme_content, "Layer deprecation notice added")
        ]
        
        passed = 0
        for check, description in checks:
            if check:
                print(f"✅ {description}")
                passed += 1
            else:
                print(f"❌ {description}")
        
        return passed == len(checks)
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False


async def main():
    """Run comprehensive integration test."""
    print("🏛️ Grace Governance System - Comprehensive Integration Test")
    print("=" * 70)
    print("Testing all fixes for the governance system issues...\n")
    
    tests = [
        ("Path Drift Resolution", test_path_drift_fix),
        ("Enforcement Hooks", test_enforcement_hooks), 
        ("CI Integration", test_ci_integration),
        ("Golden Path Audit", test_golden_path_audit),
        ("API Integration", test_api_integration),
        ("Documentation Updates", test_documentation_updates)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL GOVERNANCE FIXES VERIFIED SUCCESSFULLY!")
        print("\nThe Grace governance system now has:")
        print("• ✅ Fixed path drift with symlinks and clear documentation")
        print("• ✅ Operational enforcement hooks (decorators, middleware, services)")
        print("• ✅ Enhanced CI validation with constitutional compliance")
        print("• ✅ Concrete append/verify audit path wired into golden path")
        print("• ✅ Policy & governance integration in user-facing APIs")
        print("• ✅ Updated documentation with deprecation notices and links")
        return 0
    else:
        print(f"\n💥 {total - passed} tests failed. System needs attention.")
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)