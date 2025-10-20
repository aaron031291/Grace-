"""
Master test suite - Tests entire Grace system end-to-end
"""

import asyncio

print("=" * 80)
print("GRACE COMPLETE SYSTEM TEST")
print("Testing all components in integrated workflow")
print("=" * 80)

async def run_all_tests():
    test_results = []
    
    # Test 1: Run integration example
    print("\n[1/6] Running complete system example...")
    try:
        from examples.complete_system_example import main as example_main
        await example_main()
        test_results.append(("Complete System Example", "✅ PASSED"))
    except Exception as e:
        test_results.append(("Complete System Example", f"❌ FAILED: {e}"))
    
    # Test 2: Clarity Framework
    print("\n[2/6] Running Clarity Framework tests...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "test_clarity_framework_complete.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            test_results.append(("Clarity Framework", "✅ PASSED"))
        else:
            test_results.append(("Clarity Framework", f"❌ FAILED: {result.stderr}"))
    except Exception as e:
        test_results.append(("Clarity Framework", f"❌ FAILED: {e}"))
    
    # Test 3: Orchestration
    print("\n[3/6] Running Orchestration tests...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "test_orchestration_complete.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            test_results.append(("Orchestration", "✅ PASSED"))
        else:
            test_results.append(("Orchestration", f"❌ FAILED"))
    except Exception as e:
        test_results.append(("Orchestration", f"❌ FAILED: {e}"))
    
    # Test 4: Swarm & Transcendence
    print("\n[4/6] Running Swarm & Transcendence tests...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "test_swarm_transcendence.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            test_results.append(("Swarm & Transcendence", "✅ PASSED"))
        else:
            test_results.append(("Swarm & Transcendence", f"❌ FAILED"))
    except Exception as e:
        test_results.append(("Swarm & Transcendence", f"❌ FAILED: {e}"))
    
    # Test 5: Unified System
    print("\n[5/6] Running Unified System tests...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "test_unified_system.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            test_results.append(("Unified System", "✅ PASSED"))
        else:
            test_results.append(("Unified System", f"❌ FAILED"))
    except Exception as e:
        test_results.append(("Unified System", f"❌ FAILED: {e}"))
    
    # Test 6: Full Integration
    print("\n[6/6] Running Full Integration tests...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "test_integration_full.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            test_results.append(("Full Integration", "✅ PASSED"))
        else:
            test_results.append(("Full Integration", f"❌ FAILED"))
    except Exception as e:
        test_results.append(("Full Integration", f"❌ FAILED: {e}"))
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if "✅" in result)
    total = len(test_results)
    
    for test_name, result in test_results:
        print(f"\n{test_name:.<40} {result}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Grace system is fully operational.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    exit(exit_code)
