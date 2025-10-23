"""
Fix critical implementation gaps
Run this to see what actually needs to be done
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_entry_points():
    """Check if entry points work"""
    print("\n🔍 Checking Entry Points...")
    
    issues = []
    
    # Check unified service
    try:
        from grace.core.unified_service import create_unified_app
        app = create_unified_app()
        print("  ✓ create_unified_app() works")
    except Exception as e:
        issues.append(f"create_unified_app() fails: {e}")
        print(f"  ✗ create_unified_app() fails: {e}")
    
    # Check demos
    try:
        from grace.demo import demo_multi_os_kernel
        print("  ✓ demo imports work")
    except Exception as e:
        issues.append(f"demo imports fail: {e}")
        print(f"  ✗ demo imports fail: {e}")
    
    return issues


def check_event_usage():
    """Check if GraceEvent is actually used"""
    print("\n🔍 Checking Event Usage...")
    
    issues = []
    
    # Search for dict-based event publishing
    grace_dir = Path("grace")
    dict_publishes = []
    
    for py_file in grace_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            if 'bus.publish({' in content or "bus.publish({" in content:
                dict_publishes.append(str(py_file))
        except:
            pass
    
    if dict_publishes:
        issues.append(f"Found {len(dict_publishes)} files still using dict events")
        print(f"  ✗ Found {len(dict_publishes)} files using dict-based events:")
        for file in dict_publishes[:5]:
            print(f"     - {file}")
    else:
        print("  ✓ No dict-based event publishing found")
    
    return issues


def check_governance_api():
    """Check if async governance methods exist"""
    print("\n🔍 Checking Governance API...")
    
    issues = []
    
    try:
        from grace.governance.engine import GovernanceEngine
        import inspect
        
        engine = GovernanceEngine()
        
        # Check for validate method
        if hasattr(engine, 'validate'):
            sig = inspect.signature(engine.validate)
            if inspect.iscoroutinefunction(engine.validate):
                print("  ✓ async validate() exists")
            else:
                issues.append("validate() exists but is not async")
                print("  ✗ validate() is not async")
        else:
            issues.append("validate() method missing")
            print("  ✗ validate() method missing")
        
        # Check for escalate method
        if hasattr(engine, 'escalate'):
            if inspect.iscoroutinefunction(engine.escalate):
                print("  ✓ async escalate() exists")
            else:
                issues.append("escalate() exists but is not async")
                print("  ✗ escalate() is not async")
        else:
            issues.append("escalate() method missing")
            print("  ✗ escalate() method missing")
            
    except Exception as e:
        issues.append(f"Governance check failed: {e}")
        print(f"  ✗ Governance check failed: {e}")
    
    return issues


def check_trust_api():
    """Check if async trust methods exist with correct signatures"""
    print("\n🔍 Checking Trust API...")
    
    issues = []
    
    try:
        from grace.trust.core import TrustCoreKernel
        import inspect
        
        trust = TrustCoreKernel()
        
        # Check calculate_trust
        if hasattr(trust, 'calculate_trust'):
            if inspect.iscoroutinefunction(trust.calculate_trust):
                sig = inspect.signature(trust.calculate_trust)
                params = list(sig.parameters.keys())
                if 'entity_id' in params and 'operation_context' in params:
                    print("  ✓ async calculate_trust() with correct signature")
                else:
                    issues.append(f"calculate_trust() has wrong parameters: {params}")
                    print(f"  ✗ calculate_trust() parameters: {params}")
            else:
                issues.append("calculate_trust() is not async")
                print("  ✗ calculate_trust() is not async")
        else:
            issues.append("calculate_trust() missing")
            print("  ✗ calculate_trust() missing")
        
        # Check update_trust
        if hasattr(trust, 'update_trust'):
            if inspect.iscoroutinefunction(trust.update_trust):
                print("  ✓ async update_trust() exists")
            else:
                issues.append("update_trust() is not async")
                print("  ✗ update_trust() is not async")
        else:
            issues.append("update_trust() missing")
            print("  ✗ update_trust() missing")
            
    except Exception as e:
        issues.append(f"Trust check failed: {e}")
        print(f"  ✗ Trust check failed: {e}")
    
    return issues


def check_async_memory():
    """Check if async memory classes are actually used"""
    print("\n🔍 Checking Async Memory Integration...")
    
    issues = []
    
    # Check if files exist
    files_to_check = [
        "grace/memory/async_lightning.py",
        "grace/memory/async_fusion.py",
        "grace/memory/immutable_logs_async.py"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path} exists")
        else:
            issues.append(f"{file_path} missing")
            print(f"  ✗ {file_path} missing")
    
    # Check if they're actually imported anywhere
    grace_dir = Path("grace")
    async_imports = []
    
    for py_file in grace_dir.rglob("*.py"):
        if "async_lightning" in py_file.name or "async_fusion" in py_file.name:
            continue  # Skip the files themselves
        
        try:
            content = py_file.read_text()
            if 'AsyncLightningMemory' in content or 'AsyncFusionMemory' in content:
                async_imports.append(str(py_file))
        except:
            pass
    
    if async_imports:
        print(f"  ✓ Async memory imported in {len(async_imports)} files")
    else:
        issues.append("Async memory classes not imported anywhere (not integrated)")
        print("  ✗ Async memory classes NOT imported (not integrated)")
    
    return issues


def main():
    """Run all checks"""
    print("=" * 80)
    print("Grace System - Critical Gap Analysis")
    print("=" * 80)
    
    all_issues = []
    
    all_issues.extend(check_entry_points())
    all_issues.extend(check_event_usage())
    all_issues.extend(check_governance_api())
    all_issues.extend(check_trust_api())
    all_issues.extend(check_async_memory())
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} critical issues:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n📋 Action Items:")
        print("  1. Review IMPLEMENTATION_STATUS_ACTUAL.md")
        print("  2. Fix entry points first")
        print("  3. Convert event system to GraceEvent")
        print("  4. Implement async governance/trust APIs")
        print("  5. Wire async memory into actual use")
        
        return 1
    else:
        print("\n✅ No critical issues found!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
