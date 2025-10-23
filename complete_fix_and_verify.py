#!/usr/bin/env python3
"""
Grace AI - Complete Fix & Verification Script
==============================================
Fixes import issues and runs comprehensive verification
"""

import sys
import os

sys.path.insert(0, '/workspaces/Grace-')

print("=" * 70)
print("GRACE AI - COMPLETE FIX & VERIFICATION")
print("=" * 70)
print()

# STEP 1: Fix naming issues
print("STEP 1: Fixing naming issues in grace/core/gtrace.py...")
print("-" * 70)

gtrace_file = '/workspaces/Grace-/grace/core/gtrace.py'
try:
    with open(gtrace_file, 'r') as f:
        content = f.read()
    
    before = content.count('ImmutableLogs')
    if before > 0:
        content = content.replace('ImmutableLogs', 'ImmutableLogger')
        with open(gtrace_file, 'w') as f:
            f.write(content)
        print(f"✓ Fixed {before} occurrences of ImmutableLogs → ImmutableLogger")
    else:
        print("✓ No ImmutableLogs found (already fixed?)")
except Exception as e:
    print(f"✗ Error fixing file: {e}")
    sys.exit(1)

print()

# STEP 2: Test critical imports
print("STEP 2: Testing critical imports...")
print("-" * 70)

test_imports = [
    ('grace.core.immutable_logs', 'ImmutableLogger'),
    ('grace.core.gtrace', 'GlobalTrace'),
    ('grace.core.truth_layer', 'CoreTruthLayer'),
    ('grace.core.service_registry', 'ServiceRegistry'),
    ('grace.kernels.base_kernel', 'BaseKernel'),
    ('grace.memory', 'UnifiedMemorySystem'),
]

import_pass = 0
import_fail = 0

for module_path, class_name in test_imports:
    try:
        mod = __import__(module_path, fromlist=[class_name])
        if hasattr(mod, class_name):
            print(f"✓ {module_path}.{class_name}")
            import_pass += 1
        else:
            print(f"✗ {module_path}.{class_name} - not found in module")
            import_fail += 1
    except Exception as e:
        print(f"✗ {module_path} - {str(e)[:50]}")
        import_fail += 1

print(f"\n  Passed: {import_pass}/{len(test_imports)}")
print(f"  Failed: {import_fail}/{len(test_imports)}")

if import_fail > 0:
    print("\n⚠️  Some imports still failing - system may have other issues")
else:
    print("\n✅ All critical imports working!")

print()

# STEP 3: Run wiring audit
print("STEP 3: Running wiring audit...")
print("-" * 70)

try:
    # Import the audit module
    from grace.diagnostics import wiring_audit
    
    print("✓ Wiring audit module loaded")
    print("\nNote: Running full audit would execute the actual audit.")
    print("Audit script is ready to use:")
    print("  python grace/diagnostics/wiring_audit.py")
    
except Exception as e:
    print(f"✗ Wiring audit import failed: {e}")

print()

# STEP 4: Summary
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if import_fail == 0:
    print("✅ STATUS: SYSTEM FIXED AND READY")
    print()
    print("Next steps:")
    print("  1. Run full wiring audit:")
    print("     python grace/diagnostics/wiring_audit.py")
    print()
    print("  2. Verify system works:")
    print("     python verify_system.py")
    print()
    print("  3. Start the system:")
    print("     python -m grace.launcher --dry-run")
    print("     python -m grace.launcher")
else:
    print("⚠️  STATUS: SOME ISSUES REMAIN")
    print()
    print("Please review errors above and fix remaining issues.")

print()
