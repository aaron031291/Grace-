#!/usr/bin/env python3
"""
Grace AI - Complete Import Audit & Fix
======================================
Identifies and fixes all import issues
"""

import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, '/workspaces/Grace-')

print("=" * 70)
print("GRACE AI - IMPORT AUDIT & FIX")
print("=" * 70)
print()

# STEP 1: Fix naming inconsistencies in files
print("STEP 1: Fixing naming inconsistencies...")
print("-" * 70)

grace_path = Path('/workspaces/Grace-/grace')
fixes_applied = 0

# List of file patterns to fix
files_to_fix = [
    'core/gtrace.py',
]

for file_pattern in files_to_fix:
    filepath = grace_path / file_pattern
    if filepath.exists():
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        # Fix ImmutableLogs -> ImmutableLogger
        content = content.replace('ImmutableLogs', 'ImmutableLogger')
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            count = original.count('ImmutableLogs')
            print(f"✓ {file_pattern}: Fixed {count} occurrences")
            fixes_applied += 1

print(f"✓ Fixed {fixes_applied} files")
print()

# STEP 2: Test imports after fixing
print("STEP 2: Testing critical imports...")
print("-" * 70)

test_imports = [
    ('grace.core.immutable_logs', 'ImmutableLogger'),
    ('grace.core.gtrace', 'GlobalTrace'),
    ('grace.core.truth_layer', 'CoreTruthLayer'),
    ('grace.core.service_registry', 'ServiceRegistry'),
    ('grace.kernels.base_kernel', 'BaseKernel'),
    ('grace.kernels.learning_kernel', 'LearningKernel'),
    ('grace.kernels.orchestration_kernel', 'OrchestrationKernel'),
    ('grace.kernels.resilience_kernel', 'ResilienceKernel'),
    ('grace.multi_os', 'MultiOSKernel'),
]

passed = 0
failed = 0

for module_path, class_name in test_imports:
    try:
        module = __import__(module_path, fromlist=[class_name])
        if hasattr(module, class_name):
            print(f"✓ {module_path}.{class_name}")
            passed += 1
        else:
            print(f"✗ {module_path}.{class_name} - class not found")
            failed += 1
    except Exception as e:
        print(f"✗ {module_path} - {str(e)[:60]}")
        failed += 1

print()
print(f"✓ Passed: {passed}/{len(test_imports)}")
print(f"✗ Failed: {failed}/{len(test_imports)}")
print()

# STEP 3: Summary
print("=" * 70)
if failed == 0:
    print("✅ ALL IMPORTS WORKING - Ready for wiring audit")
    sys.exit(0)
else:
    print("⚠️  SOME IMPORTS STILL FAILING - Check errors above")
    sys.exit(1)
