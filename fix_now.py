#!/usr/bin/env python3
"""Fix ImmutableLogs naming issue"""
import sys
sys.path.insert(0, '/workspaces/Grace-')

# Fix grace/core/gtrace.py
gtrace_file = '/workspaces/Grace-/grace/core/gtrace.py'
with open(gtrace_file, 'r') as f:
    content = f.read()

before = content.count('ImmutableLogs')
content = content.replace('ImmutableLogs', 'ImmutableLogger')
after = content.count('ImmutableLogs')

with open(gtrace_file, 'w') as f:
    f.write(content)

print(f"✓ Fixed grace/core/gtrace.py: {before} replacements")

# Now test imports
print("\nTesting imports...")
try:
    from grace.core.immutable_logs import ImmutableLogger
    print("✓ ImmutableLogger imports")
except Exception as e:
    print(f"✗ ImmutableLogger import failed: {e}")
    sys.exit(1)

try:
    from grace.core.gtrace import GlobalTrace
    print("✓ GlobalTrace imports")
except Exception as e:
    print(f"✗ GlobalTrace import failed: {e}")
    sys.exit(1)

try:
    from grace import core
    print("✓ grace.core imports")
except Exception as e:
    print(f"✗ grace.core import failed: {e}")
    sys.exit(1)

print("\n✅ ALL IMPORTS FIXED AND WORKING")
