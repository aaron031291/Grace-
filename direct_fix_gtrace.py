#!/usr/bin/env python3
"""
Direct file fix for gtrace.py - ImmutableLogs naming issue
"""

import re

filepath = '/workspaces/Grace-/grace/core/gtrace.py'

print("Reading gtrace.py...")
with open(filepath, 'r') as f:
    content = f.read()

print(f"Original file size: {len(content)} bytes")

# Count occurrences before
before_count = content.count('ImmutableLogs')
print(f"Found {before_count} occurrences of 'ImmutableLogs'")

if before_count > 0:
    # Replace all occurrences
    content = content.replace('ImmutableLogs', 'ImmutableLogger')
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ Replaced {before_count} occurrences")
    print(f"✓ New file size: {len(content)} bytes")
    print(f"✓ Fixed: grace/core/gtrace.py")
else:
    print("✓ No ImmutableLogs found - file already fixed?")

print("\nNow testing imports...")
try:
    from grace.core.gtrace import GlobalTrace
    print("✓ Successfully imported GlobalTrace")
except Exception as e:
    print(f"✗ Import failed: {e}")
