#!/usr/bin/env python3
"""
Grace AI - Import Fix Script
============================
Fixes all broken imports and naming inconsistencies
"""

import os
import re
from pathlib import Path

def fix_file(filepath, fixes):
    """Apply fixes to a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        for old, new in fixes:
            content = content.replace(old, new)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

# Define all fixes needed
fixes = [
    ('from .immutable_logs import ImmutableLogs', 'from .immutable_logs import ImmutableLogger'),
    ('immutable_logs: Optional[ImmutableLogs]', 'immutable_logs: Optional[ImmutableLogger]'),
    (': ImmutableLogs =', ': ImmutableLogger ='),
]

# Find and fix all Python files
grace_path = Path('/workspaces/Grace-/grace')
fixed_count = 0
file_count = 0

for py_file in grace_path.rglob('*.py'):
    file_count += 1
    if fix_file(str(py_file), fixes):
        fixed_count += 1
        print(f"✓ Fixed: {py_file.relative_to(grace_path)}")

print(f"\n✓ Processed {file_count} files")
print(f"✓ Fixed {fixed_count} files with naming issues")
