"""
Quick fix script for ImmutableLogs -> ImmutableLogger naming issue
"""

import re
import sys

file_path = '/workspaces/Grace-/grace/core/gtrace.py'

# Read the file
with open(file_path, 'r') as f:
    content = f.read()

# Replace all instances of ImmutableLogs with ImmutableLogger
original_content = content
content = content.replace('ImmutableLogs', 'ImmutableLogger')

# Write back
with open(file_path, 'w') as f:
    f.write(content)

# Count replacements
count = original_content.count('ImmutableLogs')
print(f"✓ Fixed {count} instances of ImmutableLogs -> ImmutableLogger")
