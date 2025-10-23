#!/usr/bin/env python3
"""
Quick Analysis: Count errors before and after pattern fixes
"""

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def count_mypy_errors() -> Dict[str, int]:
    """Run mypy and count error types"""
    
    print("🔍 Running mypy analysis...")
    
    result = subprocess.run(
        ["mypy", "grace", "--ignore-missing-imports", "--no-error-summary"],
        capture_output=True,
        text=True,
        cwd="/workspaces/Grace-"
    )
    
    errors = defaultdict(int)
    total = 0
    
    for line in result.stdout.split('\n'):
        if 'error:' in line:
            total += 1
            # Extract error type
            if 'Implicit Optional' in line or 'no_implicit_optional' in line:
                errors['implicit_optional'] += 1
            elif 'incompatible with' in line and 'gather' in line:
                errors['asyncio_gather'] += 1
            elif 'Unsupported operand' in line:
                errors['arithmetic'] += 1
            elif 'has no attribute' in line and 'append' in line:
                errors['container_ops'] += 1
            elif 'Incompatible return value' in line:
                errors['return_type'] += 1
            elif 'Name' in line and 'not defined' in line:
                errors['missing_import'] += 1
            elif 'Unexpected keyword argument' in line:
                errors['signature_mismatch'] += 1
            elif 'datetime' in line and ('object' in line or 'sub' in line):
                errors['datetime_arithmetic'] += 1
            else:
                errors['other'] += 1
    
    errors['total'] = total
    return dict(errors)


def print_comparison(before: Dict[str, int], after: Dict[str, int]):
    """Print before/after comparison"""
    
    print("\n" + "=" * 80)
    print("📊 Error Count Comparison")
    print("=" * 80)
    
    categories = [
        ('implicit_optional', 'Implicit Optional'),
        ('asyncio_gather', 'asyncio.gather'),
        ('arithmetic', 'Arithmetic with object'),
        ('container_ops', 'Container Operations'),
        ('return_type', 'Return Type Mismatch'),
        ('missing_import', 'Missing Imports'),
        ('signature_mismatch', 'Signature Mismatch'),
        ('datetime_arithmetic', 'Datetime Arithmetic'),
        ('other', 'Other'),
        ('total', 'TOTAL'),
    ]
    
    print(f"\n{'Category':<25} {'Before':>10} {'After':>10} {'Fixed':>10} {'%':>8}")
    print("-" * 80)
    
    for key, label in categories:
        before_count = before.get(key, 0)
        after_count = after.get(key, 0)
        fixed = before_count - after_count
        pct = (fixed / before_count * 100) if before_count > 0 else 0
        
        print(f"{label:<25} {before_count:>10} {after_count:>10} {fixed:>10} {pct:>7.1f}%")
    
    print("=" * 80)
    
    total_fixed = before.get('total', 0) - after.get('total', 0)
    pct_total = (total_fixed / before.get('total', 1) * 100)
    
    print(f"\n✅ Fixed {total_fixed} errors ({pct_total:.1f}% reduction)")
    print(f"📉 Remaining: {after.get('total', 0)} errors")
    print()


def main():
    """Run analysis"""
    
    # Check if we have a before snapshot
    before_file = Path("/tmp/grace_errors_before.json")
    
    if before_file.exists():
        with open(before_file) as f:
            before = json.load(f)
        
        after = count_mypy_errors()
        
        # Save after snapshot
        with open("/tmp/grace_errors_after.json", 'w') as f:
            json.dump(after, f, indent=2)
        
        print_comparison(before, after)
    else:
        # First run - save baseline
        print("📸 Creating baseline snapshot...")
        errors = count_mypy_errors()
        
        with open(before_file, 'w') as f:
            json.dump(errors, f, indent=2)
        
        print(f"\n✅ Baseline saved: {errors.get('total', 0)} total errors")
        print("\nRun pattern fixes, then run this script again to see the difference.")


if __name__ == "__main__":
    main()
