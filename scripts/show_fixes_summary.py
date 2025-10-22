"""
Show what was fixed and what still needs attention
"""

import sys
from pathlib import Path
import subprocess


def show_summary():
    """Show comprehensive summary of fixes and remaining issues"""
    
    print("=" * 80)
    print("🔍 GRACE FORENSIC ANALYSIS - FIXES SUMMARY")
    print("=" * 80)
    
    print("\n📋 WHAT WAS FIXED:")
    print("-" * 80)
    
    fixes_applied = [
        "✅ Created comprehensive forensic analysis system",
        "✅ Created automatic fix scripts for common issues",
        "✅ Added missing __init__.py detection and creation",
        "✅ Added unused import detection and removal",
        "✅ Added import issue detection",
        "✅ Added circular import detection",
        "✅ Added type safety validation",
        "✅ Added syntax error detection",
        "✅ Created master fix script (fix_all_warnings.sh)",
        "✅ Created quick fix script for immediate issues",
        "✅ Added import checker script",
    ]
    
    for fix in fixes_applied:
        print(f"  {fix}")
    
    print("\n🔧 AUTOMATED FIXES AVAILABLE:")
    print("-" * 80)
    
    automated_fixes = [
        "1. Missing __init__.py files → Auto-created",
        "2. Unused imports → Auto-removed",
        "3. Python cache files → Auto-cleaned",
        "4. Code formatting → Black auto-format",
        "5. Import sorting → isort auto-sort",
        "6. Common missing imports → Auto-added",
    ]
    
    for fix in automated_fixes:
        print(f"  {fix}")
    
    print("\n🔍 WHAT THE FORENSIC ANALYZER DETECTS:")
    print("-" * 80)
    
    detections = [
        "🔴 CRITICAL - Syntax errors (breaks code)",
        "🟠 HIGH - Import errors (missing modules)",
        "🟠 HIGH - Undefined names (typos, missing imports)",
        "🟠 HIGH - Circular imports (architecture issues)",
        "🟡 MEDIUM - Type annotation issues",
        "🟡 MEDIUM - Missing __init__.py files",
        "⚪ LOW - Unused imports (cleanup)",
    ]
    
    for detection in detections:
        print(f"  {detection}")
    
    print("\n📊 RUNNING QUICK ANALYSIS...")
    print("-" * 80)
    
    # Check if grace directory exists
    grace_dir = Path("/workspaces/Grace-/grace")
    if not grace_dir.exists():
        print("❌ Grace directory not found!")
        return 1
    
    # Quick checks
    print("\n1. Checking Python files...")
    py_files = list(grace_dir.rglob("*.py"))
    print(f"   Found {len(py_files)} Python files")
    
    print("\n2. Checking for __init__.py files...")
    dirs_with_py = []
    dirs_without_init = []
    
    for dir_path in grace_dir.rglob("*"):
        if dir_path.is_dir() and not dir_path.name.startswith('.'):
            if any(dir_path.glob("*.py")):
                dirs_with_py.append(dir_path)
                if not (dir_path / "__init__.py").exists():
                    dirs_without_init.append(dir_path)
    
    print(f"   Directories with Python files: {len(dirs_with_py)}")
    print(f"   Missing __init__.py: {len(dirs_without_init)}")
    
    if dirs_without_init:
        print(f"\n   Missing in:")
        for d in dirs_without_init[:5]:
            print(f"     • {d.relative_to(Path('/workspaces/Grace-'))}")
        if len(dirs_without_init) > 5:
            print(f"     ... and {len(dirs_without_init) - 5} more")
    
    print("\n3. Checking syntax errors...")
    syntax_errors = []
    for py_file in py_files:
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append((py_file, e))
    
    if syntax_errors:
        print(f"   🔴 Found {len(syntax_errors)} syntax errors:")
        for file, error in syntax_errors[:3]:
            print(f"     • {file.relative_to(Path('/workspaces/Grace-'))}: Line {error.lineno}")
    else:
        print(f"   ✅ No syntax errors found!")
    
    print("\n4. Checking imports with pyflakes...")
    try:
        result = subprocess.run(
            ['python', '-m', 'pyflakes', 'grace/'],
            capture_output=True,
            text=True,
            cwd='/workspaces/Grace-'
        )
        
        lines = [l for l in result.stdout.split('\n') if l.strip()]
        
        if lines:
            print(f"   Found {len(lines)} issues")
            
            # Categorize
            undefined = [l for l in lines if 'undefined name' in l]
            unused = [l for l in lines if 'imported but unused' in l]
            
            if undefined:
                print(f"     🟠 Undefined names: {len(undefined)}")
                for line in undefined[:3]:
                    print(f"        {line[:100]}")
            
            if unused:
                print(f"     ⚪ Unused imports: {len(unused)}")
        else:
            print(f"   ✅ No pyflakes issues!")
            
    except Exception as e:
        print(f"   ⚠️  Pyflakes not available: {e}")
    
    print("\n" + "=" * 80)
    print("📝 NEXT STEPS:")
    print("=" * 80)
    
    next_steps = [
        "",
        "To run full forensic analysis:",
        "  python scripts/forensic_analysis.py",
        "",
        "To apply automatic fixes:",
        "  bash scripts/quick_fix.sh",
        "  bash scripts/fix_all_warnings.sh",
        "",
        "To check specific issues:",
        "  python scripts/check_imports.py",
        "  python scripts/auto_fix_issues.py",
        "",
        "Results will be saved to:",
        "  📄 forensic_report.json (detailed JSON report)",
        "",
    ]
    
    for step in next_steps:
        print(step)
    
    return 0


if __name__ == "__main__":
    sys.exit(show_summary())
