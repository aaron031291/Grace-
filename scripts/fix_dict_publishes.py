"""
Find and report any remaining dict-based event publishes
"""

import re
import sys
from pathlib import Path


def find_dict_publishes(content: str, filepath: str) -> list:
    """Find patterns that look like dict publishes"""
    issues = []
    
    # Pattern 1: .emit({...})
    pattern1 = r'\.(emit|publish)\s*\(\s*\{'
    for match in re.finditer(pattern1, content, re.MULTILINE):
        line_num = content[:match.start()].count('\n') + 1
        issues.append((line_num, "dict literal in emit/publish"))
    
    # Pattern 2: bus.emit without GraceEvent
    # This is harder to detect without full AST analysis
    
    return issues


def main():
    """Scan for dict publishes"""
    print("🔍 Scanning for dict-based event publishes...")
    print("=" * 60)
    
    grace_dir = Path("grace")
    all_issues = {}
    
    for py_file in grace_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        try:
            content = py_file.read_text()
            issues = find_dict_publishes(content, str(py_file))
            
            if issues:
                all_issues[py_file] = issues
        
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    if all_issues:
        print("\n⚠️  Found potential dict publishes:\n")
        
        for filepath, issues in all_issues.items():
            print(f"📄 {filepath}")
            for line_num, msg in issues:
                print(f"   Line {line_num}: {msg}")
        
        print("\n" + "=" * 60)
        print(f"Total: {sum(len(i) for i in all_issues.values())} potential issues")
        return 1
    else:
        print("\n✅ No dict publishes found!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
