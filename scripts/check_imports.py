"""
Check and fix import issues
"""

import sys
import ast
from pathlib import Path
from collections import defaultdict


def check_imports(root_dir: str = "."):
    """Check all imports in the codebase"""
    grace_dir = Path(root_dir) / "grace"
    issues = defaultdict(list)
    
    print("🔍 Checking imports...")
    
    for py_file in grace_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('grace'):
                        # Check if module exists
                        parts = node.module.split('.')
                        module_path = grace_dir
                        
                        for part in parts[1:]:  # Skip 'grace'
                            next_path = module_path / part
                            if next_path.is_dir():
                                module_path = next_path
                            elif (module_path / f"{part}.py").exists():
                                module_path = module_path / f"{part}.py"
                                break
                            else:
                                issues['missing_modules'].append({
                                    'file': str(py_file.relative_to(Path(root_dir))),
                                    'line': node.lineno,
                                    'module': node.module
                                })
                                break
        
        except SyntaxError as e:
            issues['syntax_errors'].append({
                'file': str(py_file.relative_to(Path(root_dir))),
                'error': str(e)
            })
    
    # Report
    total = sum(len(v) for v in issues.values())
    
    if total == 0:
        print("✅ All imports look good!")
        return 0
    
    print(f"\n⚠️  Found {total} import issues:\n")
    
    for category, items in issues.items():
        if items:
            print(f"🔍 {category}: {len(items)} issues")
            for item in items[:5]:
                print(f"  • {item}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
    
    return 1


if __name__ == "__main__":
    sys.exit(check_imports("/workspaces/Grace-"))
