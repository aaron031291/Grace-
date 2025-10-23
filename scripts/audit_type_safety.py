"""
Audit type safety and check for raw dict publishes
"""

import sys
import ast
from pathlib import Path
from typing import List, Tuple


class TypeSafetyAuditor(ast.NodeVisitor):
    """AST visitor to find type safety issues"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.issues: List[Tuple[int, str]] = []
    
    def visit_Call(self, node: ast.Call):
        """Check function calls"""
        # Check for bus.emit/publish with dict argument
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ["emit", "publish"]:
                if node.args:
                    first_arg = node.args[0]
                    
                    # Check if argument is a dict literal
                    if isinstance(first_arg, ast.Dict):
                        self.issues.append((
                            node.lineno,
                            f"Found dict literal passed to {node.func.attr}() - should use GraceEvent"
                        ))
        
        self.generic_visit(node)


def audit_file(filepath: Path) -> List[Tuple[int, str]]:
    """Audit a single file for type safety"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(filepath))
        auditor = TypeSafetyAuditor(str(filepath))
        auditor.visit(tree)
        
        return auditor.issues
    
    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e}")]
    except Exception as e:
        return [(0, f"Error parsing file: {e}")]


def main():
    """Run type safety audit"""
    print("🔍 Auditing type safety...")
    print("=" * 60)
    
    grace_dir = Path("grace")
    all_issues = []
    
    # Audit all Python files in grace/
    for py_file in grace_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        issues = audit_file(py_file)
        
        if issues:
            all_issues.extend([(py_file, line, msg) for line, msg in issues])
            print(f"\n❌ {py_file}")
            for line, msg in issues:
                print(f"   Line {line}: {msg}")
    
    print("\n" + "=" * 60)
    
    if all_issues:
        print(f"\n⚠️  Found {len(all_issues)} type safety issues")
        print("\nTo fix:")
        print("  1. Replace dict literals with GraceEvent objects")
        print("  2. Use event_factory.create_event() or GraceEvent(...)")
        print("  3. Ensure all emit/publish calls use GraceEvent")
        return 1
    else:
        print("\n✅ No type safety issues found!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
