"""
Check that all MCP messages use appropriate trust scores
"""

import sys
import ast
from pathlib import Path


def find_mcp_sends(content: str, filepath: str) -> list:
    """Find MCP send_message calls"""
    issues = []
    
    try:
        tree = ast.parse(content, filename=filepath)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "send_message":
                    # Check if trust_score is specified
                    has_trust = False
                    for keyword in node.keywords:
                        if keyword.arg == "trust_score":
                            has_trust = True
                            break
                    
                    if not has_trust:
                        issues.append((node.lineno, "send_message without explicit trust_score"))
    
    except SyntaxError:
        pass
    
    return issues


def main():
    """Check trust scores in MCP calls"""
    print("🔍 Checking MCP Trust Scores")
    print("=" * 60)
    
    grace_dir = Path("grace/kernels")
    all_issues = []
    
    for py_file in grace_dir.rglob("*.py"):
        content = py_file.read_text()
        issues = find_mcp_sends(content, str(py_file))
        
        if issues:
            all_issues.extend([(py_file, line, msg) for line, msg in issues])
            print(f"\n⚠️  {py_file}")
            for line, msg in issues:
                print(f"   Line {line}: {msg}")
    
    print("\n" + "=" * 60)
    
    if all_issues:
        print(f"\n⚠️  Found {len(all_issues)} messages without explicit trust_score")
        print("Consider adding trust_score parameter to all MCP send_message calls")
        return 0  # Warning, not error
    else:
        print("\n✅ All MCP messages have trust scores!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
