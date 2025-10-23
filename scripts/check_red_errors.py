"""
Check for common import errors that show as red in IDE
"""

import sys
from pathlib import Path
import ast

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports_in_file(file_path):
    """Check if all imports in a file can be resolved"""
    try:
        content = file_path.read_text()
        tree = ast.parse(content, filename=str(file_path))
        
        imports = []
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Try to import each
        for imp in imports:
            try:
                __import__(imp.split('.')[0])
            except ImportError as e:
                errors.append(f"{file_path}: Cannot import '{imp}' - {e}")
        
        return errors
        
    except SyntaxError as e:
        return [f"{file_path}: Syntax error - {e}"]
    except Exception as e:
        return [f"{file_path}: Error - {e}"]


def main():
    """Check all Python files"""
    print("Checking for import errors...")
    print("=" * 60)
    
    all_errors = []
    
    # Check test file
    test_file = Path("tests/test_complete_system.py")
    if test_file.exists():
        errors = check_imports_in_file(test_file)
        all_errors.extend(errors)
    
    # Check grace modules
    grace_dir = Path("grace")
    for py_file in grace_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        errors = check_imports_in_file(py_file)
        all_errors.extend(errors)
    
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} import errors:\n")
        for error in all_errors[:20]:  # Show first 20
            print(f"  {error}")
        
        if len(all_errors) > 20:
            print(f"\n  ... and {len(all_errors) - 20} more")
        
        print("\nTo fix:")
        print("  1. pip install -e .")
        print("  2. python scripts/ensure_init_files.py")
        print("  3. Run this script again")
        
        return 1
    else:
        print("\n✅ No import errors found!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
