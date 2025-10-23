"""
Automated fixes for all common Pylance issues
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def fix_missing_imports(file_path: Path) -> int:
    """Fix missing imports"""
    content = file_path.read_text()
    original = content
    fixes = 0
    
    # Fix 1: Add Any to typing imports
    if 'Any' in content and 'from typing import' in content:
        match = re.search(r'from typing import ([^\n]+)', content)
        if match and 'Any' not in match.group(1):
            imports = match.group(1).rstrip()
            # Remove trailing whitespace and add Any
            if not imports.endswith(','):
                imports += ','
            new_line = f'from typing import {imports} Any'
            content = content.replace(match.group(0), new_line)
            fixes += 1
    
    # Fix 2: Add missing standard library imports
    if 'Optional' in content and 'from typing import' in content:
        match = re.search(r'from typing import ([^\n]+)', content)
        if match and 'Optional' not in match.group(1):
            imports = match.group(1).rstrip()
            if not imports.endswith(','):
                imports += ','
            new_line = f'from typing import {imports} Optional'
            content = content.replace(match.group(0), new_line)
            fixes += 1
    
    if content != original:
        file_path.write_text(content)
    
    return fixes


def fix_optional_defaults(file_path: Path) -> int:
    """Fix Optional parameters without defaults"""
    content = file_path.read_text()
    original = content
    fixes = 0
    
    # Pattern: Optional[Type] without = None in function parameters
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Check if it's a function parameter line
        if 'Optional[' in line and '=' not in line and ('def ' in line or ',' in line or ')' in line):
            # Add = None before comma or closing paren
            modified = re.sub(
                r'(:\s*Optional\[[^\]]+\])(\s*[,)])',
                r'\1 = None\2',
                line
            )
            if modified != line:
                line = modified
                fixes += 1
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content != original:
        file_path.write_text(content)
    
    return fixes


def fix_dataclass_defaults(file_path: Path) -> int:
    """Fix mutable defaults in dataclasses"""
    content = file_path.read_text()
    original = content
    fixes = 0
    
    if '@dataclass' not in content:
        return 0
    
    # Fix Dict defaults
    content = re.sub(
        r'(\w+):\s*(Dict\[[^\]]+\])\s*=\s*\{\}',
        r'\1: \2 = field(default_factory=dict)',
        content
    )
    
    # Fix List defaults
    content = re.sub(
        r'(\w+):\s*(List\[[^\]]+\])\s*=\s*\[\]',
        r'\1: \2 = field(default_factory=list)',
        content
    )
    
    # Ensure field is imported
    if 'field(default_factory=' in content and 'from dataclasses import' in content:
        match = re.search(r'from dataclasses import ([^\n]+)', content)
        if match and 'field' not in match.group(1):
            imports = match.group(1).rstrip()
            if not imports.endswith(','):
                imports += ','
            new_line = f'from dataclasses import {imports} field'
            content = content.replace(match.group(0), new_line)
    
    if content != original:
        fixes = content.count('field(default_factory=') - original.count('field(default_factory=')
        file_path.write_text(content)
    
    return fixes


def fix_return_types(file_path: Path) -> int:
    """Add return type hints to functions"""
    content = file_path.read_text()
    original = content
    fixes = 0
    
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        # Check if it's a function definition without return type
        if line.strip().startswith('def ') and '->' not in line and '__init__' not in line and '__repr__' not in line and '__str__' not in line:
            # Add -> None before colon
            if line.rstrip().endswith(':'):
                line = line.rstrip()[:-1] + ' -> None:'
                fixes += 1
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content != original:
        file_path.write_text(content)
    
    return fixes


def process_file(file_path: Path) -> Tuple[int, List[str]]:
    """Process a single file with all fixes"""
    total_fixes = 0
    fixes_applied = []
    
    # Apply fixes in order
    n = fix_missing_imports(file_path)
    if n > 0:
        total_fixes += n
        fixes_applied.append(f"imports: {n}")
    
    n = fix_optional_defaults(file_path)
    if n > 0:
        total_fixes += n
        fixes_applied.append(f"optional: {n}")
    
    n = fix_dataclass_defaults(file_path)
    if n > 0:
        total_fixes += n
        fixes_applied.append(f"dataclass: {n}")
    
    n = fix_return_types(file_path)
    if n > 0:
        total_fixes += n
        fixes_applied.append(f"returns: {n}")
    
    return total_fixes, fixes_applied


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("Automated Pylance Issue Fixes")
    print("=" * 80)
    
    grace_dir = Path("grace")
    python_files = list(grace_dir.rglob("*.py"))
    
    print(f"\nProcessing {len(python_files)} Python files...")
    
    total_fixes = 0
    files_modified = 0
    
    for file_path in python_files:
        fixes, details = process_file(file_path)
        
        if fixes > 0:
            files_modified += 1
            total_fixes += fixes
            rel_path = file_path.relative_to('.')
            print(f"  ✓ {rel_path}: {', '.join(details)}")
    
    print("\n" + "=" * 80)
    print(f"Applied {total_fixes} fixes to {files_modified} files")
    print("=" * 80)
    
    print("\nNext steps:")
    print("  1. Review changes: git diff")
    print("  2. Run analysis: python scripts/analyze_all_problems.py")
    print("  3. Run tests: pytest")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
