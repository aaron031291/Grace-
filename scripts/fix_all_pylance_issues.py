"""
Comprehensive script to fix all Pylance issues
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track

console = Console()


def fix_file(file_path: Path) -> int:
    """Fix all issues in a file"""
    try:
        content = file_path.read_text()
        original = content
        fixes = 0
        
        # Fix 1: Ensure Any is imported where used
        if 'Any' in content and 'from typing import' in content:
            # Check if Any is already imported
            typing_import = re.search(r'from typing import ([^\n]+)', content)
            if typing_import and 'Any' not in typing_import.group(1):
                # Add Any to existing import
                new_import = typing_import.group(1).rstrip() + ', Any'
                content = content.replace(typing_import.group(0), f'from typing import {new_import}')
                fixes += 1
        
        # Fix 2: Add Optional defaults
        pattern = r'(\w+):\s*Optional\[([\w\[\],\s]+)\](?!\s*=)'
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1: Optional[\2] = None', content)
            fixes += 1
        
        # Fix 3: Fix dataclass mutable defaults
        if '@dataclass' in content:
            # Fix dict defaults
            content = re.sub(
                r'(\w+):\s*(Dict\[[\w\[\],\s]+\])\s*=\s*\{\}',
                r'\1: \2 = field(default_factory=dict)',
                content
            )
            # Fix list defaults
            content = re.sub(
                r'(\w+):\s*(List\[[\w\[\],\s]+\])\s*=\s*\[\]',
                r'\1: \2 = field(default_factory=list)',
                content
            )
            
            # Ensure field is imported
            if 'field(default_factory=' in content and 'from dataclasses import' in content:
                if 'field' not in re.search(r'from dataclasses import ([^\n]+)', content).group(1):
                    content = re.sub(
                        r'from dataclasses import dataclass',
                        'from dataclasses import dataclass, field',
                        content
                    )
        
        # Fix 4: Add type hints to function returns
        # Look for functions without return type
        pattern = r'def (\w+)\([^)]*\):(\s*""")'
        matches = list(re.finditer(pattern, content))
        for match in matches:
            func_name = match.group(1)
            # Add -> None for methods that don't return
            if func_name not in ['__init__', '__repr__', '__str__']:
                content = content.replace(
                    match.group(0),
                    match.group(0).replace('):', ') -> None:')
                )
        
        if content != original:
            file_path.write_text(content)
            return fixes
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error fixing {file_path}: {e}[/red]")
        return 0


def main():
    """Fix all Pylance issues"""
    console.print("\n[bold blue]Fixing All Pylance Issues[/bold blue]")
    console.print("=" * 70)
    
    grace_dir = Path("grace")
    python_files = list(grace_dir.rglob("*.py"))
    
    total_fixes = 0
    files_fixed = 0
    
    for file_path in track(python_files, description="Processing files..."):
        fixes = fix_file(file_path)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1
            console.print(f"  [green]✓[/green] {file_path.relative_to('.')}: {fixes} fixes")
    
    console.print(f"\n[green]✅ Fixed {total_fixes} issues in {files_fixed} files[/green]")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Run: python scripts/run_full_diagnostics.py")
    console.print("  2. Run: python scripts/master_validation.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
