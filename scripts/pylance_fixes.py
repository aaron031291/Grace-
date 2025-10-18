"""
Systematic Pylance error fixes
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def fix_any_imports(file_path: Path) -> int:
    """Fix missing Any imports"""
    content = file_path.read_text()
    fixes = 0
    
    # Check if file uses Any but doesn't import it
    if 'Any' in content and 'from typing import' in content:
        # Check if Any is already imported
        if 'from typing import' in content and 'Any' not in re.search(r'from typing import ([^\\n]+)', content).group(1):
            # Add Any to existing typing import
            content = re.sub(
                r'from typing import ([^\\n]+)',
                lambda m: f'from typing import {m.group(1)}, Any' if 'Any' not in m.group(1) else m.group(0),
                content,
                count=1
            )
            fixes += 1
    
    if fixes > 0:
        file_path.write_text(content)
    
    return fixes


def fix_optional_none_defaults(file_path: Path) -> int:
    """Fix Optional parameters without None defaults"""
    content = file_path.read_text()
    fixes = 0
    
    # Pattern: Optional[Type] without = None
    pattern = r'(\w+):\s*Optional\[([\w\[\],\s]+)\](?!\s*=)'
    
    matches = list(re.finditer(pattern, content))
    for match in reversed(matches):  # Reverse to maintain positions
        param_name = match.group(1)
        # Add = None
        old_text = match.group(0)
        new_text = f"{old_text} = None"
        content = content[:match.start()] + new_text + content[match.end():]
        fixes += 1
    
    if fixes > 0:
        file_path.write_text(content)
    
    return fixes


def fix_dataclass_fields(file_path: Path) -> int:
    """Fix dataclass fields with mutable defaults"""
    content = file_path.read_text()
    fixes = 0
    
    # Pattern: field with mutable default (dict, list)
    if '@dataclass' in content:
        # Check for dict/list defaults without field(default_factory=...)
        patterns = [
            (r'(\w+):\s*(Dict\[[\w\[\],\s]+\])\s*=\s*\{\}', r'\1: \2 = field(default_factory=dict)'),
            (r'(\w+):\s*(List\[[\w\[\],\s]+\])\s*=\s*\[\]', r'\1: \2 = field(default_factory=list)'),
        ]
        
        for pattern, replacement in patterns:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                fixes += count
                
                # Ensure field is imported
                if 'from dataclasses import' in content and 'field' not in content:
                    content = re.sub(
                        r'from dataclasses import dataclass',
                        'from dataclasses import dataclass, field',
                        content
                    )
    
    if fixes > 0:
        file_path.write_text(content)
    
    return fixes


def main():
    """Run all Pylance fixes"""
    console.print("\n[bold blue]Systematic Pylance Error Fixes[/bold blue]")
    console.print("=" * 70)
    
    grace_dir = Path("grace")
    python_files = list(grace_dir.rglob("*.py"))
    
    console.print(f"\nFound {len(python_files)} Python files")
    
    total_fixes = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Fixing errors...", total=len(python_files))
        
        for file_path in python_files:
            try:
                fixes = 0
                fixes += fix_any_imports(file_path)
                fixes += fix_optional_none_defaults(file_path)
                fixes += fix_dataclass_fields(file_path)
                
                if fixes > 0:
                    console.print(f"  Fixed {fixes} issues in {file_path.relative_to('.')}")
                    total_fixes += fixes
                
                progress.advance(task)
                
            except Exception as e:
                console.print(f"  [red]Error in {file_path}: {e}[/red]")
    
    console.print(f"\n[green]✅ Total fixes applied: {total_fixes}[/green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
