"""
Automated Pylance error fixes
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def fix_file(file_path: Path) -> int:
    """Fix all Pylance errors in a file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes = 0
        
        # Fix 1: Add Any to typing imports if missing
        if 'Any' in content and 'from typing import' in content:
            typing_imports = re.findall(r'from typing import ([^\n]+)', content)
            if typing_imports and 'Any' not in typing_imports[0]:
                content = re.sub(
                    r'from typing import ([^\n]+)',
                    lambda m: f'from typing import {m.group(1)}, Any' if 'Any' not in m.group(1) else m.group(0),
                    content,
                    count=1
                )
                fixes += 1
        
        # Fix 2: Add Optional with = None
        matches = list(re.finditer(r'(\w+):\s*Optional\[([\w\[\],\s]+)\](?!\s*=)', content))
        for match in reversed(matches):
            old = match.group(0)
            new = f"{old} = None"
            content = content[:match.start()] + new + content[match.end():]
            fixes += 1
        
        # Fix 3: Fix dataclass mutable defaults
        if '@dataclass' in content:
            # Ensure field is imported
            if 'from dataclasses import' in content and 'field' not in content:
                content = re.sub(
                    r'from dataclasses import ([^\n]+)',
                    lambda m: f'from dataclasses import {m.group(1)}, field' if 'field' not in m.group(1) else m.group(0),
                    content,
                    count=1
                )
                fixes += 1
            
            # Fix Dict defaults
            pattern = r'(\w+):\s*(Dict\[[^\]]+\])\s*=\s*\{\}'
            if re.search(pattern, content):
                content = re.sub(pattern, r'\1: \2 = field(default_factory=dict)', content)
                fixes += 1
            
            # Fix List defaults
            pattern = r'(\w+):\s*(List\[[^\]]+\])\s*=\s*\[\]'
            if re.search(pattern, content):
                content = re.sub(pattern, r'\1: \2 = field(default_factory=list)', content)
                fixes += 1
        
        # Fix 4: Add missing return types for functions
        # Pattern: def function_name(...) without -> 
        pattern = r'def (\w+)\([^)]*\)(?!\s*->):'
        matches = list(re.finditer(pattern, content))
        for match in reversed(matches):
            func_name = match.group(1)
            # Skip __init__, __repr__, etc.
            if not func_name.startswith('__'):
                # Check if it's a simple function we can infer
                if 'return' not in content[match.end():match.end()+500]:
                    # No return, assume None
                    old = match.group(0)
                    new = old[:-1] + ' -> None:'
                    content = content[:match.start()] + new + content[match.end():]
                    fixes += 1
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return fixes
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}[/red]")
        return 0


def main():
    """Run automated fixes"""
    console.print("\n[bold blue]Automated Pylance Fixes[/bold blue]")
    console.print("=" * 70)
    
    grace_dir = Path("grace")
    python_files = list(grace_dir.rglob("*.py"))
    
    console.print(f"\nProcessing {len(python_files)} Python files...\n")
    
    results: List[Tuple[Path, int]] = []
    total_fixes = 0
    
    for file_path in python_files:
        fixes = fix_file(file_path)
        if fixes > 0:
            results.append((file_path, fixes))
            total_fixes += fixes
    
    if results:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan")
        table.add_column("Fixes", justify="right", style="green")
        
        for file_path, fixes in sorted(results, key=lambda x: x[1], reverse=True)[:20]:
            table.add_row(str(file_path.relative_to('.')), str(fixes))
        
        console.print(table)
    
    console.print(f"\n[green]✅ Total fixes applied: {total_fixes}[/green]")
    console.print(f"[green]✅ Files modified: {len(results)}[/green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
