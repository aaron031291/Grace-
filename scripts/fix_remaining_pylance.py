"""
Fix all remaining Pylance errors systematically
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def add_any_to_file(file_path: Path) -> int:
    """Add Any to typing imports if used but not imported"""
    content = file_path.read_text()
    
    # Check if Any is used
    if re.search(r'\bAny\b', content):
        # Check if already imported
        if 'from typing import' in content:
            typing_line = re.search(r'from typing import ([^\n]+)', content)
            if typing_line:
                imports = typing_line.group(1)
                if 'Any' not in imports:
                    # Add Any to imports
                    new_imports = imports.rstrip() + ', Any'
                    content = content.replace(
                        f'from typing import {imports}',
                        f'from typing import {new_imports}'
                    )
                    file_path.write_text(content)
                    return 1
    return 0


def fix_optional_defaults(file_path: Path) -> int:
    """Fix Optional types without default None"""
    content = file_path.read_text()
    original = content
    
    # Pattern: Optional[...] without = None
    pattern = r'(\w+):\s*Optional\[([^\]]+)\]([,\)])'
    
    def replacer(match):
        param = match.group(1)
        type_hint = match.group(2)
        ending = match.group(3)
        # Check if there's already a default
        next_chars = content[match.end():match.end()+10]
        if not next_chars.strip().startswith('='):
            return f'{param}: Optional[{type_hint}] = None{ending}'
        return match.group(0)
    
    content = re.sub(pattern, replacer, content)
    
    if content != original:
        file_path.write_text(content)
        return 1
    return 0


def main():
    """Fix all remaining Pylance errors"""
    console.print("\n[bold blue]Fixing Remaining Pylance Errors[/bold blue]")
    console.print("=" * 70)
    
    grace_dir = Path("grace")
    files = list(grace_dir.rglob("*.py"))
    
    console.print(f"\nProcessing {len(files)} files...")
    
    stats = {
        "any_added": 0,
        "optional_fixed": 0,
        "files_modified": 0
    }
    
    modified_files = []
    
    for file_path in files:
        try:
            modified = False
            
            # Fix 1: Add Any imports
            if add_any_to_file(file_path):
                stats["any_added"] += 1
                modified = True
            
            # Fix 2: Fix Optional defaults  
            if fix_optional_defaults(file_path):
                stats["optional_fixed"] += 1
                modified = True
            
            if modified:
                stats["files_modified"] += 1
                modified_files.append(file_path)
                
        except Exception as e:
            console.print(f"[red]Error in {file_path}: {e}[/red]")
    
    # Display results
    table = Table(title="Fix Summary")
    table.add_column("Fix Type", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("Any imports added", str(stats["any_added"]))
    table.add_row("Optional defaults fixed", str(stats["optional_fixed"]))
    table.add_row("Files modified", str(stats["files_modified"]))
    
    console.print("\n")
    console.print(table)
    
    if modified_files:
        console.print("\n[bold]Modified files:[/bold]")
        for f in modified_files[:10]:
            console.print(f"  • {f.relative_to('.')}")
        if len(modified_files) > 10:
            console.print(f"  ... and {len(modified_files) - 10} more")
    
    console.print("\n[green]✅ Automated fixes complete![/green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
