"""
Automated Pylance error fixes for the entire Grace system
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Fix:
    """Represents a fix applied"""
    file: str
    line: int
    old: str
    new: str
    issue: str


class PylanceFixer:
    """Automated Pylance error fixer"""
    
    def __init__(self):
        self.fixes: List[Fix] = []
    
    def fix_file(self, file_path: Path) -> int:
        """Fix all issues in a file"""
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes_count = 0
        
        # Fix 1: Add missing Any imports
        if 'Any' in content and 'from typing import' in content:
            match = re.search(r'from typing import ([^\n]+)', content)
            if match and 'Any' not in match.group(1):
                old_import = match.group(0)
                new_import = old_import.rstrip() + ', Any'
                content = content.replace(old_import, new_import, 1)
                fixes_count += 1
        
        # Fix 2: Optional without default value
        pattern = r'(\w+):\s*Optional\[([\w\[\],\s]+)\](?!\s*=)'
        for match in re.finditer(pattern, content):
            old_text = match.group(0)
            new_text = f"{old_text} = None"
            content = content.replace(old_text, new_text, 1)
            fixes_count += 1
        
        # Fix 3: Dataclass mutable defaults
        if '@dataclass' in content:
            # Dict defaults
            pattern = r'(\w+):\s*(Dict\[[\w\[\],\s]+\])\s*=\s*\{\}'
            for match in re.finditer(pattern, content):
                old_text = match.group(0)
                var_name = match.group(1)
                type_hint = match.group(2)
                new_text = f"{var_name}: {type_hint} = field(default_factory=dict)"
                content = content.replace(old_text, new_text, 1)
                fixes_count += 1
            
            # List defaults
            pattern = r'(\w+):\s*(List\[[\w\[\],\s]+\])\s*=\s*\[\]'
            for match in re.finditer(pattern, content):
                old_text = match.group(0)
                var_name = match.group(1)
                type_hint = match.group(2)
                new_text = f"{var_name}: {type_hint} = field(default_factory=list)"
                content = content.replace(old_text, new_text, 1)
                fixes_count += 1
            
            # Ensure field is imported
            if 'field(' in content and 'from dataclasses import' in content:
                if 'field' not in re.search(r'from dataclasses import ([^\n]+)', content).group(1):
                    content = re.sub(
                        r'from dataclasses import dataclass',
                        'from dataclasses import dataclass, field',
                        content,
                        count=1
                    )
        
        # Fix 4: Add explicit return types to functions
        # Find functions without return type hints
        pattern = r'def (\w+)\(([^)]*)\):'
        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            # Skip __init__, __repr__, etc.
            if func_name.startswith('__'):
                continue
            old_text = match.group(0)
            # Check if it already has return type
            if '->' not in old_text:
                # Add -> None as default
                new_text = old_text[:-1] + ' -> None:'
                content = content.replace(old_text, new_text, 1)
                fixes_count += 1
        
        # Fix 5: Add type hints to class attributes
        # This is more complex and might need manual review
        
        # Write back if changes were made
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
        
        return fixes_count
    
    def process_directory(self, directory: Path) -> Dict[str, int]:
        """Process all Python files in directory"""
        results = {}
        
        for py_file in directory.rglob("*.py"):
            if any(part.startswith('.') for part in py_file.parts):
                continue
            if '__pycache__' in str(py_file):
                continue
            
            try:
                fixes_count = self.fix_file(py_file)
                if fixes_count > 0:
                    results[str(py_file.relative_to('.'))] = fixes_count
            except Exception as e:
                console.print(f"[red]Error processing {py_file}: {e}[/red]")
        
        return results


def main():
    """Run automated fixes"""
    console.print("\n[bold blue]Automated Pylance Error Fixes[/bold blue]")
    console.print("=" * 70)
    
    fixer = PylanceFixer()
    
    # Process grace directory
    grace_dir = Path("grace")
    
    console.print(f"\nProcessing {grace_dir}...")
    results = fixer.process_directory(grace_dir)
    
    # Display results
    if results:
        table = Table(title="Fixes Applied")
        table.add_column("File", style="cyan")
        table.add_column("Fixes", justify="right", style="green")
        
        total_fixes = 0
        for file_path, count in sorted(results.items()):
            table.add_row(file_path, str(count))
            total_fixes += count
        
        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]Total fixes applied: {total_fixes}[/bold green]")
    else:
        console.print("[yellow]No fixes needed or applied[/yellow]")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Review changes: git diff")
    console.print("  2. Run validation: python scripts/master_validation.py")
    console.print("  3. Run tests: python test_integration_full.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
