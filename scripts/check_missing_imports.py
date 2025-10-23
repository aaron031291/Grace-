"""
Check for missing imports in all Python files
"""

import sys
import ast
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for undefined names"""
    
    def __init__(self):
        self.imported_names: Set[str] = set()
        self.used_names: Set[str] = set()
        self.undefined_names: Set[str] = set()
        
        # Built-in names that don't need imports
        self.builtins = set(dir(__builtins__))
    
    def visit_Import(self, node):
        """Track import statements"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
    
    def visit_ImportFrom(self, node):
        """Track from...import statements"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
    
    def visit_Name(self, node):
        """Track name usage"""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
    
    def check_undefined(self):
        """Find undefined names"""
        for name in self.used_names:
            if name not in self.imported_names and name not in self.builtins:
                self.undefined_names.add(name)


def check_file(file_path: Path) -> Dict[str, List[str]]:
    """Check a single file for missing imports"""
    try:
        content = file_path.read_text()
        tree = ast.parse(content, filename=str(file_path))
        
        checker = ImportChecker()
        checker.visit(tree)
        checker.check_undefined()
        
        # Common type annotations that might be missing
        type_annotations = {'Any', 'Optional', 'List', 'Dict', 'Tuple', 'Union', 'Callable'}
        missing_types = checker.undefined_names & type_annotations
        
        other_undefined = checker.undefined_names - type_annotations
        
        issues = {}
        if missing_types:
            issues['typing'] = list(missing_types)
        if other_undefined:
            issues['other'] = list(other_undefined)
        
        return issues
        
    except Exception as e:
        return {'error': [str(e)]}


def main():
    """Check all Python files"""
    console.print("\n[bold blue]Checking for Missing Imports[/bold blue]")
    console.print("=" * 70)
    
    grace_dir = Path("grace")
    python_files = list(grace_dir.rglob("*.py"))
    
    files_with_issues = {}
    
    for file_path in python_files:
        issues = check_file(file_path)
        if issues:
            files_with_issues[str(file_path)] = issues
    
    if not files_with_issues:
        console.print("[green]✅ No missing imports found![/green]")
        return 0
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Missing Imports", style="red")
    
    for file_path, issues in sorted(files_with_issues.items()):
        for category, names in issues.items():
            table.add_row(
                file_path,
                category,
                ", ".join(sorted(names))
            )
    
    console.print(table)
    
    console.print(f"\n[yellow]⚠️  Found issues in {len(files_with_issues)} files[/yellow]")
    
    console.print("\n[bold]To fix automatically:[/bold]")
    console.print("  python scripts/fix_all_imports.py")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
