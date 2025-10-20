"""
Run full Pylance diagnostics and generate report
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def run_pyright_check() -> Tuple[str, int]:
    """Run pyright/pylance check"""
    try:
        result = subprocess.run(
            ["npx", "pyright", "grace", "--outputjson"],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout, result.returncode
    except FileNotFoundError:
        # Fallback to python mypy
        console.print("[yellow]Pyright not found, using mypy instead[/yellow]")
        result = subprocess.run(
            ["python", "-m", "mypy", "grace", "--show-error-codes", "--no-error-summary"],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout, result.returncode
    except Exception as e:
        return f"Error: {e}", 1


def parse_diagnostics(output: str) -> Dict[str, List[str]]:
    """Parse diagnostic output"""
    diagnostics = defaultdict(list)
    
    for line in output.split('\n'):
        if 'error:' in line.lower() or 'warning:' in line.lower():
            # Extract file path and error
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 3:
                    file_path = parts[0]
                    error = ':'.join(parts[2:]).strip()
                    diagnostics[file_path].append(error)
    
    return diagnostics


def main():
    """Run full diagnostics"""
    console.print("\n[bold blue]Grace System - Full Diagnostic Report[/bold blue]")
    console.print("=" * 80)
    
    # Run type checking
    console.print("\n[cyan]Running type checking...[/cyan]")
    output, returncode = run_pyright_check()
    
    if returncode == 0:
        console.print(Panel(
            "[green]✅ No type errors found![/green]",
            style="green"
        ))
        return 0
    
    # Parse diagnostics
    diagnostics = parse_diagnostics(output)
    
    if not diagnostics:
        console.print("\n[yellow]No structured diagnostics found. Raw output:[/yellow]")
        console.print(output)
        return 1
    
    # Display diagnostics by category
    console.print(f"\n[bold red]Found {sum(len(v) for v in diagnostics.values())} issues in {len(diagnostics)} files[/bold red]")
    
    # Group by error type
    error_types = defaultdict(int)
    for errors in diagnostics.values():
        for error in errors:
            if 'reportMissingImports' in error:
                error_types['Missing Imports'] += 1
            elif 'reportUndefinedVariable' in error:
                error_types['Undefined Variable'] += 1
            elif 'reportGeneralTypeIssues' in error:
                error_types['Type Issues'] += 1
            elif 'reportOptional' in error:
                error_types['Optional Issues'] += 1
            else:
                error_types['Other'] += 1
    
    # Display summary table
    table = Table(title="Error Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="red")
    
    for category, count in sorted(error_types.items(), key=lambda x: -x[1]):
        table.add_row(category, str(count))
    
    console.print("\n", table)
    
    # Display detailed errors
    console.print("\n[bold]Detailed Errors:[/bold]")
    for file_path, errors in sorted(diagnostics.items())[:10]:  # Show first 10 files
        console.print(f"\n[cyan]{file_path}[/cyan]")
        for error in errors[:5]:  # Show first 5 errors per file
            console.print(f"  [red]•[/red] {error}")
        if len(errors) > 5:
            console.print(f"  [yellow]... and {len(errors) - 5} more errors[/yellow]")
    
    if len(diagnostics) > 10:
        console.print(f"\n[yellow]... and {len(diagnostics) - 10} more files with errors[/yellow]")
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    console.print("  1. Run: python scripts/fix_all_imports.py")
    console.print("  2. Run: python scripts/fix_all_types.py")
    console.print("  3. Check: python scripts/check_missing_imports.py")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
