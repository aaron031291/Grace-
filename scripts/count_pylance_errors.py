"""
Count remaining Pylance errors by type
"""

import subprocess
import sys
from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Count Pylance errors"""
    console.print("\n[bold blue]Pylance Error Analysis[/bold blue]")
    console.print("=" * 70)
    
    # Run pyright to get errors (if installed)
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", "grace"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        import json
        data = json.loads(result.stdout)
        
        if "generalDiagnostics" in data:
            diagnostics = data["generalDiagnostics"]
            
            console.print(f"\nTotal errors: {len(diagnostics)}")
            
            # Group by rule
            rules = Counter(d.get("rule", "unknown") for d in diagnostics)
            
            table = Table(title="Errors by Type")
            table.add_column("Error Type", style="cyan")
            table.add_column("Count", justify="right", style="red")
            
            for rule, count in rules.most_common(10):
                table.add_row(rule, str(count))
            
            console.print("\n")
            console.print(table)
            
            # Show some examples
            console.print("\n[bold]Sample Errors:[/bold]")
            for diag in diagnostics[:5]:
                file = diag.get("file", "unknown")
                line = diag.get("range", {}).get("start", {}).get("line", 0)
                msg = diag.get("message", "")
                console.print(f"  {Path(file).name}:{line} - {msg[:80]}")
            
        else:
            console.print("[yellow]No diagnostics found[/yellow]")
            
    except FileNotFoundError:
        console.print("[yellow]Pyright not installed, using manual count[/yellow]")
        console.print("\nManual check:")
        console.print("  1. Open VS Code")
        console.print("  2. View → Problems")
        console.print("  3. Count Pylance errors")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
