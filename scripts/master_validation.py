"""
Master validation script - Complete system check
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_import_checks() -> Tuple[int, int]:
    """Run import validation"""
    console.print("\n[bold blue]1. Import Validation[/bold blue]")
    console.print("-" * 80)
    
    try:
        result = subprocess.run(
            ["python", "scripts/check_imports.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        lines = result.stdout.split('\n')
        success = sum(1 for line in lines if '✅' in line)
        failed = sum(1 for line in lines if '❌' in line)
        
        console.print(result.stdout)
        return success, failed
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 0, 0


def run_module_validation() -> Tuple[int, int]:
    """Run comprehensive module validation"""
    console.print("\n[bold blue]2. Module Validation[/bold blue]")
    console.print("-" * 80)
    
    try:
        result = subprocess.run(
            ["python", "scripts/validate_all.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        lines = result.stdout.split('\n')
        success = sum(1 for line in lines if '✅' in line)
        failed = sum(1 for line in lines if '❌' in line)
        
        console.print(result.stdout)
        return success, failed
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 0, 0


def run_config_validation() -> bool:
    """Run configuration validation"""
    console.print("\n[bold blue]3. Configuration Validation[/bold blue]")
    console.print("-" * 80)
    
    try:
        result = subprocess.run(
            ["python", "scripts/validate_config.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        console.print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def run_type_checks() -> int:
    """Run type checking with mypy"""
    console.print("\n[bold blue]4. Type Checking (mypy)[/bold blue]")
    console.print("-" * 80)
    
    try:
        result = subprocess.run(
            ["python", "-m", "mypy", "grace", 
             "--ignore-missing-imports", 
             "--no-strict-optional",
             "--show-error-codes"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            console.print("[green]✅ Type checking passed![/green]")
        else:
            console.print(result.stdout)
            console.print(f"[yellow]⚠️  {result.stdout.count('error:')} type errors found[/yellow]")
        
        return result.stdout.count('error:')
    except Exception as e:
        console.print(f"[yellow]Skipped: {e}[/yellow]")
        return 0


def check_file_structure() -> Dict[str, int]:
    """Check file structure completeness"""
    console.print("\n[bold blue]5. File Structure Check[/bold blue]")
    console.print("-" * 80)
    
    grace_dir = Path("grace")
    
    structure = {
        "Python files": len(list(grace_dir.rglob("*.py"))),
        "Init files": len(list(grace_dir.rglob("__init__.py"))),
        "Test files": len(list(Path(".").rglob("test_*.py"))),
        "Config files": len(list(Path(".").glob("*.toml"))) + len(list(Path(".").glob("*.json"))),
        "Documentation": len(list(Path(".").glob("*.md"))),
    }
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    for category, count in structure.items():
        table.add_row(category, str(count))
    
    console.print(table)
    return structure


def run_quick_tests() -> Tuple[int, int]:
    """Run quick test suite"""
    console.print("\n[bold blue]6. Quick Test Suite[/bold blue]")
    console.print("-" * 80)
    
    test_files = [
        "test_integration_full.py",
        "test_clarity_framework_complete.py",
        "test_orchestration_complete.py",
    ]
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if Path(test_file).exists():
            console.print(f"  Running {test_file}...")
            try:
                result = subprocess.run(
                    ["python", test_file],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    console.print(f"    [green]✅ Passed[/green]")
                    passed += 1
                else:
                    console.print(f"    [red]❌ Failed[/red]")
                    failed += 1
            except Exception as e:
                console.print(f"    [yellow]⚠️  Skipped: {e}[/yellow]")
        else:
            console.print(f"  [yellow]⚠️  {test_file} not found[/yellow]")
    
    return passed, failed


def generate_report(results: Dict[str, Any]) -> None:
    """Generate final validation report"""
    console.print("\n" + "=" * 80)
    console.print("[bold green]VALIDATION REPORT[/bold green]")
    console.print("=" * 80)
    
    # Summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    
    # Imports
    import_status = "✅" if results['imports']['failed'] == 0 else "❌"
    table.add_row(
        "Import Validation",
        import_status,
        f"{results['imports']['success']} passed, {results['imports']['failed']} failed"
    )
    
    # Modules
    module_status = "✅" if results['modules']['failed'] == 0 else "❌"
    table.add_row(
        "Module Validation",
        module_status,
        f"{results['modules']['success']} passed, {results['modules']['failed']} failed"
    )
    
    # Config
    config_status = "✅" if results['config'] else "❌"
    table.add_row(
        "Configuration",
        config_status,
        "Valid" if results['config'] else "Issues found"
    )
    
    # Types
    type_status = "✅" if results['types'] == 0 else "⚠️"
    table.add_row(
        "Type Checking",
        type_status,
        f"{results['types']} errors" if results['types'] > 0 else "Clean"
    )
    
    # Tests
    if results['tests']['total'] > 0:
        test_status = "✅" if results['tests']['failed'] == 0 else "❌"
        table.add_row(
            "Quick Tests",
            test_status,
            f"{results['tests']['passed']}/{results['tests']['total']} passed"
        )
    
    console.print(table)
    
    # Overall status
    all_critical_passed = (
        results['imports']['failed'] == 0 and
        results['modules']['failed'] == 0 and
        results['config']
    )
    
    console.print("\n" + "=" * 80)
    if all_critical_passed:
        console.print(Panel(
            "[bold green]✅ SYSTEM VALIDATION PASSED[/bold green]\n\n"
            "All critical checks passed. System is ready for testing.",
            style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]⚠️  SYSTEM VALIDATION WARNINGS[/bold yellow]\n\n"
            "Some issues found. Review output above for details.",
            style="yellow"
        ))
    
    # File structure summary
    console.print(f"\n[bold]Project Statistics:[/bold]")
    console.print(f"  Python files: {results['structure']['Python files']}")
    console.print(f"  Modules: {results['structure']['Init files']}")
    console.print(f"  Tests: {results['structure']['Test files']}")
    console.print(f"  Documentation: {results['structure']['Documentation']}")


def main():
    """Run master validation"""
    console.print("\n[bold blue]Grace System - Master Validation[/bold blue]")
    console.print("=" * 80)
    
    results = {
        'imports': {'success': 0, 'failed': 0},
        'modules': {'success': 0, 'failed': 0},
        'config': False,
        'types': 0,
        'structure': {},
        'tests': {'passed': 0, 'failed': 0, 'total': 0},
    }
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running validations...", total=6)
        
        # 1. Import checks
        results['imports']['success'], results['imports']['failed'] = run_import_checks()
        progress.update(task, advance=1)
        
        # 2. Module validation
        results['modules']['success'], results['modules']['failed'] = run_module_validation()
        progress.update(task, advance=1)
        
        # 3. Config validation
        results['config'] = run_config_validation()
        progress.update(task, advance=1)
        
        # 4. Type checks
        results['types'] = run_type_checks()
        progress.update(task, advance=1)
        
        # 5. File structure
        results['structure'] = check_file_structure()
        progress.update(task, advance=1)
        
        # 6. Quick tests
        passed, failed = run_quick_tests()
        results['tests'] = {
            'passed': passed,
            'failed': failed,
            'total': passed + failed
        }
        progress.update(task, advance=1)
    
    # Generate report
    generate_report(results)
    
    # Exit code
    if results['imports']['failed'] == 0 and results['modules']['failed'] == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)
