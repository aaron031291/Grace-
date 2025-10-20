"""
Comprehensive analysis of all Pylance problems
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_rich_available():
    """Check if rich is available"""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import track
        return Console(), Table, Panel, track
    except ImportError:
        return None, None, None, None


console, Table, Panel, track = check_rich_available()


def print_formatted(text: str, style: str = ""):
    """Print with or without rich"""
    if console:
        console.print(text)
    else:
        print(text.replace('[bold]', '').replace('[/bold]', '').replace('[red]', '').replace('[/red]', '').replace('[green]', '').replace('[/green]', '').replace('[yellow]', '').replace('[/yellow]', '').replace('[cyan]', '').replace('[/cyan]', ''))


def analyze_python_file(file_path: Path) -> List[Dict[str, str]]:
    """Analyze a single Python file for common issues"""
    issues = []
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        # Check 1: Missing Any import
        if 'Any' in content and 'from typing import' in content:
            typing_line = next((line for line in lines if 'from typing import' in line), None)
            if typing_line and 'Any' not in typing_line:
                issues.append({
                    'type': 'Missing Any import',
                    'file': str(file_path),
                    'fix': 'Add Any to typing imports'
                })
        
        # Check 2: Optional without default
        for i, line in enumerate(lines, 1):
            if 'Optional[' in line and '=' not in line and 'def ' not in line:
                issues.append({
                    'type': 'Optional without default',
                    'file': str(file_path),
                    'line': i,
                    'fix': 'Add = None default'
                })
        
        # Check 3: Mutable default in dataclass
        if '@dataclass' in content:
            for i, line in enumerate(lines, 1):
                if (': Dict' in line and '= {}' in line) or (': List' in line and '= []' in line):
                    issues.append({
                        'type': 'Mutable dataclass default',
                        'file': str(file_path),
                        'line': i,
                        'fix': 'Use field(default_factory=...)'
                    })
        
        # Check 4: Missing return type hints
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def ') and '->' not in line and '__init__' not in line:
                issues.append({
                    'type': 'Missing return type',
                    'file': str(file_path),
                    'line': i,
                    'fix': 'Add return type hint'
                })
        
        # Check 5: Undefined variables (basic check)
        # Look for common patterns like using variables before definition
        
    except Exception as e:
        issues.append({
            'type': 'Parse error',
            'file': str(file_path),
            'error': str(e)
        })
    
    return issues


def run_pyright_json() -> Dict:
    """Run pyright and get JSON output"""
    try:
        result = subprocess.run(
            ['npx', 'pyright', 'grace', '--outputjson'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {'generalDiagnostics': []}
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return {'generalDiagnostics': []}


def analyze_all_problems():
    """Comprehensive analysis of all problems"""
    print_formatted("\n[bold blue]Analyzing All 109 Pylance Problems[/bold blue]")
    print_formatted("=" * 80)
    
    # Method 1: Static file analysis
    print_formatted("\n[cyan]Phase 1: Static Analysis[/cyan]")
    grace_dir = Path("grace")
    python_files = list(grace_dir.rglob("*.py"))
    
    all_issues = []
    for file_path in python_files:
        issues = analyze_python_file(file_path)
        all_issues.extend(issues)
    
    print_formatted(f"Found {len(all_issues)} issues via static analysis")
    
    # Method 2: Run pyright if available
    print_formatted("\n[cyan]Phase 2: Pyright Analysis[/cyan]")
    pyright_results = run_pyright_json()
    
    if pyright_results.get('generalDiagnostics'):
        diagnostics = pyright_results['generalDiagnostics']
        print_formatted(f"Found {len(diagnostics)} issues via pyright")
        
        # Categorize diagnostics
        categories = Counter()
        file_issues = defaultdict(list)
        
        for diag in diagnostics:
            rule = diag.get('rule', 'unknown')
            file_path = diag.get('file', 'unknown')
            message = diag.get('message', '')
            
            categories[rule] += 1
            file_issues[file_path].append({
                'rule': rule,
                'message': message,
                'line': diag.get('range', {}).get('start', {}).get('line', 0)
            })
    else:
        print_formatted("Pyright not available, using static analysis only")
        categories = Counter()
        file_issues = defaultdict(list)
        
        for issue in all_issues:
            categories[issue['type']] += 1
            file_issues[issue['file']].append(issue)
    
    # Display results
    print_formatted("\n[bold]Problem Categories:[/bold]")
    print_formatted("-" * 80)
    
    for category, count in categories.most_common():
        print_formatted(f"  {category:.<50} {count:>5}")
    
    print_formatted("\n[bold]Files with Most Issues:[/bold]")
    print_formatted("-" * 80)
    
    sorted_files = sorted(file_issues.items(), key=lambda x: len(x[1]), reverse=True)
    for file_path, issues in sorted_files[:10]:
        rel_path = Path(file_path).relative_to('.') if Path(file_path).exists() else file_path
        print_formatted(f"  {str(rel_path):.<60} {len(issues):>5} issues")
    
    # Generate fix recommendations
    print_formatted("\n[bold]Fix Recommendations:[/bold]")
    print_formatted("-" * 80)
    
    fixes = []
    
    if 'reportMissingImports' in categories or 'Missing Any import' in categories:
        fixes.append("1. Fix missing imports: python scripts/fix_missing_imports.py")
    
    if 'reportUndefinedVariable' in categories or 'Optional without default' in categories:
        fixes.append("2. Fix undefined variables: python scripts/fix_undefined_vars.py")
    
    if 'reportGeneralTypeIssues' in categories or 'Missing return type' in categories:
        fixes.append("3. Fix type issues: python scripts/fix_type_issues.py")
    
    if 'Mutable dataclass default' in categories:
        fixes.append("4. Fix dataclass defaults: python scripts/fix_dataclass_defaults.py")
    
    for fix in fixes:
        print_formatted(f"  {fix}")
    
    # Create detailed report
    report_path = Path("pylance_problem_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Pylance Problem Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Problem Categories:\n")
        f.write("-" * 80 + "\n")
        for category, count in categories.most_common():
            f.write(f"  {category}: {count}\n")
        
        f.write("\n\nDetailed File Issues:\n")
        f.write("-" * 80 + "\n")
        for file_path, issues in sorted_files:
            f.write(f"\n{file_path} ({len(issues)} issues):\n")
            for issue in issues[:5]:  # First 5 issues per file
                f.write(f"  Line {issue.get('line', '?')}: {issue.get('rule', issue.get('type', 'unknown'))}\n")
                f.write(f"    {issue.get('message', issue.get('fix', 'No details'))}\n")
    
    print_formatted(f"\n[green]Detailed report saved to: {report_path}[/green]")
    
    return categories, file_issues


def main():
    """Main entry point"""
    categories, file_issues = analyze_all_problems()
    
    total_issues = sum(categories.values())
    
    print_formatted("\n" + "=" * 80)
    print_formatted(f"[bold]Total Issues Found: {total_issues}[/bold]")
    print_formatted("=" * 80)
    
    print_formatted("\n[bold]Next Steps:[/bold]")
    print_formatted("1. Review: cat pylance_problem_report.txt")
    print_formatted("2. Install rich for better output: pip install rich")
    print_formatted("3. Run automated fixes: python scripts/fix_all_issues.py")
    print_formatted("4. Manual review of remaining issues")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
