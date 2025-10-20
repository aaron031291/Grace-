"""
Fix all missing imports throughout Grace system
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


# Map of common missing imports
IMPORT_FIXES = {
    'grace/clarity/quorum_bridge.py': [
        ('from grace.mldl.quorum_aggregator import', [
            'MLDLQuorumAggregator',
            'SpecialistOutput',
            'ConsensusMethod',
            'QuorumResult'
        ])
    ],
    'grace/avn/pushback.py': [
        ('from typing import', ['Any'])
    ],
    'grace/middleware/logging.py': [
        ('from typing import', ['Any'])
    ],
    'grace/middleware/rate_limit.py': [
        ('from typing import', ['Any'])
    ],
    'grace/middleware/metrics.py': [
        ('from typing import', ['Any']),
        ('from fastapi import', ['Response'])
    ],
    'grace/api/public.py': [
        ('from typing import', ['Any'])
    ],
    'grace/swarm/coordinator.py': [
        ('from typing import', ['Any'])
    ],
    'grace/swarm/transport.py': [
        ('from typing import', ['Any'])
    ],
    'grace/swarm/consensus.py': [
        ('from typing import', ['Any'])
    ],
    'grace/swarm/discovery.py': [
        ('from typing import', ['Any'])
    ],
    'grace/integration/swarm_transcendence_integration.py': [
        ('from typing import', ['Any'])
    ],
    'grace/transcendence/quantum_library.py': [
        ('from typing import', ['Any'])
    ],
    'grace/transcendence/scientific_discovery.py': [
        ('from typing import', ['Any'])
    ],
    'grace/transcendence/societal_impact.py': [
        ('from typing import', ['Any'])
    ],
    'grace/observability/structured_logging.py': [
        ('from typing import', ['Any'])
    ],
    'grace/observability/kpi_monitor.py': [
        ('from typing import', ['Any'])
    ],
}


def add_import_to_file(file_path: Path, import_line: str, items: List[str]) -> bool:
    """Add import to file if missing"""
    try:
        content = file_path.read_text()
        
        # Check if import line exists
        if import_line not in content:
            # Add new import line at top after docstring
            lines = content.split('\n')
            insert_pos = 0
            
            # Skip docstring
            in_docstring = False
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    if not in_docstring:
                        insert_pos = i + 1
                        break
            
            # Find import section
            for i in range(insert_pos, len(lines)):
                if lines[i].startswith('import ') or lines[i].startswith('from '):
                    insert_pos = i
                    break
            
            new_line = f"{import_line} {', '.join(items)}"
            lines.insert(insert_pos, new_line)
            file_path.write_text('\n'.join(lines))
            return True
        else:
            # Check if items are in existing import
            import_pattern = re.escape(import_line) + r'\s+([^\\n]+)'
            match = re.search(import_pattern, content)
            
            if match:
                existing_imports = match.group(1)
                missing_items = [item for item in items if item not in existing_imports]
                
                if missing_items:
                    # Add missing items to existing import
                    new_imports = existing_imports.rstrip() + ', ' + ', '.join(missing_items)
                    new_line = f"{import_line} {new_imports}"
                    content = re.sub(import_pattern, new_line, content)
                    file_path.write_text(content)
                    return True
        
        return False
        
    except Exception as e:
        console.print(f"[red]Error fixing {file_path}: {e}[/red]")
        return False


def main():
    """Fix all imports"""
    console.print("\n[bold blue]Fixing All Import Issues[/bold blue]")
    console.print("=" * 70)
    
    fixed_count = 0
    total_files = len(IMPORT_FIXES)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    
    for file_path_str, imports in IMPORT_FIXES.items():
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            table.add_row(file_path_str, "⚠️", "File not found")
            continue
        
        file_fixed = False
        details = []
        
        for import_line, items in imports:
            if add_import_to_file(file_path, import_line, items):
                file_fixed = True
                details.append(f"Added: {', '.join(items)}")
        
        if file_fixed:
            fixed_count += 1
            table.add_row(
                file_path_str,
                "✅",
                "; ".join(details)
            )
        else:
            table.add_row(file_path_str, "✓", "Already correct")
    
    console.print(table)
    
    console.print(f"\n[green]✅ Fixed imports in {fixed_count}/{total_files} files[/green]")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Run: python scripts/master_validation.py")
    console.print("  2. Check remaining Pylance errors")
    console.print("  3. Run: python scripts/check_imports.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
