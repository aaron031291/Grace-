"""
Automatically fix common issues found in forensic analysis
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List


class AutoFixer:
    """Automatically fix common issues"""
    
    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir)
        self.grace_dir = self.root / "grace"
        self.fixes_applied = []
    
    def fix_all(self):
        """Apply all automatic fixes"""
        print("🔧 Starting Automatic Fixes...")
        print("=" * 80)
        
        # Load forensic report
        report_path = self.root / "forensic_report.json"
        if not report_path.exists():
            print("❌ No forensic report found. Run forensic_analysis.py first.")
            return 1
        
        with open(report_path, 'r') as f:
            issues = json.load(f)
        
        # Apply fixes
        print("\n1️⃣  Creating missing __init__.py files...")
        self._fix_missing_init(issues.get('missing_init', []))
        
        print("\n2️⃣  Removing unused imports...")
        self._fix_unused_imports(issues.get('unused_imports', []))
        
        print("\n3️⃣  Adding missing imports...")
        self._fix_missing_imports(issues.get('undefined_names', []))
        
        print("\n4️⃣  Fixing common type issues...")
        self._fix_type_issues(issues.get('type_issues', []))
        
        # Report
        print("\n" + "=" * 80)
        print(f"✅ Applied {len(self.fixes_applied)} fixes")
        
        if self.fixes_applied:
            print("\nFixed:")
            for fix in self.fixes_applied[:10]:
                print(f"  ✓ {fix}")
            if len(self.fixes_applied) > 10:
                print(f"  ... and {len(self.fixes_applied) - 10} more")
        
        return 0
    
    def _fix_missing_init(self, issues: List[Dict]):
        """Create missing __init__.py files"""
        for issue in issues:
            directory = Path(issue['directory'])
            init_file = directory / "__init__.py"
            
            if not init_file.exists():
                # Create __init__.py with appropriate content
                module_name = directory.name
                content = f'''"""
{module_name.replace('_', ' ').title()} module
"""

# Add exports here if needed
__all__ = []
'''
                
                with open(init_file, 'w') as f:
                    f.write(content)
                
                self.fixes_applied.append(f"Created {init_file}")
    
    def _fix_unused_imports(self, issues: List[Dict]):
        """Remove unused imports"""
        # Group by file
        by_file = {}
        for issue in issues:
            file_path = issue['file']
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(issue)
        
        for file_path, file_issues in by_file.items():
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Mark lines to remove
                lines_to_remove = set()
                for issue in file_issues:
                    line_no = int(issue['line']) - 1
                    if 0 <= line_no < len(lines):
                        lines_to_remove.add(line_no)
                
                # Remove lines
                if lines_to_remove:
                    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
                    
                    # Backup original
                    shutil.copy(file_path, f"{file_path}.bak")
                    
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    
                    self.fixes_applied.append(f"Removed unused imports from {file_path}")
            except Exception as e:
                print(f"  ⚠️  Could not fix {file_path}: {e}")
    
    def _fix_missing_imports(self, issues: List[Dict]):
        """Add common missing imports"""
        common_imports = {
            'Optional': 'from typing import Optional',
            'Dict': 'from typing import Dict',
            'List': 'from typing import List',
            'Any': 'from typing import Any',
            'asyncio': 'import asyncio',
            'logging': 'import logging',
            'datetime': 'from datetime import datetime',
        }
        
        by_file = {}
        for issue in issues:
            if 'undefined name' in issue.get('error', ''):
                error_text = issue['error']
                for name, import_stmt in common_imports.items():
                    if f"'{name}'" in error_text:
                        file_path = issue['file']
                        if file_path not in by_file:
                            by_file[file_path] = set()
                        by_file[file_path].add(import_stmt)
        
        for file_path, imports in by_file.items():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Find where to insert imports
                lines = content.split('\n')
                insert_pos = 0
                
                # Skip docstring and find first import
                in_docstring = False
                for i, line in enumerate(lines):
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                    elif not in_docstring and (line.startswith('import ') or line.startswith('from ')):
                        insert_pos = i
                        break
                
                # Insert imports
                for import_stmt in sorted(imports):
                    if import_stmt not in content:
                        lines.insert(insert_pos, import_stmt)
                        insert_pos += 1
                
                # Write back
                with open(file_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                self.fixes_applied.append(f"Added imports to {file_path}")
            except Exception as e:
                print(f"  ⚠️  Could not fix {file_path}: {e}")
    
    def _fix_type_issues(self, issues: List[Dict]):
        """Fix common type annotation issues"""
        # This is complex and risky, so we'll be conservative
        print("  ℹ️  Type issues require manual review")


def main():
    """Run auto-fixer"""
    fixer = AutoFixer("/workspaces/Grace-")
    return fixer.fix_all()


if __name__ == "__main__":
    sys.exit(main())
