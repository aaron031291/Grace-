"""
Forensic Analysis - Deep dive into all warnings and errors
"""

import ast
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import subprocess


class ForensicAnalyzer:
    """Deep forensic analysis of codebase issues"""
    
    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir)
        self.grace_dir = self.root / "grace"
        self.issues = defaultdict(list)
        
    def analyze_all(self):
        """Run all forensic checks"""
        print("🔍 Starting Forensic Analysis...")
        print("=" * 80)
        
        # 1. Check for syntax errors
        print("\n1️⃣  Checking for syntax errors...")
        self._check_syntax_errors()
        
        # 2. Check for import issues
        print("\n2️⃣  Checking import issues...")
        self._check_import_issues()
        
        # 3. Check for undefined names
        print("\n3️⃣  Checking undefined names...")
        self._check_undefined_names()
        
        # 4. Check for type issues
        print("\n4️⃣  Checking type issues...")
        self._check_type_issues()
        
        # 5. Check for unused imports
        print("\n5️⃣  Checking unused imports...")
        self._check_unused_imports()
        
        # 6. Check for missing __init__.py
        print("\n6️⃣  Checking missing __init__.py files...")
        self._check_missing_init_files()
        
        # 7. Check circular imports
        print("\n7️⃣  Checking circular imports...")
        self._check_circular_imports()
        
        # 8. Generate report
        print("\n📊 Generating report...")
        return self._generate_report()
    
    def _check_syntax_errors(self):
        """Check for Python syntax errors"""
        for py_file in self.grace_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    ast.parse(f.read(), filename=str(py_file))
            except SyntaxError as e:
                self.issues['syntax_errors'].append({
                    'file': str(py_file),
                    'line': e.lineno,
                    'error': str(e),
                    'severity': 'critical'
                })
    
    def _check_import_issues(self):
        """Check for import problems"""
        for py_file in self.grace_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module:
                            # Check if module exists
                            parts = node.module.split('.')
                            if parts[0] == 'grace':
                                module_path = self.grace_dir
                                for part in parts[1:]:
                                    module_path = module_path / part
                                
                                if not module_path.exists() and not (module_path.parent / f"{part}.py").exists():
                                    self.issues['import_errors'].append({
                                        'file': str(py_file),
                                        'line': node.lineno,
                                        'module': node.module,
                                        'error': f"Module not found: {node.module}",
                                        'severity': 'high'
                                    })
            except Exception as e:
                pass
    
    def _check_undefined_names(self):
        """Check for undefined variable/function names"""
        for py_file in self.grace_dir.rglob("*.py"):
            try:
                result = subprocess.run(
                    ['python', '-m', 'pyflakes', str(py_file)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    for line in result.stdout.split('\n'):
                        if 'undefined name' in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                self.issues['undefined_names'].append({
                                    'file': str(py_file),
                                    'line': parts[1],
                                    'error': ':'.join(parts[2:]),
                                    'severity': 'high'
                                })
            except Exception:
                pass
    
    def _check_type_issues(self):
        """Check for type-related issues using mypy"""
        try:
            result = subprocess.run(
                ['mypy', 'grace/', '--ignore-missing-imports', '--no-error-summary'],
                capture_output=True,
                text=True,
                cwd=self.root
            )
            
            for line in result.stdout.split('\n'):
                if line.strip() and ':' in line:
                    parts = line.split(':', 3)
                    if len(parts) >= 4:
                        severity = 'medium' if 'note' in line else 'high'
                        self.issues['type_issues'].append({
                            'file': parts[0],
                            'line': parts[1],
                            'error': parts[3].strip(),
                            'severity': severity
                        })
        except Exception as e:
            print(f"   ⚠️  Mypy not available: {e}")
    
    def _check_unused_imports(self):
        """Check for unused imports"""
        for py_file in self.grace_dir.rglob("*.py"):
            try:
                result = subprocess.run(
                    ['python', '-m', 'pyflakes', str(py_file)],
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.split('\n'):
                    if 'imported but unused' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            self.issues['unused_imports'].append({
                                'file': str(py_file),
                                'line': parts[1],
                                'error': ':'.join(parts[2:]),
                                'severity': 'low'
                            })
            except Exception:
                pass
    
    def _check_missing_init_files(self):
        """Check for directories without __init__.py"""
        for dir_path in self.grace_dir.rglob("*"):
            if dir_path.is_dir() and not dir_path.name.startswith('.'):
                # Check if it has Python files
                has_py = any(dir_path.glob("*.py"))
                has_init = (dir_path / "__init__.py").exists()
                
                if has_py and not has_init:
                    self.issues['missing_init'].append({
                        'directory': str(dir_path),
                        'error': 'Missing __init__.py file',
                        'severity': 'medium'
                    })
    
    def _check_circular_imports(self):
        """Detect potential circular imports"""
        import_graph = defaultdict(set)
        
        for py_file in self.grace_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())
                
                current_module = str(py_file.relative_to(self.root)).replace('/', '.').replace('.py', '')
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            if node.module.startswith('grace'):
                                import_graph[current_module].add(node.module)
            except Exception:
                pass
        
        # Simple cycle detection
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in import_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    self.issues['circular_imports'].append({
                        'modules': f"{node} -> {neighbor}",
                        'error': 'Circular import detected',
                        'severity': 'high'
                    })
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node in import_graph:
            if node not in visited:
                has_cycle(node, visited, set())
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        critical = sum(
            len([i for i in issues if i.get('severity') == 'critical'])
            for issues in self.issues.values()
        )
        
        high = sum(
            len([i for i in issues if i.get('severity') == 'high'])
            for issues in self.issues.values()
        )
        
        medium = sum(
            len([i for i in issues if i.get('severity') == 'medium'])
            for issues in self.issues.values()
        )
        
        print("\n" + "=" * 80)
        print("📊 FORENSIC ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\n🔴 Critical Issues: {critical}")
        print(f"🟠 High Priority:   {high}")
        print(f"🟡 Medium Priority: {medium}")
        print(f"⚪ Low Priority:    {total_issues - critical - high - medium}")
        print(f"\n📈 Total Issues: {total_issues}")
        
        # Detailed breakdown
        print("\n" + "-" * 80)
        print("DETAILED BREAKDOWN:")
        print("-" * 80)
        
        for category, issues_list in sorted(self.issues.items()):
            if issues_list:
                print(f"\n🔍 {category.upper().replace('_', ' ')} ({len(issues_list)} issues):")
                
                # Show first 5 of each type
                for issue in issues_list[:5]:
                    severity_icon = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '⚪'
                    }.get(issue.get('severity', 'low'), '⚪')
                    
                    print(f"  {severity_icon} {issue}")
                
                if len(issues_list) > 5:
                    print(f"  ... and {len(issues_list) - 5} more")
        
        # Save to file
        report_path = self.root / "forensic_report.json"
        with open(report_path, 'w') as f:
            json.dump(dict(self.issues), f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_path}")
        
        return {
            'total': total_issues,
            'critical': critical,
            'high': high,
            'medium': medium,
            'issues': dict(self.issues)
        }


def main():
    """Run forensic analysis"""
    analyzer = ForensicAnalyzer("/workspaces/Grace-")
    report = analyzer.analyze_all()
    
    # Return exit code based on severity
    if report['critical'] > 0:
        print("\n❌ CRITICAL ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
        return 1
    elif report['high'] > 0:
        print("\n⚠️  HIGH PRIORITY ISSUES FOUND - ACTION RECOMMENDED")
        return 1
    elif report['total'] > 0:
        print("\n⚡ ISSUES FOUND - REVIEW RECOMMENDED")
        return 0
    else:
        print("\n✅ NO ISSUES FOUND - CODE IS CLEAN")
        return 0


if __name__ == "__main__":
    sys.exit(main())
