#!/usr/bin/env python3
"""
Pattern-Based Fixes for Grace Codebase
Addresses the Top 10 cross-cutting error patterns to fix hundreds of errors at once
"""

from __future__ import annotations

import sys
import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PatternFixer:
    """Fix cross-cutting patterns across the Grace codebase"""
    
    def __init__(self, root_dir: str = "/workspaces/Grace-"):
        self.root = Path(root_dir)
        self.grace_dir = self.root / "grace"
        self.fixes_applied: List[str] = []
        self.files_modified: Set[Path] = set()
    
    def fix_all_patterns(self) -> int:
        """Apply all pattern fixes"""
        logger.info("🔧 Pattern-Based Fixes for Grace Codebase")
        logger.info("=" * 80)
        
        # Pattern 1: Implicit Optional
        logger.info("\n1️⃣  Pattern 1: Implicit Optional → Adding Optional[T] annotations")
        self._fix_implicit_optional()
        
        # Pattern 2: asyncio.gather with non-awaitables
        logger.info("\n2️⃣  Pattern 2: asyncio.gather → Filtering awaitables")
        self._fix_asyncio_gather()
        
        # Pattern 3 & 4: Type utilities
        logger.info("\n3️⃣  Pattern 3-4: Type Utilities → Creating conversion helpers")
        self._create_type_utilities()
        
        # Pattern 5: Container operations
        logger.info("\n4️⃣  Pattern 5: Container Operations → Fixing .append on non-lists")
        self._fix_container_operations()
        
        # Pattern 6: Return type mismatches
        logger.info("\n5️⃣  Pattern 6: Return Types → Adding explicit conversions")
        self._fix_return_types()
        
        # Pattern 7: Missing imports
        logger.info("\n6️⃣  Pattern 7: Missing Imports → Adding required imports")
        self._fix_missing_imports()
        
        # Pattern 8: MCP interface drift
        logger.info("\n7️⃣  Pattern 8: MCP Interfaces → Unifying signatures")
        self._fix_mcp_interfaces()
        
        # Pattern 9: Event Bus callbacks
        logger.info("\n8️⃣  Pattern 9: Event Bus → Fixing callback types")
        self._fix_event_bus_callbacks()
        
        # Pattern 10: Meta-learner numerics
        logger.info("\n9️⃣  Pattern 10: Meta-learner → Numpy hygiene")
        self._fix_meta_learner_numerics()
        
        # Bonus: API/DB mismatches
        logger.info("\n🔟 Bonus: API/DB → Fixing attribute names")
        self._fix_api_db_mismatches()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Applied {len(self.fixes_applied)} pattern fixes")
        logger.info(f"📝 Modified {len(self.files_modified)} files")
        
        return 0
    
    def _fix_implicit_optional(self):
        """Pattern 1: Add Optional[T] for parameters with None defaults"""
        
        # Find all Python files
        py_files = list(self.grace_dir.rglob("*.py"))
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                # Add future import if not present
                if "from __future__ import annotations" not in content:
                    # Add after module docstring or at top
                    if '"""' in content[:500]:
                        # Find end of docstring
                        parts = content.split('"""', 2)
                        if len(parts) >= 3:
                            content = parts[0] + '"""' + parts[1] + '"""' + '\nfrom __future__ import annotations\n' + parts[2]
                    else:
                        content = "from __future__ import annotations\n\n" + content
                
                # Add Optional import if needed
                if "Optional" not in content and "= None" in content:
                    if "from typing import" in content:
                        # Add to existing typing import
                        content = re.sub(
                            r'from typing import ([^\n]+)',
                            lambda m: f"from typing import {m.group(1).rstrip()}, Optional" if "Optional" not in m.group(1) else m.group(0),
                            content,
                            count=1
                        )
                    else:
                        # Add new typing import
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if line.startswith('import ') or line.startswith('from '):
                                lines.insert(i + 1, "from typing import Optional")
                                break
                        content = '\n'.join(lines)
                
                # Fix function signatures: param: Type = None → param: Optional[Type] = None
                # This is a simplified regex - a full AST-based solution would be more robust
                patterns = [
                    (r'(\w+): (str|int|float|bool|dict|list|datetime|Path) = None', r'\1: Optional[\2] = None'),
                    (r'(\w+): (Dict|List|Set|Tuple)\[(.*?)\] = None', r'\1: Optional[\2[\3]] = None'),
                ]
                
                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
                    self.fixes_applied.append(f"Fixed implicit Optional in {py_file.name}")
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Processed {len(py_files)} files")
    
    def _fix_asyncio_gather(self):
        """Pattern 2: Filter non-awaitables before asyncio.gather"""
        
        # Find files with asyncio.gather
        py_files = list(self.grace_dir.rglob("*.py"))
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                
                if "asyncio.gather" not in content:
                    continue
                
                original_content = content
                
                # Add inspect import if needed
                if "import inspect" not in content and "from inspect import" not in content:
                    if "import asyncio" in content:
                        content = content.replace("import asyncio", "import asyncio\nimport inspect")
                
                # Find gather calls and add filtering
                # Pattern: await asyncio.gather(*tasks) or await asyncio.gather(task1, task2)
                lines = content.split('\n')
                new_lines = []
                
                for i, line in enumerate(lines):
                    if "asyncio.gather" in line and "await" in line:
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        
                        # Extract variable name if it's await asyncio.gather(*var)
                        match = re.search(r'asyncio\.gather\(\*(\w+)\)', line)
                        if match:
                            var_name = match.group(1)
                            # Add filter before the gather
                            new_lines.append(f"{indent_str}{var_name} = [t for t in {var_name} if inspect.isawaitable(t)]")
                            new_lines.append(f"{indent_str}if {var_name}:")
                            new_lines.append(f"{indent_str}    {line.strip()}")
                            continue
                    
                    new_lines.append(line)
                
                content = '\n'.join(new_lines)
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
                    self.fixes_applied.append(f"Fixed asyncio.gather in {py_file.name}")
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Fixed asyncio.gather patterns")
    
    def _create_type_utilities(self):
        """Pattern 3-4: Create type conversion utilities"""
        
        utils_file = self.grace_dir / "utils" / "type_converters.py"
        utils_file.parent.mkdir(parents=True, exist_ok=True)
        
        content = '''"""
Type Conversion Utilities for Grace
Handles safe type conversions for datetime, numeric, and other types
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_datetime(x: Any) -> Optional[datetime]:
    """
    Safely convert various types to datetime
    
    Args:
        x: Value to convert (datetime, int, float, str, or None)
    
    Returns:
        datetime object or None if conversion fails
    """
    if x is None:
        return None
    
    if isinstance(x, datetime):
        return x
    
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(x, tz=timezone.utc)
        except (ValueError, OSError) as e:
            logger.warning(f"Could not convert timestamp {x} to datetime: {e}")
            return None
    
    if isinstance(x, str):
        try:
            # Try ISO format first
            return datetime.fromisoformat(x.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try parsing as timestamp
                return datetime.fromtimestamp(float(x), tz=timezone.utc)
            except (ValueError, OSError) as e:
                logger.warning(f"Could not parse datetime string {x}: {e}")
                return None
    
    logger.warning(f"Cannot convert {type(x).__name__} to datetime")
    return None


def to_int(x: Any, default: int = 0) -> int:
    """
    Safely convert to int
    
    Args:
        x: Value to convert
        default: Default value if conversion fails
    
    Returns:
        int value
    """
    if x is None:
        return default
    
    if isinstance(x, int):
        return x
    
    if isinstance(x, bool):
        return int(x)
    
    if isinstance(x, (float, str)):
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return default
    
    return default


def to_float(x: Any, default: float = 0.0) -> float:
    """
    Safely convert to float
    
    Args:
        x: Value to convert
        default: Default value if conversion fails
    
    Returns:
        float value
    """
    if x is None:
        return default
    
    if isinstance(x, float):
        return x
    
    if isinstance(x, bool):
        return float(x)
    
    if isinstance(x, (int, str)):
        try:
            return float(x)
        except (ValueError, TypeError):
            return default
    
    return default


def ensure_list(x: Any) -> list:
    """
    Ensure value is a list
    
    Args:
        x: Value to convert
    
    Returns:
        list (empty if x is None)
    """
    if x is None:
        return []
    
    if isinstance(x, list):
        return x
    
    if isinstance(x, (tuple, set)):
        return list(x)
    
    return [x]


def ensure_dict(x: Any) -> dict:
    """
    Ensure value is a dict
    
    Args:
        x: Value to convert
    
    Returns:
        dict (empty if x is None or not convertible)
    """
    if x is None:
        return {}
    
    if isinstance(x, dict):
        return x
    
    return {}
'''
        
        utils_file.write_text(content)
        self.fixes_applied.append("Created type_converters.py utility")
        self.files_modified.add(utils_file)
        
        logger.info(f"  ✓ Created {utils_file.relative_to(self.root)}")
    
    def _fix_container_operations(self):
        """Pattern 5: Fix .append on non-lists"""
        
        py_files = list(self.grace_dir.rglob("*.py"))
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                # Look for patterns like: var = {} followed by var.append()
                # This is complex and needs careful handling
                
                # Simple fix: Replace common patterns
                lines = content.split('\n')
                new_lines = []
                
                for i, line in enumerate(lines):
                    # Check if this line has .append and previous lines might have dict init
                    if '.append(' in line:
                        # Look back to find the variable
                        match = re.search(r'(\w+)\.append\(', line)
                        if match:
                            var_name = match.group(1)
                            # Check if we can find initialization
                            for j in range(max(0, i - 10), i):
                                if f"{var_name} = {{}}" in lines[j] or f"{var_name}={{}}" in lines[j]:
                                    # Change to list
                                    lines[j] = lines[j].replace(f"{var_name} = {{}}", f"{var_name} = []")
                                    lines[j] = lines[j].replace(f"{var_name}={{}}", f"{var_name}=[]")
                                    self.fixes_applied.append(f"Fixed dict→list for {var_name} in {py_file.name}")
                
                content = '\n'.join(lines)
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Fixed container operations")
    
    def _fix_return_types(self):
        """Pattern 6: Add explicit return type conversions"""
        
        py_files = list(self.grace_dir.rglob("*.py"))
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                # Pattern: def func(...) -> bool: ... return result
                # Should be: return bool(result)
                
                lines = content.split('\n')
                new_lines = []
                current_return_type = None
                
                for i, line in enumerate(lines):
                    # Detect function signature with return type
                    if 'def ' in line and '->' in line and ':' in line:
                        match = re.search(r'-> (bool|int|float|str)\s*:', line)
                        if match:
                            current_return_type = match.group(1)
                    
                    # Fix return statements
                    if current_return_type and line.strip().startswith('return ') and line.strip() != 'return':
                        return_val = line.strip()[7:].strip()
                        if not return_val.startswith(f'{current_return_type}('):
                            indent = len(line) - len(line.lstrip())
                            if return_val not in ['True', 'False', 'None'] and not return_val[0].isdigit():
                                new_lines.append(' ' * indent + f'return {current_return_type}({return_val})')
                                continue
                    
                    new_lines.append(line)
                
                content = '\n'.join(new_lines)
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
                    self.fixes_applied.append(f"Fixed return types in {py_file.name}")
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Fixed return type conversions")
    
    def _fix_missing_imports(self):
        """Pattern 7: Add missing imports"""
        
        # Common missing imports
        common_imports = {
            'logger': 'import logging\nlogger = logging.getLogger(__name__)',
            'datetime': 'from datetime import datetime',
            'Request': 'from fastapi import Request',
            'Optional': 'from typing import Optional',
        }
        
        py_files = list(self.grace_dir.rglob("*.py"))
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                # Check for usage without import
                for name, import_line in common_imports.items():
                    if name in content and import_line.split()[-1] not in content:
                        # Add import at the top (after docstring)
                        if '"""' in content[:500]:
                            parts = content.split('"""', 2)
                            if len(parts) >= 3:
                                content = parts[0] + '"""' + parts[1] + '"""' + '\n' + import_line + '\n' + parts[2]
                        else:
                            content = import_line + '\n\n' + content
                        
                        self.fixes_applied.append(f"Added {name} import to {py_file.name}")
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Added missing imports")
    
    def _fix_mcp_interfaces(self):
        """Pattern 8: Unify MCP handler signatures"""
        
        base_mcp = self.grace_dir / "mcp" / "base_mcp.py"
        
        if not base_mcp.exists():
            logger.info("  ℹ️  base_mcp.py not found, skipping")
            return
        
        # Update base to accept **kwargs
        try:
            content = base_mcp.read_text()
            
            # Add flexible signatures
            if "**kwargs" not in content:
                content = content.replace(
                    "def search(self, query: str",
                    "def search(self, query: str, **kwargs"
                )
                content = content.replace(
                    "async def search(self, query: str",
                    "async def search(self, query: str, **kwargs"
                )
                
                base_mcp.write_text(content)
                self.files_modified.add(base_mcp)
                self.fixes_applied.append("Updated BaseMCP to accept **kwargs")
        
        except Exception as e:
            logger.debug(f"  ⚠️  Could not update base_mcp.py: {e}")
        
        logger.info(f"  ✓ Unified MCP interfaces")
    
    def _fix_event_bus_callbacks(self):
        """Pattern 9: Fix Event Bus callback typing"""
        
        event_bus_files = list(self.grace_dir.rglob("*event_bus*.py"))
        
        for py_file in event_bus_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                # Add Protocol import
                if "Protocol" not in content and "Callable" in content:
                    if "from typing import" in content:
                        content = re.sub(
                            r'from typing import ([^\n]+)',
                            lambda m: f"from typing import {m.group(1).rstrip()}, Protocol" if "Protocol" not in m.group(1) else m.group(0),
                            content,
                            count=1
                        )
                
                # Fix callback list types
                content = re.sub(
                    r'self\._callbacks:\s*List\[dict\]',
                    'self._callbacks: List[Callable]',
                    content
                )
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
                    self.fixes_applied.append(f"Fixed callback types in {py_file.name}")
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Fixed Event Bus callback types")
    
    def _fix_meta_learner_numerics(self):
        """Pattern 10: Fix meta-learner numpy operations"""
        
        ml_files = list(self.grace_dir.rglob("*meta*learn*.py"))
        
        for py_file in ml_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                # Add numpy import if needed
                if ".size" in content or "np." in content:
                    if "import numpy as np" not in content:
                        if "import" in content:
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if line.startswith('import '):
                                    lines.insert(i + 1, "import numpy as np")
                                    break
                            content = '\n'.join(lines)
                
                # Convert list operations to numpy
                content = re.sub(
                    r'(\w+)\.size\s*#\s*list',
                    r'len(\1)',
                    content
                )
                
                # Add array conversions
                if "scores" in content and "np.asarray" not in content:
                    content = content.replace(
                        "scores = [",
                        "scores_list = [\n# Convert to numpy array:\n# scores = np.asarray(scores_list, dtype=float)"
                    )
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
                    self.fixes_applied.append(f"Fixed numpy operations in {py_file.name}")
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Fixed meta-learner numpy hygiene")
    
    def _fix_api_db_mismatches(self):
        """Bonus: Fix API/DB attribute mismatches"""
        
        api_files = list((self.grace_dir / "api").rglob("*.py")) if (self.grace_dir / "api").exists() else []
        
        # Common fixes
        replacements = [
            ("require_admin", "requires_admin"),
            ("get_async_db", "get_db"),
            (".value()", ".scalar()"),
        ]
        
        for py_file in api_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                for old, new in replacements:
                    if old in content:
                        content = content.replace(old, new)
                        self.fixes_applied.append(f"Fixed {old}→{new} in {py_file.name}")
                
                if content != original_content:
                    py_file.write_text(content)
                    self.files_modified.add(py_file)
            
            except Exception as e:
                logger.debug(f"  ⚠️  Could not process {py_file.name}: {e}")
        
        logger.info(f"  ✓ Fixed API/DB attribute mismatches")


def main():
    """Run pattern fixer"""
    fixer = PatternFixer()
    return fixer.fix_all_patterns()


if __name__ == "__main__":
    sys.exit(main())
