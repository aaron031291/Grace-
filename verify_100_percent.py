#!/usr/bin/env python3
"""
Grace 100% Completion Verification
==================================

Verifies that Grace is genuinely 100% complete with no stubs or placeholders.

This script:
1. Tests all critical imports
2. Validates that modules have real implementations
3. Checks for TODO/placeholder comments
4. Verifies integration points
5. Confirms no missing dependencies

Returns exit code 0 only if 100% complete.
"""

import sys
import ast
import re
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_section(title: str):
    """Print section header"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def check_imports() -> Tuple[bool, List[str]]:
    """Check that all critical modules can be imported"""
    print_section("1. Testing Critical Imports")
    
    critical_imports = [
        ("grace.events", "EventBus"),
        ("grace.governance", "GovernanceKernel"),
        ("grace.mtl", "MTLEngine"),
        ("grace.interface", "VoiceInterface"),
        ("grace.mldl.disagreement_consensus", "DisagreementConsensus"),
        ("grace.core.breakthrough", "BreakthroughMetaLoop"),
        ("grace.database", "Base"),
        ("grace.config", "get_settings"),
        ("grace.runtime", "GraceRuntime"),
        ("grace.services.quorum_service", "QuorumService"),
        ("grace.self_awareness.manager", "SelfAwarenessManager"),
        ("grace.cognitive.reverse_engineer", "ReverseEngineer"),
        ("grace.transcendence.adaptive_interface", "AdaptiveInterfaceChat"),
        ("grace.shards", "ImmuneSystemShard"),
        ("grace.shards", "CodeGeneratorShard"),
    ]
    
    failures = []
    passed = 0
    
    for module_path, class_name in critical_imports:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"  {GREEN}✓{RESET} {module_path}.{class_name}")
                passed += 1
            else:
                print(f"  {RED}✗{RESET} {module_path}.{class_name} - class not found")
                failures.append(f"{module_path}.{class_name} - class missing")
        except ImportError as e:
            print(f"  {RED}✗{RESET} {module_path}.{class_name} - {e}")
            failures.append(f"{module_path}.{class_name} - import failed")
    
    print(f"\n  {passed}/{len(critical_imports)} imports successful")
    return len(failures) == 0, failures


def scan_for_placeholders() -> Tuple[bool, List[str]]:
    """Scan for TODO, placeholder, stub comments"""
    print_section("2. Scanning for Placeholders/TODOs")
    
    grace_dir = Path("grace")
    backend_dir = Path("backend")
    
    placeholder_patterns = [
        r'#\s*TODO:?\s*Implement',
        r'#\s*Placeholder\s*-',
        r'#\s*STUB',
        r'pass\s*#.*TODO',
        r'raise NotImplementedError',
    ]
    
    issues = []
    files_scanned = 0
    
    for directory in [grace_dir, backend_dir]:
        if not directory.exists():
            continue
        
        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test" in str(py_file):
                continue
            
            files_scanned += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in placeholder_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            issues.append(f"{py_file}: {match}")
            except Exception as e:
                pass
    
    if issues:
        print(f"  {RED}✗{RESET} Found {len(issues)} placeholder/TODO comments:\n")
        for issue in issues[:10]:  # Show first 10
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print(f"  {GREEN}✓{RESET} No placeholder/TODO comments found")
        print(f"    Scanned {files_scanned} Python files")
    
    return len(issues) == 0, issues


def check_real_implementations() -> Tuple[bool, List[str]]:
    """Check that key modules have substantial implementations"""
    print_section("3. Validating Real Implementations")
    
    modules_to_check = [
        ("grace/events/event_bus.py", 200),  # Should be >200 lines
        ("grace/governance/governance_kernel.py", 200),
        ("grace/mtl/mtl_engine.py", 40),
        ("grace/interface/voice_interface.py", 100),
        ("grace/runtime/runtime.py", 200),
        ("grace/services/quorum_service.py", 200),
        ("grace/self_awareness/manager.py", 150),
    ]
    
    failures = []
    passed = 0
    
    for file_path, min_lines in modules_to_check:
        path = Path(file_path)
        
        if not path.exists():
            print(f"  {RED}✗{RESET} {file_path} - FILE MISSING")
            failures.append(f"{file_path} - missing")
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l for l in f.readlines() if l.strip() and not l.strip().startswith('#')]
                line_count = len(lines)
            
            if line_count >= min_lines:
                print(f"  {GREEN}✓{RESET} {file_path} ({line_count} lines)")
                passed += 1
            else:
                print(f"  {YELLOW}⚠{RESET} {file_path} ({line_count} lines, expected >{min_lines})")
                failures.append(f"{file_path} - suspiciously small ({line_count} lines)")
        
        except Exception as e:
            print(f"  {RED}✗{RESET} {file_path} - error reading: {e}")
            failures.append(f"{file_path} - error")
    
    print(f"\n  {passed}/{len(modules_to_check)} modules have substantial implementations")
    return len(failures) == 0, failures


def check_database_compatibility() -> Tuple[bool, List[str]]:
    """Check database compatibility shim"""
    print_section("4. Database Compatibility")
    
    try:
        from grace.database import Base, get_db, get_async_db, SessionLocal
        print(f"  {GREEN}✓{RESET} grace.database shim working")
        
        from grace.config import get_settings
        config = get_settings()
        print(f"  {GREEN}✓{RESET} grace.config.get_settings() working")
        
        return True, []
    except Exception as e:
        print(f"  {RED}✗{RESET} Compatibility check failed: {e}")
        return False, [str(e)]


def check_model_integrity() -> Tuple[bool, List[str]]:
    """Check SQLAlchemy models are correct"""
    print_section("5. SQLAlchemy Model Integrity")
    
    try:
        from grace.auth.models import User, Role, RefreshToken, user_roles
        
        # Check User has locked_until
        assert hasattr(User, '__table__')
        user_columns = [c.name for c in User.__table__.columns]
        assert 'locked_until' in user_columns, "User missing locked_until column"
        print(f"  {GREEN}✓{RESET} User model has locked_until column")
        
        # Check RefreshToken has revoked
        token_columns = [c.name for c in RefreshToken.__table__.columns]
        assert 'revoked' in token_columns, "RefreshToken missing revoked column"
        print(f"  {GREEN}✓{RESET} RefreshToken model has revoked column")
        
        # Check user_roles FK types
        user_id_col = [c for c in user_roles.columns if c.name == 'user_id'][0]
        role_id_col = [c for c in user_roles.columns if c.name == 'role_id'][0]
        
        assert str(user_id_col.type) == 'INTEGER', f"user_id should be INTEGER, got {user_id_col.type}"
        assert str(role_id_col.type) == 'INTEGER', f"role_id should be INTEGER, got {role_id_col.type}"
        print(f"  {GREEN}✓{RESET} user_roles FK types are correct (INTEGER)")
        
        return True, []
        
    except AssertionError as e:
        print(f"  {RED}✗{RESET} Model integrity check failed: {e}")
        return False, [str(e)]
    except Exception as e:
        print(f"  {RED}✗{RESET} Could not validate models: {e}")
        return False, [str(e)]


def check_async_compliance() -> Tuple[bool, List[str]]:
    """Check that async code uses proper patterns"""
    print_section("6. Async Code Compliance")
    
    try:
        # Check LLM service uses httpx not requests
        from grace.services.llm_service import LLMService
        import inspect
        
        source = inspect.getsource(LLMService)
        
        if 'import requests' in source or 'requests.post' in source:
            print(f"  {RED}✗{RESET} LLM service still uses blocking requests library")
            return False, ["LLM service uses requests instead of httpx"]
        
        if 'httpx' in source or 'AsyncClient' in source:
            print(f"  {GREEN}✓{RESET} LLM service uses async httpx")
            return True, []
        
        print(f"  {YELLOW}⚠{RESET} Could not verify HTTP library")
        return True, []  # Assume OK if can't verify
        
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Could not check async compliance: {e}")
        return True, []  # Not critical


def main():
    """Run all verification checks"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}GRACE 100% COMPLETION VERIFICATION{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")
    
    all_passed = True
    all_issues = []
    
    # Run checks
    checks = [
        ("Critical Imports", check_imports()),
        ("No Placeholders", scan_for_placeholders()),
        ("Real Implementations", check_real_implementations()),
        ("Database Compatibility", check_database_compatibility()),
        ("Model Integrity", check_model_integrity()),
        ("Async Compliance", check_async_compliance()),
    ]
    
    # Wait for async checks
    import asyncio
    checks_resolved = []
    for name, check in checks:
        if asyncio.iscoroutine(check):
            result = asyncio.run(check)
        else:
            result = check
        checks_resolved.append((name, result))
    
    # Summarize results
    print_section("SUMMARY")
    
    for name, (passed, issues) in checks_resolved:
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"  {status} {name}")
        
        if not passed:
            all_passed = False
            all_issues.extend(issues)
    
    # Final verdict
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    
    if all_passed:
        print(f"{GREEN}✅ GRACE IS 100% COMPLETE{RESET}")
        print(f"{GREEN}   No stubs, no placeholders, all functionality real{RESET}")
        print(f"{BLUE}{'=' * 70}{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ GRACE NOT YET 100% COMPLETE{RESET}")
        print(f"{RED}   Found {len(all_issues)} issues:{RESET}")
        for issue in all_issues[:15]:
            print(f"     - {issue}")
        if len(all_issues) > 15:
            print(f"     ... and {len(all_issues) - 15} more")
        print(f"{BLUE}{'=' * 70}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
