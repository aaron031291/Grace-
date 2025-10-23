"""
Quick diagnostic check without external dependencies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run quick diagnostics"""
    print("\n" + "=" * 70)
    print("Grace System - Quick Diagnostic Check")
    print("=" * 70)
    
    issues = []
    
    # Check 1: Import grace package
    print("\n1. Checking core package...")
    try:
        import grace
        print("   ✓ grace package imports successfully")
    except Exception as e:
        print(f"   ✗ Failed to import grace: {e}")
        issues.append("core_import")
    
    # Check 2: Check config
    print("\n2. Checking configuration...")
    try:
        from grace.config import get_settings
        settings = get_settings()
        print(f"   ✓ Configuration loaded: {settings.environment} environment")
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
        issues.append("config")
    
    # Check 3: Check database
    print("\n3. Checking database...")
    try:
        from grace.database import Base
        print("   ✓ Database models available")
    except Exception as e:
        print(f"   ✗ Database error: {e}")
        issues.append("database")
    
    # Check 4: Check key modules
    print("\n4. Checking key modules...")
    modules_to_check = [
        ("grace.auth", "Authentication"),
        ("grace.clarity", "Clarity Framework"),
        ("grace.mldl", "MLDL Specialists"),
        ("grace.swarm", "Swarm Intelligence"),
        ("grace.transcendence", "Transcendence Layer"),
        ("grace.observability", "Observability"),
    ]
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"   ✓ {description}")
        except Exception as e:
            print(f"   ✗ {description}: {e}")
            issues.append(module_name)
    
    # Summary
    print("\n" + "=" * 70)
    if not issues:
        print("✅ All checks passed!")
        print("\nNext steps:")
        print("  1. Install dependencies: bash scripts/install_dependencies.sh")
        print("  2. Run full validation: python scripts/master_validation.py")
        return 0
    else:
        print(f"⚠️  Found {len(issues)} issues")
        print("\nIssues found in:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  1. Install dependencies: bash scripts/install_dependencies.sh")
        print("  2. Check specific modules for errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
