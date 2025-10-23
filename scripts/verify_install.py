"""
Verify Grace installation
"""

import sys

def main():
    print("Verifying Grace installation...")
    print("=" * 60)
    
    errors = []
    
    # Test import
    try:
        import grace
        print("✅ grace package imports successfully")
    except ImportError as e:
        errors.append(f"Cannot import grace: {e}")
        print(f"❌ Cannot import grace: {e}")
    
    # Test submodules
    modules_to_test = [
        "grace.config",
        "grace.api",
        "grace.auth",
        "grace.events.schema",
        "grace.events.factory",
        "grace.integration.event_bus",
        "grace.governance.engine",
        "grace.trust.core",
        "grace.memory.async_lightning",
        "grace.memory.async_fusion",
        "grace.memory.immutable_logs_async",
        "grace.core.unified_service",
        "grace.demo.multi_os_kernel",
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            errors.append(f"{module}: {e}")
            print(f"❌ {module}: {e}")
    
    print("\n" + "=" * 60)
    
    if errors:
        print(f"\n❌ Found {len(errors)} import errors")
        print("\nTo fix, run:")
        print("  pip install -e .")
        return 1
    else:
        print("\n✅ All imports successful!")
        print("\nYou can now run:")
        print("  pytest tests/test_complete_system.py -v")
        return 0


if __name__ == "__main__":
    sys.exit(main())
