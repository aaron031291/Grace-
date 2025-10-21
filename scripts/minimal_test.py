"""
Minimal test to isolate import issues
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports step by step...")
print("=" * 60)

# Test 1: Basic import
try:
    import grace
    print("✅ Step 1: import grace")
except Exception as e:
    print(f"❌ Step 1 failed: {e}")
    sys.exit(1)

# Test 2: Config import
try:
    from grace.config import settings
    print("✅ Step 2: import grace.config.settings")
except Exception as e:
    print(f"❌ Step 2 failed: {e}")
    sys.exit(1)

# Test 3: Get settings
try:
    from grace.config import get_settings
    print("✅ Step 3: import get_settings")
except Exception as e:
    print(f"❌ Step 3 failed: {e}")
    sys.exit(1)

# Test 4: Create settings instance
try:
    settings = get_settings()
    print(f"✅ Step 4: get_settings() -> {settings.environment}")
except Exception as e:
    print(f"❌ Step 4 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Other modules
modules_to_test = [
    "grace.events.schema",
    "grace.governance.engine",
    "grace.trust.core",
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ Step 5: import {module}")
    except Exception as e:
        print(f"❌ Step 5 failed for {module}: {e}")

print("\n" + "=" * 60)
print("✅ All minimal tests passed!")
