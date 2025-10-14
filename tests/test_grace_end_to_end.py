#!/usr/bin/env python3
"""Comprehensive Grace End-to-End Test - Initial Discovery Run"""

import sys
import datetime as dt
import zoneinfo

print("=" * 70)
print("GRACE END-TO-END TEST - INITIAL DISCOVERY")
print("=" * 70)

# Test 1: Basic imports
print("\nTest 1: Testing basic Python imports...")
try:
    from pydantic import BaseModel
    print("✅ pydantic available")
except ImportError as e:
    print(f"❌ pydantic not available: {e}")

try:
    import pytest
    print("✅ pytest available")
except ImportError as e:
    print(f"❌ pytest not available: {e}")

# Test 2: Timezone support
print("\nTest 2: Testing timezone support...")
try:
    UTC = zoneinfo.ZoneInfo("UTC")
    SYDNEY = zoneinfo.ZoneInfo("Australia/Sydney")
    now_utc = dt.datetime.now(UTC)
    now_sydney = dt.datetime.now(SYDNEY)
    print(f"✅ Timezone support works")
    print(f"   UTC: {now_utc.isoformat()}")
    print(f"   Sydney: {now_sydney.isoformat()}")
except Exception as e:
    print(f"❌ Timezone error: {e}")

# Test 3: Grace module imports
print("\nTest 3: Testing Grace module imports...")
try:
    from grace.intelligence.intelligence_service import IntelligenceService
    print("✅ IntelligenceService imported")
except Exception as e:
    print(f"❌ IntelligenceService import failed: {e}")

try:
    from grace.intelligence.governance_bridge import GovernanceBridge
    print("✅ GovernanceBridge imported")
except Exception as e:
    print(f"❌ GovernanceBridge import failed: {e}")

try:
    from grace.intelligence.kernel.kernel import IntelligenceKernel
    print("✅ IntelligenceKernel imported")
except Exception as e:
    print(f"❌ IntelligenceKernel import failed: {e}")

print("\n" + "=" * 70)
print("Discovery complete. Check output above for errors.")
print("=" * 70)
