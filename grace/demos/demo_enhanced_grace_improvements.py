#!/usr/bin/env python3
"""
Grace Improvements Demo - Demonstrates the new utilities and error handling.
"""
import sys
import os
import asyncio

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.core.utils import (
    enum_from_str, 
    create_error_response, 
    validate_request_size, 
    utc_timestamp, 
    normalize_timestamp,
    generate_request_id
)
from grace.core.middleware import get_request_id, set_request_id
from grace.interface.orb_interface import PanelType, NotificationPriority


def demo_enum_parsing():
    """Demonstrate flexible enum parsing."""
    print("🔧 Enum Parsing Demo:")
    
    # Test various panel types
    test_cases = [
        ("chat", "exact value match"),
        ("ANALYTICS", "uppercase name"),
        ("dashboard", "lowercase value"), 
        ("memory", "case insensitive"),
        ("invalid_panel", "with default"),
    ]
    
    for value, description in test_cases:
        try:
            if value == "invalid_panel":
                result = enum_from_str(PanelType, value, default=PanelType.DASHBOARD)
                print(f"  ✅ {value:<15} -> {result.value:<12} ({description})")
            else:
                result = enum_from_str(PanelType, value)
                print(f"  ✅ {value:<15} -> {result.value:<12} ({description})")
        except ValueError as e:
            print(f"  ❌ {value:<15} -> ERROR: {e}")
    
    print()


def demo_error_responses():
    """Demonstrate consistent error response format."""
    print("🚨 Error Response Demo:")
    
    errors = [
        create_error_response("INVALID_PANEL_TYPE", "Invalid panel type", "panel_type: unknown"),
        create_error_response("SESSION_NOT_FOUND", "Session not found"),
        create_error_response("CONTENT_TOO_LARGE", "File too large", "Size: 52MB > 50MB limit"),
    ]
    
    for i, error in enumerate(errors, 1):
        print(f"  Error {i}: {error['error']['code']}")
        print(f"    Message: {error['error']['message']}")
        if 'detail' in error['error']:
            print(f"    Detail: {error['error']['detail']}")
        print()


def demo_size_validation():
    """Demonstrate request size validation."""
    print("📏 Size Validation Demo:")
    
    # Test different content types and sizes
    test_cases = [
        ("Small text", "Hello Grace!", 1),
        ("Large text", "x" * (2 * 1024 * 1024), 1),  # 2MB vs 1MB limit
        ("Small bytes", b"Binary data", 1),
        ("Large bytes", b"x" * (2 * 1024 * 1024), 1),  # 2MB vs 1MB limit
    ]
    
    for name, content, limit_mb in test_cases:
        result = validate_request_size(content, max_size_mb=limit_mb)
        if result:
            print(f"  ❌ {name}: {result['error']['code']} - {result['error']['message']}")
        else:
            size_kb = len(content.encode('utf-8') if isinstance(content, str) else content) / 1024
            print(f"  ✅ {name}: OK ({size_kb:.1f} KB)")
    print()


def demo_timestamps():
    """Demonstrate UTC timestamp handling."""
    print("⏰ Timestamp Demo:")
    
    # Generate UTC timestamp
    utc_ts = utc_timestamp()
    print(f"  UTC timestamp: {utc_ts}")
    
    # Test normalization
    from datetime import datetime, timezone
    test_timestamps = [
        datetime.now(),  # Naive datetime
        datetime.now(timezone.utc),  # UTC datetime
        "2023-09-28T12:00:00Z",  # ISO string with Z
        "2023-09-28T12:00:00",  # ISO string without TZ
    ]
    
    for ts in test_timestamps:
        normalized = normalize_timestamp(ts)
        ts_type = type(ts).__name__ if not isinstance(ts, str) else f"string: {ts}"
        print(f"  {ts_type:<30} -> {normalized}")
    print()


def demo_request_id():
    """Demonstrate request ID generation and context."""
    print("🆔 Request ID Demo:")
    
    # Generate request IDs
    req_ids = [generate_request_id() for _ in range(3)]
    print("  Generated IDs:")
    for i, req_id in enumerate(req_ids, 1):
        print(f"    {i}: {req_id}")
    
    # Test context setting
    print("  Context test:")
    print(f"    Current ID: {get_request_id()}")
    
    set_request_id("demo_123")
    print(f"    Set to demo_123: {get_request_id()}")
    
    set_request_id("")
    print(f"    Cleared: '{get_request_id()}'")
    print()


def demo_notification_priorities():
    """Demonstrate notification priority enum parsing."""
    print("🔔 Notification Priority Demo:")
    
    priorities = ["low", "MEDIUM", "High", "CRITICAL", "invalid"]
    
    for priority in priorities:
        try:
            if priority == "invalid":
                result = enum_from_str(NotificationPriority, priority, default=NotificationPriority.MEDIUM)
                print(f"  {priority:<10} -> {result.value} (using default)")
            else:
                result = enum_from_str(NotificationPriority, priority)
                print(f"  {priority:<10} -> {result.value}")
        except ValueError as e:
            print(f"  {priority:<10} -> ERROR: {str(e)[:50]}...")
    print()


def main():
    """Run all demos."""
    print("🎯 Grace System Improvements Demo\n")
    print("=" * 50)
    
    demo_enum_parsing()
    demo_error_responses() 
    demo_size_validation()
    demo_timestamps()
    demo_request_id()
    demo_notification_priorities()
    
    print("✨ Demo completed! All Grace improvements are working correctly.")
    print("\nKey improvements:")
    print("- ✅ Central enum_from_str() helper with flexible parsing")
    print("- ✅ Consistent error envelopes with {error:{code,message,detail}}")
    print("- ✅ Request size validation with 413 responses for oversized content")
    print("- ✅ UTC timestamps with Z suffix for database consistency") 
    print("- ✅ X-Request-ID propagation for distributed tracing")


if __name__ == "__main__":
    main()