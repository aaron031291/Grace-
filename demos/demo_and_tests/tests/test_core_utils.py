"""
Tests for Grace core utilities - enum parsing, timestamp handling, error responses.
"""

import unittest
import sys
import os
from datetime import datetime, timezone
from enum import Enum

# Add grace to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from grace.core.utils import (
    enum_from_str,
    utc_timestamp,
    normalize_timestamp,
    create_error_response,
    validate_request_size,
)


class TestEnum(Enum):
    """Test enum for validation."""

    CHAT = "chat"
    DASHBOARD = "dashboard"
    ANALYTICS = "analytics"


class TestGraceCoreUtils(unittest.TestCase):
    """Test cases for Grace core utilities."""

    def test_enum_from_str_exact_value(self):
        """Test enum parsing with exact value match."""
        result = enum_from_str(TestEnum, "chat")
        self.assertEqual(result, TestEnum.CHAT)

    def test_enum_from_str_uppercase_name(self):
        """Test enum parsing with uppercase name."""
        result = enum_from_str(TestEnum, "CHAT")
        self.assertEqual(result, TestEnum.CHAT)

    def test_enum_from_str_lowercase_value(self):
        """Test enum parsing with lowercase value."""
        result = enum_from_str(TestEnum, "dashboard")
        self.assertEqual(result, TestEnum.DASHBOARD)

    def test_enum_from_str_case_insensitive(self):
        """Test case insensitive enum parsing."""
        result = enum_from_str(TestEnum, "AnAlYtIcS")
        self.assertEqual(result, TestEnum.ANALYTICS)

    def test_enum_from_str_with_default(self):
        """Test enum parsing with default value."""
        result = enum_from_str(TestEnum, "invalid", default=TestEnum.DASHBOARD)
        self.assertEqual(result, TestEnum.DASHBOARD)

    def test_enum_from_str_invalid_no_default(self):
        """Test enum parsing failure without default."""
        with self.assertRaises(ValueError) as context:
            enum_from_str(TestEnum, "invalid")
        self.assertIn("Invalid TestEnum value", str(context.exception))

    def test_enum_from_str_non_string_input(self):
        """Test enum parsing with non-string input."""
        result = enum_from_str(TestEnum, 123, default=TestEnum.CHAT)
        self.assertEqual(result, TestEnum.CHAT)

        with self.assertRaises(ValueError):
            enum_from_str(TestEnum, 123)

    def test_utc_timestamp_format(self):
        """Test UTC timestamp generation."""
        timestamp = utc_timestamp()
        self.assertTrue(timestamp.endswith("Z"))
        self.assertRegex(timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")

    def test_normalize_timestamp_string_with_z(self):
        """Test timestamp normalization with Z suffix."""
        input_ts = "2023-09-28T12:00:00.000000Z"
        result = normalize_timestamp(input_ts)
        self.assertTrue(result.endswith("Z"))

    def test_normalize_timestamp_datetime_naive(self):
        """Test timestamp normalization with naive datetime."""
        dt = datetime(2023, 9, 28, 12, 0, 0)
        result = normalize_timestamp(dt)
        self.assertTrue(result.endswith("Z"))
        self.assertIn("2023-09-28T12:00:00", result)

    def test_normalize_timestamp_datetime_with_tz(self):
        """Test timestamp normalization with timezone-aware datetime."""
        dt = datetime(2023, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
        result = normalize_timestamp(dt)
        self.assertTrue(result.endswith("Z"))

    def test_normalize_timestamp_none(self):
        """Test timestamp normalization with None."""
        result = normalize_timestamp(None)
        self.assertIsNone(result)

    def test_create_error_response_basic(self):
        """Test basic error response creation."""
        response = create_error_response("TEST_ERROR", "Test message")
        self.assertEqual(response["error"]["code"], "TEST_ERROR")
        self.assertEqual(response["error"]["message"], "Test message")
        self.assertNotIn("detail", response["error"])

    def test_create_error_response_with_detail(self):
        """Test error response creation with detail."""
        response = create_error_response("TEST_ERROR", "Test message", "More details")
        self.assertEqual(response["error"]["code"], "TEST_ERROR")
        self.assertEqual(response["error"]["message"], "Test message")
        self.assertEqual(response["error"]["detail"], "More details")

    def test_validate_request_size_valid_string(self):
        """Test request size validation with valid string."""
        content = "This is a test message"
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNone(result)

    def test_validate_request_size_too_large_string(self):
        """Test request size validation with oversized string."""
        # Create a string larger than 1MB
        content = "x" * (1024 * 1024 + 1)  # Just over 1MB
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["error"]["code"], "CONTENT_TOO_LARGE")

    def test_validate_request_size_valid_bytes(self):
        """Test request size validation with valid bytes."""
        content = b"This is a test message"
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNone(result)

    def test_validate_request_size_too_large_bytes(self):
        """Test request size validation with oversized bytes."""
        content = b"x" * (1024 * 1024 + 1)  # Just over 1MB
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["error"]["code"], "CONTENT_TOO_LARGE")

    def test_validate_request_size_dict_with_size(self):
        """Test request size validation with dict containing size."""
        content = {"size": 1024}  # 1KB
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNone(result)

        content = {"size": 2 * 1024 * 1024}  # 2MB
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNotNone(result)

    def test_validate_request_size_unknown_type(self):
        """Test request size validation with unknown type."""
        content = {"data": "test"}  # Dict without size
        result = validate_request_size(content, max_size_mb=1)
        self.assertIsNone(result)  # Should allow unknown types


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
