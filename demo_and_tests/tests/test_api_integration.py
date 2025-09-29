"""
Integration test for Grace API endpoints with new error handling and utilities.
"""
import unittest
import sys
import os
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from grace.core.utils import enum_from_str, create_error_response, validate_request_size, utc_timestamp
from grace.core.middleware import generate_request_id, RequestIDMiddleware
from grace.interface.orb_interface import PanelType, NotificationPriority


class TestGraceAPIIntegration(unittest.TestCase):
    """Integration tests for Grace API improvements."""
    
    def test_panel_type_enum_parsing(self):
        """Test PanelType enum parsing with various inputs."""
        # Test valid cases
        self.assertEqual(enum_from_str(PanelType, "chat"), PanelType.CHAT)
        self.assertEqual(enum_from_str(PanelType, "ANALYTICS"), PanelType.ANALYTICS)
        self.assertEqual(enum_from_str(PanelType, "dashboard"), PanelType.DASHBOARD)
        
        # Test with default
        result = enum_from_str(PanelType, "invalid_panel", default=PanelType.DASHBOARD)
        self.assertEqual(result, PanelType.DASHBOARD)
        
        # Test error case
        with self.assertRaises(ValueError) as context:
            enum_from_str(PanelType, "invalid_panel")
        self.assertIn("Invalid PanelType value", str(context.exception))
    
    def test_notification_priority_enum_parsing(self):
        """Test NotificationPriority enum parsing."""
        self.assertEqual(enum_from_str(NotificationPriority, "high"), NotificationPriority.HIGH)
        self.assertEqual(enum_from_str(NotificationPriority, "LOW"), NotificationPriority.LOW)
        
        # Test with default
        result = enum_from_str(NotificationPriority, "invalid", default=NotificationPriority.MEDIUM)
        self.assertEqual(result, NotificationPriority.MEDIUM)
    
    def test_error_response_structure(self):
        """Test consistent error response structure."""
        error = create_error_response("TEST_ERROR", "Test message", "Additional details")
        
        # Validate structure
        self.assertIn("error", error)
        self.assertIn("code", error["error"])
        self.assertIn("message", error["error"])
        self.assertIn("detail", error["error"])
        
        # Validate content
        self.assertEqual(error["error"]["code"], "TEST_ERROR")
        self.assertEqual(error["error"]["message"], "Test message")
        self.assertEqual(error["error"]["detail"], "Additional details")
    
    def test_request_size_validation(self):
        """Test request size validation for different content types."""
        # Valid small content
        small_text = "This is a small message"
        result = validate_request_size(small_text, max_size_mb=1)
        self.assertIsNone(result)
        
        # Oversized content
        large_text = "x" * (2 * 1024 * 1024)  # 2MB
        result = validate_request_size(large_text, max_size_mb=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["error"]["code"], "CONTENT_TOO_LARGE")
        
        # Valid bytes content
        small_bytes = b"small content"
        result = validate_request_size(small_bytes, max_size_mb=1)
        self.assertIsNone(result)
        
        # Oversized bytes
        large_bytes = b"x" * (2 * 1024 * 1024)  # 2MB
        result = validate_request_size(large_bytes, max_size_mb=1)
        self.assertIsNotNone(result)
    
    def test_request_id_generation(self):
        """Test request ID generation."""
        req_id1 = generate_request_id()
        req_id2 = generate_request_id()
        
        # Should be different
        self.assertNotEqual(req_id1, req_id2)
        
        # Should have expected format
        self.assertTrue(req_id1.startswith("req_"))
        self.assertTrue(req_id2.startswith("req_"))
        
        # Should be reasonable length
        self.assertEqual(len(req_id1), 16)  # "req_" + 12 hex chars
        self.assertEqual(len(req_id2), 16)
    
    def test_utc_timestamp_format(self):
        """Test UTC timestamp formatting."""
        timestamp = utc_timestamp()
        
        # Should end with Z
        self.assertTrue(timestamp.endswith('Z'))
        
        # Should match expected ISO format
        import re
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
        self.assertTrue(re.match(iso_pattern, timestamp))
    
    def test_websocket_message_validation(self):
        """Test WebSocket message size validation."""
        # Small message should pass
        small_msg = {"type": "chat_message", "content": "Hello"}
        content_size_check = validate_request_size(small_msg.get("content", ""), max_size_mb=1)
        self.assertIsNone(content_size_check)
        
        # Large message should fail
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        large_msg = {"type": "chat_message", "content": large_content}
        content_size_check = validate_request_size(large_msg.get("content", ""), max_size_mb=1)
        self.assertIsNotNone(content_size_check)
        self.assertEqual(content_size_check["error"]["code"], "CONTENT_TOO_LARGE")
    
    def test_api_error_consistency(self):
        """Test that API errors follow consistent format."""
        # Simulate different error types
        errors = [
            create_error_response("INVALID_PANEL_TYPE", "Invalid panel type", "panel_type: unknown"),
            create_error_response("SESSION_NOT_FOUND", "Session not found", "session_id: test123"),
            create_error_response("CONTENT_TOO_LARGE", "Content too large", "size: 5MB"),
            create_error_response("UNAUTHORIZED", "Authentication required"),
        ]
        
        for error in errors:
            # All should have error envelope
            self.assertIn("error", error)
            self.assertIn("code", error["error"])
            self.assertIn("message", error["error"])
            
            # Code should be uppercase with underscores
            code = error["error"]["code"]
            self.assertTrue(code.isupper())
            self.assertNotIn(" ", code)
            
            # Message should be human readable
            message = error["error"]["message"]
            self.assertIsInstance(message, str)
            self.assertGreater(len(message), 0)


class TestGraceAPIValidation(unittest.TestCase):
    """Test API validation scenarios."""
    
    def test_chat_message_size_limits(self):
        """Test chat message size validation."""
        # 10MB limit for chat messages
        valid_message = "Hello Grace, how are you today?"
        result = validate_request_size(valid_message, max_size_mb=10)
        self.assertIsNone(result)
        
        # Create a message just over 10MB
        large_message = "x" * (11 * 1024 * 1024)  # 11MB
        result = validate_request_size(large_message, max_size_mb=10)
        self.assertIsNotNone(result)
        self.assertEqual(result["error"]["code"], "CONTENT_TOO_LARGE")
        self.assertIn("10MB", result["error"]["message"])
    
    def test_file_upload_size_limits(self):
        """Test file upload size validation."""
        # 50MB limit for file uploads
        small_file_content = b"This is a small file" * 1000  # ~20KB
        result = validate_request_size(small_file_content, max_size_mb=50)
        self.assertIsNone(result)
        
        # Create content larger than 50MB
        large_file_content = b"x" * (51 * 1024 * 1024)  # 51MB
        result = validate_request_size(large_file_content, max_size_mb=50)
        self.assertIsNotNone(result)
        self.assertEqual(result["error"]["code"], "CONTENT_TOO_LARGE")
        self.assertIn("50MB", result["error"]["message"])
    
    def test_websocket_message_size_limits(self):
        """Test WebSocket message size validation."""
        # 1MB limit for WebSocket messages
        valid_ws_message = "Real-time message content"
        result = validate_request_size(valid_ws_message, max_size_mb=1)
        self.assertIsNone(result)
        
        # Create content larger than 1MB
        large_ws_message = "x" * (2 * 1024 * 1024)  # 2MB
        result = validate_request_size(large_ws_message, max_size_mb=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["error"]["code"], "CONTENT_TOO_LARGE")
        self.assertIn("1MB", result["error"]["message"])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)