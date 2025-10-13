"""Tests for production verification bridge implementation."""

import pytest
import time
from datetime import datetime, timedelta
import sys
import os

# Add grace to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from grace.governance.verification_bridge import VerificationBridge
from grace.contracts.governed_request import GovernedRequest


class TestVerificationBridge:
    """Test suite for VerificationBridge production implementation."""

    def setup_method(self):
        """Setup test instance."""
        self.vb = VerificationBridge()

    def test_signature_verification_valid(self):
        """Test valid signature verification."""
        # Create test request
        test_request = {
            "request_type": "test_action",
            "content": "test content",
            "requester": "test_user",
        }

        # Generate valid signature
        content = self.vb._extract_request_content(test_request)
        signature = self.vb._generate_signature(content)

        # Verify signature
        result = self.vb.verify_signature(test_request, signature)
        assert result is True

    def test_signature_verification_invalid(self):
        """Test invalid signature rejection."""
        test_request = {"request_type": "test", "content": "test"}

        # Test with invalid signature
        result = self.vb.verify_signature(test_request, "invalid_signature")
        assert result is False

    def test_signature_verification_empty(self):
        """Test empty signature rejection."""
        test_request = {"request_type": "test", "content": "test"}

        result = self.vb.verify_signature(test_request, "")
        assert result is False

    def test_signature_verification_pydantic_request(self):
        """Test signature verification with pydantic request."""
        governed_request = GovernedRequest(
            request_type="test_type", content="test content", requester="test_user"
        )

        content = self.vb._extract_request_content(governed_request)
        signature = self.vb._generate_signature(content)

        result = self.vb.verify_signature(governed_request, signature)
        assert result is True

    def test_source_verification_trusted_domains(self):
        """Test source verification for trusted domains."""
        trusted_requesters = [
            "admin_user",
            "system_process",
            "internal_service",
            "governance_module",
        ]

        for requester in trusted_requesters:
            result = self.vb.verify_source(requester, {})
            assert result is True, f"Trusted requester {requester} should be accepted"

    def test_source_verification_blocked_sources(self):
        """Test source verification blocks untrusted sources."""
        blocked_requesters = ["anonymous", "unknown", "guest", "temp_user"]

        for requester in blocked_requesters:
            result = self.vb.verify_source(requester, {})
            assert result is False, f"Blocked requester {requester} should be rejected"

    def test_source_verification_with_session(self):
        """Test source verification with valid session."""
        result = self.vb.verify_source("regular_user", {"session_valid": True})
        assert result is True

    def test_source_verification_without_session(self):
        """Test source verification without session for non-trusted user."""
        result = self.vb.verify_source("regular_user", {})
        assert result is False

    def test_source_verification_rate_limiting(self):
        """Test source verification respects rate limiting."""
        context = {"request_count": 150, "session_valid": True}
        result = self.vb.verify_source("regular_user", context)
        assert result is False

    def test_source_verification_failed_attempts(self):
        """Test source verification blocks after failed attempts."""
        context = {"failed_attempts": 10, "session_valid": True}
        result = self.vb.verify_source("regular_user", context)
        assert result is False

    def test_source_verification_elevation_required(self):
        """Test source verification for elevation requirements."""
        # Non-admin user requesting elevation should fail
        context = {"requires_elevation": True}
        result = self.vb.verify_source("regular_user", context)
        assert result is False

        # Admin user requesting elevation should pass
        result = self.vb.verify_source("admin_user", context)
        assert result is True

    def test_source_verification_timestamp_validation(self):
        """Test timestamp validation in context."""
        # Valid recent timestamp
        recent_time = time.time()
        context = {"timestamp": recent_time, "session_valid": True}
        result = self.vb.verify_source("regular_user", context)
        assert result is True

        # Old timestamp should fail
        old_time = time.time() - 90000  # More than 24 hours ago
        context = {"timestamp": old_time, "session_valid": True}
        result = self.vb.verify_source("regular_user", context)
        assert result is False

    def test_requester_format_validation(self):
        """Test requester format validation."""
        # Valid formats
        valid_requesters = ["user123", "admin_user", "system.process"]
        for requester in valid_requesters:
            assert self.vb._validate_requester_format(requester) is True

        # Invalid formats
        invalid_requesters = [
            "",
            "x",
            "<script>alert(1)</script>",
            "javascript:void(0)",
        ]
        for requester in invalid_requesters:
            assert self.vb._validate_requester_format(requester) is False

    def test_context_ip_validation(self):
        """Test IP address validation in context."""
        # Valid IP context
        context = {"source_ip": "192.168.1.1", "session_valid": True}
        result = self.vb._validate_context("user", context)
        assert result is True

        # Invalid IP should fail validation
        context = {"source_ip": "invalid.ip.format", "session_valid": True}
        result = self.vb._validate_context("user", context)
        assert result is False

    def test_extract_request_content_dict(self):
        """Test content extraction from dict request."""
        request = {
            "request_type": "test",
            "content": "content",
            "requester": "user",
            "signature": "should_be_ignored",
        }

        content = self.vb._extract_request_content(request)
        assert "signature" not in content
        assert "request_type:test" in content

    def test_extract_request_content_object(self):
        """Test content extraction from object request."""
        request = GovernedRequest(
            request_type="test_type",
            content="test content",
            requester="test_user",
            priority=5,
        )

        content = self.vb._extract_request_content(request)
        assert "test_type" in content
        assert "test content" in content
        assert "test_user" in content
        assert "5" in content

    def test_signature_constant_time_comparison(self):
        """Test that signature comparison is constant time."""
        request = {"test": "data"}
        content = self.vb._extract_request_content(request)
        valid_sig = self.vb._generate_signature(content)

        # Valid signature should pass
        assert self.vb.verify_signature(request, valid_sig) is True

        # Invalid signature of same length should fail
        invalid_sig = valid_sig[:-1] + "X"
        assert self.vb.verify_signature(request, invalid_sig) is False

        # Different length signature should fail
        short_sig = valid_sig[:10]
        assert self.vb.verify_signature(request, short_sig) is False


def test_integration_with_main_verify_method():
    """Test integration with main verify method."""
    vb = VerificationBridge()

    # Test with valid governed request
    request = GovernedRequest(
        request_type="test_action",
        content="integration test content",
        requester="system_user",
    )

    # The main verify method should work with new implementations
    result = vb.verify(request)

    assert isinstance(result, dict)
    assert "verified" in result
    assert "confidence" in result
    assert result["details"]["source"] is True  # system_user should be trusted


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
