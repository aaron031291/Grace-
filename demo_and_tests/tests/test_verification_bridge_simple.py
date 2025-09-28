"""Simple test runner for verification bridge implementation."""
import time
from datetime import datetime, timedelta
import sys
import os

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from grace.governance.verification_bridge import VerificationBridge
from grace.contracts.governed_request import GovernedRequest


def run_test(test_name, test_func):
    """Run a single test function."""
    try:
        test_func()
        print(f"✅ {test_name}")
        return True
    except Exception as e:
        print(f"❌ {test_name}: {e}")
        return False


def test_signature_verification():
    """Test signature verification functionality."""
    vb = VerificationBridge()
    
    # Test valid signature
    test_request = {'request_type': 'test', 'content': 'test content', 'requester': 'user'}
    content = vb._extract_request_content(test_request)
    signature = vb._generate_signature(content)
    
    assert vb.verify_signature(test_request, signature) is True
    assert vb.verify_signature(test_request, 'invalid_sig') is False
    assert vb.verify_signature(test_request, '') is False


def test_source_verification():
    """Test source verification functionality."""
    vb = VerificationBridge()
    
    # Trusted sources should pass
    assert vb.verify_source('admin_user', {}) is True
    assert vb.verify_source('system_process', {}) is True
    
    # Blocked sources should fail
    assert vb.verify_source('anonymous', {}) is False
    assert vb.verify_source('unknown', {}) is False
    
    # Regular users need session
    assert vb.verify_source('regular_user', {'session_valid': True}) is True
    assert vb.verify_source('regular_user', {}) is False


def test_context_validation():
    """Test context validation."""
    vb = VerificationBridge()
    
    # Rate limiting
    context = {'request_count': 150, 'session_valid': True}
    assert vb.verify_source('regular_user', context) is False
    
    # Failed attempts
    context = {'failed_attempts': 10, 'session_valid': True}
    assert vb.verify_source('regular_user', context) is False
    
    # Elevation requirements
    context = {'requires_elevation': True}
    assert vb.verify_source('regular_user', context) is False
    assert vb.verify_source('admin_user', context) is True


def test_format_validation():
    """Test format validation."""
    vb = VerificationBridge()
    
    # Valid formats
    assert vb._validate_requester_format('user123') is True
    assert vb._validate_requester_format('admin.user') is True
    
    # Invalid formats
    assert vb._validate_requester_format('') is False
    assert vb._validate_requester_format('x') is False
    assert vb._validate_requester_format('<script>') is False


def test_pydantic_integration():
    """Test integration with pydantic models."""
    vb = VerificationBridge()
    
    request = GovernedRequest(
        request_type='test_type',
        content='test content',
        requester='system_user'
    )
    
    # Should extract content properly
    content = vb._extract_request_content(request)
    assert 'test_type' in content
    assert 'test content' in content
    
    # Should verify signature
    signature = vb._generate_signature(content)
    assert vb.verify_signature(request, signature) is True


def test_integration():
    """Test integration with main verify method."""
    vb = VerificationBridge()
    
    request = GovernedRequest(
        request_type='test_action',
        content='integration test',
        requester='system_user'
    )
    
    result = vb.verify(request)
    
    assert isinstance(result, dict)
    assert 'verified' in result
    assert 'confidence' in result
    assert result['details']['source'] is True


def test_error_handling():
    """Test error handling in verification methods."""
    vb = VerificationBridge()
    
    # Should handle None gracefully
    assert vb.verify_signature(None, 'sig') is False
    assert vb.verify_source(None, {}) is False
    
    # Should handle malformed input
    assert vb.verify_signature({}, None) is False
    

def run_all_tests():
    """Run all verification bridge tests."""
    print("🧪 Testing Verification Bridge Production Implementation\n")
    
    tests = [
        ("Signature Verification", test_signature_verification),
        ("Source Verification", test_source_verification),
        ("Context Validation", test_context_validation),
        ("Format Validation", test_format_validation),
        ("Pydantic Integration", test_pydantic_integration),
        ("Integration Test", test_integration),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)