"""Verification bridge - verifies request authenticity and integrity."""
from typing import Dict, Any
import logging
import hashlib
import hmac
import base64
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VerificationBridge:
    """Handles request verification and validation."""
    
    def __init__(self):
        self.verification_methods = ["signature", "integrity", "source"]
        # In production, this should be loaded from secure configuration
        self._signature_secret = "grace_verification_secret_key_change_in_production"
        # Trusted source patterns and validation rules
        self._trusted_domains = {
            "internal", "admin", "system", "governance", "verified_user"
        }
        self._blocked_sources = {
            "anonymous", "unknown", "guest", "temp", "test"
        }
    
    def verify(self, request) -> Dict[str, Any]:
        """Verify request authenticity and integrity.
        
        Args:
            request: Either GovernedRequest (pydantic) or dict/other request format
            
        Returns:
            Dict with verification results
        """
        checks = []
        
        # Check 1: Basic request structure  
        structure_valid = self._check_structure(request)
        checks.append(("structure", structure_valid))
        
        # Check 2: Content integrity (simplified)
        content_integrity = self._check_content_integrity(request)
        checks.append(("content_integrity", content_integrity))
        
        # Check 3: Source verification (simplified)
        source_valid = self._check_source(request)
        checks.append(("source", source_valid))
        
        # Aggregate results
        passed_checks = sum(1 for _, passed in checks if passed)
        total_checks = len(checks)
        confidence = passed_checks / total_checks if total_checks > 0 else 0.0
        
        all_passed = passed_checks == total_checks
        
        details = {
            check_name: passed for check_name, passed in checks
        }
        details["passed_checks"] = passed_checks
        details["total_checks"] = total_checks
        
        return {
            "verified": all_passed,
            "confidence": confidence,
            "verification_method": "multi_check",
            "details": details
        }
    
    def _check_structure(self, request) -> bool:
        """Check basic request structure."""
        if hasattr(request, 'request_type') and hasattr(request, 'content') and hasattr(request, 'requester'):
            return bool(request.request_type and request.content and request.requester)
        elif isinstance(request, dict):
            return bool(
                request.get('request_type') and 
                request.get('content') and 
                request.get('requester')
            )
        else:
            # For other request formats, just check it's not empty
            return request is not None
    
    def _check_content_integrity(self, request) -> bool:
        """Check content integrity."""
        content = ""
        if hasattr(request, 'content'):
            content = str(request.content)
        elif isinstance(request, dict):
            content = str(request.get('content', ''))
        else:
            content = str(request) if request else ""
            
        return len(content.strip()) > 0
    
    def _check_source(self, request) -> bool:
        """Check source validity."""
        requester = ""
        if hasattr(request, 'requester'):
            requester = str(request.requester)
        elif isinstance(request, dict):
            requester = str(request.get('requester', ''))
            
        return requester != "unknown" and bool(requester)
    
    def verify_signature(self, request, signature: str) -> bool:
        """Verify cryptographic signature for request authenticity.
        
        Args:
            request: The request object to verify
            signature: Base64-encoded HMAC-SHA256 signature to verify
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        if not signature:
            logger.warning("Empty signature provided for verification")
            return False
            
        try:
            # Extract request content for signature verification
            request_content = self._extract_request_content(request)
            if not request_content:
                logger.warning("Unable to extract content from request for signature verification")
                return False
            
            # Generate expected signature
            expected_signature = self._generate_signature(request_content)
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {str(e)}")
            return False
    
    def verify_source(self, requester: str, context: Dict[str, Any]) -> bool:
        """Verify the request source authenticity and authorization.
        
        Args:
            requester: The identity of the requester
            context: Additional context information for verification
            
        Returns:
            bool: True if source is valid and authorized, False otherwise
        """
        if not requester:
            logger.warning("Empty requester provided for source verification")
            return False
            
        try:
            # Check if requester is blocked
            if requester.lower() in self._blocked_sources:
                logger.warning(f"Blocked source attempted access: {requester}")
                return False
            
            # Validate requester format and content
            if not self._validate_requester_format(requester):
                logger.warning(f"Invalid requester format: {requester}")
                return False
            
            # Check context for additional validation
            if context and not self._validate_context(requester, context):
                logger.warning(f"Context validation failed for requester: {requester}")
                return False
            
            # Check if requester is from trusted domain
            is_trusted = any(trusted in requester.lower() for trusted in self._trusted_domains)
            
            # For non-trusted sources, perform additional checks
            if not is_trusted:
                if not self._perform_additional_source_checks(requester, context):
                    logger.info(f"Additional source checks failed for: {requester}")
                    return False
            
            logger.debug(f"Source verification successful for: {requester}")
            return True
            
        except Exception as e:
            logger.error(f"Source verification failed for {requester}: {str(e)}")
            return False
    
    def get_verification_strength(self, result: Dict[str, Any]) -> str:
        """Get human-readable verification strength."""
        confidence = result.get("confidence", 0.0)
        if confidence >= 0.9:
            return "strong"
        elif confidence >= 0.7:
            return "moderate"
        elif confidence >= 0.5:
            return "weak"
        else:
            return "failed"
    
    def _extract_request_content(self, request) -> str:
        """Extract content from request for signature verification."""
        content_parts = []
        
        if hasattr(request, 'request_type'):
            content_parts.append(str(request.request_type))
        if hasattr(request, 'content'):
            content_parts.append(str(request.content))
        if hasattr(request, 'requester'):
            content_parts.append(str(request.requester))
        if hasattr(request, 'priority'):
            content_parts.append(str(request.priority))
        
        elif isinstance(request, dict):
            # For dict-based requests
            for key in sorted(request.keys()):
                if key not in ['signature', 'timestamp']:  # Exclude signature fields
                    content_parts.append(f"{key}:{request[key]}")
        else:
            # For other request types
            content_parts.append(str(request))
        
        return "|".join(content_parts)
    
    def _generate_signature(self, content: str) -> str:
        """Generate HMAC-SHA256 signature for content."""
        signature = hmac.new(
            self._signature_secret.encode('utf-8'),
            content.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    def _validate_requester_format(self, requester: str) -> bool:
        """Validate requester string format and content."""
        # Basic format validation
        if len(requester.strip()) < 2:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = ['<script', 'javascript:', 'data:', '..', '//', '\\\\']
        requester_lower = requester.lower()
        
        if any(pattern in requester_lower for pattern in suspicious_patterns):
            return False
        
        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in requester):
            return False
        
        return True
    
    def _validate_context(self, requester: str, context: Dict[str, Any]) -> bool:
        """Validate context information for the requester."""
        # Check for required context fields for certain operations
        if context.get('requires_elevation') and 'admin' not in requester.lower():
            return False
        
        # Validate timestamp if provided
        if 'timestamp' in context:
            try:
                timestamp = context['timestamp']
                if isinstance(timestamp, (int, float)):
                    # Check if timestamp is within reasonable bounds (last 24 hours)
                    now = time.time()
                    if abs(now - timestamp) > 86400:  # 24 hours in seconds
                        return False
                elif isinstance(timestamp, str):
                    # Try to parse ISO format timestamp
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if abs((datetime.now() - dt).total_seconds()) > 86400:
                        return False
            except (ValueError, TypeError):
                return False
        
        # Validate IP address if provided
        if 'source_ip' in context:
            ip = context['source_ip']
            # Basic IP format validation (simplified)
            if not isinstance(ip, str) or not ip.replace('.', '').replace(':', '').isdigit():
                return False
        
        return True
    
    def _perform_additional_source_checks(self, requester: str, context: Dict[str, Any]) -> bool:
        """Perform additional checks for non-trusted sources."""
        context = context or {}
        
        # Check for rate limiting context
        if context.get('request_count', 0) > 100:  # Basic rate limiting
            return False
        
        # Check for suspicious activity patterns
        if context.get('failed_attempts', 0) > 5:
            return False
        
        # Check if requester has valid session/token context
        if not context.get('session_valid') and not context.get('token_valid'):
            # For non-trusted sources, require some form of valid session
            if 'system' not in requester.lower() and 'internal' not in requester.lower():
                return False
        
        return True