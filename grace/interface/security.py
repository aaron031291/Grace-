"""
Security utilities for the Grace interface including authentication and validation.
"""
import hashlib
import hmac
import secrets
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TokenManager:
    """Simple token management for WebSocket authentication."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        self.token_expiry_minutes = 60  # Tokens expire after 1 hour
    
    def generate_token(self, user_id: str, session_id: str, tenant_id: Optional[str] = None) -> str:
        """Generate a secure token for WebSocket authentication."""
        # Create token payload
        timestamp = datetime.utcnow().isoformat()
        payload = f"{user_id}:{session_id}:{tenant_id or 'default'}:{timestamp}"
        
        # Generate token using HMAC
        token = hmac.new(
            self.secret_key.encode(), 
            payload.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        # Store token metadata
        self.active_tokens[token] = {
            "user_id": user_id,
            "session_id": session_id,
            "tenant_id": tenant_id or "default",
            "created_at": timestamp,
            "expires_at": (datetime.utcnow() + timedelta(minutes=self.token_expiry_minutes)).isoformat()
        }
        
        logger.info(f"Generated token for user {user_id}, session {session_id}")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return user/session info if valid."""
        if not token or token not in self.active_tokens:
            return None
        
        token_data = self.active_tokens[token]
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        
        # Check if token has expired
        if datetime.utcnow() > expires_at:
            logger.warning(f"Token expired for user {token_data['user_id']}")
            del self.active_tokens[token]
            return None
        
        return token_data
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self.active_tokens:
            user_id = self.active_tokens[token]["user_id"]
            logger.info(f"Revoked token for user {user_id}")
            del self.active_tokens[token]
            return True
        return False
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens from memory."""
        current_time = datetime.utcnow()
        expired_tokens = []
        
        for token, data in self.active_tokens.items():
            expires_at = datetime.fromisoformat(data["expires_at"])
            if current_time > expires_at:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_tokens[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")


class FileValidator:
    """Validates uploaded files for security."""
    
    # MIME type mapping for common file extensions
    ALLOWED_MIME_TYPES = {
        'txt': 'text/plain',
        'md': 'text/markdown',
        'json': 'application/json',
        'csv': 'text/csv',
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'py': 'text/x-python',
        'js': 'text/javascript',
        'html': 'text/html',
        'css': 'text/css',
        'xml': 'text/xml',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'mp3': 'audio/mpeg',
        'mp4': 'video/mp4',
        'wav': 'audio/wav'
    }
    
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_file_size(self, file_size: int) -> bool:
        """Check if file size is within limits."""
        return file_size <= self.max_file_size_bytes
    
    def validate_file_type(self, filename: str, content_type: Optional[str] = None) -> bool:
        """Validate file type by extension and optionally MIME type."""
        if not filename or '.' not in filename:
            return False
        
        extension = filename.rsplit('.', 1)[-1].lower()
        if extension not in self.ALLOWED_MIME_TYPES:
            return False
        
        # If content type is provided, verify it matches expected MIME type
        if content_type:
            expected_mime = self.ALLOWED_MIME_TYPES[extension]
            # Allow some flexibility in MIME type checking
            if not (content_type.startswith(expected_mime.split('/')[0]) or 
                    content_type == expected_mime):
                logger.warning(f"MIME type mismatch for {filename}: expected {expected_mime}, got {content_type}")
                return False
        
        return True
    
    def get_safe_filename(self, filename: str) -> str:
        """Generate a safe filename by removing potentially dangerous characters."""
        import re
        # Remove any path components and dangerous characters
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename.split('/')[-1])
        # Ensure filename is not empty and has reasonable length
        if not safe_name or len(safe_name) > 255:
            safe_name = f"file_{secrets.token_hex(8)}"
        return safe_name
    
    def should_scan_for_viruses(self, filename: str) -> bool:
        """Determine if file should be scanned for viruses based on type."""
        if not filename or '.' not in filename:
            return True  # Scan unknown files
        
        extension = filename.rsplit('.', 1)[-1].lower()
        # Scan executable and document types that could contain malware
        risky_extensions = {'exe', 'bat', 'sh', 'scr', 'com', 'pif', 'doc', 'docx', 'pdf', 'zip', 'rar'}
        return extension in risky_extensions


async def virus_scan_hook(file_path: str, filename: str) -> bool:
    """
    Placeholder for virus scanning integration.
    In production, this would integrate with ClamAV, VirusTotal API, or similar.
    """
    logger.info(f"Virus scan requested for {filename} at {file_path}")
    
    # Placeholder implementation - always returns safe
    # In real implementation, this would:
    # 1. Call ClamAV daemon (clamd) if available
    # 2. Or use VirusTotal API for cloud scanning
    # 3. Or integrate with enterprise security tools
    
    return True  # File is safe