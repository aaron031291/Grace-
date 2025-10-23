"""
Encryption Manager - Data encryption at rest and in transit
"""

from typing import Dict, Any, Optional
import base64
import hashlib
import secrets
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Manages encryption for sensitive data
    
    Features:
    - Field-level encryption
    - Key derivation
    - Secure token generation
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption manager
        
        Args:
            master_key: Master encryption key (32 bytes)
                       If None, generates a new key
        """
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt string data
        
        Args:
            data: Plain text to encrypt
        
        Returns:
            Base64-encoded encrypted data
        """
        if not data:
            return ""
        
        try:
            encrypted = self.fernet.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt string data
        
        Args:
            encrypted_data: Base64-encoded encrypted data
        
        Returns:
            Decrypted plain text
        """
        if not encrypted_data:
            return ""
        
        try:
            decoded = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(
        self,
        data: Dict[str, Any],
        fields_to_encrypt: list
    ) -> Dict[str, Any]:
        """
        Encrypt specific fields in dictionary
        
        Args:
            data: Dictionary containing data
            fields_to_encrypt: List of field names to encrypt
        
        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()
        
        for field in fields_to_encrypt:
            if field in encrypted_data:
                value = str(encrypted_data[field])
                encrypted_data[field] = self.encrypt(value)
                encrypted_data[f"{field}_encrypted"] = True
        
        return encrypted_data
    
    def decrypt_dict(
        self,
        data: Dict[str, Any],
        fields_to_decrypt: list
    ) -> Dict[str, Any]:
        """
        Decrypt specific fields in dictionary
        
        Args:
            data: Dictionary containing encrypted data
            fields_to_decrypt: List of field names to decrypt
        
        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()
        
        for field in fields_to_decrypt:
            if field in decrypted_data and data.get(f"{field}_encrypted"):
                encrypted_value = decrypted_data[field]
                decrypted_data[field] = self.decrypt(encrypted_value)
                decrypted_data.pop(f"{field}_encrypted", None)
        
        return decrypted_data
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """
        Hash password with salt
        
        Args:
            password: Plain text password
            salt: Optional salt (generates if not provided)
        
        Returns:
            (hashed_password, salt) tuple as base64 strings
        """
        if not salt:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        return (
            base64.b64encode(key).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8')
        )
    
    @staticmethod
    def verify_password(
        password: str,
        hashed_password: str,
        salt: str
    ) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password to verify
            hashed_password: Base64-encoded hash
            salt: Base64-encoded salt
        
        Returns:
            True if password matches
        """
        try:
            computed_hash, _ = EncryptionManager.hash_password(
                password,
                base64.b64decode(salt.encode('utf-8'))
            )
            return computed_hash == hashed_password
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate secure random token
        
        Args:
            length: Token length in bytes
        
        Returns:
            URL-safe base64-encoded token
        """
        return secrets.token_urlsafe(length)
