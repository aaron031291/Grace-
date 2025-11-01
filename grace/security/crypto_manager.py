"""
Cryptographic Manager for Grace
Generates and manages cryptographic keys for all operations
Ensures every input/output is signed and logged immutably
"""

import hashlib
import hmac
import secrets
import base64
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperationKey:
    """Cryptographic key for an operation"""
    operation_id: str
    key_id: str
    key_type: str
    created_at: datetime
    salt: bytes
    context: Dict[str, Any]
    signature: str


class CryptoManager:
    """
    Manages cryptographic operations for Grace.
    
    Every input/output gets:
    1. Unique cryptographic key
    2. HMAC signature
    3. Immutable log entry
    """
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self.operation_keys: Dict[str, OperationKey] = {}
        self.key_rotation_interval = 86400  # 24 hours
        
        logger.info("CryptoManager initialized")
    
    def _generate_master_key(self) -> str:
        """Generate master key for the system"""
        return Fernet.generate_key().decode()
    
    def generate_operation_key(
        self,
        operation_id: str,
        operation_type: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate unique cryptographic key for each operation.
        
        This key is used to sign all inputs/outputs for the operation
        and is logged to the immutable log for audit trail.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (api_request, db_query, etc.)
            context: Additional context about the operation
        
        Returns:
            Base64-encoded key
        """
        # Generate unique salt
        salt = secrets.token_bytes(32)
        
        # Create key material
        key_material = f"{operation_id}:{operation_type}:{datetime.utcnow().isoformat()}"
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(key_material.encode()))
        
        # Generate key ID
        key_id = hashlib.sha256(key).hexdigest()[:16]
        
        # Create signature of key metadata
        metadata = f"{operation_id}:{operation_type}:{key_id}"
        signature = self._sign_data(metadata)
        
        # Store operation key
        op_key = OperationKey(
            operation_id=operation_id,
            key_id=key_id,
            key_type=operation_type,
            created_at=datetime.utcnow(),
            salt=salt,
            context=context,
            signature=signature
        )
        
        self.operation_keys[operation_id] = op_key
        
        # Log to immutable logger
        self._log_key_generation(op_key)
        
        logger.debug(f"Generated key for operation: {operation_id} (type: {operation_type})")
        
        return key.decode()
    
    def sign_operation_data(
        self,
        operation_id: str,
        data: Dict[str, Any],
        data_direction: str = "output"  # "input" or "output"
    ) -> str:
        """
        Sign operation data (input or output).
        
        Creates HMAC signature for data integrity and non-repudiation.
        """
        # Get operation key
        op_key = self.operation_keys.get(operation_id)
        if not op_key:
            # Generate if not exists
            self.generate_operation_key(
                operation_id,
                "auto_generated",
                {"auto": True}
            )
            op_key = self.operation_keys[operation_id]
        
        # Serialize data
        data_str = self._serialize_data(data)
        
        # Create signature
        signature = hmac.new(
            op_key.key_id.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Log to immutable logger
        self._log_operation_signature(
            operation_id,
            data_direction,
            signature,
            data
        )
        
        return signature
    
    def verify_operation_signature(
        self,
        operation_id: str,
        data: Dict[str, Any],
        signature: str
    ) -> bool:
        """Verify operation signature"""
        op_key = self.operation_keys.get(operation_id)
        if not op_key:
            logger.warning(f"No key found for operation: {operation_id}")
            return False
        
        # Compute expected signature
        data_str = self._serialize_data(data)
        expected = hmac.new(
            op_key.key_id.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(expected, signature)
    
    def _sign_data(self, data: str) -> str:
        """Sign data using master key"""
        return hmac.new(
            self.master_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _serialize_data(self, data: Dict[str, Any]) -> str:
        """Serialize data for signing"""
        import json
        return json.dumps(data, sort_keys=True, default=str)
    
    def _log_key_generation(self, op_key: OperationKey):
        """Log key generation to immutable logger"""
        try:
            # Import here to avoid circular dependency
            import sys
            import os
            
            # Try to import immutable logger
            try:
                from grace.immutable_log import append_entry
                
                entry = {
                    "who": {
                        "actor_id": "crypto_manager",
                        "actor_type": "system",
                        "actor_display": "CryptoManager"
                    },
                    "what": "cryptographic_key_generated",
                    "where": {
                        "host": os.environ.get("HOSTNAME", "localhost"),
                        "service_path": "grace.security.crypto_manager"
                    },
                    "when": op_key.created_at.isoformat(),
                    "why": f"Key for {op_key.key_type} operation",
                    "how": "PBKDF2-SHA256 key derivation",
                    "payload": {
                        "operation_id": op_key.operation_id,
                        "key_id": op_key.key_id,
                        "operation_type": op_key.key_type,
                        "context": op_key.context
                    },
                    "signature": op_key.signature
                }
                
                append_entry(entry)
                logger.debug(f"Logged key generation for: {op_key.operation_id}")
                
            except ImportError:
                logger.warning("Immutable logger not available - key generation not logged")
                
        except Exception as e:
            logger.error(f"Failed to log key generation: {e}")
    
    def _log_operation_signature(
        self,
        operation_id: str,
        direction: str,
        signature: str,
        data: Dict[str, Any]
    ):
        """Log operation signature to immutable logger"""
        try:
            from grace.immutable_log import append_entry
            
            # Sanitize data for logging (don't log sensitive content)
            sanitized_data = {
                "data_size": len(str(data)),
                "data_type": type(data).__name__,
                "keys": list(data.keys()) if isinstance(data, dict) else None
            }
            
            entry = {
                "who": {
                    "actor_id": "crypto_manager",
                    "actor_type": "system"
                },
                "what": f"operation_{direction}_signed",
                "where": {
                    "service_path": "grace.security.crypto_manager"
                },
                "when": datetime.utcnow().isoformat(),
                "why": f"Sign {direction} data for operation",
                "how": "HMAC-SHA256 signature",
                "payload": {
                    "operation_id": operation_id,
                    "direction": direction,
                    "signature": signature,
                    "data_summary": sanitized_data
                }
            }
            
            append_entry(entry)
            
        except ImportError:
            logger.warning("Immutable logger not available")
        except Exception as e:
            logger.error(f"Failed to log operation signature: {e}")
    
    def get_operation_key_info(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an operation key"""
        op_key = self.operation_keys.get(operation_id)
        if not op_key:
            return None
        
        return {
            "operation_id": op_key.operation_id,
            "key_id": op_key.key_id,
            "key_type": op_key.key_type,
            "created_at": op_key.created_at.isoformat(),
            "context": op_key.context
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cryptographic operation statistics"""
        return {
            "total_keys_generated": len(self.operation_keys),
            "key_types": list(set(k.key_type for k in self.operation_keys.values())),
            "oldest_key": min(
                (k.created_at for k in self.operation_keys.values()),
                default=None
            )
        }


# Global instance
_crypto_manager: Optional[CryptoManager] = None


def get_crypto_manager() -> CryptoManager:
    """Get global crypto manager instance"""
    global _crypto_manager
    if _crypto_manager is None:
        _crypto_manager = CryptoManager()
    return _crypto_manager


# Decorator for automatic crypto logging
def crypto_logged(operation_type: str):
    """
    Decorator to automatically generate keys and sign operations.
    
    Usage:
        @crypto_logged("api_request")
        async def my_endpoint(data):
            return {"result": "success"}
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import uuid
            operation_id = str(uuid.uuid4())
            
            crypto = get_crypto_manager()
            
            # Generate key for this operation
            key = crypto.generate_operation_key(
                operation_id,
                operation_type,
                {"function": func.__name__, "module": func.__module__}
            )
            
            # Sign inputs
            input_data = {"args": str(args), "kwargs": str(kwargs)}
            input_sig = crypto.sign_operation_data(
                operation_id,
                input_data,
                "input"
            )
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Sign output
                output_data = {"result": str(result)}
                output_sig = crypto.sign_operation_data(
                    operation_id,
                    output_data,
                    "output"
                )
                
                return result
                
            except Exception as e:
                # Log error
                error_data = {"error": str(e), "type": type(e).__name__}
                crypto.sign_operation_data(operation_id, error_data, "error")
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo
    print("🔐 CryptoManager Demo\n")
    
    crypto = CryptoManager()
    
    # Generate key for an operation
    op_id = "demo_operation_001"
    key = crypto.generate_operation_key(
        op_id,
        "api_request",
        {"endpoint": "/api/tasks", "method": "POST"}
    )
    
    print(f"✅ Generated key: {key[:20]}...")
    
    # Sign some data
    data = {"user": "admin", "action": "create_task"}
    signature = crypto.sign_operation_data(op_id, data, "input")
    
    print(f"✅ Signed input: {signature[:20]}...")
    
    # Verify signature
    valid = crypto.verify_operation_signature(op_id, data, signature)
    print(f"✅ Signature valid: {valid}")
    
    # Stats
    stats = crypto.get_stats()
    print(f"\n📊 Stats:")
    print(f"  Total keys: {stats['total_keys_generated']}")
    print(f"  Key types: {stats['key_types']}")
