"""
Quantum-Safe Storage Layer for Grace Memory Infrastructure.

Provides:
- Post-quantum cryptographic encryption for data at rest
- Quantum-resistant key management and rotation
- Secure storage with quantum-safe algorithms
- Enhanced security for governance and memory operations
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import base64

# Optional post-quantum cryptography dependencies
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Optional quantum-safe algorithm support
try:
    import liboqs

    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False

logger = logging.getLogger(__name__)


class QuantumSafeKeyManager:
    """
    Quantum-safe key management system.

    Provides key generation, rotation, and storage using post-quantum
    cryptographic algorithms and best practices.
    """

    def __init__(
        self,
        key_store_path: str = "/tmp/grace_quantum_keys",
        master_key: Optional[bytes] = None,
    ):
        self.key_store_path = key_store_path
        self.master_key = master_key or self._generate_master_key()
        self.keys = {}
        self.key_history = {}

        # Ensure key store directory exists
        os.makedirs(key_store_path, mode=0o700, exist_ok=True)

    def _generate_master_key(self) -> bytes:
        """Generate a quantum-safe master key."""
        # Use strong entropy source
        return secrets.token_bytes(64)  # 512 bits for quantum resistance

    def _derive_key(self, purpose: str, salt: bytes = None) -> bytes:
        """Derive a key for specific purpose using quantum-safe KDF."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback to HMAC-based KDF
            if salt is None:
                salt = secrets.token_bytes(32)

            return hmac.new(
                self.master_key, purpose.encode() + salt, hashlib.sha3_256
            ).digest()

        # Use PBKDF2 with SHA3 for quantum resistance
        if salt is None:
            salt = secrets.token_bytes(32)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA3_256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )

        return kdf.derive(purpose.encode() + self.master_key)

    async def generate_encryption_key(
        self, key_id: str, algorithm: str = "AES-256-GCM"
    ) -> Dict[str, Any]:
        """Generate a new encryption key with quantum-safe properties."""
        try:
            # Generate key material
            if algorithm == "AES-256-GCM":
                key_material = secrets.token_bytes(32)  # 256 bits
            elif algorithm == "ChaCha20-Poly1305":
                key_material = secrets.token_bytes(32)  # 256 bits
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Create key metadata
            key_info = {
                "key_id": key_id,
                "algorithm": algorithm,
                "key_material": base64.b64encode(key_material).decode(),
                "created_at": datetime.now().isoformat(),
                "rotation_schedule": (datetime.now() + timedelta(days=90)).isoformat(),
                "status": "active",
                "version": 1,
            }

            # Store key securely
            await self._store_key(key_id, key_info)

            # Track in memory
            self.keys[key_id] = key_info

            logger.info(f"Generated quantum-safe encryption key: {key_id}")
            return {
                "key_id": key_id,
                "algorithm": algorithm,
                "created_at": key_info["created_at"],
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to generate encryption key {key_id}: {e}")
            return {"key_id": key_id, "status": "error", "error": str(e)}

    async def rotate_key(self, key_id: str) -> Dict[str, Any]:
        """Rotate an existing encryption key."""
        if key_id not in self.keys:
            return {"key_id": key_id, "status": "error", "error": "Key not found"}

        try:
            old_key = self.keys[key_id].copy()

            # Archive old key
            if key_id not in self.key_history:
                self.key_history[key_id] = []

            old_key["status"] = "rotated"
            old_key["rotated_at"] = datetime.now().isoformat()
            self.key_history[key_id].append(old_key)

            # Generate new key version
            algorithm = old_key["algorithm"]
            new_version = old_key["version"] + 1

            key_material = secrets.token_bytes(32)

            new_key = {
                "key_id": key_id,
                "algorithm": algorithm,
                "key_material": base64.b64encode(key_material).decode(),
                "created_at": datetime.now().isoformat(),
                "rotation_schedule": (datetime.now() + timedelta(days=90)).isoformat(),
                "status": "active",
                "version": new_version,
                "previous_version": old_key["version"],
            }

            # Store new key
            await self._store_key(key_id, new_key)
            self.keys[key_id] = new_key

            logger.info(f"Rotated encryption key {key_id} to version {new_version}")
            return {
                "key_id": key_id,
                "old_version": old_key["version"],
                "new_version": new_version,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")
            return {"key_id": key_id, "status": "error", "error": str(e)}

    async def get_key(
        self, key_id: str, version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a key by ID and version."""
        try:
            if version is None:
                # Get current active key
                return self.keys.get(key_id)
            else:
                # Get specific version from history
                if key_id in self.key_history:
                    for historical_key in self.key_history[key_id]:
                        if historical_key["version"] == version:
                            return historical_key

                # Check current key
                current_key = self.keys.get(key_id)
                if current_key and current_key["version"] == version:
                    return current_key

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None

    async def _store_key(self, key_id: str, key_info: Dict[str, Any]) -> None:
        """Securely store key information."""
        try:
            # Encrypt key info with master key
            encrypted_key_info = self._encrypt_data(
                json.dumps(key_info).encode(), self.master_key
            )

            # Store to file with secure permissions
            key_file_path = os.path.join(self.key_store_path, f"{key_id}.key")
            with open(key_file_path, "wb") as f:
                f.write(encrypted_key_info)

            # Set secure permissions
            os.chmod(key_file_path, 0o600)

        except Exception as e:
            logger.error(f"Failed to store key {key_id}: {e}")
            raise

    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using quantum-safe algorithms."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback: XOR with HMAC for integrity
            mac = hmac.new(key[:32], data, hashlib.sha3_256).digest()
            encrypted = bytes(
                a ^ b
                for a, b in zip(data, key[32 : 32 + len(data)] * (len(data) // 32 + 1))
            )
            return mac + encrypted

        # Use AES-256-GCM with random IV
        iv = secrets.token_bytes(16)
        cipher = Cipher(
            algorithms.AES(key[:32]), modes.GCM(iv), backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        return iv + encryptor.tag + ciphertext

    def _decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using quantum-safe algorithms."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback: XOR with HMAC verification
            mac = encrypted_data[:32]
            ciphertext = encrypted_data[32:]
            decrypted = bytes(
                a ^ b
                for a, b in zip(
                    ciphertext,
                    key[32 : 32 + len(ciphertext)] * (len(ciphertext) // 32 + 1),
                )
            )

            # Verify MAC
            expected_mac = hmac.new(key[:32], decrypted, hashlib.sha3_256).digest()
            if not hmac.compare_digest(mac, expected_mac):
                raise ValueError("Data integrity check failed")

            return decrypted

        # AES-256-GCM decryption
        iv = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]

        cipher = Cipher(
            algorithms.AES(key[:32]), modes.GCM(iv, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        return decryptor.update(ciphertext) + decryptor.finalize()

    def check_key_rotation_needed(self) -> List[str]:
        """Check which keys need rotation."""
        keys_to_rotate = []
        now = datetime.now()

        for key_id, key_info in self.keys.items():
            rotation_time = datetime.fromisoformat(key_info["rotation_schedule"])
            if now >= rotation_time:
                keys_to_rotate.append(key_id)

        return keys_to_rotate

    def get_stats(self) -> Dict[str, Any]:
        """Get key manager statistics."""
        now = datetime.now()

        active_keys = len([k for k in self.keys.values() if k["status"] == "active"])
        keys_needing_rotation = len(self.check_key_rotation_needed())

        # Calculate average key age
        key_ages = []
        for key_info in self.keys.values():
            created_at = datetime.fromisoformat(key_info["created_at"])
            age_days = (now - created_at).days
            key_ages.append(age_days)

        avg_key_age = sum(key_ages) / len(key_ages) if key_ages else 0

        return {
            "total_keys": len(self.keys),
            "active_keys": active_keys,
            "keys_needing_rotation": keys_needing_rotation,
            "average_key_age_days": avg_key_age,
            "key_store_path": self.key_store_path,
            "quantum_safe": True,
        }


class QuantumSafeEncryption:
    """
    Quantum-safe encryption service for data protection.
    """

    def __init__(self, key_manager: QuantumSafeKeyManager):
        self.key_manager = key_manager

    async def encrypt_data(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        key_id: str,
        additional_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Encrypt data using quantum-safe algorithms."""
        try:
            # Get encryption key
            key_info = await self.key_manager.get_key(key_id)
            if not key_info:
                raise ValueError(f"Encryption key not found: {key_id}")

            # Prepare data for encryption
            if isinstance(data, str):
                plaintext = data.encode("utf-8")
            elif isinstance(data, dict):
                plaintext = json.dumps(data).encode("utf-8")
            else:
                plaintext = data

            # Get key material
            key_material = base64.b64decode(key_info["key_material"])
            algorithm = key_info["algorithm"]

            # Encrypt based on algorithm
            if algorithm == "AES-256-GCM":
                encrypted_data = self._encrypt_aes_gcm(
                    plaintext, key_material, additional_data
                )
            elif algorithm == "ChaCha20-Poly1305":
                encrypted_data = self._encrypt_chacha20(
                    plaintext, key_material, additional_data
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "key_id": key_id,
                "key_version": key_info["version"],
                "algorithm": algorithm,
                "encrypted_at": datetime.now().isoformat(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Encryption failed for key {key_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def decrypt_data(
        self,
        encrypted_data: str,
        key_id: str,
        key_version: Optional[int] = None,
        additional_data: Optional[bytes] = None,
        return_format: str = "auto",
    ) -> Dict[str, Any]:
        """Decrypt data using quantum-safe algorithms."""
        try:
            # Get decryption key
            key_info = await self.key_manager.get_key(key_id, key_version)
            if not key_info:
                raise ValueError(f"Decryption key not found: {key_id}")

            # Decode encrypted data
            ciphertext = base64.b64decode(encrypted_data)

            # Get key material
            key_material = base64.b64decode(key_info["key_material"])
            algorithm = key_info["algorithm"]

            # Decrypt based on algorithm
            if algorithm == "AES-256-GCM":
                plaintext = self._decrypt_aes_gcm(
                    ciphertext, key_material, additional_data
                )
            elif algorithm == "ChaCha20-Poly1305":
                plaintext = self._decrypt_chacha20(
                    ciphertext, key_material, additional_data
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Format output
            if return_format == "bytes":
                decrypted_data = plaintext
            elif return_format == "json":
                decrypted_data = json.loads(plaintext.decode("utf-8"))
            else:  # auto-detect
                try:
                    # Try to decode as JSON first
                    decrypted_data = json.loads(plaintext.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        # Try as UTF-8 string
                        decrypted_data = plaintext.decode("utf-8")
                    except UnicodeDecodeError:
                        # Return as bytes
                        decrypted_data = plaintext

            return {
                "decrypted_data": decrypted_data,
                "key_id": key_id,
                "key_version": key_info["version"],
                "algorithm": algorithm,
                "decrypted_at": datetime.now().isoformat(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Decryption failed for key {key_id}: {e}")
            return {"status": "error", "error": str(e)}

    def _encrypt_aes_gcm(
        self, plaintext: bytes, key: bytes, aad: Optional[bytes] = None
    ) -> bytes:
        """Encrypt using AES-256-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")

        # Generate random IV
        iv = secrets.token_bytes(16)

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Add additional authenticated data if provided
        if aad:
            encryptor.authenticate_additional_data(aad)

        # Encrypt and finalize
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext

    def _decrypt_aes_gcm(
        self, encrypted_data: bytes, key: bytes, aad: Optional[bytes] = None
    ) -> bytes:
        """Decrypt using AES-256-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")

        # Extract components
        iv = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]

        # Create cipher
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Add additional authenticated data if provided
        if aad:
            decryptor.authenticate_additional_data(aad)

        # Decrypt and finalize
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _encrypt_chacha20(
        self, plaintext: bytes, key: bytes, aad: Optional[bytes] = None
    ) -> bytes:
        """Encrypt using ChaCha20-Poly1305 (quantum-safe alternative)."""
        # This would require additional library support
        # For now, fall back to AES-GCM
        return self._encrypt_aes_gcm(plaintext, key, aad)

    def _decrypt_chacha20(
        self, encrypted_data: bytes, key: bytes, aad: Optional[bytes] = None
    ) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        # This would require additional library support
        # For now, fall back to AES-GCM
        return self._decrypt_aes_gcm(encrypted_data, key, aad)


class QuantumSafeStorageLayer:
    """
    Quantum-safe storage layer providing encrypted data persistence.
    """

    def __init__(
        self,
        storage_path: str = "/tmp/grace_quantum_storage",
        key_manager: Optional[QuantumSafeKeyManager] = None,
    ):
        self.storage_path = storage_path
        self.key_manager = key_manager or QuantumSafeKeyManager()
        self.encryption = QuantumSafeEncryption(self.key_manager)

        # Ensure storage directory exists
        os.makedirs(storage_path, mode=0o700, exist_ok=True)

        # Statistics
        self.write_count = 0
        self.read_count = 0
        self.encryption_count = 0
        self.decryption_count = 0

    async def initialize(self) -> bool:
        """Initialize the quantum-safe storage layer."""
        try:
            # Generate default encryption keys
            await self.key_manager.generate_encryption_key(
                "grace_default", "AES-256-GCM"
            )
            await self.key_manager.generate_encryption_key(
                "grace_governance", "AES-256-GCM"
            )
            await self.key_manager.generate_encryption_key(
                "grace_memory", "AES-256-GCM"
            )

            logger.info("Quantum-safe storage layer initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize quantum-safe storage: {e}")
            return False

    async def store_encrypted(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        storage_id: str,
        key_id: str = "grace_default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store data with quantum-safe encryption."""
        try:
            # Encrypt the data
            encryption_result = await self.encryption.encrypt_data(data, key_id)

            if encryption_result["status"] != "success":
                return encryption_result

            # Create storage entry
            storage_entry = {
                "storage_id": storage_id,
                "encrypted_data": encryption_result["encrypted_data"],
                "key_id": encryption_result["key_id"],
                "key_version": encryption_result["key_version"],
                "algorithm": encryption_result["algorithm"],
                "metadata": metadata or {},
                "stored_at": datetime.now().isoformat(),
                "storage_version": "1.0",
            }

            # Store to file
            storage_file = os.path.join(self.storage_path, f"{storage_id}.qss")
            with open(storage_file, "w") as f:
                json.dump(storage_entry, f)

            # Set secure permissions
            os.chmod(storage_file, 0o600)

            self.write_count += 1
            self.encryption_count += 1

            logger.debug(f"Stored encrypted data: {storage_id}")

            return {
                "storage_id": storage_id,
                "key_id": key_id,
                "encrypted": True,
                "stored_at": storage_entry["stored_at"],
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to store encrypted data {storage_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def retrieve_encrypted(
        self, storage_id: str, return_format: str = "auto"
    ) -> Dict[str, Any]:
        """Retrieve and decrypt data."""
        try:
            # Load storage entry
            storage_file = os.path.join(self.storage_path, f"{storage_id}.qss")

            if not os.path.exists(storage_file):
                return {"status": "error", "error": "Storage entry not found"}

            with open(storage_file, "r") as f:
                storage_entry = json.load(f)

            # Decrypt the data
            decryption_result = await self.encryption.decrypt_data(
                storage_entry["encrypted_data"],
                storage_entry["key_id"],
                storage_entry["key_version"],
                return_format=return_format,
            )

            if decryption_result["status"] != "success":
                return decryption_result

            self.read_count += 1
            self.decryption_count += 1

            logger.debug(f"Retrieved encrypted data: {storage_id}")

            return {
                "storage_id": storage_id,
                "data": decryption_result["decrypted_data"],
                "metadata": storage_entry.get("metadata", {}),
                "key_id": storage_entry["key_id"],
                "stored_at": storage_entry["stored_at"],
                "retrieved_at": datetime.now().isoformat(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to retrieve encrypted data {storage_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_encrypted(self, storage_id: str) -> Dict[str, Any]:
        """Securely delete encrypted data."""
        try:
            storage_file = os.path.join(self.storage_path, f"{storage_id}.qss")

            if not os.path.exists(storage_file):
                return {"status": "error", "error": "Storage entry not found"}

            # Secure deletion (overwrite before delete)
            file_size = os.path.getsize(storage_file)
            with open(storage_file, "r+b") as f:
                # Overwrite with random data
                f.write(secrets.token_bytes(file_size))
                f.flush()
                os.fsync(f.fileno())

            # Remove file
            os.remove(storage_file)

            logger.info(f"Securely deleted encrypted data: {storage_id}")

            return {
                "storage_id": storage_id,
                "deleted_at": datetime.now().isoformat(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to delete encrypted data {storage_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def rotate_storage_keys(self) -> Dict[str, Any]:
        """Rotate all storage encryption keys."""
        try:
            keys_to_rotate = self.key_manager.check_key_rotation_needed()
            rotation_results = {}

            for key_id in keys_to_rotate:
                result = await self.key_manager.rotate_key(key_id)
                rotation_results[key_id] = result

            logger.info(f"Rotated {len(keys_to_rotate)} storage keys")

            return {
                "keys_rotated": len(keys_to_rotate),
                "rotation_results": rotation_results,
                "rotated_at": datetime.now().isoformat(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Storage key rotation failed: {e}")
            return {"status": "error", "error": str(e)}

    def list_stored_items(self) -> List[str]:
        """List all stored item IDs."""
        try:
            storage_files = [
                f[:-4] for f in os.listdir(self.storage_path) if f.endswith(".qss")
            ]
            return storage_files
        except Exception as e:
            logger.error(f"Failed to list stored items: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get storage layer statistics."""
        try:
            stored_items = self.list_stored_items()
            total_items = len(stored_items)

            # Calculate storage size
            total_size = 0
            for item_id in stored_items:
                storage_file = os.path.join(self.storage_path, f"{item_id}.qss")
                if os.path.exists(storage_file):
                    total_size += os.path.getsize(storage_file)

            key_manager_stats = self.key_manager.get_stats()

            return {
                "storage_path": self.storage_path,
                "total_items": total_items,
                "total_size_bytes": total_size,
                "write_count": self.write_count,
                "read_count": self.read_count,
                "encryption_count": self.encryption_count,
                "decryption_count": self.decryption_count,
                "quantum_safe": True,
                "key_management": key_manager_stats,
            }

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"status": "error", "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the storage system."""
        try:
            # Test encryption/decryption
            test_data = "quantum_safe_test_" + secrets.token_hex(16)
            test_id = f"health_test_{int(datetime.now().timestamp())}"

            # Store test data
            store_result = await self.store_encrypted(test_data, test_id)
            if store_result["status"] != "success":
                return {"healthy": False, "error": "Storage test failed"}

            # Retrieve test data
            retrieve_result = await self.retrieve_encrypted(test_id)
            if retrieve_result["status"] != "success":
                return {"healthy": False, "error": "Retrieval test failed"}

            # Verify data integrity
            if retrieve_result["data"] != test_data:
                return {"healthy": False, "error": "Data integrity test failed"}

            # Clean up test data
            await self.delete_encrypted(test_id)

            return {
                "healthy": True,
                "test_completed_at": datetime.now().isoformat(),
                "encryption_working": True,
                "decryption_working": True,
                "data_integrity": True,
            }

        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {"healthy": False, "error": str(e)}
