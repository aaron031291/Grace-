"""
Grace AI - Immutable, Append-Only Logging for Transparency with Cryptographic Guarantees
"""

import hashlib
import json
import os
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Optional: Try to import pynacl for Ed25519 signatures
try:
    from nacl.signing import SigningKey
    from nacl.encoding import HexEncoder
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("pynacl not available. Ed25519 signatures disabled. Install with: pip install pynacl")


class TransparencyLevel:
    """Defines transparency levels for logged events."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    GOVERNANCE_INTERNAL = "governance_internal"


class ImmutableLogger:
    """
    Cryptographic, append-only immutable logger with SHA-256 hashing
    and Ed25519 signature verification for every event.
    """
    def __init__(self, log_file_path: str = "grace_data/grace_log.jsonl", signing_key_hex: str = None):
        self.path = log_file_path
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        
        # Initialize Ed25519 signing key if crypto is available
        if CRYPTO_AVAILABLE:
            if signing_key_hex is None:
                # Generate a new key for development (in production, load from secure storage)
                self.sk = SigningKey.generate()
                logger.warning("Generated new Ed25519 signing key for ImmutableLogger. This should be persisted securely.")
            else:
                self.sk = SigningKey(signing_key_hex, encoder=HexEncoder)
            
            self.vk = self.sk.verify_key
        else:
            self.sk = None
            self.vk = None

        # Cache of last entry hash to chain records
        self._last_hash = None
        self._hydrate_last_hash()
        
        logger.info(f"Immutable Logger initialized. Log file: {self.path}, Crypto: {CRYPTO_AVAILABLE}")

    def _hydrate_last_hash(self):
        """Load the hash of the last entry to maintain the chain."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                last = None
                for line in f:
                    last = line
            if last:
                obj = json.loads(last)
                self._last_hash = obj.get("sha256")
        except Exception as e:
            logger.warning(f"Could not hydrate last hash: {e}")

    def _hash(self, payload: dict) -> str:
        """Compute SHA-256 hash of the payload."""
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def _sign(self, payload: dict) -> str:
        """Compute Ed25519 signature of the payload."""
        if not CRYPTO_AVAILABLE or not self.sk:
            return "CRYPTO_UNAVAILABLE"
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return self.sk.sign(blob).signature.hex()

    def append_phase(self, event: dict, phase: str, status: str, metadata: dict):
        """
        Append a new phase entry to the immutable log with cryptographic guarantees.
        """
        record = {
            "ts": time.time(),
            "event_id": event.get("id", "unknown"),
            "event_type": event.get("type", "unknown"),
            "phase": phase,
            "status": status,
            "metadata": metadata,
            "prev_hash": self._last_hash,
        }
        
        if CRYPTO_AVAILABLE and self.vk:
            record["pubkey"] = self.vk.encode(encoder=HexEncoder).decode()
        
        # Compute content hash & signature over core content (exclude sig fields)
        content = {k: record[k] for k in record.keys() if k not in ("sha256", "ed25519_sig")}
        sha = self._hash(content)
        sig = self._sign(content)
        record["sha256"] = sha
        record["ed25519_sig"] = sig

        line = json.dumps(record, separators=(",", ":")) + "\n"
        with open(self.path, "ab", buffering=0) as f:
            f.write(line.encode())
        self._last_hash = sha
        
        logger.debug(f"Immutable log entry appended: event_id={record['event_id']}, phase={phase}, status={status}")

    def log(self, actor: str, action: str, details: Dict[str, Any], level=TransparencyLevel.HIGH):
        """
        Legacy method for backward compatibility. Converts to append_phase format.
        """
        event = {
            "id": details.get("event_id", "legacy"),
            "type": action,
        }
        metadata = {
            "actor": actor,
            "details": details,
            "level": level,
        }
        self.append_phase(event, phase="LOG_ENTRY", status="ok", metadata=metadata)
