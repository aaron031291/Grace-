#!/usr/bin/env python3
"""
Grace AI - Immutable Log Verification Utility

Verifies cryptographic integrity of the immutable audit log:
- Recomputes SHA-256 hashes
- Verifies Ed25519 signatures
- Checks hash chain integrity
- Displays event flow for a given event_id

Usage:
    python verify_immutable_log.py <event_id>
    python verify_immutable_log.py --all
    python verify_immutable_log.py --last 10
"""
import sys
import json
import hashlib
import argparse
from pathlib import Path

# Add grace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace import config

try:
    from nacl.signing import VerifyKey
    from nacl.encoding import HexEncoder
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("WARNING: pynacl not available. Signature verification disabled.")


def recompute_hash(record: dict) -> str:
    """Recompute SHA-256 hash for a record."""
    # Extract fields that were hashed
    content = {
        k: record[k] for k in ("ts", "event_id", "event_type", "phase", "status", "metadata", "prev_hash", "pubkey")
        if k in record
    }
    blob = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def verify_signature(record: dict) -> bool:
    """Verify Ed25519 signature for a record."""
    if not CRYPTO_AVAILABLE:
        return None  # Cannot verify
    
    if record.get("ed25519_sig") == "CRYPTO_UNAVAILABLE":
        return None  # Signature not available
    
    try:
        pubkey = record.get("pubkey")
        if not pubkey:
            return False
        
        vk = VerifyKey(pubkey, encoder=HexEncoder)
        
        # Reconstruct content that was signed
        content = {
            k: record[k] for k in ("ts", "event_id", "event_type", "phase", "status", "metadata", "prev_hash", "pubkey")
            if k in record
        }
        blob = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        
        sig_hex = record.get("ed25519_sig")
        sig = bytes.fromhex(sig_hex)
        
        vk.verify(blob, sig)
        return True
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False


def verify_chain(records: list) -> dict:
    """Verify hash chain integrity."""
    results = {
        "total": len(records),
        "hash_valid": 0,
        "hash_invalid": 0,
        "chain_breaks": [],
        "sig_valid": 0,
        "sig_invalid": 0,
        "sig_unavailable": 0
    }
    
    for i, record in enumerate(records):
        # Verify hash
        expected_hash = recompute_hash(record)
        actual_hash = record.get("sha256")
        
        if expected_hash == actual_hash:
            results["hash_valid"] += 1
        else:
            results["hash_invalid"] += 1
            print(f"⚠ Hash mismatch at record {i}: expected {expected_hash[:16]}..., got {actual_hash[:16] if actual_hash else 'None'}...")
        
        # Verify chain link
        if i > 0:
            expected_prev = records[i-1].get("sha256")
            actual_prev = record.get("prev_hash")
            if expected_prev != actual_prev:
                results["chain_breaks"].append(i)
                print(f"⚠ Chain break at record {i}: expected prev {expected_prev[:16] if expected_prev else 'None'}..., got {actual_prev[:16] if actual_prev else 'None'}...")
        
        # Verify signature
        sig_result = verify_signature(record)
        if sig_result is True:
            results["sig_valid"] += 1
        elif sig_result is False:
            results["sig_invalid"] += 1
        else:
            results["sig_unavailable"] += 1
    
    return results


def display_event_flow(event_id: str, log_path: str):
    """Display complete flow for a given event_id."""
    records = []
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("event_id") == event_id:
                    records.append(record)
            except json.JSONDecodeError:
                continue
    
    if not records:
        print(f"No records found for event_id: {event_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"Event Flow for: {event_id}")
    print(f"{'='*80}\n")
    
    for i, record in enumerate(records):
        print(f"[{i}] Phase: {record.get('phase'):20s} Status: {record.get('status'):10s}")
        print(f"    Timestamp: {record.get('ts')}")
        print(f"    Event Type: {record.get('event_type')}")
        print(f"    Hash: {record.get('sha256')}")
        print(f"    Prev Hash: {record.get('prev_hash') or 'None (first record)'}")
        print(f"    Metadata: {json.dumps(record.get('metadata', {}), indent=2)}")
        print()
    
    # Verify integrity
    print(f"{'='*80}")
    print(f"Cryptographic Verification")
    print(f"{'='*80}\n")
    
    results = verify_chain(records)
    
    print(f"Total Records: {results['total']}")
    print(f"Hash Valid: {results['hash_valid']}")
    print(f"Hash Invalid: {results['hash_invalid']}")
    print(f"Chain Breaks: {len(results['chain_breaks'])}")
    print(f"Signatures Valid: {results['sig_valid']}")
    print(f"Signatures Invalid: {results['sig_invalid']}")
    print(f"Signatures Unavailable: {results['sig_unavailable']}")
    
    if results['hash_invalid'] == 0 and len(results['chain_breaks']) == 0 and results['sig_invalid'] == 0:
        print(f"\n✓✓✓ VERIFICATION PASSED ✓✓✓")
    else:
        print(f"\n✗✗✗ VERIFICATION FAILED ✗✗✗")


def main():
    parser = argparse.ArgumentParser(description="Verify Grace AI immutable log integrity")
    parser.add_argument("event_id", nargs="?", help="Event ID to verify")
    parser.add_argument("--all", action="store_true", help="Verify entire log")
    parser.add_argument("--last", type=int, metavar="N", help="Verify last N records")
    parser.add_argument("--log", help="Path to log file (default: from config)")
    
    args = parser.parse_args()
    
    # Get log path from config or args
    log_path = args.log
    if not log_path:
        try:
            from grace import config
            log_path = config.IMMUTABLE_LOG_PATH
        except Exception as e:
            log_path = "grace_data/grace_log.jsonl"
            print(f"WARNING: Could not load config, using default path: {log_path}")
    
    if not Path(log_path).exists():
        print(f"ERROR: Log file not found: {log_path}")
        return 1
    
    if args.event_id:
        display_event_flow(args.event_id, args.log)
    elif args.all:
        print("Verifying entire log...")
        records = []
        with open(args.log, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        results = verify_chain(records)
        print(f"\nTotal Records: {results['total']}")
        print(f"Hash Valid: {results['hash_valid']}")
        print(f"Hash Invalid: {results['hash_invalid']}")
        print(f"Chain Breaks: {len(results['chain_breaks'])}")
        print(f"Signatures Valid: {results['sig_valid']}")
        print(f"Signatures Invalid: {results['sig_invalid']}")
        if results['hash_invalid'] == 0 and len(results['chain_breaks']) == 0 and results['sig_invalid'] == 0:
            print(f"\n✓ VERIFICATION PASSED")
            return 0
        else:
            print(f"\n✗ VERIFICATION FAILED")
            return 1
    elif args.last:
        print(f"Verifying last {args.last} records...")
        records = []
        with open(args.log, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        records = records[-args.last:]
        results = verify_chain(records)
        print(f"\nRecords Checked: {results['total']}")
        print(f"Hash Valid: {results['hash_valid']}")
        print(f"Hash Invalid: {results['hash_invalid']}")
        print(f"Chain Breaks: {len(results['chain_breaks'])}")
        return 0 if results['hash_invalid'] == 0 and len(results['chain_breaks']) == 0 else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
