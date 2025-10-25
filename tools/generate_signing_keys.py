#!/usr/bin/env python3
"""
Grace AI - Cryptographic Key Generator
Generates Ed25519 key pair for persistent signing
"""
import sys

try:
    from nacl.signing import SigningKey
    from nacl.encoding import HexEncoder
except ImportError:
    print("ERROR: pynacl not installed")
    print("Install with: pip install pynacl")
    sys.exit(1)

# Generate new key pair
sk = SigningKey.generate()
sk_hex = sk.encode(encoder=HexEncoder).decode()
pk_hex = sk.verify_key.encode(encoder=HexEncoder).decode()

print("\n" + "="*80)
print("Grace AI - Ed25519 Key Pair Generated")
print("="*80)
print("\nAdd these to your shell environment (e.g., ~/.bashrc or .env):\n")
print(f'export GRACE_ED25519_SK="{sk_hex}"')
print(f'export GRACE_ED25519_PUB="{pk_hex}"')
print("\n" + "="*80)
print("\nFor this session only:")
print("="*80)
print(f"\nexport GRACE_ED25519_SK={sk_hex}")
print(f"export GRACE_ED25519_PUB={pk_hex}\n")
print("="*80)
print("\n⚠️  SECURITY WARNING:")
print("  - Keep the private key (GRACE_ED25519_SK) secret")
print("  - Never commit keys to version control")
print("  - Store in secure key management for production")
print("="*80 + "\n")
