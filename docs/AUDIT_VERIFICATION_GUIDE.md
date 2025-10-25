# Grace AI - Cryptographic Audit & Verification Guide

## Overview

Grace AI implements a **cryptographic, immutable audit trail** for every event that flows through the system. Each event is tracked through distinct phases, with each phase logged using SHA-256 hashing and Ed25519 signatures to create a tamper-evident chain.

## Event Flow Phases

Every event goes through the following phases:

1. **RECEIVED** - Event enters the TriggerMesh
2. **MATCHED** or **NO_MATCH** - Event is routed to workflows
3. **HANDLER_EXECUTED** - Workflow handler begins execution
4. **HANDLER_COMMITTED** - Workflow successfully completes
5. **FAILED** - Workflow or handler fails (alternative to COMMITTED)

## Environment Variables

### Required

- `GRACE_ED25519_SK` - Ed25519 private signing key (hex-encoded)
  - If not set, a new key will be generated and saved to `grace_data/.grace_signing_key`
  - **Production**: Set this environment variable to persist keys across runs

### Optional

- `GRACE_ED25519_PK` - Ed25519 public verification key (hex-encoded)
- `GRACE_LOG_LEVEL` - Logging level (default: INFO)
- `GRACE_TRUST_THRESHOLD` - Minimum trust score for events (default: 0.3)
- `GRACE_REFLECTION_INTERVAL` - Self-reflection interval in seconds (default: 3600)
- `GRACE_DEV_MODE` - Enable development mode (default: true)

## File Paths

All data is stored in `grace_data/`:

- `grace_data/grace_log.jsonl` - Immutable audit log
- `grace_data/.grace_signing_key` - Ed25519 private key (generated if missing)
- `grace_data/memory/` - Knowledge graph and memory storage
- `grace_data/vectors/` - Vector embeddings

## Running Grace

### Basic Execution

```bash
python run_e2e_simulation.py
```

### With Persistent Keys

```bash
# Generate a new key pair (one time)
python -c "from nacl.signing import SigningKey; from nacl.encoding import HexEncoder; sk = SigningKey.generate(); print(f'Private: {sk.encode(encoder=HexEncoder).decode()}'); print(f'Public: {sk.verify_key.encode(encoder=HexEncoder).decode()}')"

# Export the private key
export GRACE_ED25519_SK="<your_private_key_hex>"

# Run Grace
python run_e2e_simulation.py
```

## Verifying the Audit Log

Grace provides a verification utility to validate cryptographic integrity:

### Verify a Specific Event

```bash
python tools/verify_immutable_log.py <event_id>
```

Output shows:
- Complete phase flow
- SHA-256 hash verification
- Ed25519 signature verification
- Chain integrity status

### Verify Entire Log

```bash
python tools/verify_immutable_log.py --all
```

### Verify Last N Records

```bash
python tools/verify_immutable_log.py --last 10
```

## End-to-End Testing

Run the comprehensive E2E test with phase verification:

```bash
python tests/test_e2e_phase_verification.py
```

This test:
1. Initializes Grace system
2. Dispatches a test event
3. Verifies all phases are logged
4. Validates cryptographic hash chain
5. Fails if any phase is missing or chain is broken

Expected output:
```
✓✓✓ E2E TEST PASSED ✓✓✓
All phases logged with valid cryptographic chain
```

## Audit Log Format

Each record in `grace_log.jsonl` contains:

```json
{
  "ts": 1234567890.123,
  "event_id": "uuid-here",
  "event_type": "external_data_received",
  "phase": "RECEIVED",
  "status": "ok",
  "metadata": {"size": 1024},
  "prev_hash": "sha256-of-previous-record",
  "pubkey": "ed25519-public-key-hex",
  "sha256": "sha256-of-this-record",
  "ed25519_sig": "signature-hex"
}
```

## Cryptographic Guarantees

### Hash Chaining

Each record includes:
- `sha256`: Hash of the current record's content
- `prev_hash`: Hash of the previous record

This creates a blockchain-like chain where tampering with any record breaks the chain.

### Digital Signatures

Each record is signed with Ed25519:
- **Private key** (`GRACE_ED25519_SK`): Used to sign records
- **Public key** (`pubkey` in record): Used to verify signatures

### Verification Process

The `verify_immutable_log.py` tool:

1. **Recomputes hashes**: Calculates SHA-256 of each record's content
2. **Compares**: Verifies computed hash matches stored `sha256`
3. **Checks chain**: Verifies each `prev_hash` matches previous record's `sha256`
4. **Verifies signatures**: Uses public key to verify Ed25519 signatures

## Workflows

### Available Workflows

1. **handle_external_data_received** - Demo workflow (Python object)
2. **data_ingestion_pipeline** - Full ingestion workflow with normalization, semantic tagging, and trust evaluation

### Creating New Workflows

Create a Python file in `grace/workflows/`:

```python
import logging

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "my_workflow"
EVENTS = ["my_event_type"]

class MyWorkflow:
    name = WORKFLOW_NAME
    EVENTS = EVENTS

    async def execute(self, event: dict):
        event_id = event.get("id", "unknown")
        logger.info(f"HANDLER_START {self.name} event_id={event_id}")
        
        # Your workflow logic here
        
        logger.info(f"HANDLER_DONE {self.name} event_id={event_id}")
        return {"status": "success"}

workflow = MyWorkflow()
```

## Troubleshooting

### Missing Phases

If verification fails with missing phases:

1. Check that workflows are executing: `grep HANDLER_START grace.log`
2. Verify immutable logger is initialized: `grep "Immutable Logger initialized"`
3. Check for errors during workflow execution

### Broken Chain

If hash chain verification fails:

1. Check for manual edits to `grace_log.jsonl`
2. Verify no concurrent writes to the log file
3. Check disk space and file permissions

### Signature Verification Fails

1. Ensure `pynacl` is installed: `pip install pynacl`
2. Verify the public key in records matches your signing key
3. Check that `GRACE_ED25519_SK` hasn't changed between runs

## Security Best Practices

1. **Protect Private Keys**: Never commit `GRACE_ED25519_SK` to version control
2. **Read-Only Logs**: Set `grace_log.jsonl` to read-only after writing (optional)
3. **Backup Keys**: Store Ed25519 keys in secure key management (production)
4. **Monitor Chain**: Regularly run `verify_immutable_log.py --all`
5. **Audit Access**: Log all access to the audit log itself

## Development Mode

When `GRACE_DEV_MODE=true` (default):
- Private keys are logged to console (for debugging)
- Detailed phase logging is enabled
- Permissive trust thresholds

**Production**: Set `GRACE_DEV_MODE=false`

## Support

For issues or questions:
1. Check logs in console output
2. Run verification tool: `python tools/verify_immutable_log.py --all`
3. Review this documentation
4. Check `grace/config.py` for configuration options
