# Grace AI - Quick Start Guide

## 🚀 Setup & Installation

### **1. Generate Persistent Signing Keys (One-Time Setup)**

Grace uses Ed25519 cryptographic signatures for immutable audit logs. Generate a persistent key pair:

```bash
# Generate keys
python tools/generate_signing_keys.py

# Copy the export commands to your shell
# Example output:
# export GRACE_ED25519_SK="abc123..."
# export GRACE_ED25519_PUB="def456..."
```

**For persistent setup**, add to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export GRACE_ED25519_SK="<your-private-key>"' >> ~/.bashrc
echo 'export GRACE_ED25519_PUB="<your-public-key>"' >> ~/.bashrc
source ~/.bashrc
```

**Alternatively**, Grace will auto-generate and save keys to `grace_data/.grace_signing_key` if not provided.

---

### **2. Install Dependencies (Optional)**

For cryptographic signatures (recommended):

```bash
pip install pynacl
```

If `pynacl` is not installed, Grace will run without Ed25519 signatures (logging only SHA-256 hashes).

---

## 🧪 Running Tests

### **Quick Test - VWX Verification**

```bash
python test_vwx.py
```

**Expected Output:**
```
✓ Grace initialized
✓ High veracity event processed
✓ Medium veracity event processed
✓ Low veracity event processed
✓✓✓ VWX verification working correctly
```

---

### **Comprehensive E2E Test**

```bash
python run_full_e2e_test.py
```

**This tests:**
- ✅ Component initialization
- ✅ Workflow loading
- ✅ Event dispatch & routing
- ✅ Immutable logging & cryptographic chain
- ✅ Trust ledger (dynamic scoring)
- ✅ VWX verification (5D veracity vector)

**Expected Output:**
```
================================================================================
FINAL TEST REPORT
================================================================================
  ✓ PASS   Component Initialization
  ✓ PASS   Workflow Loading
  ✓ PASS   Event Dispatch & Routing
  ✓ PASS   Immutable Logging
  ✓ PASS   Trust Ledger
  ✓ PASS   VWX Verification

================================================================================
✓✓✓ ALL TESTS PASSED ✓✓✓
Grace AI system is fully operational!
================================================================================
```

---

## 🔍 Verifying the Audit Trail

### **Verify Entire Log**

```bash
python tools/verify_immutable_log.py --all
```

**Expected Output:**
```
Total Records: 45
Hash Valid: 45
Hash Invalid: 0
Chain Breaks: 0
Signatures Valid: 45
Signatures Invalid: 0

✓ VERIFICATION PASSED
```

---

### **Verify Specific Event**

```bash
python tools/verify_immutable_log.py <event_id>
```

**Shows:**
- Complete phase flow
- SHA-256 hash verification
- Ed25519 signature verification
- Chain integrity status

---

### **Verify Last N Records**

```bash
python tools/verify_immutable_log.py --last 10
```

---

## 📊 Inspecting Data

### **View Immutable Log**

```bash
cat grace_data/grace_log.jsonl | jq | head -50
```

**Sample Record:**
```json
{
  "ts": 1234567890.123,
  "event_id": "abc-123",
  "event_type": "verification_request",
  "phase": "HANDLER_COMMITTED",
  "status": "ok",
  "metadata": {
    "workflow": "veracity_continuity_kernel",
    "veracity_aggregate": 0.95
  },
  "prev_hash": "sha256-of-previous-record",
  "pubkey": "ed25519-public-key-hex",
  "sha256": "sha256-of-this-record",
  "ed25519_sig": "signature-hex"
}
```

---

### **View Trust Ledger**

```bash
cat grace_data/trust_ledger.jsonl | jq
```

**Sample Record:**
```json
{
  "entity_id": "verified_api",
  "entity_type": "source",
  "trust_score": 0.95,
  "confidence": 0.85,
  "last_updated": 1234567890.123,
  "total_interactions": 42,
  "successful_verifications": 40,
  "failed_verifications": 2,
  "quarantine_count": 0,
  "metadata": {
    "last_event_id": "abc-123",
    "last_reason": "VWX verification delta=+0.10"
  }
}
```

---

## ⚙️ Configuration

All configuration is in `grace/config.py` and can be overridden via environment variables:

### **Paths**

```bash
export GRACE_DATA_DIR="grace_data"                    # Data directory
export GRACE_IMMUTABLE_LOG="grace_data/grace_log.jsonl"  # Audit log
export GRACE_TRUST_LEDGER="grace_data/trust_ledger.jsonl"  # Trust scores
export GRACE_WORKFLOW_DIR="grace/workflows"           # Workflows directory
```

### **Cryptographic Keys**

```bash
export GRACE_ED25519_SK="<hex-private-key>"   # Required for signatures
export GRACE_ED25519_PUB="<hex-public-key>"   # Optional (derived from SK)
```

### **Verification Settings**

```bash
export GRACE_CHECKPOINT_EVERY_N="100"         # Merkle checkpoint interval
export GRACE_TRUST_THRESHOLD="0.3"            # Minimum trust score
```

### **Development Mode**

```bash
export GRACE_DEV_MODE="true"                  # Enable dev mode
export GRACE_LOG_LEVEL="INFO"                 # Logging level
```

---

## 🎯 Common Workflows

### **1. Run a Single Test**

```bash
# VWX verification test
python test_vwx.py

# Phase verification test
python tests/test_e2e_phase_verification.py
```

---

### **2. Full System Test + Verification**

```bash
# Run comprehensive test
python run_full_e2e_test.py

# Verify audit trail
python tools/verify_immutable_log.py --all

# Check trust ledger stats
cat grace_data/trust_ledger.jsonl | jq -s 'group_by(.entity_type) | map({type: .[0].entity_type, count: length})'
```

---

### **3. Clean Start**

```bash
# Remove all data (preserves signing keys if in environment)
rm -rf grace_data/*

# Run test to regenerate
python run_full_e2e_test.py
```

---

## 📁 Project Structure

```
Grace-/
├── grace/
│   ├── config.py                          # Central configuration ✅
│   ├── launcher.py                        # System launcher ✅
│   ├── core/
│   │   ├── immutable_logs.py              # Cryptographic logging ✅
│   │   ├── trust_ledger.py                # Trust scoring ✅
│   │   ├── service_registry.py            # Service registry
│   │   └── event_bus.py                   # Event bus
│   ├── orchestration/
│   │   ├── trigger_mesh.py                # Event dispatcher
│   │   ├── event_router.py                # Event routing
│   │   ├── workflow_engine.py             # Execution engine
│   │   └── workflow_registry.py           # Workflow loader
│   └── workflows/
│       ├── verification_workflow.py       # VWX v2 Kernel ✅
│       ├── data_ingestion_pipeline.py     # Data ingestion ✅
│       └── handle_external_data_received.py  # Demo workflow
├── tools/
│   ├── verify_immutable_log.py            # Audit verification CLI ✅
│   └── generate_signing_keys.py           # Key generator ✅
├── tests/
│   └── test_e2e_phase_verification.py     # E2E tests ✅
├── test_vwx.py                            # VWX tests ✅
├── run_full_e2e_test.py                   # Comprehensive E2E ✅
└── grace_data/
    ├── grace_log.jsonl                    # Immutable audit trail
    ├── trust_ledger.jsonl                 # Trust scores
    └── .grace_signing_key                 # Ed25519 private key (auto-generated)
```

---

## 🔐 Security Notes

### **Production Deployment**

1. **Set persistent keys:**
   ```bash
   export GRACE_ED25519_SK="<secure-key>"
   ```

2. **Disable dev mode:**
   ```bash
   export GRACE_DEV_MODE="false"
   ```

3. **Secure key storage:**
   - Use environment variables (not files)
   - Use secret management (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Never commit keys to version control

4. **File permissions:**
   ```bash
   chmod 600 grace_data/.grace_signing_key
   chmod 644 grace_data/*.jsonl
   ```

---

## 🆘 Troubleshooting

### **"pynacl not available" Warning**

```bash
pip install pynacl
```

### **"Log file not found" Error**

Run a test first to generate logs:
```bash
python test_vwx.py
```

### **Missing Phases in Verification**

Check that workflows are loaded:
```bash
ls grace/workflows/*.py
```

Ensure `grace/workflows/__init__.py` exists (can be empty).

### **"No factory registered" Warning**

This is normal if a service is not yet implemented. The system will continue with available services.

---

## 📚 Next Steps

1. **Read the Architecture:** `docs/ARCHITECTURE.md`
2. **Read the Verification Guide:** `docs/AUDIT_VERIFICATION_GUIDE.md`
3. **Explore Workflows:** `grace/workflows/`
4. **Create Custom Workflows:** See `grace/workflows/verification_workflow.py` as example

---

**Built with ❤️ for verifiable intelligence**
