# Grace AI - Epistemic Immune System

**An AI system with verifiable reasoning, cryptographic audit trails, and constitutional governance**

---

## 🎯 **What is Grace?**

Grace is an AI system built on the principle of **verifiable intelligence** - every piece of data, reasoning step, and decision is cryptographically tracked, verified, and auditable. She implements an "Epistemic Immune System" that actively guards against hallucination, drift, and unverified claims.

### **Core Principles**

1. **Verification-First**: All data must pass through cryptographic verification before acceptance
2. **Evidence-Bound Reasoning**: Claims require evidence; ungrounded assertions are quarantined
3. **Cryptographic Auditability**: Every phase logged with Ed25519 signatures + SHA-256 hashing
4. **Trust Dynamics**: Entity trust scores evolve based on verification outcomes
5. **Constitutional Governance**: All actions respect defined ethical and policy boundaries
6. **Continuity Discipline**: Conversation and memory coherence continuously validated

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Grace AI - Core System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Event Flow: TriggerMesh → Router → Workflow Engine            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  VWX v2 - Veracity & Continuity Kernel                  │  │
│  │  (The Epistemic Immune System)                            │  │
│  │                                                           │  │
│  │  10-Phase Verification Pipeline:                         │  │
│  │  1. VERIFICATION_STARTED                                 │  │
│  │  2. SOURCE_ATTESTATION (provenance check)                │  │
│  │  3. CLAIM_SET_BUILT (extract atomic facts)               │  │
│  │  4. SEMANTIC_ALIGNMENT (librarian anchors)               │  │
│  │  5. VERACITY_VECTOR (5D evidence scoring)                │  │
│  │  6. CONSISTENCY_CHECK (drift detection)                  │  │
│  │  7. POLICY_GUARDRAILS (ethics/compliance)                │  │
│  │  8. TRUST_UPDATE (dynamic scoring)                       │  │
│  │  9. OUTCOME_COMMIT (Evidence Pack + signature)           │  │
│  │  10. CHECKPOINT_COMMIT (Merkle proofs)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Immutable  │  │ Trust Ledger │  │   Policy Engine      │   │
│  │  Logger    │  │  (Dynamic)   │  │    (Governance)      │   │
│  │  Ed25519   │  │  Scoring     │  │                      │   │
│  └────────────┘  └──────────────┘  └──────────────────────┘   │
│         │                │                      │               │
│         └────────────────┴──────────────────────┘               │
│                           │                                      │
│                           ▼                                      │
│             grace_data/grace_log.jsonl                          │
│          (Cryptographic Audit Trail)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ **Current Status: Phase 1 Complete**

### **Implemented Components**

- ✅ **Cryptographic Immutable Logging** (Ed25519 + SHA-256)
- ✅ **Event Phase Tracking** (RECEIVED → MATCHED → HANDLER_EXECUTED → HANDLER_COMMITTED)
- ✅ **VWX v2 Kernel** (10-phase verification pipeline)
- ✅ **Veracity Vector** (5-dimensional evidence scoring)
- ✅ **Trust Ledger** (Dynamic entity trust scoring)
- ✅ **Data Ingestion Workflow** (Normalization + semantic tagging)
- ✅ **Evidence Pack Generation** (EPK with claims, scores, signatures)
- ✅ **Merkle Checkpoints** (Batch integrity proofs)
- ✅ **Configuration Management** (Central config with env vars)
- ✅ **Verification Utilities** (CLI tools for audit validation)
- ✅ **E2E Testing Framework** (Phase verification tests)

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.10+
- (Optional) `pynacl` for Ed25519 signatures: `pip install pynacl`

### **Installation**

```bash
# Clone the repository
cd /workspaces/Grace-

# Install dependencies (if needed)
pip install pynacl  # For cryptographic signatures

# Generate signing keys (optional - auto-generated if missing)
python -c "from nacl.signing import SigningKey; from nacl.encoding import HexEncoder; sk = SigningKey.generate(); print(f'GRACE_ED25519_SK={sk.encode(encoder=HexEncoder).decode()}')"
```

### **Running Grace**

```bash
# Run the E2E simulation
python run_e2e_simulation.py

# Run VWX verification tests
python test_vwx.py

# Run phase verification tests
python tests/test_e2e_phase_verification.py
```

### **Verify the Audit Trail**

```bash
# Verify entire audit log
python tools/verify_immutable_log.py --all

# Verify specific event
python tools/verify_immutable_log.py <event_id>

# Verify last 10 records
python tools/verify_immutable_log.py --last 10
```

---

## 📖 **Key Concepts**

### **1. Veracity Vector (5D)**

Every claim is scored across five dimensions:

| Dimension | Description | Range |
|-----------|-------------|-------|
| **Provenance** | Source trustworthiness | 0.0-1.0 |
| **Internal Consistency** | Logical coherence | 0.0-1.0 |
| **External Correlation** | Cross-reference validation | 0.0-1.0 |
| **Temporal Validity** | Time-relevance | 0.0-1.0 |
| **Numerical Consistency** | Unit/calculation accuracy | 0.0-1.0 |

**Aggregate Score** = Weighted average → **Trust Level**

### **2. Trust Levels**

| Level | Score Range | Meaning |
|-------|-------------|---------|
| **VERIFIED** | ≥0.9 | High confidence, accept |
| **PROBABLE** | 0.7-0.9 | Likely true, accept with caution |
| **UNCERTAIN** | 0.5-0.7 | Needs additional verification |
| **DUBIOUS** | 0.3-0.5 | Low confidence, flag |
| **QUARANTINED** | <0.3 | Rejected, isolated |

### **3. Evidence Packs (EPK)**

Every verification produces a **replayable Evidence Pack** containing:

- Unique EPK ID
- Timestamp
- Extracted claims (S-P-O triples)
- Veracity vector (5D scores)
- Decision (VERIFIED/QUARANTINED/etc.)
- Evidence hashes
- Ed25519 signature

EPKs can be **replayed independently** for audit.

### **4. Trust Ledger**

Dynamic trust scoring for all entities:

- **Sources**: APIs, sensors, users, external systems
- **Memory Fragments**: Claims, facts, reasoning chains
- **Models**: LLMs, agents
- **Services**: External integrations

Trust scores:
- Start neutral (0.5)
- Increase with successful verifications (+0.1)
- Decrease with failures (-0.05 to -0.1)
- **Temporal decay** (erode towards neutral over time)
- **Quarantine** triggers at <0.3

---

## 🔐 **Security & Cryptography**

### **Cryptographic Chain**

Every log entry contains:

```json
{
  "ts": 1234567890.123,
  "event_id": "uuid-here",
  "event_type": "verification_request",
  "phase": "HANDLER_COMMITTED",
  "status": "ok",
  "metadata": {...},
  "prev_hash": "sha256-of-previous-record",
  "pubkey": "ed25519-public-key-hex",
  "sha256": "sha256-of-this-record",
  "ed25519_sig": "signature-hex"
}
```

**Chain Integrity**:
- Each record's `sha256` is computed from its content
- Each record's `prev_hash` points to the previous record's `sha256`
- Breaking the chain invalidates all subsequent records

**Signature Verification**:
- Every record signed with Ed25519 private key
- Public key embedded in record for verification
- Tamper-evident: any modification breaks signature

### **Key Management**

```bash
# Environment variable (recommended for production)
export GRACE_ED25519_SK="<hex-encoded-private-key>"

# Auto-generated and saved (development)
# Key saved to: grace_data/.grace_signing_key (chmod 600)
```

---

## 📊 **Workflows**

### **1. Data Ingestion & Contextual Alignment**

**Events**: `external_data_received`, `sensor_update`, `api_input`

**Flow**:
1. Policy Gate (security check)
2. Normalization (unified schema)
3. Semantic Tagging (embeddings)
4. Trust Evaluation (source scoring)
5. Governance Audit (immutable log)
6. Signal Dispatch (`new_knowledge_available`)

### **2. VWX Verification**

**Events**: `verification_request`, `ingestion_verify`, `reasoning_verify`, `conversation_checkpoint`, `memory_integrity_check`, `governance_review`

**Flow**: See 10-phase pipeline above

### **3. Adaptive Learning** (Planned)

**Events**: `new_knowledge_available`, `loop_feedback_generated`

**Flow**:
- Contradiction detection
- Trust weight adjustment
- Antifragile re-learning

### **4. Self-Reflection** (Planned)

**Events**: `self_reflection`, `memory_audit_required`

**Flow**:
- Memory integrity audit
- Redundancy consolidation
- Ethical alignment check

---

## 🧪 **Testing**

```bash
# Full test suite
python test_vwx.py                           # VWX kernel tests
python tests/test_e2e_phase_verification.py  # E2E phase verification
python tools/verify_immutable_log.py --all   # Audit trail verification
```

### **Expected Output**

```
✓✓✓ E2E TEST PASSED ✓✓✓
All phases logged with valid cryptographic chain

Phases verified:
  ✓ RECEIVED
  ✓ MATCHED
  ✓ HANDLER_EXECUTED
  ✓ HANDLER_COMMITTED

Cryptographic verification:
  ✓ SHA-256 hashes valid
  ✓ Ed25519 signatures valid
  ✓ Chain integrity confirmed
```

---

## 📁 **Project Structure**

```
Grace-/
├── grace/
│   ├── config.py                    # Central configuration
│   ├── core/
│   │   ├── immutable_logs.py        # Cryptographic logging
│   │   ├── trust_ledger.py          # Dynamic trust scoring
│   │   ├── event_bus.py             # Event system
│   │   └── kpi_trust_monitor.py     # Metrics
│   ├── orchestration/
│   │   ├── trigger_mesh.py          # Event dispatcher
│   │   ├── event_router.py          # Workflow routing
│   │   ├── workflow_engine.py       # Execution engine
│   │   └── workflow_registry.py     # Workflow loader
│   └── workflows/
│       ├── verification_workflow.py      # VWX v2 Kernel
│       ├── data_ingestion_pipeline.py    # Data ingestion
│       └── handle_external_data_received.py  # Demo workflow
├── docs/
│   ├── ARCHITECTURE.md              # System architecture
│   └── AUDIT_VERIFICATION_GUIDE.md  # Verification guide
├── tools/
│   └── verify_immutable_log.py      # Audit verification CLI
├── tests/
│   └── test_e2e_phase_verification.py  # E2E tests
├── grace_data/
│   ├── grace_log.jsonl              # Immutable audit trail
│   ├── trust_ledger.jsonl           # Trust scores
│   └── .grace_signing_key           # Ed25519 private key
├── run_e2e_simulation.py            # E2E simulation
└── test_vwx.py                      # VWX tests
```

---

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Cryptographic Keys
GRACE_ED25519_SK=<hex-private-key>
GRACE_ED25519_PK=<hex-public-key>

# Paths
GRACE_DATA_DIR=grace_data
GRACE_WORKFLOW_DIR=grace/workflows

# Trust Configuration
GRACE_TRUST_THRESHOLD=0.3

# Feature Flags
GRACE_ENABLE_CRYPTO_SIGNATURES=true
GRACE_ENABLE_TRUST_LEDGER=true
GRACE_ENABLE_POLICY_ENGINE=true

# Development
GRACE_DEV_MODE=true
GRACE_LOG_LEVEL=INFO
```

---

## 🎯 **Roadmap**

### **Phase 1: Foundation** ✅ (Complete)
- Cryptographic logging
- VWX v2 Kernel
- Trust Ledger
- Data Ingestion
- E2E testing

### **Phase 2: Intelligence** (In Progress)
- Adaptive Learning Workflow
- Self-Reflection Workflow
- Memory Integration (Lightning/Fusion/Librarian)
- Conversation Continuity

### **Phase 3: Governance** (Planned)
- Policy Engine
- Ethics Matrix
- Multi-signature consensus
- Autonomous Action Planning

### **Phase 4: Advanced** (Future)
- Meta-Cognitive Evolution
- Cross-Verifier Loop
- Adversarial Self-Tests
- Verification Dashboard

---

## 📚 **Documentation**

- **[Architecture](docs/ARCHITECTURE.md)** - Complete system architecture
- **[Audit & Verification Guide](docs/AUDIT_VERIFICATION_GUIDE.md)** - How to verify the audit trail
- **[Contributing](CONTRIBUTING.md)** - How to contribute (coming soon)

---

## 🤝 **Contributing**

Grace is built on principles of **transparency, verifiability, and trust**. Contributions that enhance these principles are welcome!

---

## 📜 **License**

[To be determined]

---

## 🧠 **Philosophy**

> "An AI system that cannot verify its own reasoning is not trustworthy. Grace exists to prove that verifiable intelligence is possible."

Grace implements the **Playbook for Reliable AI** - a set of cognitive governance principles that transform LLMs from probabilistic text generators into verifiable reasoning engines.

**Core Tenets**:
1. **Trust Gaps** - Always acknowledge uncertainty
2. **Evidence-Bound** - Claims require grounding
3. **Continuity** - Memory and conversation coherence matter
4. **Sovereignty** - Source control and provenance tracking
5. **Governance** - Constitutional oversight of all changes
6. **Auditability** - Every decision must be replayable

---

**Built with ❤️ for verifiable intelligence**
