import os

GRACE_HOME = os.getenv("GRACE_HOME", os.path.abspath("./"))
GRACE_DATA_DIR = os.getenv("GRACE_DATA_DIR", os.path.join(GRACE_HOME, "grace_data"))

# Core dirs
LOG_DIR        = os.getenv("GRACE_LOG_DIR",     os.path.join(GRACE_DATA_DIR, "logs"))
AUDIT_LOG_DIR  = os.getenv("AUDIT_LOG_DIR",     os.path.join(GRACE_DATA_DIR, "audit"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH",    os.path.join(GRACE_DATA_DIR, "vector_db"))
TMP_DIR        = os.getenv("GRACE_TMP_DIR",     os.path.join(GRACE_DATA_DIR, "tmp"))
REPORTS_DIR    = os.getenv("REPORTS_DIR",       os.path.join(GRACE_DATA_DIR, "reports"))
SYSTEM_DIR     = os.getenv("SYSTEM_DIR",        os.path.join(GRACE_DATA_DIR, "system"))

# Files (parents will be created by launcher)
TRUST_LEDGER_PATH   = os.getenv("TRUST_LEDGER_PATH",   os.path.join(GRACE_DATA_DIR, "trust_ledger.jsonl"))
IMMUTABLE_LOG_PATH  = os.getenv("IMMUTABLE_LOG_PATH",  os.path.join(AUDIT_LOG_DIR,  "immutable_logs.jsonl"))
SYSTEM_STATUS_PATH  = os.getenv("SYSTEM_STATUS_PATH",  os.path.join(SYSTEM_DIR,     "status.json"))
E2E_REPORT_PATH     = os.getenv("E2E_REPORT_PATH",     os.path.join(REPORTS_DIR,    "e2e_report.json"))
HEALTHCHECK_REPORT  = os.getenv("HEALTHCHECK_REPORT",  os.path.join(REPORTS_DIR,    "healthcheck.json"))
VERIFICATION_REPORT = os.getenv("VERIFICATION_REPORT", os.path.join(REPORTS_DIR,    "verification.json"))

# --- Crypto keys ---
ED25519_SK_HEX      = os.getenv("GRACE_ED25519_SK", "").strip()  # required in non-dev
ED25519_PUB_HEX     = os.getenv("GRACE_ED25519_PUB", "").strip()

# --- Workflows ---
WORKFLOW_DIR        = os.getenv("GRACE_WORKFLOW_DIR", "grace/workflows")

# --- Verification/checkpointing ---
CHECKPOINT_EVERY_N  = int(os.getenv("GRACE_CHECKPOINT_EVERY_N", "100"))

# Optional external services (won’t crash if unset)
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://localhost:5432/grace")
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY= os.getenv("ANTHROPIC_API_KEY", "")