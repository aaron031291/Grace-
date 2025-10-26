"""
Grace AI - Central Configuration Module
Manages all system paths, keys, and runtime settings.
This module is designed to be self-contained and robust.
"""
from pathlib import Path
import os
import logging

# --- Path Configuration ---

# Use the location of this file to robustly find the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory (creates ./grace_data by default; override with GRACE_DATA_DIR env var)
# This is the single source of truth for the data directory path.
GRACE_DATA_DIR = Path(os.environ.get("GRACE_DATA_DIR", _PROJECT_ROOT / "grace_data"))

# Core subdirectories
LOG_DIR = GRACE_DATA_DIR / "logs"
AUDIT_LOG_DIR = GRACE_DATA_DIR / "audit"
VECTOR_DB_PATH = GRACE_DATA_DIR / "vector_db"
TMP_DIR = GRACE_DATA_DIR / "tmp"
REPORTS_DIR = GRACE_DATA_DIR / "reports"
SYSTEM_DIR = GRACE_DATA_DIR / "system"
WORKFLOW_DIR = _PROJECT_ROOT / "grace" / "workflows"

# Canonical file paths
# Services should import and use these constants directly.
IMMUTABLE_LOG_PATH = AUDIT_LOG_DIR / "immutable_log.jsonl"
TRUST_LEDGER_PATH = GRACE_DATA_DIR / "trust_ledger.jsonl"
SYSTEM_STATUS_PATH = SYSTEM_DIR / "status.json"
E2E_REPORT_PATH = REPORTS_DIR / "e2e_report.json"

# --- Create all necessary directories and files on import ---
# This ensures that any part of the system can import this module and be
# confident that the necessary file structure exists.
_ALL_DIRS = {
    GRACE_DATA_DIR, LOG_DIR, AUDIT_LOG_DIR, VECTOR_DB_PATH,
    TMP_DIR, REPORTS_DIR, SYSTEM_DIR, WORKFLOW_DIR
}
_ALL_FILES = {IMMUTABLE_LOG_PATH, TRUST_LEDGER_PATH, SYSTEM_STATUS_PATH, E2E_REPORT_PATH}

for directory in _ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

for file_path in _ALL_FILES:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)

logging.getLogger(__name__).info(f"Grace config initialized. Data directory: {GRACE_DATA_DIR}")


# --- Cryptographic Keys ---
ED25519_SK_HEX = os.getenv("GRACE_ED25519_SK", "").strip()
ED25519_PUB_HEX = os.getenv("GRACE_ED25519_PUB", "").strip()

# --- Verification & Checkpointing ---
CHECKPOINT_EVERY_N = int(os.getenv("GRACE_CHECKPOINT_EVERY_N", "100"))

# --- Optional External Services ---
POSTGRES_URL = os.getenv("POSTGRES_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")