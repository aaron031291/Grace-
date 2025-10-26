from pathlib import Path
import os

# Resolve repo root two levels up from this file or package
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure data dir is a Path and exists
GRACE_DATA_DIR = Path(os.environ.get("GRACE_DATA_DIR", _REPO_ROOT / "grace_data"))
GRACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Required paths
IMMUTABLE_LOG_PATH = GRACE_DATA_DIR / "immutable.audit.log"
TRUST_LEDGER_PATH  = GRACE_DATA_DIR / "trust_ledger.jsonl"

# Touch files so downstream code can open them
IMMUTABLE_LOG_PATH.touch(exist_ok=True)
TRUST_LEDGER_PATH.touch(exist_ok=True)

# Additional config attributes
ED25519_SK_HEX = os.getenv("GRACE_ED25519_SK", "").strip()
ED25519_PUB_HEX = os.getenv("GRACE_ED25519_PUB", "").strip()
WORKFLOW_DIR = os.getenv("GRACE_WORKFLOW_DIR", "grace/workflows")
CHECKPOINT_EVERY_N = int(os.getenv("GRACE_CHECKPOINT_EVERY_N", "100"))


def get_config():
    """Return a dictionary of all configuration values."""
    return {
        "GRACE_DATA_DIR": GRACE_DATA_DIR,
        "IMMUTABLE_LOG_PATH": IMMUTABLE_LOG_PATH,
        "TRUST_LEDGER_PATH": TRUST_LEDGER_PATH,
        "ED25519_SK_HEX": ED25519_SK_HEX,
        "ED25519_PUB_HEX": ED25519_PUB_HEX,
        "WORKFLOW_DIR": WORKFLOW_DIR,
        "CHECKPOINT_EVERY_N": CHECKPOINT_EVERY_N,
    }
