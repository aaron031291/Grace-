from pathlib import Path
import os

# Resolve repo root two levels up from this file (…/Grace-/grace/config.py -> repo)
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Directory for all persistent data
GRACE_DATA_DIR = Path(os.environ.get("GRACE_DATA_DIR", _REPO_ROOT / "grace_data"))
GRACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Key file paths (Path objects, not strings)
IMMUTABLE_LOG_PATH = GRACE_DATA_DIR / "immutable.audit.log"
TRUST_LEDGER_PATH  = GRACE_DATA_DIR / "trust_ledger.jsonl"
