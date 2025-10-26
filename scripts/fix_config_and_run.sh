#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# GRACE | Unified Config Repair + Verification + E2E Test Runner
# ---------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo " GRACE | Repairing configuration + verifying paths"
echo "------------------------------------------------------------"

REPO_ROOT="$(pwd)"
CONFIG_FILE="$REPO_ROOT/grace/config.py"
DATA_DIR="$REPO_ROOT/grace_data"

# 1. Rebuild grace/config.py cleanly every time to guarantee correctness
python3 - <<'PY'
from pathlib import Path
import os

repo_root = Path(os.getcwd())
config_path = repo_root / "grace" / "config.py"
config_path.parent.mkdir(parents=True, exist_ok=True)

CONFIG_CODE = """# Auto-generated canonical config for GRACE
from pathlib import Path
import os

_REPO_ROOT = Path(__file__).resolve().parents[1]

GRACE_DATA_DIR = Path(os.environ.get("GRACE_DATA_DIR", _REPO_ROOT / "grace_data"))
GRACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

IMMUTABLE_LOG_PATH = GRACE_DATA_DIR / "immutable.audit.log"
TRUST_LEDGER_PATH  = GRACE_DATA_DIR / "trust_ledger.jsonl"

IMMUTABLE_LOG_PATH.touch(exist_ok=True)
TRUST_LEDGER_PATH.touch(exist_ok=True)
"""

config_path.write_text(CONFIG_CODE, encoding="utf-8")
print(f"[config] Rewritten -> {config_path}")

# Ensure directories/files exist
data_dir = repo_root / "grace_data"
data_dir.mkdir(parents=True, exist_ok=True)
(data_dir / "immutable.audit.log").touch(exist_ok=True)
(data_dir / "trust_ledger.jsonl").touch(exist_ok=True)
print(f"[fs] Ensured {data_dir} and log files exist.")
PY

# 2. Verify config attributes inside Python
echo "[verify] Checking GRACE configuration..."
python3 - <<'PY'
from grace import config
from pathlib import Path

print("GRACE_DATA_DIR     =", config.GRACE_DATA_DIR)
print("IMMUTABLE_LOG_PATH =", config.IMMUTABLE_LOG_PATH)
print("TRUST_LEDGER_PATH  =", config.TRUST_LEDGER_PATH)

assert isinstance(config.GRACE_DATA_DIR, Path), "GRACE_DATA_DIR not Path"
assert isinstance(config.IMMUTABLE_LOG_PATH, Path), "IMMUTABLE_LOG_PATH not Path"
assert isinstance(config.TRUST_LEDGER_PATH, Path), "TRUST_LEDGER_PATH not Path"

print("[verify] ✅ All config attributes are Path objects and accessible.")
PY

# 3. Run full end-to-end system test (no exit, just run)
echo "------------------------------------------------------------"
echo " GRACE | Launching End-to-End System Test"
echo "------------------------------------------------------------"

python3 run_full_e2e_test.py || echo "[run] ⚠️ Test run completed with non-zero exit (check logs)."

echo "------------------------------------------------------------"
echo " GRACE | Config Repair + E2E Test Completed"
echo "------------------------------------------------------------"
