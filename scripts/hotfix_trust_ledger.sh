#!/usr/bin/env bash
set -euo pipefail

echo "------------------------------------------------------------"
echo " GRACE | Hotfix: TrustLedger factory + DEV_MODE in config"
echo "------------------------------------------------------------"

REPO_ROOT="$(pwd)"
CFG_PKG="$REPO_ROOT/grace/config/__init__.py"
LAUNCHER="$REPO_ROOT/grace/launcher.py"

# A) Ensure DEV_MODE exists in config (some code paths reference it)
if grep -q "^DEV_MODE" "$CFG_PKG"; then
  echo "[config] DEV_MODE already present"
else
  printf "\n# Development toggle used by subsystems\nDEV_MODE = bool(int(os.environ.get('GRACE_DEV_MODE', '0')))\n" >> "$CFG_PKG"
  echo "[config] Added DEV_MODE to grace/config/__init__.py"
fi

# B) Patch launcher to adapt to TrustLedger.__init__ signature
#    We wrap the factory with a small helper that inspects the constructor.
if ! grep -q "def _make_trust_ledger(" "$LAUNCHER"; then
  python3 - <<'PY'
from pathlib import Path
p = Path("grace/launcher.py")
src = p.read_text(encoding="utf-8")

HELPER = r'''
# --- BEGIN HOTFIX: Adaptive TrustLedger factory ---
def _make_trust_ledger(cfg):
    from inspect import signature
    from grace.core.trust_ledger import TrustLedger
    ledger_path = str(cfg.TRUST_LEDGER_PATH)
    try:
        sig = signature(TrustLedger.__init__)
        params = sig.parameters
        if "persistence_path" in params:
            return TrustLedger(persistence_path=ledger_path)
        elif "path" in params:
            return TrustLedger(path=ledger_path)
        elif "storage_path" in params:
            return TrustLedger(storage_path=ledger_path)
        elif "file_path" in params:
            return TrustLedger(file_path=ledger_path)
        else:
            # try positional
            return TrustLedger(ledger_path)
    except Exception:
        # last resort: positional
        return TrustLedger(ledger_path)
# --- END HOTFIX ---
'''

# insert helper near top-level imports (once)
if "def _make_trust_ledger(" not in src:
    # place right after imports block
    lines = src.splitlines(True)
    # find first blank line after imports to insert helper
    insert_at = 0
    for i, L in enumerate(lines[:120]):  # only scan early header
        if L.strip() == "" and i > 0:
            insert_at = i
            break
    lines.insert(insert_at, HELPER + "\n")
    src = "".join(lines)

# replace original TrustLedger factory line with call to helper
src = src.replace(
    'lambda reg: TrustLedger(persistence_path=str(config.GRACE_DATA_DIR / "trust_ledger.jsonl"))',
    "lambda reg: _make_trust_ledger(config)"
)

Path("grace/launcher.py").write_text(src, encoding="utf-8")
print("[launcher] Patched adaptive TrustLedger factory into grace/launcher.py")
PY
else
  echo "[launcher] Adaptive TrustLedger factory already present"
fi

echo "[run] Re-running E2E tests..."
python3 run_full_e2e_test.py || echo "[run] ⚠️ Non-zero exit; see logs above."

echo "------------------------------------------------------------"
echo " GRACE | Hotfix complete"
echo "------------------------------------------------------------"
