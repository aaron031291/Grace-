#!/usr/bin/env bash
set -euo pipefail

LAUNCHER="grace/launcher.py"

echo "---- Inspecting TrustLedger signature ----"
python3 - <<'PY'
import inspect, importlib, sys
tl = importlib.import_module("grace.core.trust_ledger").TrustLedger
sig = inspect.signature(tl.__init__)
print("TrustLedger.__init__", sig)
PY

echo "---- Patching launcher with adaptive factory ----"
python3 - <<'PY'
from pathlib import Path
import re

p = Path("grace/launcher.py")
src = p.read_text(encoding="utf-8")

HELPER = r'''
# --- BEGIN: Adaptive TrustLedger factory (auto-inserted) ---
def _make_trust_ledger(cfg):
    from inspect import signature
    from grace.core.trust_ledger import TrustLedger
    ledger_path = str(cfg.TRUST_LEDGER_PATH)
    try:
        params = signature(TrustLedger.__init__).parameters
        if "persistence_path" in params:  return TrustLedger(persistence_path=ledger_path)
        if "path" in params:              return TrustLedger(path=ledger_path)
        if "storage_path" in params:      return TrustLedger(storage_path=ledger_path)
        if "file_path" in params:         return TrustLedger(file_path=ledger_path)
        if "filename" in params:          return TrustLedger(filename=ledger_path)
        if "filepath" in params:          return TrustLedger(filepath=ledger_path)
        # Positional fallback
        return TrustLedger(ledger_path)
    except Exception:
        return TrustLedger(ledger_path)
# --- END: Adaptive TrustLedger factory ---
'''

if "_make_trust_ledger(" not in src:
    # insert helper after first blank line following imports
    lines = src.splitlines(True)
    insert_at = 0
    # find a good insertion point near top
    for i, L in enumerate(lines[:200]):
        if L.strip() == "":
            insert_at = i+1
            break
    lines.insert(insert_at, HELPER + "\n")
    src = "".join(lines)

# Replace any lambda that directly calls TrustLedger(...) with our helper.
# Use a regex so spacing/quotes don't matter.
pattern = re.compile(r"lambda\s+reg\s*:\s*TrustLedger\s*\([^)]*\)")
repl    = r"lambda reg: _make_trust_ledger(config)"
new_src = re.sub(pattern, repl, src)

# If nothing matched (edge case), also try a broader replacement in the trust_ledger factory line.
if src == new_src:
    new_src = src.replace("TrustLedger(", "_make_trust_ledger(config)  # ")

Path("grace/launcher.py").write_text(new_src, encoding="utf-8")
print("[launcher] Patched to use _make_trust_ledger(config)")
PY

echo "---- Re-running tests ----"
python3 run_full_e2e_test.py || echo "⚠️ Tests returned non-zero; check the Trust Ledger section above."
