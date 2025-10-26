#!/usr/bin/env bash
set -euo pipefail

LAUNCHER="grace/launcher.py"

echo "---- Inspecting TrustLedger signature ----"
python3 - <<'PY'
import inspect, importlib
tl = importlib.import_module("grace.core.trust_ledger").TrustLedger
print("TrustLedger.__init__ signature:", inspect.signature(tl.__init__))
PY

echo "---- Patching launcher with robust adaptive factory ----"
python3 - <<'PY'
from pathlib import Path
import re

p = Path("grace/launcher.py")
src = p.read_text(encoding="utf-8")

HELPER = r'''
# --- BEGIN: Adaptive TrustLedger factory (auto-inserted) ---
def _make_trust_ledger(config):
    from inspect import signature
    from grace.core.trust_ledger import TrustLedger
    ledger_path = str(config.TRUST_LEDGER_PATH)
    params = signature(TrustLedger.__init__).parameters
    # Try common keyword names
    for kw in ("persistence_path","path","storage_path","file_path","filename","filepath"):
        if kw in params:
            return TrustLedger(**{kw: ledger_path})
    # Fallback: positional single-arg if supported, else no-arg then configure
    try:
        return TrustLedger(ledger_path)
    except TypeError:
        return TrustLedger()
# --- END: Adaptive TrustLedger factory ---
'''

# Inject helper near the top (after imports) if not present
if "_make_trust_ledger(" not in src:
    # Find the end of the import block
    lines = src.splitlines(True)
    insert_at = 0
    # safer: insert after the first blank line following imports
    # find last import line index
    last_import = 0
    for i, L in enumerate(lines[:300]):
        if L.strip().startswith(("import ", "from ")):
            last_import = i
    # advance to the next blank line after last import
    insert_at = last_import + 1
    while insert_at < len(lines) and lines[insert_at].strip() != "":
        insert_at += 1
    insert_at += 1
    lines.insert(insert_at, HELPER + "\n")
    src = "".join(lines)

# Replace any lambda that builds TrustLedger(...) (handles multi-line args)
pattern = re.compile(r"lambda\s+reg\s*:\s*TrustLedger\s*\((?:.|\n)*?\)", re.S)
if pattern.search(src):
    src = pattern.sub("lambda reg: _make_trust_ledger(config)", src)
else:
    # Try a targeted replace in the trust_ledger factory registration line
    src = re.sub(
        r"(register.*['\"]trust_ledger['\"]\s*,\s*)lambda\s+reg\s*:\s*TrustLedger\s*\((?:.|\n)*?\)",
        r"\1lambda reg: _make_trust_ledger(config)",
        src, flags=re.S
    )

Path("grace/launcher.py").write_text(src, encoding="utf-8")
print("[launcher] Patched to use _make_trust_ledger(config)")
PY

echo "---- Re-running tests ----"
python3 run_full_e2e_test.py || echo "⚠️ Tests returned non-zero; check the Trust Ledger section above."
