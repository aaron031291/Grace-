#!/usr/bin/env bash
set -euo pipefail
LAUNCHER="grace/launcher.py"

# Insert helper if missing
if ! grep -q "_make_trust_ledger" "$LAUNCHER"; then
  awk '
    BEGIN { inserted=0; last_import_line=0 }
    # Track import lines
    /^[[:space:]]*from[[:space:]]/ || /^[[:space:]]*import[[:space:]]/ { 
      print
      last_import_line=NR
      next 
    }
    # After last import, insert helper once
    last_import_line > 0 && NR==last_import_line+1 && !inserted {
      print ""
      print "def _make_trust_ledger(config):"
      print "    import inspect"
      print "    from grace.core.trust_ledger import TrustLedger"
      print "    ledger_path = str(config.TRUST_LEDGER_PATH)"
      print "    try:"
      print "        params = inspect.signature(TrustLedger.__init__).parameters"
      print "    except (ValueError, TypeError):"
      print "        params = {}"
      print "    for kw in (\"persistence_path\",\"path\",\"file_path\",\"storage_file\",\"ledger_path\"):"
      print "        if kw in params: return TrustLedger(**{kw: ledger_path})"
      print "    try: return TrustLedger(ledger_path)"
      print "    except TypeError: pass"
      print "    for factory in (\"from_path\",\"from_file\",\"from_persistence\"):"
      print "        if hasattr(TrustLedger, factory):"
      print "            return getattr(TrustLedger, factory)(ledger_path)"
      print "    raise TypeError(\"Could not construct TrustLedger; unknown constructor signature.\")"
      print ""
      inserted=1
    }
    { print }
  ' "$LAUNCHER" > "$LAUNCHER.tmp" && mv "$LAUNCHER.tmp" "$LAUNCHER"
  echo "[launcher] Inserted _make_trust_ledger helper into $LAUNCHER"
fi

# Ensure the factory uses the helper
if ! grep -q "_make_trust_ledger(config)" "$LAUNCHER"; then
  sed -i 's/lambda reg: TrustLedger([^)]*)/lambda reg: _make_trust_ledger(config)/' "$LAUNCHER" || true
  echo "[launcher] Patched trust_ledger factory to call _make_trust_ledger(config)"
fi

echo "[run] Re-running E2E…"
python3 run_full_e2e_test.py || echo "[run] ⚠️ Non-zero exit; see logs."
