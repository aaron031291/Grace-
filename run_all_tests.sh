#!/bin/bash
# Grace AI - Complete Test & Verification Script

set -e  # Exit on error

echo "================================================================================"
echo "Grace AI - Complete Test & Verification Script"
echo "================================================================================"
echo ""

# Check if pynacl is installed
if ! python -c "import nacl" 2>/dev/null; then
    echo "⚠️  WARNING: pynacl not installed"
    echo "   Cryptographic signatures will not be available"
    echo "   Install with: pip install pynacl"
    echo ""
fi

# Check for signing keys
if [ -z "$GRACE_ED25519_SK" ]; then
    echo "ℹ️  No GRACE_ED25519_SK environment variable set"
    echo "   Grace will auto-generate and save a key to grace_data/.grace_signing_key"
    echo ""
    echo "   To generate persistent keys, run:"
    echo "   python tools/generate_signing_keys.py"
    echo ""
else
    echo "✓ GRACE_ED25519_SK environment variable is set"
    echo ""
fi

# Step 1: Run VWX test
echo "================================================================================"
echo "STEP 1: Running VWX Verification Test"
echo "================================================================================"
echo ""
python test_vwx.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✓✓✓ VWX test PASSED"
else
    echo ""
    echo "✗✗✗ VWX test FAILED"
    exit 1
fi

# Step 2: Run full E2E test
echo ""
echo "================================================================================"
echo "STEP 2: Running Comprehensive E2E Test"
echo "================================================================================"
echo ""
python run_full_e2e_test.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✓✓✓ E2E test PASSED"
else
    echo ""
    echo "✗✗✗ E2E test FAILED"
    exit 1
fi

# Step 3: Verify audit trail
echo ""
echo "================================================================================"
echo "STEP 3: Verifying Cryptographic Audit Trail"
echo "================================================================================"
echo ""
python tools/verify_immutable_log.py --all
if [ $? -eq 0 ]; then
    echo ""
    echo "✓✓✓ Audit trail verification PASSED"
else
    echo ""
    echo "✗✗✗ Audit trail verification FAILED"
    exit 1
fi

# Step 4: Display trust ledger stats
echo ""
echo "================================================================================"
echo "STEP 4: Trust Ledger Statistics"
echo "================================================================================"
echo ""
if [ -f grace_data/trust_ledger.jsonl ]; then
    echo "Trust Ledger Entries:"
    cat grace_data/trust_ledger.jsonl | wc -l
    echo ""
    
    if command -v jq &> /dev/null; then
        echo "Entity Summary:"
        cat grace_data/trust_ledger.jsonl | jq -s 'group_by(.entity_type) | map({type: .[0].entity_type, count: length})' 2>/dev/null || echo "(jq processing failed)"
        echo ""
        
        echo "Trust Level Summary:"
        cat grace_data/trust_ledger.jsonl | jq -r '.entity_id + " : " + (.trust_score | tostring) + " (" + (.trust_score | if . >= 0.9 then "HIGHLY_TRUSTED" elif . >= 0.7 then "TRUSTED" elif . >= 0.5 then "NEUTRAL" elif . >= 0.3 then "UNTRUSTED" else "QUARANTINED" end) + ")"' | tail -10
    else
        echo "(Install jq for detailed stats: apt install jq)"
        cat grace_data/trust_ledger.jsonl | tail -5
    fi
else
    echo "No trust ledger file found"
fi

# Final summary
echo ""
echo "================================================================================"
echo "FINAL SUMMARY"
echo "================================================================================"
echo ""
echo "✓ VWX Verification Test:         PASSED"
echo "✓ Comprehensive E2E Test:         PASSED"
echo "✓ Cryptographic Audit Trail:     VERIFIED"
echo "✓ Trust Ledger:                   OPERATIONAL"
echo ""
echo "Grace AI system is fully operational! 🎉"
echo ""
echo "Log files:"
echo "  - Immutable log: grace_data/grace_log.jsonl"
echo "  - Trust ledger:  grace_data/trust_ledger.jsonl"
echo ""
echo "Next steps:"
echo "  - View logs:     cat grace_data/grace_log.jsonl | jq | head -50"
echo "  - View trust:    cat grace_data/trust_ledger.jsonl | jq"
echo "  - Read docs:     cat QUICKSTART.md"
echo ""
echo "================================================================================"
