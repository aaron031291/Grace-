#!/bin/bash
#
# Grace AI - Security & Sanity Check Script
# =========================================
# This script performs two main functions:
# 1. Scans the repository for hardcoded private keys.
# 2. Runs a lightweight verification to ensure TriggerMesh is wired correctly.
#

set -e

# --- ANSI Color Codes ---
C_RED='\033[91m'
C_GREEN='\033[92m'
C_YELLOW='\033[93m'
C_BLUE='\033[94m'
C_MAGENTA='\033[95m'
C_END='\033[0m'

echo -e "${C_MAGENTA}=========================================${C_END}"
echo -e "${C_MAGENTA}  Grace AI - Security & Sanity Check   ${C_END}"
echo -e "${C_MAGENTA}=========================================${C_END}\n"

# --- STEP 1: Scan for Hardcoded Secrets ---
echo -e "${C_BLUE}--- STEP 1: Scanning for hardcoded private keys ---${C_END}"
KEY_PATTERN="PRIVATE KEY|BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY|BEGIN DSA PRIVATE KEY|BEGIN EC PRIVATE KEY"
# Exclude this script itself from the search
EXCLUSIONS="--exclude-dir=.git --exclude=$(basename "$0")"

# The `grep` command will exit with 1 if no lines are selected, so we handle it.
if grep -R --line-number -E "$KEY_PATTERN" . $EXCLUSIONS; then
    echo -e "\n${C_RED}✗ DANGER: Potential private key(s) found in the files listed above.${C_END}"
    echo -e "${C_YELLOW}Please remove them immediately, add them to .gitignore, and rotate the keys!${C_END}"
    exit 1
else
    echo -e "${C_GREEN}✓ PASS: No hardcoded private keys found.${C_END}"
fi
echo -e "\n"


# --- STEP 2: Verify TriggerMesh Wiring ---
echo -e "${C_BLUE}--- STEP 2: Verifying TriggerMesh service wiring ---${C_END}"
# Add workspace to Python path to ensure imports work
export PYTHONPATH="/workspaces/Grace-:$PYTHONPATH"

VERIFICATION_CMD="
import asyncio
from grace.launcher import GraceLauncher

class MockArgs:
    debug = False
    kernel = None
    dry_run = True

async def verify():
    try:
        launcher = GraceLauncher(MockArgs())
        await launcher.initialize()
        trigger_mesh_instance = launcher.registry.get('trigger_mesh')
        
        if trigger_mesh_instance:
            print(f'${C_GREEN}✓ PASS: ServiceRegistry successfully returned a TriggerMesh instance.${C_END}')
            print(f'Instance: {trigger_mesh_instance}')
        else:
            print(f'${C_RED}✗ FAIL: ServiceRegistry returned None for "trigger_mesh".${C_END}')
            exit(1)
            
        cortex = next((k for k in launcher.kernels if k.name == 'cognitive_cortex'), None)
        if cortex and cortex.get_service('trigger_mesh'):
             print(f'${C_GREEN}✓ PASS: CognitiveCortex has a wired TriggerMesh instance.${C_END}')
        else:
             print(f'${C_RED}✗ FAIL: CognitiveCortex is missing the TriggerMesh service.${C_END}')
             exit(1)

    except Exception as e:
        print(f'${C_RED}✗ FAIL: Verification script failed with an exception: {e}${C_END}')
        exit(1)

asyncio.run(verify())
"

# Execute the Python verification command
python -c "$VERIFICATION_CMD"

echo -e "\n${C_GREEN}=========================================${C_END}"
echo -e "${C_GREEN}  ✅ All Checks Passed Successfully!  ${C_END}"
echo -e "${C_GREEN}=========================================${C_END}\n"
