#!/bin/bash
#
# Grace AI - Push All Updates to GitHub
# =====================================
# This script stages all changes, commits them with a comprehensive
# message, and pushes to the remote repository.
#

set -e

echo "=========================================="
echo "Grace AI - Pushing Updates to GitHub"
echo "=========================================="
echo ""

# --- Configuration ---
# The branch to push to. Change if you are not on 'main' or 'master'.
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
REMOTE_NAME="origin"

# --- Commit Message ---
# A detailed commit message summarizing all recent changes.
COMMIT_TITLE="feat: Implement real kernels, unified launcher, and fix critical errors"
COMMIT_BODY=$(cat <<-END
This major update includes:

- **Real Kernels:** Replaced placeholder kernel launchers with real, async-ready kernel implementations (Learning, Orchestration, Resilience, MultiOS).
- **Unified Launcher:** Introduced a unified launcher (grace/launcher.py) to manage the entire system lifecycle with CLI support.
- **Core Infrastructure:** Implemented a ServiceRegistry for dependency injection and a BaseKernel for consistent kernel structure.
- **Critical Fixes:** Fixed a cascading import error ('ImmutableLogs' vs 'ImmutableLogger') that blocked the entire system.
- **Module Consolidation:** Merged 'avn' and 'immune_system' modules into a unified 'resilience' module.
- **Diagnostics & Cleanup:** Added comprehensive diagnostic, cleanup, merge, and audit scripts.

The system is now operational and can be started via the new launcher. All major structural issues identified have been addressed.
END
)

# Step 1: Show current status
echo "STEP 1: Reviewing Git Status"
echo "------------------------------"
git status
echo ""
echo "The status above shows all changes that will be committed."
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Step 2: Add all changes to the staging area
echo "STEP 2: Staging all changes..."
git add -A
echo "✓ All changes have been staged."
echo ""

# Step 3: Commit the changes
echo "STEP 3: Committing changes..."
git commit -m "$COMMIT_TITLE" -m "$COMMIT_BODY"
echo "✓ Changes committed successfully."
echo ""

# Step 4: Push to GitHub
echo "STEP 4: Pushing to GitHub remote '$REMOTE_NAME' on branch '$BRANCH_NAME'..."
git push $REMOTE_NAME $BRANCH_NAME
echo ""
echo "=========================================="
echo "✅ PUSH COMPLETE"
echo "=========================================="
echo "All local changes have been successfully pushed to the GitHub repository."
echo ""
