#!/bin/bash

# Test script to validate the commit parameter handling fix
# Tests that git-workflow.sh correctly handles commits with and without scope

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

echo -e "${YELLOW}Testing Git Workflow Commit Fix${NC}"
echo "=================================="
echo ""

# Setup test environment
TEST_FILE="test-commit-temp.txt"
INITIAL_COMMIT=$(git rev-parse HEAD)

cleanup() {
    echo ""
    echo "Cleaning up test commits..."
    git reset --hard "$INITIAL_COMMIT" >/dev/null 2>&1
    rm -f "$TEST_FILE"
}

trap cleanup EXIT

# Test 1: Commit without scope
echo "Test 1: Commit WITHOUT scope"
echo "Command: ./scripts/git-workflow.sh commit feat 'add feature'"
echo "test 1" > "$TEST_FILE"
OUTPUT=$(./scripts/git-workflow.sh commit feat "add feature" 2>&1)
COMMIT_MSG=$(git log -1 --pretty=%B)

if [ "$COMMIT_MSG" = "feat: add feature" ]; then
    echo -e "${GREEN}✅ PASS${NC}: Created 'feat: add feature'"
else
    echo -e "${RED}❌ FAIL${NC}: Expected 'feat: add feature', got '$COMMIT_MSG'"
    exit 1
fi
echo ""

# Test 2: Commit with scope
echo "Test 2: Commit WITH scope"
echo "Command: ./scripts/git-workflow.sh commit fix scripts 'fix bug'"
echo "test 2" >> "$TEST_FILE"
OUTPUT=$(./scripts/git-workflow.sh commit fix scripts "fix bug" 2>&1)
COMMIT_MSG=$(git log -1 --pretty=%B)

if [ "$COMMIT_MSG" = "fix(scripts): fix bug" ]; then
    echo -e "${GREEN}✅ PASS${NC}: Created 'fix(scripts): fix bug'"
else
    echo -e "${RED}❌ FAIL${NC}: Expected 'fix(scripts): fix bug', got '$COMMIT_MSG'"
    exit 1
fi
echo ""

# Test 3: Missing message should fail
echo "Test 3: Missing message (should fail)"
echo "Command: ./scripts/git-workflow.sh commit test"
OUTPUT=$(./scripts/git-workflow.sh commit test 2>&1 || true)

if echo "$OUTPUT" | grep -q "Type and message are required"; then
    echo -e "${GREEN}✅ PASS${NC}: Correctly validates missing message"
else
    echo -e "${RED}❌ FAIL${NC}: Should require message"
    exit 1
fi
echo ""

# Test 4: Verify conventional commit format
echo "Test 4: Verify conventional commit formats"
COMMITS=$(git log -2 --pretty=%B)
echo "Recent commits:"
git log -2 --oneline | sed 's/^/  /'

if echo "$COMMITS" | grep -qE '^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+'; then
    echo -e "${GREEN}✅ PASS${NC}: Commits follow conventional format"
else
    echo -e "${RED}❌ FAIL${NC}: Commits don't follow conventional format"
    exit 1
fi
echo ""

echo "=================================="
echo -e "${GREEN}✅ All tests passed!${NC}"
echo "=================================="
