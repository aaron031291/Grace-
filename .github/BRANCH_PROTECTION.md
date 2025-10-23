# Branch Protection Setup Guide

## Required Configuration

To enforce green CI before merging to `main`, configure these settings in GitHub:

### Step 1: Access Branch Protection Settings

1. Go to **Repository Settings** → **Branches**
2. Click **Add branch protection rule**
3. Enter branch name pattern: `main`

### Step 2: Required Status Checks

Enable the following:

- ✅ **Require status checks to pass before merging**
- ✅ **Require branches to be up to date before merging**

Select these required checks:

1. **Type Safety & Linting**
2. **Unit Tests**
3. **EventBus Feature Tests**
4. **TriggerMesh Tests**
5. **Build Check**
6. **CI Summary**

### Step 3: Pull Request Requirements

Enable:

- ✅ **Require pull request reviews before merging**
  - Minimum: 1 approval
- ✅ **Dismiss stale pull request approvals when new commits are pushed**
- ✅ **Require review from Code Owners** (optional)

### Step 4: Additional Protections

Enable:

- ✅ **Require conversation resolution before merging**
- ✅ **Include administrators** (enforce rules for admins)
- ✅ **Allow force pushes**: **DISABLED**
- ✅ **Allow deletions**: **DISABLED**

### Step 5: Status Check Badges

Add to `README.md`:

```markdown
[![CI](https://github.com/yourorg/grace/workflows/CI/badge.svg)](https://github.com/yourorg/grace/actions/workflows/ci.yml)
[![Quick Check](https://github.com/yourorg/grace/workflows/Quick%20Check/badge.svg)](https://github.com/yourorg/grace/actions/workflows/quick-check.yml)
```

## What Gets Checked

### Type Safety & Linting
- Black code formatting
- Ruff linting
- Type safety audit (no dict publishes)

### Unit Tests
- All unit tests in `tests/`
- With Postgres and Redis services
- 30-second timeout per test

### EventBus Feature Tests
- emit/subscribe/wait_for
- TTL expiry
- Idempotency
- Dead Letter Queue
- Backpressure

### TriggerMesh Tests
- YAML config loading
- Route matching
- Filter application
- Subscription binding

### Consensus Tests
- Governance → MLDL request/response
- Multi-specialist consensus
- Timeout handling

### Integration Tests
- End-to-end flows
- Memory integration
- Kernel coordination

### Build Check
- Package builds successfully
- `twine check` passes

### Security Scan
- Bandit (code security)
- Safety (dependency vulnerabilities)
- Results are informational (non-blocking)

### Coverage
- Code coverage collection
- Upload to Codecov
- Results are informational (non-blocking)

## Bypassing Branch Protection

Branch protection can only be bypassed if:

1. **Include administrators** is disabled, AND
2. You are a repository administrator

Otherwise, all checks must pass.

## Troubleshooting

### Check fails with "Required status check missing"

Ensure the check name in branch protection **exactly matches** the job name in workflow:

```yaml
jobs:
  unit-tests:  # ← This becomes "Unit Tests" in GitHub
    name: Unit Tests  # ← Use this exact name
```

### Pull request cannot merge

Common causes:

1. **Not up to date** - Click "Update branch"
2. **Checks still running** - Wait for completion
3. **Check failed** - Fix issues and push new commit
4. **Missing approval** - Request review

### False positive security scan

Security scans are `continue-on-error: true` so they won't block merges.
Review findings in the Actions artifacts.

## Local Pre-commit Checks

Run these locally before pushing:

```bash
# Full test suite
bash scripts/run_all_event_tests.sh

# Quick check
python -m pytest tests/test_event_bus_features.py -v

# Type safety
python scripts/audit_type_safety.py

# Formatting
black grace/ tests/ scripts/
```

## Workflow Triggers

- **CI**: Runs on push to `main`/`develop` and all PRs
- **Quick Check**: Runs on PR open/sync
- **Branch Protection Info**: Manual trigger only

## Artifacts

CI uploads these artifacts:

- `test-results` - Coverage data
- `security-reports` - Bandit JSON
- `dist-packages` - Built wheels/sdist
- `coverage-report` - HTML coverage report
