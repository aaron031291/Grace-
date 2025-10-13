Title: Timezone safety and background-task cleanup

Summary:
This branch (copilot/fix-timezone-tests) centralizes timezone handling and fixes several issues causing warnings and potential bugs in tests.

What changed:
- Added `grace.utils.time` helpers (now_utc, iso_now_utc, to_utc) — centralized timezone-aware helpers.
- Replaced all uses of naive `datetime.utcnow()` across the codebase with `now_utc()` or `iso_now_utc()`.
- Converted Pydantic model default factories to use `now_utc` where appropriate.
- Added an autouse pytest fixture to cancel leftover asyncio tasks between tests.
- Introduced explicit start()/stop lifecycle approaches for components that spawn background tasks (several kernels and services).
- Fixed multiple indentation/compile regressions found during the refactor.

Files changed (representative):
- grace/utils/time.py (new)
- pytest fixture (conftest.py)
- grace/interface/models.py
- grace/contracts/dto_common.py
- grace/mtl_kernel/schemas.py
- grace/intelligence_kernel/specialists/specialist.py
- grace/interface/bridges/gov_bridge.py
- grace/learning_kernel/kernel.py
- grace/ingress_kernel/adapters/base.py
- demo_ingress_kernel.py
- many other small per-file fixes discovered while testing

Tests:
- Ran pytest repeatedly during refactor: final run shows 11 passed.

Notes and recommendations:
- The refactor reduces timezone-related warnings and ensures timestamps are timezone-aware everywhere.
- Upstream merges with `main` should be handled carefully — previous attempt generated many add/add conflicts. Recommend a manual per-file merge after review.
- I recommend opening this branch as a PR for code review. After approval, perform a controlled merge into `main` resolving conflicts interactively.

Next steps (optional actions I can take):
- Draft a full PR on GitHub (title + body) and open it.
- Run a linting pass (flake8/pyright) and fix minor style issues.
- Help with a manual interactive merge against `origin/main` and resolve conflicts.

If you'd like, I can open a PR now with this description, or prepare a smaller change set if you prefer a more incremental merge strategy.

PR checklist (for reviewers)
 - [ ] Verify timezone helper API and edge cases (now_utc / iso_now_utc / to_utc)
 - [ ] Review model default factory changes for Pydantic compatibility
 - [ ] Confirm background-task lifecycle changes don't break long-running services
 - [ ] Sanity-check demo and tests for date/time formatting expectations
 - [ ] Approve and merge, or request follow-up changes

Recommended next actions (I can do any):
 - Run and fix high-value linter errors (unused imports, missing newlines, trivial style fixes)
 - Run a selective static-type check (pyright) and fix critical type issues
 - Help resolve an interactive merge with `main` if reviewers want the full original branch merged
 - Keep this PR focused (merge as-is) and address the larger lint cleanup in a follow-up PR

PR link: https://github.com/aaron031291/Grace-/pull/82

# commit marker for PR update

# commit marker