"""
DELETION MANIFEST - Files to be removed
========================================

PHASE 1: Duplicate Documentation
---------------------------------
X /workspaces/Grace-/ARCHITECTURE_REFACTORED.md
  Reason: Superseded by ARCHITECTURE.md and ARCHITECTURE_VISUAL.txt
  Size: ~10 KB
  Risk: NONE - Documentation only


PHASE 2: Cache & Build Artifacts  
----------------------------------
NOTE: These would be removed via CLI commands, not via tool
  
  rm -rf $(find /workspaces/Grace- -type d -name '__pycache__')
  find /workspaces/Grace- -type f -name '*.pyc' -delete
  find /workspaces/Grace- -type f -name '*.pyo' -delete
  rm -rf $(find /workspaces/Grace- -type d -name '.pytest_cache')
  rm -rf $(find /workspaces/Grace- -type d -name '.egg-info')
  rm -rf /workspaces/Grace-/dist /workspaces/Grace-/build


PHASE 3: Redundant Code (Merged functionality)
-----------------------------------------------
X /workspaces/Grace-/grace/kernels/resilience_kernel.py
  Functionality merged into:
    - grace/immune_system/core.py (ImmuneSystem class)
    - grace/immune_system/avn_healer.py (AVNHealer class)
  Size: ~5 KB
  Risk: LOW - All imports updated, references removed
  Status: ✓ Import references removed from grace/kernels/__init__.py

X /workspaces/Grace-/grace/services/observability.py
  Functionality merged into:
    - grace/core/truth_layer.py (SystemMetrics class)
  Size: ~3 KB
  Risk: LOW - Functionality preserved in SystemMetrics
  Status: ✓ Import references removed from grace/services/__init__.py


SUMMARY
=======
Total files to delete (PHASE 1-3):
  - 1 documentation file
  - 2 redundant code files
  - Cache/artifacts (dozens of small files)

Estimated space freed: ~200-300 MB (including __pycache__)

Safety status:
  ✓ All imports updated
  ✓ No broken references in main.py
  ✓ Functionality preserved
  ✓ Deletions are reversible via git


EXECUTION INSTRUCTIONS
======================

To complete cleanup, run:

# Delete PHASE 1 (Documentation)
rm /workspaces/Grace-/ARCHITECTURE_REFACTORED.md

# Delete PHASE 2 (Cache - requires shell)
find /workspaces/Grace- -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find /workspaces/Grace- -type f -name '*.pyc' -delete
find /workspaces/Grace- -type f -name '*.pyo' -delete
find /workspaces/Grace- -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
find /workspaces/Grace- -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
rm -rf /workspaces/Grace-/dist /workspaces/Grace-/build

# Delete PHASE 3 (Redundant code)
rm /workspaces/Grace-/grace/kernels/resilience_kernel.py
rm /workspaces/Grace-/grace/services/observability.py
"""
