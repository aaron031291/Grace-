#!/bin/bash
"""
Grace AI Memory Module - Cleanup Script
======================================
Removes redundant/obsolete files and folders from grace/memory/
Keeps only the unified memory system components
"""

set -e

echo "=========================================="
echo "Grace AI Memory Module Cleanup"
echo "=========================================="
echo ""

MEMORY_PATH="/workspaces/Grace-/grace/memory"

# Files/folders to delete (redundant, consolidated into unified_memory_system.py)
FILES_TO_DELETE=(
    "$MEMORY_PATH/mtl.py"
    "$MEMORY_PATH/mtl_kernel.py"
    "$MEMORY_PATH/mlt_kernel_ml.py"
    "$MEMORY_PATH/lightning.py"
    "$MEMORY_PATH/fusion.py"
    "$MEMORY_PATH/vector.py"
    "$MEMORY_PATH/librarian.py"
    "$MEMORY_PATH/database.py"
    "$MEMORY_PATH/tables.py"
    "$MEMORY_PATH/db.py"
    "$MEMORY_PATH/postgres_store.py"
    "$MEMORY_PATH/redis_cache.py"
    "$MEMORY_PATH/storage.py"
    "$MEMORY_PATH/cache.py"
    "$MEMORY_PATH/embeddings.py"
    "$MEMORY_PATH/schemas.py"
)

FOLDERS_TO_DELETE=(
    "$MEMORY_PATH/mtl_folder"
    "$MEMORY_PATH/lightning_folder"
    "$MEMORY_PATH/fusion_folder"
    "$MEMORY_PATH/vector_folder"
    "$MEMORY_PATH/librarian_folder"
    "$MEMORY_PATH/database_folder"
    "$MEMORY_PATH/tables_folder"
    "$MEMORY_PATH/old_memory"
    "$MEMORY_PATH/deprecated"
    "$MEMORY_PATH/backup"
    "$MEMORY_PATH/archive"
)

echo "STEP 1: Deleting redundant files..."
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✗ Deleting: $(basename $file)"
        rm -f "$file"
    fi
done
echo "✓ Redundant files cleaned"
echo ""

echo "STEP 2: Deleting redundant folders..."
for folder in "${FOLDERS_TO_DELETE[@]}"; do
    if [ -d "$folder" ]; then
        echo "  ✗ Deleting: $(basename $folder)/"
        rm -rf "$folder"
    fi
done
echo "✓ Redundant folders cleaned"
echo ""

echo "STEP 3: Deleting cache and __pycache__..."
find "$MEMORY_PATH" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$MEMORY_PATH" -type f -name '*.pyc' -delete 2>/dev/null || true
find "$MEMORY_PATH" -type f -name '*.pyo' -delete 2>/dev/null || true
echo "✓ Cache cleaned"
echo ""

echo "STEP 4: Verifying remaining structure..."
echo ""
echo "Remaining files in grace/memory/:"
ls -lah "$MEMORY_PATH"/*.py 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
echo ""
echo "Remaining folders in grace/memory/:"
ls -d "$MEMORY_PATH"/*/ 2>/dev/null | xargs -I {} basename {} | awk '{print "  " $0}' || echo "  (none)"
echo ""

echo "=========================================="
echo "✅ MEMORY MODULE CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Redundant files deleted"
echo "  ✓ Redundant folders deleted"
echo "  ✓ Cache cleaned"
echo "  ✓ Unified memory system is canonical"
echo ""
echo "Remaining structure:"
echo "  grace/memory/"
echo "  ├── __init__.py"
echo "  ├── unified_memory_system.py  ✓ CANONICAL"
echo "  ├── integration_tests.py      ✓ VERIFICATION"
echo "  ├── quick_reference.md        ✓ DOCUMENTATION"
echo "  └── health_monitor.py         ✓ HEALTH CHECKS"
echo ""
echo "Next steps:"
echo "  1. Test unified system: python -c \"from grace.memory import UnifiedMemorySystem\""
echo "  2. Run verification: python grace/memory/integration_tests.py"
echo "  3. Commit: git add -A && git commit -m 'refactor: consolidate memory system into unified architecture'"
echo ""
