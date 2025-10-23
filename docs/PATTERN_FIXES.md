# Grace Pattern-Based Error Fixes

## Overview

This document describes the **pattern-based fixing strategy** for resolving the ~3,000 errors in the Grace codebase. Instead of fixing errors one-by-one, we identify cross-cutting patterns and apply fixes across the entire codebase at once.

## 🎯 The 10 Cross-Cutting Patterns

### Pattern 1: Implicit Optional (Est. 200-300 fixes)
**Problem:** Parameters with `None` defaults lack `Optional[T]` annotation
```python
# ❌ Before
def process(data: str = None):
    pass

# ✅ After  
def process(data: Optional[str] = None):
    pass
```

**Fix:** Add `from __future__ import annotations` and `Optional[T]` everywhere

---

### Pattern 2: asyncio.gather with non-awaitables (Est. 50-100 fixes)
**Problem:** Passing `None` or non-awaitable objects to `asyncio.gather()`
```python
# ❌ Before
await asyncio.gather(*tasks)  # tasks may contain None

# ✅ After
tasks = [t for t in tasks if inspect.isawaitable(t)]
if tasks:
    await asyncio.gather(*tasks)
```

**Fix:** Filter awaitables before gathering

---

### Pattern 3-4: Type Arithmetic (Est. 150-250 fixes)
**Problem:** Unannotated variables cause arithmetic errors
```python
# ❌ Before
result = counter + 1  # counter is 'object' type

# ✅ After
from grace.utils.type_converters import to_int
result = to_int(counter) + 1
```

**Fix:** Created utility functions:
- `ensure_datetime(x) -> datetime | None`
- `to_int(x, default=0) -> int`
- `to_float(x, default=0.0) -> float`

---

### Pattern 5: Container Operations (Est. 75-125 fixes)
**Problem:** Calling `.append()` on dicts or wrong element types
```python
# ❌ Before
results = {}
results.append(item)  # dict has no append

# ✅ After
results = []
results.append(item)
```

**Fix:** Initialize as lists, not dicts; add type guards

---

### Pattern 6: Return Type Mismatches (Est. 100-150 fixes)
**Problem:** Returning `Any` where concrete type expected
```python
# ❌ Before
def get_status() -> bool:
    return result  # result is Any

# ✅ After
def get_status() -> bool:
    return bool(result)
```

**Fix:** Add explicit type conversions in return statements

---

### Pattern 7: Missing Imports (Est. 50-75 fixes)
**Problem:** Using `logger`, `datetime`, `Request` without importing
```python
# ❌ Before
logger.info("Starting...")  # NameError

# ✅ After
import logging
logger = logging.getLogger(__name__)
logger.info("Starting...")
```

**Fix:** Add missing imports; install type stubs (`types-PyYAML`)

---

### Pattern 8: MCP Interface Drift (Est. 80-120 fixes)
**Problem:** Handlers pass kwargs that base signature doesn't accept
```python
# ❌ Before (base_mcp.py)
def search(self, query: str):
    pass

# Caller
result = await handler.search(query, collection="docs")  # Error

# ✅ After
def search(self, query: str, **kwargs):
    pass
```

**Fix:** Add `**kwargs` to base signatures; remove `await` on sync functions

---

### Pattern 9: Event Bus Callbacks (Est. 40-60 fixes)
**Problem:** Callback lists typed as `List[dict]` but storing callables
```python
# ❌ Before
self._callbacks: List[dict] = []

# ✅ After
from typing import Protocol

class Subscriber(Protocol):
    def __call__(self, message: GraceMessageEnvelope) -> None: ...

self._callbacks: List[Subscriber] = []
```

**Fix:** Use `Protocol` for callback typing

---

### Pattern 10: Meta-learner Numerics (Est. 60-90 fixes)
**Problem:** Treating Python lists as numpy arrays
```python
# ❌ Before
scores: List[float] = [...]
n = scores.size  # list has no .size

# ✅ After
import numpy as np
scores_array = np.asarray(scores, dtype=float)
scores_array = np.nan_to_num(scores_array)  # handle None/NaN
n = scores_array.size
```

**Fix:** Convert to numpy arrays; sanitize None/NaN

---

### Bonus: API/DB Mismatches (Est. 30-50 fixes)
**Problem:** Wrong attribute/function names
```python
# ❌ Before
@require_admin  # doesn't exist
db = get_async_db()  # wrong name

# ✅ After
@requires_admin
db = get_db()
```

**Fix:** Align names with actual module contents

---

## 🚀 Usage

### 1. Measure Baseline
```bash
cd /workspaces/Grace-
python scripts/measure_fix_impact.py
```

This creates a baseline snapshot of current errors.

### 2. Apply All Pattern Fixes
```bash
chmod +x scripts/apply_pattern_fixes.sh
bash scripts/apply_pattern_fixes.sh
```

This applies all 10 patterns across the codebase.

### 3. Measure Impact
```bash
python scripts/measure_fix_impact.py
```

Shows before/after comparison with percentage reduction.

### 4. Verify
```bash
# Run full analysis
python scripts/forensic_analysis.py

# Or quick mypy check
mypy grace --ignore-missing-imports | head -50
```

---

## 📊 Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Errors | ~3,000 | ~1,100-1,600 | 46-63% |
| Blocking Errors | ~800 | ~200-400 | 50-75% |
| Type Errors | ~1,500 | ~400-700 | 53-73% |
| Import Errors | ~200 | ~50-80 | 60-75% |

**Total estimated fixes: 835-1,395 errors** in a single pass.

---

## 🔧 Files Created

1. **`scripts/pattern_fixes.py`** - Main pattern fixing engine
2. **`scripts/apply_pattern_fixes.sh`** - Execution wrapper
3. **`scripts/measure_fix_impact.py`** - Impact measurement
4. **`grace/utils/type_converters.py`** - Type conversion utilities

---

## 🎯 Next Steps After Pattern Fixes

After applying pattern fixes, remaining errors will be:
1. **Module-specific logic errors** (~400-600)
2. **Complex type inference issues** (~200-300)  
3. **Architecture mismatches** (~150-250)
4. **Edge cases** (~100-200)

These require targeted, file-by-file fixes but are much more manageable than 3,000 errors.

---

## 💡 Key Insights

**Why Pattern-Based Fixing Works:**
1. ✅ **Leverage repetition** - Same mistakes repeated across codebase
2. ✅ **Multiplicative impact** - One fix solves 10-100+ instances
3. ✅ **Faster execution** - Automated vs. manual fixes
4. ✅ **Consistency** - Same solution applied uniformly
5. ✅ **Measurable** - Clear before/after metrics

**Traditional Approach:**
- Fix 25 errors at a time
- ~120 batches needed for 3,000 errors
- High risk of inconsistency

**Pattern Approach:**
- Fix 10 patterns at once
- ~1,000+ errors resolved in single pass
- Consistent solution across codebase

---

## 🔍 Verification Checklist

After running fixes:

- [ ] `python scripts/measure_fix_impact.py` shows significant reduction
- [ ] `mypy grace` output is noticeably shorter
- [ ] `git diff` shows expected pattern changes
- [ ] Core modules (`flow_control.py`, `semantic_bridge.py`) now type-check
- [ ] Test suite runs without import errors: `pytest tests/ -v`
- [ ] No new errors introduced (verify with git diff)

---

## 📚 References

- **Original Error Log:** `Grace errors.docx` - Full forensic analysis
- **Pattern Analysis:** User-provided top 10 patterns
- **Type Utilities:** `grace/utils/type_converters.py`
- **Fix Scripts:** `scripts/pattern_fixes.py`

---

## ⚠️ Important Notes

1. **Backup First:** All fixes modify files in-place
   ```bash
   git add -A
   git commit -m "Pre-pattern-fix snapshot"
   ```

2. **Review Changes:** Check git diff for unexpected modifications
   ```bash
   git diff | less
   ```

3. **Iterative:** Some patterns may need refinement after first pass

4. **Not Perfect:** Automated fixes may miss edge cases; manual review recommended

---

## 🎉 Success Criteria

Pattern fixes are successful if:
1. ✅ Total error count reduced by 40%+
2. ✅ No new import/syntax errors introduced
3. ✅ Core subsystems (flow_control, semantic_bridge, event_bus) type-check
4. ✅ Test suite can import all modules
5. ✅ Remaining errors are module-specific, not systemic

---

**Ready to fix Grace?** Run the baseline measurement and apply pattern fixes! 🚀
