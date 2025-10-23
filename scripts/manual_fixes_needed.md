# Manual Pylance Fixes Needed

## Common Issues to Fix

### 1. Missing Type Hints in Function Signatures

**Pattern**: Functions without return type hints
**Fix**: Add `-> ReturnType` or `-> None`

Example:
```python
# Before
def my_function(param: str):
    return param.upper()

# After  
def my_function(param: str) -> str:
    return param.upper()
```

### 2. Optional Parameters Without Defaults

**Pattern**: `param: Optional[Type]` without `= None`
**Fix**: Add `= None`

Example:
```python
# Before
def func(param: Optional[str]):
    pass

# After
def func(param: Optional[str] = None):
    pass
```

### 3. Dataclass Mutable Defaults

**Pattern**: `field: List[T] = []` or `field: Dict[K, V] = {}`
**Fix**: Use `field(default_factory=...)`

Example:
```python
# Before
@dataclass
class MyClass:
    items: List[str] = []

# After
from dataclasses import dataclass, field

@dataclass
class MyClass:
    items: List[str] = field(default_factory=list)
```

### 4. Missing Any Import

**Pattern**: Using `Any` without importing
**Fix**: Add to typing imports

Example:
```python
# Before
from typing import Dict, List

# After
from typing import Dict, List, Any
```

### 5. Abstract Method Return Types

**Pattern**: Abstract methods without proper signatures
**Fix**: Match base class signature exactly

Example:
```python
# Before
@abstractmethod
async def send(self, destination: str, message: dict):
    pass

# After
@abstractmethod
async def send(self, destination: str, message: Dict[str, Any]) -> bool:
    pass
```

## Files Requiring Attention

Run this to find files with issues:
```bash
find grace -name "*.py" | xargs -I {} python -m pylint {} 2>&1 | grep -E "(missing-return|no-member)"
```

## Quick Fixes

### Run Automated Fixer
```bash
python scripts/auto_fix_pylance.py
```

### Format Code
```bash
black grace/
```

### Validate
```bash
python scripts/master_validation.py
```
