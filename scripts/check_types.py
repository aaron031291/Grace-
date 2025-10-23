"""
Type checking validation using mypy
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run type checking"""
    print("Running type checks with mypy...")
    print("=" * 80)
    
    result = subprocess.run(
        [
            "python", "-m", "mypy",
            "grace",
            "--ignore-missing-imports",
            "--no-strict-optional",
            "--show-error-codes",
            "--pretty"
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print("=" * 80)
    
    if result.returncode == 0:
        print("✅ Type checking passed!")
        return 0
    else:
        print("❌ Type checking found issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
