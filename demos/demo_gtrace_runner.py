"""Run gtrace demo/test functions without being collected by pytest.

This file replaces the top-level `test_gtrace.py` so pytest doesn't collect
it as a duplicate test module. It provides a small CLI to run the demo
functions manually.
"""
import asyncio
import sys
from pathlib import Path

# Ensure package root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from test_gtrace import run_all_tests


def main():
    print("Running demo gtrace runner (not collected by pytest)")
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("Cancelled")


if __name__ == "__main__":
    main()
