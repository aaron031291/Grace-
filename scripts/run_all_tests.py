"""
Automated test runner
Runs all tests and generates report
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """Run all tests with pytest"""
    print("=" * 80)
    print("Grace System - Automated Test Suite")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run pytest
    result = subprocess.run(
        [
            "python", "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--color=yes",
            "--durations=10",
            "-x"  # Stop on first failure
        ],
        capture_output=False,
        text=True
    )
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result.returncode}")
    
    return result.returncode


def run_with_coverage():
    """Run tests with coverage report"""
    print("=" * 80)
    print("Grace System - Test Coverage Report")
    print("=" * 80)
    
    result = subprocess.run(
        [
            "python", "-m", "pytest",
            "tests/",
            "--cov=grace",
            "--cov-report=html",
            "--cov-report=term",
            "-v"
        ],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ Coverage report generated in htmlcov/")
    
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Grace system tests")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    args = parser.parse_args()
    
    if args.coverage:
        sys.exit(run_with_coverage())
    else:
        sys.exit(run_tests())
