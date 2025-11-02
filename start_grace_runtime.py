#!/usr/bin/env python3
"""
Grace AI - Unified Runtime Startup
==================================

Simple one-command startup for Grace AI system.
Handles all initialization and runs the complete integrated system.

Usage:
    python start_grace_runtime.py                 # Full system
    python start_grace_runtime.py --production    # Production mode
    python start_grace_runtime.py --api           # API server mode
    python start_grace_runtime.py --autonomous    # Autonomous mode
"""

import sys
import subprocess
from pathlib import Path

# Ensure we're in the correct directory
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))


def main():
    """Start Grace Runtime"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Grace AI - Unified Runtime Startup'
    )
    parser.add_argument('--production', action='store_true', help='Production mode')
    parser.add_argument('--api', action='store_true', help='API server mode')
    parser.add_argument('--autonomous', action='store_true', help='Autonomous mode')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (test config)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.api:
        mode = 'api-server'
    elif args.autonomous:
        mode = 'autonomous'
    elif args.production:
        mode = 'production'
    else:
        mode = 'full-system'
    
    # Build command
    cmd = [sys.executable, '-m', 'grace.launcher', '--mode', mode]
    
    if args.debug:
        cmd.append('--debug')
    
    if args.dry_run:
        cmd.append('--dry-run')
    
    print("=" * 70)
    print("GRACE AI - UNIFIED RUNTIME")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    print()
    
    # Run launcher
    try:
        result = subprocess.run(cmd, cwd=str(root_dir))
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nShutdown requested")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
