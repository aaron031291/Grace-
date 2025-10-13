"""Compatibility shim: expose MultiOSService at top-level for tests.

This file delegates to multi_os.multi_os_service.MultiOSService so tests that
import `multi_os_service` continue to work without changing their imports.
"""
import sys
from pathlib import Path

# Ensure the multi_os package's subpackages can be imported as top-level modules
multi_os_path = str(Path(__file__).parent / "multi_os")
if multi_os_path not in sys.path:
	sys.path.insert(0, multi_os_path)

from multi_os.multi_os_service import MultiOSService

__all__ = ["MultiOSService"]
