"""
Grace Test Suite Configuration

Enables intelligent test quality monitoring with KPI integration.
"""

import sys
from pathlib import Path

# Add Grace root to path
grace_root = Path(__file__).parent
sys.path.insert(0, str(grace_root))

# Enable Grace test quality plugin
pytest_plugins = ['grace.testing.pytest_quality_plugin']
