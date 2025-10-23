"""
Root conftest for pytest - ensures proper imports
"""

import sys
from pathlib import Path

# Ensure grace is importable
sys.path.insert(0, str(Path(__file__).parent))

# Set test environment variables
import os
os.environ.setdefault('ENVIRONMENT', 'testing')
os.environ.setdefault('AUTH_SECRET_KEY', 'test-secret-key-minimum-32-characters-long')
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')
