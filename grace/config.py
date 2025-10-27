import os
from pathlib import Path

# --- Core Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# --- Configuration ----------------------------------------------------------------
# LLM Configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# WebSocket Configuration
WEBSOCKET_HOST = os.environ.get("GRACE_WEBSOCKET_HOST", "127.0.0.1")
try:
    WEBSOCKET_PORT = int(os.environ.get("GRACE_WEBSOCKET_PORT", "8765"))
except Exception:
    WEBSOCKET_PORT = 8765

# Kernel-specific settings can be added below
# e.g., LEARNING_KERNEL_INTERVAL = 5
