# ... existing code ...
# LLM Configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# WebSocket Configuration
WEBSOCKET_HOST = os.getenv("GRACE_WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("GRACE_WEBSOCKET_PORT", 8765))

# Kernel-specific settings can be added below
# e.g., LEARNING_KERNEL_INTERVAL = 5
