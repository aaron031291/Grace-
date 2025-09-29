"""API router initialization."""

from . import auth, health, memory, tasks, governance, collab, websocket

__all__ = ["auth", "health", "memory", "tasks", "governance", "collab", "websocket"]