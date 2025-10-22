"""
Unified Service - Main Grace service orchestrator
"""

from typing import Optional, List
import logging
import asyncio

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_unified_app(config: Optional[dict] = None) -> FastAPI:
    """
    Create unified Grace application and initialize runtime kernels.
    - Registers routers lazily
    - Starts kernel background tasks on startup
    """
    from grace.config import get_settings

    settings = get_settings()
    app = FastAPI(title=settings.api_title, version=settings.api_version, debug=settings.debug)

    # Lazy import middleware and routers to avoid circular imports
    from grace.middleware.logging import setup_logging, LoggingMiddleware
    from grace.middleware.metrics import MetricsMiddleware
    from grace.middleware.rate_limit import RateLimitMiddleware

    # Configure logging
    setup_logging(log_level=settings.observability.log_level, json_output=settings.observability.json_logs,
                  log_file=settings.observability.log_file)

    # Add middlewares
    app.add_middleware(LoggingMiddleware)
    if settings.rate_limit.enabled:
        app.add_middleware(RateLimitMiddleware)
    if settings.observability.metrics_enabled:
        app.add_middleware(MetricsMiddleware)

    # Include core routers (lazy)
    try:
        from grace.api.v1.auth import router as auth_router
        from grace.api.v1.documents import router as documents_router
        from grace.api.v1.kernels import router as kernels_router  # NEW

        app.include_router(auth_router, prefix=settings.api_prefix)
        app.include_router(documents_router, prefix=settings.api_prefix)
        app.include_router(kernels_router, prefix=settings.api_prefix)  # NEW
    except Exception:
        logger.debug("Some routers not available during initialization; continuing.")

    # Kernel handles to cancel on shutdown
    app.state._kernel_tasks: List[asyncio.Task] = []

    async def _start_kernels():
        """
        Start configured kernels (non-blocking). Kernels are small coroutines that register with EventBus.
        """
        # Import kernel modules lazily
        from importlib import import_module

        kernel_modules = [
            "grace.kernels.multi_os",
            "grace.kernels.mldl",
            "grace.kernels.resilience",
        ]

        for km in kernel_modules:
            try:
                mod = import_module(km)
                if hasattr(mod, "start"):
                    task = asyncio.create_task(mod.start())
                    app.state._kernel_tasks.append(task)
                    logger.info(f"Started kernel: {km}")
            except Exception as e:
                logger.warning(f"Failed to start kernel {km}: {e}")

    async def _stop_kernels():
        # attempt graceful cancel
        for t in list(getattr(app.state, "_kernel_tasks", [])):
            try:
                t.cancel()
                await asyncio.wait_for(t, timeout=2.0)
            except Exception:
                pass

    @app.on_event("startup")
    async def _on_startup():
        logger.info("Unified app startup: launching kernels")
        await _start_kernels()

    @app.on_event("shutdown")
    async def _on_shutdown():
        logger.info("Unified app shutdown: stopping kernels")
        await _stop_kernels()

    # health endpoint
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.api_version}

    return app


# Backwards compatibility
UnifiedService = create_unified_app
