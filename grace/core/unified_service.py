"""
Unified Service - Initializes all Grace components
"""

from typing import Optional, List
import logging
import asyncio
from datetime import datetime
from fastapi import FastAPI

from grace.config import get_config
from grace.watchdog import install_watchdog, get_watchdog
from grace.scheduler import get_scheduler
from grace.integration.event_bus import get_event_bus
from grace.governance.engine import GovernanceEngine
from grace.trust.core import TrustCoreKernel
from grace.memory.core import MemoryCore
from grace.memory.async_lightning import AsyncLightningMemory
from grace.memory.async_fusion import AsyncFusionMemory
from grace.memory.immutable_logs_async import AsyncImmutableLogs
from grace.trigger_mesh import TriggerMesh

logger = logging.getLogger(__name__)


class UnifiedService:
    """
    Grace Unified Service
    
    Initializes and manages all core components:
    - Configuration
    - Watchdog
    - EventBus
    - MemoryCore
    - GovernanceEngine
    - TrustCore
    - Kernels
    - Scheduler
    - Health reporters
    """
    
    def __init__(self):
        self.config = get_config()
        self.watchdog = None
        self.event_bus = None
        self.governance = None
        self.trust = None
        self.scheduler = None
        self.app: Optional[FastAPI] = None
        
        self.memory_core = None
        self.lightning = None
        self.fusion = None
        self.immutable_logs = None
        
        self.trigger_mesh = None
        
        self._kernel_tasks: List[asyncio.Task] = []
        self._initialized = False
        self._start_time: Optional[datetime] = None
    
    async def initialize(self) -> FastAPI:
        """
        Initialize all components
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.warning("Service already initialized")
            return self.app
        
        logger.info("=" * 60)
        logger.info("Initializing Grace Unified Service")
        logger.info("=" * 60)
        
        self._start_time = datetime.utcnow()
        
        try:
            # 1. Install watchdog
            logger.info("1. Installing watchdog...")
            self.watchdog = install_watchdog()
            
            # 2. Initialize EventBus
            logger.info("2. Initializing EventBus...")
            self.event_bus = get_event_bus()
            logger.info(f"   EventBus initialized (max_queue={self.config.event_bus_max_queue})")
            
            # 2.5. Initialize TriggerMesh (after EventBus)
            logger.info("2.5. Initializing TriggerMesh...")
            self.trigger_mesh = TriggerMesh(self.event_bus)
            self.trigger_mesh.load_config()
            self.trigger_mesh.bind_subscriptions()
            logger.info(f"   TriggerMesh initialized ({len(self.trigger_mesh.routes)} routes)")
            
            # 3. Initialize Governance
            if self.config.governance_enabled:
                logger.info("3. Initializing GovernanceEngine...")
                self.governance = GovernanceEngine()
                logger.info("   GovernanceEngine initialized")
            else:
                logger.info("3. GovernanceEngine disabled")
            
            # 4. Initialize Trust
            if self.config.trust_enabled:
                logger.info("4. Initializing TrustCore...")
                self.trust = TrustCoreKernel()
                self.trust.thresholds.acceptable = self.config.trust_default_threshold
                logger.info(f"   TrustCore initialized (threshold={self.config.trust_default_threshold})")
            else:
                logger.info("4. TrustCore disabled")
            
            # 5. Initialize Scheduler
            if self.config.scheduler_enabled:
                logger.info("5. Initializing Scheduler...")
                self.scheduler = get_scheduler()
                self._setup_periodic_tasks()
                logger.info("   Scheduler initialized")
            else:
                logger.info("5. Scheduler disabled")
            
            # 6. Create FastAPI app
            logger.info("6. Creating FastAPI application...")
            self.app = self._create_app()
            logger.info("   FastAPI application created")
            
            # 7. Start kernels
            if self.config.kernels_auto_start:
                logger.info("7. Starting kernels...")
                await self._start_kernels()
                logger.info(f"   Started {len(self._kernel_tasks)} kernels")
            else:
                logger.info("7. Kernel auto-start disabled")
            
            # 8. Setup health reporters
            if self.config.health_reporters_enabled:
                logger.info("8. Setting up health reporters...")
                self._setup_health_reporters()
                logger.info("   Health reporters configured")
            else:
                logger.info("8. Health reporters disabled")
            
            # Initialize Memory Layers
            logger.info("X. Initializing Memory Layers...")
            
            # Lightning (Cache)
            if self.config.memory_lightning_enabled:
                self.lightning = AsyncLightningMemory(redis_url=self.config.redis_url)
                await self.lightning.connect()
                logger.info("   Lightning memory initialized")
            
            # Fusion (Durable Store)
            if self.config.memory_fusion_enabled:
                self.fusion = AsyncFusionMemory(self.config.database_url)
                await self.fusion.connect()
                logger.info("   Fusion memory initialized")
            
            # Immutable Logs
            self.immutable_logs = AsyncImmutableLogs(
                self.config.database_url,
                batch_size=100
            )
            await self.immutable_logs.connect()
            logger.info("   Immutable logs initialized")
            
            # MemoryCore (Coordinator)
            from grace.events.factory import GraceEventFactory
            
            self.memory_core = MemoryCore(
                lightning_memory=self.lightning,
                fusion_memory=self.fusion,
                vector_store=None,  # TODO: Add vector store
                trust_core=self.trust,
                immutable_logs=self.immutable_logs,
                event_bus=self.event_bus,
                event_factory=GraceEventFactory()
            )
            logger.info("   MemoryCore initialized")
            
            self._initialized = True
            
            logger.info("=" * 60)
            logger.info("Grace Unified Service initialized successfully")
            logger.info(f"Environment: {self.config.environment}")
            logger.info(f"Service: {self.config.service_name}")
            logger.info("=" * 60)
            
            return self.app
        
        except Exception as e:
            logger.critical(f"Service initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Grace service: {e}") from e
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with middleware"""
        from grace.api import create_app
        
        app = create_app(config=self.config.to_dict())
        
        # Register shutdown handlers
        @app.on_event("shutdown")
        async def _shutdown():
            await self.shutdown()
        
        return app
    
    async def _start_kernels(self):
        """Start configured kernels with dependency injection"""
        from grace.kernels.multi_os import MultiOSKernel
        from grace.kernels.mldl import MLDLKernel
        from grace.kernels.resilience import ResilienceKernel
        from grace.events.factory import GraceEventFactory
        
        factory = GraceEventFactory()
        kernel_instances = {}
        
        for kernel_name in self.config.kernels_enabled:
            kernel_name = kernel_name.strip()
            
            try:
                if kernel_name == "multi_os":
                    kernel = MultiOSKernel(
                        self.event_bus,
                        factory,
                        trigger_mesh=self.trigger_mesh  # Pass TriggerMesh
                    )
                    await kernel.start()
                    kernel_instances["multi_os"] = kernel
                    logger.info("   ✓ Started kernel: multi_os")
                
                elif kernel_name == "mldl":
                    # Try to get LLM components
                    model_manager = None
                    inference_router = None
                    try:
                        from grace.llm import ModelManager, InferenceRouter
                        model_manager = ModelManager()
                        inference_router = InferenceRouter(model_manager)
                    except:
                        pass
                    
                    kernel = MLDLKernel(
                        self.event_bus,
                        factory,
                        model_manager,
                        inference_router,
                        trigger_mesh=self.trigger_mesh  # Pass TriggerMesh
                    )
                    await kernel.start()
                    kernel_instances["mldl"] = kernel
                    logger.info("   ✓ Started kernel: mldl")
                
                elif kernel_name == "resilience":
                    if not self.governance:
                        logger.warning("   ✗ Resilience kernel requires governance (disabled)")
                        continue
                    
                    kernel = ResilienceKernel(
                        self.event_bus,
                        factory,
                        self.governance,
                        trigger_mesh=self.trigger_mesh  # Pass TriggerMesh
                    )
                    await kernel.start()
                    kernel_instances["resilience"] = kernel
                    logger.info("   ✓ Started kernel: resilience")
                
                else:
                    logger.warning(f"   ✗ Unknown kernel: {kernel_name}")
            
            except Exception as e:
                logger.error(f"   ✗ Failed to start kernel {kernel_name}: {e}")
                if self.config.debug:
                    raise
        
        # Store kernel instances for shutdown
        self.kernel_instances = kernel_instances
        
        # Register shutdown handlers
        for kernel in kernel_instances.values():
            self.watchdog.register_shutdown_handler(kernel.stop)

    def _setup_periodic_tasks(self):
        """Setup periodic maintenance tasks"""
        if not self.scheduler:
            return
        
        # Health check task
        async def _health_check():
            metrics = self.get_health()
            logger.debug("Health check", extra=metrics)
        
        self.scheduler.schedule(
            name="health_check",
            func=_health_check,
            interval_seconds=self.config.health_check_interval
        )
        
        # EventBus metrics
        def _log_event_metrics():
            if self.event_bus:
                metrics = self.event_bus.get_metrics()
                logger.info("EventBus metrics", extra=metrics)
        
        self.scheduler.schedule(
            name="event_bus_metrics",
            func=_log_event_metrics,
            interval_seconds=60
        )
    
    def _setup_health_reporters(self):
        """Setup health check endpoints"""
        # Health reporters are already in the API
        # This method can be extended for custom health checks
        pass
    
    def get_health(self) -> dict:
        """Get overall system health"""
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        health = {
            "status": "healthy" if self._initialized else "initializing",
            "uptime_seconds": uptime,
            "environment": self.config.environment,
            "service_name": self.config.service_name,
            "components": {
                "watchdog": self.watchdog is not None,
                "event_bus": self.event_bus is not None,
                "governance": self.governance is not None,
                "trust": self.trust is not None,
                "scheduler": self.scheduler is not None
            }
        }
        
        # Add component-specific health
        if self.event_bus:
            health["event_bus_metrics"] = self.event_bus.get_metrics()
        
        if self.scheduler:
            health["scheduler_stats"] = self.scheduler.get_stats()
        
        return health
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Grace Unified Service...")
        
        # Stop scheduler
        if self.scheduler:
            await self.scheduler.stop_all()
        
        # Stop kernels via instances
        if hasattr(self, 'kernel_instances'):
            for name, kernel in self.kernel_instances.items():
                try:
                    logger.info(f"Stopping kernel: {name}")
                    await kernel.stop()
                except Exception as e:
                    logger.error(f"Error stopping kernel {name}: {e}")
        
        # Shutdown memory layers
        if self.immutable_logs:
            await self.immutable_logs.disconnect()
        
        if self.fusion:
            await self.fusion.disconnect()
        
        if self.lightning:
            await self.lightning.disconnect()
        
        # Shutdown EventBus
        if self.event_bus:
            await self.event_bus.shutdown()
        
        logger.info("Grace Unified Service shutdown complete")


def create_unified_app(config: Optional[dict] = None) -> FastAPI:
    """
    Create and initialize unified Grace application
    
    This is the main entry point for service mode
    
    Returns:
        FastAPI application
    
    Raises:
        RuntimeError: If initialization fails
    """
    service = UnifiedService()
    
    # Run async initialization
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(service.initialize())
    
    return app
