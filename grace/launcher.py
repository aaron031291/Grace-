#!/usr/bin/env python3
"""
Grace AI - Unified Launcher
==========================
Single entry point for starting all Grace systems

Usage:
  python -m grace.launcher
  python -m grace.launcher --debug
  python -m grace.launcher --kernel learning
  python -m grace.launcher --dry-run
"""

import sys
import asyncio
import logging
import argparse
import signal
from typing import Optional, List

# Add workspace to path
sys.path.insert(0, '/workspaces/Grace-')

from grace.core.service_registry import initialize_global_registry, get_global_registry
from grace.kernels import (
    CognitiveCortex, SentinelKernel, SwarmKernel, MetaLearningKernel,
    LearningKernel, OrchestrationKernel, ResilienceKernel, BaseKernel
)
from grace.multi_os import MultiOSKernel
from grace.services import (
    TaskManager, CommunicationChannel, NotificationService,
    LLMService, WebSocketService, PolicyEngine, TrustLedger, SandboxManager,
    ResilienceService
)
from grace.core.immutable_logs import ImmutableLogger
from grace.orchestration.trigger_mesh import TriggerMesh
from grace.core.truth_layer import CoreTruthLayer

logger = logging.getLogger(__name__)


class GraceLauncher:
    """Main launcher for Grace AI system"""
    
    def __init__(self, args):
        self.args = args
        self.registry = None
        self.kernels: List[BaseKernel] = []
        self.running = False
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging based on debug flag"""
        level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info(f"Logging level: {logging.getLevelName(level)}")
    
    async def initialize(self):
        """Initialize all services and registry"""
        logger.info("=" * 60)
        logger.info("GRACE AI - Initializing System")
        logger.info("=" * 60)
        
        # Initialize registry
        self.registry = initialize_global_registry()
        logger.info("✓ Service registry initialized")
        
        # Register services
        self._register_services()
        logger.info("✓ All services registered")
        
        # Initialize registry (lazy-loads services)
        self.registry.initialize()
        logger.info("✓ All services initialized")
        
        logger.info("✓ System initialization complete")
    
    def _register_services(self):
        """Register all services in the registry"""
        # Register core services
        self.registry.register_factory(
            'task_manager',
            lambda reg: TaskManager()
        )
        self.registry.register_factory(
            'communication_channel',
            lambda reg: CommunicationChannel()
        )
        self.registry.register_factory(
            'notification_service',
            lambda reg: NotificationService()
        )
        self.registry.register_factory(
            'llm_service',
            lambda reg: LLMService()
        )
        self.registry.register_factory(
            'websocket_service',
            lambda reg: WebSocketService()
        )
        self.registry.register_factory(
            'policy_engine',
            lambda reg: PolicyEngine()
        )
        self.registry.register_factory(
            'trust_ledger',
            lambda reg: TrustLedger()
        )
        self.registry.register_factory(
            'sandbox_manager',
            lambda reg: SandboxManager()
        )
        self.registry.register_factory(
            'immune_system', # Registering the resilience service under 'immune_system'
            lambda reg: ResilienceService(reg)
        )
        self.registry.register_factory(
            'immutable_logger',
            lambda reg: ImmutableLogger(log_file_path="grace_log.jsonl")
        )
        self.registry.register_factory(
            'trigger_mesh',
            lambda reg: TriggerMesh(reg)
        )
        self.registry.register_factory(
            'truth_layer',
            lambda reg: CoreTruthLayer()
        )
    
    async def create_kernels(self):
        """Create kernel instances"""
        logger.info("Creating kernels...")
        
        kernel_classes = [
            ('cognitive_cortex', CognitiveCortex),
            ('sentinel', SentinelKernel),
            ('swarm', SwarmKernel),
            ('meta_learning', MetaLearningKernel),
            ('learning', LearningKernel),
            ('orchestration', OrchestrationKernel),
            ('resilience', ResilienceKernel),
            ('multi_os', MultiOSKernel),
        ]
        
        # Filter by requested kernel if specified
        if self.args.kernel:
            kernel_classes = [
                (name, cls) for name, cls in kernel_classes
                if self.args.kernel.lower() in name.lower()
            ]
        
        for name, kernel_class in kernel_classes:
            try:
                kernel = kernel_class(self.registry)
                self.kernels.append(kernel)
                logger.info(f"✓ Created: {name} kernel")
            except Exception as e:
                logger.error(f"✗ Failed to create {name} kernel: {e}")
        
        logger.info(f"✓ Created {len(self.kernels)} kernels")
    
    async def start_kernels(self):
        """Start all kernels"""
        logger.info(f"Starting {len(self.kernels)} kernels...")
        
        tasks = []
        for kernel in self.kernels:
            task = asyncio.create_task(kernel.run_loop(interval=1.0))
            tasks.append(task)
        
        logger.info("✓ All kernels started")
        
        # Wait for all kernels to complete
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Kernels cancelled")
    
    async def run_dry(self):
        """Run in dry-run mode (no actual execution)"""
        logger.info("DRY RUN MODE - Initialization only")
        await self.initialize()
        await self.create_kernels()
        
        # Print status
        logger.info("=" * 60)
        logger.info("DRY RUN STATUS")
        logger.info("=" * 60)
        logger.info(f"Kernels created: {len(self.kernels)}")
        for kernel in self.kernels:
            logger.info(f"  - {kernel.name}")
        
        services = self.registry.get_all_services()
        logger.info(f"Services initialized: {len(services)}")
        for name in services:
            logger.info(f"  - {name}")
        
        logger.info("DRY RUN COMPLETE - System ready")
    
    async def run(self):
        """Run the full system"""
        logger.info("=" * 60)
        logger.info("GRACE AI - Starting System")
        logger.info("=" * 60)
        
        try:
            await self.initialize()
            await self.create_kernels()
            
            if not self.kernels:
                logger.error("✗ No kernels created")
                return
            
            self.running = True
            
            logger.info("=" * 60)
            logger.info("System started - Kernels running")
            logger.info("Press Ctrl+C to shutdown")
            logger.info("=" * 60)
            
            await self.start_kernels()
        
        except KeyboardInterrupt:
            logger.info("Shutdown requested (Ctrl+C)")
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown all kernels and services"""
        logger.info("Shutting down...")
        
        self.running = False
        
        # Stop all kernels
        for kernel in self.kernels:
            await kernel.stop()
        
        # Shutdown registry
        if self.registry:
            self.registry.shutdown()
        
        logger.info("✓ System shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Grace AI Launcher - Start the complete system'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--kernel',
        help='Start only specific kernel (e.g., learning, orchestration)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry-run mode (initialize only, do not start kernels)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Grace AI v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = GraceLauncher(args)
    
    # Run
    try:
        if args.dry_run:
            asyncio.run(launcher.run_dry())
        else:
            asyncio.run(launcher.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
