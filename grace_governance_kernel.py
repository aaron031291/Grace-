"""
Grace Governance Kernel - Main integration and initialization.
"""
import asyncio
import logging
from typing import Dict, Any, Optional

# Core infrastructure
from core import EventBus, MemoryCore

# Governance components
from layer_01_governance import (
    VerificationEngine, UnifiedLogic, GovernanceEngine, 
    Parliament, TrustCoreKernel
)

# Supporting infrastructure
from layer_04_audit_logs import ImmutableLogs
from layer_02_event_mesh import TriggerMesh
from immune import EnhancedAVNCore
from mldl import MLDLQuorum


logger = logging.getLogger(__name__)


class GraceGovernanceKernel:
    """
    Main Grace Governance Kernel that orchestrates all components.
    Implements the complete governance architecture with all 21 specialists,
    event routing, audit trails, and health monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.components = {}
        self.is_initialized = False
        self.is_running = False
        
    async def initialize(self):
        """Initialize all governance kernel components."""
        if self.is_initialized:
            logger.warning("Governance kernel already initialized")
            return
        
        logger.info("Initializing Grace Governance Kernel...")
        
        try:
            # Initialize core infrastructure
            await self._initialize_core_infrastructure()
            
            # Initialize governance components
            await self._initialize_governance_components()
            
            # Initialize supporting systems
            await self._initialize_support_systems()
            
            # Setup component integration
            await self._setup_component_integration()
            
            self.is_initialized = True
            logger.info("Grace Governance Kernel initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize governance kernel: {e}")
            raise
    
    async def _initialize_core_infrastructure(self):
        """Initialize core event bus and memory systems."""
        logger.info("Initializing core infrastructure...")
        
        # Event bus for system-wide communication
        self.components['event_bus'] = EventBus()
        
        # Memory core for persistent storage
        memory_db_path = self.config.get('memory_db_path', 'grace_governance.db')
        self.components['memory_core'] = MemoryCore(memory_db_path)
        
        logger.info("Core infrastructure initialized")
    
    async def _initialize_governance_components(self):
        """Initialize main governance components."""
        logger.info("Initializing governance components...")
        
        event_bus = self.components['event_bus']
        memory_core = self.components['memory_core']
        
        # Verification engine for claim analysis
        self.components['verification_engine'] = VerificationEngine(event_bus, memory_core)
        
        # Unified logic for decision synthesis
        self.components['unified_logic'] = UnifiedLogic(event_bus, memory_core)
        
        # Trust core for credibility management
        self.components['trust_core'] = TrustCoreKernel(event_bus, memory_core)
        
        # Parliament for democratic review
        self.components['parliament'] = Parliament(event_bus, memory_core)
        
        # Main governance engine (orchestrator)
        self.components['governance_engine'] = GovernanceEngine(
            event_bus, memory_core,
            self.components['verification_engine'],
            self.components['unified_logic']
        )
        
        logger.info("Governance components initialized")
    
    async def _initialize_support_systems(self):
        """Initialize supporting infrastructure systems."""
        logger.info("Initializing support systems...")
        
        event_bus = self.components['event_bus']
        memory_core = self.components['memory_core']
        
        # Immutable audit logs
        audit_db_path = self.config.get('audit_db_path', 'governance_audit.db')
        self.components['immutable_logs'] = ImmutableLogs(audit_db_path)
        
        # Trigger mesh for event routing
        self.components['trigger_mesh'] = TriggerMesh(event_bus, memory_core)
        
        # Enhanced AVN core for health monitoring
        self.components['avn_core'] = EnhancedAVNCore(event_bus, memory_core)
        
        # MLDL quorum with 21 specialists
        self.components['mldl_quorum'] = MLDLQuorum(event_bus, memory_core)
        
        logger.info("Support systems initialized")
    
    async def _setup_component_integration(self):
        """Setup integration between components."""
        logger.info("Setting up component integration...")
        
        # Register components with trigger mesh
        trigger_mesh = self.components['trigger_mesh']
        
        # Register governance components
        trigger_mesh.register_component(
            "governance_engine", "governance", 
            ["GOVERNANCE_VALIDATION", "GOVERNANCE_ROLLBACK"]
        )
        
        trigger_mesh.register_component(
            "verification_engine", "governance",
            ["CLAIM_VERIFICATION"]
        )
        
        trigger_mesh.register_component(
            "unified_logic", "governance",
            ["CONSENSUS_REQUEST"]
        )
        
        trigger_mesh.register_component(
            "parliament", "governance",
            ["GOVERNANCE_NEEDS_REVIEW", "PARLIAMENT_VOTE_CAST"]
        )
        
        trigger_mesh.register_component(
            "trust_core", "trust",
            ["TRUST_UPDATED"]
        )
        
        # Register support systems
        trigger_mesh.register_component(
            "immutable_logs", "audit",
            ["GOVERNANCE_APPROVED", "GOVERNANCE_REJECTED", "GOVERNANCE_SNAPSHOT_CREATED"]
        )
        
        trigger_mesh.register_component(
            "avn_core", "health",
            ["ANOMALY_DETECTED", "COMPONENT_FAILOVER"]
        )
        
        trigger_mesh.register_component(
            "mldl_quorum", "ml",
            ["MLDL_CONSENSUS_REQUEST"]
        )
        
        # Register components for health monitoring
        avn_core = self.components['avn_core']
        
        avn_core.register_component("governance_engine")
        avn_core.register_component("verification_engine")
        avn_core.register_component("unified_logic")
        avn_core.register_component("parliament")
        avn_core.register_component("trust_core")
        avn_core.register_component("mldl_quorum")
        
        # Update governance engine with AVN core reference
        self.components['governance_engine'].avn_core = avn_core
        
        logger.info("Component integration completed")
    
    async def start(self):
        """Start the governance kernel."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            logger.warning("Governance kernel already running")
            return
        
        logger.info("Starting Grace Governance Kernel...")
        
        try:
            self.is_running = True
            
            # Create initial snapshot
            governance_engine = self.components['governance_engine']
            snapshot = await governance_engine.create_snapshot()
            logger.info(f"Created initial snapshot: {snapshot.snapshot_id}")
            
            # Log kernel startup
            immutable_logs = self.components['immutable_logs']
            await immutable_logs.log_governance_action(
                "kernel_startup",
                {
                    "timestamp": snapshot.created_at.isoformat(),
                    "snapshot_id": snapshot.snapshot_id,
                    "components_initialized": list(self.components.keys())
                },
                "democratic_oversight"
            )
            
            logger.info("Grace Governance Kernel started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start governance kernel: {e}")
            self.is_running = False
            raise
    
    async def shutdown(self):
        """Shutdown the governance kernel gracefully."""
        if not self.is_running:
            logger.warning("Governance kernel not running")
            return
        
        logger.info("Shutting down Grace Governance Kernel...")
        
        try:
            # Create final snapshot
            governance_engine = self.components.get('governance_engine')
            if governance_engine:
                snapshot = await governance_engine.create_snapshot()
                logger.info(f"Created shutdown snapshot: {snapshot.snapshot_id}")
            
            # Log shutdown
            immutable_logs = self.components.get('immutable_logs')
            if immutable_logs:
                await immutable_logs.log_governance_action(
                    "kernel_shutdown",
                    {
                        "timestamp": snapshot.created_at.isoformat() if snapshot else None,
                        "snapshot_id": snapshot.snapshot_id if snapshot else None,
                        "reason": "graceful_shutdown"
                    },
                    "democratic_oversight"
                )
            
            # Close memory connections
            memory_core = self.components.get('memory_core')
            if memory_core:
                memory_core.close()
            
            self.is_running = False
            logger.info("Grace Governance Kernel shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during governance kernel shutdown: {e}")
    
    async def process_governance_request(self, decision_subject: str, 
                                       inputs: Dict[str, Any],
                                       thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Process a governance validation request.
        
        Args:
            decision_subject: Type of decision ("action", "policy", "claim", "deployment")
            inputs: Request inputs including claims, context, etc.
            thresholds: Optional threshold overrides
            
        Returns:
            Governance decision result
        """
        if not self.is_running:
            raise RuntimeError("Governance kernel not running")
        
        try:
            # Route through trigger mesh
            trigger_mesh = self.components['trigger_mesh']
            
            correlation_id = await trigger_mesh.route_event(
                "GOVERNANCE_VALIDATION",
                {
                    "decision_subject": decision_subject,
                    "inputs": inputs,
                    "thresholds": thresholds or {}
                },
                correlation_id=None,
                source_component="external_request"
            )
            
            # Wait for result (simplified - in practice would use async result handling)
            await asyncio.sleep(0.1)  # Allow processing
            
            # Get events by correlation ID
            event_bus = self.components['event_bus']
            related_events = event_bus.get_events_by_correlation(correlation_id)
            
            # Find the governance decision result
            for event in related_events:
                if event['type'] in ['GOVERNANCE_APPROVED', 'GOVERNANCE_REJECTED', 'GOVERNANCE_NEEDS_REVIEW']:
                    return event['payload']
            
            # If no result found, return timeout
            return {
                "correlation_id": correlation_id,
                "outcome": "GOVERNANCE_TIMEOUT",
                "rationale": "Request processing timeout",
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error processing governance request: {e}")
            return {
                "outcome": "GOVERNANCE_ERROR",
                "rationale": f"Processing error: {str(e)}",
                "error": True
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        status = {
            "status": "running" if self.is_running else "initialized",
            "components": {},
            "system_health": {}
        }
        
        # Component status
        for name, component in self.components.items():
            if hasattr(component, 'get_status'):
                status['components'][name] = component.get_status()
            elif hasattr(component, 'health_check'):
                status['components'][name] = {"has_health_check": True}
            else:
                status['components'][name] = {"status": "active"}
        
        # System health from AVN core
        if 'avn_core' in self.components:
            status['system_health'] = self.components['avn_core'].get_system_health_summary()
        
        return status
    
    def get_governance_metrics(self) -> Dict[str, Any]:
        """Get governance performance metrics."""
        metrics = {}
        
        # Trigger mesh routing metrics
        if 'trigger_mesh' in self.components:
            metrics['routing'] = self.components['trigger_mesh'].get_routing_metrics()
        
        # Parliament statistics
        if 'parliament' in self.components:
            metrics['parliament'] = self.components['parliament'].get_member_stats()
        
        # MLDL quorum status
        if 'mldl_quorum' in self.components:
            metrics['mldl_quorum'] = self.components['mldl_quorum'].get_quorum_status()
        
        # Audit log statistics
        if 'immutable_logs' in self.components:
            metrics['audit'] = self.components['immutable_logs'].get_audit_statistics()
        
        # Trust system statistics
        if 'trust_core' in self.components:
            metrics['trust'] = self.components['trust_core'].get_trust_statistics()
        
        return metrics


async def main():
    """Main entry point for the Grace Governance Kernel."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start governance kernel
    kernel = GraceGovernanceKernel()
    
    try:
        await kernel.start()
        
        # Example governance request
        result = await kernel.process_governance_request(
            "policy",
            {
                "claims": [
                    {
                        "id": "claim_001",
                        "statement": "This policy update improves system security",
                        "sources": [{"uri": "https://security-report.example.com", "credibility": 0.8}],
                        "evidence": [{"type": "doc", "pointer": "security_audit_2024.pdf"}],
                        "confidence": 0.85,
                        "logical_chain": [{"step": "Security audit identified vulnerabilities"}]
                    }
                ],
                "context": {
                    "decision_type": "policy",
                    "urgency": "normal"
                }
            }
        )
        
        print("Governance Decision Result:", result)
        
        # Get system status
        status = kernel.get_system_status()
        print("System Status:", status['status'])
        
        # Keep running for a while
        print("Governance kernel running... (Press Ctrl+C to stop)")
        await asyncio.sleep(60)
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(main())