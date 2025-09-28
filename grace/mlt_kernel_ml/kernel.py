"""
MLT Kernel ML - Main orchestrating kernel for machine learning tuning and adaptation.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .contracts import Experience, Insight, AdaptationPlan, ExperienceSource
from .experience_collector import ExperienceCollector
from .insight_generator import InsightGenerator
from .adaptation_planner import AdaptationPlanner
from .policy_tuner import PolicyTuner
from .snapshot_manager import SnapshotManager
from .mlt_service import MLTService
from .mlt_gov_bridge import MLTGovernanceBridge


logger = logging.getLogger(__name__)


class MLTKernelML:
    """
    Main MLT Kernel for Machine Learning Tuning.
    
    Transforms raw experiences into insights and adaptation plans,
    auto-tunes thresholds, re-weights specialists, and manages model lifecycle.
    """
    
    def __init__(self, event_bus=None, governance_engine=None):
        self.event_bus = event_bus
        self.governance_engine = governance_engine
        
        # Initialize core components
        self.experience_collector = ExperienceCollector(event_bus)
        self.insight_generator = InsightGenerator()
        self.adaptation_planner = AdaptationPlanner()
        self.policy_tuner = PolicyTuner()
        self.snapshot_manager = SnapshotManager()
        
        # Initialize service interfaces
        self.mlt_service = MLTService(self)
        self.gov_bridge = MLTGovernanceBridge(governance_engine, event_bus)
        
        # State
        self.running = False
        self.learning_loop_task = None
        self.learning_interval = 300  # 5 minutes
        
        logger.info("MLT Kernel ML initialized")
    
    async def start(self):
        """Start the MLT kernel and begin processing."""
        try:
            self.running = True
            
            # Create initial snapshot
            await self.snapshot_manager.create_snapshot({
                "planner_version": "1.0.0",
                "search_spaces": {
                    "tabular.classification": "xgb,rf,lr,svm",
                    "tabular.regression": "xgb,rf,lr,svr",
                    "nlp": "transformer,lstm,cnn",
                    "vision": "cnn,resnet,efficientnet"
                },
                "weights": {
                    "Performance": 1.0,
                    "Fairness": 1.0,
                    "Anomaly": 1.0,
                    "Uncertainty": 1.0
                },
                "policies": {
                    "min_calibration": 0.95,
                    "canary_steps": [5, 25, 50, 100]
                },
                "active_jobs": []
            })
            
            # Start continuous learning loop
            self.learning_loop_task = asyncio.create_task(self._learning_loop())
            
            # Subscribe to governance events if event bus available
            if self.event_bus:
                await self._subscribe_to_events()
            
            logger.info("MLT Kernel ML started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MLT Kernel ML: {e}")
            raise
    
    async def stop(self):
        """Stop the MLT kernel."""
        try:
            self.running = False
            
            if self.learning_loop_task:
                self.learning_loop_task.cancel()
                try:
                    await self.learning_loop_task
                except asyncio.CancelledError:
                    pass
            
            # Create final snapshot
            await self.snapshot_manager.create_snapshot()
            
            logger.info("MLT Kernel ML stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping MLT Kernel ML: {e}")
    
    async def _learning_loop(self):
        """Main learning loop that processes experiences and generates adaptations."""
        while self.running:
            try:
                await self._process_learning_cycle()
                await asyncio.sleep(self.learning_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _process_learning_cycle(self):
        """Process one complete learning cycle."""
        try:
            # 1. Collect recent experiences
            experiences = self.experience_collector.get_recent_experiences(100)
            
            if len(experiences) < 5:  # Need sufficient data
                logger.debug(f"Insufficient experiences ({len(experiences)}) for learning cycle")
                return
            
            # 2. Generate insights from experiences
            insights = await self.insight_generator.generate_insights(experiences)
            
            if not insights:
                logger.debug("No insights generated this cycle")
                return
            
            # 3. Create adaptation plan from insights
            adaptation_plan = await self.adaptation_planner.create_adaptation_plan(insights)
            
            if not adaptation_plan:
                logger.debug("No adaptation plan generated this cycle")
                return
            
            # 4. Generate policy recommendations
            current_policies = self.snapshot_manager.current_state.get("policies", {})
            policy_recommendations = await self.policy_tuner.generate_policy_recommendations(
                insights, current_policies
            )
            
            # 5. Submit plan to governance for approval
            proposal_id = await self.gov_bridge.submit_plan_for_approval(adaptation_plan)
            
            logger.info(f"Completed learning cycle: {len(experiences)} exp -> {len(insights)} insights -> 1 plan -> proposal {proposal_id}")
            
            # 6. Emit insight ready event
            if self.event_bus:
                for insight in insights:
                    await self.event_bus.publish("MLT_INSIGHT_READY", {
                        "payload": insight.to_dict()
                    })
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events from the event bus."""
        if not self.event_bus:
            return
        
        # Subscribe to governance decisions
        await self.event_bus.subscribe("GOVERNANCE_APPROVED", self._handle_governance_approved)
        await self.event_bus.subscribe("GOVERNANCE_REJECTED", self._handle_governance_rejected)
        
        # Subscribe to rollback requests
        await self.event_bus.subscribe("ROLLBACK_REQUESTED", self._handle_rollback_request)
        
        logger.info("Subscribed to governance and rollback events")
    
    async def _handle_governance_approved(self, event_data: Dict[str, Any]):
        """Handle governance approval events."""
        await self.gov_bridge.handle_governance_decision({
            "event_type": "GOVERNANCE_APPROVED",
            "payload": event_data
        })
    
    async def _handle_governance_rejected(self, event_data: Dict[str, Any]):
        """Handle governance rejection events."""
        await self.gov_bridge.handle_governance_decision({
            "event_type": "GOVERNANCE_REJECTED",
            "payload": event_data
        })
    
    async def _handle_rollback_request(self, event_data: Dict[str, Any]):
        """Handle rollback requests."""
        try:
            latest_snapshot = self.snapshot_manager.get_latest_snapshot()
            if latest_snapshot:
                success = await self.snapshot_manager.rollback_to_snapshot(
                    latest_snapshot.snapshot_id
                )
                
                if success and self.event_bus:
                    await self.event_bus.publish("ROLLBACK_COMPLETED", {
                        "snapshot_id": latest_snapshot.snapshot_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.info(f"Rollback completed to snapshot {latest_snapshot.snapshot_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle rollback request: {e}")
    
    # Public API methods
    async def submit_experience(self, source: ExperienceSource, raw_data: Dict[str, Any]) -> Experience:
        """Submit a new experience for processing."""
        return await self.experience_collector.collect_experience(source, raw_data)
    
    async def get_current_insights(self, limit: int = 20) -> List[Insight]:
        """Get current insights."""
        return self.insight_generator.get_recent_insights(limit)
    
    async def get_adaptation_plans(self, limit: int = 10) -> List[AdaptationPlan]:
        """Get recent adaptation plans."""
        return self.adaptation_planner.get_recent_plans(limit)
    
    async def force_learning_cycle(self) -> Dict[str, Any]:
        """Force immediate execution of learning cycle."""
        await self._process_learning_cycle()
        return {"status": "learning_cycle_completed", "timestamp": datetime.now().isoformat()}
    
    async def create_manual_snapshot(self) -> str:
        """Create a manual snapshot."""
        snapshot = await self.snapshot_manager.create_snapshot()
        return snapshot.snapshot_id
    
    async def check_rollback_conditions(self, metrics: Dict[str, Any]) -> tuple[bool, str]:
        """Check if rollback conditions are met."""
        return self.snapshot_manager.should_rollback(metrics)
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "kernel": {
                "running": self.running,
                "learning_interval": self.learning_interval,
                "uptime": "running" if self.running else "stopped"
            },
            "experience_collector": self.experience_collector.get_stats(),
            "insight_generator": self.insight_generator.get_stats(),
            "adaptation_planner": self.adaptation_planner.get_stats(),
            "policy_tuner": self.policy_tuner.get_stats(),
            "snapshot_manager": self.snapshot_manager.get_stats(),
            "governance_bridge": self.gov_bridge.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_service_app(self):
        """Get the FastAPI service application."""
        return self.mlt_service.get_app()
    
    # Integration methods for other systems
    def set_governance_engine(self, governance_engine):
        """Set the governance engine reference."""
        self.governance_engine = governance_engine
        self.gov_bridge.governance_engine = governance_engine
    
    def set_event_bus(self, event_bus):
        """Set the event bus reference."""
        self.event_bus = event_bus
        self.experience_collector.event_bus = event_bus
        self.gov_bridge.event_bus = event_bus
    
    # Health and monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check each component
            components = [
                ("experience_collector", self.experience_collector),
                ("insight_generator", self.insight_generator),
                ("adaptation_planner", self.adaptation_planner),
                ("policy_tuner", self.policy_tuner),
                ("snapshot_manager", self.snapshot_manager),
                ("gov_bridge", self.gov_bridge)
            ]
            
            all_healthy = True
            for name, component in components:
                try:
                    # Basic health check - component exists and has expected methods
                    component_healthy = hasattr(component, 'get_stats')
                    health["components"][name] = "healthy" if component_healthy else "degraded"
                    
                    if not component_healthy:
                        all_healthy = False
                        
                except Exception as e:
                    health["components"][name] = f"error: {str(e)}"
                    all_healthy = False
            
            health["status"] = "healthy" if all_healthy else "degraded"
            
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
        
        return health