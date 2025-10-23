"""
Grace AI Consciousness Module - The perpetual self-reflection and decision-making loop
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class Consciousness:
    """The main consciousness loop - Grace's perpetual self-awareness and decision engine."""
    
    def __init__(
        self,
        cognitive_cortex,
        task_manager,
        kpi_monitor,
        component_registry,
        event_bus,
        immutable_logger
    ):
        self.cognitive_cortex = cognitive_cortex
        self.task_manager = task_manager
        self.kpi_monitor = kpi_monitor
        self.component_registry = component_registry
        self.event_bus = event_bus
        self.immutable_logger = immutable_logger
        self.tick_count = 0
        self.running = False
    
    async def run(self, tick_interval: float = 1.0):
        """Run the consciousness loop continuously."""
        self.running = True
        logger.info("Consciousness loop started")
        
        while self.running:
            try:
                await self._tick(tick_interval)
            except Exception as e:
                logger.error(f"Error in consciousness tick: {str(e)}")
                await asyncio.sleep(tick_interval)
    
    async def _tick(self, interval: float):
        """Single tick of the consciousness loop."""
        self.tick_count += 1
        logger.info(f"[Tick {self.tick_count}] Consciousness loop iteration started.")
        
        # Phase 1: Self-Reflection
        logger.info(f"[Tick {self.tick_count}] Phase 1: Self-Reflection...")
        await self._self_reflect()
        
        # Phase 2: Review Goals
        logger.info(f"[Tick {self.tick_count}] Phase 2: Reviewing Goals...")
        await self._review_goals()
        
        # Phase 3: Synthesize and Decide
        logger.info(f"[Tick {self.tick_count}] Phase 3: Synthesizing and Deciding...")
        decision = await self._synthesize_and_decide()
        
        # Phase 4: Act on Decision
        logger.info(f"[Tick {self.tick_count}] Phase 4: Acting on Decision...")
        await self._act_on_decision(decision)
        
        # Phase 5: Self-Mutate (reload improved components)
        logger.info(f"[Tick {self.tick_count}] Phase 5: Self-Mutation Check...")
        await self._self_mutate()
        
        await asyncio.sleep(interval)
    
    async def _self_reflect(self):
        """Gather system state for reflection."""
        kpis = self.kpi_monitor.get_all_kpis()
        overall_trust = self.kpi_monitor.get_overall_trust()
        logger.info(f"Self-reflection: KPIs={kpis}, Overall Trust={overall_trust:.1f}")
    
    async def _review_goals(self):
        """Review active tasks and goals."""
        open_tasks = self.task_manager.get_open_tasks()
        logger.info(f"Found {len(open_tasks)} open tasks.")
    
    async def _synthesize_and_decide(self):
        """Use CognitiveCortex to make strategic decisions."""
        system_state = {
            "kpis": self.kpi_monitor.get_all_kpis(),
            "trust": self.kpi_monitor.get_overall_trust(),
            "tasks": len(self.task_manager.get_open_tasks())
        }
        
        decision = await self.cognitive_cortex.synthesize_and_decide(system_state)
        return decision
    
    async def _act_on_decision(self, decision: dict):
        """Execute the decision from CognitiveCortex."""
        if decision and decision.get("action"):
            logger.info(f"Acting on decision: {decision['action']}")
            await self.event_bus.publish("consciousness.decision_made", decision)
    
    async def _self_mutate(self):
        """Check if components need to be reloaded after self-improvement."""
        logger.info("Checking for improved components to reload...")
        # TODO: Implement dynamic component reloading based on MentorEngine improvements
    
    async def stop(self):
        """Stop the consciousness loop."""
        self.running = False
        logger.info("Consciousness loop stopped")
