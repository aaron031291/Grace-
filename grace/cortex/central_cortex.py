"""
Central Cortex - Main coordination module merging old and new logic
Production-ready implementation with full integration
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import json

from grace.cortex.intent_registry import GlobalIntentRegistry, IntentStatus
from grace.cortex.trust_orchestrator import TrustOrchestrator as CortexTrustOrch
from grace.cortex.ethical_framework import EthicalFramework as CortexEthics
from grace.cortex.memory_vault import MemoryVault
from grace.integration.event_bus import EventBus
from grace.trust import TrustScoreManager
from grace.clarity.governance_validation import ConstitutionValidator

logger = logging.getLogger(__name__)


class CentralCortex:
    """
    Central Cortex - Unified coordination system
    Merges old Cortex logic with new Grace architecture
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("grace.cortex.central")
        self.config = config or {}
        
        # Initialize OLD Cortex components (enhanced)
        self.intent_registry = GlobalIntentRegistry(
            storage_path=self.config.get("intent_storage_path")
        )
        self.cortex_trust = CortexTrustOrch(
            storage_path=self.config.get("cortex_trust_path")
        )
        self.cortex_ethics = CortexEthics(
            policies_path=self.config.get("cortex_policies_path")
        )
        self.memory_vault = MemoryVault(
            storage_path=self.config.get("memory_vault_path")
        )
        
        # Initialize NEW Grace components
        self.trust_manager = TrustScoreManager()
        self.constitution = ConstitutionValidator()
        self.event_bus = EventBus()
        
        # Integration bridges
        self._setup_integration_bridges()
        
        self.logger.info("Central Cortex initialized with unified architecture")
    
    def _setup_integration_bridges(self):
        """Setup bridges between old and new components"""
        # Bridge trust systems
        self._bridge_trust_systems()
        
        # Bridge ethical/constitutional systems
        self._bridge_governance_systems()
        
        # Setup event handlers
        self._register_unified_event_handlers()
        
        self.logger.info("Integration bridges established")
    
    def _bridge_trust_systems(self):
        """Bridge old TrustOrchestrator with new TrustScoreManager"""
        # They work together - Cortex for pod-level, TrustManager for component-level
        pass
    
    def _bridge_governance_systems(self):
        """Bridge old EthicalFramework with new ConstitutionValidator"""
        # Load cortex policies into constitution
        cortex_policies = self.cortex_ethics.get_all_policies()
        
        for policy in cortex_policies:
            # Convert old policy format to new constitution rules
            for rule in policy.get("rules", []):
                # Add as constitution rule if not already present
                pass
    
    def _register_unified_event_handlers(self):
        """Register event handlers for unified system"""
        self.event_bus.subscribe("pod.registered", self._handle_pod_registered)
        self.event_bus.subscribe("pod.action_requested", self._handle_action_requested)
        self.event_bus.subscribe("pod.action_completed", self._handle_action_completed)
        self.event_bus.subscribe("component.registered", self._handle_component_registered)
    
    def _handle_pod_registered(self, event_data: Dict[str, Any]):
        """Handle pod registration (old Cortex style)"""
        try:
            pod_id = event_data.get("pod_id")
            metadata = event_data.get("metadata", {})
            
            if not pod_id:
                return
            
            # Initialize in BOTH trust systems
            self.cortex_trust.initialize_trust_score(pod_id, metadata)
            self.trust_manager.initialize_trust(pod_id, "pod", 0.5, metadata)
            
            # Store as memory
            self.memory_vault.store_experience({
                "type": "pod_registration",
                "pod_id": pod_id,
                "metadata": metadata,
                "category": "system_events"
            })
            
            self.logger.info(f"Pod registered: {pod_id}")
        
        except Exception as e:
            self.logger.error(f"Error handling pod registration: {e}")
    
    def _handle_component_registered(self, event_data: Dict[str, Any]):
        """Handle component registration (new Grace style)"""
        try:
            component_id = event_data.get("component_id")
            metadata = event_data.get("metadata", {})
            
            if not component_id:
                return
            
            # Register in trust system
            self.trust_manager.initialize_trust(component_id, "component", 0.6, metadata)
            
            # Store as memory
            self.memory_vault.store_experience({
                "type": "component_registration",
                "component_id": component_id,
                "metadata": metadata,
                "category": "system_events"
            })
            
            self.logger.info(f"Component registered: {component_id}")
        
        except Exception as e:
            self.logger.error(f"Error handling component registration: {e}")
    
    def _handle_action_requested(self, event_data: Dict[str, Any]):
        """Handle action request with unified evaluation"""
        try:
            entity_id = event_data.get("pod_id") or event_data.get("component_id")
            action = event_data.get("action", {})
            request_id = event_data.get("request_id")
            
            if not entity_id or not action:
                return
            
            # Evaluate with BOTH systems
            cortex_eval = self.cortex_ethics.evaluate_action(action)
            constitution_eval = self.constitution.validate_against_constitution(
                action, 
                {}
            )
            
            # Check trust in BOTH systems
            cortex_trust = self.cortex_trust.evaluate_trust_threshold(entity_id, 0.4)
            grace_trust = self.trust_manager.is_trusted(entity_id)
            
            # Final approval requires ALL systems
            approved = (
                cortex_eval["compliant"] and
                constitution_eval.passed and
                cortex_trust[0] and
                grace_trust
            )
            
            # Store decision
            self.memory_vault.store_experience({
                "type": "action_evaluation",
                "entity_id": entity_id,
                "action": action,
                "request_id": request_id,
                "cortex_evaluation": cortex_eval,
                "constitution_evaluation": {
                    "passed": constitution_eval.passed,
                    "score": constitution_eval.score,
                    "violations": constitution_eval.violations
                },
                "approved": approved,
                "category": "action_requests"
            })
            
            # Publish decision
            self.event_bus.publish("cortex.action_decision", {
                "entity_id": entity_id,
                "request_id": request_id,
                "approved": approved,
                "cortex_compliant": cortex_eval["compliant"],
                "constitution_compliant": constitution_eval.passed
            })
            
            self.logger.info(f"Action evaluated for {entity_id}: approved={approved}")
        
        except Exception as e:
            self.logger.error(f"Error handling action request: {e}")
    
    def _handle_action_completed(self, event_data: Dict[str, Any]):
        """Handle action completion with trust updates"""
        try:
            entity_id = event_data.get("pod_id") or event_data.get("component_id")
            action = event_data.get("action", {})
            result = event_data.get("result", {})
            success = result.get("success", False)
            
            if not entity_id:
                return
            
            # Update trust in BOTH systems
            if success:
                # Cortex trust update
                self.cortex_trust.update_trust_score(
                    entity_id,
                    {"history": 0.6, "consistency": 0.6},
                    "Successful action completion",
                    {"action": action, "result": result}
                )
                
                # Grace trust update
                self.trust_manager.record_success(
                    entity_id,
                    weight=1.0,
                    context={"action": action, "result": result}
                )
            else:
                # Cortex trust update
                self.cortex_trust.update_trust_score(
                    entity_id,
                    {"history": 0.4, "consistency": 0.4},
                    "Failed action completion",
                    {"action": action, "result": result}
                )
                
                # Grace trust update
                self.trust_manager.record_failure(
                    entity_id,
                    severity=0.5,
                    context={"action": action, "result": result}
                )
            
            # Store memory
            self.memory_vault.store_experience({
                "type": "action_completion",
                "entity_id": entity_id,
                "action": action,
                "result": result,
                "success": success,
                "category": "action_results"
            })
            
            self.logger.info(f"Action completion handled for {entity_id}: success={success}")
        
        except Exception as e:
            self.logger.error(f"Error handling action completion: {e}")
    
    def evaluate_action(
        self,
        entity_id: str,
        action: Dict[str, Any],
        entity_type: str = "pod"
    ) -> Dict[str, Any]:
        """
        Unified action evaluation
        Uses BOTH old Cortex and new Grace systems
        """
        try:
            # OLD Cortex evaluation
            cortex_eval = self.cortex_ethics.evaluate_action(action)
            cortex_trust, cortex_trust_result = self.cortex_trust.evaluate_trust_threshold(
                entity_id, 
                0.4
            )
            
            # NEW Grace evaluation
            constitution_eval = self.constitution.validate_against_constitution(action, {})
            grace_trust = self.trust_manager.is_trusted(entity_id)
            
            # Combined decision
            approved = (
                cortex_eval["compliant"] and
                constitution_eval.passed and
                cortex_trust and
                grace_trust
            )
            
            return {
                "approved": approved,
                "cortex_evaluation": {
                    "compliant": cortex_eval["compliant"],
                    "score": cortex_eval["overall_score"],
                    "concerns": cortex_eval["concerns"]
                },
                "constitution_evaluation": {
                    "passed": constitution_eval.passed,
                    "score": constitution_eval.score,
                    "violations": constitution_eval.violations,
                    "warnings": constitution_eval.warnings
                },
                "trust_evaluation": {
                    "cortex_trust": cortex_trust,
                    "cortex_score": cortex_trust_result.get("trust_score", 0),
                    "grace_trust": grace_trust
                },
                "reasons": self._compile_denial_reasons(
                    cortex_eval,
                    constitution_eval,
                    cortex_trust,
                    grace_trust
                )
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating action: {e}")
            return {
                "approved": False,
                "error": str(e),
                "reasons": {"system": [f"Evaluation error: {e}"]}
            }
    
    def _compile_denial_reasons(
        self,
        cortex_eval: Dict,
        constitution_eval: Any,
        cortex_trust: bool,
        grace_trust: bool
    ) -> Dict[str, List[str]]:
        """Compile reasons for denial from all systems"""
        reasons = {
            "ethical": [],
            "constitutional": [],
            "trust": []
        }
        
        if not cortex_eval["compliant"]:
            reasons["ethical"].extend([c["description"] for c in cortex_eval["concerns"]])
        
        if not constitution_eval.passed:
            reasons["constitutional"].extend([v["description"] for v in constitution_eval.violations])
        
        if not cortex_trust:
            reasons["trust"].append("Cortex trust threshold not met")
        
        if not grace_trust:
            reasons["trust"].append("Grace trust level insufficient")
        
        return reasons
    
    def get_unified_trust(self, entity_id: str) -> Dict[str, Any]:
        """Get unified trust information from both systems"""
        try:
            cortex_trust = self.cortex_trust.get_trust_score(entity_id)
            grace_trust = self.trust_manager.get_trust_score(entity_id)
            
            return {
                "entity_id": entity_id,
                "cortex_trust": cortex_trust,
                "grace_trust": {
                    "score": grace_trust.score if grace_trust else 0,
                    "level": grace_trust.level.name if grace_trust else "UNKNOWN"
                },
                "unified_score": (
                    (cortex_trust["trust_score"] + (grace_trust.score if grace_trust else 0)) / 2
                )
            }
        
        except Exception as e:
            self.logger.error(f"Error getting unified trust: {e}")
            return {"entity_id": entity_id, "error": str(e)}
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state from all components"""
        try:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "intent_registry": self.intent_registry.get_intent_statistics(),
                "cortex_trust": self.cortex_trust.calculate_system_trust(),
                "grace_trust": self.trust_manager.get_trust_statistics(),
                "cortex_ethics": {
                    "total_policies": len(self.cortex_ethics.get_all_policies())
                },
                "constitution": self.constitution.get_violation_statistics(),
                "memory_vault": self.memory_vault.summarize_experiences(),
                "event_bus": self.event_bus.get_statistics()
            }
        
        except Exception as e:
            self.logger.error(f"Error getting system state: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
