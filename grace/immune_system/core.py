"""
Grace AI Immune System - Core
Central immune system coordinating threat detection, AVN healing, and recovery
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Severity levels for detected threats."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class Threat:
    """Represents a detected threat or anomaly."""
    
    def __init__(self, threat_id: str, threat_type: str, level: ThreatLevel, description: str):
        self.threat_id = threat_id
        self.threat_type = threat_type
        self.level = level
        self.description = description
        self.detected_at = datetime.now().isoformat()
        self.status = "detected"
        self.remediation: Optional[str] = None

class RemediationStrategy:
    """Strategy for remediating a detected threat."""
    
    def __init__(self, threat_id: str, strategy_type: str, actions: List[str]):
        self.threat_id = threat_id
        self.strategy_type = strategy_type
        self.actions = actions
        self.created_at = datetime.now().isoformat()
        self.status = "pending"

class ImmuneSystem:
    """
    Grace's unified immune system combining:
    - Threat detection and classification
    - Autonomous healing network (AVN)
    - Recovery and remediation strategies
    - Self-healing protocols
    """
    
    def __init__(self, event_bus=None, resilience_kernel=None):
        self.event_bus = event_bus
        self.resilience_kernel = resilience_kernel
        self.threats: Dict[str, Threat] = {}
        self.remediation_strategies: Dict[str, RemediationStrategy] = {}
        self.infection_history: List[Dict[str, Any]] = []
        self.recovery_log: List[Dict[str, Any]] = []
    
    async def detect_threat(self, threat_type: str, level: ThreatLevel, description: str) -> str:
        """Detect and register a threat."""
        import uuid
        threat_id = str(uuid.uuid4())[:8]
        
        threat = Threat(threat_id, threat_type, level, description)
        self.threats[threat_id] = threat
        
        self.infection_history.append({
            "threat_id": threat_id,
            "threat_type": threat_type,
            "level": level.value,
            "detected_at": threat.detected_at
        })
        
        logger.warning(f"Threat detected: {threat_type} ({level.value}) - {threat_id}")
        
        # Publish event
        if self.event_bus:
            await self.event_bus.publish("immune.threat_detected", {
                "threat_id": threat_id,
                "threat_type": threat_type,
                "level": level.value
            })
        
        # Initiate automated response for critical threats
        if level == ThreatLevel.CRITICAL:
            await self._initiate_emergency_healing(threat_id)
        
        return threat_id
    
    async def create_remediation_strategy(self, threat_id: str, strategy_type: str, actions: List[str]) -> str:
        """Create a remediation strategy for a threat."""
        strategy = RemediationStrategy(threat_id, strategy_type, actions)
        self.remediation_strategies[threat_id] = strategy
        
        logger.info(f"Remediation strategy created for threat {threat_id}")
        return threat_id
    
    async def execute_remediation(self, threat_id: str) -> bool:
        """Execute remediation for a detected threat."""
        threat = self.threats.get(threat_id)
        strategy = self.remediation_strategies.get(threat_id)
        
        if not threat or not strategy:
            logger.warning(f"Cannot execute remediation: missing threat or strategy for {threat_id}")
            return False
        
        logger.info(f"Executing remediation for threat {threat_id}")
        
        try:
            # Execute each action in the strategy
            for action in strategy.actions:
                logger.info(f"Executing action: {action}")
                # Action execution would be implemented here
            
            # Mark threat as remediated
            threat.status = "remediated"
            strategy.status = "completed"
            
            self.recovery_log.append({
                "threat_id": threat_id,
                "remediated_at": datetime.now().isoformat(),
                "strategy_type": strategy.strategy_type
            })
            
            logger.info(f"Threat {threat_id} successfully remediated")
            
            if self.event_bus:
                await self.event_bus.publish("immune.threat_remediated", {
                    "threat_id": threat_id
                })
            
            return True
        
        except Exception as e:
            logger.error(f"Remediation failed for threat {threat_id}: {str(e)}")
            return False
    
    async def _initiate_emergency_healing(self, threat_id: str):
        """Initiate emergency autonomous healing for critical threats."""
        logger.critical(f"EMERGENCY HEALING INITIATED for critical threat {threat_id}")
        
        if self.event_bus:
            await self.event_bus.publish("immune.emergency_healing", {
                "threat_id": threat_id,
                "timestamp": datetime.now().isoformat()
            })
    
    async def perform_immune_scan(self) -> Dict[str, Any]:
        """Perform a full immune system scan."""
        active_threats = len([t for t in self.threats.values() if t.status == "detected"])
        remediated_threats = len([t for t in self.threats.values() if t.status == "remediated"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_threats_detected": len(self.threats),
            "active_threats": active_threats,
            "remediated_threats": remediated_threats,
            "total_recoveries": len(self.recovery_log)
        }
    
    def get_threat_status(self, threat_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a threat."""
        threat = self.threats.get(threat_id)
        if threat:
            return {
                "threat_id": threat.threat_id,
                "threat_type": threat.threat_type,
                "level": threat.level.value,
                "status": threat.status,
                "detected_at": threat.detected_at
            }
        return None
    
    def get_infection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get infection history."""
        return self.infection_history[-limit:]
    
    def get_recovery_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recovery log."""
        return self.recovery_log[-limit:]
