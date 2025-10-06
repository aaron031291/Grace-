"""MTL Service - Unified API combining Memory + Trust + Logs."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .memory_orchestrator import MemoryOrchestrator
from .trust_core import TrustCore
from .immutable_logger import ImmutableLogger

logger = logging.getLogger(__name__)


class MTLService:
    """
    Unified API for all MTL operations.
    
    Combines:
    - Memory (via MemoryOrchestrator)
    - Trust (via TrustCore)
    - Logs (via ImmutableLogger)
    
    Features:
    - Constitutional validation hooks
    - Event mesh integration points
    - Performance monitoring
    - Health checks
    """
    
    def __init__(self):
        # Initialize core components
        self.memory = MemoryOrchestrator()
        self.trust = TrustCore()
        self.logger = ImmutableLogger()
        
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_operations": 0,
            "constitutional_checks": 0,
            "violations_detected": 0,
            "start_time": time.time()
        }
        
        logger.info("MTL Service initialized")
    
    async def store_with_governance(
        self,
        data: Dict[str, Any],
        trust_score: float,
        constitutional_check: bool = True,
        component_id: str = "unknown"
    ) -> str:
        """
        Store data with governance validation.
        
        Args:
            data: Data to store
            trust_score: Trust score for the data
            constitutional_check: Perform constitutional validation
            component_id: ID of component storing data
        
        Returns:
            Entry ID
        """
        start = time.time()
        self._stats["total_operations"] += 1
        
        # Constitutional check
        compliance_score = 1.0
        if constitutional_check:
            self._stats["constitutional_checks"] += 1
            compliance_score = await self._check_constitutional_compliance(data)
            
            if compliance_score < 0.5:
                self._stats["violations_detected"] += 1
                
                # Log violation
                await self.logger.log(
                    event_type="POLICY_VIOLATION",
                    component_id=component_id,
                    payload={
                        "data": data,
                        "compliance_score": compliance_score,
                        "action": "rejected"
                    },
                    trust_score=trust_score,
                    constitutional_compliance=compliance_score
                )
                
                raise ValueError(f"Constitutional compliance too low: {compliance_score}")
        
        # Store in memory
        key = data.get("key", f"data_{int(time.time() * 1000)}")
        entry_id = await self.memory.store_with_trust(
            data,
            trust_score=trust_score
        )
        
        # Log storage
        await self.logger.log(
            event_type="MEMORY_STORED",
            component_id=component_id,
            payload={
                "entry_id": entry_id,
                "key": key,
                "trust_score": trust_score
            },
            trust_score=trust_score,
            constitutional_compliance=compliance_score
        )
        
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Stored with governance in {elapsed:.2f}ms")
        
        return entry_id
    
    async def retrieve_with_trust(
        self,
        key: str,
        min_trust: float = 0.5,
        component_id: str = "unknown"
    ) -> Optional[Any]:
        """
        Retrieve data with trust validation.
        
        Args:
            key: Key to retrieve
            min_trust: Minimum required trust score
            component_id: ID of component retrieving data
        
        Returns:
            Data if found and trusted, None otherwise
        """
        start = time.time()
        self._stats["total_operations"] += 1
        
        # Retrieve from memory
        value = await self.memory.get(key)
        
        if value is None:
            # Log retrieval miss
            await self.logger.log(
                event_type="MEMORY_RETRIEVED",
                component_id=component_id,
                payload={
                    "key": key,
                    "found": False
                },
                trust_score=0.0
            )
            return None
        
        # Check trust score
        entity_trust = await self.trust.get_trust_score(key)
        
        if entity_trust < min_trust:
            logger.warning(f"Trust too low for '{key}': {entity_trust} < {min_trust}")
            
            # Log low trust access attempt
            await self.logger.log(
                event_type="MEMORY_RETRIEVED",
                component_id=component_id,
                payload={
                    "key": key,
                    "found": True,
                    "trust_score": entity_trust,
                    "min_trust": min_trust,
                    "action": "rejected"
                },
                trust_score=entity_trust
            )
            
            return None
        
        # Log successful retrieval
        await self.logger.log(
            event_type="MEMORY_RETRIEVED",
            component_id=component_id,
            payload={
                "key": key,
                "found": True,
                "trust_score": entity_trust
            },
            trust_score=entity_trust
        )
        
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Retrieved with trust in {elapsed:.2f}ms")
        
        return value
    
    async def query_with_audit(
        self,
        query: str,
        log_access: bool = True,
        component_id: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """
        Query with audit logging.
        
        Args:
            query: Query string
            log_access: Log the query access
            component_id: ID of component querying
        
        Returns:
            Query results
        """
        start = time.time()
        self._stats["total_operations"] += 1
        
        # Execute query
        results = await self.memory.query(query, search_type="hybrid")
        
        # Log access if requested
        if log_access:
            await self.logger.log(
                event_type="MEMORY_RETRIEVED",
                component_id=component_id,
                payload={
                    "query": query,
                    "result_count": len(results)
                },
                trust_score=0.8
            )
        
        elapsed = (time.time() - start) * 1000
        logger.debug(f"Query with audit in {elapsed:.2f}ms")
        
        return results
    
    async def update_entity_trust(
        self,
        entity_id: str,
        performance_data: Dict[str, Any],
        component_id: str = "unknown"
    ) -> float:
        """
        Update trust score for entity.
        
        Args:
            entity_id: Entity to update
            performance_data: Performance metrics
            component_id: Component reporting performance
        
        Returns:
            New trust score
        """
        self._stats["total_operations"] += 1
        
        # Update trust
        new_trust = await self.trust.update_trust(entity_id, performance_data)
        
        # Log trust update
        await self.logger.log(
            event_type="TRUST_UPDATED",
            component_id=component_id,
            payload={
                "entity_id": entity_id,
                "new_trust": new_trust,
                "performance_data": performance_data
            },
            trust_score=new_trust
        )
        
        return new_trust
    
    async def log_decision(
        self,
        decision_type: str,
        decision_data: Dict[str, Any],
        component_id: str = "governance_kernel"
    ) -> str:
        """
        Log a governance decision.
        
        Args:
            decision_type: Type of decision
            decision_data: Decision details
            component_id: Component making decision
        
        Returns:
            Audit ID
        """
        self._stats["total_operations"] += 1
        
        # Determine compliance based on approval
        compliance = 1.0 if decision_data.get("approved", True) else 0.5
        
        audit_id = await self.logger.log(
            event_type="DECISION_MADE",
            component_id=component_id,
            payload={
                "decision_type": decision_type,
                **decision_data
            },
            trust_score=decision_data.get("confidence", 0.8),
            constitutional_compliance=compliance
        )
        
        return audit_id
    
    async def health_check(self) -> Dict[str, str]:
        """
        Check health of all MTL components.
        
        Returns:
            Health status for each component
        """
        health = {}
        
        # Check memory
        memory_health = await self.memory.health_check()
        health.update(memory_health)
        
        # Check trust
        try:
            await self.trust.get_stats()
            health["trust"] = "healthy"
        except Exception as e:
            health["trust"] = f"unhealthy: {e}"
        
        # Check logger
        try:
            integrity_ok = await self.logger.verify_chain_integrity()
            health["logger"] = "healthy" if integrity_ok else "unhealthy: chain integrity failed"
        except Exception as e:
            health["logger"] = f"unhealthy: {e}"
        
        return health
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive MTL statistics."""
        memory_stats = await self.memory.get_stats()
        trust_stats = await self.trust.get_stats()
        logger_stats = await self.logger.get_stats()
        
        uptime = time.time() - self._stats["start_time"]
        
        return {
            "mtl_service": {
                "total_operations": self._stats["total_operations"],
                "constitutional_checks": self._stats["constitutional_checks"],
                "violations_detected": self._stats["violations_detected"],
                "uptime_seconds": round(uptime, 1)
            },
            "memory": memory_stats,
            "trust": trust_stats,
            "logger": logger_stats
        }
    
    async def _check_constitutional_compliance(self, data: Dict[str, Any]) -> float:
        """
        Check constitutional compliance of data.
        
        This is a placeholder - would integrate with governance engine.
        
        Returns:
            Compliance score (0.0-1.0)
        """
        # Simple checks
        score = 1.0
        
        # Check for privacy violations (PII in data)
        if "email" in str(data).lower() or "ssn" in str(data).lower():
            score -= 0.3
        
        # Check for sensitive operations
        if data.get("operation") in ["delete_user", "grant_admin"]:
            score -= 0.2
        
        return max(0.0, score)
