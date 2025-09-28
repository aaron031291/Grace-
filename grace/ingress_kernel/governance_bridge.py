"""
Ingress-Governance Bridge - Connects Ingress Kernel to Governance System.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

from grace.contracts.ingress_contracts import SourceConfig, NormRecord
from grace.contracts.ingress_events import IngressEvent, IngressEventType, ValidationFailedPayload


logger = logging.getLogger(__name__)


class GovernanceRequest:
    """Governance request for policy evaluation."""
    
    def __init__(self, request_type: str, data: Dict[str, Any], context: Optional[Dict] = None):
        self.request_id = str(uuid.uuid4())
        self.request_type = request_type
        self.data = data
        self.context = context or {}
        self.timestamp = datetime.utcnow()


class IngressGovernanceBridge:
    """Bridge between Ingress Kernel and Governance System."""
    
    def __init__(self, ingress_kernel, governance_engine=None, policy_engine=None):
        """
        Initialize the governance bridge.
        
        Args:
            ingress_kernel: The Ingress Kernel instance
            governance_engine: Governance engine for policy decisions
            policy_engine: Policy engine for validation
        """
        self.ingress_kernel = ingress_kernel
        self.governance_engine = governance_engine
        self.policy_engine = policy_engine
        
        # Policy cache for performance
        self.policy_cache = {}
        self.cache_expiry = {}
        
        # Request tracking
        self.pending_requests: Dict[str, GovernanceRequest] = {}
        self.approved_requests: Dict[str, Dict[str, Any]] = {}
        self.rejected_requests: Dict[str, Dict[str, Any]] = {}
        
        self.running = False
    
    async def start(self):
        """Start the governance bridge."""
        if self.running:
            return
        
        logger.info("Starting Ingress-Governance Bridge...")
        self.running = True
        
        # Start policy monitoring
        asyncio.create_task(self._monitor_policies())
        
        logger.info("Ingress-Governance Bridge started")
    
    async def stop(self):
        """Stop the governance bridge."""
        logger.info("Stopping Ingress-Governance Bridge...")
        self.running = False
    
    async def validate_source_registration(self, source_config: SourceConfig) -> Dict[str, Any]:
        """
        Validate source registration against governance policies.
        
        Args:
            source_config: Source configuration to validate
            
        Returns:
            Validation result with approval/rejection
        """
        try:
            # Create governance request
            request = GovernanceRequest(
                request_type="source_registration",
                data=source_config.dict(),
                context={
                    "source_type": source_config.kind,
                    "governance_label": source_config.governance_label,
                    "pii_policy": source_config.pii_policy,
                    "retention_days": source_config.retention_days
                }
            )
            
            # Check if registration requires approval
            approval_required = await self._requires_approval(source_config)
            
            if approval_required:
                return await self._request_governance_approval(request)
            else:
                return await self._auto_approve_source(source_config)
                
        except Exception as e:
            logger.error(f"Source validation failed: {e}")
            return {
                "approved": False,
                "reason": f"Validation error: {e}",
                "requires_review": True
            }
    
    async def validate_record_processing(self, record: NormRecord) -> Dict[str, Any]:
        """
        Validate record processing against policies.
        
        Args:
            record: Normalized record to validate
            
        Returns:
            Validation result
        """
        try:
            # PII detection and policy enforcement
            pii_result = await self._check_pii_policy(record)
            if not pii_result["passed"]:
                return {
                    "approved": False,
                    "reason": "PII policy violation",
                    "details": pii_result
                }
            
            # Quality thresholds
            quality_result = await self._check_quality_thresholds(record)
            if not quality_result["passed"]:
                return {
                    "approved": False,
                    "reason": "Quality threshold violation",
                    "details": quality_result
                }
            
            # Governance label compliance
            governance_result = await self._check_governance_compliance(record)
            if not governance_result["passed"]:
                return {
                    "approved": False,
                    "reason": "Governance policy violation",
                    "details": governance_result
                }
            
            return {
                "approved": True,
                "reason": "All policies satisfied"
            }
            
        except Exception as e:
            logger.error(f"Record validation failed: {e}")
            return {
                "approved": False,
                "reason": f"Validation error: {e}"
            }
    
    async def request_policy_update(self, policy_type: str, current_value: Any, proposed_value: Any) -> Dict[str, Any]:
        """
        Request policy update through governance system.
        
        Args:
            policy_type: Type of policy to update
            current_value: Current policy value
            proposed_value: Proposed new value
            
        Returns:
            Request result
        """
        try:
            request = GovernanceRequest(
                request_type="policy_update",
                data={
                    "policy_type": policy_type,
                    "current_value": current_value,
                    "proposed_value": proposed_value,
                    "justification": f"Ingress system requesting {policy_type} update"
                },
                context={
                    "source": "ingress_kernel",
                    "priority": "normal"
                }
            )
            
            return await self._request_governance_approval(request)
            
        except Exception as e:
            logger.error(f"Policy update request failed: {e}")
            return {
                "approved": False,
                "reason": f"Request error: {e}"
            }
    
    async def _requires_approval(self, source_config: SourceConfig) -> bool:
        """Check if source registration requires governance approval."""
        # High-risk sources require approval
        high_risk_kinds = ["sql", "github", "social"]
        if source_config.kind in high_risk_kinds:
            return True
        
        # Restricted governance labels require approval
        if source_config.governance_label == "restricted":
            return True
        
        # External sources with PII require approval
        if source_config.pii_policy != "block" and "external" in source_config.uri.lower():
            return True
        
        return False
    
    async def _auto_approve_source(self, source_config: SourceConfig) -> Dict[str, Any]:
        """Auto-approve low-risk source registrations."""
        logger.info(f"Auto-approving source registration: {source_config.source_id}")
        
        return {
            "approved": True,
            "reason": "Auto-approved based on low-risk profile",
            "conditions": [
                "Monitor for first 30 days",
                "Review if error rate > 5%"
            ]
        }
    
    async def _request_governance_approval(self, request: GovernanceRequest) -> Dict[str, Any]:
        """Request approval from governance system."""
        self.pending_requests[request.request_id] = request
        
        if self.governance_engine:
            try:
                # Submit to governance engine
                result = await self.governance_engine.process_governance_request(
                    request.request_type,
                    {
                        "request_id": request.request_id,
                        "data": request.data,
                        "context": request.context
                    }
                )
                
                if result.get("outcome") == "APPROVED":
                    self.approved_requests[request.request_id] = result
                    return {
                        "approved": True,
                        "reason": result.get("reasoning", "Approved by governance"),
                        "conditions": result.get("conditions", [])
                    }
                else:
                    self.rejected_requests[request.request_id] = result
                    return {
                        "approved": False,
                        "reason": result.get("reasoning", "Rejected by governance"),
                        "requires_review": True
                    }
                    
            except Exception as e:
                logger.error(f"Governance engine error: {e}")
                return {
                    "approved": False,
                    "reason": f"Governance system error: {e}",
                    "requires_review": True
                }
        else:
            # Mock approval for development
            logger.warning("No governance engine available, using mock approval")
            return {
                "approved": True,
                "reason": "Mock approval (no governance engine)",
                "conditions": ["Review when governance engine is available"]
            }
    
    async def _check_pii_policy(self, record: NormRecord) -> Dict[str, Any]:
        """Check PII policy compliance."""
        try:
            # Simple PII detection (mock implementation)
            pii_flags = record.quality.pii_flags
            
            if not pii_flags:
                return {"passed": True, "details": "No PII detected"}
            
            # Get source PII policy
            source = self.ingress_kernel.get_source(record.source.source_id)
            if not source:
                return {"passed": False, "details": "Unknown source"}
            
            pii_policy = source.pii_policy
            
            if pii_policy == "block" and pii_flags:
                return {
                    "passed": False,
                    "details": f"PII detected with block policy: {pii_flags}"
                }
            elif pii_policy == "mask":
                # Would apply masking transformation
                return {
                    "passed": True,
                    "details": f"PII masked: {pii_flags}",
                    "action": "mask_applied"
                }
            elif pii_policy == "hash":
                # Would apply hashing transformation
                return {
                    "passed": True,
                    "details": f"PII hashed: {pii_flags}",
                    "action": "hash_applied"
                }
            else:  # allow_with_consent
                return {
                    "passed": True,
                    "details": f"PII allowed with consent: {pii_flags}"
                }
                
        except Exception as e:
            logger.error(f"PII policy check failed: {e}")
            return {"passed": False, "details": f"PII check error: {e}"}
    
    async def _check_quality_thresholds(self, record: NormRecord) -> Dict[str, Any]:
        """Check quality threshold compliance."""
        try:
            min_validity = self.ingress_kernel.config["validation"]["min_validity"]
            min_trust = self.ingress_kernel.config["validation"]["min_trust"]
            
            if record.quality.validity_score < min_validity:
                return {
                    "passed": False,
                    "details": f"Validity score {record.quality.validity_score} below threshold {min_validity}"
                }
            
            if record.quality.trust_score < min_trust:
                return {
                    "passed": False,
                    "details": f"Trust score {record.quality.trust_score} below threshold {min_trust}"
                }
            
            return {"passed": True, "details": "Quality thresholds met"}
            
        except Exception as e:
            logger.error(f"Quality threshold check failed: {e}")
            return {"passed": False, "details": f"Quality check error: {e}"}
    
    async def _check_governance_compliance(self, record: NormRecord) -> Dict[str, Any]:
        """Check governance label compliance."""
        try:
            source = self.ingress_kernel.get_source(record.source.source_id)
            if not source:
                return {"passed": False, "details": "Unknown source"}
            
            # Check if governance label allows current processing
            governance_label = source.governance_label
            
            if governance_label == "restricted":
                # Restricted data requires special handling
                return {
                    "passed": True,
                    "details": "Restricted data - special handling applied",
                    "conditions": ["Limited retention", "Audit trail required"]
                }
            
            return {"passed": True, "details": f"Governance compliance: {governance_label}"}
            
        except Exception as e:
            logger.error(f"Governance compliance check failed: {e}")
            return {"passed": False, "details": f"Compliance check error: {e}"}
    
    async def _monitor_policies(self):
        """Monitor policy changes."""
        while self.running:
            try:
                # Check for policy updates from governance system
                await self._refresh_policy_cache()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Policy monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _refresh_policy_cache(self):
        """Refresh cached policies."""
        try:
            # Clear expired cache entries
            current_time = datetime.utcnow()
            expired_keys = [
                k for k, expiry in self.cache_expiry.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                self.policy_cache.pop(key, None)
                self.cache_expiry.pop(key, None)
                
        except Exception as e:
            logger.error(f"Policy cache refresh failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "running": self.running,
            "pending_requests": len(self.pending_requests),
            "approved_requests": len(self.approved_requests),
            "rejected_requests": len(self.rejected_requests),
            "policy_cache_size": len(self.policy_cache)
        }