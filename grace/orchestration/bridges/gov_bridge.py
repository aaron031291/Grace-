"""
Grace Orchestration Governance Bridge - Integration with governance kernel.

Provides integration with the governance kernel for policy validation,
compliance checking, and governance-driven orchestration decisions.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class GovernanceBridge:
    """Bridge for integrating Orchestration Kernel with the Governance Kernel."""
    
    def __init__(self, governance_kernel=None):
        self.governance_kernel = governance_kernel
        
        # Validation requests and results
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.validation_results: List[Dict[str, Any]] = []
        
        # Configuration
        self.auto_validation_enabled = True
        self.policy_enforcement_enabled = True
        self.violation_threshold = 3  # Number of violations before action
        
        # Statistics
        self.total_validations = 0
        self.successful_validations = 0
        self.policy_violations = 0
        self.enforcement_actions = 0
        
        self.running = False
    
    async def start(self):
        """Start the governance bridge."""
        if self.running:
            return
        
        logger.info("Starting Orchestration-Governance Bridge...")
        self.running = True
        logger.info("Orchestration-Governance Bridge started")
    
    async def stop(self):
        """Stop the governance bridge."""
        logger.info("Stopping Orchestration-Governance Bridge...")
        self.running = False
    
    async def validate_orchestration_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an orchestration decision against governance policies."""
        request_id = f"gov_req_{int(datetime.now().timestamp() * 1000)}"
        
        validation_request = {
            "request_id": request_id,
            "context": decision_context,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self.pending_requests[request_id] = validation_request
        self.total_validations += 1
        
        try:
            logger.debug(f"Validating orchestration decision: {request_id}")
            
            # Perform governance validation
            result = await self._perform_governance_validation(decision_context)
            
            validation_request["status"] = "completed"
            validation_request["result"] = result
            
            # Store result
            self.validation_results.append({
                "request_id": request_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only recent results
            if len(self.validation_results) > 1000:
                self.validation_results = self.validation_results[-1000:]
            
            if result["compliant"]:
                self.successful_validations += 1
                logger.debug(f"Validation successful: {request_id}")
            else:
                self.policy_violations += 1
                logger.warning(f"Policy violations found: {result['violations']}")
                
                # Handle policy violations
                if self.policy_enforcement_enabled:
                    await self._handle_policy_violations(decision_context, result)
            
            return result
            
        except Exception as e:
            validation_request["status"] = "failed"
            validation_request["error"] = str(e)
            
            logger.error(f"Governance validation failed for {request_id}: {e}")
            return {
                "compliant": False,
                "violations": [{"message": f"Validation error: {e}"}],
                "error": str(e)
            }
        
        finally:
            # Clean up pending request
            self.pending_requests.pop(request_id, None)
    
    async def _perform_governance_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual governance validation."""
        if self.governance_kernel and hasattr(self.governance_kernel, 'validate_decision'):
            # Use actual governance kernel
            try:
                return await self.governance_kernel.validate_decision(context)
            except Exception as e:
                logger.error(f"Governance kernel validation error: {e}")
        
        # Fallback to basic validation
        return await self._basic_governance_validation(context)
    
    async def _basic_governance_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic governance validation (fallback when no governance kernel)."""
        violations = []
        
        # Basic validation rules
        decision_type = context.get("decision_type", "unknown")
        
        # Check for required fields
        required_fields = ["decision_type", "timestamp", "source"]
        for field in required_fields:
            if field not in context:
                violations.append({
                    "type": "missing_field",
                    "message": f"Required field missing: {field}",
                    "severity": "high"
                })
        
        # Validate decision type
        valid_decision_types = [
            "loop_execution", "task_dispatch", "scaling_action", 
            "rollback", "policy_update", "resource_allocation"
        ]
        if decision_type not in valid_decision_types:
            violations.append({
                "type": "invalid_decision_type",
                "message": f"Invalid decision type: {decision_type}",
                "severity": "medium"
            })
        
        # Check for sensitive operations
        if decision_type in ["rollback", "policy_update"]:
            if not context.get("authorization"):
                violations.append({
                    "type": "missing_authorization",
                    "message": f"Sensitive operation {decision_type} requires authorization",
                    "severity": "critical"
                })
        
        # Resource limits validation
        if decision_type == "scaling_action":
            requested_instances = context.get("requested_instances", 0)
            if requested_instances > 10:  # Example limit
                violations.append({
                    "type": "resource_limit_exceeded",
                    "message": f"Requested instances ({requested_instances}) exceeds limit (10)",
                    "severity": "high"
                })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "validated_at": datetime.now().isoformat(),
            "validator": "orchestration_governance_bridge"
        }
    
    async def _handle_policy_violations(self, context: Dict[str, Any], result: Dict[str, Any]):
        """Handle policy violations."""
        violations = result.get("violations", [])
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        
        if critical_violations:
            # Block the operation for critical violations
            self.enforcement_actions += 1
            logger.error(f"Blocking operation due to critical violations: {critical_violations}")
            
            # Could also:
            # - Send alert to administrators
            # - Create emergency snapshot
            # - Initiate emergency procedures
        
        elif len(violations) >= self.violation_threshold:
            # Take action for multiple violations
            self.enforcement_actions += 1
            logger.warning(f"Multiple violations detected, taking enforcement action")
            
            # Could:
            # - Reduce operation priority
            # - Require additional approval
            # - Log for audit
    
    async def validate_loop_execution(self, loop_id: str, loop_config: Dict[str, Any]) -> bool:
        """Validate loop execution against governance policies."""
        context = {
            "decision_type": "loop_execution",
            "loop_id": loop_id,
            "loop_config": loop_config,
            "timestamp": datetime.now().isoformat(),
            "source": "orchestration_scheduler"
        }
        
        result = await self.validate_orchestration_decision(context)
        return result.get("compliant", False)
    
    async def validate_task_dispatch(self, task_id: str, loop_id: str, inputs: Dict[str, Any]) -> bool:
        """Validate task dispatch against governance policies."""
        context = {
            "decision_type": "task_dispatch",
            "task_id": task_id,
            "loop_id": loop_id,
            "inputs": inputs,
            "timestamp": datetime.now().isoformat(),
            "source": "orchestration_scheduler"
        }
        
        result = await self.validate_orchestration_decision(context)
        return result.get("compliant", False)
    
    async def validate_scaling_action(self, action: str, loop_id: str, 
                                    current_instances: int, target_instances: int) -> bool:
        """Validate scaling actions against governance policies."""
        context = {
            "decision_type": "scaling_action",
            "action": action,
            "loop_id": loop_id,
            "current_instances": current_instances,
            "requested_instances": target_instances,
            "timestamp": datetime.now().isoformat(),
            "source": "orchestration_scaling_manager"
        }
        
        result = await self.validate_orchestration_decision(context)
        return result.get("compliant", False)
    
    async def validate_rollback_request(self, snapshot_id: str, reason: str,
                                      authorized_by: str = None) -> bool:
        """Validate rollback requests against governance policies."""
        context = {
            "decision_type": "rollback",
            "snapshot_id": snapshot_id,
            "reason": reason,
            "authorization": authorized_by,
            "timestamp": datetime.now().isoformat(),
            "source": "orchestration_snapshot_manager"
        }
        
        result = await self.validate_orchestration_decision(context)
        return result.get("compliant", False)
    
    async def report_governance_violation(self, violation: Dict[str, Any]):
        """Report a governance violation to the governance kernel."""
        try:
            if self.governance_kernel and hasattr(self.governance_kernel, 'record_violation'):
                await self.governance_kernel.record_violation(violation)
            else:
                # Log violation locally
                logger.warning(f"Governance violation reported: {violation}")
            
        except Exception as e:
            logger.error(f"Failed to report governance violation: {e}")
    
    async def get_governance_policies(self, policy_type: str = None) -> List[Dict[str, Any]]:
        """Get governance policies relevant to orchestration."""
        try:
            if self.governance_kernel and hasattr(self.governance_kernel, 'get_policies'):
                return await self.governance_kernel.get_policies(
                    scope="orchestration",
                    policy_type=policy_type
                )
            else:
                # Return basic policies
                return self._get_default_policies()
                
        except Exception as e:
            logger.error(f"Failed to get governance policies: {e}")
            return []
    
    def _get_default_policies(self) -> List[Dict[str, Any]]:
        """Get default governance policies for orchestration."""
        return [
            {
                "policy_id": "orch_resource_limits",
                "name": "Orchestration Resource Limits",
                "type": "resource",
                "rules": {
                    "max_instances_per_loop": 10,
                    "max_concurrent_tasks": 100,
                    "max_memory_per_instance": "4GB"
                }
            },
            {
                "policy_id": "orch_security_controls",
                "name": "Orchestration Security Controls",
                "type": "security",
                "rules": {
                    "require_authorization": ["rollback", "policy_update"],
                    "audit_sensitive_operations": True,
                    "encryption_required": True
                }
            },
            {
                "policy_id": "orch_operational_limits",
                "name": "Orchestration Operational Limits",
                "type": "operational",
                "rules": {
                    "max_rollbacks_per_day": 5,
                    "min_snapshot_interval_hours": 1,
                    "max_loop_execution_time_minutes": 60
                }
            }
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get governance bridge statistics."""
        success_rate = 0.0
        if self.total_validations > 0:
            success_rate = self.successful_validations / self.total_validations
        
        return {
            "running": self.running,
            "auto_validation_enabled": self.auto_validation_enabled,
            "policy_enforcement_enabled": self.policy_enforcement_enabled,
            "pending_requests": len(self.pending_requests),
            "statistics": {
                "total_validations": self.total_validations,
                "successful_validations": self.successful_validations,
                "policy_violations": self.policy_violations,
                "enforcement_actions": self.enforcement_actions,
                "success_rate": success_rate
            },
            "recent_results_count": len(self.validation_results),
            "connected_governance_kernel": self.governance_kernel is not None
        }