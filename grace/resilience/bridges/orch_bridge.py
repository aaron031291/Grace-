"""Bridge to Orchestration kernel for healing actions."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename

logger = logging.getLogger(__name__)


class OrchestrationBridge:
    """
    Bridge to Orchestration kernel for executing healing actions.
    
    Handles communication with the Grace Orchestration kernel for
    restarting services, reprovisioning resources, and coordinating
    recovery actions across the system.
    """
    
    def __init__(self, orchestration_client=None):
        """Initialize orchestration bridge."""
        self.orchestration_client = orchestration_client
        self.healing_actions: List[Dict] = []
        
        logger.debug("Orchestration bridge initialized")
    
    async def request_restart(self, service_id: str, reason: str = "resilience_healing") -> Dict[str, Any]:
        """
        Request service restart via orchestration.
        
        Args:
            service_id: Service to restart
            reason: Reason for restart request
            
        Returns:
            Result of restart operation
        """
        try:
            request = {
                "action": "restart",
                "target": service_id,
                "reason": reason,
                "requester": "resilience",
                "timestamp": iso_format()
            }
            
            if self.orchestration_client:
                result = await self.orchestration_client.restart_service(service_id, reason)
            else:
                # Simulate restart
                await asyncio.sleep(2)  # Simulate restart delay
                result = {
                    "status": "success",
                    "message": f"Service {service_id} restart initiated",
                    "estimated_completion": "30s"
                }
            
            # Record healing action
            healing_record = {
                **request,
                "result": result,
                "completed_at": iso_format()
            }
            self.healing_actions.append(healing_record)
            
            logger.info(f"Requested restart for service {service_id}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to request restart for service {service_id}: {e}")
            return {
                "status": "failed",
                "message": str(e)
            }
    
    async def request_reprovision(
        self, 
        service_id: str, 
        resource_type: str = "instance",
        reason: str = "resilience_healing"
    ) -> Dict[str, Any]:
        """
        Request resource reprovisioning via orchestration.
        
        Args:
            service_id: Service requiring reprovisioning
            resource_type: Type of resource to reprovision
            reason: Reason for reprovision request
            
        Returns:
            Result of reprovision operation
        """
        try:
            request = {
                "action": "reprovision",
                "target": service_id,
                "resource_type": resource_type,
                "reason": reason,
                "requester": "resilience",
                "timestamp": iso_format()
            }
            
            if self.orchestration_client:
                result = await self.orchestration_client.reprovision_resource(
                    service_id, resource_type, reason
                )
            else:
                # Simulate reprovision
                await asyncio.sleep(5)  # Simulate reprovision delay
                result = {
                    "status": "success",
                    "message": f"Resource reprovisioning initiated for {service_id}",
                    "estimated_completion": "5m",
                    "new_instances": 1
                }
            
            # Record healing action
            healing_record = {
                **request,
                "result": result,
                "completed_at": iso_format()
            }
            self.healing_actions.append(healing_record)
            
            logger.info(f"Requested reprovision for service {service_id}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to request reprovision for service {service_id}: {e}")
            return {
                "status": "failed",
                "message": str(e)
            }
    
    async def request_scale(
        self, 
        service_id: str, 
        target_instances: int,
        reason: str = "resilience_scaling"
    ) -> Dict[str, Any]:
        """
        Request service scaling via orchestration.
        
        Args:
            service_id: Service to scale
            target_instances: Target number of instances
            reason: Reason for scaling request
            
        Returns:
            Result of scaling operation
        """
        try:
            request = {
                "action": "scale",
                "target": service_id,
                "target_instances": target_instances,
                "reason": reason,
                "requester": "resilience",
                "timestamp": iso_format()
            }
            
            if self.orchestration_client:
                result = await self.orchestration_client.scale_service(
                    service_id, target_instances, reason
                )
            else:
                # Simulate scaling
                await asyncio.sleep(1)
                result = {
                    "status": "success",
                    "message": f"Scaling {service_id} to {target_instances} instances",
                    "current_instances": target_instances,
                    "estimated_completion": "2m"
                }
            
            # Record healing action
            healing_record = {
                **request,
                "result": result,
                "completed_at": iso_format()
            }
            self.healing_actions.append(healing_record)
            
            logger.info(f"Requested scaling for service {service_id} to {target_instances}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to request scaling for service {service_id}: {e}")
            return {
                "status": "failed",
                "message": str(e)
            }
    
    async def request_isolate(
        self, 
        service_id: str,
        isolation_type: str = "network",
        reason: str = "resilience_isolation"
    ) -> Dict[str, Any]:
        """
        Request service isolation via orchestration.
        
        Args:
            service_id: Service to isolate
            isolation_type: Type of isolation (network, process, container)
            reason: Reason for isolation request
            
        Returns:
            Result of isolation operation
        """
        try:
            request = {
                "action": "isolate",
                "target": service_id,
                "isolation_type": isolation_type,
                "reason": reason,
                "requester": "resilience",
                "timestamp": iso_format()
            }
            
            if self.orchestration_client:
                result = await self.orchestration_client.isolate_service(
                    service_id, isolation_type, reason
                )
            else:
                # Simulate isolation
                result = {
                    "status": "success",
                    "message": f"Service {service_id} isolated ({isolation_type})",
                    "isolation_id": f"iso_{service_id}_{int(utc_now().timestamp())}"
                }
            
            # Record healing action
            healing_record = {
                **request,
                "result": result,
                "completed_at": iso_format()
            }
            self.healing_actions.append(healing_record)
            
            logger.info(f"Requested isolation for service {service_id}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to request isolation for service {service_id}: {e}")
            return {
                "status": "failed",
                "message": str(e)
            }
    
    async def request_health_check(self, service_id: str) -> Dict[str, Any]:
        """
        Request health check via orchestration.
        
        Args:
            service_id: Service to health check
            
        Returns:
            Health check result
        """
        try:
            if self.orchestration_client:
                result = await self.orchestration_client.health_check(service_id)
            else:
                # Simulate health check
                result = {
                    "service_id": service_id,
                    "status": "healthy",
                    "instances": {
                        "total": 3,
                        "healthy": 3,
                        "unhealthy": 0
                    },
                    "last_check": iso_format()
                }
            
            logger.debug(f"Health check for service {service_id}: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to health check service {service_id}: {e}")
            return {
                "service_id": service_id,
                "status": "error",
                "message": str(e)
            }
    
    async def execute_playbook_step(
        self, 
        service_id: str, 
        step: Dict[str, Any],
        incident_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a playbook step via orchestration.
        
        Args:
            service_id: Target service
            step: Playbook step definition
            incident_id: Related incident ID
            
        Returns:
            Step execution result
        """
        try:
            actions = step.get("do", [])
            results = []
            
            for action in actions:
                if action == "restart":
                    result = await self.request_restart(service_id, f"playbook_step_{incident_id}")
                elif action == "reprovision":
                    result = await self.request_reprovision(service_id, "instance", f"playbook_step_{incident_id}")
                elif action.startswith("scale_"):
                    # Extract target from action like "scale_5"
                    target = int(action.split("_")[1]) if "_" in action else 3
                    result = await self.request_scale(service_id, target, f"playbook_step_{incident_id}")
                else:
                    result = {
                        "status": "skipped",
                        "message": f"Unknown action: {action}"
                    }
                
                results.append({"action": action, "result": result})
            
            overall_success = all(r["result"]["status"] == "success" for r in results)
            
            return {
                "step_status": "success" if overall_success else "partial_failure",
                "actions_executed": len(results),
                "results": results,
                "executed_at": iso_format()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute playbook step for service {service_id}: {e}")
            return {
                "step_status": "failed",
                "message": str(e)
            }
    
    def get_healing_history(self, service_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get healing action history."""
        history = self.healing_actions
        
        if service_id:
            history = [h for h in history if h.get("target") == service_id]
        
        # Trim history
        if len(self.healing_actions) > 1000:
            self.healing_actions = self.healing_actions[-500:]
        
        return history[-limit:] if limit else history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestration bridge statistics."""
        total_actions = len(self.healing_actions)
        successful_actions = len([h for h in self.healing_actions if h.get("result", {}).get("status") == "success"])
        
        action_counts = {}
        for action in self.healing_actions:
            action_type = action.get("action", "unknown")
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return {
            "total_healing_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate_pct": (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            "action_counts": action_counts,
            "generated_at": iso_format()
        }