"""
Deployment Manager - handles canary/shadow deployments, blue/green, promotion rules.
"""
import asyncio
import logging
import sqlite3
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    CANARY = "canary"
    SHADOW = "shadow"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"
    FAILED = "failed"


class DeploymentManager:
    """Manages model deployments with canary, shadow, and blue/green strategies."""
    
    def __init__(self, event_bus=None, governance_bridge=None, db_path: str = ":memory:"):
        self.event_bus = event_bus
        self.governance_bridge = governance_bridge
        self.db_path = db_path
        self.conn = None
        
        # Deployment configuration
        self.default_canary_steps = [5, 25, 50, 100]
        self.default_promotion_window = 3600  # 1 hour in seconds
        self.max_concurrent_deployments = 5
        
        # Active deployment tracking
        self.active_deployments = {}
        self.deployment_tasks = {}
        
        # Initialize database
        self._initialize_db()
        
        logger.info("Deployment Manager initialized")
    
    def _initialize_db(self):
        """Initialize SQLite database for deployment tracking."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create deployments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT UNIQUE NOT NULL,
                model_key TEXT NOT NULL,
                version TEXT NOT NULL,
                environment TEXT NOT NULL,
                status TEXT NOT NULL,
                deployment_spec TEXT,  -- JSON string
                canary_pct INTEGER DEFAULT 0,
                shadow_mode BOOLEAN DEFAULT FALSE,
                started_at TEXT NOT NULL,
                updated_at TEXT,
                completed_at TEXT,
                rollback_reason TEXT,
                metadata TEXT  -- JSON string
            )
        """)
        
        # Create canary progress table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS canary_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                target_pct INTEGER NOT NULL,
                actual_pct INTEGER,
                success BOOLEAN,
                metrics TEXT,  -- JSON string
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
            )
        """)
        
        # Create rollback history table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rollback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                rollback_id TEXT NOT NULL,
                reason TEXT,
                triggered_by TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
            )
        """)
        
        self.conn.commit()
        logger.info("Deployment database initialized")
    
    async def request(self, model_key: str, version: str, deployment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request a new model deployment.
        
        Args:
            model_key: Model identifier
            version: Model version
            deployment_spec: DeploymentSpec dictionary
            
        Returns:
            Deployment information
        """
        try:
            # Check if governance approval is required
            if self.governance_bridge:
                approval_required = await self._check_governance_approval(
                    model_key, version, deployment_spec
                )
                if not approval_required["approved"]:
                    raise ValueError(f"Governance approval required: {approval_required['reason']}")
            
            # Generate deployment ID
            deployment_id = f"deploy_{uuid.uuid4().hex[:12]}"
            
            # Extract deployment parameters
            target_env = deployment_spec.get("target_env", "staging")
            canary_pct = deployment_spec.get("canary_pct", 5)
            shadow = deployment_spec.get("shadow", False)
            guardrails = deployment_spec.get("guardrails", {})
            
            # Validate deployment request
            validation_result = await self._validate_deployment_request(
                model_key, version, deployment_spec
            )
            if not validation_result["valid"]:
                raise ValueError(f"Invalid deployment request: {validation_result['reason']}")
            
            # Create deployment record
            self.conn.execute("""
                INSERT INTO deployments (
                    deployment_id, model_key, version, environment, status,
                    deployment_spec, canary_pct, shadow_mode, started_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deployment_id, model_key, version, target_env,
                DeploymentStatus.CANARY.value if not shadow else DeploymentStatus.SHADOW.value,
                json.dumps(deployment_spec), canary_pct, shadow,
                iso_format(),
                json.dumps({"guardrails": guardrails})
            ))
            
            self.conn.commit()
            
            # Track active deployment
            deployment_info = {
                "deployment_id": deployment_id,
                "model_key": model_key,
                "version": version,
                "environment": target_env,
                "status": DeploymentStatus.SHADOW.value if shadow else DeploymentStatus.CANARY.value,
                "canary_pct": canary_pct if not shadow else 0,
                "shadow_mode": shadow,
                "started_at": iso_format()
            }
            
            self.active_deployments[deployment_id] = deployment_info
            
            # Start deployment process
            if shadow:
                task = asyncio.create_task(self._run_shadow_deployment(deployment_id))
            else:
                task = asyncio.create_task(self._run_canary_deployment(deployment_id))
            
            self.deployment_tasks[deployment_id] = task
            
            # Publish deployment event
            if self.event_bus:
                await self.event_bus.publish("MLDL_DEPLOYMENT_REQUESTED", {
                    "model_key": model_key,
                    "version": version,
                    "spec": deployment_spec
                })
            
            logger.info(f"Deployment {deployment_id} requested for {model_key}@{version}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Deployment request failed: {e}")
            raise
    
    async def promote(self, deployment_id: str, target_pct: Optional[int] = None) -> Dict[str, Any]:
        """
        Promote a canary deployment to next stage or full production.
        
        Args:
            deployment_id: Deployment identifier
            target_pct: Target percentage (optional)
            
        Returns:
            Promotion result
        """
        try:
            # Get deployment info
            deployment = await self.get_deployment(deployment_id)
            if not deployment:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            if deployment["status"] not in [DeploymentStatus.CANARY.value, DeploymentStatus.ACTIVE.value]:
                raise ValueError(f"Cannot promote deployment with status {deployment['status']}")
            
            # Check promotion eligibility
            eligibility = await self._check_promotion_eligibility(deployment_id)
            if not eligibility["eligible"]:
                raise ValueError(f"Promotion not eligible: {eligibility['reason']}")
            
            # Determine next canary percentage
            current_pct = deployment["canary_pct"]
            if target_pct is not None:
                next_pct = target_pct
            else:
                next_pct = self._get_next_canary_step(current_pct)
            
            # Update deployment
            self.conn.execute("""
                UPDATE deployments 
                SET canary_pct = ?, updated_at = ?
                WHERE deployment_id = ?
            """, (next_pct, iso_format(), deployment_id))
            
            # Record canary progress
            self.conn.execute("""
                INSERT INTO canary_progress (
                    deployment_id, step, target_pct, success, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                deployment_id, 
                len(self.default_canary_steps), 
                next_pct, 
                True,
                iso_format()
            ))
            
            self.conn.commit()
            
            # Update active deployment tracking
            if deployment_id in self.active_deployments:
                self.active_deployments[deployment_id]["canary_pct"] = next_pct
                if next_pct >= 100:
                    self.active_deployments[deployment_id]["status"] = DeploymentStatus.ACTIVE.value
            
            # If promoted to 100%, mark as fully active
            if next_pct >= 100:
                await self._complete_deployment(deployment_id)
            
            # Publish promotion event
            if self.event_bus:
                await self.event_bus.publish("MLDL_DEPLOYMENT_PROMOTED", {
                    "deployment": self.active_deployments.get(deployment_id, {})
                })
            
            logger.info(f"Deployment {deployment_id} promoted to {next_pct}%")
            
            return {
                "deployment_id": deployment_id,
                "previous_pct": current_pct,
                "current_pct": next_pct,
                "status": "promoted",
                "promoted_at": iso_format()
            }
            
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            raise
    
    async def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment information."""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM deployments WHERE deployment_id = ?
            """, (deployment_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            deployment_data = dict(row)
            
            # Parse JSON fields
            if deployment_data["deployment_spec"]:
                deployment_data["deployment_spec"] = json.loads(deployment_data["deployment_spec"])
            if deployment_data["metadata"]:
                deployment_data["metadata"] = json.loads(deployment_data["metadata"])
            
            # Get canary progress
            deployment_data["canary_progress"] = await self._get_canary_progress(deployment_id)
            
            return deployment_data
            
        except Exception as e:
            logger.error(f"Failed to get deployment {deployment_id}: {e}")
            return None
    
    async def rollback(self, deployment_id: str, reason: str = None, triggered_by: str = None) -> Dict[str, Any]:
        """
        Rollback a deployment.
        
        Args:
            deployment_id: Deployment identifier
            reason: Rollback reason
            triggered_by: Who/what triggered the rollback
            
        Returns:
            Rollback result
        """
        try:
            # Get deployment info
            deployment = await self.get_deployment(deployment_id)
            if not deployment:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            rollback_id = f"rollback_{uuid.uuid4().hex[:8]}"
            
            # Update deployment status
            self.conn.execute("""
                UPDATE deployments 
                SET status = ?, rollback_reason = ?, updated_at = ?, completed_at = ?
                WHERE deployment_id = ?
            """, (
                DeploymentStatus.ROLLED_BACK.value,
                reason,
                iso_format(),
                iso_format(),
                deployment_id
            ))
            
            # Record rollback
            self.conn.execute("""
                INSERT INTO rollback_history (
                    deployment_id, rollback_id, reason, triggered_by, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                deployment_id, rollback_id, reason, triggered_by,
                iso_format()
            ))
            
            self.conn.commit()
            
            # Update active deployment tracking
            if deployment_id in self.active_deployments:
                self.active_deployments[deployment_id]["status"] = DeploymentStatus.ROLLED_BACK.value
            
            # Cancel deployment task if running
            if deployment_id in self.deployment_tasks:
                self.deployment_tasks[deployment_id].cancel()
                del self.deployment_tasks[deployment_id]
            
            # Publish rollback event
            if self.event_bus:
                await self.event_bus.publish("ROLLBACK_REQUESTED", {
                    "target": "mldl",
                    "to_snapshot": rollback_id
                })
            
            logger.info(f"Deployment {deployment_id} rolled back: {reason}")
            
            return {
                "deployment_id": deployment_id,
                "rollback_id": rollback_id,
                "reason": reason,
                "triggered_by": triggered_by,
                "rolled_back_at": iso_format()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    async def _validate_deployment_request(self, model_key: str, version: str, 
                                         deployment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate deployment request."""
        try:
            # Check if model exists (would integrate with registry)
            # For now, assume valid
            
            # Check environment
            target_env = deployment_spec.get("target_env", "staging")
            if target_env not in ["staging", "prod"]:
                return {"valid": False, "reason": f"Invalid environment: {target_env}"}
            
            # Check canary percentage
            canary_pct = deployment_spec.get("canary_pct", 5)
            if not 0 <= canary_pct <= 100:
                return {"valid": False, "reason": f"Invalid canary percentage: {canary_pct}"}
            
            # Check concurrent deployments
            active_count = len([d for d in self.active_deployments.values() 
                              if d["status"] in [DeploymentStatus.CANARY.value, DeploymentStatus.SHADOW.value]])
            
            if active_count >= self.max_concurrent_deployments:
                return {"valid": False, "reason": f"Too many active deployments: {active_count}"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "reason": str(e)}
    
    async def _check_governance_approval(self, model_key: str, version: str, 
                                       deployment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Check if governance approval is required and obtained."""
        try:
            # Check if production deployment
            if deployment_spec.get("target_env") == "prod":
                # For production, require governance approval
                if self.governance_bridge:
                    # Would integrate with governance system
                    # For now, mock approval
                    return {"approved": True, "approver": "governance_system"}
                else:
                    return {"approved": False, "reason": "Governance approval required for production"}
            
            return {"approved": True}
            
        except Exception as e:
            return {"approved": False, "reason": str(e)}
    
    async def _run_canary_deployment(self, deployment_id: str):
        """Run canary deployment process."""
        try:
            deployment = await self.get_deployment(deployment_id)
            if not deployment:
                return
            
            canary_steps = deployment.get("deployment_spec", {}).get("canary_steps", self.default_canary_steps)
            promotion_window = deployment.get("deployment_spec", {}).get("promotion_window", self.default_promotion_window)
            
            for step, target_pct in enumerate(canary_steps):
                if target_pct <= deployment["canary_pct"]:
                    continue  # Already at or past this step
                
                logger.info(f"Canary deployment {deployment_id} step {step}: {target_pct}%")
                
                # Simulate canary validation
                await asyncio.sleep(30)  # Wait for metrics
                
                # Check metrics and guardrails
                metrics_ok = await self._check_canary_metrics(deployment_id)
                
                if metrics_ok:
                    # Record successful step
                    self.conn.execute("""
                        INSERT INTO canary_progress (
                            deployment_id, step, target_pct, actual_pct, success, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        deployment_id, step, target_pct, target_pct, True,
                        iso_format()
                    ))
                    
                    # Update deployment percentage
                    self.conn.execute("""
                        UPDATE deployments 
                        SET canary_pct = ?, updated_at = ?
                        WHERE deployment_id = ?
                    """, (target_pct, iso_format(), deployment_id))
                    
                    self.conn.commit()
                    
                    # Publish progress event
                    if self.event_bus:
                        await self.event_bus.publish("MLDL_CANARY_PROGRESS", {
                            "deployment_id": deployment_id,
                            "step": step,
                            "pass": True,
                            "notes": f"Successfully promoted to {target_pct}%"
                        })
                    
                    # Wait before next step (unless it's the final step)
                    if target_pct < 100:
                        await asyncio.sleep(promotion_window)
                    
                else:
                    # Rollback on failure
                    await self.rollback(deployment_id, "Canary metrics failed", "automated")
                    return
            
            # Complete deployment
            await self._complete_deployment(deployment_id)
            
        except Exception as e:
            logger.error(f"Canary deployment {deployment_id} failed: {e}")
            await self.rollback(deployment_id, str(e), "automated")
    
    async def _run_shadow_deployment(self, deployment_id: str):
        """Run shadow deployment process."""
        try:
            deployment = await self.get_deployment(deployment_id)
            if not deployment:
                return
            
            logger.info(f"Shadow deployment {deployment_id} started")
            
            # Shadow deployments run for a fixed period
            shadow_duration = deployment.get("deployment_spec", {}).get("shadow_duration", 3600)  # 1 hour
            
            # Monitor shadow traffic for the duration
            await asyncio.sleep(shadow_duration)
            
            # Check shadow metrics
            metrics_ok = await self._check_shadow_metrics(deployment_id)
            
            if metrics_ok:
                # Shadow completed successfully
                self.conn.execute("""
                    UPDATE deployments 
                    SET status = ?, completed_at = ?
                    WHERE deployment_id = ?
                """, (DeploymentStatus.ACTIVE.value, iso_format(), deployment_id))
                
                self.conn.commit()
                
                logger.info(f"Shadow deployment {deployment_id} completed successfully")
            else:
                await self.rollback(deployment_id, "Shadow metrics failed", "automated")
            
        except Exception as e:
            logger.error(f"Shadow deployment {deployment_id} failed: {e}")
            await self.rollback(deployment_id, str(e), "automated")
    
    async def _check_canary_metrics(self, deployment_id: str) -> bool:
        """Check canary metrics against guardrails."""
        # Mock metric checking - real implementation would integrate with monitoring
        return True  # Assume success for now
    
    async def _check_shadow_metrics(self, deployment_id: str) -> bool:
        """Check shadow metrics against guardrails."""
        # Mock metric checking - real implementation would integrate with monitoring
        return True  # Assume success for now
    
    async def _check_promotion_eligibility(self, deployment_id: str) -> Dict[str, Any]:
        """Check if deployment is eligible for promotion."""
        deployment = await self.get_deployment(deployment_id)
        if not deployment:
            return {"eligible": False, "reason": "Deployment not found"}
        
        # Check if enough time has passed since last promotion
        if deployment["updated_at"]:
            last_update = datetime.fromisoformat(deployment["updated_at"])
            if utc_now() - last_update < timedelta(minutes=10):
                return {"eligible": False, "reason": "Insufficient time since last promotion"}
        
        return {"eligible": True}
    
    def _get_next_canary_step(self, current_pct: int) -> int:
        """Get next canary percentage step."""
        for step_pct in self.default_canary_steps:
            if step_pct > current_pct:
                return step_pct
        return 100  # Full deployment
    
    async def _complete_deployment(self, deployment_id: str):
        """Mark deployment as completed."""
        self.conn.execute("""
            UPDATE deployments 
            SET status = ?, completed_at = ?
            WHERE deployment_id = ?
        """, (DeploymentStatus.ACTIVE.value, iso_format(), deployment_id))
        
        self.conn.commit()
        
        # Clean up active tracking
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id]["status"] = DeploymentStatus.ACTIVE.value
        
        if deployment_id in self.deployment_tasks:
            del self.deployment_tasks[deployment_id]
        
        logger.info(f"Deployment {deployment_id} completed")
    
    async def _get_canary_progress(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Get canary progress history."""
        try:
            cursor = self.conn.execute("""
                SELECT * FROM canary_progress 
                WHERE deployment_id = ? 
                ORDER BY created_at ASC
            """, (deployment_id,))
            
            progress = []
            for row in cursor.fetchall():
                progress_data = dict(row)
                if progress_data["metrics"]:
                    progress_data["metrics"] = json.loads(progress_data["metrics"])
                progress.append(progress_data)
            
            return progress
            
        except Exception as e:
            logger.warning(f"Failed to get canary progress: {e}")
            return []
    
    def close(self):
        """Close database connection and cleanup."""
        # Cancel active deployment tasks
        for task in self.deployment_tasks.values():
            task.cancel()
        
        if self.conn:
            self.conn.close()
            logger.info("Deployment manager database connection closed")