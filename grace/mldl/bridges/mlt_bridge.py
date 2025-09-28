"""
MLDL-MLT Bridge - Connects MLDL Kernel to MLT (Meta-Learning Tuning) Kernel.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

logger = logging.getLogger(__name__)


class MLDLMLTBridge:
    """Bridge between MLDL Kernel and MLT Kernel for experience sharing and adaptation."""
    
    def __init__(self, mlt_kernel=None, event_bus=None):
        self.mlt_kernel = mlt_kernel
        self.event_bus = event_bus
        
        logger.info("MLDL MLT Bridge initialized")
    
    async def send_training_experience(self, job_result: Dict[str, Any]) -> str:
        """Send training experience to MLT kernel."""
        try:
            # Extract training metrics for MLT
            metrics = job_result.get("metrics", {})
            training_params = job_result.get("training_params", {})
            
            experience = {
                "exp_id": f"mldl_train_{uuid.uuid4().hex[:8]}",
                "stage": "train",
                "metrics": {
                    "cv_score": metrics.get("cv_score", 0.0),
                    "trials": metrics.get("trials", 1),
                    "cost_units": training_params.get("compute_cost", 0.0)
                },
                "context": {
                    "model_family": job_result.get("model_family", "unknown"),
                    "task_type": job_result.get("task_type", "unknown"),
                    "dataset_size": training_params.get("dataset_size", 0),
                    "hyperparams": training_params.get("best_params", {})
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to MLT kernel
            if self.mlt_kernel:
                exp_id = await self.mlt_kernel.submit_experience("training", experience)
                logger.info(f"Sent training experience {exp_id} to MLT")
                return exp_id
            else:
                logger.warning("MLT kernel not available for experience submission")
                return f"mock_exp_{uuid.uuid4().hex[:8]}"
                
        except Exception as e:
            logger.error(f"Failed to send training experience: {e}")
            raise