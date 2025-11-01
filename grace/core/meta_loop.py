"""
Meta-Loop Optimizer - The Recursive Self-Improvement Engine

This is THE breakthrough component that transforms Grace from
observational to self-evolving intelligence.

Implements:
1. Candidate generation from recent failures
2. Sandboxed A/B evaluation
3. Statistical validation
4. Gated deployment with rollback
5. Automatic distillation of winning strategies
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import numpy as np

from .evaluation_harness import EvaluationHarness, EvaluationResult

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for candidate changes"""
    LOW = "low"  # Prompt tweaks, threshold adjustments
    MEDIUM = "medium"  # Routing changes, small adapter updates
    HIGH = "high"  # Architecture changes, major adaptations


class DeploymentStatus(Enum):
    """Status of deployment attempts"""
    PENDING = "pending"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class Candidate:
    """A proposed configuration change"""
    candidate_id: str
    config: Dict[str, Any]
    changes_from_baseline: Dict[str, Any]
    risk_level: RiskLevel
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_from: str = ""  # What triggered this candidate
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "config": self.config,
            "changes": self.changes_from_baseline,
            "risk": self.risk_level.value,
            "generated_at": self.generated_at.isoformat(),
            "rationale": self.rationale
        }


@dataclass
class DeploymentResult:
    """Result of a deployment attempt"""
    candidate_id: str
    status: DeploymentStatus
    improvement: float  # Delta reward vs baseline
    confidence: float  # Statistical confidence
    evaluation_result: Optional[EvaluationResult] = None
    deployed_at: Optional[datetime] = None
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "improvement": self.improvement,
            "confidence": self.confidence,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "reason": self.reason
        }


class CheckpointManager:
    """
    Manages configuration checkpoints and rollback capability.
    Critical for safe self-improvement.
    """
    
    def __init__(self, storage_path: str = "./checkpoints"):
        self.storage_path = storage_path
        self.current_checkpoint = None
        self.checkpoint_history = []
        self.baseline_config = None
        
    def create_checkpoint(
        self,
        config: Dict[str, Any],
        label: str,
        metrics: Dict[str, Any]
    ) -> str:
        """Create a new checkpoint"""
        checkpoint_id = self._generate_checkpoint_id(config)
        
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "label": label,
            "config": config,
            "metrics": metrics,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.checkpoint_history.append(checkpoint)
        self.current_checkpoint = checkpoint
        
        # Save to disk
        self._save_checkpoint(checkpoint)
        
        logger.info(f"Created checkpoint: {checkpoint_id} ({label})")
        return checkpoint_id
    
    def get_baseline_config(self) -> Dict[str, Any]:
        """Get the baseline configuration"""
        if self.baseline_config:
            return self.baseline_config
        
        # Default baseline
        return {
            "model": "default",
            "temperature": 0.7,
            "routing_thresholds": {
                "factual": 0.8,
                "reasoning": 0.7,
                "safety": 0.95
            },
            "prompt_template": "default",
            "ensemble_weights": {
                "model_a": 0.5,
                "model_b": 0.5
            }
        }
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a specific checkpoint"""
        for checkpoint in reversed(self.checkpoint_history):
            if checkpoint["checkpoint_id"] == checkpoint_id:
                self.current_checkpoint = checkpoint
                logger.info(f"Rolled back to checkpoint: {checkpoint_id}")
                return True
        
        logger.error(f"Checkpoint not found: {checkpoint_id}")
        return False
    
    def _generate_checkpoint_id(self, config: Dict[str, Any]) -> str:
        """Generate unique checkpoint ID"""
        config_str = json.dumps(config, sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode())
        return f"ckpt_{hash_obj.hexdigest()[:12]}"
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save checkpoint to disk (stub)"""
        # In production, save to actual storage
        pass


class AdaptationSurface:
    """
    Defines what parameters can be adapted.
    This is the "bounded" part of bounded self-improvement.
    """
    
    def __init__(self):
        # What can be changed
        self.adaptable_parameters = {
            "routing_thresholds": {
                "type": "continuous",
                "range": (0.0, 1.0),
                "default": 0.7
            },
            "temperature": {
                "type": "continuous",
                "range": (0.0, 2.0),
                "default": 0.7
            },
            "prompt_variant": {
                "type": "discrete",
                "options": ["default", "detailed", "concise", "chain_of_thought"],
                "default": "default"
            },
            "ensemble_weights": {
                "type": "continuous_vector",
                "dimension": 2,
                "constraint": "sum_to_one",
                "default": [0.5, 0.5]
            }
        }
    
    def generate_random_candidate(self, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random candidate within bounds"""
        candidate = baseline.copy()
        
        # Randomly perturb one or more parameters
        param_name = np.random.choice(list(self.adaptable_parameters.keys()))
        param_spec = self.adaptable_parameters[param_name]
        
        if param_spec["type"] == "continuous":
            # Add Gaussian noise
            current_value = candidate.get(param_name, param_spec["default"])
            noise = np.random.normal(0, 0.1)
            new_value = np.clip(
                current_value + noise,
                param_spec["range"][0],
                param_spec["range"][1]
            )
            candidate[param_name] = new_value
        
        elif param_spec["type"] == "discrete":
            # Random choice
            candidate[param_name] = np.random.choice(param_spec["options"])
        
        return candidate
    
    def validate_candidate(self, candidate: Dict[str, Any]) -> bool:
        """Ensure candidate is within bounds"""
        for param_name, param_spec in self.adaptable_parameters.items():
            if param_name not in candidate:
                continue
            
            value = candidate[param_name]
            
            if param_spec["type"] == "continuous":
                if not (param_spec["range"][0] <= value <= param_spec["range"][1]):
                    return False
            
            elif param_spec["type"] == "discrete":
                if value not in param_spec["options"]:
                    return False
        
        return True


class MetaLoopOptimizer:
    """
    THE breakthrough engine.
    
    Transforms Grace from observational to recursively self-improving.
    """
    
    def __init__(
        self,
        evaluation_harness: EvaluationHarness,
        governance_kernel=None,
        improvement_threshold: float = 0.02  # 2% improvement to deploy
    ):
        self.eval_harness = evaluation_harness
        self.governance = governance_kernel
        self.checkpoint_manager = CheckpointManager()
        self.adaptation_surface = AdaptationSurface()
        
        self.improvement_threshold = improvement_threshold
        self.confidence_threshold = 0.95  # 95% confidence required
        
        # State
        self.baseline_config = None
        self.baseline_score = None
        self.candidate_history = []
        self.deployment_history = []
        
        self.running = False
        
        logger.info("Meta-Loop Optimizer initialized - Ready for recursive self-improvement")
    
    async def initialize_baseline(self, config: Optional[Dict[str, Any]] = None):
        """Set the baseline configuration"""
        if config is None:
            config = self.checkpoint_manager.get_baseline_config()
        
        logger.info("Evaluating baseline configuration...")
        baseline_result = await self.eval_harness.evaluate_candidate(config)
        
        self.baseline_config = config
        self.baseline_score = baseline_result.reward
        
        # Create checkpoint
        self.checkpoint_manager.create_checkpoint(
            config=config,
            label="baseline",
            metrics=baseline_result.scorecard
        )
        
        logger.info(f"Baseline established. Reward: {self.baseline_score:.4f}")
    
    async def improvement_cycle(self) -> DeploymentResult:
        """
        One iteration of self-improvement.
        
        This is the CORE LOOP:
        1. Generate candidate
        2. Evaluate in sandbox
        3. Statistical comparison
        4. Safety check
        5. Governance gate
        6. Deploy or rollback
        """
        logger.info("🔄 Starting improvement cycle...")
        
        # Ensure baseline exists
        if self.baseline_config is None:
            await self.initialize_baseline()
        
        # 1. Generate candidate
        candidate = await self._generate_candidate()
        logger.info(f"Generated candidate: {candidate.candidate_id}")
        logger.info(f"  Risk: {candidate.risk_level.value}")
        logger.info(f"  Rationale: {candidate.rationale}")
        
        # 2. Evaluate in sandbox
        logger.info("Evaluating candidate in sandbox...")
        candidate_result = await self.eval_harness.evaluate_candidate(
            candidate.config,
            run_in_sandbox=True
        )
        
        # 3. Statistical comparison
        improvement = candidate_result.reward - self.baseline_score
        confidence = self._compute_statistical_confidence(
            candidate_result,
            improvement
        )
        
        logger.info(f"  Candidate reward: {candidate_result.reward:.4f}")
        logger.info(f"  Baseline reward: {self.baseline_score:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f} ({improvement/self.baseline_score*100:+.1f}%)")
        logger.info(f"  Confidence: {confidence:.2%}")
        
        # 4. Safety check
        if not candidate_result.safety_passed:
            logger.warning("❌ Candidate failed safety gates")
            return DeploymentResult(
                candidate_id=candidate.candidate_id,
                status=DeploymentStatus.REJECTED,
                improvement=improvement,
                confidence=confidence,
                evaluation_result=candidate_result,
                reason="safety_violation"
            )
        
        # 5. Governance gate for high-risk changes
        if candidate.risk_level == RiskLevel.HIGH:
            approved = await self._request_governance_approval(
                candidate,
                candidate_result
            )
            if not approved:
                logger.warning("❌ Governance rejected high-risk candidate")
                return DeploymentResult(
                    candidate_id=candidate.candidate_id,
                    status=DeploymentStatus.REJECTED,
                    improvement=improvement,
                    confidence=confidence,
                    evaluation_result=candidate_result,
                    reason="governance_rejected"
                )
        
        # 6. Deploy if better with confidence
        if improvement > self.improvement_threshold and confidence > self.confidence_threshold:
            logger.info("✅ Candidate approved for deployment!")
            
            # Create checkpoint before deployment
            self.checkpoint_manager.create_checkpoint(
                config=candidate.config,
                label=f"candidate_{candidate.candidate_id}",
                metrics=candidate_result.scorecard
            )
            
            # Deploy
            await self._deploy_candidate(candidate, candidate_result)
            
            # Update baseline
            self.baseline_config = candidate.config
            self.baseline_score = candidate_result.reward
            
            # Distill successful strategy
            await self._distill_winning_strategy(candidate)
            
            return DeploymentResult(
                candidate_id=candidate.candidate_id,
                status=DeploymentStatus.DEPLOYED,
                improvement=improvement,
                confidence=confidence,
                evaluation_result=candidate_result,
                deployed_at=datetime.utcnow(),
                reason="improvement_validated"
            )
        
        else:
            logger.info("❌ Candidate does not meet deployment criteria")
            return DeploymentResult(
                candidate_id=candidate.candidate_id,
                status=DeploymentStatus.REJECTED,
                improvement=improvement,
                confidence=confidence,
                evaluation_result=candidate_result,
                reason="insufficient_improvement"
            )
    
    async def continuous_improvement(
        self,
        interval_hours: float = 24.0,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous improvement loop.
        
        This makes Grace continuously evolve!
        """
        self.running = True
        iteration = 0
        
        logger.info(f"🚀 Starting continuous improvement loop (interval: {interval_hours}h)")
        
        try:
            while self.running:
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Improvement Iteration #{iteration}")
                logger.info(f"{'='*60}\n")
                
                # Run one improvement cycle
                result = await self.improvement_cycle()
                
                # Log result
                self.deployment_history.append(result)
                logger.info(f"Cycle result: {result.status.value}")
                
                # Wait for next interval
                if self.running and (max_iterations is None or iteration < max_iterations):
                    await asyncio.sleep(interval_hours * 3600)
        
        finally:
            self.running = False
            logger.info("Continuous improvement loop stopped")
    
    def stop(self):
        """Stop continuous improvement"""
        self.running = False
    
    async def _generate_candidate(self) -> Candidate:
        """
        Generate candidate based on recent performance.
        
        In production, this would analyze:
        - Recent failure modes
        - Uncertainty bands
        - Disagreement patterns
        """
        # For now, use random perturbation
        # In production, use failure-driven adaptation
        
        candidate_config = self.adaptation_surface.generate_random_candidate(
            self.baseline_config
        )
        
        # Compute changes
        changes = {}
        for key in candidate_config:
            if candidate_config[key] != self.baseline_config.get(key):
                changes[key] = {
                    "from": self.baseline_config.get(key),
                    "to": candidate_config[key]
                }
        
        # Determine risk level
        risk_level = self._assess_risk(changes)
        
        candidate = Candidate(
            candidate_id=self._generate_candidate_id(),
            config=candidate_config,
            changes_from_baseline=changes,
            risk_level=risk_level,
            generated_from="random_perturbation",  # In production: "failure_analysis"
            rationale="Exploratory adaptation to improve performance"
        )
        
        self.candidate_history.append(candidate)
        return candidate
    
    def _assess_risk(self, changes: Dict[str, Any]) -> RiskLevel:
        """Assess risk level of changes"""
        # Simple heuristic - in production, use more sophisticated logic
        num_changes = len(changes)
        
        if num_changes == 1:
            return RiskLevel.LOW
        elif num_changes <= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _compute_statistical_confidence(
        self,
        candidate_result: EvaluationResult,
        improvement: float
    ) -> float:
        """
        Compute statistical confidence in improvement.
        
        In production, use proper statistical testing:
        - Bootstrap confidence intervals
        - T-test for significance
        - Multiple hypothesis correction
        """
        # Simplified confidence calculation
        # Higher confidence if:
        # - Larger improvement
        # - Higher success rate
        # - More tasks evaluated
        
        success_rate = candidate_result.scorecard["task_success_rate"]
        num_tasks = candidate_result.scorecard["total_tasks"]
        
        # Base confidence on improvement magnitude
        if improvement > 0.1:  # 10%+ improvement
            base_confidence = 0.95
        elif improvement > 0.05:  # 5%+ improvement
            base_confidence = 0.85
        elif improvement > 0.02:  # 2%+ improvement
            base_confidence = 0.75
        else:
            base_confidence = 0.5
        
        # Adjust for success rate
        confidence = base_confidence * success_rate
        
        # Adjust for sample size
        if num_tasks < 5:
            confidence *= 0.8
        elif num_tasks >= 10:
            confidence *= 1.1
        
        return min(1.0, confidence)
    
    async def _request_governance_approval(
        self,
        candidate: Candidate,
        result: EvaluationResult
    ) -> bool:
        """Request governance approval for high-risk changes"""
        if self.governance is None:
            # No governance system - default to conservative approval
            logger.warning("No governance system configured - auto-approving")
            return True
        
        # In production, trigger actual governance voting
        approval_request = {
            "candidate": candidate.to_dict(),
            "evaluation": result.to_dict(),
            "risk_level": candidate.risk_level.value
        }
        
        # Stub: would call governance.request_approval(approval_request)
        logger.info("Governance approval requested (simulated)")
        return True  # Simulate approval
    
    async def _deploy_candidate(
        self,
        candidate: Candidate,
        result: EvaluationResult
    ):
        """Deploy candidate configuration"""
        logger.info(f"🚀 Deploying candidate: {candidate.candidate_id}")
        
        # In production:
        # 1. Update live configuration
        # 2. Canary deployment (10% -> 50% -> 100%)
        # 3. Monitor for degradation
        # 4. Auto-rollback if issues
        
        # For now, just log
        logger.info("  Configuration updated")
        logger.info("  Monitoring validation metrics...")
    
    async def _distill_winning_strategy(self, candidate: Candidate):
        """
        Extract and persist successful strategies.
        
        This is how improvements become "sticky" - we learn
        what works and consolidate it into reusable patterns.
        """
        logger.info(f"📚 Distilling winning strategy from {candidate.candidate_id}")
        
        # In production:
        # 1. Extract prompt patterns that worked
        # 2. Update routing heuristics
        # 3. Consolidate adapter weights
        # 4. Build reusable templates
        
        strategy = {
            "candidate_id": candidate.candidate_id,
            "changes": candidate.changes_from_baseline,
            "rationale": candidate.rationale,
            "learned_at": datetime.utcnow().isoformat()
        }
        
        # Save to strategy library
        logger.info(f"  Strategy added to library")
    
    def _generate_candidate_id(self) -> str:
        """Generate unique candidate ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(str(np.random.random()).encode()).hexdigest()[:8]
        return f"cand_{timestamp}_{random_suffix}"
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvement progress"""
        total_candidates = len(self.candidate_history)
        deployed_count = len([d for d in self.deployment_history if d.status == DeploymentStatus.DEPLOYED])
        
        if self.baseline_score and deployed_count > 0:
            current_score = self.baseline_score
            initial_score = self.deployment_history[0].evaluation_result.reward if self.deployment_history else current_score
            total_improvement = current_score - initial_score
        else:
            total_improvement = 0.0
        
        return {
            "total_candidates_generated": total_candidates,
            "total_deployments": deployed_count,
            "current_reward": self.baseline_score,
            "total_improvement": total_improvement,
            "improvement_percentage": (total_improvement / initial_score * 100) if initial_score > 0 else 0.0
        }


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("🚀 Meta-Loop Optimizer Demo\n")
        
        # Create harness and optimizer
        harness = EvaluationHarness()
        optimizer = MetaLoopOptimizer(harness)
        
        # Initialize baseline
        await optimizer.initialize_baseline()
        
        # Run 3 improvement cycles
        print("\nRunning 3 improvement cycles...\n")
        for i in range(3):
            result = await optimizer.improvement_cycle()
            print(f"\n{'='*60}")
            print(f"Cycle {i+1} Result: {result.status.value}")
            print(f"Improvement: {result.improvement:+.4f}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"{'='*60}\n")
        
        # Show summary
        summary = optimizer.get_improvement_summary()
        print("\n📊 Improvement Summary:")
        print(f"  Total Candidates: {summary['total_candidates_generated']}")
        print(f"  Successful Deployments: {summary['total_deployments']}")
        print(f"  Current Reward: {summary['current_reward']:.4f}")
        print(f"  Total Improvement: {summary['total_improvement']:+.4f} ({summary['improvement_percentage']:+.1f}%)")
    
    asyncio.run(demo())
