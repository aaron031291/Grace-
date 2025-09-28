"""Chaos engineering experiment runner."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of chaos experiments."""
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    SERVICE_KILL = "service_kill"
    DEPENDENCY_FAILURE = "dependency_failure"


class ExperimentStatus(Enum):
    """Chaos experiment status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ChaosExperiment:
    """
    Individual chaos experiment definition and execution.
    """
    
    def __init__(
        self,
        experiment_id: str,
        target: str,
        experiment_type: ExperimentType,
        blast_radius_pct: float,
        duration_s: int,
        parameters: Optional[Dict] = None
    ):
        """Initialize chaos experiment."""
        self.experiment_id = experiment_id
        self.target = target
        self.experiment_type = experiment_type
        self.blast_radius_pct = blast_radius_pct
        self.duration_s = duration_s
        self.parameters = parameters or {}
        
        self.status = ExperimentStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.results = {}
        self.metrics_before = {}
        self.metrics_during = {}
        self.metrics_after = {}
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the chaos experiment."""
        try:
            logger.info(f"Starting chaos experiment {self.experiment_id} on {self.target}")
            
            self.status = ExperimentStatus.RUNNING
            self.started_at = datetime.now()
            
            # Collect baseline metrics
            self.metrics_before = await self._collect_metrics()
            
            # Execute the experiment
            await self._run_experiment()
            
            # Wait for experiment duration
            await asyncio.sleep(self.duration_s)
            
            # Stop the experiment
            await self._stop_experiment()
            
            # Collect post-experiment metrics
            self.metrics_after = await self._collect_metrics()
            
            # Analyze results
            self.results = await self._analyze_results()
            
            self.status = ExperimentStatus.COMPLETED
            self.completed_at = datetime.now()
            
            logger.info(f"Chaos experiment {self.experiment_id} completed successfully")
            
            return {
                "experiment_id": self.experiment_id,
                "status": self.status.value,
                "outcome": "pass" if self.results.get("hypothesis_confirmed", False) else "fail",
                "findings": self.results
            }
            
        except Exception as e:
            logger.error(f"Chaos experiment {self.experiment_id} failed: {e}")
            self.status = ExperimentStatus.FAILED
            self.completed_at = datetime.now()
            
            return {
                "experiment_id": self.experiment_id,
                "status": self.status.value,
                "outcome": "fail",
                "error": str(e)
            }
    
    async def _run_experiment(self):
        """Execute the specific experiment type."""
        if self.experiment_type == ExperimentType.LATENCY_INJECTION:
            await self._inject_latency()
        elif self.experiment_type == ExperimentType.ERROR_INJECTION:
            await self._inject_errors()
        elif self.experiment_type == ExperimentType.RESOURCE_EXHAUSTION:
            await self._exhaust_resources()
        elif self.experiment_type == ExperimentType.NETWORK_PARTITION:
            await self._partition_network()
        elif self.experiment_type == ExperimentType.SERVICE_KILL:
            await self._kill_service()
        elif self.experiment_type == ExperimentType.DEPENDENCY_FAILURE:
            await self._fail_dependency()
    
    async def _inject_latency(self):
        """Inject artificial latency."""
        latency_ms = self.parameters.get("latency_ms", 1000)
        logger.info(f"Injecting {latency_ms}ms latency to {self.target}")
        # Implementation would integrate with service mesh or proxy
        await asyncio.sleep(0.1)  # Simulate setup
    
    async def _inject_errors(self):
        """Inject artificial errors."""
        error_rate = self.parameters.get("error_rate_pct", 10.0)
        logger.info(f"Injecting {error_rate}% error rate to {self.target}")
        # Implementation would integrate with service mesh or proxy
        await asyncio.sleep(0.1)  # Simulate setup
    
    async def _exhaust_resources(self):
        """Exhaust system resources."""
        resource_type = self.parameters.get("resource_type", "cpu")
        logger.info(f"Exhausting {resource_type} resources on {self.target}")
        # Implementation would integrate with container orchestrator
        await asyncio.sleep(0.1)  # Simulate setup
    
    async def _partition_network(self):
        """Create network partition."""
        logger.info(f"Creating network partition for {self.target}")
        # Implementation would integrate with network controls
        await asyncio.sleep(0.1)  # Simulate setup
    
    async def _kill_service(self):
        """Kill service instances."""
        kill_percentage = self.parameters.get("kill_percentage", 50.0)
        logger.info(f"Killing {kill_percentage}% of {self.target} instances")
        # Implementation would integrate with orchestrator
        await asyncio.sleep(0.1)  # Simulate setup
    
    async def _fail_dependency(self):
        """Fail a service dependency."""
        dependency = self.parameters.get("dependency", "database")
        logger.info(f"Failing dependency {dependency} for {self.target}")
        # Implementation would integrate with dependency management
        await asyncio.sleep(0.1)  # Simulate setup
    
    async def _stop_experiment(self):
        """Stop the experiment and restore normal conditions."""
        logger.info(f"Stopping experiment {self.experiment_id}")
        # Restore normal conditions
        await asyncio.sleep(0.1)  # Simulate cleanup
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics."""
        # Placeholder for metric collection
        return {
            "latency_p95": 150.0,
            "error_rate": 2.0,
            "throughput": 1000.0,
            "cpu_usage": 45.0,
            "memory_usage": 60.0
        }
    
    async def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        # Compare metrics before/after
        latency_increase = (
            self.metrics_after.get("latency_p95", 0) - 
            self.metrics_before.get("latency_p95", 0)
        )
        
        error_increase = (
            self.metrics_after.get("error_rate", 0) - 
            self.metrics_before.get("error_rate", 0)
        )
        
        # Determine if system remained stable
        system_stable = (
            latency_increase < 500 and  # Less than 500ms increase
            error_increase < 5.0        # Less than 5% error increase
        )
        
        return {
            "hypothesis_confirmed": system_stable,
            "latency_impact_ms": latency_increase,
            "error_rate_impact_pct": error_increase,
            "system_stability": "stable" if system_stable else "degraded",
            "recommendations": self._generate_recommendations(system_stable)
        }
    
    def _generate_recommendations(self, system_stable: bool) -> List[str]:
        """Generate recommendations based on experiment results."""
        if system_stable:
            return [
                "System demonstrated good resilience to failure",
                "Consider increasing experiment blast radius gradually",
                "Monitor for any delayed effects"
            ]
        else:
            return [
                "System showed instability under failure conditions",
                "Consider implementing additional resilience patterns",
                "Review circuit breaker and retry configurations",
                "Investigate cascade failure potential"
            ]


class ChaosRunner:
    """
    Chaos engineering experiment runner and orchestrator.
    """
    
    def __init__(self, max_blast_radius_pct: float = 5.0):
        """Initialize chaos runner."""
        self.max_blast_radius_pct = max_blast_radius_pct
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[Dict] = []
        self.governance_approval_required = True
        
        logger.debug(f"Chaos runner initialized with max blast radius {max_blast_radius_pct}%")
    
    async def start_experiment(
        self,
        target: str,
        blast_radius_pct: float,
        duration_s: int,
        experiment_type: str = "latency_injection",
        parameters: Optional[Dict] = None
    ) -> str:
        """
        Start a chaos experiment.
        
        Args:
            target: Target service/component
            blast_radius_pct: Percentage of instances to affect
            duration_s: Experiment duration in seconds
            experiment_type: Type of experiment to run
            parameters: Additional experiment parameters
            
        Returns:
            Experiment ID
        """
        # Validate blast radius
        if blast_radius_pct > self.max_blast_radius_pct:
            raise ValueError(f"Blast radius {blast_radius_pct}% exceeds maximum {self.max_blast_radius_pct}%")
        
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Create experiment
        exp_type = ExperimentType(experiment_type)
        experiment = ChaosExperiment(
            experiment_id=experiment_id,
            target=target,
            experiment_type=exp_type,
            blast_radius_pct=blast_radius_pct,
            duration_s=duration_s,
            parameters=parameters
        )
        
        # Store experiment
        self.active_experiments[experiment_id] = experiment
        
        # Start experiment asynchronously
        asyncio.create_task(self._run_experiment_async(experiment))
        
        logger.info(f"Started chaos experiment {experiment_id} on {target}")
        return experiment_id
    
    async def _run_experiment_async(self, experiment: ChaosExperiment):
        """Run experiment asynchronously and handle completion."""
        try:
            result = await experiment.execute()
            
            # Move from active to history
            if experiment.experiment_id in self.active_experiments:
                del self.active_experiments[experiment.experiment_id]
            
            self.experiment_history.append(result)
            
            # Trim history
            if len(self.experiment_history) > 1000:
                self.experiment_history = self.experiment_history[-500:]
                
        except Exception as e:
            logger.error(f"Error running chaos experiment {experiment.experiment_id}: {e}")
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict]:
        """Get status of an experiment."""
        if experiment_id in self.active_experiments:
            exp = self.active_experiments[experiment_id]
            return {
                "experiment_id": experiment_id,
                "status": exp.status.value,
                "target": exp.target,
                "blast_radius_pct": exp.blast_radius_pct,
                "started_at": exp.started_at.isoformat() if exp.started_at else None,
                "duration_s": exp.duration_s
            }
        
        # Check history
        for result in reversed(self.experiment_history):
            if result["experiment_id"] == experiment_id:
                return result
        
        return None
    
    def get_active_experiments(self) -> List[Dict]:
        """Get all active experiments."""
        return [
            {
                "experiment_id": exp_id,
                "status": exp.status.value,
                "target": exp.target,
                "blast_radius_pct": exp.blast_radius_pct,
                "started_at": exp.started_at.isoformat() if exp.started_at else None
            }
            for exp_id, exp in self.active_experiments.items()
        ]
    
    def get_experiment_history(self, limit: int = 50) -> List[Dict]:
        """Get experiment history."""
        return self.experiment_history[-limit:] if limit else self.experiment_history
    
    async def abort_experiment(self, experiment_id: str) -> bool:
        """Abort a running experiment."""
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            experiment.status = ExperimentStatus.ABORTED
            
            try:
                await experiment._stop_experiment()
                logger.info(f"Aborted chaos experiment {experiment_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to abort experiment {experiment_id}: {e}")
                return False
        
        return False


# Global chaos runner instance
_chaos_runner = ChaosRunner()


async def start_experiment(target: str, blast_radius_pct: float, duration_s: int) -> str:
    """Global function to start chaos experiment."""
    return await _chaos_runner.start_experiment(target, blast_radius_pct, duration_s)


def get_chaos_runner() -> ChaosRunner:
    """Get the global chaos runner instance."""
    return _chaos_runner