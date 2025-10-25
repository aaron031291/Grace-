"""
Grace AI Meta-Learning Kernel - Top-level intelligence optimization
Observes all kernel actions and learns to optimize the system
Applies intelligence to improve trigger rules, handlers, and strategies
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from grace.kernels.base_kernel import BaseKernel
from .models import KernelInsight

logger = logging.getLogger(__name__)

class MetaLearningKernel(BaseKernel):
    """
    The self-improvement and optimization kernel for Grace.
    It integrates with the PolicyEngine, TrustLedger, and LLMService to
    analyze system performance and suggest improvements.
    """
    
    def __init__(self, service_registry=None):
        super().__init__("meta_learning_kernel", service_registry)
        self.policy_engine = self.get_service('policy_engine')
        self.trust_ledger = self.get_service('trust_ledger')
        self.llm_service = self.get_service('llm_service')
        self.insights_generated = 0
        self.logger.info("Meta-Learning Kernel initialized and services wired.")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a meta-learning task, like analyzing performance or updating policies.

        Args:
            task: A dictionary defining the task.
                  Example: {'type': 'analyze_kpi', 'kpi_data': {...}}
                           {'type': 'propose_policy_update'}
        """
        task_type = task.get('type', 'unknown')
        self.logger.info(f"Executing meta-learning task of type: {task_type}")

        if not all([self.policy_engine, self.trust_ledger, self.llm_service]):
            error_msg = "One or more required services are not available."
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        try:
            if task_type == 'analyze_performance':
                result = await self._analyze_performance(task)
            elif task_type == 'update_policy':
                result = await self._update_policy(task)
            else:
                result = {'success': False, 'error': f"Unknown task type: {task_type}"}

            self.insights_generated += 1
            return result
        except Exception as e:
            self.logger.error(f"Error during meta-learning execution: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _analyze_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance data to generate insights."""
        perf_data = task.get('data', {})
        self.logger.info(f"Analyzing performance data with {len(perf_data)} points.")

        # Use LLM to generate an insight
        prompt = f"Analyze the following performance data and generate a key insight: {perf_data}"
        llm_response = await self.llm_service.query(prompt)
        insight_text = llm_response.get('response', 'No insight generated.')

        insight = KernelInsight(
            source_kernel=self.name,
            insight_type='performance_analysis',
            content=insight_text,
            confidence=0.85
        )

        # Log insight to trust ledger
        self.trust_ledger.log_event(
            actor='meta_learning_kernel',
            action='generate_insight',
            details={'insight': insight.content}
        )

        return {'success': True, 'insight_generated': True, 'insight': insight.to_dict()}

    async def _update_policy(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Propose and apply an update to a system policy."""
        policy_name = task.get('policy_name', 'default_policy')
        proposed_change = task.get('change', 'No change specified.')
        self.logger.info(f"Updating policy '{policy_name}'.")

        # Use LLM to validate the proposed change
        prompt = f"Is the following a safe and effective policy change? '{proposed_change}'"
        llm_response = await self.llm_service.query(prompt)

        if 'yes' in llm_response.get('response', '').lower():
            # Apply policy change
            self.policy_engine.update_policy(policy_name, {'rule': proposed_change})
            self.logger.info(f"Policy '{policy_name}' updated successfully.")
            return {'success': True, 'policy_updated': True, 'policy_name': policy_name}
        else:
            self.logger.warning(f"Policy update for '{policy_name}' was rejected by LLM validation.")
            return {'success': False, 'policy_updated': False, 'reason': 'LLM validation failed.'}

    async def health_check(self) -> Dict[str, Any]:
        """Return the health status of the kernel."""
        return {
            'name': self.name,
            'running': self.is_running,
            'services': {
                'policy_engine': 'wired' if self.policy_engine else 'missing',
                'trust_ledger': 'wired' if self.trust_ledger else 'missing',
                'llm_service': 'wired' if self.llm_service else 'missing',
            },
            'insights_generated': self.insights_generated,
        }
