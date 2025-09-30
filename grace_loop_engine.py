#!/usr/bin/env python3
"""
Grace Loop Engine Implementation
===============================

Implements the canonical 7-phase loop engine as specified in the Grace Build & Policy Contract:
perceive -> reason -> plan -> act -> reflect -> learn -> log
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LoopPhase(Enum):
    """Loop execution phases."""

    PERCEIVE = "perceive"
    REASON = "reason"
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"
    LEARN = "learn"
    LOG = "log"


@dataclass
class LoopMetrics:
    """Loop execution metrics."""

    turn_latency_ms: float = 0.0
    tool_success_rate: float = 0.0
    error_rate: float = 0.0
    hallucination_flag_rate: float = 0.0
    governance_block_rate: float = 0.0
    phase_durations: Dict[str, float] = field(default_factory=dict)
    total_actions: int = 0
    successful_actions: int = 0


@dataclass
class LoopResult:
    """Result of a complete loop execution."""

    success: bool
    ui_instructions: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[LoopMetrics] = None
    memory_deltas: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    governance_blocks: List[str] = field(default_factory=list)


class GraceLoopEngine:
    """
    Grace canonical loop engine implementation.

    Enforces sandbox-only execution and governance constraints.
    """

    def __init__(self):
        self.constraints = {
            "sandbox_only": True,
            "max_actions_per_turn": 8,
            "max_tokens_per_turn": 8192,
        }
        self.retry_config = {"max_retries": 2, "backoff_ms": [200, 800]}
        self.governance_required_actions = [
            "modify_code",
            "change_constraints",
            "alter_memory_rules",
            "modify_api_contract",
        ]

    async def execute_turn(
        self,
        user_input: str,
        context: Dict[str, Any],
        grace_intelligence=None,
        governance_engine=None,
    ) -> LoopResult:
        """Execute a complete turn through all loop phases."""
        start_time = time.time()
        result = LoopResult(success=False)
        metrics = LoopMetrics()

        try:
            # Phase 1: Perceive
            phase_start = time.time()
            perceive_result = await self._phase_perceive(user_input, context)
            metrics.phase_durations["perceive"] = (time.time() - phase_start) * 1000

            # Phase 2: Reason
            phase_start = time.time()
            reason_result = await self._phase_reason(
                perceive_result, grace_intelligence
            )
            metrics.phase_durations["reason"] = (time.time() - phase_start) * 1000

            # Phase 3: Plan
            phase_start = time.time()
            plan_result = await self._phase_plan(reason_result)
            metrics.phase_durations["plan"] = (time.time() - phase_start) * 1000

            # Phase 4: Act
            phase_start = time.time()
            act_result = await self._phase_act(plan_result, governance_engine)
            metrics.phase_durations["act"] = (time.time() - phase_start) * 1000

            # Update metrics from action phase
            metrics.total_actions = act_result.get("total_actions", 0)
            metrics.successful_actions = act_result.get("successful_actions", 0)
            if metrics.total_actions > 0:
                metrics.tool_success_rate = (
                    metrics.successful_actions / metrics.total_actions
                )

            # Phase 5: Reflect
            phase_start = time.time()
            reflect_result = await self._phase_reflect(act_result)
            metrics.phase_durations["reflect"] = (time.time() - phase_start) * 1000

            # Phase 6: Learn
            phase_start = time.time()
            learn_result = await self._phase_learn(reflect_result)
            metrics.phase_durations["learn"] = (time.time() - phase_start) * 1000

            # Phase 7: Log
            phase_start = time.time()
            log_result = await self._phase_log(
                {
                    "perceive": perceive_result,
                    "reason": reason_result,
                    "plan": plan_result,
                    "act": act_result,
                    "reflect": reflect_result,
                    "learn": learn_result,
                },
                metrics,
            )
            metrics.phase_durations["log"] = (time.time() - phase_start) * 1000

            # Calculate final metrics
            total_time = time.time() - start_time
            metrics.turn_latency_ms = total_time * 1000

            # Build result
            result.success = True
            result.ui_instructions = act_result.get("ui_instructions", [])
            result.messages = act_result.get("messages", [])
            result.memory_deltas = learn_result.get("memory_deltas", [])
            result.metrics = metrics

            logger.info(
                f"Loop turn completed successfully in {metrics.turn_latency_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Loop execution failed: {e}")
            result.errors.append(str(e))
            metrics.error_rate = 1.0
            result.metrics = metrics

        return result

    async def _phase_perceive(
        self, user_input: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 1: Gather input, attachments, context, memory recall."""
        return {
            "user_input": user_input,
            "context": context,
            "attachments": context.get("attachments", []),
            "memory_context": await self._recall_relevant_memories(user_input),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    async def _phase_reason(
        self, perceive_result: Dict[str, Any], grace_intelligence=None
    ) -> Dict[str, Any]:
        """Phase 2: Deliberate via GraceIntelligence; produce ReasoningResult."""
        if grace_intelligence:
            # Use actual Grace Intelligence if available
            try:
                from grace.intelligence.grace_intelligence import (
                    ReasoningContext,
                    ReasoningStage,
                )

                context = ReasoningContext(
                    user_id="system",
                    session_id="system",
                    conversation_history=[],
                    current_panels=[],
                )
                reasoning_result = await grace_intelligence.process_request(
                    perceive_result["user_input"], context
                )
                return {
                    "reasoning_result": reasoning_result,
                    "confidence": getattr(reasoning_result, "confidence", 0.8),
                    "domain": getattr(reasoning_result, "domain", "general"),
                }
            except Exception as e:
                logger.warning(f"Grace Intelligence not available: {e}")

        # Fallback reasoning
        return {
            "intent": "process_user_request",
            "confidence": 0.7,
            "domain": "general",
            "key_points": [perceive_result["user_input"]],
            "reasoning_trace": ["Basic intent recognition performed"],
        }

    async def _phase_plan(self, reason_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Produce explicit Plan with steps."""
        return {
            "steps": [
                {
                    "type": "response",
                    "action": "generate_response",
                    "params": {"intent": reason_result.get("intent", "unknown")},
                }
            ],
            "estimated_duration_ms": 1000,
            "requires_governance": False,
            "sandbox_safe": True,
        }

    async def _phase_act(
        self, plan_result: Dict[str, Any], governance_engine=None
    ) -> Dict[str, Any]:
        """Phase 4: Execute tool/API calls (bounded by sandbox + allowlist)."""
        actions_executed = 0
        successful_actions = 0
        ui_instructions = []
        messages = []
        governance_blocks = []

        for step in plan_result.get("steps", []):
            if actions_executed >= self.constraints["max_actions_per_turn"]:
                logger.warning("Max actions per turn reached")
                break

            # Check governance requirements
            if step.get("action") in self.governance_required_actions:
                if governance_engine:
                    # Check with governance
                    approval = await self._check_governance_approval(
                        step, governance_engine
                    )
                    if not approval:
                        governance_blocks.append(f"Action blocked: {step['action']}")
                        continue
                else:
                    governance_blocks.append(
                        f"Governance required but not available: {step['action']}"
                    )
                    continue

            # Execute action with retry logic
            success = await self._execute_action_with_retry(step)
            actions_executed += 1

            if success:
                successful_actions += 1
                # Generate response based on action
                if step.get("action") == "generate_response":
                    messages.append(
                        {
                            "type": "assistant",
                            "content": "Request processed successfully",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                    )

        return {
            "total_actions": actions_executed,
            "successful_actions": successful_actions,
            "ui_instructions": ui_instructions,
            "messages": messages,
            "governance_blocks": governance_blocks,
        }

    async def _phase_reflect(self, act_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Summarize outcome, detect errors, generate improvements."""
        success_rate = 0.0
        if act_result["total_actions"] > 0:
            success_rate = act_result["successful_actions"] / act_result["total_actions"]

        return {
            "success_rate": success_rate,
            "errors_detected": len(act_result.get("governance_blocks", [])),
            "improvements": [],
            "outcome_summary": f"Executed {act_result['total_actions']} actions with {success_rate:.1%} success rate",
        }

    async def _phase_learn(self, reflect_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Write deltas to memory (short-term ↔ long-term), update skills."""
        memory_deltas = []

        # Example: promote successful patterns to long-term memory
        if reflect_result["success_rate"] > 0.8:
            memory_deltas.append(
                {
                    "type": "promote",
                    "fragment": "successful_interaction_pattern",
                    "trust_score": 0.8,
                    "tags": ["stable_fact"],
                }
            )

        return {"memory_deltas": memory_deltas, "skills_updated": []}

    async def _phase_log(
        self, all_results: Dict[str, Any], metrics: LoopMetrics
    ) -> Dict[str, Any]:
        """Phase 7: Append immutable log entries."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "phases": {phase: result for phase, result in all_results.items()},
            "metrics": {
                "turn_latency_ms": metrics.turn_latency_ms,
                "tool_success_rate": metrics.tool_success_rate,
                "error_rate": metrics.error_rate,
                "phase_durations": metrics.phase_durations,
            },
            "hash": self._calculate_log_hash(all_results, metrics),
        }

        # In production, this would be written to immutable storage
        logger.info(f"Loop execution logged: {log_entry['hash'][:8]}")
        return {"log_entry": log_entry}

    async def _recall_relevant_memories(self, user_input: str) -> List[Dict[str, Any]]:
        """Recall relevant memories for context."""
        # Placeholder - would integrate with memory system
        return []

    async def _check_governance_approval(
        self, step: Dict[str, Any], governance_engine
    ) -> bool:
        """Check if action is approved by governance."""
        # Placeholder - would integrate with governance system
        return False  # Default to blocking for safety

    async def _execute_action_with_retry(self, step: Dict[str, Any]) -> bool:
        """Execute action with retry logic."""
        for attempt in range(self.retry_config["max_retries"] + 1):
            try:
                if not self.constraints["sandbox_only"]:
                    logger.error("Production actions not allowed - sandbox only!")
                    return False

                # Simulate action execution
                if step.get("action") == "generate_response":
                    return True

                return True

            except Exception as e:
                if attempt < self.retry_config["max_retries"]:
                    await self._wait_backoff(attempt)
                    continue
                logger.error(f"Action failed after retries: {e}")
                return False

        return False

    async def _wait_backoff(self, attempt: int):
        """Wait with exponential backoff."""
        import asyncio

        wait_ms = self.retry_config["backoff_ms"][
            min(attempt, len(self.retry_config["backoff_ms"]) - 1)
        ]
        await asyncio.sleep(wait_ms / 1000.0)

    def _calculate_log_hash(
        self, all_results: Dict[str, Any], metrics: LoopMetrics
    ) -> str:
        """Calculate hash for immutable log chain."""
        import hashlib
        import json

        data = json.dumps(
            {"results": all_results, "metrics": metrics.__dict__}, sort_keys=True
        )
        return hashlib.sha256(data.encode()).hexdigest()