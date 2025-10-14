"""
Constitutional Decorator - Runtime governance enforcement for Grace operations.

This decorator ensures all user-facing operations are validated against
constitutional principles before execution.
"""

import asyncio
import logging
import functools
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .constitutional_validator import (
    ConstitutionalValidator,
    ConstitutionalValidationResult,
)
from ..audit.immutable_logs import ImmutableLogs

logger = logging.getLogger(__name__)


class ConstitutionalCheckError(Exception):
    """Exception raised when constitutional check fails."""

    def __init__(
        self,
        message: str,
        violations: list,
        validation_result: ConstitutionalValidationResult,
    ):
        self.message = message
        self.violations = violations
        self.validation_result = validation_result
        super().__init__(message)


def constitutional_check(
    policy: str = "default",
    transparency_level: str = "democratic_oversight",
    require_rationale: bool = True,
    allow_override: bool = False,
):
    """
    Decorator to enforce constitutional compliance on functions.

    Args:
        policy: The governance policy to apply ("default", "strict", "permissive")
        transparency_level: Audit transparency level
        require_rationale: Whether to require rationale in response
        allow_override: Whether to allow constitutional overrides (dangerous)

    Usage:
        @constitutional_check(policy="strict")
        async def sensitive_operation(data):
            return {"result": "processed"}
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Initialize validator
            validator = ConstitutionalValidator()
            audit_logger = ImmutableLogs()

            # Extract action details from function call
            action = {
                "type": "function_execution",
                "function_name": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                "policy": policy,
                "timestamp": datetime.now().isoformat(),
                "id": f"{func.__name__}_{int(datetime.now().timestamp() * 1000)}",
            }

            # Add rationale if provided in kwargs
            if "rationale" in kwargs:
                action["rationale"] = kwargs.pop("rationale")
            elif require_rationale:
                action["rationale"] = f"Automated execution of {func.__name__}"

            context = {
                "transparency_level": transparency_level,
                "policy": policy,
                "allow_override": allow_override,
                "has_rationale": "rationale" in action,
            }

            # Log pre-execution
            await audit_logger.log_governance_action(
                action_type="constitutional_check_start",
                data={"action": action, "context": context, "function": func.__name__},
                transparency_level=transparency_level,
            )

            # Validate against constitution
            validation_result = await validator.validate_against_constitution(
                action, context
            )

            if not validation_result.is_valid:
                # Log violation
                await audit_logger.log_governance_action(
                    action_type="constitutional_violation",
                    data={
                        "action": action,
                        "violations": [vars(v) for v in validation_result.violations],
                        "severity": "critical"
                        if any(
                            v.severity == "critical"
                            for v in validation_result.violations
                        )
                        else "major",
                    },
                    transparency_level="public",  # Violations are always public
                )

                if not allow_override:
                    raise ConstitutionalCheckError(
                        f"Constitutional violation in {func.__name__}: {validation_result.summary}",
                        [vars(v) for v in validation_result.violations],
                        validation_result,
                    )

            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Ensure result includes governance metadata
                if isinstance(result, dict):
                    result.setdefault("governance", {})
                    result["governance"].update(
                        {
                            "constitutional_compliance": validation_result.is_valid,
                            "policy_applied": policy,
                            "transparency_level": transparency_level,
                            "audit_trail": validation_result.audit_trail,
                            "rationale": action.get("rationale"),
                            "timestamp": action["timestamp"],
                        }
                    )

                    if validation_result.violations:
                        result["governance"]["violations"] = [
                            vars(v) for v in validation_result.violations
                        ]

                # Log successful execution
                await audit_logger.log_governance_action(
                    action_type="constitutional_check_success",
                    data={
                        "action": action,
                        "result_type": type(result).__name__,
                        "has_governance_metadata": isinstance(result, dict)
                        and "governance" in result,
                    },
                    transparency_level=transparency_level,
                )

                return result

            except Exception as e:
                # Log execution failure
                await audit_logger.log_governance_action(
                    action_type="constitutional_check_failure",
                    data={
                        "action": action,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    transparency_level=transparency_level,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, run the async wrapper in event loop
            return asyncio.run(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def trust_middleware(
    min_trust_score: float = 0.75, trust_sources: Optional[list] = None
):
    """
    Decorator to enforce minimum trust score requirements.

    Args:
        min_trust_score: Minimum required trust score (0.0-1.0)
        trust_sources: List of trust sources to consider
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Initialize trust evaluation
            from ..governance.trust_core_kernel import TrustCoreKernel

            trust_kernel = TrustCoreKernel()

            # Extract trust context
            trust_context = {
                "function": func.__name__,
                "module": func.__module__,
                "sources": trust_sources or ["system"],
                "timestamp": datetime.now().isoformat(),
            }

            # Evaluate trust score
            trust_score = await trust_kernel.evaluate_trust_score(trust_context)

            if trust_score < min_trust_score:
                raise ConstitutionalCheckError(
                    f"Insufficient trust score for {func.__name__}: {trust_score:.2f} < {min_trust_score}",
                    [
                        {
                            "type": "trust_violation",
                            "score": trust_score,
                            "required": min_trust_score,
                        }
                    ],
                    None,
                )

            # Add trust metadata to kwargs
            kwargs["_trust_score"] = trust_score
            kwargs["_trust_context"] = trust_context

            return await func(*args, **kwargs)

        return wrapper

    return decorator


class ContradictionService:
    """
    Service to detect and resolve contradictions in governance decisions.
    """

    def __init__(self):
        self.validator = ConstitutionalValidator()
        self.audit_logger = ImmutableLogs()

    async def detect_contradictions(self, actions: list) -> Dict[str, Any]:
        """Detect contradictions between multiple actions."""
        contradictions = []

        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions[i + 1 :], i + 1):
                contradiction = await self._check_contradiction(action1, action2)
                if contradiction:
                    contradictions.append(
                        {
                            "action1_index": i,
                            "action2_index": j,
                            "contradiction": contradiction,
                        }
                    )

        return {
            "contradictions_found": len(contradictions),
            "contradictions": contradictions,
            "actions_analyzed": len(actions),
        }

    async def _check_contradiction(
        self, action1: Dict, action2: Dict
    ) -> Optional[Dict]:
        """Check if two actions contradict each other."""
        # Simple contradiction detection logic
        # In practice, this would be more sophisticated

        if (
            action1.get("type") == "allow"
            and action2.get("type") == "deny"
            and action1.get("resource") == action2.get("resource")
        ):
            return {
                "type": "allow_deny_contradiction",
                "resource": action1.get("resource"),
                "description": f"Action 1 allows while Action 2 denies access to {action1.get('resource')}",
            }

        return None


def uniform_envelope_builder(
    envelope_type: str = "governance_response", include_audit_trail: bool = True
):
    """
    Decorator to ensure uniform response envelope structure.

    Args:
        envelope_type: Type of envelope to build
        include_audit_trail: Whether to include audit trail in response
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()

            try:
                result = await func(*args, **kwargs)

                # Build uniform envelope
                envelope = {
                    "envelope_type": envelope_type,
                    "status": "success",
                    "timestamp": start_time.isoformat(),
                    "processing_time_ms": int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    ),
                    "data": result,
                    "metadata": {
                        "function": func.__name__,
                        "module": func.__module__,
                        "version": "1.0.0",
                    },
                }

                if include_audit_trail:
                    envelope["audit_trail"] = {
                        "execution_id": f"{func.__name__}_{int(start_time.timestamp() * 1000)}",
                        "constitutional_checks": getattr(result, "governance", {}).get(
                            "constitutional_compliance", True
                        ),
                        "trust_score": kwargs.get("_trust_score"),
                        "policy_applied": getattr(result, "governance", {}).get(
                            "policy_applied"
                        ),
                    }

                return envelope

            except Exception as e:
                # Build error envelope
                envelope = {
                    "envelope_type": envelope_type,
                    "status": "error",
                    "timestamp": start_time.isoformat(),
                    "processing_time_ms": int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    ),
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "function": func.__name__,
                    },
                    "metadata": {
                        "function": func.__name__,
                        "module": func.__module__,
                        "version": "1.0.0",
                    },
                }

                if include_audit_trail:
                    envelope["audit_trail"] = {
                        "execution_id": f"{func.__name__}_{int(start_time.timestamp() * 1000)}",
                        "error_logged": True,
                    }

                return envelope

        return wrapper

    return decorator
