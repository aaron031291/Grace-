"""
Policy enforcement middleware for Grace API.

Enforces policy checks on API operations.
"""

import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from .rules import get_policy_engine

logger = logging.getLogger(__name__)


class PolicyEnforcementMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce policy rules on API operations."""

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.policy_engine = get_policy_engine()

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request through policy engine."""
        if not self.enabled:
            return await call_next(request)

        # Extract operation details from request
        operation = await self._extract_operation(request)

        if operation:
            # Evaluate against policies
            policy_result = self.policy_engine.evaluate_operation(operation)

            if not policy_result["allowed"]:
                # Block the operation
                violations_summary = "; ".join(
                    [
                        f"{v['rule_name']} ({v['severity']})"
                        for v in policy_result["violations"]
                    ]
                )

                logger.warning(
                    f"Blocked operation due to policy violations: {violations_summary}"
                )

                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Operation blocked by policy",
                        "violations": policy_result["violations"],
                        "policy_result": policy_result,
                    },
                )

        # Continue with request
        return await call_next(request)

    async def _extract_operation(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract operation details from request."""
        path = request.url.path
        method = request.method

        # Get user context if available
        user_id = getattr(request.state, "user_id", None)
        user_roles = getattr(request.state, "user_roles", [])
        user_scopes = getattr(request.state, "user_scopes", [])

        operation = {
            "user_id": user_id,
            "user_roles": user_roles,
            "user_scopes": user_scopes,
            "request_method": method,
            "request_path": path,
        }

        # Determine operation type based on endpoint
        if path.startswith("/api/v1/memory/ingest"):
            operation["type"] = "memory_ingestion"

            # For POST requests, check body content
            if method == "POST":
                try:
                    body = await request.body()
                    if body:
                        # Note: This consumes the body, so we'd need to restore it
                        # In a real implementation, you'd want to buffer this properly
                        import json

                        data = json.loads(body.decode())
                        operation["content"] = str(data)

                        if "file_path" in data:
                            operation["file_path"] = data["file_path"]
                            operation["type"] = "file_ingestion"

                except Exception:
                    pass  # Ignore JSON parsing errors

        elif path.startswith("/api/v1/search"):
            operation["type"] = "memory_search"

        elif "code" in path or "ide" in path:
            operation["type"] = "ide_operation"

        elif method in ["PUT", "POST", "DELETE"]:
            # Potentially dangerous operations
            operation["type"] = "data_modification"

        else:
            # Read-only operations generally don't need policy checks
            return None

        return operation


def create_policy_middleware(enabled: bool = True):
    """Factory function to create policy enforcement middleware."""

    def middleware(app):
        return PolicyEnforcementMiddleware(app, enabled=enabled)

    return middleware


class PolicyValidator:
    """Utility class for manual policy validation."""

    def __init__(self):
        self.policy_engine = get_policy_engine()

    def validate_file_operation(
        self,
        file_path: str,
        operation_type: str,
        user_id: str = None,
        user_roles: list = None,
        user_scopes: list = None,
    ) -> Dict[str, Any]:
        """Validate a file operation against policies."""
        operation = {
            "type": operation_type,
            "file_path": file_path,
            "user_id": user_id or "unknown",
            "user_roles": user_roles or [],
            "user_scopes": user_scopes or [],
        }

        return self.policy_engine.evaluate_operation(operation)

    def validate_code_execution(
        self,
        code_content: str,
        user_id: str = None,
        user_roles: list = None,
        user_scopes: list = None,
    ) -> Dict[str, Any]:
        """Validate code execution against policies."""
        operation = {
            "type": "code_execution",
            "content": code_content,
            "user_id": user_id or "unknown",
            "user_roles": user_roles or [],
            "user_scopes": user_scopes or [],
        }

        return self.policy_engine.evaluate_operation(operation)

    def validate_ide_changes(
        self,
        changes: Dict[str, Any],
        branch_name: str = None,
        user_id: str = None,
        user_roles: list = None,
        user_scopes: list = None,
    ) -> Dict[str, Any]:
        """Validate IDE changes against policies."""
        operation = {
            "type": "ide_apply_changes",
            "content": str(changes),
            "branch_name": branch_name,
            "user_id": user_id or "unknown",
            "user_roles": user_roles or [],
            "user_scopes": user_scopes or [],
        }

        result = self.policy_engine.evaluate_operation(operation)

        # Check specific IDE requirements
        if result["violations"]:
            for violation in result["violations"]:
                if violation["rule_name"] == "ide_apply_changes":
                    metadata = violation.get("metadata", {})

                    # Check sandbox branch requirement
                    if "require_sandbox_branch" in violation["actions"]:
                        sandbox_prefix = metadata.get(
                            "sandbox_branch_prefix", "sandbox/"
                        )
                        if not branch_name or not branch_name.startswith(
                            sandbox_prefix
                        ):
                            result["sandbox_branch_required"] = True
                            result["sandbox_prefix"] = sandbox_prefix

                    # Check policy:pass label requirement
                    if "require_policy_pass" in violation["actions"]:
                        result["policy_pass_required"] = True
                        result["required_labels"] = metadata.get(
                            "required_labels", ["policy:pass"]
                        )

                    # Check human approval requirement
                    if "require_human_approval" in violation["actions"]:
                        result["human_approval_required"] = True
                        result["required_approvals"] = metadata.get(
                            "required_approvals", 1
                        )

        return result


# Global validator instance
_validator = None


def get_policy_validator() -> PolicyValidator:
    """Get global policy validator instance."""
    global _validator
    if _validator is None:
        _validator = PolicyValidator()
    return _validator
