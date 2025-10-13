"""
Grace Governance API Service - FastAPI service for governance approvals and enforcement.

Provides:
- /approve, /reject, /queue, /status endpoints
- RBAC & consent enforcement
- Immutable audit logging
- Event publishing to Grace Event Bus
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback for environments without FastAPI
    FASTAPI_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None
    JSONResponse = None

from ..contracts.message_envelope_simple import (
    EventTypes,
    RBACContext,
)
from ..layer_04_audit_logs.immutable_logs import ImmutableLogs as ImmutableLogger
from ..core.utils import create_error_response, utc_timestamp, normalize_timestamp
from ..core.middleware import get_request_id

logger = logging.getLogger(__name__)


# Request/Response Models
class GovernanceRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Governance approval request."""

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    action_type: str = Field(..., description="Type of action requiring approval")
    resource_id: str = Field(..., description="ID of resource to be acted upon")
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal", pattern="^(critical|high|normal|low)$")
    requester: str = Field(..., description="Who is requesting approval")
    reason: Optional[str] = Field(None, description="Reason for the request")
    timeout_seconds: int = Field(default=3600, ge=1, le=86400)


class GovernanceDecision(BaseModel if FASTAPI_AVAILABLE else object):
    """Governance approval decision."""

    request_id: str
    decision: str = Field(..., pattern="^(approved|rejected|deferred)$")
    approver: str = Field(..., description="Who made the decision")
    reason: Optional[str] = Field(None, description="Reason for the decision")
    conditions: List[str] = Field(
        default_factory=list, description="Conditions for approval"
    )
    expires_at: Optional[datetime] = Field(None, description="When approval expires")


class GovernanceStatus(BaseModel if FASTAPI_AVAILABLE else object):
    """Governance request status."""

    request_id: str
    status: str  # pending, approved, rejected, expired, executing, completed, failed
    created_at: datetime
    updated_at: datetime
    approvals_required: int
    approvals_received: int
    rejections_received: int
    current_approvers: List[str]
    decision_log: List[Dict[str, Any]]


class RBACCheck:
    """RBAC enforcement utilities."""

    @staticmethod
    def extract_rbac_from_headers(request: "Request") -> Optional[RBACContext]:
        """Extract RBAC context from request headers."""
        user_id = request.headers.get("X-User-ID")
        roles = (
            request.headers.get("X-User-Roles", "").split(",")
            if request.headers.get("X-User-Roles")
            else []
        )
        permissions = (
            request.headers.get("X-User-Permissions", "").split(",")
            if request.headers.get("X-User-Permissions")
            else []
        )

        if user_id:
            return RBACContext(
                user_id=user_id,
                roles=[r.strip() for r in roles if r.strip()],
                permissions=[p.strip() for p in permissions if p.strip()],
            )
        return None

    @staticmethod
    def check_permission(rbac: RBACContext, required_permission: str) -> bool:
        """Check if RBAC context has required permission."""
        if not rbac:
            return False

        # Check direct permissions
        if required_permission in rbac.permissions:
            return True

        # Check role-based permissions
        admin_roles = {"admin", "governance_admin", "super_user"}
        if any(role in admin_roles for role in rbac.roles):
            return True

        # Check specific governance roles
        governance_roles = {"governance_approver", "governance_reviewer"}
        governance_permissions = {
            "governance.approve",
            "governance.review",
            "governance.override",
        }

        if required_permission in governance_permissions:
            return any(role in governance_roles for role in rbac.roles)

        return False

    @staticmethod
    def check_consent(request: "Request", required_scopes: List[str]) -> bool:
        """Check if request has required consent scopes."""
        consent_header = request.headers.get("X-Consent-Scopes", "")
        consent_scopes = [s.strip() for s in consent_header.split(",") if s.strip()]

        return all(scope in consent_scopes for scope in required_scopes)


class GovernanceAPIService:
    """Grace Governance API Service."""

    def __init__(
        self,
        event_bus=None,
        governance_kernel=None,
        immutable_logger: Optional[ImmutableLogger] = None,
    ):
        self.event_bus = event_bus
        self.governance_kernel = governance_kernel
        self.immutable_logger = immutable_logger or ImmutableLogger()

        # In-memory state (could be replaced with Redis/database)
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.request_history: Dict[str, Dict[str, Any]] = {}

        # RBAC configuration
        self.rbac_policies = self._load_rbac_policies()
        self.consent_requirements = self._load_consent_requirements()

        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Grace Governance API",
                description="Governance approvals and enforcement",
                version="1.0.0",
            )
            self._register_routes()
        else:
            logger.warning("FastAPI not available, running in compatibility mode")
            self.app = None

    def _load_rbac_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load RBAC policies for different actions."""
        return {
            "mldl.deploy": {
                "required_permissions": ["mldl.deploy", "governance.approve"],
                "required_roles": ["mldl_admin", "governance_approver"],
                "min_approvers": 2,
            },
            "resilience.chaos": {
                "required_permissions": ["resilience.chaos", "governance.approve"],
                "required_roles": ["resilience_admin", "governance_approver"],
                "min_approvers": 1,
            },
            "memory.delete": {
                "required_permissions": ["memory.admin", "governance.approve"],
                "required_roles": ["memory_admin", "governance_approver"],
                "min_approvers": 2,
            },
            "system.rollback": {
                "required_permissions": ["system.admin", "governance.approve"],
                "required_roles": ["system_admin", "governance_approver"],
                "min_approvers": 1,
            },
        }

    def _load_consent_requirements(self) -> Dict[str, List[str]]:
        """Load consent requirements for different actions."""
        return {
            "mldl.deploy": ["model_deployment", "data_processing"],
            "memory.write": ["data_processing", "memory_storage"],
            "memory.delete": ["data_deletion"],
            "resilience.chaos": ["system_testing"],
        }

    def _register_routes(self):
        """Register FastAPI routes."""
        if not self.app:
            return

        @self.app.post("/api/governance/v1/approve", response_model=dict)
        async def approve_request(decision: GovernanceDecision, request: Request):
            return await self._handle_approval(decision, request)

        @self.app.post("/api/governance/v1/reject", response_model=dict)
        async def reject_request(decision: GovernanceDecision, request: Request):
            return await self._handle_rejection(decision, request)

        @self.app.post("/api/governance/v1/queue", response_model=dict)
        async def queue_request(
            governance_request: GovernanceRequest, request: Request
        ):
            return await self._handle_queue_request(governance_request, request)

        @self.app.get(
            "/api/governance/v1/status/{request_id}", response_model=GovernanceStatus
        )
        async def get_status(request_id: str, request: Request):
            return await self._handle_get_status(request_id, request)

        @self.app.get("/api/governance/v1/pending", response_model=List[dict])
        async def list_pending(request: Request, limit: int = 50):
            return await self._handle_list_pending(request, limit)

        @self.app.post("/api/governance/v1/override", response_model=dict)
        async def override_decision(request_id: str, reason: str, request: Request):
            return await self._handle_override(request_id, reason, request)

    async def _handle_queue_request(
        self, governance_request: GovernanceRequest, request: Request
    ) -> Dict[str, Any]:
        """Handle queuing a new governance request."""
        try:
            # Extract RBAC context
            rbac_context = RBACCheck.extract_rbac_from_headers(request)
            if not rbac_context:
                return JSONResponse(
                    status_code=401,
                    content=create_error_response(
                        "UNAUTHORIZED", "Authentication required"
                    ),
                )

            # Check if requester can submit this type of request
            if not self._can_submit_request(
                governance_request.action_type, rbac_context
            ):
                return JSONResponse(
                    status_code=403,
                    content=create_error_response(
                        "INSUFFICIENT_PERMISSIONS",
                        "Insufficient permissions to submit request",
                        f"Action type: {governance_request.action_type}",
                    ),
                )

            # Check consent requirements
            consent_scopes = self.consent_requirements.get(
                governance_request.action_type, []
            )
            if consent_scopes and not RBACCheck.check_consent(request, consent_scopes):
                return JSONResponse(
                    status_code=403,
                    content=create_error_response(
                        "MISSING_CONSENT",
                        f"Missing required consent: {consent_scopes}",
                        f"Action type: {governance_request.action_type}",
                    ),
                )

            # Create request record
            request_record = {
                "request_id": governance_request.request_id,
                "action_type": governance_request.action_type,
                "resource_id": governance_request.resource_id,
                "payload": governance_request.payload,
                "priority": governance_request.priority,
                "requester": governance_request.requester,
                "reason": governance_request.reason,
                "status": "pending",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "expires_at": datetime.utcnow()
                + timedelta(seconds=governance_request.timeout_seconds),
                "approvals_required": self._get_required_approvals(
                    governance_request.action_type
                ),
                "approvals_received": 0,
                "rejections_received": 0,
                "approvers": [],
                "decision_log": [],
            }

            # Store pending request
            self.pending_requests[governance_request.request_id] = request_record

            # Log to immutable logs
            await self.immutable_logger.log_governance_decision(
                {
                    "event_type": "GOVERNANCE_REQUEST_QUEUED",
                    "request_id": governance_request.request_id,
                    "action_type": governance_request.action_type,
                    "requester": governance_request.requester,
                    "timestamp": utc_timestamp(),
                    "rbac_context": rbac_context.to_dict(),
                    "http_request_id": get_request_id(),
                }
            )

            # Publish event
            if self.event_bus:
                await self.event_bus.publish(
                    event_type=EventTypes.GOVERNANCE_NEEDS_REVIEW,
                    payload={
                        "request_id": governance_request.request_id,
                        "action_type": governance_request.action_type,
                        "priority": governance_request.priority,
                        "requester": governance_request.requester,
                        "http_request_id": get_request_id(),
                    },
                    source="governance_api",
                    priority=governance_request.priority,
                )

            logger.info(
                f"Queued governance request {governance_request.request_id} for {governance_request.action_type}",
                extra={"request_id": get_request_id()},
            )

            return {
                "request_id": governance_request.request_id,
                "status": "pending",
                "message": "Request queued for approval",
                "approvals_required": request_record["approvals_required"],
                "estimated_approval_time": "1-4 hours",
                "created_at": normalize_timestamp(request_record["created_at"]),
                "expires_at": normalize_timestamp(request_record["expires_at"]),
                "http_request_id": get_request_id(),
            }

        except Exception as e:
            logger.error(
                f"Error queuing governance request: {e}",
                extra={"request_id": get_request_id()},
            )
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    "GOVERNANCE_QUEUE_FAILED",
                    "Failed to queue governance request",
                    str(e),
                ),
            )

    async def _handle_approval(
        self, decision: GovernanceDecision, request: Request
    ) -> Dict[str, Any]:
        """Handle approval decision."""
        try:
            # Extract RBAC context
            rbac_context = RBACCheck.extract_rbac_from_headers(request)
            if not rbac_context:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Get pending request
            if decision.request_id not in self.pending_requests:
                raise HTTPException(status_code=404, detail="Request not found")

            request_record = self.pending_requests[decision.request_id]

            # Check if request is still pending
            if request_record["status"] != "pending":
                raise HTTPException(
                    status_code=400,
                    detail=f"Request is {request_record['status']}, not pending",
                )

            # Check if request has expired
            if datetime.utcnow() > request_record["expires_at"]:
                request_record["status"] = "expired"
                raise HTTPException(status_code=400, detail="Request has expired")

            # Check approver permissions
            if not self._can_approve_request(
                request_record["action_type"], rbac_context
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions to approve this request",
                )

            # Check if approver already voted
            if rbac_context.user_id in request_record["approvers"]:
                raise HTTPException(
                    status_code=400, detail="You have already approved this request"
                )

            # Record approval
            approval_record = {
                "decision": "approved",
                "approver": decision.approver,
                "reason": decision.reason,
                "conditions": decision.conditions,
                "timestamp": datetime.utcnow().isoformat(),
                "rbac_context": rbac_context.to_dict(),
            }

            request_record["approvals_received"] += 1
            request_record["approvers"].append(rbac_context.user_id)
            request_record["decision_log"].append(approval_record)
            request_record["updated_at"] = datetime.utcnow()

            # Check if we have enough approvals
            if (
                request_record["approvals_received"]
                >= request_record["approvals_required"]
            ):
                request_record["status"] = "approved"

                # Move to history
                self.request_history[decision.request_id] = self.pending_requests.pop(
                    decision.request_id
                )

                # Log approval
                await self.immutable_logger.log_governance_decision(
                    {
                        "event_type": "GOVERNANCE_APPROVED",
                        "request_id": decision.request_id,
                        "action_type": request_record["action_type"],
                        "approver": decision.approver,
                        "final_decision": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "approvals_received": request_record["approvals_received"],
                    }
                )

                # Publish approval event
                if self.event_bus:
                    await self.event_bus.publish(
                        event_type=EventTypes.GOVERNANCE_APPROVED,
                        payload={
                            "request_id": decision.request_id,
                            "action_type": request_record["action_type"],
                            "resource_id": request_record["resource_id"],
                            "approver": decision.approver,
                            "conditions": decision.conditions,
                        },
                        source="governance_api",
                        priority=request_record["priority"],
                    )

                return {
                    "request_id": decision.request_id,
                    "status": "approved",
                    "message": "Request approved",
                    "conditions": decision.conditions,
                    "next_steps": "Request can now be executed",
                }
            else:
                # Still need more approvals
                remaining = (
                    request_record["approvals_required"]
                    - request_record["approvals_received"]
                )

                return {
                    "request_id": decision.request_id,
                    "status": "pending",
                    "message": f"Approval recorded. {remaining} more approvals needed.",
                    "approvals_received": request_record["approvals_received"],
                    "approvals_required": request_record["approvals_required"],
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling approval: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_rejection(
        self, decision: GovernanceDecision, request: Request
    ) -> Dict[str, Any]:
        """Handle rejection decision."""
        try:
            # Extract RBAC context
            rbac_context = RBACCheck.extract_rbac_from_headers(request)
            if not rbac_context:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Get pending request
            if decision.request_id not in self.pending_requests:
                raise HTTPException(status_code=404, detail="Request not found")

            request_record = self.pending_requests[decision.request_id]

            # Check approver permissions
            if not self._can_approve_request(
                request_record["action_type"], rbac_context
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions to reject this request",
                )

            # Record rejection
            rejection_record = {
                "decision": "rejected",
                "approver": decision.approver,
                "reason": decision.reason,
                "timestamp": datetime.utcnow().isoformat(),
                "rbac_context": rbac_context.to_dict(),
            }

            request_record["status"] = "rejected"
            request_record["rejections_received"] += 1
            request_record["decision_log"].append(rejection_record)
            request_record["updated_at"] = datetime.utcnow()

            # Move to history
            self.request_history[decision.request_id] = self.pending_requests.pop(
                decision.request_id
            )

            # Log rejection
            await self.immutable_logger.log_governance_decision(
                {
                    "event_type": "GOVERNANCE_REJECTED",
                    "request_id": decision.request_id,
                    "action_type": request_record["action_type"],
                    "approver": decision.approver,
                    "reason": decision.reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Publish rejection event
            if self.event_bus:
                await self.event_bus.publish(
                    event_type=EventTypes.GOVERNANCE_REJECTED,
                    payload={
                        "request_id": decision.request_id,
                        "action_type": request_record["action_type"],
                        "resource_id": request_record["resource_id"],
                        "approver": decision.approver,
                        "reason": decision.reason,
                    },
                    source="governance_api",
                    priority=request_record["priority"],
                )

            return {
                "request_id": decision.request_id,
                "status": "rejected",
                "message": "Request rejected",
                "reason": decision.reason,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling rejection: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_get_status(
        self, request_id: str, request: Request
    ) -> GovernanceStatus:
        """Get request status."""
        try:
            # Check pending requests first
            if request_id in self.pending_requests:
                record = self.pending_requests[request_id]
            elif request_id in self.request_history:
                record = self.request_history[request_id]
            else:
                raise HTTPException(status_code=404, detail="Request not found")

            return GovernanceStatus(
                request_id=request_id,
                status=record["status"],
                created_at=record["created_at"],
                updated_at=record["updated_at"],
                approvals_required=record["approvals_required"],
                approvals_received=record["approvals_received"],
                rejections_received=record["rejections_received"],
                current_approvers=record["approvers"],
                decision_log=record["decision_log"],
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting request status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_list_pending(
        self, request: Request, limit: int
    ) -> List[Dict[str, Any]]:
        """List pending requests."""
        try:
            # Extract RBAC context for filtering
            rbac_context = RBACCheck.extract_rbac_from_headers(request)

            pending = []
            for request_id, record in self.pending_requests.items():
                # Filter based on permissions
                if rbac_context and self._can_view_request(
                    record["action_type"], rbac_context
                ):
                    pending.append(
                        {
                            "request_id": request_id,
                            "action_type": record["action_type"],
                            "resource_id": record["resource_id"],
                            "requester": record["requester"],
                            "priority": record["priority"],
                            "created_at": record["created_at"].isoformat(),
                            "approvals_received": record["approvals_received"],
                            "approvals_required": record["approvals_required"],
                        }
                    )

            # Sort by priority and creation time
            priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
            pending.sort(
                key=lambda x: (priority_order.get(x["priority"], 4), x["created_at"])
            )

            return pending[:limit]

        except Exception as e:
            logger.error(f"Error listing pending requests: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_override(
        self, request_id: str, reason: str, request: Request
    ) -> Dict[str, Any]:
        """Handle governance override (admin only)."""
        try:
            # Extract RBAC context
            rbac_context = RBACCheck.extract_rbac_from_headers(request)
            if not rbac_context:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Check override permissions (admin only)
            if not RBACCheck.check_permission(rbac_context, "governance.override"):
                raise HTTPException(
                    status_code=403, detail="Override permission required"
                )

            # Get request
            if request_id not in self.pending_requests:
                raise HTTPException(status_code=404, detail="Request not found")

            request_record = self.pending_requests[request_id]

            # Override approval
            override_record = {
                "decision": "override_approved",
                "approver": rbac_context.user_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "rbac_context": rbac_context.to_dict(),
            }

            request_record["status"] = "approved"
            request_record["decision_log"].append(override_record)
            request_record["updated_at"] = datetime.utcnow()

            # Move to history
            self.request_history[request_id] = self.pending_requests.pop(request_id)

            # Log override
            await self.immutable_logger.log_governance_decision(
                {
                    "event_type": "GOVERNANCE_OVERRIDDEN",
                    "request_id": request_id,
                    "action_type": request_record["action_type"],
                    "approver": rbac_context.user_id,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return {
                "request_id": request_id,
                "status": "approved",
                "message": "Request approved via override",
                "reason": reason,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling override: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _can_submit_request(self, action_type: str, rbac_context: RBACContext) -> bool:
        """Check if user can submit this type of request."""
        policy = self.rbac_policies.get(action_type, {})
        required_permissions = policy.get("required_permissions", [])

        # Check if user has any of the required permissions
        return any(
            RBACCheck.check_permission(rbac_context, perm)
            for perm in required_permissions
        )

    def _can_approve_request(self, action_type: str, rbac_context: RBACContext) -> bool:
        """Check if user can approve this type of request."""
        policy = self.rbac_policies.get(action_type, {})
        required_roles = policy.get("required_roles", [])
        required_permissions = policy.get("required_permissions", [])

        # Check roles or permissions
        has_role = any(role in rbac_context.roles for role in required_roles)
        has_permission = any(
            RBACCheck.check_permission(rbac_context, perm)
            for perm in required_permissions
        )

        return has_role or has_permission

    def _can_view_request(self, action_type: str, rbac_context: RBACContext) -> bool:
        """Check if user can view this type of request."""
        # More permissive than approval - can view if can approve or has general governance access
        return (
            self._can_approve_request(action_type, rbac_context)
            or RBACCheck.check_permission(rbac_context, "governance.view")
            or "governance_viewer" in rbac_context.roles
        )

    def _get_required_approvals(self, action_type: str) -> int:
        """Get number of approvals required for action type."""
        policy = self.rbac_policies.get(action_type, {})
        return policy.get("min_approvers", 1)

    async def cleanup_expired_requests(self):
        """Clean up expired requests (should be run periodically)."""
        now = datetime.utcnow()
        expired_requests = []

        for request_id, record in list(self.pending_requests.items()):
            if now > record["expires_at"]:
                record["status"] = "expired"
                record["updated_at"] = now
                self.request_history[request_id] = self.pending_requests.pop(request_id)
                expired_requests.append(request_id)

        if expired_requests:
            logger.info(f"Expired {len(expired_requests)} governance requests")

        return len(expired_requests)

    async def get_stats(self) -> Dict[str, Any]:
        """Get governance API statistics."""
        return {
            "pending_requests": len(self.pending_requests),
            "total_requests": len(self.pending_requests) + len(self.request_history),
            "approval_rates": self._calculate_approval_rates(),
            "average_approval_time": self._calculate_average_approval_time(),
            "requests_by_type": self._get_requests_by_type(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _calculate_approval_rates(self) -> Dict[str, float]:
        """Calculate approval rates by action type."""
        rates = {}
        type_counts = {}
        type_approvals = {}

        for record in self.request_history.values():
            action_type = record["action_type"]
            type_counts[action_type] = type_counts.get(action_type, 0) + 1

            if record["status"] == "approved":
                type_approvals[action_type] = type_approvals.get(action_type, 0) + 1

        for action_type, total in type_counts.items():
            approved = type_approvals.get(action_type, 0)
            rates[action_type] = approved / total if total > 0 else 0

        return rates

    def _calculate_average_approval_time(self) -> float:
        """Calculate average approval time in hours."""
        approval_times = []

        for record in self.request_history.values():
            if record["status"] == "approved":
                approval_time = (
                    record["updated_at"] - record["created_at"]
                ).total_seconds()
                approval_times.append(approval_time)

        return (
            (sum(approval_times) / len(approval_times) / 3600) if approval_times else 0
        )

    def _get_requests_by_type(self) -> Dict[str, int]:
        """Get request counts by type."""
        type_counts = {}

        for record in list(self.pending_requests.values()) + list(
            self.request_history.values()
        ):
            action_type = record["action_type"]
            type_counts[action_type] = type_counts.get(action_type, 0) + 1

        return type_counts


# Compatibility wrapper for environments without FastAPI
if not FASTAPI_AVAILABLE:
    logger.warning(
        "FastAPI not available. Governance API running in compatibility mode."
    )

    class MockGovernanceAPIService:
        """Mock service for environments without FastAPI."""

        def __init__(self, **kwargs):
            self.app = None
            logger.info("Mock Governance API Service initialized")

        async def get_stats(self):
            return {"status": "mock", "message": "FastAPI not available"}

    GovernanceAPIService = MockGovernanceAPIService
