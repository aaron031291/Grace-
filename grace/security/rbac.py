"""
Role-Based Access Control (RBAC) implementation
"""

from typing import Dict, Set, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # Read permissions
    READ_EVENTS = "read:events"
    READ_METRICS = "read:metrics"
    READ_LOGS = "read:logs"
    READ_KERNELS = "read:kernels"
    READ_MEMORY = "read:memory"
    
    # Write permissions
    WRITE_EVENTS = "write:events"
    WRITE_MEMORY = "write:memory"
    
    # Admin permissions
    MANAGE_KERNELS = "manage:kernels"
    MANAGE_USERS = "manage:users"
    MANAGE_POLICIES = "manage:policies"
    
    # Governance permissions
    VALIDATE_EVENTS = "validate:events"
    ESCALATE_ISSUES = "escalate:issues"
    OVERRIDE_TRUST = "override:trust"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    CONSTITUTIONAL_OVERRIDE = "constitutional:override"


class Role(Enum):
    """System roles"""
    GUEST = "guest"
    USER = "user"
    OPERATOR = "operator"
    GOVERNANCE_REVIEWER = "governance_reviewer"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class RoleDefinition:
    """Role with assigned permissions"""
    name: Role
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    constitutional_required: bool = False


class RBACManager:
    """
    RBAC Manager - Enforces role-based access control
    
    Constitutional compliance:
    - All administrative actions require constitutional validation
    - Permission changes are logged immutably
    - Trust scores affect permission grants
    """
    
    def __init__(self, governance_engine=None, immutable_logs=None):
        self.governance = governance_engine
        self.logs = immutable_logs
        
        self.roles: Dict[Role, RoleDefinition] = {}
        self.user_roles: Dict[str, Set[Role]] = {}
        
        self._define_roles()
    
    def _define_roles(self):
        """Define system roles and their permissions"""
        # Guest - minimal read access
        self.roles[Role.GUEST] = RoleDefinition(
            name=Role.GUEST,
            permissions={
                Permission.READ_METRICS,
            },
            description="Read-only access to public metrics"
        )
        
        # User - standard read access
        self.roles[Role.USER] = RoleDefinition(
            name=Role.USER,
            permissions={
                Permission.READ_EVENTS,
                Permission.READ_METRICS,
                Permission.READ_LOGS,
                Permission.WRITE_EVENTS,
            },
            description="Standard user with read/write event access"
        )
        
        # Operator - manage kernels
        self.roles[Role.OPERATOR] = RoleDefinition(
            name=Role.OPERATOR,
            permissions={
                Permission.READ_EVENTS,
                Permission.READ_METRICS,
                Permission.READ_LOGS,
                Permission.READ_KERNELS,
                Permission.READ_MEMORY,
                Permission.WRITE_EVENTS,
                Permission.WRITE_MEMORY,
                Permission.MANAGE_KERNELS,
            },
            description="Operator with kernel management access",
            constitutional_required=True
        )
        
        # Governance Reviewer - validate and escalate
        self.roles[Role.GOVERNANCE_REVIEWER] = RoleDefinition(
            name=Role.GOVERNANCE_REVIEWER,
            permissions={
                Permission.READ_EVENTS,
                Permission.READ_METRICS,
                Permission.READ_LOGS,
                Permission.VALIDATE_EVENTS,
                Permission.ESCALATE_ISSUES,
            },
            description="Governance reviewer with validation powers",
            constitutional_required=True
        )
        
        # Admin - full access except constitutional override
        self.roles[Role.ADMIN] = RoleDefinition(
            name=Role.ADMIN,
            permissions=set(p for p in Permission if p != Permission.CONSTITUTIONAL_OVERRIDE),
            description="Administrator with full access",
            constitutional_required=True
        )
        
        # System - full access including constitutional override
        self.roles[Role.SYSTEM] = RoleDefinition(
            name=Role.SYSTEM,
            permissions=set(Permission),
            description="System role with constitutional override",
            constitutional_required=True
        )
    
    async def assign_role(
        self,
        user_id: str,
        role: Role,
        assigned_by: str,
        trust_score: float = 1.0
    ) -> bool:
        """
        Assign role to user with constitutional validation
        
        Args:
            user_id: User identifier
            role: Role to assign
            assigned_by: Who is assigning the role
            trust_score: User's trust score
        
        Returns:
            True if assignment successful
        """
        role_def = self.roles.get(role)
        if not role_def:
            logger.error(f"Unknown role: {role}")
            return False
        
        # Constitutional validation for privileged roles
        if role_def.constitutional_required:
            if self.governance:
                from grace.schemas.events import GraceEvent
                
                event = GraceEvent(
                    event_type="rbac.role_assignment",
                    source=assigned_by,
                    payload={
                        "user_id": user_id,
                        "role": role.value,
                        "assigned_by": assigned_by
                    },
                    constitutional_validation_required=True,
                    trust_score=trust_score
                )
                
                result = await self.governance.validate(event)
                if not result.passed:
                    logger.warning(f"Role assignment blocked by governance: {result.violations}")
                    return False
        
        # Assign role
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role)
        
        # Log assignment
        if self.logs:
            await self.logs.log(
                operation_type="rbac_role_assigned",
                actor=assigned_by,
                action={
                    "user_id": user_id,
                    "role": role.value
                },
                result={"success": True},
                severity="info"
            )
        
        logger.info(f"Role {role.value} assigned to {user_id} by {assigned_by}")
        return True
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if user has permission
        
        Args:
            user_id: User identifier
            permission: Permission to check
        
        Returns:
            True if user has permission
        """
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            role_def = self.roles.get(role)
            if role_def and permission in role_def.permissions:
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user"""
        permissions = set()
        
        user_roles = self.user_roles.get(user_id, set())
        for role in user_roles:
            role_def = self.roles.get(role)
            if role_def:
                permissions.update(role_def.permissions)
        
        return permissions
    
    async def check_constitutional_compliance(
        self,
        user_id: str,
        action: str,
        context: Dict
    ) -> bool:
        """
        Check if action complies with constitutional principles
        
        All privileged operations must pass this check
        """
        if not self.governance:
            logger.warning("Governance engine not available for constitutional check")
            return True
        
        from grace.schemas.events import GraceEvent
        
        event = GraceEvent(
            event_type=f"constitutional.check.{action}",
            source=user_id,
            payload=context,
            constitutional_validation_required=True
        )
        
        result = await self.governance.validate(event)
        
        if not result.passed:
            logger.warning(f"Constitutional compliance check failed for {user_id}: {result.violations}")
        
        return result.passed
