"""
Grace Orchestration State Manager - FSM, policies, and orchestrator state management.

Manages the finite state machine for orchestration, system-wide policies,
and persistent state for reliable operation and recovery.
"""

import asyncio
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OrchestrationState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class PolicyType(Enum):
    SECURITY = "security"
    GOVERNANCE = "governance"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    COMPLIANCE = "compliance"


class PolicyScope(Enum):
    GLOBAL = "global"
    KERNEL = "kernel"
    LOOP = "loop"
    TASK = "task"


class Policy:
    """System policy definition."""
    
    def __init__(self, policy_id: str, name: str, policy_type: PolicyType,
                 scope: PolicyScope, rules: Dict[str, Any], 
                 enabled: bool = True, priority: int = 5):
        self.policy_id = policy_id
        self.name = name
        self.policy_type = policy_type
        self.scope = scope
        self.rules = rules
        self.enabled = enabled
        self.priority = priority
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.applied_count = 0
        self.violation_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "type": self.policy_type.value,
            "scope": self.scope.value,
            "rules": self.rules,
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "applied_count": self.applied_count,
            "violation_count": self.violation_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """Create policy from dictionary."""
        policy = cls(
            policy_id=data["policy_id"],
            name=data["name"],
            policy_type=PolicyType(data["type"]),
            scope=PolicyScope(data["scope"]),
            rules=data["rules"],
            enabled=data.get("enabled", True),
            priority=data.get("priority", 5)
        )
        
        if "created_at" in data:
            policy.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            policy.updated_at = datetime.fromisoformat(data["updated_at"])
        if "applied_count" in data:
            policy.applied_count = data["applied_count"]
        if "violation_count" in data:
            policy.violation_count = data["violation_count"]
        
        return policy


class StateTransition:
    """State transition record."""
    
    def __init__(self, from_state: OrchestrationState, to_state: OrchestrationState,
                 trigger: str, context: Dict[str, Any] = None):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class StateManager:
    """Orchestration state and policy manager."""
    
    def __init__(self, storage_path: str = "/tmp/grace_orchestration_state"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_state = OrchestrationState.INITIALIZING
        self.state_since = datetime.now()
        self.state_context = {}
        
        # Policies
        self.policies: Dict[str, Policy] = {}
        self.policy_cache: Dict[str, List[Policy]] = {}
        
        # State machine
        self.valid_transitions = self._define_valid_transitions()
        self.state_history: List[StateTransition] = []
        
        # Database
        self.db_path = self.storage_path / "orchestration_state.db"
        self._init_database()
        
        # Configuration
        self.max_history_size = 1000
        self.policy_cache_ttl = 300  # 5 minutes
        
        # Load persisted state
        asyncio.create_task(self._load_persisted_state())
    
    def _define_valid_transitions(self) -> Dict[OrchestrationState, Set[OrchestrationState]]:
        """Define valid state transitions."""
        return {
            OrchestrationState.INITIALIZING: {
                OrchestrationState.ACTIVE,
                OrchestrationState.ERROR
            },
            OrchestrationState.ACTIVE: {
                OrchestrationState.DEGRADED,
                OrchestrationState.MAINTENANCE,
                OrchestrationState.SHUTDOWN,
                OrchestrationState.ERROR
            },
            OrchestrationState.DEGRADED: {
                OrchestrationState.ACTIVE,
                OrchestrationState.RECOVERING,
                OrchestrationState.ERROR,
                OrchestrationState.SHUTDOWN
            },
            OrchestrationState.MAINTENANCE: {
                OrchestrationState.ACTIVE,
                OrchestrationState.ERROR,
                OrchestrationState.SHUTDOWN
            },
            OrchestrationState.RECOVERING: {
                OrchestrationState.ACTIVE,
                OrchestrationState.DEGRADED,
                OrchestrationState.ERROR
            },
            OrchestrationState.ERROR: {
                OrchestrationState.RECOVERING,
                OrchestrationState.SHUTDOWN
            },
            OrchestrationState.SHUTDOWN: {}  # Terminal state
        }
    
    def _init_database(self):
        """Initialize SQLite database for state persistence."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orchestration_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    context TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    rules TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    applied_count INTEGER DEFAULT 0,
                    violation_count INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    context TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_state_timestamp ON orchestration_state(timestamp);
                CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON state_transitions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_policies_type ON policies(type);
            """)
    
    async def _load_persisted_state(self):
        """Load persisted state from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load current state
                cursor = conn.execute("""
                    SELECT state, context, timestamp FROM orchestration_state 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    state_name, context_json, timestamp = row
                    self.current_state = OrchestrationState(state_name)
                    self.state_context = json.loads(context_json)
                    self.state_since = datetime.fromisoformat(timestamp)
                
                # Load policies
                cursor = conn.execute("SELECT * FROM policies")
                for row in cursor.fetchall():
                    policy_data = {
                        "policy_id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "scope": row[3],
                        "rules": json.loads(row[4]),
                        "enabled": bool(row[5]),
                        "priority": row[6],
                        "created_at": row[7],
                        "updated_at": row[8],
                        "applied_count": row[9],
                        "violation_count": row[10]
                    }
                    policy = Policy.from_dict(policy_data)
                    self.policies[policy.policy_id] = policy
                
                logger.info(f"Loaded state: {self.current_state.value}, policies: {len(self.policies)}")
                
        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")
    
    async def transition_state(self, to_state: OrchestrationState, trigger: str,
                             context: Dict[str, Any] = None) -> bool:
        """Transition to a new orchestration state."""
        if not self._is_valid_transition(self.current_state, to_state):
            logger.warning(f"Invalid state transition: {self.current_state.value} -> {to_state.value}")
            return False
        
        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            trigger=trigger,
            context=context or {}
        )
        
        # Update current state
        old_state = self.current_state
        self.current_state = to_state
        self.state_since = datetime.now()
        self.state_context = context or {}
        
        # Add to history
        self.state_history.append(transition)
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
        
        # Persist state
        await self._persist_state()
        await self._persist_transition(transition)
        
        logger.info(f"State transition: {old_state.value} -> {to_state.value} (trigger: {trigger})")
        return True
    
    def _is_valid_transition(self, from_state: OrchestrationState, 
                           to_state: OrchestrationState) -> bool:
        """Check if state transition is valid."""
        return to_state in self.valid_transitions.get(from_state, set())
    
    async def _persist_state(self):
        """Persist current state to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO orchestration_state (state, context, timestamp)
                    VALUES (?, ?, ?)
                """, (
                    self.current_state.value,
                    json.dumps(self.state_context),
                    self.state_since.isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
    
    async def _persist_transition(self, transition: StateTransition):
        """Persist state transition to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO state_transitions (from_state, to_state, trigger, context, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    transition.from_state.value,
                    transition.to_state.value,
                    transition.trigger,
                    json.dumps(transition.context),
                    transition.timestamp.isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to persist transition: {e}")
    
    async def add_policy(self, policy: Policy) -> bool:
        """Add a new policy."""
        try:
            self.policies[policy.policy_id] = policy
            
            # Clear relevant cache
            self._clear_policy_cache(policy.policy_type, policy.scope)
            
            # Persist policy
            await self._persist_policy(policy)
            
            logger.info(f"Added policy: {policy.name} ({policy.policy_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add policy {policy.policy_id}: {e}")
            return False
    
    async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing policy."""
        if policy_id not in self.policies:
            return False
        
        try:
            policy = self.policies[policy_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            policy.updated_at = datetime.now()
            
            # Clear cache
            self._clear_policy_cache(policy.policy_type, policy.scope)
            
            # Persist changes
            await self._persist_policy(policy)
            
            logger.info(f"Updated policy: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update policy {policy_id}: {e}")
            return False
    
    async def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        if policy_id not in self.policies:
            return False
        
        try:
            policy = self.policies.pop(policy_id)
            
            # Clear cache
            self._clear_policy_cache(policy.policy_type, policy.scope)
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM policies WHERE policy_id = ?", (policy_id,))
            
            logger.info(f"Removed policy: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove policy {policy_id}: {e}")
            return False
    
    async def _persist_policy(self, policy: Policy):
        """Persist policy to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO policies 
                    (policy_id, name, type, scope, rules, enabled, priority, 
                     created_at, updated_at, applied_count, violation_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    policy.policy_id,
                    policy.name,
                    policy.policy_type.value,
                    policy.scope.value,
                    json.dumps(policy.rules),
                    int(policy.enabled),
                    policy.priority,
                    policy.created_at.isoformat(),
                    policy.updated_at.isoformat(),
                    policy.applied_count,
                    policy.violation_count
                ))
        except Exception as e:
            logger.error(f"Failed to persist policy {policy.policy_id}: {e}")
    
    def get_applicable_policies(self, policy_type: PolicyType = None, 
                              scope: PolicyScope = None,
                              context: Dict[str, Any] = None) -> List[Policy]:
        """Get policies applicable to the given context."""
        # Build cache key
        cache_key = f"{policy_type.value if policy_type else 'any'}_{scope.value if scope else 'any'}"
        
        # Check cache (with TTL)
        if cache_key in self.policy_cache:
            cached_policies, cache_time = self.policy_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.policy_cache_ttl:
                return cached_policies
        
        # Filter policies
        applicable = []
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            if policy_type and policy.policy_type != policy_type:
                continue
            
            if scope and policy.scope != scope and policy.scope != PolicyScope.GLOBAL:
                continue
            
            # Additional context-based filtering could go here
            
            applicable.append(policy)
        
        # Sort by priority (higher priority first)
        applicable.sort(key=lambda p: p.priority, reverse=True)
        
        # Cache result
        self.policy_cache[cache_key] = (applicable, datetime.now())
        
        return applicable
    
    def _clear_policy_cache(self, policy_type: PolicyType = None, 
                          scope: PolicyScope = None):
        """Clear policy cache for specific type/scope."""
        if policy_type is None and scope is None:
            self.policy_cache.clear()
            return
        
        keys_to_remove = []
        for key in self.policy_cache.keys():
            key_type, key_scope = key.split('_')
            
            should_remove = False
            if policy_type and (key_type == policy_type.value or key_type == 'any'):
                should_remove = True
            if scope and (key_scope == scope.value or key_scope == 'any'):
                should_remove = True
            
            if should_remove:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.policy_cache[key]
    
    async def evaluate_policy_compliance(self, context: Dict[str, Any],
                                       policy_type: PolicyType = None) -> Dict[str, Any]:
        """Evaluate policy compliance for a given context."""
        applicable_policies = self.get_applicable_policies(policy_type=policy_type)
        
        results = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "policies_evaluated": len(applicable_policies)
        }
        
        for policy in applicable_policies:
            try:
                # Simple rule evaluation (can be extended)
                violation = await self._evaluate_policy_rules(policy, context)
                
                if violation:
                    results["compliant"] = False
                    results["violations"].append({
                        "policy_id": policy.policy_id,
                        "policy_name": policy.name,
                        "violation": violation,
                        "severity": policy.rules.get("severity", "medium")
                    })
                    
                    # Update policy metrics
                    policy.violation_count += 1
                else:
                    # Update policy metrics
                    policy.applied_count += 1
                
            except Exception as e:
                logger.error(f"Error evaluating policy {policy.policy_id}: {e}")
                results["warnings"].append({
                    "policy_id": policy.policy_id,
                    "error": str(e)
                })
        
        return results
    
    async def _evaluate_policy_rules(self, policy: Policy, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate policy rules against context."""
        # Simple rule evaluation - can be extended with more sophisticated logic
        rules = policy.rules
        
        # Check required fields
        if "required_fields" in rules:
            for field in rules["required_fields"]:
                if field not in context:
                    return f"Required field missing: {field}"
        
        # Check value constraints
        if "constraints" in rules:
            for field, constraint in rules["constraints"].items():
                if field not in context:
                    continue
                
                value = context[field]
                
                if "min" in constraint and value < constraint["min"]:
                    return f"Value {field}={value} below minimum {constraint['min']}"
                
                if "max" in constraint and value > constraint["max"]:
                    return f"Value {field}={value} above maximum {constraint['max']}"
                
                if "allowed_values" in constraint and value not in constraint["allowed_values"]:
                    return f"Value {field}={value} not in allowed values"
        
        return None
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary."""
        return {
            "current_state": self.current_state.value,
            "state_since": self.state_since.isoformat(),
            "context": self.state_context,
            "uptime_seconds": (datetime.now() - self.state_since).total_seconds(),
            "transition_count": len(self.state_history),
            "policy_count": len(self.policies)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        return {
            "state": self.get_state_summary(),
            "policies": {
                "total": len(self.policies),
                "enabled": sum(1 for p in self.policies.values() if p.enabled),
                "by_type": {
                    policy_type.value: sum(1 for p in self.policies.values() 
                                         if p.policy_type == policy_type)
                    for policy_type in PolicyType
                },
                "by_scope": {
                    scope.value: sum(1 for p in self.policies.values() 
                                   if p.scope == scope)
                    for scope in PolicyScope
                }
            },
            "cache": {
                "entries": len(self.policy_cache),
                "hit_rate": "N/A"  # Could implement cache hit tracking
            },
            "history_size": len(self.state_history),
            "database_path": str(self.db_path)
        }