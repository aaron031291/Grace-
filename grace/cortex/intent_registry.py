"""
Global Intent Registry - Manages pod intents across the GRACE system
Enhanced version with timezone-aware timestamps
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from threading import Lock
import json
import logging
import uuid
import os

logger = logging.getLogger(__name__)


class IntentStatus(Enum):
    """Status of an intent in the system"""
    REGISTERED = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REVOKED = auto()


class IntentPriority(Enum):
    """Priority levels for intents"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Intent:
    """Intent data structure"""
    id: str
    pod_id: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    priority: IntentPriority
    status: IntentStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    status_history: List[Dict[str, Any]] = field(default_factory=list)


class GlobalIntentRegistry:
    """
    Registry for managing pod intents across the GRACE system
    PRODUCTION IMPLEMENTATION with full persistence
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = logging.getLogger("grace.cortex.intent_registry")
        self.intents: Dict[str, Intent] = {}
        self.pod_intents: Dict[str, Set[str]] = {}
        self.intent_lock = Lock()
        
        # Setup storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "intents"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Intent registry initialized with storage at {self.storage_path}")
        
        # Load existing intents
        self._load_intents()
    
    def _load_intents(self) -> None:
        """Load intents from storage"""
        try:
            intent_files = list(self.storage_path.glob("*.json"))
            self.logger.info(f"Loading {len(intent_files)} intents from storage")
            
            for intent_file in intent_files:
                try:
                    with open(intent_file, "r") as f:
                        intent_data = json.load(f)
                    
                    intent_id = intent_data.get("id")
                    if not intent_id:
                        continue
                    
                    # Convert strings back to enums and datetimes
                    if "status" in intent_data and isinstance(intent_data["status"], str):
                        intent_data["status"] = IntentStatus[intent_data["status"]]
                    
                    if "priority" in intent_data and isinstance(intent_data["priority"], str):
                        intent_data["priority"] = IntentPriority[intent_data["priority"]]
                    
                    if "created_at" in intent_data:
                        intent_data["created_at"] = datetime.fromisoformat(intent_data["created_at"])
                    
                    if "updated_at" in intent_data:
                        intent_data["updated_at"] = datetime.fromisoformat(intent_data["updated_at"])
                    
                    # Create Intent object
                    intent = Intent(**intent_data)
                    
                    with self.intent_lock:
                        self.intents[intent_id] = intent
                        if intent.pod_id not in self.pod_intents:
                            self.pod_intents[intent.pod_id] = set()
                        self.pod_intents[intent.pod_id].add(intent_id)
                    
                except Exception as e:
                    self.logger.error(f"Error loading intent from {intent_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error loading intents: {e}")
    
    def _save_intent(self, intent_id: str) -> bool:
        """Save intent to storage"""
        try:
            with self.intent_lock:
                if intent_id not in self.intents:
                    return False
                
                intent = self.intents[intent_id]
                
                # Convert to serializable dict
                intent_data = {
                    "id": intent.id,
                    "pod_id": intent.pod_id,
                    "description": intent.description,
                    "parameters": intent.parameters,
                    "dependencies": intent.dependencies,
                    "priority": intent.priority.name,
                    "status": intent.status.name,
                    "created_at": intent.created_at.isoformat(),
                    "updated_at": intent.updated_at.isoformat(),
                    "metadata": intent.metadata,
                    "status_history": intent.status_history
                }
                
                file_path = self.storage_path / f"{intent_id}.json"
                with open(file_path, "w") as f:
                    json.dump(intent_data, f, indent=2)
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error saving intent {intent_id}: {e}")
            return False
    
    def register_intent(
        self,
        pod_id: str,
        intent_data: Dict[str, Any]
    ) -> Intent:
        """Register a new intent from a pod"""
        try:
            intent_id = intent_data.get("id", str(uuid.uuid4()))
            timestamp = datetime.now(timezone.utc)  # FIXED: timezone-aware
            
            # Parse priority
            priority = intent_data.get("priority", IntentPriority.MEDIUM)
            if isinstance(priority, str):
                try:
                    priority = IntentPriority[priority.upper()]
                except KeyError:
                    priority = IntentPriority.MEDIUM
            
            # Create intent
            intent = Intent(
                id=intent_id,
                pod_id=pod_id,
                description=intent_data.get("description", ""),
                parameters=intent_data.get("parameters", {}),
                dependencies=intent_data.get("dependencies", []),
                priority=priority,
                status=IntentStatus.REGISTERED,
                created_at=timestamp,
                updated_at=timestamp,
                metadata=intent_data.get("metadata", {})
            )
            
            # Register
            with self.intent_lock:
                self.intents[intent_id] = intent
                if pod_id not in self.pod_intents:
                    self.pod_intents[pod_id] = set()
                self.pod_intents[pod_id].add(intent_id)
            
            # Save to storage
            self._save_intent(intent_id)
            
            self.logger.info(f"Registered intent {intent_id} from pod {pod_id}")
            
            return intent
        
        except Exception as e:
            self.logger.error(f"Error registering intent from pod {pod_id}: {e}")
            raise ValueError(f"Failed to register intent: {e}")
    
    def update_intent_status(
        self,
        intent_id: str,
        status: IntentStatus,
        details: Optional[Dict[str, Any]] = None
    ) -> Intent:
        """Update intent status"""
        try:
            with self.intent_lock:
                if intent_id not in self.intents:
                    raise ValueError(f"Intent {intent_id} not found")
                
                intent = self.intents[intent_id]
                old_status = intent.status
                
                intent.status = status
                intent.updated_at = datetime.now(timezone.utc)  # FIXED: timezone-aware
                
                # Add to history
                history_entry = {
                    "status": status.name,
                    "previous_status": old_status.name,
                    "timestamp": intent.updated_at.isoformat(),
                    "details": details or {}
                }
                intent.status_history.append(history_entry)
                
                # Save changes
                self._save_intent(intent_id)
                
                self.logger.info(f"Updated intent {intent_id} status to {status.name}")
                
                return intent
        
        except Exception as e:
            self.logger.error(f"Error updating intent {intent_id} status: {e}")
            raise ValueError(f"Failed to update intent status: {e}")
    
    def get_intent(self, intent_id: str) -> Intent:
        """Get intent by ID"""
        with self.intent_lock:
            if intent_id not in self.intents:
                raise ValueError(f"Intent {intent_id} not found")
            return self.intents[intent_id]
    
    def get_pod_intents(self, pod_id: str) -> List[Intent]:
        """Get all intents for a pod"""
        with self.intent_lock:
            if pod_id not in self.pod_intents:
                return []
            return [self.intents[iid] for iid in self.pod_intents[pod_id]]
    
    def validate_intent(self, intent_id: str) -> Tuple[bool, str]:
        """Validate an intent"""
        try:
            with self.intent_lock:
                if intent_id not in self.intents:
                    return False, "Intent not found"
                
                intent = self.intents[intent_id]
                
                # Check dependencies
                for dep_id in intent.dependencies:
                    if dep_id not in self.intents:
                        return False, f"Dependency {dep_id} not found"
                    
                    dep_status = self.intents[dep_id].status
                    if dep_status != IntentStatus.COMPLETED:
                        return False, f"Dependency {dep_id} is not completed"
                
                return True, ""
        
        except Exception as e:
            self.logger.error(f"Error validating intent {intent_id}: {e}")
            return False, f"Validation error: {e}"
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get intent statistics"""
        with self.intent_lock:
            total = len(self.intents)
            
            status_counts = {status.name: 0 for status in IntentStatus}
            priority_counts = {priority.name: 0 for priority in IntentPriority}
            
            for intent in self.intents.values():
                status_counts[intent.status.name] += 1
                priority_counts[intent.priority.name] += 1
            
            return {
                "total": total,
                "by_status": status_counts,
                "by_priority": priority_counts,
                "by_pod": {pod_id: len(intents) for pod_id, intents in self.pod_intents.items()}
            }
