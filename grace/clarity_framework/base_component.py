"""
Class 1: Structural Ambiguity Resolution

BaseComponent provides a standardized abstract base class that all Grace components
must inherit from. This eliminates inconsistent component definitions and unclear
lifecycle methods across the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging


class ComponentStatus(Enum):
    """Standardized component status values."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ComponentMetadata:
    """Standard metadata for all Grace components."""
    component_id: str
    component_type: str
    version: str
    created_at: datetime
    last_updated: datetime
    instance_id: str


class BaseComponent(ABC):
    """
    Abstract base class for all Grace system components.
    
    Enforces standardized lifecycle methods, status tracking, and metadata
    to eliminate structural ambiguity across the system.
    """
    
    def __init__(self, component_type: str, version: str = "1.0.0"):
        self.component_id = str(uuid.uuid4())
        self.instance_id = f"{component_type}_{self.component_id[:8]}"
        self.status = ComponentStatus.INACTIVE
        self.logger = logging.getLogger(f"grace.{component_type}")
        
        self.metadata = ComponentMetadata(
            component_id=self.component_id,
            component_type=component_type,
            version=version,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            instance_id=self.instance_id
        )
        
        self._status_history = []
        self._error_count = 0
        self._last_error = None
    
    @abstractmethod
    async def activate(self) -> bool:
        """
        Activate the component and prepare it for operation.
        Returns True if activation was successful.
        """
        pass
    
    @abstractmethod
    async def deactivate(self) -> bool:
        """
        Deactivate the component and clean up resources.
        Returns True if deactivation was successful.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check component health and return status information.
        Returns dict with health metrics and status details.
        """
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get component-specific performance and operational metrics.
        Returns dict with metric names and values.
        """
        pass
    
    def get_status(self) -> ComponentStatus:
        """Get current component status."""
        return self.status
    
    def get_metadata(self) -> ComponentMetadata:
        """Get component metadata."""
        return self.metadata
    
    def update_status(self, new_status: ComponentStatus, details: Optional[str] = None):
        """Update component status with optional details."""
        old_status = self.status
        self.status = new_status
        self.metadata.last_updated = datetime.now()
        
        # Track status history
        self._status_history.append({
            "from_status": old_status.value,
            "to_status": new_status.value,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
        
        # Keep only last 100 status changes
        if len(self._status_history) > 100:
            self._status_history = self._status_history[-100:]
        
        self.logger.info(f"Status changed: {old_status.value} -> {new_status.value}")
        if details:
            self.logger.info(f"Status change details: {details}")
    
    def record_error(self, error: Exception, context: Optional[str] = None):
        """Record an error for this component."""
        self._error_count += 1
        self._last_error = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update status to error if not already in error state
        if self.status != ComponentStatus.ERROR:
            self.update_status(ComponentStatus.ERROR, f"Error: {str(error)}")
        
        self.logger.error(f"Component error recorded: {error}", exc_info=error)
    
    def get_status_history(self, limit: int = 50) -> list:
        """Get recent status change history."""
        return self._status_history[-limit:]
    
    def get_error_info(self) -> Dict[str, Any]:
        """Get error information for this component."""
        return {
            "error_count": self._error_count,
            "last_error": self._last_error
        }
    
    async def safe_activate(self) -> bool:
        """Safely activate component with error handling."""
        try:
            self.update_status(ComponentStatus.INITIALIZING, "Starting activation")
            success = await self.activate()
            
            if success:
                self.update_status(ComponentStatus.ACTIVE, "Activation completed successfully")
                return True
            else:
                self.update_status(ComponentStatus.ERROR, "Activation failed")
                return False
                
        except Exception as e:
            self.record_error(e, "During component activation")
            return False
    
    async def safe_deactivate(self) -> bool:
        """Safely deactivate component with error handling."""
        try:
            self.update_status(ComponentStatus.SHUTTING_DOWN, "Starting deactivation")
            success = await self.deactivate()
            
            if success:
                self.update_status(ComponentStatus.INACTIVE, "Deactivation completed successfully")
                return True
            else:
                self.update_status(ComponentStatus.ERROR, "Deactivation failed")
                return False
                
        except Exception as e:
            self.record_error(e, "During component deactivation")
            return False
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.instance_id}, {self.status.value})>"