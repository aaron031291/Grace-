"""
Centralized configuration for Grace system
"""

from typing import Optional, Dict, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GraceConfig:
    """
    Central configuration manager
    
    Loads from environment variables with fallback defaults
    """
    
    def __init__(self):
        # Environment
        self.environment = os.getenv("GRACE_ENV", "development")
        self.debug = os.getenv("GRACE_DEBUG", "false").lower() == "true"
        
        # Service
        self.service_name = os.getenv("GRACE_SERVICE_NAME", "grace-unified")
        self.service_host = os.getenv("GRACE_HOST", "0.0.0.0")
        self.service_port = int(os.getenv("GRACE_PORT", "8000"))
        
        # Database
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./grace.db")
        self.database_pool_size = int(os.getenv("DATABASE_POOL_SIZE", "5"))
        
        # Redis
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_enabled = os.getenv("REDIS_ENABLED", "true").lower() == "true"
        
        # Event Bus
        self.event_bus_max_queue = int(os.getenv("EVENT_BUS_MAX_QUEUE", "10000"))
        self.event_bus_dlq_max = int(os.getenv("EVENT_BUS_DLQ_MAX", "1000"))
        self.event_bus_ttl_cleanup = os.getenv("EVENT_BUS_TTL_CLEANUP", "true").lower() == "true"
        
        # Kernels
        self.kernels_enabled = os.getenv("KERNELS_ENABLED", "multi_os,mldl,resilience").split(",")
        self.kernels_auto_start = os.getenv("KERNELS_AUTO_START", "true").lower() == "true"
        
        # Memory
        self.memory_lightning_enabled = os.getenv("MEMORY_LIGHTNING_ENABLED", "true").lower() == "true"
        self.memory_fusion_enabled = os.getenv("MEMORY_FUSION_ENABLED", "true").lower() == "true"
        self.memory_vector_enabled = os.getenv("MEMORY_VECTOR_ENABLED", "false").lower() == "true"
        self.memory_fanout_enabled = os.getenv("MEMORY_FANOUT_ENABLED", "true").lower() == "true"
        
        # Governance
        self.governance_enabled = os.getenv("GOVERNANCE_ENABLED", "true").lower() == "true"
        self.governance_auto_escalate = os.getenv("GOVERNANCE_AUTO_ESCALATE", "true").lower() == "true"
        
        # Trust
        self.trust_enabled = os.getenv("TRUST_ENABLED", "true").lower() == "true"
        self.trust_default_threshold = float(os.getenv("TRUST_DEFAULT_THRESHOLD", "0.7"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "json")  # json or text
        self.log_file = os.getenv("LOG_FILE", None)
        
        # Health
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        self.health_reporters_enabled = os.getenv("HEALTH_REPORTERS", "true").lower() == "true"
        
        # Scheduler
        self.scheduler_enabled = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
        
        # Watchdog
        self.watchdog_enabled = os.getenv("WATCHDOG_ENABLED", "true").lower() == "true"
        self.watchdog_restart_on_failure = os.getenv("WATCHDOG_RESTART", "false").lower() == "true"
        
        # TriggerMesh
        self.trigger_mesh_config = os.getenv("TRIGGER_MESH_CONFIG", "config/trigger_mesh.yaml")
        self.trigger_mesh_enabled = os.getenv("TRIGGER_MESH_ENABLED", "true").lower() == "true"
        
        # Validate critical config
        self._validate()
    
    def _validate(self):
        """Validate critical configuration"""
        errors = []
        
        if self.service_port < 1024 or self.service_port > 65535:
            errors.append(f"Invalid port: {self.service_port}")
        
        if self.event_bus_max_queue < 100:
            errors.append(f"Event queue too small: {self.event_bus_max_queue}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info(f"Configuration validated for {self.environment} environment")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self) -> str:
        return f"<GraceConfig env={self.environment} service={self.service_name}>"


# Global configuration instance
_config: Optional[GraceConfig] = None


def get_config() -> GraceConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = GraceConfig()
    return _config


def reload_config():
    """Reload configuration from environment"""
    global _config
    _config = GraceConfig()
    return _config
