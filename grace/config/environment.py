"""
Environment configuration loader for Grace Governance system.
Loads configuration from environment variables and integrates with existing config.
"""
import os
from typing import Dict, Any, Optional
from .governance_config import GRACE_CONFIG


class EnvironmentLoader:
    """Load and integrate environment variables with Grace configuration."""
    
    def __init__(self):
        self._loaded = False
        self._config = None
    
    def load_environment(self) -> Dict[str, Any]:
        """Load environment variables and merge with base configuration."""
        if self._loaded and self._config:
            return self._config
            
        # Copy base configuration
        config = GRACE_CONFIG.copy()
        
        # Load AI provider keys
        config["ai_config"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
        config["ai_config"]["openai"]["org_id"] = os.getenv("OPENAI_ORG_ID")
        config["ai_config"]["anthropic"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        # Load database URLs
        config["database_config"]["postgres_url"] = os.getenv(
            "DATABASE_URL", 
            config["database_config"]["postgres_url"]
        )
        config["database_config"]["redis_url"] = os.getenv(
            "REDIS_URL",
            config["database_config"]["redis_url"]
        )
        config["database_config"]["chroma_url"] = os.getenv(
            "CHROMA_URL",
            config["database_config"]["chroma_url"]
        )
        
        # Load instance configuration
        config["environment_config"]["instance_id"] = os.getenv(
            "GRACE_INSTANCE_ID",
            config["environment_config"]["instance_id"]
        )
        config["environment_config"]["version"] = os.getenv(
            "GRACE_VERSION",
            config["environment_config"]["version"]
        )
        config["environment_config"]["log_level"] = os.getenv(
            "GRACE_LOG_LEVEL",
            config["environment_config"]["log_level"]
        )
        config["environment_config"]["debug_mode"] = os.getenv(
            "GRACE_DEBUG", "false"
        ).lower() == "true"
        
        # Load service configuration
        config["environment_config"]["api_host"] = os.getenv(
            "API_HOST",
            config["environment_config"]["api_host"]
        )
        config["environment_config"]["api_port"] = int(os.getenv(
            "API_PORT",
            str(config["environment_config"]["api_port"])
        ))
        config["environment_config"]["orchestrator_port"] = int(os.getenv(
            "ORCHESTRATOR_PORT",
            str(config["environment_config"]["orchestrator_port"])
        ))
        
        # Load infrastructure settings
        config["infrastructure_config"]["enable_telemetry"] = os.getenv(
            "ENABLE_TELEMETRY", "true"
        ).lower() == "true"
        config["infrastructure_config"]["enable_health_monitoring"] = os.getenv(
            "ENABLE_HEALTH_MONITORING", "true"
        ).lower() == "true"
        config["infrastructure_config"]["auto_rollback_enabled"] = os.getenv(
            "AUTO_ROLLBACK_ENABLED", "true"
        ).lower() == "true"
        config["infrastructure_config"]["governance_strict_mode"] = os.getenv(
            "GOVERNANCE_STRICT_MODE", "true"
        ).lower() == "true"
        config["infrastructure_config"]["constitutional_enforcement"] = os.getenv(
            "CONSTITUTIONAL_ENFORCEMENT", "true"
        ).lower() == "true"
        
        # Load feature flags
        config["database_config"]["use_postgres"] = os.getenv(
            "USE_POSTGRES", "true"
        ).lower() == "true"
        config["database_config"]["use_redis_cache"] = os.getenv(
            "USE_REDIS_CACHE", "true"
        ).lower() == "true"
        config["database_config"]["use_chroma_vectors"] = os.getenv(
            "USE_CHROMA_VECTORS", "true"
        ).lower() == "true"
        
        # Load MLDL configuration
        if os.getenv("MLDL_MIN_SPECIALISTS"):
            config["mldl_config"]["min_participating_specialists"] = int(
                os.getenv("MLDL_MIN_SPECIALISTS")
            )
        if os.getenv("MLDL_MAX_SPECIALISTS"):
            config["mldl_config"]["max_participating_specialists"] = int(
                os.getenv("MLDL_MAX_SPECIALISTS", "5")
            )
        if os.getenv("MLDL_CONSENSUS_THRESHOLD"):
            config["mldl_config"]["consensus_threshold"] = float(
                os.getenv("MLDL_CONSENSUS_THRESHOLD")
            )
            
        # Load monitoring configuration
        if os.getenv("METRICS_EXPORT_INTERVAL"):
            config["infrastructure_config"]["metrics_export_interval"] = int(
                os.getenv("METRICS_EXPORT_INTERVAL")
            )
        
        self._config = config
        self._loaded = True
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if not self._loaded:
            return self.load_environment()
        return self._config
    
    def validate_required_env_vars(self) -> list[str]:
        """Validate that required environment variables are set."""
        missing = []
        
        # Check for AI provider keys (at least one required)
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")
            
        # Check database URLs if postgres is enabled
        if os.getenv("USE_POSTGRES", "true").lower() == "true":
            if not os.getenv("DATABASE_URL"):
                missing.append("DATABASE_URL")
                
        if os.getenv("USE_REDIS_CACHE", "true").lower() == "true":
            if not os.getenv("REDIS_URL"):
                missing.append("REDIS_URL")
                
        if os.getenv("USE_CHROMA_VECTORS", "true").lower() == "true":
            if not os.getenv("CHROMA_URL"):
                missing.append("CHROMA_URL")
                
        return missing


# Global instance
env_loader = EnvironmentLoader()

def get_grace_config() -> Dict[str, Any]:
    """Get the Grace configuration with environment variables loaded."""
    return env_loader.get_config()

def validate_environment() -> list[str]:
    """Validate required environment variables are set."""
    return env_loader.validate_required_env_vars()