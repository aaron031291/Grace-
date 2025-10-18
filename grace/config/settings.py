"""
Centralized configuration using Pydantic BaseSettings
"""

from typing import Optional, Literal, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = Field(
        default="sqlite:///./grace.db",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL queries (dev only)"
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Max overflow connections"
    )
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class AuthSettings(BaseSettings):
    """Authentication configuration"""
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="JWT secret key"
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Access token expiration (minutes)"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Refresh token expiration (days)"
    )
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if v == "dev-secret-key-change-in-production":
            logger.warning("⚠️  Using default secret key - CHANGE IN PRODUCTION!")
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        return v
    
    model_config = SettingsConfigDict(env_prefix="AUTH_")


class EmbeddingSettings(BaseSettings):
    """Embedding provider configuration"""
    provider: Literal["openai", "huggingface", "local"] = Field(
        default="huggingface",
        description="Embedding provider"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    huggingface_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name"
    )
    dimension: int = Field(
        default=384,
        ge=128,
        le=1536,
        description="Embedding dimension"
    )
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v: Optional[str], info) -> Optional[str]:
        values = info.data
        if values.get('provider') == 'openai' and not v:
            raise ValueError("OpenAI API key required when provider is 'openai'")
        return v
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration"""
    type: Literal["faiss", "pgvector"] = Field(
        default="faiss",
        description="Vector store type"
    )
    faiss_index_path: str = Field(
        default="./data/vectors/grace_index.bin",
        description="FAISS index file path"
    )
    pgvector_table: str = Field(
        default="document_embeddings",
        description="pgvector table name"
    )
    
    model_config = SettingsConfigDict(env_prefix="VECTOR_")


class SwarmSettings(BaseSettings):
    """Swarm intelligence configuration"""
    enabled: bool = Field(
        default=False,
        description="Enable swarm coordination"
    )
    node_id: Optional[str] = Field(
        default=None,
        description="Unique node identifier"
    )
    transport: Literal["http", "grpc", "kafka"] = Field(
        default="http",
        description="Transport protocol"
    )
    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Swarm communication port"
    )
    discovery_interval: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Peer discovery interval (seconds)"
    )
    
    model_config = SettingsConfigDict(env_prefix="SWARM_")


class TranscendenceSettings(BaseSettings):
    """Transcendence layer configuration"""
    quantum_enabled: bool = Field(
        default=False,
        description="Enable quantum-inspired algorithms"
    )
    discovery_enabled: bool = Field(
        default=False,
        description="Enable scientific discovery"
    )
    impact_enabled: bool = Field(
        default=False,
        description="Enable societal impact evaluation"
    )
    
    model_config = SettingsConfigDict(env_prefix="TRANSCENDENCE_")


class ObservabilitySettings(BaseSettings):
    """Observability configuration"""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    json_logs: bool = Field(
        default=True,
        description="Output logs as JSON"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Metrics endpoint port"
    )
    
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration"""
    enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for distributed rate limiting"
    )
    default_limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Default requests per window"
    )
    window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit window (seconds)"
    )
    
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")


class Settings(BaseSettings):
    """Main configuration class for Grace system"""
    
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    api_title: str = Field(
        default="Grace AI System API",
        description="API title"
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="API path prefix"
    )
    
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        description="Database settings"
    )
    auth: AuthSettings = Field(
        default_factory=AuthSettings,
        description="Authentication settings"
    )
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="Embedding settings"
    )
    vector_store: VectorStoreSettings = Field(
        default_factory=VectorStoreSettings,
        description="Vector store settings"
    )
    swarm: SwarmSettings = Field(
        default_factory=SwarmSettings,
        description="Swarm settings"
    )
    transcendence: TranscendenceSettings = Field(
        default_factory=TranscendenceSettings,
        description="Transcendence settings"
    )
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings,
        description="Observability settings"
    )
    rate_limit: RateLimitSettings = Field(
        default_factory=RateLimitSettings,
        description="Rate limiting settings"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def validate_production_config(self) -> List[str]:
        """Validate configuration for production deployment"""
        issues: List[str] = []
        
        if self.environment == "production":
            if self.auth.secret_key == "dev-secret-key-change-in-production":
                issues.append("CRITICAL: Using default secret key in production!")
            
            if self.debug:
                issues.append("WARNING: Debug mode enabled in production")
            
            if self.database.echo:
                issues.append("WARNING: SQL echo enabled in production")
            
            if not self.rate_limit.enabled:
                issues.append("WARNING: Rate limiting disabled in production")
            
            if self.embedding.provider == "openai" and not self.embedding.openai_api_key:
                issues.append("ERROR: OpenAI provider selected but no API key")
        
        return issues
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "api_version": self.api_version,
            "features": {
                "swarm": self.swarm.enabled,
                "quantum": self.transcendence.quantum_enabled,
                "discovery": self.transcendence.discovery_enabled,
                "impact": self.transcendence.impact_enabled,
                "metrics": self.observability.metrics_enabled,
                "rate_limiting": self.rate_limit.enabled
            },
            "providers": {
                "embedding": self.embedding.provider,
                "vector_store": self.vector_store.type,
                "transport": self.swarm.transport if self.swarm.enabled else None
            }
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Usage:
        from grace.config import get_settings
        settings = get_settings()
    """
    settings = Settings()
    
    # Validate production config
    if settings.environment == "production":
        issues = settings.validate_production_config()
        for issue in issues:
            if "CRITICAL" in issue or "ERROR" in issue:
                logger.error(issue)
            else:
                logger.warning(issue)
    
    logger.info(f"Configuration loaded: {settings.environment} environment")
    return settings


# Backwards compatibility exports (deprecated - use get_settings() instead)
def _get_setting_value(key: str, default=None):
    """Get setting value - deprecated"""
    import warnings
    warnings.warn(
        "Direct setting imports are deprecated. Use get_settings() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    settings = get_settings()
    return getattr(settings, key, default)
