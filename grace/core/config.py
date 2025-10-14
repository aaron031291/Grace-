"""
Grace Configuration Management

Handles environment variables and settings for the Grace system.
"""

import os
import logging
from typing import Optional
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class GraceSettings(BaseSettings):
    """Grace system configuration settings."""

    # Service Configuration
    service_mode: str = Field(default="api", env="SERVICE_MODE")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    worker_queues: str = Field(
        default="ingestion,embeddings,media", env="WORKER_QUEUES"
    )

    # Database Configuration
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")

    # Object Storage Configuration
    s3_endpoint: Optional[str] = Field(default=None, env="S3_ENDPOINT")
    s3_access_key: Optional[str] = Field(default=None, env="S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = Field(default=None, env="S3_SECRET_KEY")
    s3_bucket_name: str = Field(default="grace-storage", env="S3_BUCKET_NAME")

    # Vector Database Configuration
    vector_url: Optional[str] = Field(default=None, env="VECTOR_URL")

    # Core Configuration
    grace_instance_id: str = Field(default="grace_main_001", env="GRACE_INSTANCE_ID")
    grace_version: str = Field(default="1.0.0", env="GRACE_VERSION")
    log_level: str = Field(default="INFO", env="GRACE_LOG_LEVEL")
    debug: bool = Field(default=False, env="GRACE_DEBUG")

    # Security Configuration
    jwt_secret_key: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")

    # Governance Configuration
    governance_strict_mode: bool = Field(default=True, env="GOVERNANCE_STRICT_MODE")
    constitutional_enforcement: bool = Field(
        default=True, env="CONSTITUTIONAL_ENFORCEMENT"
    )
    auto_rollback_enabled: bool = Field(default=True, env="AUTO_ROLLBACK_ENABLED")

    # Monitoring Configuration
    enable_telemetry: bool = Field(default=True, env="ENABLE_TELEMETRY")
    enable_health_monitoring: bool = Field(default=True, env="ENABLE_HEALTH_MONITORING")
    metrics_export_interval: int = Field(default=30, env="METRICS_EXPORT_INTERVAL")
    prometheus_endpoint: str = Field(
        default="http://localhost:9090", env="PROMETHEUS_ENDPOINT"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> GraceSettings:
    """Get cached settings instance."""
    return GraceSettings()


def setup_environment():
    """Set up environment variables and logging configuration."""
    settings = get_settings()

    # Configure logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/grace.log")
            if os.path.exists("logs")
            else logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Grace {settings.grace_version} - {settings.grace_instance_id}")
    logger.info(f"Service mode: {settings.service_mode}")
    logger.info(f"Debug mode: {settings.debug}")

    return settings
