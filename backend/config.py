"""Configuration management for Grace Backend."""

import os
from functools import lru_cache
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Basic settings
    debug: bool = Field(default=False, env="DEBUG")
    secret_key: str = Field(default_factory=lambda: os.getenv("SECRET_KEY", ""), env="SECRET_KEY")
    
    # Database
    database_url: str = Field(default="postgresql://grace_user:grace_pass@localhost:5432/grace_governance", env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Vector database
    vector_db_url: str = Field(default="http://localhost:8000", env="VECTOR_DB_URL")
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")  # chroma or qdrant
    
    # Storage
    storage_type: str = Field(default="minio", env="STORAGE_TYPE")  # minio or s3
    storage_endpoint: str = Field(default="http://localhost:9000", env="STORAGE_ENDPOINT")
    storage_access_key: str = Field(default_factory=lambda: os.getenv("STORAGE_ACCESS_KEY", ""), env="STORAGE_ACCESS_KEY")
    storage_secret_key: str = Field(default_factory=lambda: os.getenv("STORAGE_SECRET_KEY", ""), env="STORAGE_SECRET_KEY")
    storage_bucket: str = Field(default="grace-media", env="STORAGE_BUCKET")
    
    # JWT
    jwt_secret_key: str = Field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""))
    jwt_algorithm: str = Field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    jwt_access_token_expire_minutes: int = Field(default_factory=lambda: int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")))
    jwt_refresh_token_expire_days: int = Field(default_factory=lambda: int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")))
    
    # Rate limiting
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    # File upload
    max_upload_size: int = Field(default=100 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 100MB
    allowed_file_types: List[str] = Field(
        default=["pdf", "docx", "txt", "html", "md", "json"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # User quotas
    default_storage_quota: int = Field(default=1024 * 1024 * 1024, env="DEFAULT_STORAGE_QUOTA")  # 1GB
    default_memory_fragments_quota: int = Field(default=10000, env="DEFAULT_MEMORY_FRAGMENTS_QUOTA")
    
    # Session limits
    max_panels_per_session: int = Field(default=20, env="MAX_PANELS_PER_SESSION")
    session_timeout_minutes: int = Field(default=480, env="SESSION_TIMEOUT_MINUTES")  # 8 hours
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # Worker settings
    worker_type: str = Field(default="celery", env="WORKER_TYPE")  # celery, rq, or arq
    worker_concurrency: int = Field(default=4, env="WORKER_CONCURRENCY")
    
    # Observability
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Policy engine
    policy_engine_type: str = Field(default="yaml", env="POLICY_ENGINE_TYPE")  # yaml or opa
    policies_path: str = Field(default="policies", env="POLICIES_PATH")
    
    # Security
    enable_antivirus: bool = Field(default=False, env="ENABLE_ANTIVIRUS")
    antivirus_endpoint: str = Field(default="", env="ANTIVIRUS_ENDPOINT")

    # Pool limits and timeouts
    pg_pool_size: int = Field(default=10, env="PG_POOL_SIZE")
    pg_pool_timeout: int = Field(default=30, env="PG_POOL_TIMEOUT")
    redis_pool_size: int = Field(default=10, env="REDIS_POOL_SIZE")
    redis_pool_timeout: int = Field(default=30, env="REDIS_POOL_TIMEOUT")

    # Circuit breaker config
    circuit_breaker_fail_max: int = Field(default=5, env="CB_FAIL_MAX")
    circuit_breaker_reset_timeout: int = Field(default=60, env="CB_RESET_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()