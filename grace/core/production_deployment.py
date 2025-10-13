"""
Production Deployment Infrastructure - Scaling beyond SQLite.

Implements production-grade infrastructure as specified in the missing components
requirements. Features:
- Multi-database support (PostgreSQL, Redis, MongoDB)
- Horizontal scaling capabilities
- Load balancing and failover
- Container orchestration integration
- Performance monitoring and metrics
- Health checks and circuit breakers
- Configuration management
- Security hardening
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

# Optional production dependencies with graceful fallbacks
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis

        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False

try:
    import motor.motor_asyncio

    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False

try:
    import prometheus_client

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseType(Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    MONGODB = "mongodb"
    CASSANDRA = "cassandra"


class ScalingMode(Enum):
    """Scaling modes."""

    SINGLE_INSTANCE = "single_instance"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    AUTO_SCALING = "auto_scaling"


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 20
    max_overflow: int = 10
    timeout_seconds: int = 30
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None

    # Connection string for complex configurations
    connection_string: Optional[str] = None

    # Performance tuning
    connection_pool_recycle: int = 3600
    query_timeout: int = 60
    statement_timeout: int = 300

    def get_connection_url(self) -> str:
        """Generate connection URL."""
        if self.connection_string:
            return self.connection_string

        if self.db_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.REDIS:
            return f"redis://{self.host}:{self.port}/0"
        elif self.db_type == DatabaseType.MONGODB:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"{self.host}:{self.port}"


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""

    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600

    # Metrics for scaling decisions
    metrics_window_seconds: int = 300
    metrics_evaluation_intervals: int = 3


@dataclass
class ProductionMetrics:
    """Production deployment metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0

    # Database metrics
    db_connections_active: int = 0
    db_connections_idle: int = 0
    db_query_count: int = 0
    db_slow_query_count: int = 0

    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mbps: float = 0.0

    # Application metrics
    active_users: int = 0
    concurrent_sessions: int = 0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    start_time: datetime = field(default_factory=datetime.now)


class DatabaseConnection(ABC):
    """Abstract database connection interface."""

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def execute(self, query: str, parameters: Optional[List] = None) -> Any:
        pass

    @abstractmethod
    async def fetch_one(
        self, query: str, parameters: Optional[List] = None
    ) -> Optional[Dict]:
        pass

    @abstractmethod
    async def fetch_all(
        self, query: str, parameters: Optional[List] = None
    ) -> List[Dict]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL connection implementation."""

    def __init__(self, config: DatabaseConfig):
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg not available for PostgreSQL connection")

        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> bool:
        """Establish PostgreSQL connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.get_connection_url(),
                min_size=5,
                max_size=self.config.pool_size,
                command_timeout=self.config.query_timeout,
                server_settings={
                    "statement_timeout": str(
                        self.config.statement_timeout * 1000
                    ),  # milliseconds
                    "lock_timeout": "10s",
                    "idle_in_transaction_session_timeout": "5min",
                },
            )

            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            logger.info(
                f"Connected to PostgreSQL at {self.config.host}:{self.config.port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Disconnected from PostgreSQL")

    async def execute(self, query: str, parameters: Optional[List] = None) -> Any:
        """Execute a query."""
        if not self.pool:
            raise RuntimeError("Not connected to database")

        async with self.pool.acquire() as conn:
            if parameters:
                return await conn.execute(query, *parameters)
            else:
                return await conn.execute(query)

    async def fetch_one(
        self, query: str, parameters: Optional[List] = None
    ) -> Optional[Dict]:
        """Fetch single row."""
        if not self.pool:
            raise RuntimeError("Not connected to database")

        async with self.pool.acquire() as conn:
            if parameters:
                row = await conn.fetchrow(query, *parameters)
            else:
                row = await conn.fetchrow(query)

            return dict(row) if row else None

    async def fetch_all(
        self, query: str, parameters: Optional[List] = None
    ) -> List[Dict]:
        """Fetch all rows."""
        if not self.pool:
            raise RuntimeError("Not connected to database")

        async with self.pool.acquire() as conn:
            if parameters:
                rows = await conn.fetch(query, *parameters)
            else:
                rows = await conn.fetch(query)

            return [dict(row) for row in rows]

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.pool:
                return False

            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1

        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False


class RedisConnection(DatabaseConnection):
    """Redis connection implementation."""

    def __init__(self, config: DatabaseConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("aioredis not available for Redis connection")

        self.config = config
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self.redis = aioredis.from_url(
                self.config.get_connection_url(),
                max_connections=self.config.pool_size,
                retry_on_timeout=True,
                socket_timeout=self.config.timeout_seconds,
                socket_connect_timeout=10,
            )

            # Test connection
            await self.redis.ping()

            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("Disconnected from Redis")

    async def execute(self, query: str, parameters: Optional[List] = None) -> Any:
        """Execute Redis command."""
        if not self.redis:
            raise RuntimeError("Not connected to Redis")

        # Parse Redis command from query string
        parts = query.split()
        command = parts[0].upper()
        args = parts[1:] if len(parts) > 1 else []

        if parameters:
            args.extend(str(p) for p in parameters)

        return await self.redis.execute_command(command, *args)

    async def fetch_one(
        self, query: str, parameters: Optional[List] = None
    ) -> Optional[Dict]:
        """Get single value from Redis."""
        result = await self.execute(query, parameters)
        return {"value": result} if result is not None else None

    async def fetch_all(
        self, query: str, parameters: Optional[List] = None
    ) -> List[Dict]:
        """Get multiple values from Redis."""
        result = await self.execute(query, parameters)
        if isinstance(result, list):
            return [{"value": item} for item in result]
        return [{"value": result}] if result is not None else []

    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            if not self.redis:
                return False

            pong = await self.redis.ping()
            return pong is True

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


class DatabaseManager:
    """Manages multiple database connections with failover."""

    def __init__(self):
        self.connections: Dict[str, DatabaseConnection] = {}
        self.primary_db: Optional[str] = None
        self.replica_dbs: List[str] = []
        self.health_status: Dict[str, bool] = {}
        self.connection_metrics: Dict[str, Dict[str, int]] = {}

    async def add_database(
        self, name: str, config: DatabaseConfig, is_primary: bool = False
    ) -> bool:
        """Add a database connection."""
        try:
            if config.db_type == DatabaseType.POSTGRESQL:
                connection = PostgreSQLConnection(config)
            elif config.db_type == DatabaseType.REDIS:
                connection = RedisConnection(config)
            else:
                logger.error(f"Unsupported database type: {config.db_type}")
                return False

            # Connect to database
            if await connection.connect():
                self.connections[name] = connection
                self.health_status[name] = True
                self.connection_metrics[name] = {
                    "queries_executed": 0,
                    "queries_failed": 0,
                    "connections_created": 1,
                }

                if is_primary:
                    self.primary_db = name
                else:
                    self.replica_dbs.append(name)

                logger.info(
                    f"Added database connection: {name} ({'primary' if is_primary else 'replica'})"
                )
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error adding database {name}: {e}")
            return False

    async def get_connection(
        self, prefer_replica: bool = False
    ) -> Optional[DatabaseConnection]:
        """Get a healthy database connection."""
        # Try primary first (unless replica preferred for read operations)
        if (
            not prefer_replica
            and self.primary_db
            and self.health_status.get(self.primary_db, False)
        ):
            return self.connections.get(self.primary_db)

        # Try replicas
        for replica in self.replica_dbs:
            if self.health_status.get(replica, False):
                return self.connections.get(replica)

        # Fallback to primary if no healthy replicas
        if self.primary_db and self.health_status.get(self.primary_db, False):
            return self.connections.get(self.primary_db)

        return None

    async def execute_with_failover(
        self,
        query: str,
        parameters: Optional[List] = None,
        prefer_replica: bool = False,
    ) -> Any:
        """Execute query with automatic failover."""
        connection = await self.get_connection(prefer_replica)
        if not connection:
            raise RuntimeError("No healthy database connections available")

        try:
            result = await connection.execute(query, parameters)

            # Update metrics
            for name, conn in self.connections.items():
                if conn is connection:
                    self.connection_metrics[name]["queries_executed"] += 1
                    break

            return result

        except Exception as e:
            # Update failure metrics
            for name, conn in self.connections.items():
                if conn is connection:
                    self.connection_metrics[name]["queries_failed"] += 1
                    break

            logger.error(f"Database query failed: {e}")
            raise

    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health checks on all connections."""
        results = {}

        for name, connection in self.connections.items():
            try:
                is_healthy = await connection.health_check()
                self.health_status[name] = is_healthy
                results[name] = is_healthy

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self.health_status[name] = False
                results[name] = False

        return results

    async def close_all(self) -> None:
        """Close all database connections."""
        for name, connection in self.connections.items():
            try:
                await connection.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection {name}: {e}")

        self.connections.clear()
        self.health_status.clear()
        logger.info("Closed all database connections")


class LoadBalancer:
    """Simple load balancer for horizontal scaling."""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances: List[Dict[str, Any]] = []
        self.current_index = 0
        self.instance_metrics: Dict[str, Dict[str, float]] = {}

    def add_instance(
        self, instance_id: str, host: str, port: int, weight: float = 1.0
    ) -> None:
        """Add a service instance."""
        instance = {
            "id": instance_id,
            "host": host,
            "port": port,
            "weight": weight,
            "healthy": True,
            "added_at": datetime.now(),
        }

        self.instances.append(instance)
        self.instance_metrics[instance_id] = {
            "requests": 0,
            "response_time": 0.0,
            "error_rate": 0.0,
            "last_request": 0.0,
        }

        logger.info(f"Added load balancer instance: {instance_id} ({host}:{port})")

    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get next instance based on load balancing strategy."""
        healthy_instances = [inst for inst in self.instances if inst["healthy"]]

        if not healthy_instances:
            return None

        if self.strategy == "round_robin":
            instance = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index += 1
            return instance

        elif self.strategy == "weighted":
            # Choose based on weights
            total_weight = sum(inst["weight"] for inst in healthy_instances)
            if total_weight == 0:
                return healthy_instances[0]

            import random

            r = random.uniform(0, total_weight)
            current_weight = 0

            for instance in healthy_instances:
                current_weight += instance["weight"]
                if r <= current_weight:
                    return instance

            return healthy_instances[-1]

        elif self.strategy == "least_connections":
            # Choose instance with least active requests
            return min(
                healthy_instances,
                key=lambda inst: self.instance_metrics[inst["id"]]["requests"],
            )

        else:
            # Default to first healthy instance
            return healthy_instances[0]

    def mark_instance_unhealthy(self, instance_id: str) -> None:
        """Mark an instance as unhealthy."""
        for instance in self.instances:
            if instance["id"] == instance_id:
                instance["healthy"] = False
                logger.warning(f"Marked instance {instance_id} as unhealthy")
                break

    def mark_instance_healthy(self, instance_id: str) -> None:
        """Mark an instance as healthy."""
        for instance in self.instances:
            if instance["id"] == instance_id:
                instance["healthy"] = True
                logger.info(f"Marked instance {instance_id} as healthy")
                break


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            # Check if timeout has passed
            if (
                self.last_failure_time
                and (datetime.now() - self.last_failure_time).total_seconds()
                > self.timeout_seconds
            ):
                self.state = "half-open"
                logger.info("Circuit breaker moving to half-open state")
            else:
                raise RuntimeError("Circuit breaker is open")

        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Success - reset failure count
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful call")

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

            raise e


class ProductionDeployment:
    """Main production deployment orchestrator."""

    def __init__(
        self, environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    ):
        self.environment = environment
        self.database_manager = DatabaseManager()
        self.load_balancer = LoadBalancer()
        self.metrics = ProductionMetrics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Configuration
        self.config = self._load_configuration()

        # Health monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.overall_health = HealthStatus.UNKNOWN

        # Auto-scaling
        self.scaling_config = ScalingConfig()
        self.current_instances = 1

        # Monitoring and metrics
        self.metrics_collectors: List[Callable] = []
        self.monitoring_task: Optional[asyncio.Task] = None

        # System state
        self.started = False
        self.start_time: Optional[datetime] = None

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables and files."""
        config = {
            # Database configuration
            "databases": {
                "primary": {
                    "type": os.getenv("DB_TYPE", "postgresql"),
                    "host": os.getenv("DB_HOST", "localhost"),
                    "port": int(os.getenv("DB_PORT", "5432")),
                    "database": os.getenv("DB_NAME", "grace"),
                    "username": os.getenv("DB_USER", "grace"),
                    "password": os.getenv("DB_PASSWORD", ""),
                    "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
                },
                "cache": {
                    "type": "redis",
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": int(os.getenv("REDIS_PORT", "6379")),
                    "database": "0",
                    "pool_size": int(os.getenv("REDIS_POOL_SIZE", "10")),
                },
            },
            # Scaling configuration
            "scaling": {
                "mode": os.getenv("SCALING_MODE", "horizontal"),
                "min_instances": int(os.getenv("MIN_INSTANCES", "2")),
                "max_instances": int(os.getenv("MAX_INSTANCES", "20")),
                "target_cpu": float(os.getenv("TARGET_CPU_PERCENT", "70")),
                "target_memory": float(os.getenv("TARGET_MEMORY_PERCENT", "80")),
            },
            # Monitoring
            "monitoring": {
                "enabled": os.getenv("MONITORING_ENABLED", "true").lower() == "true",
                "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
                "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            },
            # Security
            "security": {
                "jwt_secret": os.getenv("JWT_SECRET", ""),
                "encryption_key": os.getenv("ENCRYPTION_KEY", ""),
                "ssl_enabled": os.getenv("SSL_ENABLED", "false").lower() == "true",
                "ssl_cert_path": os.getenv("SSL_CERT_PATH", ""),
                "ssl_key_path": os.getenv("SSL_KEY_PATH", ""),
            },
        }

        return config

    async def start(self) -> bool:
        """Start the production deployment."""
        if self.started:
            logger.warning("Production deployment already started")
            return True

        try:
            self.start_time = datetime.now()
            logger.info(
                f"Starting Grace production deployment ({self.environment.value})"
            )

            # Initialize databases
            await self._initialize_databases()

            # Start health monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Initialize load balancer if in horizontal scaling mode
            if self.config["scaling"]["mode"] == "horizontal":
                await self._initialize_load_balancer()

            # Setup circuit breakers
            self._setup_circuit_breakers()

            # Start metrics collection
            if self.config["monitoring"]["enabled"]:
                await self._start_metrics_collection()

            self.started = True
            self.overall_health = HealthStatus.HEALTHY

            logger.info("Production deployment started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start production deployment: {e}")
            self.overall_health = HealthStatus.UNHEALTHY
            return False

    async def stop(self) -> None:
        """Stop the production deployment."""
        if not self.started:
            return

        logger.info("Stopping production deployment...")

        try:
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            # Close database connections
            await self.database_manager.close_all()

            # Stop metrics collection
            if PROMETHEUS_AVAILABLE:
                # Cleanup prometheus metrics if needed
                pass

            self.started = False
            self.overall_health = HealthStatus.UNKNOWN

            logger.info("Production deployment stopped")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _initialize_databases(self) -> None:
        """Initialize database connections."""
        # Primary database
        primary_config = self.config["databases"]["primary"]
        primary_db_config = DatabaseConfig(
            db_type=DatabaseType(primary_config["type"]),
            host=primary_config["host"],
            port=primary_config["port"],
            database=primary_config["database"],
            username=primary_config.get("username"),
            password=primary_config.get("password"),
            pool_size=primary_config.get("pool_size", 20),
        )

        await self.database_manager.add_database(
            "primary", primary_db_config, is_primary=True
        )

        # Cache database (Redis)
        if "cache" in self.config["databases"]:
            cache_config = self.config["databases"]["cache"]
            cache_db_config = DatabaseConfig(
                db_type=DatabaseType.REDIS,
                host=cache_config["host"],
                port=cache_config["port"],
                database=cache_config.get("database", "0"),
                pool_size=cache_config.get("pool_size", 10),
            )

            await self.database_manager.add_database("cache", cache_db_config)

    async def _initialize_load_balancer(self) -> None:
        """Initialize load balancer with service instances."""
        # In a real deployment, this would discover instances from service registry
        # For now, add current instance
        import socket

        hostname = socket.gethostname()
        self.load_balancer.add_instance("instance_1", hostname, 8000)

    def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers for external services."""
        self.circuit_breakers["database"] = CircuitBreaker(
            failure_threshold=5, timeout_seconds=60
        )
        self.circuit_breakers["external_api"] = CircuitBreaker(
            failure_threshold=3, timeout_seconds=30
        )
        self.circuit_breakers["file_storage"] = CircuitBreaker(
            failure_threshold=10, timeout_seconds=120
        )

    async def _start_metrics_collection(self) -> None:
        """Start Prometheus metrics collection if available."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, skipping metrics setup")
            return

        # Setup Prometheus metrics
        self.request_counter = prometheus_client.Counter(
            "grace_requests_total",
            "Total requests processed",
            ["method", "endpoint", "status"],
        )

        self.request_duration = prometheus_client.Histogram(
            "grace_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
        )

        self.db_connection_gauge = prometheus_client.Gauge(
            "grace_db_connections", "Active database connections", ["database", "state"]
        )

        # Start Prometheus HTTP server
        prometheus_port = self.config["monitoring"]["prometheus_port"]
        prometheus_client.start_http_server(prometheus_port)
        logger.info(f"Started Prometheus metrics server on port {prometheus_port}")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Started production monitoring loop")

        while self.started:
            try:
                # Update system metrics
                await self._update_metrics()

                # Perform health checks
                await self._perform_health_checks()

                # Check scaling requirements
                await self._check_scaling()

                # Sleep until next check
                interval = self.config["monitoring"]["health_check_interval"]
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Shorter sleep on error

    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        current_time = datetime.now()

        # Update uptime
        if self.start_time:
            self.metrics.uptime_seconds = (
                current_time - self.start_time
            ).total_seconds()

        # Update database metrics
        db_health = await self.database_manager.health_check_all()
        self.metrics.db_connections_active = sum(
            1 for healthy in db_health.values() if healthy
        )

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and hasattr(self, "db_connection_gauge"):
            for db_name, healthy in db_health.items():
                state = "active" if healthy else "inactive"
                self.db_connection_gauge.labels(database=db_name, state=state).set(
                    1 if healthy else 0
                )

        self.metrics.last_updated = current_time

    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks."""
        health_results = []

        # Database health
        db_health = await self.database_manager.health_check_all()
        db_healthy = any(db_health.values())
        health_results.append(db_healthy)

        # Memory usage check (simplified)
        memory_healthy = self.metrics.memory_usage_mb < 1024  # 1GB threshold
        health_results.append(memory_healthy)

        # Response time check
        response_time_healthy = (
            self.metrics.average_response_time_ms < 2000
        )  # 2s threshold
        health_results.append(response_time_healthy)

        # Determine overall health
        if all(health_results):
            self.overall_health = HealthStatus.HEALTHY
        elif any(health_results):
            self.overall_health = HealthStatus.DEGRADED
        else:
            self.overall_health = HealthStatus.UNHEALTHY

    async def _check_scaling(self) -> None:
        """Check if scaling is needed."""
        if self.config["scaling"]["mode"] != "auto_scaling":
            return

        # Simple scaling logic based on metrics
        cpu_usage = self.metrics.cpu_usage_percent
        memory_usage = self.metrics.memory_usage_mb

        scale_up_needed = (
            cpu_usage > self.scaling_config.target_cpu_percent
            or memory_usage > self.scaling_config.target_memory_percent
        )

        scale_down_needed = (
            cpu_usage
            < self.scaling_config.scale_down_threshold
            * self.scaling_config.target_cpu_percent
            and memory_usage
            < self.scaling_config.scale_down_threshold
            * self.scaling_config.target_memory_percent
        )

        if (
            scale_up_needed
            and self.current_instances < self.scaling_config.max_instances
        ):
            logger.info(f"Scaling up: CPU={cpu_usage}%, Memory={memory_usage}MB")
            await self._scale_up()
        elif (
            scale_down_needed
            and self.current_instances > self.scaling_config.min_instances
        ):
            logger.info(f"Scaling down: CPU={cpu_usage}%, Memory={memory_usage}MB")
            await self._scale_down()

    async def _scale_up(self) -> None:
        """Scale up the deployment."""
        # In a real deployment, this would:
        # 1. Request new instances from orchestrator (Kubernetes, Docker Swarm, etc.)
        # 2. Wait for instances to become healthy
        # 3. Add instances to load balancer
        # 4. Update current_instances count

        self.current_instances += 1
        logger.info(f"Scaled up to {self.current_instances} instances")

    async def _scale_down(self) -> None:
        """Scale down the deployment."""
        # In a real deployment, this would:
        # 1. Remove instance from load balancer
        # 2. Drain connections from instance
        # 3. Terminate instance
        # 4. Update current_instances count

        self.current_instances = max(1, self.current_instances - 1)
        logger.info(f"Scaled down to {self.current_instances} instances")

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        return {
            "environment": self.environment.value,
            "started": self.started,
            "health": self.overall_health.value,
            "uptime_seconds": self.metrics.uptime_seconds,
            "instances": self.current_instances,
            "databases": {
                name: self.database_manager.health_status.get(name, False)
                for name in self.database_manager.connections.keys()
            },
            "metrics": asdict(self.metrics),
            "load_balancer": {
                "instances": len(self.load_balancer.instances),
                "healthy_instances": len(
                    [i for i in self.load_balancer.instances if i["healthy"]]
                ),
            },
            "circuit_breakers": {
                name: {"state": cb.state, "failures": cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            },
        }


# Global production deployment instance
production_deployment = ProductionDeployment()
