# Grace Governance Kernel Configuration

# Default governance thresholds
GOVERNANCE_THRESHOLDS = {
    "min_confidence": 0.78,
    "min_trust": 0.72,
    "constitutional_compliance_min": 0.85,
    "shadow_switchover_accuracy_delta": 0.02,
    "rollback_compliance_threshold": 0.98,
    "anomaly_tolerance": 0.1
}

# Database paths and connection configuration
DATABASE_CONFIG = {
    "memory_db_path": "grace_governance.db",  # SQLite fallback
    "audit_db_path": "governance_audit.db",  # SQLite fallback
    "postgres_url": "postgresql://grace_user:grace_pass@localhost:5432/grace_governance",
    "redis_url": "redis://:grace_redis_pass@localhost:6379",
    "chroma_url": "http://localhost:8000",
    "use_postgres": True,  # Use PostgreSQL in production
    "use_redis_cache": True,
    "use_chroma_vectors": True
}

# Event routing configuration
EVENT_ROUTING = {
    "priority_workers": {
        "critical": 4,
        "high": 3,
        "normal": 2,
        "low": 1
    },
    "timeout_defaults": {
        "critical": 500,    # ms
        "high": 1000,       # ms
        "normal": 2000,     # ms
        "low": 10000        # ms
    }
}

# MLDL specialist configuration
MLDL_CONFIG = {
    "min_participating_specialists": 11,  # Majority of 21
    "consensus_threshold": 0.65,
    "confidence_threshold": 0.6,
    "prediction_timeout": 30.0  # seconds
}

# Parliament configuration
PARLIAMENT_CONFIG = {
    "voting_thresholds": {
        "policy": 0.6,          # 60% for policy changes
        "constitutional": 0.75,  # 75% for constitutional changes
        "operational": 0.5       # 50% for operational decisions
    },
    "review_deadlines": {
        "critical": 1,  # days
        "high": 3,      # days
        "normal": 7,    # days
        "low": 14       # days
    }
}

# Trust system configuration
TRUST_CONFIG = {
    "decay_rate": 0.01,  # Daily decay rate
    "min_interactions_for_reliability": 5,
    "stale_profile_threshold_days": 90
}

# Health monitoring configuration
HEALTH_CONFIG = {
    "monitoring_interval": 30,      # seconds
    "predictive_window": 300,       # seconds
    "anomaly_thresholds": {
        "decision_latency": {
            "warning": 2.0,     # seconds
            "error": 5.0,       # seconds
            "critical": 10.0    # seconds
        },
        "confidence_score": {
            "warning": 0.6,
            "error": 0.4,
            "critical": 0.2
        },
        "trust_score": {
            "warning": 0.7,
            "error": 0.5,
            "critical": 0.3
        }
    }
}

# Audit log configuration
AUDIT_CONFIG = {
    "transparency_levels": {
        "public": {
            "retention_days": 2555,  # ~7 years
            "access_level": 0
        },
        "democratic_oversight": {
            "retention_days": 1825,  # ~5 years
            "access_level": 1
        },
        "governance_internal": {
            "retention_days": 365,   # 1 year
            "access_level": 2
        },
        "audit_only": {
            "retention_days": 2555,  # ~7 years
            "access_level": 3
        },
        "security_sensitive": {
            "retention_days": 90,    # 90 days
            "access_level": 4
        }
    }
}

# Constitutional principles
CONSTITUTIONAL_PRINCIPLES = {
    "transparency": {
        "description": "All decisions must be transparent and auditable",
        "weight": 1.0,
        "required": True
    },
    "fairness": {
        "description": "Decisions must be fair and unbiased",
        "weight": 1.0,
        "required": True
    },
    "accountability": {
        "description": "Decision makers must be accountable",
        "weight": 0.9,
        "required": True
    },
    "consistency": {
        "description": "Similar cases should have similar outcomes",
        "weight": 0.8,
        "required": True
    },
    "harm_prevention": {
        "description": "Decisions must not cause unnecessary harm",
        "weight": 1.0,
        "required": True
    }
}

# AI Provider configuration
AI_CONFIG = {
    "openai": {
        "api_key": None,  # Set via environment variable
        "org_id": None,   # Set via environment variable
        "model_default": "gpt-4",
        "max_tokens": 4096
    },
    "anthropic": {
        "api_key": None,  # Set via environment variable  
        "model_default": "claude-3-sonnet-20240229",
        "max_tokens": 4096
    }
}

# Environment integration
ENVIRONMENT_CONFIG = {
    "instance_id": "grace_main_001",
    "version": "1.0.0",
    "log_level": "INFO",
    "debug_mode": False,
    "api_host": "0.0.0.0",
    "api_port": 8080,
    "orchestrator_port": 8081
}

# Infrastructure configuration  
INFRASTRUCTURE_CONFIG = {
    "enable_telemetry": True,
    "enable_health_monitoring": True,
    "metrics_export_interval": 30,
    "auto_rollback_enabled": True,
    "governance_strict_mode": True,
    "constitutional_enforcement": True
}

# Complete configuration
GRACE_CONFIG = {
    "governance_thresholds": GOVERNANCE_THRESHOLDS,
    "database_config": DATABASE_CONFIG,
    "event_routing": EVENT_ROUTING,
    "mldl_config": MLDL_CONFIG,
    "parliament_config": PARLIAMENT_CONFIG,
    "trust_config": TRUST_CONFIG,
    "health_config": HEALTH_CONFIG,
    "audit_config": AUDIT_CONFIG,
    "constitutional_principles": CONSTITUTIONAL_PRINCIPLES,
    "ai_config": AI_CONFIG,
    "environment_config": ENVIRONMENT_CONFIG,
    "infrastructure_config": INFRASTRUCTURE_CONFIG
}