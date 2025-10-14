"""
Event Schema Governance system for Grace.

Provides schema registry with versioning, validation, deprecation policies,
and forward/backward compatibility management for event contracts.
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import hashlib
import re

logger = logging.getLogger(__name__)


class SchemaVersion:
    """Semantic versioning for schemas."""

    def __init__(self, version_string: str):
        self.version_string = version_string
        self.major, self.minor, self.patch = self._parse_version(version_string)

    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string."""
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ValueError(f"Invalid version format: {version}")

        parts = version.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if versions are backward compatible."""
        # Same major version is compatible
        return self.major == other.major

    def is_newer_than(self, other: "SchemaVersion") -> bool:
        """Check if this version is newer."""
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )

    def __str__(self):
        return self.version_string

    def __eq__(self, other):
        return (
            isinstance(other, SchemaVersion)
            and self.version_string == other.version_string
        )

    def __hash__(self):
        return hash(self.version_string)


class SchemaStatus(Enum):
    """Schema lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ValidationRule(Enum):
    """Schema validation rules."""

    STRICT = "strict"  # Exact schema match required
    COMPATIBLE = "compatible"  # Forward/backward compatible
    LENIENT = "lenient"  # Best effort validation


@dataclass
class SchemaField:
    """Individual schema field definition."""

    name: str
    type: str
    required: bool = True
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    default_value: Any = None

    def validate_value(self, value: Any) -> Tuple[bool, str]:
        """Validate a value against this field definition."""
        if value is None:
            if self.required and self.default_value is None:
                return False, f"Required field '{self.name}' is missing"
            return True, ""

        # Type validation
        if not self._validate_type(value):
            return (
                False,
                f"Field '{self.name}' type mismatch: expected {self.type}, got {type(value).__name__}",
            )

        # Constraints validation
        for constraint, constraint_value in self.constraints.items():
            if not self._validate_constraint(value, constraint, constraint_value):
                return False, f"Field '{self.name}' constraint violation: {constraint}"

        return True, ""

    def _validate_type(self, value: Any) -> bool:
        """Validate value type."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
            "object": dict,
            "array": list,
        }

        expected_type = type_map.get(self.type.lower())
        if expected_type is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected_type)

    def _validate_constraint(
        self, value: Any, constraint: str, constraint_value: Any
    ) -> bool:
        """Validate specific constraint."""
        if constraint == "min_length" and hasattr(value, "__len__"):
            return len(value) >= constraint_value
        elif constraint == "max_length" and hasattr(value, "__len__"):
            return len(value) <= constraint_value
        elif constraint == "min_value" and isinstance(value, (int, float)):
            return value >= constraint_value
        elif constraint == "max_value" and isinstance(value, (int, float)):
            return value <= constraint_value
        elif constraint == "pattern" and isinstance(value, str):
            return re.match(constraint_value, value) is not None
        elif constraint == "enum" and isinstance(constraint_value, list):
            return value in constraint_value

        return True  # Unknown constraint, skip


@dataclass
class EventSchema:
    """Complete event schema definition."""

    name: str
    version: SchemaVersion
    description: str
    fields: Dict[str, SchemaField]
    status: SchemaStatus = SchemaStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deprecated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    compatibility_notes: List[str] = field(default_factory=list)
    migration_rules: Dict[str, Any] = field(default_factory=dict)
    hash: str = field(init=False)

    def __post_init__(self):
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute schema content hash."""
        content = {
            "name": self.name,
            "version": str(self.version),
            "fields": {name: asdict(field) for name, field in self.fields.items()},
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_event(
        self,
        event_data: Dict[str, Any],
        rule: ValidationRule = ValidationRule.COMPATIBLE,
    ) -> Tuple[bool, List[str]]:
        """Validate event data against this schema."""
        errors = []

        if rule == ValidationRule.STRICT:
            # Check for extra fields not in schema
            event_fields = set(event_data.keys())
            schema_fields = set(self.fields.keys())
            extra_fields = event_fields - schema_fields
            if extra_fields:
                errors.append(
                    f"Extra fields not allowed in strict mode: {list(extra_fields)}"
                )

        # Validate each field
        for field_name, field_def in self.fields.items():
            value = event_data.get(field_name)
            is_valid, error = field_def.validate_value(value)
            if not is_valid:
                errors.append(error)

        # Add defaults for missing optional fields
        if rule in [ValidationRule.COMPATIBLE, ValidationRule.LENIENT]:
            for field_name, field_def in self.fields.items():
                if field_name not in event_data and field_def.default_value is not None:
                    event_data[field_name] = field_def.default_value

        return len(errors) == 0, errors

    def is_compatible_with(self, other: "EventSchema") -> Tuple[bool, List[str]]:
        """Check compatibility with another schema version."""
        compatibility_issues = []

        if self.name != other.name:
            compatibility_issues.append("Schema names don't match")
            return False, compatibility_issues

        if not self.version.is_compatible_with(other.version):
            compatibility_issues.append(
                f"Version incompatibility: {self.version} vs {other.version}"
            )

        # Check field compatibility
        for field_name, field_def in self.fields.items():
            other_field = other.fields.get(field_name)
            if other_field is None:
                if field_def.required:
                    compatibility_issues.append(
                        f"Required field '{field_name}' missing in other schema"
                    )
                continue

            # Type compatibility
            if field_def.type != other_field.type:
                compatibility_issues.append(
                    f"Field '{field_name}' type mismatch: {field_def.type} vs {other_field.type}"
                )

        return len(compatibility_issues) == 0, compatibility_issues

    def migrate_event(
        self, event_data: Dict[str, Any], target_schema: "EventSchema"
    ) -> Dict[str, Any]:
        """Migrate event data to target schema version."""
        migrated_data = event_data.copy()

        # Apply migration rules if available
        for rule_name, rule_config in self.migration_rules.items():
            migrated_data = self._apply_migration_rule(
                migrated_data, rule_name, rule_config, target_schema
            )

        # Add default values for new fields
        for field_name, field_def in target_schema.fields.items():
            if field_name not in migrated_data and field_def.default_value is not None:
                migrated_data[field_name] = field_def.default_value

        return migrated_data

    def _apply_migration_rule(
        self,
        data: Dict[str, Any],
        rule_name: str,
        rule_config: Dict[str, Any],
        target_schema: "EventSchema",
    ) -> Dict[str, Any]:
        """Apply specific migration rule."""
        if rule_name == "rename_field":
            old_name = rule_config.get("from")
            new_name = rule_config.get("to")
            if old_name in data and new_name in target_schema.fields:
                data[new_name] = data.pop(old_name)

        elif rule_name == "transform_field":
            field_name = rule_config.get("field")
            transform_type = rule_config.get("type")
            if field_name in data:
                if transform_type == "string_to_int":
                    try:
                        data[field_name] = int(data[field_name])
                    except (ValueError, TypeError):
                        pass
                elif transform_type == "split_string":
                    delimiter = rule_config.get("delimiter", ",")
                    if isinstance(data[field_name], str):
                        data[field_name] = data[field_name].split(delimiter)

        elif rule_name == "remove_field":
            field_name = rule_config.get("field")
            data.pop(field_name, None)

        return data


class SchemaRegistry:
    """
    Central schema registry for event governance.

    Manages schema versions, compatibility checks, validation,
    and schema evolution with deprecation policies.
    """

    def __init__(self, storage_path: Optional[Path] = None, event_bus=None):
        self.storage_path = storage_path or Path("schemas")
        self.event_bus = event_bus

        # Schema storage
        self.schemas: Dict[str, Dict[SchemaVersion, EventSchema]] = {}
        self.active_schemas: Dict[str, EventSchema] = {}

        # Configuration
        self.validation_rule = ValidationRule.COMPATIBLE
        self.deprecation_grace_period = timedelta(days=90)
        self.archive_after_period = timedelta(days=365)

        # Statistics
        self.validation_stats = {
            "total_validations": 0,
            "validation_failures": 0,
            "schema_migrations": 0,
            "compatibility_checks": 0,
        }

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start the schema registry."""
        if self.running:
            return

        self.running = True
        logger.info("Starting Schema Registry...")

        # Load existing schemas
        await self._load_schemas_from_storage()

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            f"Schema Registry started with {len(self.active_schemas)} active schemas"
        )

    async def stop(self):
        """Stop the schema registry."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping Schema Registry...")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Save schemas
        await self._save_schemas_to_storage()

        logger.info("Schema Registry stopped")

    async def register_schema(self, schema: EventSchema) -> bool:
        """Register a new schema or version."""
        schema_name = schema.name
        schema_version = schema.version

        # Check if schema name exists
        if schema_name not in self.schemas:
            self.schemas[schema_name] = {}

        # Check if this version already exists
        if schema_version in self.schemas[schema_name]:
            existing = self.schemas[schema_name][schema_version]
            if existing.hash != schema.hash:
                logger.warning(
                    f"Schema {schema_name} v{schema_version} already exists with different content"
                )
                return False
            return True  # Same schema already registered

        # Validate compatibility with existing versions
        compatibility_issues = []
        for existing_version, existing_schema in self.schemas[schema_name].items():
            if existing_schema.status == SchemaStatus.ACTIVE:
                compatible, issues = schema.is_compatible_with(existing_schema)
                if not compatible and schema_version.major == existing_version.major:
                    compatibility_issues.extend(issues)

        if compatibility_issues:
            logger.error(
                f"Schema compatibility issues for {schema_name} v{schema_version}: {compatibility_issues}"
            )
            return False

        # Register the schema
        self.schemas[schema_name][schema_version] = schema

        # Update active schema if this is newer
        current_active = self.active_schemas.get(schema_name)
        if current_active is None or schema.version.is_newer_than(
            current_active.version
        ):
            self.active_schemas[schema_name] = schema

        await self._save_schema_to_storage(schema)

        # Publish schema registration event
        if self.event_bus:
            await self.event_bus.publish(
                "schema_registered",
                {
                    "schema_name": schema_name,
                    "version": str(schema_version),
                    "status": schema.status.value,
                    "hash": schema.hash,
                },
            )

        logger.info(f"Registered schema {schema_name} v{schema_version}")
        return True

    async def validate_event(
        self, event_name: str, event_data: Dict[str, Any], version: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validate event against its schema."""
        self.validation_stats["total_validations"] += 1

        # Get schema
        if version:
            schema_version = SchemaVersion(version)
            schema = self.schemas.get(event_name, {}).get(schema_version)
        else:
            schema = self.active_schemas.get(event_name)

        if not schema:
            error = f"No schema found for event '{event_name}'" + (
                f" v{version}" if version else ""
            )
            self.validation_stats["validation_failures"] += 1
            return False, [error]

        # Validate
        is_valid, errors = schema.validate_event(event_data, self.validation_rule)

        if not is_valid:
            self.validation_stats["validation_failures"] += 1

            # Publish validation failure event
            if self.event_bus:
                await self.event_bus.publish(
                    "schema_validation_failed",
                    {
                        "event_name": event_name,
                        "schema_version": str(schema.version),
                        "errors": errors,
                        "event_data_keys": list(event_data.keys()),
                    },
                )

        return is_valid, errors

    async def migrate_event(
        self,
        event_name: str,
        event_data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """Migrate event data between schema versions."""
        from_schema_version = SchemaVersion(from_version)
        to_schema_version = SchemaVersion(to_version)

        from_schema = self.schemas.get(event_name, {}).get(from_schema_version)
        to_schema = self.schemas.get(event_name, {}).get(to_schema_version)

        if not from_schema:
            raise ValueError(f"Source schema not found: {event_name} v{from_version}")
        if not to_schema:
            raise ValueError(f"Target schema not found: {event_name} v{to_version}")

        # Perform migration
        migrated_data = from_schema.migrate_event(event_data, to_schema)
        self.validation_stats["schema_migrations"] += 1

        # Publish migration event
        if self.event_bus:
            await self.event_bus.publish(
                "schema_migration_performed",
                {
                    "event_name": event_name,
                    "from_version": from_version,
                    "to_version": to_version,
                    "migration_rules_applied": len(from_schema.migration_rules),
                },
            )

        return migrated_data

    async def deprecate_schema(self, event_name: str, version: str, reason: str = ""):
        """Mark a schema version as deprecated."""
        schema_version = SchemaVersion(version)
        schema = self.schemas.get(event_name, {}).get(schema_version)

        if not schema:
            raise ValueError(f"Schema not found: {event_name} v{version}")

        if schema.status == SchemaStatus.DEPRECATED:
            return  # Already deprecated

        schema.status = SchemaStatus.DEPRECATED
        schema.deprecated_at = datetime.utcnow()

        await self._save_schema_to_storage(schema)

        # Publish deprecation event
        if self.event_bus:
            await self.event_bus.publish(
                "schema_deprecated",
                {
                    "event_name": event_name,
                    "version": version,
                    "reason": reason,
                    "deprecated_at": schema.deprecated_at.isoformat(),
                    "grace_period_ends": (
                        schema.deprecated_at + self.deprecation_grace_period
                    ).isoformat(),
                },
            )

        logger.info(f"Deprecated schema {event_name} v{version}: {reason}")

    async def get_schema_info(
        self, event_name: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get schema information."""
        if version:
            schema_version = SchemaVersion(version)
            schema = self.schemas.get(event_name, {}).get(schema_version)
        else:
            schema = self.active_schemas.get(event_name)

        if not schema:
            return None

        return {
            "name": schema.name,
            "version": str(schema.version),
            "status": schema.status.value,
            "created_at": schema.created_at.isoformat(),
            "updated_at": schema.updated_at.isoformat(),
            "deprecated_at": schema.deprecated_at.isoformat()
            if schema.deprecated_at
            else None,
            "description": schema.description,
            "field_count": len(schema.fields),
            "hash": schema.hash,
            "compatibility_notes": schema.compatibility_notes,
        }

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_schemas = sum(len(versions) for versions in self.schemas.values())
        active_count = len(self.active_schemas)
        deprecated_count = sum(
            1
            for versions in self.schemas.values()
            for schema in versions.values()
            if schema.status == SchemaStatus.DEPRECATED
        )

        return {
            "total_schemas": total_schemas,
            "active_schemas": active_count,
            "deprecated_schemas": deprecated_count,
            "schema_families": len(self.schemas),
            "validation_stats": self.validation_stats.copy(),
            "running": self.running,
        }

    async def _load_schemas_from_storage(self):
        """Load schemas from persistent storage."""
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True, exist_ok=True)
            return

        loaded_count = 0

        # Load YAML schema files
        for schema_file in self.storage_path.glob("*.yaml"):
            try:
                with open(schema_file, "r") as f:
                    schema_data = yaml.safe_load(f)

                schema = self._deserialize_schema(schema_data)
                if schema:
                    if schema.name not in self.schemas:
                        self.schemas[schema.name] = {}
                    self.schemas[schema.name][schema.version] = schema

                    # Update active schema
                    current_active = self.active_schemas.get(schema.name)
                    if current_active is None or schema.version.is_newer_than(
                        current_active.version
                    ):
                        if schema.status == SchemaStatus.ACTIVE:
                            self.active_schemas[schema.name] = schema

                    loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load schema from {schema_file}: {e}")

        logger.info(f"Loaded {loaded_count} schemas from storage")

    async def _save_schemas_to_storage(self):
        """Save all schemas to persistent storage."""
        for schema_name, versions in self.schemas.items():
            for version, schema in versions.items():
                await self._save_schema_to_storage(schema)

    async def _save_schema_to_storage(self, schema: EventSchema):
        """Save individual schema to storage."""
        filename = f"{schema.name}_v{schema.version}.yaml"
        file_path = self.storage_path / filename

        try:
            schema_data = self._serialize_schema(schema)
            with open(file_path, "w") as f:
                yaml.dump(schema_data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save schema {schema.name} v{schema.version}: {e}")

    def _serialize_schema(self, schema: EventSchema) -> Dict[str, Any]:
        """Serialize schema to storage format."""
        return {
            "name": schema.name,
            "version": str(schema.version),
            "description": schema.description,
            "status": schema.status.value,
            "created_at": schema.created_at.isoformat(),
            "updated_at": schema.updated_at.isoformat(),
            "deprecated_at": schema.deprecated_at.isoformat()
            if schema.deprecated_at
            else None,
            "archived_at": schema.archived_at.isoformat()
            if schema.archived_at
            else None,
            "fields": {
                name: {
                    "type": field.type,
                    "required": field.required,
                    "description": field.description,
                    "constraints": field.constraints,
                    "default_value": field.default_value,
                }
                for name, field in schema.fields.items()
            },
            "compatibility_notes": schema.compatibility_notes,
            "migration_rules": schema.migration_rules,
            "hash": schema.hash,
        }

    def _deserialize_schema(self, data: Dict[str, Any]) -> Optional[EventSchema]:
        """Deserialize schema from storage format."""
        try:
            fields = {}
            for field_name, field_data in data.get("fields", {}).items():
                fields[field_name] = SchemaField(
                    name=field_name,
                    type=field_data.get("type", "str"),
                    required=field_data.get("required", True),
                    description=field_data.get("description", ""),
                    constraints=field_data.get("constraints", {}),
                    default_value=field_data.get("default_value"),
                )

            return EventSchema(
                name=data["name"],
                version=SchemaVersion(data["version"]),
                description=data.get("description", ""),
                fields=fields,
                status=SchemaStatus(data.get("status", "active")),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                deprecated_at=datetime.fromisoformat(data["deprecated_at"])
                if data.get("deprecated_at")
                else None,
                archived_at=datetime.fromisoformat(data["archived_at"])
                if data.get("archived_at")
                else None,
                compatibility_notes=data.get("compatibility_notes", []),
                migration_rules=data.get("migration_rules", {}),
            )
        except Exception as e:
            logger.error(f"Failed to deserialize schema: {e}")
            return None

    async def _cleanup_loop(self):
        """Background cleanup loop for old schemas."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_schemas()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Schema cleanup error: {e}")

    async def _cleanup_old_schemas(self):
        """Archive very old deprecated schemas."""
        now = datetime.utcnow()
        archived_count = 0

        for schema_name, versions in self.schemas.items():
            for version, schema in versions.items():
                if (
                    schema.status == SchemaStatus.DEPRECATED
                    and schema.deprecated_at
                    and now - schema.deprecated_at > self.archive_after_period
                ):
                    schema.status = SchemaStatus.ARCHIVED
                    schema.archived_at = now
                    archived_count += 1

                    # Publish archival event
                    if self.event_bus:
                        await self.event_bus.publish(
                            "schema_archived",
                            {
                                "event_name": schema_name,
                                "version": str(version),
                                "archived_at": now.isoformat(),
                            },
                        )

        if archived_count > 0:
            logger.info(f"Archived {archived_count} old deprecated schemas")


# Helper functions for common schema patterns


def create_basic_event_schema(event_name: str, version: str = "1.0.0") -> EventSchema:
    """Create a basic event schema with common fields."""
    fields = {
        "timestamp": SchemaField(
            "timestamp",
            "str",
            required=True,
            description="Event timestamp in ISO format",
        ),
        "source": SchemaField(
            "source", "str", required=True, description="Event source component"
        ),
        "correlation_id": SchemaField(
            "correlation_id",
            "str",
            required=False,
            description="Correlation ID for tracking",
        ),
        "payload": SchemaField(
            "payload", "object", required=True, description="Event payload data"
        ),
    }

    return EventSchema(
        name=event_name,
        version=SchemaVersion(version),
        description=f"Basic schema for {event_name} events",
        fields=fields,
    )


def create_governance_event_schema() -> EventSchema:
    """Create schema for governance validation events."""
    fields = {
        "correlation_id": SchemaField("correlation_id", "str", required=True),
        "decision_subject": SchemaField(
            "decision_subject",
            "str",
            required=True,
            constraints={"enum": ["action", "policy", "claim", "deployment"]},
        ),
        "confidence": SchemaField(
            "confidence",
            "float",
            required=True,
            constraints={"min_value": 0.0, "max_value": 1.0},
        ),
        "trust_score": SchemaField(
            "trust_score",
            "float",
            required=True,
            constraints={"min_value": 0.0, "max_value": 1.0},
        ),
        "rationale": SchemaField("rationale", "str", required=True),
        "timestamp": SchemaField("timestamp", "str", required=True),
    }

    return EventSchema(
        name="governance_decision",
        version=SchemaVersion("1.0.0"),
        description="Schema for governance decision events",
        fields=fields,
    )
