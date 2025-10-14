"""
Grace Core Utilities - Central helper functions for common patterns across Grace.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Type variable for enum types
EnumT = TypeVar("EnumT", bound=Enum)


def enum_from_str(
    enum_class: Type[EnumT], value: str, *, default: Optional[EnumT] = None
) -> EnumT:
    """
    Central helper to parse enum values from strings with flexible matching.

    Attempts to match by:
    1. Exact enum value (e.g., "chat" for PanelType.CHAT)
    2. Uppercase enum name (e.g., "CHAT" for PanelType.CHAT)
    3. Lowercase enum value (e.g., "chat" for PanelType.CHAT)
    4. Returns default if provided, otherwise raises ValueError

    Args:
        enum_class: The enum class to parse into
        value: String value to parse
        default: Optional default value to return if parsing fails

    Returns:
        Enum instance

    Raises:
        ValueError: If value cannot be parsed and no default provided

    Examples:
        >>> enum_from_str(PanelType, "chat")
        PanelType.CHAT
        >>> enum_from_str(PanelType, "CHAT")
        PanelType.CHAT
        >>> enum_from_str(PanelType, "invalid", default=PanelType.DASHBOARD)
        PanelType.DASHBOARD
    """
    if not isinstance(value, str):
        if default is not None:
            return default
        raise ValueError(
            f"Expected string value for enum parsing, got {type(value)}: {value}"
        )

    # Try exact value match first (most common case)
    try:
        return enum_class(value)
    except ValueError:
        pass

    # Try lowercase value match
    try:
        return enum_class(value.lower())
    except ValueError:
        pass

    # Try uppercase name match
    try:
        return enum_class[value.upper()]
    except (ValueError, KeyError):
        pass

    # Try value match against all enum values (case insensitive)
    value_lower = value.lower()
    for enum_item in enum_class:
        if enum_item.value.lower() == value_lower:
            return enum_item

    # Return default or raise error
    if default is not None:
        return default

    valid_values = [item.value for item in enum_class]
    valid_names = [item.name for item in enum_class]
    raise ValueError(
        f"Invalid {enum_class.__name__} value: '{value}'. "
        f"Valid values: {valid_values} or names: {valid_names}"
    )


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:12]}"


def utc_timestamp() -> str:
    """Generate UTC timestamp in ISO8601 format with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_timestamp(dt: Union[datetime, str, None]) -> Optional[str]:
    """
    Normalize datetime to UTC ISO8601 with Z suffix.

    Args:
        dt: datetime object, ISO string, or None

    Returns:
        Normalized UTC ISO8601 string with Z or None
    """
    if dt is None:
        return None

    if isinstance(dt, str):
        try:
            # Parse string timestamp
            if dt.endswith("Z"):
                dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            elif "+" in dt or dt.endswith("+00:00"):
                dt_obj = datetime.fromisoformat(dt)
            else:
                # Assume naive datetime is UTC
                dt_obj = datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.warning(f"Could not parse timestamp '{dt}': {e}")
            return None
    elif isinstance(dt, datetime):
        dt_obj = dt
        # Convert naive datetime to UTC
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        # Convert to UTC if not already
        elif dt_obj.tzinfo != timezone.utc:
            dt_obj = dt_obj.astimezone(timezone.utc)
    else:
        logger.warning(f"Unsupported timestamp type: {type(dt)}")
        return None

    return dt_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def create_error_response(
    code: str, message: str, detail: Optional[str] = None, status_code: int = 400
) -> Dict[str, Any]:
    """
    Create consistent error response envelope.

    Args:
        code: Error code (e.g., "INVALID_ENUM", "VALIDATION_ERROR")
        message: Human-readable error message
        detail: Optional additional detail
        status_code: HTTP status code

    Returns:
        Error response dictionary with consistent structure
    """
    error_response = {"error": {"code": code, "message": message}}

    if detail:
        error_response["error"]["detail"] = detail

    return error_response


def validate_request_size(
    content: Any, max_size_mb: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Validate request content size.

    Args:
        content: Content to validate (string, bytes, or dict with size info)
        max_size_mb: Maximum size in megabytes

    Returns:
        Error response dict if too large, None if valid
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    try:
        if isinstance(content, str):
            size_bytes = len(content.encode("utf-8"))
        elif isinstance(content, bytes):
            size_bytes = len(content)
        elif isinstance(content, dict) and "size" in content:
            size_bytes = content["size"]
        else:
            # Can't determine size, allow it
            return None

        if size_bytes > max_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            return create_error_response(
                "CONTENT_TOO_LARGE",
                f"Content size {size_mb:.1f}MB exceeds limit of {max_size_mb}MB",
                f"Request body size: {size_bytes} bytes",
                413,
            )

    except Exception as e:
        logger.warning(f"Could not validate content size: {e}")

    return None
