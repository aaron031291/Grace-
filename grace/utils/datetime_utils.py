"""
ISO 8601 compliant datetime utilities for Grace system.

Provides centralized, timezone-aware datetime handling with full ISO 8601 compliance.
All datetime operations use UTC timezone and proper ISO formatting.
"""

from datetime import datetime, timezone
from typing import Optional, Union
import time

# ISO 8601 datetime format with timezone (e.g., "2024-01-15T10:30:45.123456+00:00")
ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"

def utc_now() -> datetime:
    """
    Get current UTC datetime with timezone information.
    
    Replaces deprecated datetime.utcnow() with timezone-aware alternative.
    
    Returns:
        datetime: Current UTC time with timezone info
    """
    return datetime.now(timezone.utc)

def utc_timestamp() -> float:
    """
    Get current UTC timestamp as float.
    
    Returns:
        float: UTC timestamp
    """
    return utc_now().timestamp()

def iso_format(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as ISO 8601 string with timezone information.
    
    Args:
        dt: datetime to format, defaults to current UTC time
        
    Returns:
        str: ISO 8601 formatted datetime string
    """
    if dt is None:
        dt = utc_now()
    
    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.isoformat()

def parse_iso(iso_string: str) -> datetime:
    """
    Parse ISO 8601 datetime string to timezone-aware datetime.
    
    Args:
        iso_string: ISO 8601 formatted datetime string
        
    Returns:
        datetime: Parsed timezone-aware datetime
        
    Raises:
        ValueError: If string is not valid ISO 8601 format
    """
    try:
        # Handle various ISO 8601 formats
        if iso_string.endswith('Z'):
            # Replace Z with +00:00 for proper parsing
            iso_string = iso_string[:-1] + '+00:00'
        
        return datetime.fromisoformat(iso_string)
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime string '{iso_string}': {e}")

def datetime_from_timestamp(timestamp: Union[float, int], tz: timezone = timezone.utc) -> datetime:
    """
    Create timezone-aware datetime from timestamp.
    
    Args:
        timestamp: Unix timestamp
        tz: Target timezone, defaults to UTC
        
    Returns:
        datetime: Timezone-aware datetime object
    """
    return datetime.fromtimestamp(timestamp, tz=tz)

def ensure_timezone_aware(dt: datetime, default_tz: timezone = timezone.utc) -> datetime:
    """
    Ensure datetime object is timezone-aware.
    
    Args:
        dt: datetime object
        default_tz: Default timezone if datetime is naive
        
    Returns:
        datetime: Timezone-aware datetime
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz)
    return dt

def format_for_filename(dt: Optional[datetime] = None) -> str:
    """
    Format datetime for safe use in filenames.
    
    Args:
        dt: datetime to format, defaults to current UTC time
        
    Returns:
        str: Filename-safe datetime string (YYYYMMDD_HHMMSS format)
    """
    if dt is None:
        dt = utc_now()
    
    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.strftime('%Y%m%d_%H%M%S')

def format_for_audit(dt: Optional[datetime] = None) -> str:
    """
    Format datetime for audit logging with microsecond precision.
    
    Args:
        dt: datetime to format, defaults to current UTC time
        
    Returns:
        str: High-precision ISO 8601 datetime for audit trails
    """
    if dt is None:
        dt = utc_now()
    
    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.isoformat(timespec='microseconds')

# Legacy compatibility aliases (deprecated, use utc_now() instead)
def get_utc_now() -> datetime:
    """DEPRECATED: Use utc_now() instead."""
    import warnings
    warnings.warn("get_utc_now() is deprecated, use utc_now() instead", DeprecationWarning, stacklevel=2)
    return utc_now()

def get_iso_timestamp() -> str:
    """DEPRECATED: Use iso_format() instead."""
    import warnings
    warnings.warn("get_iso_timestamp() is deprecated, use iso_format() instead", DeprecationWarning, stacklevel=2)
    return iso_format()
