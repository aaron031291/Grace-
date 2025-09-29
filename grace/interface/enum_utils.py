"""
Utilities for safe enum parsing and mapping across the Grace interface.
"""
from typing import Type, TypeVar, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

E = TypeVar('E', bound=Enum)


def safe_enum_parse(enum_class: Type[E], value: Any, default: Optional[E] = None) -> Optional[E]:
    """
    Safely parse an enum value from string or enum, with fallback to default.
    
    Args:
        enum_class: The enum class to parse into
        value: String value, enum value, or any other type
        default: Default value if parsing fails
        
    Returns:
        Parsed enum value or default
    """
    if value is None:
        return default
        
    # If already the correct enum type, return it
    if isinstance(value, enum_class):
        return value
        
    # Convert to string for parsing
    str_value = str(value).strip()
    
    if not str_value:
        return default
    
    # Try direct value match (case insensitive)
    try:
        return enum_class(str_value.lower())
    except ValueError:
        pass
    
    # Try name match (case insensitive)
    try:
        return enum_class[str_value.upper()]
    except KeyError:
        pass
    
    # Try partial matches for common variations
    for enum_item in enum_class:
        # Check if string matches enum value
        if str_value.lower() == enum_item.value.lower():
            return enum_item
        # Check if string matches enum name
        if str_value.upper() == enum_item.name.upper():
            return enum_item
        # Check for partial matches (for backwards compatibility)
        if str_value.lower().replace('_', '') == enum_item.value.lower().replace('_', ''):
            return enum_item
    
    logger.warning(f"Could not parse '{value}' as {enum_class.__name__}, using default: {default}")
    return default


def create_enum_mapper(mapping_dict: dict, enum_class: Type[E], default: Optional[E] = None):
    """
    Create a safe mapper function for converting legacy string values to enum values.
    
    Args:
        mapping_dict: Dictionary of legacy string -> enum mappings
        enum_class: Target enum class
        default: Default enum value for unmapped strings
        
    Returns:
        Function that safely maps strings to enum values
    """
    def mapper(value: Any) -> Optional[E]:
        # If it's already an enum of the right type, return it
        if isinstance(value, enum_class):
            return value
            
        str_value = str(value).strip()
        
        # Check explicit mapping first
        if str_value in mapping_dict:
            return mapping_dict[str_value]
            
        # Fall back to safe enum parsing
        return safe_enum_parse(enum_class, value, default)
    
    return mapper