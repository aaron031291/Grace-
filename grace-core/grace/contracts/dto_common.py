"""
Common DTO utilities, pagination, cursors, and enums.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class SortOrder(str, Enum):
    """Sort order for queries."""
    ASC = "asc"
    DESC = "desc"


class FilterOperator(str, Enum):
    """Filter operators for queries."""
    EQ = "eq"          # equals
    NE = "ne"          # not equals
    GT = "gt"          # greater than
    GTE = "gte"        # greater than or equal
    LT = "lt"          # less than
    LTE = "lte"        # less than or equal
    IN = "in"          # in list
    NOT_IN = "not_in"  # not in list
    LIKE = "like"      # pattern matching
    REGEX = "regex"    # regex matching
    EXISTS = "exists"  # field exists
    NULL = "null"      # field is null


class FilterCondition(BaseModel):
    """A single filter condition."""
    field: str = Field(..., description="Field name to filter on")
    operator: FilterOperator = Field(..., description="Filter operator")
    value: Union[str, int, float, bool, List[Any], None] = Field(
        ..., description="Filter value"
    )
    
    class Config:
        use_enum_values = True


class Cursor(BaseModel):
    """Cursor for pagination."""
    offset: int = Field(0, ge=0, description="Offset from start")
    limit: int = Field(50, ge=1, le=1000, description="Maximum items to return")
    sort_field: str = Field("created_at", description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order")
    filters: List[FilterCondition] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class Pagination(BaseModel):
    """Pagination information for responses."""
    total_count: int = Field(..., ge=0, description="Total number of items")
    page_count: int = Field(..., ge=0, description="Total number of pages")
    current_page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")