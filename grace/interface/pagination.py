"""
Pagination utilities for Grace API endpoints.
"""

from typing import List, Optional, TypeVar, Generic
from pydantic import BaseModel
from dataclasses import dataclass
import math

T = TypeVar("T")


@dataclass
class PaginationParams:
    """Parameters for pagination."""

    page: int = 1
    limit: int = 20

    def __post_init__(self):
        # Validate pagination parameters
        self.page = max(1, self.page)
        self.limit = max(1, min(100, self.limit))  # Max 100 items per page

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""

    items: List[T]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool

    @classmethod
    def create(cls, items: List[T], total: int, pagination: PaginationParams):
        """Create a paginated response."""
        total_pages = math.ceil(total / pagination.limit) if total > 0 else 1

        return cls(
            items=items,
            total=total,
            page=pagination.page,
            limit=pagination.limit,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_prev=pagination.page > 1,
        )


class CursorPaginationParams:
    """Parameters for cursor-based pagination."""

    def __init__(self, cursor: Optional[str] = None, limit: int = 20):
        self.cursor = cursor
        self.limit = max(1, min(100, limit))


def paginate_list(items: List[T], pagination: PaginationParams) -> PaginatedResponse[T]:
    """Paginate a list of items."""
    total = len(items)
    start = pagination.offset
    end = start + pagination.limit

    paginated_items = items[start:end]

    return PaginatedResponse.create(paginated_items, total, pagination)


def filter_and_paginate(
    items: List[T],
    filter_fn: Optional[callable] = None,
    sort_fn: Optional[callable] = None,
    pagination: PaginationParams = None,
) -> PaginatedResponse[T]:
    """Filter, sort and paginate items."""
    if pagination is None:
        pagination = PaginationParams()

    # Apply filtering
    if filter_fn:
        filtered_items = [item for item in items if filter_fn(item)]
    else:
        filtered_items = items.copy()

    # Apply sorting
    if sort_fn:
        filtered_items.sort(key=sort_fn)

    # Paginate
    return paginate_list(filtered_items, pagination)
