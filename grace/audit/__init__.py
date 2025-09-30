"""
Audit System - Convenience access to audit-related components.

This module provides easy access to audit components that may be distributed
across different kernel locations for architectural reasons.
"""

# Import from the actual implementation
from ..layer_04_audit_logs.immutable_logs import ImmutableLogs
from ..core.immutable_logs import ImmutableLogs as CoreImmutableLogs
from ..mtl_kernel.immutable_log_service import ImmutableLogService

__all__ = [
    'ImmutableLogs',
    'CoreImmutableLogs', 
    'ImmutableLogService'
]