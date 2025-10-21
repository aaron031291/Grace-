"""
Middleware components
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'LoggingMiddleware',
    'RateLimitMiddleware', 
    'MetricsMiddleware',
    'get_metrics_response'
]

def __getattr__(name):
    """Lazy import middleware components"""
    if name == 'LoggingMiddleware' or name == 'setup_logging':
        from .logging import LoggingMiddleware, setup_logging
        return LoggingMiddleware if name == 'LoggingMiddleware' else setup_logging
    
    elif name == 'RateLimitMiddleware':
        from .rate_limit import RateLimitMiddleware
        return RateLimitMiddleware
    
    elif name == 'MetricsMiddleware' or name == 'get_metrics_response':
        from .metrics import MetricsMiddleware, get_metrics_response
        return MetricsMiddleware if name == 'MetricsMiddleware' else get_metrics_response
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
