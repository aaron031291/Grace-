"""
Grace - Business Operations Kernel

A sophisticated external action and plugin execution layer that takes approved 
decisions from the governance layer and executes them in the real world via 
controlled, sandboxed plugins.
"""

__version__ = "0.1.0"
__author__ = "Grace Development Team"

from .business_ops_kernel import BusinessOpsKernel
from .plugin_registry import PluginRegistry
from .plugins.base_plugin import BasePlugin

__all__ = [
    "BusinessOpsKernel",
    "PluginRegistry", 
    "BasePlugin",
]