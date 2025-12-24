"""
Infrastructure Factory Module - Service creation and lifecycle management.

This module provides:
- ServiceFactory: Creates service instances with proper configuration
- ServiceRegistry: Manages service instances and dependencies
- Dependency injection support
"""

from .service_factory import ServiceFactory
from .service_registry import (
    ServiceRegistry,
    get_registry,
    async_init_services,
    get_services,
)

__all__ = [
    "ServiceFactory",
    "ServiceRegistry",
    "get_registry",
    "async_init_services",
    "get_services",
]
