"""
Service Lifecycle Management.

Provides protocols and base classes for managing service lifecycle,
including initialization, shutdown, and health checks.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@runtime_checkable
class AsyncInitializable(Protocol):
    """Protocol for services that support async initialization."""

    async def async_init(self) -> None:
        """Initialize service asynchronously."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        ...


class ServiceLifecycle(ABC):
    """
    Base class for services with lifecycle management.

    Provides a consistent interface for:
    - Async/sync initialization
    - Health checks
    - Graceful shutdown
    - Status tracking
    """

    def __init__(self):
        self._status: ServiceStatus = ServiceStatus.NOT_INITIALIZED
        self._error: Optional[str] = None

    @property
    def status(self) -> ServiceStatus:
        """Get current service status."""
        return self._status

    @property
    def is_initialized(self) -> bool:
        """Check if service is ready."""
        return self._status == ServiceStatus.READY

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._status in (ServiceStatus.READY, ServiceStatus.DEGRADED)

    @property
    def error(self) -> Optional[str]:
        """Get last error if any."""
        return self._error

    async def async_init(self) -> None:
        """Initialize service asynchronously."""
        if self._status in (ServiceStatus.READY, ServiceStatus.INITIALIZING):
            return

        self._status = ServiceStatus.INITIALIZING
        self._error = None

        try:
            await self._do_init()
            self._status = ServiceStatus.READY
        except Exception as e:
            self._status = ServiceStatus.ERROR
            self._error = str(e)
            raise

    def init(self) -> None:
        """Initialize service synchronously."""
        if self._status in (ServiceStatus.READY, ServiceStatus.INITIALIZING):
            return

        self._status = ServiceStatus.INITIALIZING
        self._error = None

        try:
            self._do_init_sync()
            self._status = ServiceStatus.READY
        except Exception as e:
            self._status = ServiceStatus.ERROR
            self._error = str(e)
            raise

    async def shutdown(self) -> None:
        """Shutdown service gracefully."""
        if self._status == ServiceStatus.SHUTDOWN:
            return

        self._status = ServiceStatus.SHUTTING_DOWN
        try:
            await self._do_shutdown()
        finally:
            self._status = ServiceStatus.SHUTDOWN

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        result = {
            "status": self._status.value,
            "healthy": self.is_healthy,
        }
        if self._error:
            result["error"] = self._error

        # Add custom health info
        custom_health = self._get_health_info()
        if custom_health:
            result.update(custom_health)

        return result

    @abstractmethod
    async def _do_init(self) -> None:
        """Implement async initialization logic."""
        ...

    def _do_init_sync(self) -> None:
        """
        Implement sync initialization logic.

        Override if sync initialization differs from async.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support synchronous initialization"
        )

    async def _do_shutdown(self) -> None:
        """
        Implement shutdown logic.

        Override to clean up resources.
        Default implementation does nothing.
        """
        pass

    def _get_health_info(self) -> Optional[Dict[str, Any]]:
        """
        Get additional health check information.

        Override to provide custom health metrics.
        Default implementation returns None.
        """
        return None


class CompositeService(ServiceLifecycle):
    """
    Base class for services composed of multiple sub-services.

    Manages lifecycle of all sub-services together.
    """

    def __init__(self):
        super().__init__()
        self._services: Dict[str, ServiceLifecycle] = {}

    def register_service(self, name: str, service: ServiceLifecycle) -> None:
        """Register a sub-service."""
        self._services[name] = service

    def get_service(self, name: str) -> Optional[ServiceLifecycle]:
        """Get a sub-service by name."""
        return self._services.get(name)

    async def _do_init(self) -> None:
        """Initialize all sub-services."""
        for name, service in self._services.items():
            try:
                await service.async_init()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize {name}: {e}") from e

    async def _do_shutdown(self) -> None:
        """Shutdown all sub-services."""
        errors = []
        for name, service in self._services.items():
            try:
                await service.shutdown()
            except Exception as e:
                errors.append(f"{name}: {e}")

        if errors:
            raise RuntimeError(f"Shutdown errors: {'; '.join(errors)}")

    def _get_health_info(self) -> Optional[Dict[str, Any]]:
        """Get health info from all sub-services."""
        return {
            "services": {
                name: service.health_check()
                for name, service in self._services.items()
            }
        }
