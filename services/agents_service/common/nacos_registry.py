"""
Nacos Service Registry for A2A Agents.

Provides service registration, discovery, and heartbeat functionality
for multi-agent system integration with Nacos.
"""

import asyncio
import atexit
import logging
import socket
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .config import get_nacos_config, NacosConfig

logger = logging.getLogger(__name__)

# Lazy import nacos to avoid import errors when not installed
_nacos = None


def _get_nacos():
    """Lazy load nacos module."""
    global _nacos
    if _nacos is None:
        try:
            import nacos
            _nacos = nacos
        except ImportError:
            raise ImportError(
                "nacos-sdk-python is required. Install it with: pip install nacos-sdk-python"
            )
    return _nacos


@dataclass
class ServiceInstance:
    """Represents a service instance."""
    service_id: str
    service_name: str
    address: str
    port: int
    healthy: bool = True
    weight: float = 1.0
    metadata: Dict[str, str] = None

    @property
    def base_url(self) -> str:
        """Get base URL for HTTP requests."""
        return f"http://{self.address}:{self.port}"


class NacosServiceRegistry:
    """
    Nacos Service Registry for A2A Agents.

    Features:
    - Service registration with automatic heartbeat
    - Service discovery with caching
    - Graceful shutdown and deregistration
    """

    _instance: Optional["NacosServiceRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[NacosConfig] = None):
        """Singleton pattern for registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[NacosConfig] = None):
        """
        Initialize Nacos registry.

        Args:
            config: Nacos configuration (uses default if not provided)
        """
        if self._initialized:
            return

        self.config = config or get_nacos_config()
        self._client = None
        self._registered_services: Dict[str, Dict[str, Any]] = {}
        self._service_cache: Dict[str, List[ServiceInstance]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 30  # seconds
        self._heartbeat_threads: Dict[str, threading.Thread] = {}
        self._running = True
        self._initialized = True

        # Register cleanup on exit
        atexit.register(self._cleanup_sync)

    def _get_client(self):
        """Get or create Nacos client."""
        if not self.config.enabled:
            return None

        if self._client is None:
            try:
                nacos = _get_nacos()
                self._client = nacos.NacosClient(
                    self.config.server_addresses,
                    namespace=self.config.namespace,
                    username=self.config.username or None,
                    password=self.config.password or None,
                )
                logger.info(f"Nacos client connected to {self.config.server_addresses}")
            except Exception as e:
                logger.error(f"Failed to create Nacos client: {e}")
                return None
        return self._client

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Try to get the actual IP by connecting to an external address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def register_service(
        self,
        service_name: str,
        port: int,
        address: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> Optional[str]:
        """
        Register a service with Nacos.

        Args:
            service_name: Name of the service (e.g., "travel-assistant-agent")
            port: Service port
            address: Service address (auto-detected if not provided)
            metadata: Additional metadata (e.g., {"version": "1.0.0"})
            tags: Service tags for filtering
            weight: Instance weight for load balancing

        Returns:
            Service ID if successful, None otherwise
        """
        if not self.config.enabled:
            logger.info(f"Nacos disabled, skipping registration for {service_name}")
            return None

        client = self._get_client()
        if not client:
            logger.warning(f"Nacos client not available, skipping registration for {service_name}")
            return None

        # Determine address
        if address is None:
            address = self._get_local_ip()

        # Generate service ID
        service_id = f"{service_name}-{address}:{port}"

        # Prepare metadata
        full_metadata = metadata or {}
        if tags:
            full_metadata["tags"] = ",".join(tags)
        full_metadata["service_id"] = service_id

        try:
            # Register with Nacos
            client.add_naming_instance(
                service_name,
                address,
                port,
                cluster_name=self.config.cluster_name,
                weight=weight,
                metadata=full_metadata,
                enable=True,
                healthy=True,
                group_name=self.config.group,
            )

            # Store registration info
            self._registered_services[service_id] = {
                "name": service_name,
                "address": address,
                "port": port,
                "cluster_name": self.config.cluster_name,
                "group": self.config.group,
                "weight": weight,
                "metadata": full_metadata,
            }

            # Start heartbeat thread
            self._start_heartbeat(service_id)

            logger.info(f"Registered service with Nacos: {service_name} ({address}:{port})")
            return service_id

        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
            return None

    def _start_heartbeat(self, service_id: str):
        """Start heartbeat thread for a service."""
        if service_id in self._heartbeat_threads:
            return

        def heartbeat_loop():
            client = self._get_client()
            if not client:
                return

            service_info = self._registered_services.get(service_id)
            if not service_info:
                return

            while self._running and service_id in self._registered_services:
                try:
                    client.send_heartbeat(
                        service_info["name"],
                        service_info["address"],
                        service_info["port"],
                        cluster_name=service_info["cluster_name"],
                        weight=service_info["weight"],
                        metadata=service_info["metadata"],
                        group_name=service_info["group"],
                    )
                    logger.debug(f"Heartbeat sent for {service_id}")
                except Exception as e:
                    logger.warning(f"Heartbeat failed for {service_id}: {e}")

                # Sleep with early exit check
                for _ in range(self.config.heartbeat_interval * 10):
                    if not self._running or service_id not in self._registered_services:
                        break
                    import time
                    time.sleep(0.1)

        thread = threading.Thread(target=heartbeat_loop, daemon=True, name=f"heartbeat-{service_id}")
        thread.start()
        self._heartbeat_threads[service_id] = thread
        logger.debug(f"Started heartbeat thread for {service_id}")

    def deregister_service(self, service_id: str) -> bool:
        """
        Deregister a service from Nacos.

        Args:
            service_id: Service ID to deregister

        Returns:
            True if successful
        """
        if not self.config.enabled:
            return True

        if service_id not in self._registered_services:
            logger.warning(f"Service not found for deregistration: {service_id}")
            return False

        service_info = self._registered_services.pop(service_id, None)
        if not service_info:
            return False

        client = self._get_client()
        if not client:
            return False

        try:
            client.remove_naming_instance(
                service_info["name"],
                service_info["address"],
                service_info["port"],
                cluster_name=service_info["cluster_name"],
                group_name=service_info["group"],
            )
            logger.info(f"Deregistered service from Nacos: {service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

    def discover_services(
        self,
        service_name: str,
        healthy_only: bool = True,
        use_cache: bool = True,
    ) -> List[ServiceInstance]:
        """
        Discover service instances.

        Args:
            service_name: Name of the service to discover
            healthy_only: Only return healthy instances
            use_cache: Use cached results if available

        Returns:
            List of service instances
        """
        if not self.config.enabled:
            return []

        import time

        cache_key = f"{service_name}:{self.config.group}"

        # Check cache
        if use_cache and cache_key in self._service_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self._cache_ttl:
                cached = self._service_cache[cache_key]
                if healthy_only:
                    return [i for i in cached if i.healthy]
                return cached

        client = self._get_client()
        if not client:
            # Return cached if available
            return self._service_cache.get(cache_key, [])

        try:
            result = client.list_naming_instance(
                service_name,
                healthy_only=healthy_only,
                group_name=self.config.group,
            )

            instances = []
            hosts = result.get("hosts", [])

            for host in hosts:
                metadata = host.get("metadata", {})
                instance = ServiceInstance(
                    service_id=metadata.get("service_id", f"{service_name}-{host.get('ip')}:{host.get('port')}"),
                    service_name=service_name,
                    address=host.get("ip"),
                    port=host.get("port"),
                    healthy=host.get("healthy", False),
                    weight=host.get("weight", 1.0),
                    metadata=metadata,
                )
                instances.append(instance)

            # Update cache
            self._service_cache[cache_key] = instances
            self._cache_timestamps[cache_key] = time.time()

            return instances

        except Exception as e:
            logger.error(f"Failed to discover service {service_name}: {e}")
            return self._service_cache.get(cache_key, [])

    def get_service_instance(
        self,
        service_name: str,
        strategy: str = "round_robin",
    ) -> Optional[ServiceInstance]:
        """
        Get a healthy service instance using load balancing.

        Args:
            service_name: Service name
            strategy: Load balancing strategy ("round_robin", "random", "weight")

        Returns:
            A service instance or None
        """
        import random

        instances = self.discover_services(service_name, healthy_only=True)
        if not instances:
            return None

        if strategy == "random":
            return random.choice(instances)
        elif strategy == "weight":
            total_weight = sum(i.weight for i in instances)
            if total_weight <= 0:
                return random.choice(instances)
            r = random.uniform(0, total_weight)
            current = 0
            for instance in instances:
                current += instance.weight
                if current >= r:
                    return instance
            return instances[-1]
        else:  # round_robin
            # Simple round-robin using hash
            if not hasattr(self, "_rr_counters"):
                self._rr_counters = {}
            counter = self._rr_counters.get(service_name, 0)
            self._rr_counters[service_name] = counter + 1
            return instances[counter % len(instances)]

    def _cleanup_sync(self):
        """Synchronous cleanup for atexit."""
        logger.info("Cleaning up Nacos registry...")
        self._running = False

        # Deregister all services
        for service_id in list(self._registered_services.keys()):
            self.deregister_service(service_id)

        # Stop heartbeat threads
        for thread in self._heartbeat_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._heartbeat_threads.clear()

        logger.info("Nacos registry cleanup complete")

    async def cleanup(self):
        """Async cleanup method."""
        self._cleanup_sync()


# Convenience functions
def get_registry(config: Optional[NacosConfig] = None) -> NacosServiceRegistry:
    """Get the singleton registry instance."""
    return NacosServiceRegistry(config)


def register_agent(
    agent_name: str,
    port: int,
    address: Optional[str] = None,
    version: str = "1.0.0",
    tags: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Register an A2A agent with Nacos.

    Args:
        agent_name: Agent name (will be converted to service name format)
        port: Agent port
        address: Agent address (auto-detected if not provided)
        version: Agent version
        tags: Additional tags

    Returns:
        Service ID if successful
    """
    registry = get_registry()

    # Convert agent name to service name format (e.g., "travel_assistant" -> "travel-assistant-agent")
    service_name = agent_name.replace("_", "-") + "-agent"

    # Prepare tags
    all_tags = ["a2a", "agent"]
    if tags:
        all_tags.extend(tags)

    return registry.register_service(
        service_name=service_name,
        port=port,
        address=address,
        metadata={
            "version": version,
            "protocol": "a2a",
            "agent_name": agent_name,
        },
        tags=all_tags,
    )


def deregister_agent(service_id: str) -> bool:
    """Deregister an A2A agent."""
    registry = get_registry()
    return registry.deregister_service(service_id)


def discover_agents(agent_name: Optional[str] = None) -> List[ServiceInstance]:
    """
    Discover A2A agents.

    Args:
        agent_name: Specific agent name (discovers all if not provided)

    Returns:
        List of agent instances
    """
    registry = get_registry()

    if agent_name:
        service_name = agent_name.replace("_", "-") + "-agent"
        return registry.discover_services(service_name)

    # Discover all known agent types
    all_agents = []
    agent_types = [
        "travel-assistant-agent",
        "charging-manager-agent",
        "billing-advisor-agent",
        "emergency-support-agent",
        "data-analyst-agent",
        "maintenance-expert-agent",
        "energy-advisor-agent",
        "scheduling-advisor-agent",
    ]

    for agent_type in agent_types:
        instances = registry.discover_services(agent_type)
        all_agents.extend(instances)

    return all_agents