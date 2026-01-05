"""Nacos service discovery client using nacos-sdk-python."""

import asyncio
import random
import socket
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import json
import threading

from .logging import get_logger

logger = get_logger(__name__)

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
    tags: List[str]
    meta: Dict[str, str]
    healthy: bool = True
    weight: float = 1.0
    cluster_name: str = "DEFAULT"

    @property
    def http_address(self) -> str:
        """Get HTTP address."""
        return f"{self.address}:{self.port}"

    @property
    def base_url(self) -> str:
        """Get base URL for HTTP requests."""
        return f"http://{self.address}:{self.port}"


class NacosClient:
    """Nacos client for service registration and discovery."""

    def __init__(
        self,
        server_addresses: str = "localhost:8848",
        namespace: str = "public",
        username: Optional[str] = None,
        password: Optional[str] = None,
        group: str = "DEFAULT_GROUP",
    ):
        """
        Initialize Nacos client.

        Args:
            server_addresses: Nacos server addresses (e.g., "localhost:8848" or "host1:8848,host2:8848")
            namespace: Nacos namespace (default: "public")
            username: Nacos username for authentication
            password: Nacos password for authentication
            group: Default service group
        """
        self.server_addresses = server_addresses
        self.namespace = namespace
        self.username = username
        self.password = password
        self.group = group

        self._client = None
        self._registered_services: Dict[str, Dict[str, Any]] = {}
        self._service_cache: Dict[str, List[ServiceInstance]] = {}
        self._cache_ttl = 30  # seconds
        self._cache_timestamps: Dict[str, float] = {}
        self._round_robin_index: Dict[str, int] = {}
        self._heartbeat_threads: Dict[str, threading.Thread] = {}
        self._running = True

    def _get_client(self):
        """Get or create Nacos client."""
        if self._client is None:
            nacos = _get_nacos()
            self._client = nacos.NacosClient(
                self.server_addresses,
                namespace=self.namespace,
                username=self.username,
                password=self.password,
            )
        return self._client

    async def close(self):
        """Close the client and cleanup resources."""
        self._running = False
        # Stop all heartbeat threads
        for service_id, thread in self._heartbeat_threads.items():
            if thread.is_alive():
                logger.debug(f"Stopping heartbeat for service: {service_id}")
        self._heartbeat_threads.clear()
        self._client = None

    async def register_service(
        self,
        name: str,
        port: int,
        address: Optional[str] = None,
        service_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, str]] = None,
        weight: float = 1.0,
        cluster_name: str = "DEFAULT",
        group: Optional[str] = None,
        enable_heartbeat: bool = True,
        health_check_url: Optional[str] = None,
        health_check_interval: str = "30s",
        health_check_timeout: str = "10s",
    ) -> str:
        """
        Register a service with Nacos.

        Args:
            name: Service name
            port: Service port
            address: Service address (defaults to hostname)
            service_id: Unique service ID (auto-generated if not provided)
            tags: Service tags (stored in metadata)
            meta: Service metadata
            weight: Instance weight for load balancing
            cluster_name: Cluster name
            group: Service group (uses default if not provided)
            enable_heartbeat: Enable heartbeat to keep service alive
            health_check_url: Health check URL (stored in metadata)
            health_check_interval: Health check interval (stored in metadata)
            health_check_timeout: Health check timeout (stored in metadata)

        Returns:
            Service ID
        """
        if address is None:
            address = socket.gethostname()
            # Try to get actual IP address
            try:
                address = socket.gethostbyname(address)
            except socket.gaierror:
                address = "127.0.0.1"

        if service_id is None:
            service_id = f"{name}-{uuid.uuid4().hex[:8]}"

        group = group or self.group

        # Prepare metadata
        metadata = meta or {}
        if tags:
            metadata["tags"] = ",".join(tags)
        if health_check_url:
            metadata["health_check_url"] = health_check_url
            metadata["health_check_interval"] = health_check_interval
            metadata["health_check_timeout"] = health_check_timeout

        try:
            client = self._get_client()

            # Run in executor since nacos-sdk-python is synchronous
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.add_naming_instance(
                    name,
                    address,
                    port,
                    cluster_name=cluster_name,
                    weight=weight,
                    metadata=metadata,
                    enable=True,
                    healthy=True,
                    group_name=group,
                )
            )

            # Store registration info for deregistration
            self._registered_services[service_id] = {
                "name": name,
                "address": address,
                "port": port,
                "cluster_name": cluster_name,
                "group": group,
            }

            # Start heartbeat thread if enabled
            if enable_heartbeat:
                self._start_heartbeat(service_id, name, address, port, cluster_name, group, weight, metadata)

            logger.info(f"Registered service with Nacos: {name} (ID: {service_id}, Address: {address}:{port})")
            return service_id
        except Exception as e:
            logger.error(f"Failed to register service {name} with Nacos: {e}")
            raise

    def _start_heartbeat(
        self,
        service_id: str,
        name: str,
        address: str,
        port: int,
        cluster_name: str,
        group: str,
        weight: float,
        metadata: Dict[str, str],
    ):
        """Start heartbeat thread for a service."""
        def heartbeat_loop():
            client = self._get_client()
            while self._running and service_id in self._registered_services:
                try:
                    client.send_heartbeat(
                        name,
                        address,
                        port,
                        cluster_name=cluster_name,
                        weight=weight,
                        metadata=metadata,
                        group_name=group,
                    )
                except Exception as e:
                    logger.warning(f"Heartbeat failed for {service_id}: {e}")
                # Sleep for 5 seconds between heartbeats
                for _ in range(50):  # 50 * 0.1 = 5 seconds
                    if not self._running or service_id not in self._registered_services:
                        break
                    import time
                    time.sleep(0.1)

        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        self._heartbeat_threads[service_id] = thread

    async def deregister_service(self, service_id: str) -> bool:
        """
        Deregister a service from Nacos.

        Args:
            service_id: Service ID to deregister

        Returns:
            True if successful
        """
        if service_id not in self._registered_services:
            logger.warning(f"Service not found for deregistration: {service_id}")
            return False

        service_info = self._registered_services.pop(service_id)

        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.remove_naming_instance(
                    service_info["name"],
                    service_info["address"],
                    service_info["port"],
                    cluster_name=service_info["cluster_name"],
                    group_name=service_info["group"],
                )
            )
            logger.info(f"Deregistered service from Nacos: {service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id} from Nacos: {e}")
            return False

    async def discover_service(
        self,
        name: str,
        tags: Optional[List[str]] = None,
        passing_only: bool = True,
        use_cache: bool = True,
        group: Optional[str] = None,
        clusters: Optional[List[str]] = None,
    ) -> List[ServiceInstance]:
        """
        Discover service instances.

        Args:
            name: Service name to discover
            tags: Filter by tags (stored in metadata)
            passing_only: Only return healthy instances
            use_cache: Use cached results if available
            group: Service group
            clusters: Filter by clusters

        Returns:
            List of service instances
        """
        import time

        group = group or self.group
        cache_key = f"{name}:{group}:{','.join(tags or [])}"

        # Check cache
        if use_cache and cache_key in self._service_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self._cache_ttl:
                cached = self._service_cache[cache_key]
                if passing_only:
                    return [i for i in cached if i.healthy]
                return cached

        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()

            # Get instances from Nacos
            result = await loop.run_in_executor(
                None,
                lambda: client.list_naming_instance(
                    name,
                    healthy_only=passing_only,
                    group_name=group,
                    clusters=",".join(clusters) if clusters else None,
                )
            )

            instances = []
            hosts = result.get("hosts", [])

            for host in hosts:
                metadata = host.get("metadata", {})

                # Parse tags from metadata
                instance_tags = []
                if "tags" in metadata:
                    instance_tags = metadata["tags"].split(",")

                # Filter by tags
                if tags:
                    if not all(tag in instance_tags for tag in tags):
                        continue

                instance = ServiceInstance(
                    service_id=f"{name}-{host.get('ip')}:{host.get('port')}",
                    service_name=name,
                    address=host.get("ip"),
                    port=host.get("port"),
                    tags=instance_tags,
                    meta=metadata,
                    healthy=host.get("healthy", False),
                    weight=host.get("weight", 1.0),
                    cluster_name=host.get("clusterName", "DEFAULT"),
                )
                instances.append(instance)

            # Update cache
            self._service_cache[cache_key] = instances
            self._cache_timestamps[cache_key] = time.time()

            if passing_only:
                return [i for i in instances if i.healthy]
            return instances
        except Exception as e:
            logger.error(f"Failed to discover service {name} from Nacos: {e}")
            # Return cached results if available
            if cache_key in self._service_cache:
                cached = self._service_cache[cache_key]
                if passing_only:
                    return [i for i in cached if i.healthy]
                return cached
            return []

    async def get_healthy_instance(
        self,
        name: str,
        strategy: str = "round_robin",
        group: Optional[str] = None,
    ) -> Optional[ServiceInstance]:
        """
        Get a healthy service instance using load balancing.

        Args:
            name: Service name
            strategy: Load balancing strategy ("round_robin", "random", "first", "weight")
            group: Service group

        Returns:
            A service instance or None if no healthy instances
        """
        instances = await self.discover_service(name, passing_only=True, group=group)

        if not instances:
            return None

        healthy = [i for i in instances if i.healthy]
        if not healthy:
            return None

        if strategy == "random":
            return random.choice(healthy)
        elif strategy == "first":
            return healthy[0]
        elif strategy == "weight":
            # Weighted random selection
            total_weight = sum(i.weight for i in healthy)
            if total_weight <= 0:
                return random.choice(healthy)
            r = random.uniform(0, total_weight)
            current = 0
            for instance in healthy:
                current += instance.weight
                if current >= r:
                    return instance
            return healthy[-1]
        else:  # round_robin
            key = f"{name}:{group or self.group}"
            if key not in self._round_robin_index:
                self._round_robin_index[key] = 0

            index = self._round_robin_index[key] % len(healthy)
            self._round_robin_index[key] = index + 1
            return healthy[index]

    async def get_config(
        self,
        data_id: str,
        group: Optional[str] = None,
        timeout: int = 3000,
    ) -> Optional[str]:
        """
        Get configuration from Nacos Config Center.

        Args:
            data_id: Configuration data ID
            group: Configuration group
            timeout: Request timeout in milliseconds

        Returns:
            Configuration content or None
        """
        group = group or self.group
        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()
            config = await loop.run_in_executor(
                None,
                lambda: client.get_config(data_id, group, timeout)
            )
            return config
        except Exception as e:
            logger.error(f"Failed to get config {data_id} from Nacos: {e}")
            return None

    async def publish_config(
        self,
        data_id: str,
        content: str,
        group: Optional[str] = None,
        config_type: str = "text",
    ) -> bool:
        """
        Publish configuration to Nacos Config Center.

        Args:
            data_id: Configuration data ID
            content: Configuration content
            group: Configuration group
            config_type: Configuration type (text, json, yaml, etc.)

        Returns:
            True if successful
        """
        group = group or self.group
        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.publish_config(data_id, group, content, config_type=config_type)
            )
            return result
        except Exception as e:
            logger.error(f"Failed to publish config {data_id} to Nacos: {e}")
            return False

    async def remove_config(
        self,
        data_id: str,
        group: Optional[str] = None,
    ) -> bool:
        """
        Remove configuration from Nacos Config Center.

        Args:
            data_id: Configuration data ID
            group: Configuration group

        Returns:
            True if successful
        """
        group = group or self.group
        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.remove_config(data_id, group)
            )
            return result
        except Exception as e:
            logger.error(f"Failed to remove config {data_id} from Nacos: {e}")
            return False

    def add_config_watcher(
        self,
        data_id: str,
        callback: Callable[[str], None],
        group: Optional[str] = None,
    ):
        """
        Add a configuration watcher.

        Args:
            data_id: Configuration data ID to watch
            callback: Callback function when config changes
            group: Configuration group
        """
        group = group or self.group
        try:
            client = self._get_client()
            client.add_config_watcher(data_id, group, callback)
            logger.info(f"Added config watcher for {data_id}")
        except Exception as e:
            logger.error(f"Failed to add config watcher for {data_id}: {e}")

    def subscribe_service(
        self,
        name: str,
        callback: Callable[[List[Dict]], None],
        group: Optional[str] = None,
        clusters: Optional[List[str]] = None,
    ):
        """
        Subscribe to service instance changes.

        Args:
            name: Service name to subscribe
            callback: Callback function when instances change
            group: Service group
            clusters: Filter by clusters
        """
        group = group or self.group
        try:
            client = self._get_client()
            client.subscribe(
                callback,
                0,  # listener_interval (0 = use default)
                name,
                group_name=group,
                clusters=",".join(clusters) if clusters else None,
            )
            logger.info(f"Subscribed to service {name}")
        except Exception as e:
            logger.error(f"Failed to subscribe to service {name}: {e}")

    def invalidate_cache(self, service_name: Optional[str] = None):
        """Invalidate service cache."""
        if service_name:
            keys_to_remove = [k for k in self._service_cache if k.startswith(service_name)]
            for key in keys_to_remove:
                del self._service_cache[key]
                self._cache_timestamps.pop(key, None)
        else:
            self._service_cache.clear()
            self._cache_timestamps.clear()


# Convenience functions for backward compatibility
async def create_nacos_client(
    server_addresses: str = "localhost:8848",
    namespace: str = "public",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> NacosClient:
    """Create and return a NacosClient instance."""
    return NacosClient(
        server_addresses=server_addresses,
        namespace=namespace,
        username=username,
        password=password,
    )
