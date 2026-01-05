"""Agent registry for managing specialized agents."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from services.common.logging import get_logger

logger = get_logger(__name__)


class AgentStatus(Enum):
    """Agent status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Information about a specialized agent."""
    id: str
    name: str
    description: str
    capabilities: List[str]
    url: str
    status: AgentStatus = AgentStatus.OFFLINE
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    average_latency_ms: float = 0.0


class AgentRegistry:
    """
    Registry for specialized A2A agents.

    Features:
    - Agent registration and discovery
    - Health monitoring
    - Capability-based matching
    - Load balancing
    """

    def __init__(
        self,
        health_check_interval: float = 30.0,
        nacos_client=None,
    ):
        """
        Initialize agent registry.

        Args:
            health_check_interval: Interval between health checks
            nacos_client: Nacos client for service discovery
        """
        self.health_check_interval = health_check_interval
        self.nacos_client = nacos_client

        self._agents: Dict[str, AgentInfo] = {}
        self._capability_index: Dict[str, List[str]] = {}  # capability -> agent_ids
        self._health_check_task = None
        self._running = False

    async def start(self):
        """Start the registry and health checking."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Agent registry started")

    async def stop(self):
        """Stop the registry."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    def register_agent(self, agent: AgentInfo) -> bool:
        """
        Register an agent.

        Args:
            agent: Agent information

        Returns:
            True if registered successfully
        """
        self._agents[agent.id] = agent

        # Update capability index
        for capability in agent.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            if agent.id not in self._capability_index[capability]:
                self._capability_index[capability].append(agent.id)

        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID

        Returns:
            True if unregistered
        """
        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]

        # Remove from capability index
        for capability in agent.capabilities:
            if capability in self._capability_index:
                self._capability_index[capability] = [
                    aid for aid in self._capability_index[capability]
                    if aid != agent_id
                ]

        del self._agents[agent_id]
        logger.info(f"Unregistered agent: {agent_id}")
        return True

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            AgentInfo or None
        """
        return self._agents.get(agent_id)

    def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """
        Get agents with a specific capability.

        Args:
            capability: Capability to match

        Returns:
            List of matching agents
        """
        agent_ids = self._capability_index.get(capability, [])
        return [
            self._agents[aid] for aid in agent_ids
            if aid in self._agents and self._agents[aid].status == AgentStatus.ONLINE
        ]

    def find_best_agent(
        self,
        capabilities: List[str],
        prefer_fastest: bool = True,
    ) -> Optional[AgentInfo]:
        """
        Find the best agent for given capabilities.

        Args:
            capabilities: Required capabilities
            prefer_fastest: Prefer agent with lowest latency

        Returns:
            Best matching agent or None
        """
        candidates = []

        for agent in self._agents.values():
            if agent.status != AgentStatus.ONLINE:
                continue

            # Check if agent has all required capabilities
            agent_caps = set(agent.capabilities)
            required_caps = set(capabilities)

            if required_caps.issubset(agent_caps):
                candidates.append(agent)

        if not candidates:
            # Try partial match
            for agent in self._agents.values():
                if agent.status != AgentStatus.ONLINE:
                    continue

                agent_caps = set(agent.capabilities)
                if agent_caps.intersection(set(capabilities)):
                    candidates.append(agent)

        if not candidates:
            return None

        # Sort by preference
        if prefer_fastest:
            candidates.sort(key=lambda a: a.average_latency_ms)
        else:
            # Sort by success rate
            candidates.sort(
                key=lambda a: a.successful_requests / max(a.total_requests, 1),
                reverse=True,
            )

        return candidates[0]

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
    ) -> List[AgentInfo]:
        """
        List all agents.

        Args:
            status: Filter by status

        Returns:
            List of agents
        """
        agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents

    def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
    ):
        """
        Update agent status.

        Args:
            agent_id: Agent ID
            status: New status
        """
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            self._agents[agent_id].last_health_check = datetime.utcnow()

    def update_agent_metrics(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
    ):
        """
        Update agent performance metrics.

        Args:
            agent_id: Agent ID
            success: Whether request was successful
            latency_ms: Request latency
        """
        if agent_id not in self._agents:
            return

        agent = self._agents[agent_id]
        agent.total_requests += 1
        if success:
            agent.successful_requests += 1

        # Update running average
        n = agent.total_requests
        agent.average_latency_ms = (
            (agent.average_latency_ms * (n - 1) + latency_ms) / n
        )

    async def discover_agents(self):
        """Discover agents from Nacos."""
        if not self.nacos_client:
            return

        try:
            # Find all A2A agent services
            services = await self.nacos_client.discover_service(
                "a2a-agent",
                passing_only=True,
            )

            for service in services:
                agent_id = service.service_id
                if agent_id not in self._agents:
                    # Get agent metadata
                    agent_info = AgentInfo(
                        id=agent_id,
                        name=service.meta.get("name", service.service_id),
                        description=service.meta.get("description", ""),
                        capabilities=service.meta.get("capabilities", "").split(","),
                        url=f"http://{service.address}:{service.port}",
                        status=AgentStatus.ONLINE,
                    )
                    self.register_agent(agent_info)

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_agents()
                await self.discover_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_agents(self):
        """Check health of all registered agents."""
        import httpx

        for agent_id, agent in self._agents.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{agent.url}/health")
                    if response.status_code == 200:
                        self.update_agent_status(agent_id, AgentStatus.ONLINE)
                    else:
                        self.update_agent_status(agent_id, AgentStatus.ERROR)
            except Exception:
                self.update_agent_status(agent_id, AgentStatus.OFFLINE)
