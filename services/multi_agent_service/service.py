"""Multi-Agent Service - Main service implementation."""

import time
import uuid
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass

from services.common.logging import get_logger
from services.common.config import get_service_config

from .routing import RoutingAgent, AgentRegistry
from .routing.agent_registry import AgentInfo, AgentStatus
from .a2a import A2AClient
from .a2a.client import A2AResponse, TaskState

logger = get_logger(__name__)


@dataclass
class MultiAgentResponse:
    """Response from multi-agent coordination."""
    message: str
    agent_responses: List[Dict[str, Any]]
    routing_info: Dict[str, Any]
    total_latency_ms: int = 0


class MultiAgentService:
    """
    Multi-Agent Service for A2A coordination.

    Features:
    - Intelligent routing to specialized agents
    - A2A protocol communication
    - Response aggregation
    - Fallback handling
    """

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        routing_agent: Optional[RoutingAgent] = None,
        a2a_client: Optional[A2AClient] = None,
        llm_client=None,
        monitoring_client=None,
    ):
        """
        Initialize multi-agent service.

        Args:
            registry: Agent registry
            routing_agent: Routing agent
            a2a_client: A2A client
            llm_client: LLM client for aggregation
            monitoring_client: Monitoring client
        """
        self.registry = registry or AgentRegistry()
        self.routing_agent = routing_agent or RoutingAgent(self.registry)
        self.a2a_client = a2a_client or A2AClient()
        self.llm_client = llm_client
        self.monitoring_client = monitoring_client

        self._initialized = False

    @classmethod
    def from_config(cls, config=None) -> "MultiAgentService":
        """
        Create service from configuration.

        Args:
            config: Service configuration

        Returns:
            MultiAgentService instance
        """
        if config is None:
            config = get_service_config("multi-agent-service")

        return cls()

    async def initialize(self):
        """Initialize service and start registry."""
        if self._initialized:
            return

        await self.registry.start()

        # Register some default agents (for demo)
        await self._register_default_agents()

        self._initialized = True
        logger.info("Multi-Agent Service initialized")

    async def shutdown(self):
        """Shutdown service."""
        await self.registry.stop()
        self._initialized = False
        logger.info("Multi-Agent Service shutdown")

    async def _register_default_agents(self):
        """Register default specialized agents."""
        # These would normally be discovered via Nacos
        default_agents = [
            AgentInfo(
                id="charging-agent",
                name="Charging Agent",
                description="Handles charging station queries",
                capabilities=["charging", "ev_service", "station_info"],
                url="http://charging-agent:9001",
                status=AgentStatus.OFFLINE,
            ),
            AgentInfo(
                id="billing-agent",
                name="Billing Agent",
                description="Handles billing and payment queries",
                capabilities=["billing", "payment", "pricing"],
                url="http://billing-agent:9002",
                status=AgentStatus.OFFLINE,
            ),
            AgentInfo(
                id="support-agent",
                name="Support Agent",
                description="Handles support and maintenance queries",
                capabilities=["support", "maintenance", "troubleshooting"],
                url="http://support-agent:9003",
                status=AgentStatus.OFFLINE,
            ),
            AgentInfo(
                id="general-agent",
                name="General Agent",
                description="General purpose assistant",
                capabilities=["general", "qa", "assistant"],
                url="http://general-agent:9004",
                status=AgentStatus.OFFLINE,
            ),
        ]

        for agent in default_agents:
            self.registry.register_agent(agent)

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[List[Dict]] = None,
    ) -> MultiAgentResponse:
        """
        Process a query using multi-agent coordination.

        Args:
            query: User query
            session_id: Session ID
            context: Conversation context

        Returns:
            MultiAgentResponse with aggregated result
        """
        await self.initialize()

        start_time = time.time()

        # Step 1: Route to appropriate agents
        routing_decision = await self.routing_agent.route(query)

        logger.info(
            f"Routing decision: {len(routing_decision.selected_agents)} agents, "
            f"confidence={routing_decision.confidence:.2f}"
        )

        # Step 2: Handle no agents available
        if not routing_decision.selected_agents:
            fallback = await self.routing_agent.get_fallback_agent()
            if fallback:
                routing_decision.selected_agents = [fallback]
                routing_decision.reasoning = "Using fallback agent"
            else:
                return MultiAgentResponse(
                    message="I'm sorry, no agents are currently available to handle your request.",
                    agent_responses=[],
                    routing_info={
                        "selected_agents": [],
                        "reasoning": "No agents available",
                        "confidence": 0.0,
                    },
                    total_latency_ms=int((time.time() - start_time) * 1000),
                )

        # Step 3: Send to agents
        agent_responses = await self.a2a_client.send_to_multiple(
            agents=routing_decision.selected_agents,
            message=query,
            parallel=routing_decision.requires_coordination,
        )

        # Step 4: Update metrics
        for agent, response in zip(routing_decision.selected_agents, agent_responses):
            self.registry.update_agent_metrics(
                agent_id=agent.id,
                success=response.state == TaskState.COMPLETED,
                latency_ms=response.latency_ms,
            )

        # Step 5: Aggregate responses
        final_message = await self._aggregate_responses(
            query=query,
            responses=agent_responses,
            requires_coordination=routing_decision.requires_coordination,
        )

        total_latency_ms = int((time.time() - start_time) * 1000)

        return MultiAgentResponse(
            message=final_message,
            agent_responses=[
                {
                    "agent_id": r.agent_id,
                    "agent_name": r.agent_name,
                    "result": r.result,
                    "state": r.state.value,
                    "latency_ms": r.latency_ms,
                }
                for r in agent_responses
            ],
            routing_info={
                "selected_agents": [a.name for a in routing_decision.selected_agents],
                "reasoning": routing_decision.reasoning,
                "confidence": routing_decision.confidence,
                "requires_coordination": routing_decision.requires_coordination,
            },
            total_latency_ms=total_latency_ms,
        )

    async def stream_query(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[MultiAgentResponse]:
        """
        Stream query processing.

        Args:
            query: User query
            session_id: Session ID

        Yields:
            Partial responses
        """
        await self.initialize()

        start_time = time.time()

        # Route
        routing_decision = await self.routing_agent.route(query)

        if not routing_decision.selected_agents:
            fallback = await self.routing_agent.get_fallback_agent()
            if fallback:
                routing_decision.selected_agents = [fallback]

        if not routing_decision.selected_agents:
            yield MultiAgentResponse(
                message="No agents available",
                agent_responses=[],
                routing_info={"error": "No agents available"},
            )
            return

        # Stream from first agent
        primary_agent = routing_decision.selected_agents[0]

        async for response in self.a2a_client.send_task_streaming(
            agent=primary_agent,
            message=query,
            session_id=session_id,
        ):
            yield MultiAgentResponse(
                message=response.result,
                agent_responses=[{
                    "agent_id": response.agent_id,
                    "agent_name": response.agent_name,
                    "result": response.result,
                    "state": response.state.value,
                    "latency_ms": response.latency_ms,
                }],
                routing_info={
                    "selected_agents": [primary_agent.name],
                    "streaming": True,
                },
                total_latency_ms=int((time.time() - start_time) * 1000),
            )

    async def _aggregate_responses(
        self,
        query: str,
        responses: List[A2AResponse],
        requires_coordination: bool,
    ) -> str:
        """
        Aggregate responses from multiple agents.

        Args:
            query: Original query
            responses: Agent responses
            requires_coordination: Whether responses need coordination

        Returns:
            Aggregated response text
        """
        # Filter successful responses
        successful = [r for r in responses if r.state == TaskState.COMPLETED]

        if not successful:
            # All failed
            errors = [r.result for r in responses if r.state == TaskState.FAILED]
            return f"I apologize, but I couldn't process your request. Errors: {'; '.join(errors)}"

        if len(successful) == 1:
            # Single response
            return successful[0].result

        if not requires_coordination or not self.llm_client:
            # Simple concatenation
            parts = []
            for r in successful:
                parts.append(f"[{r.agent_name}]: {r.result}")
            return "\n\n".join(parts)

        # Use LLM to aggregate
        responses_text = "\n\n".join([
            f"Response from {r.agent_name}:\n{r.result}"
            for r in successful
        ])

        prompt = f"""You are coordinating responses from multiple specialized agents.
Combine their responses into a coherent, helpful answer.

User query: {query}

Agent responses:
{responses_text}

Combined response:"""

        try:
            combined = await self.llm_client.generate(prompt, max_tokens=500)
            return combined.strip()
        except Exception as e:
            logger.warning(f"LLM aggregation failed: {e}")
            return "\n\n".join([r.result for r in successful])

    async def list_agents(self) -> List[Dict[str, Any]]:
        """
        List available agents.

        Returns:
            List of agent info
        """
        agents = self.registry.list_agents()
        return [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "capabilities": a.capabilities,
                "status": a.status.value,
                "metrics": {
                    "total_requests": a.total_requests,
                    "success_rate": a.successful_requests / max(a.total_requests, 1),
                    "avg_latency_ms": a.average_latency_ms,
                },
            }
            for a in agents
        ]

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status
        """
        agents = self.registry.list_agents()
        online = len([a for a in agents if a.status == AgentStatus.ONLINE])

        return {
            "status": "healthy",
            "total_agents": len(agents),
            "online_agents": online,
        }
