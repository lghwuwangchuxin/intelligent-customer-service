"""Routing agent for multi-agent coordination."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from services.common.logging import get_logger
from .agent_registry import AgentRegistry, AgentInfo

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Routing decision result."""
    query: str
    selected_agents: List[AgentInfo]
    reasoning: str
    confidence: float
    requires_coordination: bool = False


class RoutingAgent:
    """
    Routing agent for task distribution.

    Analyzes user queries and routes to appropriate specialized agents.
    Supports:
    - Intent classification
    - Capability matching
    - Multi-agent coordination
    - Fallback handling
    """

    def __init__(
        self,
        registry: AgentRegistry,
        llm_client=None,
        use_llm_routing: bool = False,
    ):
        """
        Initialize routing agent.

        Args:
            registry: Agent registry
            llm_client: LLM client for intelligent routing
            use_llm_routing: Use LLM for routing decisions
        """
        self.registry = registry
        self.llm_client = llm_client
        self.use_llm_routing = use_llm_routing and llm_client is not None

        # Intent to capability mapping
        self.intent_mapping = {
            # Customer service intents
            "充电": ["charging", "ev_service"],
            "电桩": ["charging", "ev_service"],
            "收费": ["billing", "payment"],
            "价格": ["billing", "pricing"],
            "故障": ["maintenance", "support"],
            "报修": ["maintenance", "support"],
            "预约": ["booking", "scheduling"],
            "订单": ["order", "tracking"],
            "账户": ["account", "user_service"],
            "会员": ["membership", "account"],

            # English intents
            "charging": ["charging", "ev_service"],
            "payment": ["billing", "payment"],
            "booking": ["booking", "scheduling"],
            "support": ["support", "maintenance"],
        }

    async def route(self, query: str) -> RoutingDecision:
        """
        Route a query to appropriate agents.

        Args:
            query: User query

        Returns:
            RoutingDecision with selected agents
        """
        if self.use_llm_routing:
            return await self._route_with_llm(query)
        else:
            return await self._route_with_rules(query)

    async def _route_with_rules(self, query: str) -> RoutingDecision:
        """Route using rule-based matching."""
        query_lower = query.lower()

        # Find matching capabilities from intent mapping
        matched_capabilities = set()
        for keyword, capabilities in self.intent_mapping.items():
            if keyword in query_lower:
                matched_capabilities.update(capabilities)

        # Find agents with matching capabilities
        selected_agents = []
        if matched_capabilities:
            for capability in matched_capabilities:
                agents = self.registry.get_agents_by_capability(capability)
                for agent in agents:
                    if agent not in selected_agents:
                        selected_agents.append(agent)

        # If no specific match, try to find a general agent
        if not selected_agents:
            general_agent = self.registry.find_best_agent(["general", "qa"])
            if general_agent:
                selected_agents.append(general_agent)

        requires_coordination = len(selected_agents) > 1

        return RoutingDecision(
            query=query,
            selected_agents=selected_agents,
            reasoning=f"Matched capabilities: {list(matched_capabilities)}" if matched_capabilities else "No specific match, using general agent",
            confidence=0.8 if matched_capabilities else 0.5,
            requires_coordination=requires_coordination,
        )

    async def _route_with_llm(self, query: str) -> RoutingDecision:
        """Route using LLM for intent understanding."""
        # Get available agents
        available_agents = self.registry.list_agents()
        if not available_agents:
            return RoutingDecision(
                query=query,
                selected_agents=[],
                reasoning="No agents available",
                confidence=0.0,
            )

        # Build prompt for LLM
        agent_descriptions = "\n".join([
            f"- {agent.name}: {agent.description} (capabilities: {', '.join(agent.capabilities)})"
            for agent in available_agents
        ])

        prompt = f"""You are a routing agent that selects the best specialized agent(s) to handle a user query.

Available agents:
{agent_descriptions}

User query: {query}

Analyze the query and select the most appropriate agent(s).
If the query requires multiple agents, list them in order of importance.
If no agent is suitable, respond with "NONE".

Respond in JSON format:
{{
    "agents": ["agent_name1", "agent_name2"],
    "reasoning": "Brief explanation",
    "confidence": 0.0-1.0,
    "requires_coordination": true/false
}}"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=300)

            # Parse response
            import json
            result = json.loads(response)

            # Map agent names to AgentInfo
            selected_agents = []
            for agent_name in result.get("agents", []):
                for agent in available_agents:
                    if agent.name.lower() == agent_name.lower():
                        selected_agents.append(agent)
                        break

            return RoutingDecision(
                query=query,
                selected_agents=selected_agents,
                reasoning=result.get("reasoning", "LLM routing"),
                confidence=result.get("confidence", 0.7),
                requires_coordination=result.get("requires_coordination", False),
            )

        except Exception as e:
            logger.warning(f"LLM routing failed, falling back to rules: {e}")
            return await self._route_with_rules(query)

    async def get_fallback_agent(self) -> Optional[AgentInfo]:
        """
        Get a fallback agent when no specific agent is available.

        Returns:
            Fallback AgentInfo or None
        """
        # Try to find a general-purpose agent
        fallback_capabilities = ["general", "qa", "support", "assistant"]

        for capability in fallback_capabilities:
            agents = self.registry.get_agents_by_capability(capability)
            if agents:
                return agents[0]

        # Return any available online agent
        all_agents = self.registry.list_agents()
        online_agents = [a for a in all_agents if a.status.value == "online"]
        return online_agents[0] if online_agents else None


class CoordinationStrategy:
    """Strategy for coordinating multiple agents."""

    @staticmethod
    def parallel(agents: List[AgentInfo]) -> List[List[AgentInfo]]:
        """Execute all agents in parallel."""
        return [agents]

    @staticmethod
    def sequential(agents: List[AgentInfo]) -> List[List[AgentInfo]]:
        """Execute agents sequentially."""
        return [[agent] for agent in agents]

    @staticmethod
    def hierarchical(agents: List[AgentInfo], primary_index: int = 0) -> List[List[AgentInfo]]:
        """Execute primary agent first, then others in parallel."""
        if not agents:
            return []

        primary = agents[primary_index]
        others = [a for i, a in enumerate(agents) if i != primary_index]

        result = [[primary]]
        if others:
            result.append(others)

        return result
