"""Single Agent Service - Main service implementation."""

import time
import uuid
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass

import httpx

from services.common.logging import get_logger
from services.common.config import get_service_config, ServiceConfig

from .agent import AgentState, create_agent_graph

logger = get_logger(__name__)


class OllamaLLMClient:
    """
    Simple Ollama LLM client for single-agent-service.

    Returns response in format expected by ChatNode:
    {"content": str, "tool_calls": list}
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Chat with Ollama LLM.

        Returns dict with 'content' and 'tool_calls' keys.
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=30.0)) as client:
            try:
                logger.debug(f"Calling Ollama API: {url}, model={self.model}")
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                message = data.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                logger.debug(f"Ollama response: {content[:100]}..." if content else "Empty response")

                return {
                    "content": content,
                    "tool_calls": tool_calls,
                }

            except httpx.TimeoutException as e:
                logger.error(f"Ollama API timeout after {self.timeout}s: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama API HTTP error: {e.response.status_code}")
                raise
            except Exception as e:
                logger.error(f"Ollama API error: {type(e).__name__}: {e}")
                raise


@dataclass
class ChatConfig:
    """Chat configuration."""
    enable_tools: bool = True
    enable_rag: bool = True
    knowledge_base_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class ChatResponse:
    """Chat response."""
    message: str
    conversation_id: str
    tool_calls: List[Dict[str, Any]]
    sources: List[str]
    is_final: bool = True
    latency_ms: int = 0


class SingleAgentService:
    """
    Single Agent Service using LangGraph.

    Features:
    - LangGraph-based agent
    - Tool integration via MCP
    - RAG integration
    - Conversation history management
    - Streaming support
    """

    def __init__(
        self,
        llm_client=None,
        rag_client=None,
        mcp_client=None,
        monitoring_client=None,
    ):
        """
        Initialize single agent service.

        Args:
            llm_client: LLM client
            rag_client: RAG service client
            mcp_client: MCP service client
            monitoring_client: Monitoring service client
        """
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.mcp_client = mcp_client
        self.monitoring_client = monitoring_client

        # Conversation history storage (in-memory, use Redis for production)
        self._conversations: Dict[str, List[Dict]] = {}

        # Available tools
        self._tools: List[Dict] = []

        self._initialized = False

    @classmethod
    def from_config(cls, config: ServiceConfig = None) -> "SingleAgentService":
        """
        Create service from configuration.

        Args:
            config: Service configuration

        Returns:
            SingleAgentService instance
        """
        if config is None:
            config = get_service_config("single-agent-service")

        # Initialize LLM client
        llm_client = OllamaLLMClient(
            base_url=config.llm_base_url,
            model=config.llm_model,
            timeout=120.0,
        )
        logger.info(f"LLM client initialized: {config.llm_provider} - {config.llm_model} @ {config.llm_base_url}")

        # TODO: Initialize RAG and MCP clients when needed
        # For now, only LLM client is initialized

        return cls(llm_client=llm_client)

    async def initialize(self):
        """Initialize service and load tools."""
        if self._initialized:
            return

        # Load tools from MCP service
        if self.mcp_client:
            try:
                tools = await self.mcp_client.list_tools()
                self._tools = tools
                logger.info(f"Loaded {len(self._tools)} tools from MCP")
            except Exception as e:
                logger.warning(f"Failed to load tools: {e}")

        self._initialized = True
        logger.info("Single Agent Service initialized")

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        config: Optional[ChatConfig] = None,
    ) -> ChatResponse:
        """
        Process a chat message.

        Args:
            message: User message
            conversation_id: Conversation ID (auto-generated if not provided)
            config: Chat configuration

        Returns:
            ChatResponse
        """
        await self.initialize()

        start_time = time.time()
        config = config or ChatConfig()
        conversation_id = conversation_id or str(uuid.uuid4())

        # Get conversation history
        history = self._conversations.get(conversation_id, [])

        # Build agent state
        state = AgentState(
            user_message=message,
            conversation_id=conversation_id,
            enable_tools=config.enable_tools,
            enable_rag=config.enable_rag,
            knowledge_base_id=config.knowledge_base_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Add history to state
        for msg in history:
            state.messages.append(state.Message(
                role=msg["role"],
                content=msg["content"],
            ))

        # Create and run agent graph
        graph = create_agent_graph(
            llm_client=self.llm_client,
            rag_client=self.rag_client,
            mcp_client=self.mcp_client,
            tools=self._tools,
        )

        result_state = await graph.invoke(state)

        # Update conversation history
        self._conversations[conversation_id] = [
            {"role": m.role, "content": m.content}
            for m in result_state.messages
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        # Log to monitoring
        if self.monitoring_client:
            await self._log_to_monitoring(result_state, latency_ms)

        return ChatResponse(
            message=result_state.final_response or "",
            conversation_id=conversation_id,
            tool_calls=[
                {"name": tc.name, "result": tc.result, "status": tc.status}
                for tc in result_state.tool_calls
            ],
            sources=result_state.sources,
            latency_ms=latency_ms,
        )

    async def stream_chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        config: Optional[ChatConfig] = None,
    ) -> AsyncIterator[ChatResponse]:
        """
        Process a chat message with streaming.

        Args:
            message: User message
            conversation_id: Conversation ID
            config: Chat configuration

        Yields:
            ChatResponse for each step
        """
        await self.initialize()

        start_time = time.time()
        config = config or ChatConfig()
        conversation_id = conversation_id or str(uuid.uuid4())

        # Get conversation history
        history = self._conversations.get(conversation_id, [])

        # Build agent state
        state = AgentState(
            user_message=message,
            conversation_id=conversation_id,
            enable_tools=config.enable_tools,
            enable_rag=config.enable_rag,
            knowledge_base_id=config.knowledge_base_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Add history to state
        for msg in history:
            from .agent.state import Message
            state.messages.append(Message(
                role=msg["role"],
                content=msg["content"],
            ))

        # Create and run agent graph with streaming
        graph = create_agent_graph(
            llm_client=self.llm_client,
            rag_client=self.rag_client,
            mcp_client=self.mcp_client,
            tools=self._tools,
        )

        async for step in graph.stream(state):
            node_name = step["node"]
            current_state = step["state"]

            latency_ms = int((time.time() - start_time) * 1000)

            # Yield intermediate response
            yield ChatResponse(
                message=current_state.llm_response or "",
                conversation_id=conversation_id,
                tool_calls=[
                    {"name": tc.name, "result": tc.result, "status": tc.status}
                    for tc in current_state.tool_calls
                ],
                sources=current_state.sources,
                is_final=current_state.phase.value == "completed",
                latency_ms=latency_ms,
            )

        # Update conversation history with final state
        final_state = step["state"]
        self._conversations[conversation_id] = [
            {"role": m.role, "content": m.content}
            for m in final_state.messages
        ]

    async def get_history(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        history = self._conversations.get(conversation_id, [])
        return history[-limit:] if limit else history

    async def clear_history(self, conversation_id: str) -> bool:
        """
        Clear conversation history.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if cleared
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    async def _log_to_monitoring(self, state: AgentState, latency_ms: int):
        """Log execution to monitoring service."""
        try:
            # Log trace
            await self.monitoring_client.record_trace([{
                "trace_id": state.conversation_id,
                "span_id": str(uuid.uuid4()),
                "name": "single_agent.chat",
                "attributes": {
                    "user_message": state.user_message[:100],
                    "tool_calls": len(state.tool_calls),
                    "contexts": len(state.retrieved_contexts),
                    "latency_ms": latency_ms,
                },
                "status": "OK" if not state.error else "ERROR",
            }])
        except Exception as e:
            logger.warning(f"Failed to log to monitoring: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status
        """
        return {
            "status": "healthy",
            "tools_loaded": len(self._tools),
            "active_conversations": len(self._conversations),
        }
