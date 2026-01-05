"""
Base Agent Executor for Multi-Agent System.
Provides common functionality for all specialized agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgentExecutor(AgentExecutor, ABC):
    """
    Base class for specialized agent executors.

    Provides:
    - LLM client integration
    - Common tool execution patterns
    - Structured response generation
    """

    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize base executor.

        Args:
            agent_name: Name of the agent
            agent_description: Description of agent capabilities
            llm_client: Optional LLM client (creates default if not provided)
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.llm_client = llm_client or LLMClient()

    @abstractmethod
    async def process_request(
        self,
        user_message: str,
        context: RequestContext,
    ) -> str:
        """
        Process user request and return response.

        Must be implemented by specialized agents.

        Args:
            user_message: User's message text
            context: Request context with task/message info

        Returns:
            Response text
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Must be implemented by specialized agents.

        Returns:
            System prompt string
        """
        pass

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute agent logic (A2A interface).

        Args:
            context: Request context from A2A
            event_queue: Event queue for sending responses
        """
        try:
            # Extract user message from context
            user_message = self._extract_user_message(context)
            logger.info(f"[{self.agent_name}] Processing: {user_message[:100]}...")

            # Process the request
            response = await self.process_request(user_message, context)

            # Send response via event queue
            await event_queue.enqueue_event(new_agent_text_message(response))
            logger.info(f"[{self.agent_name}] Response sent successfully")

        except Exception as e:
            logger.error(f"[{self.agent_name}] Error: {e}")
            error_response = f"抱歉，{self.agent_name}处理您的请求时遇到了问题: {str(e)}"
            await event_queue.enqueue_event(new_agent_text_message(error_response))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel task execution."""
        logger.info(f"[{self.agent_name}] Cancel requested")
        await event_queue.enqueue_event(
            new_agent_text_message(f"{self.agent_name}已取消当前任务")
        )

    def _extract_user_message(self, context: RequestContext) -> str:
        """Extract user message text from context."""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    return part.text
                elif hasattr(part, "root") and hasattr(part.root, "text"):
                    return part.root.text
        return ""

    async def call_llm(
        self,
        user_message: str,
        context_info: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Call LLM with agent-specific system prompt.

        Args:
            user_message: User's message
            context_info: Optional additional context
            temperature: Sampling temperature

        Returns:
            LLM response
        """
        system_prompt = self.get_system_prompt()
        if context_info:
            system_prompt += f"\n\n相关上下文信息:\n{context_info}"

        return await self.llm_client.generate(
            prompt=user_message,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        error: Optional[str] = None,
    ) -> str:
        """
        Format tool execution result for LLM context.

        Args:
            tool_name: Name of the tool
            result: Tool execution result
            error: Optional error message

        Returns:
            Formatted result string
        """
        if error:
            return f"[工具 {tool_name} 执行失败]: {error}"

        if isinstance(result, dict):
            # Format dict as readable text
            lines = [f"[工具 {tool_name} 执行结果]:"]
            for key, value in result.items():
                if isinstance(value, list):
                    lines.append(f"  {key}:")
                    for item in value[:5]:  # Limit items
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {key}: {value}")
            return "\n".join(lines)

        elif isinstance(result, list):
            lines = [f"[工具 {tool_name} 执行结果]:"]
            for i, item in enumerate(result[:5], 1):
                lines.append(f"  {i}. {item}")
            return "\n".join(lines)

        else:
            return f"[工具 {tool_name} 执行结果]: {result}"
