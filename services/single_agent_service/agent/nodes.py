"""LangGraph node implementations."""

import asyncio
import json
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from services.common.logging import get_logger
from .state import AgentState, AgentPhase, ToolCall, RetrievedContext, Message

logger = get_logger(__name__)


class BaseNode(ABC):
    """Base class for graph nodes."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute the node."""
        pass


class RouterNode(BaseNode):
    """Route to appropriate processing based on input."""

    def __init__(self, llm_client=None):
        super().__init__("router")
        self.llm_client = llm_client

    async def __call__(self, state: AgentState) -> AgentState:
        """Determine processing path."""
        state.phase = AgentPhase.ROUTING

        # Add user message to history
        state.add_user_message(state.user_message)

        # Simple routing logic - can be enhanced with LLM
        message_lower = state.user_message.lower()

        # Check for tool indicators
        tool_keywords = ['搜索', '查询', '执行', 'search', 'find', 'calculate', 'run']
        state.should_use_tools = state.enable_tools and any(
            kw in message_lower for kw in tool_keywords
        )

        # Check for RAG indicators
        rag_keywords = ['知识库', '文档', '资料', 'knowledge', 'document', 'what is', '是什么']
        state.should_use_rag = state.enable_rag and any(
            kw in message_lower for kw in rag_keywords
        )

        # Default to RAG for question-like queries
        if state.enable_rag and ('?' in state.user_message or '？' in state.user_message):
            state.should_use_rag = True

        logger.debug(f"Routing: tools={state.should_use_tools}, rag={state.should_use_rag}")

        return state


class RAGNode(BaseNode):
    """Retrieve context from RAG service."""

    def __init__(self, rag_client=None):
        super().__init__("rag")
        self.rag_client = rag_client

    async def __call__(self, state: AgentState) -> AgentState:
        """Retrieve relevant context."""
        if not state.should_use_rag:
            return state

        state.phase = AgentPhase.RETRIEVING
        state.rag_query = state.user_message

        if not self.rag_client:
            logger.warning("RAG client not configured")
            return state

        try:
            # Call RAG service
            result = await self.rag_client.retrieve(
                query=state.rag_query,
                knowledge_base_id=state.knowledge_base_id,
                top_k=5,
            )

            # Convert to RetrievedContext
            for doc in result.get("documents", []):
                state.retrieved_contexts.append(RetrievedContext(
                    id=doc["id"],
                    content=doc["content"],
                    score=doc["score"],
                    source=doc.get("metadata", {}).get("source", "unknown"),
                    metadata=doc.get("metadata", {}),
                ))

            logger.info(f"Retrieved {len(state.retrieved_contexts)} contexts")

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            state.error = f"Retrieval failed: {e}"

        return state


class ChatNode(BaseNode):
    """Generate response using LLM."""

    def __init__(self, llm_client=None, tools: Optional[List[Dict]] = None):
        super().__init__("chat")
        self.llm_client = llm_client
        self.tools = tools or []

    async def __call__(self, state: AgentState) -> AgentState:
        """Generate LLM response."""
        state.phase = AgentPhase.GENERATING

        if not self.llm_client:
            state.error = "LLM client not configured"
            state.phase = AgentPhase.ERROR
            return state

        try:
            # Build system message
            system_message = self._build_system_message(state)

            # Build messages
            messages = [{"role": "system", "content": system_message}]
            messages.extend(state.get_messages_for_llm())

            # Add context if available
            if state.retrieved_contexts:
                context_msg = f"\n\nRelevant context:\n{state.get_context_string()}"
                messages[-1]["content"] += context_msg

            # Call LLM
            response = await self.llm_client.chat(
                messages=messages,
                tools=self.tools if state.should_use_tools else None,
                temperature=state.temperature,
                max_tokens=state.max_tokens,
            )

            # Process response
            state.llm_response = response.get("content", "")
            state.llm_tool_calls = response.get("tool_calls", [])

            # Check for tool calls
            if state.llm_tool_calls:
                for tc in state.llm_tool_calls:
                    state.pending_tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    ))

            logger.debug(f"LLM response: {state.llm_response[:100]}...")

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            state.error = f"Generation failed: {e}"
            state.phase = AgentPhase.ERROR

        return state

    def _build_system_message(self, state: AgentState) -> str:
        """Build system message for LLM."""
        system = """You are a helpful customer service assistant for an intelligent customer service system.

Your capabilities:
- Answer questions using provided context
- Use tools when needed to search or retrieve information
- Provide accurate and helpful responses

Guidelines:
- Be concise and professional
- Cite sources when using retrieved context
- Ask for clarification if the question is unclear
- Admit when you don't know something
"""

        if state.retrieved_contexts:
            system += "\n\nUse the provided context to answer the user's question."

        if state.should_use_tools:
            system += "\n\nYou have access to tools. Use them when appropriate."

        return system


class ToolNode(BaseNode):
    """Execute tool calls."""

    def __init__(self, mcp_client=None):
        super().__init__("tool")
        self.mcp_client = mcp_client

    async def __call__(self, state: AgentState) -> AgentState:
        """Execute pending tool calls."""
        if not state.pending_tool_calls:
            return state

        state.phase = AgentPhase.TOOL_CALLING

        if not self.mcp_client:
            logger.warning("MCP client not configured")
            for tc in state.pending_tool_calls:
                tc.status = "failed"
                tc.error = "Tool service not available"
            state.tool_calls.extend(state.pending_tool_calls)
            state.pending_tool_calls.clear()
            return state

        # Execute tools concurrently
        tasks = [
            self._execute_tool(tc)
            for tc in state.pending_tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for tc, result in zip(state.pending_tool_calls, results):
            if isinstance(result, Exception):
                tc.status = "failed"
                tc.error = str(result)
            else:
                tc.status = "completed"
                tc.result = result

            # Add to tool messages
            state.add_tool_result(
                tool_call_id=tc.id,
                name=tc.name,
                result=tc.result or tc.error or "",
            )

            state.tool_calls.append(tc)

        state.pending_tool_calls.clear()

        logger.info(f"Executed {len(state.tool_calls)} tool calls")

        return state

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a single tool call."""
        try:
            result = await self.mcp_client.execute_tool(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
            )
            return json.dumps(result) if isinstance(result, dict) else str(result)
        except Exception as e:
            raise Exception(f"Tool execution failed: {e}")


class ResponseNode(BaseNode):
    """Generate final response."""

    def __init__(self):
        super().__init__("response")

    async def __call__(self, state: AgentState) -> AgentState:
        """Generate final response."""
        state.phase = AgentPhase.RESPONDING

        # Use LLM response as final response
        if state.llm_response:
            state.final_response = state.llm_response
        elif state.error:
            state.final_response = f"I apologize, but I encountered an error: {state.error}"
        else:
            state.final_response = "I apologize, but I couldn't generate a response."

        # Add assistant message
        state.add_assistant_message(state.final_response)

        # Collect sources
        state.sources = list(set(
            ctx.source for ctx in state.retrieved_contexts
            if ctx.source != "unknown"
        ))

        state.phase = AgentPhase.COMPLETED

        return state


class ErrorNode(BaseNode):
    """Handle errors."""

    def __init__(self):
        super().__init__("error")

    async def __call__(self, state: AgentState) -> AgentState:
        """Handle error state."""
        state.phase = AgentPhase.ERROR

        if state.retry_count < state.max_retries:
            state.retry_count += 1
            state.error = None
            state.phase = AgentPhase.ROUTING
            logger.info(f"Retrying (attempt {state.retry_count})")
        else:
            state.final_response = f"I apologize, but I encountered an error after {state.max_retries} attempts: {state.error}"
            state.add_assistant_message(state.final_response)
            state.phase = AgentPhase.COMPLETED

        return state
