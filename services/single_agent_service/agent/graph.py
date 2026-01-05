"""LangGraph agent graph definition."""

from typing import Dict, Any, Optional, List

from services.common.logging import get_logger
from .state import AgentState, AgentPhase
from .nodes import RouterNode, RAGNode, ChatNode, ToolNode, ResponseNode, ErrorNode

logger = get_logger(__name__)


def should_retrieve(state: AgentState) -> str:
    """Determine if we should retrieve context."""
    if state.should_use_rag and not state.retrieved_contexts:
        return "rag"
    return "chat"


def should_use_tools(state: AgentState) -> str:
    """Determine if we should execute tools."""
    if state.pending_tool_calls:
        return "tool"
    return "response"


def should_continue_after_tools(state: AgentState) -> str:
    """Determine next step after tool execution."""
    # Check if we need to generate a response with tool results
    if state.tool_calls and not state.llm_response:
        return "chat"
    return "response"


def check_error(state: AgentState) -> str:
    """Check if in error state."""
    if state.phase == AgentPhase.ERROR or state.error:
        return "error"
    return "continue"


class AgentGraph:
    """
    LangGraph-style agent graph.

    Flow:
    1. Router -> Determine processing path
    2. RAG -> Retrieve context (if needed)
    3. Chat -> Generate LLM response
    4. Tool -> Execute tools (if requested)
    5. Response -> Format final response
    """

    def __init__(
        self,
        llm_client=None,
        rag_client=None,
        mcp_client=None,
        tools: Optional[List[Dict]] = None,
    ):
        """
        Initialize agent graph.

        Args:
            llm_client: LLM client
            rag_client: RAG service client
            mcp_client: MCP service client
            tools: Available tools
        """
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.mcp_client = mcp_client
        self.tools = tools or []

        # Initialize nodes
        self.router = RouterNode(llm_client)
        self.rag = RAGNode(rag_client)
        self.chat = ChatNode(llm_client, tools)
        self.tool = ToolNode(mcp_client)
        self.response = ResponseNode()
        self.error = ErrorNode()

    async def invoke(self, state: AgentState) -> AgentState:
        """
        Invoke the agent graph.

        Args:
            state: Initial agent state

        Returns:
            Final agent state
        """
        try:
            # Step 1: Route
            state = await self.router(state)
            if state.error:
                return await self.error(state)

            # Step 2: RAG (if needed)
            if state.should_use_rag:
                state = await self.rag(state)
                if state.error:
                    return await self.error(state)

            # Step 3: Chat loop with tools
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Generate response
                state = await self.chat(state)
                if state.error:
                    return await self.error(state)

                # Execute tools if requested
                if state.pending_tool_calls:
                    state = await self.tool(state)
                    if state.error:
                        return await self.error(state)
                    # Continue loop to process tool results
                    continue

                # No more tools, break
                break

            # Step 4: Generate response
            state = await self.response(state)

            return state

        except Exception as e:
            logger.error(f"Agent graph error: {e}")
            state.error = str(e)
            return await self.error(state)

    async def stream(self, state: AgentState):
        """
        Stream agent execution.

        Yields intermediate states during execution.

        Args:
            state: Initial agent state

        Yields:
            Intermediate states
        """
        try:
            # Step 1: Route
            state = await self.router(state)
            yield {"node": "router", "state": state}

            if state.error:
                state = await self.error(state)
                yield {"node": "error", "state": state}
                return

            # Step 2: RAG
            if state.should_use_rag:
                state = await self.rag(state)
                yield {"node": "rag", "state": state}

                if state.error:
                    state = await self.error(state)
                    yield {"node": "error", "state": state}
                    return

            # Step 3: Chat loop
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                state = await self.chat(state)
                yield {"node": "chat", "state": state}

                if state.error:
                    state = await self.error(state)
                    yield {"node": "error", "state": state}
                    return

                if state.pending_tool_calls:
                    state = await self.tool(state)
                    yield {"node": "tool", "state": state}

                    if state.error:
                        state = await self.error(state)
                        yield {"node": "error", "state": state}
                        return
                    continue

                break

            # Step 4: Response
            state = await self.response(state)
            yield {"node": "response", "state": state}

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            state.error = str(e)
            state = await self.error(state)
            yield {"node": "error", "state": state}


def create_agent_graph(
    llm_client=None,
    rag_client=None,
    mcp_client=None,
    tools: Optional[List[Dict]] = None,
) -> AgentGraph:
    """
    Create an agent graph.

    Args:
        llm_client: LLM client
        rag_client: RAG service client
        mcp_client: MCP service client
        tools: Available tools

    Returns:
        AgentGraph instance
    """
    return AgentGraph(
        llm_client=llm_client,
        rag_client=rag_client,
        mcp_client=mcp_client,
        tools=tools,
    )


# For LangGraph-compatible implementation
try:
    from langgraph.graph import StateGraph, END

    def create_langgraph_agent(
        llm_client=None,
        rag_client=None,
        mcp_client=None,
        tools: Optional[List[Dict]] = None,
    ):
        """Create a LangGraph StateGraph."""
        # Initialize nodes
        router = RouterNode(llm_client)
        rag = RAGNode(rag_client)
        chat = ChatNode(llm_client, tools)
        tool = ToolNode(mcp_client)
        response = ResponseNode()
        error = ErrorNode()

        # Build graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", router)
        workflow.add_node("rag", rag)
        workflow.add_node("chat", chat)
        workflow.add_node("tool", tool)
        workflow.add_node("response", response)
        workflow.add_node("error", error)

        # Add edges
        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            should_retrieve,
            {"rag": "rag", "chat": "chat"},
        )

        workflow.add_edge("rag", "chat")

        workflow.add_conditional_edges(
            "chat",
            should_use_tools,
            {"tool": "tool", "response": "response"},
        )

        workflow.add_conditional_edges(
            "tool",
            should_continue_after_tools,
            {"chat": "chat", "response": "response"},
        )

        workflow.add_edge("response", END)
        workflow.add_edge("error", END)

        return workflow.compile()

except ImportError:
    def create_langgraph_agent(*args, **kwargs):
        """Fallback when langgraph is not installed."""
        logger.warning("langgraph not installed, using custom implementation")
        return create_agent_graph(*args, **kwargs)
