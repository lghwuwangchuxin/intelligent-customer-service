"""Agent state definitions for LangGraph."""

from typing import List, Dict, Any, Optional, Annotated
from dataclasses import dataclass, field
from enum import Enum
import operator


class AgentPhase(Enum):
    """Current phase of agent execution."""
    ROUTING = "routing"
    RETRIEVING = "retrieving"
    TOOL_CALLING = "tool_calling"
    GENERATING = "generating"
    RESPONDING = "responding"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Message:
    """Chat message."""
    role: str  # user, assistant, system, tool
    content: str
    name: Optional[str] = None  # Tool name for tool messages
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Tool call information."""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class RetrievedContext:
    """Retrieved context from RAG."""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    Agent state for LangGraph.

    This state is passed through the graph and updated by each node.
    Uses Annotated types for LangGraph reducers.
    """
    # Input
    user_message: str = ""
    conversation_id: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Configuration
    enable_tools: bool = True
    enable_rag: bool = True
    knowledge_base_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048

    # Conversation history
    messages: List[Message] = field(default_factory=list)

    # Current phase
    phase: AgentPhase = AgentPhase.ROUTING

    # Routing decision
    should_use_tools: bool = False
    should_use_rag: bool = False
    selected_tools: List[str] = field(default_factory=list)

    # RAG context
    retrieved_contexts: List[RetrievedContext] = field(default_factory=list)
    rag_query: Optional[str] = None

    # Tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)
    pending_tool_calls: List[ToolCall] = field(default_factory=list)

    # LLM generations
    llm_response: Optional[str] = None
    llm_tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Final response
    final_response: Optional[str] = None
    sources: List[str] = field(default_factory=list)

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Metrics
    total_tokens: int = 0
    latency_ms: int = 0

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content, **kwargs))

    def add_user_message(self, content: str):
        """Add user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        """Add assistant message."""
        self.add_message("assistant", content)

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        """Add tool result message."""
        self.add_message("tool", result, name=name, tool_call_id=tool_call_id)

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Get messages in LLM-compatible format."""
        result = []
        for msg in self.messages:
            item = {"role": msg.role, "content": msg.content}
            if msg.name:
                item["name"] = msg.name
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            result.append(item)
        return result

    def get_context_string(self) -> str:
        """Get retrieved contexts as a string."""
        if not self.retrieved_contexts:
            return ""

        parts = []
        for i, ctx in enumerate(self.retrieved_contexts, 1):
            parts.append(f"[{i}] {ctx.content}")
            self.sources.append(ctx.source)

        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "user_message": self.user_message,
            "conversation_id": self.conversation_id,
            "phase": self.phase.value,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in self.messages
            ],
            "tool_calls": [
                {"name": tc.name, "status": tc.status}
                for tc in self.tool_calls
            ],
            "retrieved_contexts_count": len(self.retrieved_contexts),
            "final_response": self.final_response,
            "error": self.error,
        }


def merge_states(left: AgentState, right: AgentState) -> AgentState:
    """Merge two agent states (for LangGraph reducers)."""
    # Right state takes precedence for most fields
    merged = AgentState(
        user_message=right.user_message or left.user_message,
        conversation_id=right.conversation_id or left.conversation_id,
        user_id=right.user_id or left.user_id,
        session_id=right.session_id or left.session_id,
        enable_tools=right.enable_tools,
        enable_rag=right.enable_rag,
        knowledge_base_id=right.knowledge_base_id or left.knowledge_base_id,
        temperature=right.temperature,
        max_tokens=right.max_tokens,
        phase=right.phase,
        should_use_tools=right.should_use_tools,
        should_use_rag=right.should_use_rag,
        llm_response=right.llm_response or left.llm_response,
        final_response=right.final_response or left.final_response,
        error=right.error or left.error,
        retry_count=right.retry_count,
    )

    # Merge lists
    merged.messages = left.messages + right.messages
    merged.retrieved_contexts = left.retrieved_contexts + right.retrieved_contexts
    merged.tool_calls = left.tool_calls + right.tool_calls
    merged.sources = list(set(left.sources + right.sources))

    return merged
