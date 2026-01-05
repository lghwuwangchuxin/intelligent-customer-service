"""
Agent State Management - State definitions and transitions for the ReAct agent.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import operator


class AgentPhase(str, Enum):
    """Current phase of the agent."""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    RESPONDING = "responding"
    COMPLETED = "completed"
    ERROR = "error"


class ToolCall(BaseModel):
    """Record of a tool call."""
    id: str
    name: str
    args: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ThoughtStep(BaseModel):
    """A single thought in the reasoning process."""
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskStep(BaseModel):
    """A step in a task plan."""
    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None


class LongTermMemoryContext(BaseModel):
    """Context from long-term memory for the current session."""
    user_id: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    relevant_entities: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_knowledge: List[Dict[str, Any]] = Field(default_factory=list)
    session_facts: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    """
    State for the ReAct agent.

    Uses TypedDict for LangGraph compatibility.
    """
    # Conversation context
    messages: Annotated[List[Dict[str, str]], operator.add]
    conversation_id: Optional[str]
    user_id: Optional[str]  # For user-specific memory

    # ReAct reasoning
    thoughts: List[ThoughtStep]
    current_thought: Optional[str]
    current_step: int
    max_iterations: int

    # Tool execution
    tool_calls: List[ToolCall]
    pending_tool_call: Optional[Dict[str, Any]]

    # Task planning
    plan: Optional[List[TaskStep]]
    current_task_id: Optional[int]

    # Long-term memory context
    long_term_memory: Optional[LongTermMemoryContext]
    memory_updates: List[Dict[str, Any]]  # Pending updates to long-term memory

    # Control flow
    phase: str
    should_continue: bool
    final_response: Optional[str]

    # Error handling
    error: Optional[str]
    retry_count: int


def create_initial_state(
    messages: List[Dict[str, str]],
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    max_iterations: int = 10,
    long_term_memory: Optional[LongTermMemoryContext] = None,
) -> AgentState:
    """
    Create initial agent state.

    Args:
        messages: Initial conversation messages.
        conversation_id: Optional conversation ID.
        user_id: Optional user ID for personalized memory.
        max_iterations: Maximum reasoning iterations.
        long_term_memory: Optional pre-loaded long-term memory context.

    Returns:
        Initial AgentState.
    """
    return AgentState(
        messages=messages,
        conversation_id=conversation_id,
        user_id=user_id,
        thoughts=[],
        current_thought=None,
        current_step=0,
        max_iterations=max_iterations,
        tool_calls=[],
        pending_tool_call=None,
        plan=None,
        current_task_id=None,
        long_term_memory=long_term_memory,
        memory_updates=[],
        phase=AgentPhase.THINKING.value,
        should_continue=True,
        final_response=None,
        error=None,
        retry_count=0,
    )


class AgentStateManager:
    """
    Helper class for managing agent state transitions.
    """

    @staticmethod
    def add_thought(state: AgentState, thought: str) -> AgentState:
        """Add a thought to the state."""
        step = ThoughtStep(
            step=state["current_step"] + 1,
            thought=thought,
        )
        state["thoughts"].append(step)
        state["current_thought"] = thought
        state["current_step"] += 1
        return state

    @staticmethod
    def record_action(
        state: AgentState,
        action: str,
        action_input: Dict[str, Any],
    ) -> AgentState:
        """Record an action in the current thought."""
        if state["thoughts"]:
            state["thoughts"][-1].action = action
            state["thoughts"][-1].action_input = action_input
        state["phase"] = AgentPhase.ACTING.value
        return state

    @staticmethod
    def record_observation(state: AgentState, observation: str) -> AgentState:
        """Record an observation from tool execution."""
        if state["thoughts"]:
            state["thoughts"][-1].observation = observation
        state["phase"] = AgentPhase.OBSERVING.value
        return state

    @staticmethod
    def add_tool_call(state: AgentState, tool_call: ToolCall) -> AgentState:
        """Add a tool call record."""
        state["tool_calls"].append(tool_call)
        return state

    @staticmethod
    def set_final_response(state: AgentState, response: str) -> AgentState:
        """Set the final response and mark as completed."""
        state["final_response"] = response
        state["phase"] = AgentPhase.COMPLETED.value
        state["should_continue"] = False
        return state

    @staticmethod
    def set_error(state: AgentState, error: str) -> AgentState:
        """Set an error state."""
        state["error"] = error
        state["phase"] = AgentPhase.ERROR.value
        state["should_continue"] = False
        return state

    @staticmethod
    def should_continue(state: AgentState) -> bool:
        """Check if the agent should continue reasoning."""
        if not state["should_continue"]:
            return False
        if state["current_step"] >= state["max_iterations"]:
            return False
        if state["phase"] in (AgentPhase.COMPLETED.value, AgentPhase.ERROR.value):
            return False
        return True

    # ============== Long-term Memory Methods ==============

    @staticmethod
    def set_long_term_memory(
        state: AgentState,
        memory_context: LongTermMemoryContext,
    ) -> AgentState:
        """Set the long-term memory context."""
        state["long_term_memory"] = memory_context
        return state

    @staticmethod
    def queue_memory_update(
        state: AgentState,
        update_type: str,
        data: Dict[str, Any],
    ) -> AgentState:
        """
        Queue a memory update to be persisted later.

        Args:
            state: Current agent state.
            update_type: Type of update (user_preference, entity, knowledge, fact).
            data: Update data.

        Returns:
            Updated state.
        """
        update = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        state["memory_updates"].append(update)
        return state

    @staticmethod
    def add_learned_preference(
        state: AgentState,
        key: str,
        value: Any,
        confidence: float = 0.8,
    ) -> AgentState:
        """Add a learned user preference to be stored."""
        if not state.get("user_id"):
            return state

        return AgentStateManager.queue_memory_update(
            state,
            update_type="user_preference",
            data={
                "user_id": state["user_id"],
                "key": key,
                "value": value,
                "confidence": confidence,
            },
        )

    @staticmethod
    def add_learned_entity(
        state: AgentState,
        entity_type: str,
        entity_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Add a learned entity to be stored."""
        return AgentStateManager.queue_memory_update(
            state,
            update_type="entity",
            data={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "name": name,
                "attributes": attributes or {},
            },
        )

    @staticmethod
    def add_learned_fact(
        state: AgentState,
        topic: str,
        content: str,
        source: Optional[str] = None,
    ) -> AgentState:
        """Add a learned fact/knowledge to be stored."""
        return AgentStateManager.queue_memory_update(
            state,
            update_type="knowledge",
            data={
                "topic": topic,
                "content": content,
                "source": source,
            },
        )

    @staticmethod
    def get_pending_memory_updates(state: AgentState) -> List[Dict[str, Any]]:
        """Get all pending memory updates."""
        return state.get("memory_updates", [])

    @staticmethod
    def clear_memory_updates(state: AgentState) -> AgentState:
        """Clear pending memory updates after they've been persisted."""
        state["memory_updates"] = []
        return state
