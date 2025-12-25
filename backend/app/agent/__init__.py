"""
Agent Framework - ReAct and LangGraph agents with tool calling and memory management.

Memory Architecture:
- Short-term: ConversationMemory, MemoryManager (per-session context)
- Long-term: Store, StoreManager (cross-session knowledge)
"""

from app.agent.react_agent import ReActAgent
from app.agent.memory import MemoryManager, ConversationMemory, AgentInteraction
from app.agent.state import (
    AgentState,
    AgentPhase,
    ToolCall,
    ThoughtStep,
    LongTermMemoryContext,
    AgentStateManager,
)
from app.agent.store import (
    BaseStore,
    InMemoryStore,
    PersistentStore,
    StoreManager,
    MemoryItem,
    MemoryType,
    UserPreference,
    EntityMemory,
    KnowledgeItem,
    create_memory_store,
)

# LangGraph agent (optional, requires langgraph package)
try:
    from app.agent.langgraph_agent import LangGraphAgent, create_langgraph_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LangGraphAgent = None
    create_langgraph_agent = None
    LANGGRAPH_AVAILABLE = False

__all__ = [
    # Agents
    "ReActAgent",
    "LangGraphAgent",
    "create_langgraph_agent",
    "LANGGRAPH_AVAILABLE",
    # Short-term Memory
    "MemoryManager",
    "ConversationMemory",
    "AgentInteraction",
    # Long-term Memory (Store)
    "BaseStore",
    "InMemoryStore",
    "PersistentStore",
    "StoreManager",
    "MemoryItem",
    "MemoryType",
    "UserPreference",
    "EntityMemory",
    "KnowledgeItem",
    "create_memory_store",
    # State Management
    "AgentState",
    "AgentPhase",
    "ToolCall",
    "ThoughtStep",
    "LongTermMemoryContext",
    "AgentStateManager",
]
