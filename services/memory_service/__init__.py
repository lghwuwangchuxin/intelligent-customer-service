"""
Memory Service - Short-term and long-term memory management.

Memory Architecture:
- Short-term: ConversationMemory, MemoryManager (per-session context)
- Long-term: Store, StoreManager (cross-session knowledge)

Usage:
    # As a microservice
    python -m services.memory_service.run

    # As a library
    from services.memory_service import MemoryManager, StoreManager
"""

# Local imports for memory management
from .memory import MemoryManager, ConversationMemory, AgentInteraction
from .state import (
    AgentState,
    AgentPhase,
    ToolCall,
    ThoughtStep,
    LongTermMemoryContext,
    AgentStateManager,
)
from .store import (
    BaseStore,
    InMemoryStore,
    PersistentStore,
    StoreManager,
    MemoryItem,
    MemoryType,
    UserPreference,
    EntityMemory,
    KnowledgeItem,
)

# Service components
from .service import MemoryService, MemoryServiceConfig

# Agent implementations (optional)
try:
    from .react_agent import ReActAgent
    REACT_AVAILABLE = True
except ImportError:
    ReActAgent = None
    REACT_AVAILABLE = False

try:
    from .langgraph_agent import LangGraphAgent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LangGraphAgent = None
    LANGGRAPH_AVAILABLE = False

__all__ = [
    # Service
    "MemoryService",
    "MemoryServiceConfig",
    # Agents
    "ReActAgent",
    "LangGraphAgent",
    "REACT_AVAILABLE",
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
    # State Management
    "AgentState",
    "AgentPhase",
    "ToolCall",
    "ThoughtStep",
    "LongTermMemoryContext",
    "AgentStateManager",
]
