"""
Agent Framework - ReAct and LangGraph agents with tool calling and memory management.
"""

from app.agent.react_agent import ReActAgent
from app.agent.memory import MemoryManager, ConversationMemory
from app.agent.state import AgentState, AgentPhase, ToolCall, ThoughtStep

# LangGraph agent (optional, requires langgraph package)
try:
    from app.agent.langgraph_agent import LangGraphAgent, create_langgraph_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LangGraphAgent = None
    create_langgraph_agent = None
    LANGGRAPH_AVAILABLE = False

__all__ = [
    "ReActAgent",
    "LangGraphAgent",
    "create_langgraph_agent",
    "MemoryManager",
    "ConversationMemory",
    "AgentState",
    "AgentPhase",
    "ToolCall",
    "ThoughtStep",
    "LANGGRAPH_AVAILABLE",
]
