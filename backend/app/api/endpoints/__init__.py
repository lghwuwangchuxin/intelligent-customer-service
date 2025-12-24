"""
API Endpoints Module - Domain-specific route handlers.

Each endpoint module corresponds to a bounded context:
- chat: Chat and RAG conversation endpoints
- knowledge: Knowledge base management endpoints
- agent: Agent (ReAct, LangGraph) endpoints
- mcp: MCP tool endpoints
- system: System info and health endpoints
- config: Model configuration endpoints
"""

from .chat import router as chat_router
from .knowledge import router as knowledge_router
from .agent import router as agent_router
from .mcp import router as mcp_router
from .system import router as system_router
from .config import router as config_router

__all__ = [
    "chat_router",
    "knowledge_router",
    "agent_router",
    "mcp_router",
    "system_router",
    "config_router",
]