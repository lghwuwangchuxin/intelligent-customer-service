"""
Domain Module - Core business logic and abstractions.

This module contains:
- base: Base interfaces, entities, and lifecycle management
- chat: Chat domain (future expansion)
- knowledge: Knowledge base domain (future expansion)
- agent: Agent domain (future expansion)
"""

from .base import (
    # Interfaces
    IEmbeddingService,
    ILLMService,
    IVectorStore,
    IRAGService,
    IChatService,
    IAgentService,
    IToolRegistry,
    IDocumentProcessor,
    # Entities
    Message,
    Document,
    SearchResult,
    AgentThought,
    ToolCall,
    # Lifecycle
    ServiceLifecycle,
    AsyncInitializable,
)

__all__ = [
    # Interfaces
    "IEmbeddingService",
    "ILLMService",
    "IVectorStore",
    "IRAGService",
    "IChatService",
    "IAgentService",
    "IToolRegistry",
    "IDocumentProcessor",
    # Entities
    "Message",
    "Document",
    "SearchResult",
    "AgentThought",
    "ToolCall",
    # Lifecycle
    "ServiceLifecycle",
    "AsyncInitializable",
]