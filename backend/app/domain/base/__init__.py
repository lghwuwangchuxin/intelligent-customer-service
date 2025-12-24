"""
Domain Base Module - Core abstractions and interfaces for DDD architecture.

This module provides:
- Base interfaces (ports) for domain services
- Common domain value objects and entities
- Service lifecycle protocols
"""

from .interfaces import (
    IEmbeddingService,
    ILLMService,
    IVectorStore,
    IRAGService,
    IChatService,
    IAgentService,
    IToolRegistry,
    IDocumentProcessor,
)
from .entities import (
    Message,
    Document,
    SearchResult,
    AgentThought,
    ToolCall,
)
from .lifecycle import (
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
