"""
Memory Service - Main service implementation.
记忆管理服务，提供短期和长期记忆管理功能。

Features:
- Short-term conversation memory with auto-summarization
- Long-term memory storage (user preferences, entities, knowledge)
- Memory persistence with JSON files
- Memory search and retrieval
"""

import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from services.common.logging import get_logger
from services.common.config import get_service_config, ServiceConfig

from .memory import MemoryManager, ConversationMemory, AgentInteraction
from .store import (
    StoreManager,
    PersistentStore,
    InMemoryStore,
    MemoryItem,
    MemoryType,
    UserPreference,
    EntityMemory,
    KnowledgeItem,
)

logger = get_logger(__name__)


@dataclass
class MemoryServiceConfig:
    """Memory service configuration."""
    # Short-term memory settings
    max_messages: int = 50
    summary_threshold: int = 20
    keep_recent: int = 10

    # Long-term memory settings
    persist_enabled: bool = True
    persist_path: str = "./data/memory_store"
    auto_save_interval: int = 60  # seconds

    # Memory search settings
    default_search_limit: int = 10


@dataclass
class ConversationInfo:
    """Conversation information."""
    conversation_id: str
    title: Optional[str]
    message_count: int
    has_summary: bool
    created_at: str
    updated_at: str


@dataclass
class MessageInfo:
    """Message information."""
    role: str
    content: str
    timestamp: Optional[str] = None


@dataclass
class MemoryStats:
    """Memory statistics."""
    total_conversations: int
    total_messages: int
    total_memories: int
    namespaces: List[str]
    memory_types: Dict[str, int]


class MemoryService:
    """
    Memory Service for managing short-term and long-term memory.

    Features:
    - Conversation memory management
    - Auto-summarization
    - Long-term memory storage
    - Memory persistence
    """

    def __init__(
        self,
        config: MemoryServiceConfig = None,
        llm_client=None,
    ):
        """
        Initialize memory service.

        Args:
            config: Service configuration
            llm_client: LLM client for summarization
        """
        self.config = config or MemoryServiceConfig()
        self.llm_client = llm_client

        # Initialize short-term memory manager
        self.memory_manager = MemoryManager(
            llm_manager=llm_client,
            max_messages=self.config.max_messages,
            summary_threshold=self.config.summary_threshold,
            keep_recent=self.config.keep_recent,
            persist_path=self.config.persist_path if self.config.persist_enabled else None,
        )

        # Initialize long-term memory store
        if self.config.persist_enabled:
            self.store = PersistentStore(
                storage_path=self.config.persist_path,
                auto_save=True,
                save_interval=10,
            )
        else:
            self.store = InMemoryStore()

        self.store_manager = StoreManager(self.store)
        self._initialized = False

    @classmethod
    def from_config(cls, config: ServiceConfig = None) -> "MemoryService":
        """
        Create service from service configuration.

        Args:
            config: Service configuration

        Returns:
            MemoryService instance
        """
        if config is None:
            config = get_service_config("memory-service")

        # Build memory config from service config
        memory_config = MemoryServiceConfig(
            max_messages=int(config.extra.get("memory_max_messages", 50)),
            summary_threshold=int(config.extra.get("memory_summary_threshold", 20)),
            keep_recent=int(config.extra.get("memory_keep_recent", 10)),
            persist_enabled=config.extra.get("memory_persist_enabled", "true").lower() == "true",
            persist_path=config.extra.get("memory_persist_path", "./data/memory_store"),
            auto_save_interval=int(config.extra.get("memory_auto_save_interval", 60)),
        )

        # Initialize LLM client for summarization (optional)
        llm_client = None
        if config.llm_base_url:
            from .llm_client import OllamaLLMClient
            llm_client = OllamaLLMClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
            )
            logger.info(f"LLM client initialized: {config.llm_model} @ {config.llm_base_url}")

        return cls(config=memory_config, llm_client=llm_client)

    async def initialize(self):
        """Initialize service."""
        if self._initialized:
            return

        # Load persisted memories
        if self.config.persist_enabled:
            try:
                self.memory_manager.load_persisted_memories()
                logger.info("Loaded persisted short-term memories")
            except Exception as e:
                logger.warning(f"Failed to load persisted memories: {e}")

        self._initialized = True
        logger.info("Memory Service initialized")

    async def shutdown(self):
        """Shutdown service and persist data."""
        if self.config.persist_enabled:
            try:
                # Persist short-term memory
                self.memory_manager._persist_memory()
                # Flush long-term store
                if hasattr(self.store, 'flush'):
                    await self.store.flush()
                logger.info("Memory persisted on shutdown")
            except Exception as e:
                logger.error(f"Failed to persist memory on shutdown: {e}")

    # ==================== Short-term Memory Operations ====================

    async def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> ConversationInfo:
        """
        Create a new conversation.

        Args:
            conversation_id: Optional conversation ID
            title: Optional title
            metadata: Optional metadata

        Returns:
            ConversationInfo
        """
        conversation_id = conversation_id or str(uuid.uuid4())

        # Get or create conversation memory
        memory = self.memory_manager.get_or_create(conversation_id)
        if title:
            memory.title = title
        if metadata:
            memory.metadata.update(metadata)

        return ConversationInfo(
            conversation_id=conversation_id,
            title=memory.title,
            message_count=len(memory.messages),
            has_summary=memory.summary is not None,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Add a message to conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant/system)
            content: Message content

        Returns:
            Result with message count and summary status
        """
        await self.memory_manager.add_message(conversation_id, role, content)
        memory = self.memory_manager.get_or_create(conversation_id)

        return {
            "conversation_id": conversation_id,
            "message_count": len(memory.messages),
            "has_summary": memory.summary is not None,
            "summary_triggered": len(memory.messages) >= self.config.summary_threshold,
        }

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation details.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation details or None
        """
        memory = self.memory_manager._memories.get(conversation_id)
        if not memory:
            return None

        return {
            "conversation_id": conversation_id,
            "title": memory.title or memory.generate_title(),
            "messages": memory.messages,
            "summary": memory.summary,
            "interactions": [i.dict() for i in memory.interactions],
            "message_count": len(memory.messages),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "metadata": memory.metadata,
        }

    async def get_context(
        self,
        conversation_id: str,
        max_messages: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM.

        Args:
            conversation_id: Conversation ID
            max_messages: Maximum messages to return

        Returns:
            List of messages for LLM context
        """
        # get_context is synchronous in MemoryManager
        return self.memory_manager.get_context(conversation_id)

    async def list_conversations(
        self,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List conversations.

        Args:
            page: Page number
            page_size: Page size
            user_id: Optional user ID filter

        Returns:
            Paginated conversation list
        """
        # Convert page/page_size to limit/offset
        offset = (page - 1) * page_size
        limit = page_size

        result = self.memory_manager.list_conversations(
            limit=limit,
            offset=offset,
        )

        total = len(self.memory_manager._memories)

        conversations = [
            ConversationInfo(
                conversation_id=c["conversation_id"],
                title=c["title"],
                message_count=c["message_count"],
                has_summary=c.get("has_summary", False),
                created_at=c["created_at"],
                updated_at=c["updated_at"],
            ).__dict__
            for c in result
        ]

        return {
            "conversations": conversations,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear conversation messages but keep metadata.

        Args:
            conversation_id: Conversation ID

        Returns:
            Success status
        """
        self.memory_manager.clear_memory(conversation_id)
        return True

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete conversation completely.

        Args:
            conversation_id: Conversation ID

        Returns:
            Success status
        """
        return self.memory_manager.delete_memory(conversation_id)

    async def summarize_conversation(
        self,
        conversation_id: str,
        force: bool = False,
    ) -> Optional[str]:
        """
        Summarize conversation.

        Args:
            conversation_id: Conversation ID
            force: Force summarization even if threshold not reached

        Returns:
            Summary text or None
        """
        memory = self.memory_manager._memories.get(conversation_id)
        if not memory:
            return None

        if force or len(memory.messages) >= self.config.summary_threshold:
            await self.memory_manager._summarize(memory)
            return memory.summary

        return memory.summary

    # ==================== Long-term Memory Operations ====================

    async def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        category: str = "general",
    ) -> bool:
        """
        Set user preference.

        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
            category: Preference category

        Returns:
            Success status
        """
        try:
            await self.store_manager.set_user_preference(
                user_id=user_id,
                key=key,
                value=value,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set user preference: {e}")
            return False

    async def get_user_preference(
        self,
        user_id: str,
        key: str,
    ) -> Optional[Any]:
        """
        Get user preference.

        Args:
            user_id: User ID
            key: Preference key

        Returns:
            Preference value or None
        """
        return await self.store_manager.get_user_preference(user_id, key)

    async def get_user_preferences(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Get all user preferences.

        Args:
            user_id: User ID

        Returns:
            Dict of preferences
        """
        return await self.store_manager.search_user_preferences(user_id)

    async def store_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        attributes: Dict[str, Any] = None,
        relationships: List[Dict[str, str]] = None,
    ) -> bool:
        """
        Store entity memory.

        Args:
            entity_id: Entity ID
            entity_type: Entity type (person, organization, etc.)
            name: Entity name
            attributes: Entity attributes
            relationships: Entity relationships

        Returns:
            Success status
        """
        try:
            entity = EntityMemory(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                attributes=attributes or {},
                relationships=relationships or [],
            )
            await self.store_manager.store_entity(entity)
            return True
        except Exception as e:
            logger.error(f"Failed to store entity: {e}")
            return False

    async def get_entity(
        self,
        entity_id: str,
        entity_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID and type.

        Args:
            entity_id: Entity ID
            entity_type: Entity type

        Returns:
            Entity data or None
        """
        return await self.store_manager.get_entity(entity_type, entity_id)

    async def search_entities(
        self,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search entities.

        Args:
            entity_type: Optional entity type filter
            query: Optional search query
            limit: Maximum results

        Returns:
            List of matching entities
        """
        return await self.store_manager.search_entities(
            entity_type=entity_type,
            query=query,
            limit=limit,
        )

    async def store_knowledge(
        self,
        topic: str,
        content: str,
        source: Optional[str] = None,
        tags: List[str] = None,
    ) -> bool:
        """
        Store knowledge item.

        Args:
            topic: Knowledge topic
            content: Knowledge content
            source: Optional source
            tags: Optional tags

        Returns:
            Success status
        """
        try:
            knowledge = KnowledgeItem(
                topic=topic,
                content=content,
                source=source,
                tags=tags or [],
            )
            await self.store_manager.store_knowledge(knowledge)
            return True
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            return False

    async def search_knowledge(
        self,
        query: str,
        tags: List[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge items.

        Args:
            query: Search query
            tags: Optional tag filter
            limit: Maximum results

        Returns:
            List of matching knowledge items
        """
        return await self.store_manager.search_knowledge(
            query=query,
            tags=tags,
            limit=limit,
        )

    async def set_session_data(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """
        Set session-scoped data.

        Args:
            session_id: Session ID
            key: Data key
            value: Data value

        Returns:
            Success status
        """
        try:
            await self.store_manager.set_session_data(session_id, key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set session data: {e}")
            return False

    async def get_session_data(
        self,
        session_id: str,
        key: str,
    ) -> Optional[Any]:
        """
        Get session-scoped data.

        Args:
            session_id: Session ID
            key: Data key

        Returns:
            Data value or None
        """
        return await self.store_manager.get_session_data(session_id, key)

    async def get_user_context(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Get full context for a user.

        Args:
            user_id: User ID

        Returns:
            Combined context including preferences and entities
        """
        return await self.store_manager.get_context_for_user(user_id)

    # ==================== Statistics and Health ====================

    async def get_stats(self) -> MemoryStats:
        """
        Get memory statistics.

        Returns:
            Memory statistics
        """
        # Short-term memory stats
        total_conversations = len(self.memory_manager._memories)
        total_messages = sum(
            len(m.messages) for m in self.memory_manager._memories.values()
        )

        # Long-term memory stats
        store_stats = self.store.get_stats()

        return MemoryStats(
            total_conversations=total_conversations,
            total_messages=total_messages,
            total_memories=store_stats.get("total_items", 0),
            namespaces=store_stats.get("namespaces", []),
            memory_types=store_stats.get("by_type", {}),
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status
        """
        stats = await self.get_stats()

        return {
            "status": "healthy",
            "initialized": self._initialized,
            "persist_enabled": self.config.persist_enabled,
            "total_conversations": stats.total_conversations,
            "total_messages": stats.total_messages,
            "total_memories": stats.total_memories,
        }
