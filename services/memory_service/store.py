"""
Long-term Memory Store - Knowledge storage for LangGraph Agent.
长期记忆存储模块，用于跨会话、跨用户的知识管理。

架构设计：
---------
```
Store Architecture:
├── BaseStore (抽象基类)
│   ├── InMemoryStore (内存存储，快速访问)
│   └── PersistentStore (JSON文件持久化)
│
├── Memory Types:
│   ├── UserPreference - 用户偏好设置
│   ├── KnowledgeItem - 知识条目
│   ├── EntityMemory - 实体记忆（人、物、事件）
│   └── SemanticMemory - 语义记忆（支持向量检索）
│
└── Namespaces:
    ├── user:{user_id} - 用户相关记忆
    ├── global - 全局共享知识
    └── session:{session_id} - 会话级别记忆
```
"""
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============== Memory Types ==============

class MemoryType(str, Enum):
    """Types of long-term memory."""
    USER_PREFERENCE = "user_preference"  # 用户偏好
    KNOWLEDGE = "knowledge"              # 知识条目
    ENTITY = "entity"                    # 实体记忆
    FACT = "fact"                        # 事实记忆
    SEMANTIC = "semantic"                # 语义记忆


class MemoryPriority(str, Enum):
    """Priority levels for memory items."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryItem(BaseModel):
    """Base model for memory items stored in the Store."""
    key: str
    namespace: str = "global"
    memory_type: MemoryType = MemoryType.KNOWLEDGE
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: MemoryPriority = MemoryPriority.NORMAL
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None  # 可选的过期时间
    embedding: Optional[List[float]] = None  # 用于语义检索

    def is_expired(self) -> bool:
        """Check if the memory item has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class UserPreference(BaseModel):
    """User preference memory."""
    user_id: str
    preference_key: str
    preference_value: Any
    category: str = "general"  # general, language, notification, etc.
    confidence: float = 1.0  # 置信度 0-1

    def to_memory_item(self) -> MemoryItem:
        """Convert to MemoryItem."""
        return MemoryItem(
            key=f"pref:{self.user_id}:{self.preference_key}",
            namespace=f"user:{self.user_id}",
            memory_type=MemoryType.USER_PREFERENCE,
            content={
                "key": self.preference_key,
                "value": self.preference_value,
                "category": self.category,
            },
            metadata={"confidence": self.confidence},
        )


class EntityMemory(BaseModel):
    """Entity memory for people, things, events."""
    entity_id: str
    entity_type: str  # person, organization, product, event, etc.
    name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, str]] = Field(default_factory=list)
    source: Optional[str] = None

    def to_memory_item(self) -> MemoryItem:
        """Convert to MemoryItem."""
        return MemoryItem(
            key=f"entity:{self.entity_type}:{self.entity_id}",
            namespace="global",
            memory_type=MemoryType.ENTITY,
            content={
                "entity_id": self.entity_id,
                "entity_type": self.entity_type,
                "name": self.name,
                "attributes": self.attributes,
                "relationships": self.relationships,
            },
            metadata={"source": self.source},
        )


class KnowledgeItem(BaseModel):
    """Knowledge item for storing facts and information."""
    topic: str
    content: str
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0

    def to_memory_item(self) -> MemoryItem:
        """Convert to MemoryItem."""
        key = hashlib.md5(f"{self.topic}:{self.content[:50]}".encode()).hexdigest()[:16]
        return MemoryItem(
            key=f"knowledge:{key}",
            namespace="global",
            memory_type=MemoryType.KNOWLEDGE,
            content={
                "topic": self.topic,
                "content": self.content,
                "source": self.source,
                "tags": self.tags,
            },
            metadata={"confidence": self.confidence},
        )


# ============== Store Interface ==============

class BaseStore(ABC):
    """
    Abstract base class for long-term memory stores.

    Implements the Store interface for LangGraph agents.
    """

    @abstractmethod
    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a value with the given key in the specified namespace.

        Args:
            namespace: The namespace for the key (e.g., "user:123", "global").
            key: The key to store the value under.
            value: The value to store.
            memory_type: Type of memory being stored.
            metadata: Optional metadata for the stored item.
        """
        pass

    @abstractmethod
    async def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[MemoryItem]:
        """
        Retrieve a value by its key from the specified namespace.

        Args:
            namespace: The namespace to search in.
            key: The key to retrieve.

        Returns:
            The stored MemoryItem or None if not found.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """
        Delete a value by its key from the specified namespace.

        Args:
            namespace: The namespace to delete from.
            key: The key to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def list(
        self,
        namespace: str,
        prefix: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """
        List all items in a namespace, optionally filtered by prefix and type.

        Args:
            namespace: The namespace to list items from.
            prefix: Optional key prefix to filter by.
            memory_type: Optional memory type to filter by.
            limit: Maximum number of items to return.

        Returns:
            List of MemoryItems.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search for items matching the query.

        Args:
            query: Search query string.
            namespace: Optional namespace to search in.
            memory_type: Optional memory type to filter by.
            limit: Maximum number of results.

        Returns:
            List of (MemoryItem, relevance_score) tuples.
        """
        pass

    @abstractmethod
    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all items in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            Number of items deleted.
        """
        pass


# ============== In-Memory Store ==============

class InMemoryStore(BaseStore):
    """
    In-memory implementation of the Store.

    Fast access but non-persistent. Ideal for:
    - Development and testing
    - Caching frequently accessed items
    - Short-lived session data
    """

    def __init__(self):
        """Initialize the in-memory store."""
        self._store: Dict[str, Dict[str, MemoryItem]] = {}
        logger.info("[Store] InMemoryStore initialized")

    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a value in memory."""
        if namespace not in self._store:
            self._store[namespace] = {}

        # Check if updating existing item
        existing = self._store[namespace].get(key)
        now = datetime.utcnow()

        item = MemoryItem(
            key=key,
            namespace=namespace,
            memory_type=memory_type,
            content=value,
            metadata=metadata or {},
            created_at=existing.created_at if existing else now,
            updated_at=now,
            access_count=existing.access_count if existing else 0,
        )

        self._store[namespace][key] = item
        logger.debug(f"[Store] Put: {namespace}/{key} (type={memory_type.value})")

    async def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[MemoryItem]:
        """Retrieve a value from memory."""
        if namespace not in self._store:
            return None

        item = self._store[namespace].get(key)
        if item:
            # Check expiration
            if item.is_expired():
                del self._store[namespace][key]
                return None
            item.touch()
            logger.debug(f"[Store] Get: {namespace}/{key} (access_count={item.access_count})")

        return item

    async def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """Delete a value from memory."""
        if namespace not in self._store:
            return False

        if key in self._store[namespace]:
            del self._store[namespace][key]
            logger.debug(f"[Store] Delete: {namespace}/{key}")
            return True

        return False

    async def list(
        self,
        namespace: str,
        prefix: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """List items in a namespace."""
        if namespace not in self._store:
            return []

        items = []
        for key, item in self._store[namespace].items():
            # Skip expired items
            if item.is_expired():
                continue

            # Filter by prefix
            if prefix and not key.startswith(prefix):
                continue

            # Filter by type
            if memory_type and item.memory_type != memory_type:
                continue

            items.append(item)

            if len(items) >= limit:
                break

        logger.debug(f"[Store] List: {namespace} (prefix={prefix}, type={memory_type}, count={len(items)})")
        return items

    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Tuple[MemoryItem, float]]:
        """Search for items (simple string matching for in-memory store)."""
        results = []
        query_lower = query.lower()

        namespaces = [namespace] if namespace else list(self._store.keys())

        for ns in namespaces:
            if ns not in self._store:
                continue

            for item in self._store[ns].values():
                # Skip expired items
                if item.is_expired():
                    continue

                # Filter by type
                if memory_type and item.memory_type != memory_type:
                    continue

                # Simple relevance scoring based on string matching
                content_str = json.dumps(item.content, ensure_ascii=False).lower()
                if query_lower in content_str:
                    # Calculate simple relevance score
                    score = content_str.count(query_lower) / len(content_str) * 100
                    results.append((item, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"[Store] Search: '{query}' (namespace={namespace}, results={len(results[:limit])})")
        return results[:limit]

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all items in a namespace."""
        if namespace not in self._store:
            return 0

        count = len(self._store[namespace])
        del self._store[namespace]
        logger.info(f"[Store] Clear namespace: {namespace} (deleted={count})")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total_items = sum(len(items) for items in self._store.values())
        return {
            "type": "InMemoryStore",
            "namespaces": list(self._store.keys()),
            "namespace_count": len(self._store),
            "total_items": total_items,
        }


# ============== Persistent Store ==============

class PersistentStore(BaseStore):
    """
    JSON file-based persistent store.

    Stores data in JSON files for persistence across restarts.
    Supports both in-memory caching and disk persistence.
    """

    def __init__(
        self,
        storage_path: str,
        auto_save: bool = True,
        save_interval: int = 10,  # Save after N writes
    ):
        """
        Initialize the persistent store.

        Args:
            storage_path: Path to store JSON files.
            auto_save: Whether to auto-save on writes.
            save_interval: Number of writes before auto-save (if auto_save=True).
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._write_count = 0

        # In-memory cache
        self._cache: Dict[str, Dict[str, MemoryItem]] = {}
        self._dirty_namespaces: set = set()

        # Load existing data
        self._load_all()

        logger.info(f"[Store] PersistentStore initialized at {self.storage_path}")

    def _get_namespace_file(self, namespace: str) -> Path:
        """Get the file path for a namespace."""
        # Sanitize namespace for filename
        safe_name = namespace.replace(":", "_").replace("/", "_")
        return self.storage_path / f"{safe_name}.json"

    def _load_all(self) -> None:
        """Load all persisted data into memory."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                namespace = data.get("namespace", file_path.stem)
                items = {}

                for key, item_data in data.get("items", {}).items():
                    try:
                        items[key] = MemoryItem(**item_data)
                    except Exception as e:
                        logger.warning(f"[Store] Failed to load item {key}: {e}")

                self._cache[namespace] = items
                logger.debug(f"[Store] Loaded namespace: {namespace} ({len(items)} items)")

            except Exception as e:
                logger.error(f"[Store] Failed to load {file_path}: {e}")

        logger.info(f"[Store] Loaded {len(self._cache)} namespaces from disk")

    async def _save_namespace(self, namespace: str) -> None:
        """Save a namespace to disk."""
        if namespace not in self._cache:
            return

        file_path = self._get_namespace_file(namespace)
        data = {
            "namespace": namespace,
            "updated_at": datetime.utcnow().isoformat(),
            "items": {
                key: item.model_dump(mode="json")
                for key, item in self._cache[namespace].items()
            },
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[Store] Saved namespace: {namespace}")
        except Exception as e:
            logger.error(f"[Store] Failed to save {namespace}: {e}")

    async def _maybe_auto_save(self, namespace: str) -> None:
        """Auto-save if conditions are met."""
        self._dirty_namespaces.add(namespace)
        self._write_count += 1

        if self.auto_save and self._write_count >= self.save_interval:
            await self.flush()
            self._write_count = 0

    async def flush(self) -> None:
        """Flush all dirty namespaces to disk."""
        for namespace in list(self._dirty_namespaces):
            await self._save_namespace(namespace)
        self._dirty_namespaces.clear()
        logger.debug(f"[Store] Flushed all dirty namespaces")

    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a value persistently."""
        if namespace not in self._cache:
            self._cache[namespace] = {}

        existing = self._cache[namespace].get(key)
        now = datetime.utcnow()

        item = MemoryItem(
            key=key,
            namespace=namespace,
            memory_type=memory_type,
            content=value,
            metadata=metadata or {},
            created_at=existing.created_at if existing else now,
            updated_at=now,
            access_count=existing.access_count if existing else 0,
        )

        self._cache[namespace][key] = item
        await self._maybe_auto_save(namespace)
        logger.debug(f"[Store] Put: {namespace}/{key}")

    async def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[MemoryItem]:
        """Retrieve a value from the store."""
        if namespace not in self._cache:
            return None

        item = self._cache[namespace].get(key)
        if item:
            if item.is_expired():
                del self._cache[namespace][key]
                self._dirty_namespaces.add(namespace)
                return None
            item.touch()

        return item

    async def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """Delete a value from the store."""
        if namespace not in self._cache:
            return False

        if key in self._cache[namespace]:
            del self._cache[namespace][key]
            await self._maybe_auto_save(namespace)
            return True

        return False

    async def list(
        self,
        namespace: str,
        prefix: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """List items in a namespace."""
        if namespace not in self._cache:
            return []

        items = []
        for key, item in self._cache[namespace].items():
            if item.is_expired():
                continue
            if prefix and not key.startswith(prefix):
                continue
            if memory_type and item.memory_type != memory_type:
                continue

            items.append(item)
            if len(items) >= limit:
                break

        return items

    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Tuple[MemoryItem, float]]:
        """Search for items."""
        results = []
        query_lower = query.lower()

        namespaces = [namespace] if namespace else list(self._cache.keys())

        for ns in namespaces:
            if ns not in self._cache:
                continue

            for item in self._cache[ns].values():
                if item.is_expired():
                    continue
                if memory_type and item.memory_type != memory_type:
                    continue

                content_str = json.dumps(item.content, ensure_ascii=False).lower()
                if query_lower in content_str:
                    score = content_str.count(query_lower) / len(content_str) * 100
                    results.append((item, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all items in a namespace."""
        if namespace not in self._cache:
            return 0

        count = len(self._cache[namespace])
        del self._cache[namespace]

        # Remove the file
        file_path = self._get_namespace_file(namespace)
        if file_path.exists():
            file_path.unlink()

        logger.info(f"[Store] Clear namespace: {namespace} (deleted={count})")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total_items = sum(len(items) for items in self._cache.values())
        return {
            "type": "PersistentStore",
            "storage_path": str(self.storage_path),
            "namespaces": list(self._cache.keys()),
            "namespace_count": len(self._cache),
            "total_items": total_items,
            "dirty_namespaces": len(self._dirty_namespaces),
        }


# ============== Store Manager ==============

class StoreManager:
    """
    Manager for long-term memory stores.

    Provides high-level APIs for common memory operations
    and manages store lifecycle.
    """

    def __init__(self, store: BaseStore):
        """
        Initialize the store manager.

        Args:
            store: The underlying store implementation.
        """
        self.store = store
        logger.info(f"[StoreManager] Initialized with {type(store).__name__}")

    # ============== User Preferences ==============

    async def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        category: str = "general",
        confidence: float = 1.0,
    ) -> None:
        """Set a user preference."""
        pref = UserPreference(
            user_id=user_id,
            preference_key=key,
            preference_value=value,
            category=category,
            confidence=confidence,
        )
        item = pref.to_memory_item()
        await self.store.put(
            namespace=item.namespace,
            key=item.key,
            value=item.content,
            memory_type=item.memory_type,
            metadata=item.metadata,
        )
        logger.info(f"[StoreManager] Set user preference: {user_id}/{key}={value}")

    async def get_user_preference(
        self,
        user_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get a user preference value."""
        item = await self.store.get(
            namespace=f"user:{user_id}",
            key=f"pref:{user_id}:{key}",
        )
        if item:
            return item.content.get("value", default)
        return default

    async def get_all_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get all preferences for a user."""
        items = await self.store.list(
            namespace=f"user:{user_id}",
            prefix="pref:",
            memory_type=MemoryType.USER_PREFERENCE,
        )

        prefs = {}
        for item in items:
            if category and item.content.get("category") != category:
                continue
            prefs[item.content["key"]] = item.content["value"]

        return prefs

    # ============== Entity Memory ==============

    async def store_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, str]]] = None,
        source: Optional[str] = None,
    ) -> None:
        """Store an entity in memory."""
        entity = EntityMemory(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            attributes=attributes or {},
            relationships=relationships or [],
            source=source,
        )
        item = entity.to_memory_item()
        await self.store.put(
            namespace=item.namespace,
            key=item.key,
            value=item.content,
            memory_type=item.memory_type,
            metadata=item.metadata,
        )
        logger.info(f"[StoreManager] Stored entity: {entity_type}/{name}")

    async def get_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by type and ID."""
        item = await self.store.get(
            namespace="global",
            key=f"entity:{entity_type}:{entity_id}",
        )
        return item.content if item else None

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for entities."""
        results = await self.store.search(
            query=query,
            namespace="global",
            memory_type=MemoryType.ENTITY,
            limit=limit,
        )

        entities = []
        for item, score in results:
            if entity_type and item.content.get("entity_type") != entity_type:
                continue
            entities.append({
                **item.content,
                "relevance_score": score,
            })

        return entities

    # ============== Knowledge Management ==============

    async def store_knowledge(
        self,
        topic: str,
        content: str,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
    ) -> str:
        """Store a knowledge item."""
        knowledge = KnowledgeItem(
            topic=topic,
            content=content,
            source=source,
            tags=tags or [],
            confidence=confidence,
        )
        item = knowledge.to_memory_item()
        await self.store.put(
            namespace=item.namespace,
            key=item.key,
            value=item.content,
            memory_type=item.memory_type,
            metadata=item.metadata,
        )
        logger.info(f"[StoreManager] Stored knowledge: {topic}")
        return item.key

    async def search_knowledge(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for knowledge items."""
        results = await self.store.search(
            query=query,
            namespace="global",
            memory_type=MemoryType.KNOWLEDGE,
            limit=limit * 2,  # Get more to filter
        )

        knowledge_items = []
        for item, score in results:
            # Filter by tags if specified
            if tags:
                item_tags = item.content.get("tags", [])
                if not any(tag in item_tags for tag in tags):
                    continue

            knowledge_items.append({
                **item.content,
                "relevance_score": score,
            })

            if len(knowledge_items) >= limit:
                break

        return knowledge_items

    # ============== Session Memory ==============

    async def set_session_data(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set session-specific data."""
        await self.store.put(
            namespace=f"session:{session_id}",
            key=key,
            value=value,
            memory_type=MemoryType.FACT,
        )

    async def get_session_data(
        self,
        session_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get session-specific data."""
        item = await self.store.get(
            namespace=f"session:{session_id}",
            key=key,
        )
        return item.content if item else default

    async def clear_session(self, session_id: str) -> int:
        """Clear all session data."""
        return await self.store.clear_namespace(f"session:{session_id}")

    # ============== Utility Methods ==============

    async def get_context_for_user(
        self,
        user_id: str,
        include_preferences: bool = True,
        include_entities: bool = True,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Get relevant context for a user.

        Returns combined preferences and related entities.
        """
        context = {
            "user_id": user_id,
            "preferences": {},
            "entities": [],
        }

        if include_preferences:
            context["preferences"] = await self.get_all_user_preferences(user_id)

        if include_entities:
            # Get user-specific entities
            user_items = await self.store.list(
                namespace=f"user:{user_id}",
                memory_type=MemoryType.ENTITY,
                limit=limit,
            )
            context["entities"] = [item.content for item in user_items]

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        if hasattr(self.store, "get_stats"):
            return self.store.get_stats()
        return {"type": type(self.store).__name__}


# ============== Factory Functions ==============

def create_memory_store(
    store_type: str = "memory",
    storage_path: Optional[str] = None,
    **kwargs,
) -> BaseStore:
    """
    Create a memory store instance.

    Args:
        store_type: Type of store ("memory" or "persistent").
        storage_path: Path for persistent storage.
        **kwargs: Additional arguments for the store.

    Returns:
        A BaseStore instance.
    """
    if store_type == "memory":
        return InMemoryStore()
    elif store_type == "persistent":
        if not storage_path:
            storage_path = "./data/memory_store"
        return PersistentStore(storage_path, **kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
