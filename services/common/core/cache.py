"""
Redis Cache Manager / Redis 缓存管理器
=====================================

统一的 Redis 缓存接口，支持多种数据类型和自动回退到内存缓存。

功能特性
--------
- 支持 Redis 和内存缓存两种模式
- 自动序列化/反序列化 (JSON, pickle for numpy)
- TTL 过期机制
- 连接池管理
- 优雅的错误处理和回退

缓存类型
--------
- EmbeddingCache: 向量嵌入缓存 (numpy arrays)
- SearchCache: 搜索结果缓存 (JSON serializable)
- GeneralCache: 通用键值缓存

使用示例
--------
```python
from app.core.cache import get_cache_manager, CacheType

cache = get_cache_manager()

# 通用缓存
await cache.set("key", {"data": "value"}, ttl=3600)
value = await cache.get("key")

# 嵌入缓存 (支持 numpy)
await cache.set_embedding("text_hash", embedding_array)
embedding = await cache.get_embedding("text_hash")

# 搜索结果缓存
await cache.set_search_results("query", results, ttl=300)
results = await cache.get_search_results("query")
```

Author: Intelligent Customer Service Team
Version: 1.0.0
"""
import asyncio
import json
import logging
import pickle
import hashlib
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class CacheType(str, Enum):
    """缓存类型枚举"""
    EMBEDDING = "embedding"  # 嵌入向量
    SEARCH = "search"  # 搜索结果
    GENERAL = "general"  # 通用缓存


class CacheBackend(str, Enum):
    """缓存后端类型"""
    REDIS = "redis"
    MEMORY = "memory"


# ============================================================================
# Memory Cache (Fallback)
# ============================================================================

class MemoryCache:
    """
    内存缓存实现 (LRU + TTL)。

    作为 Redis 不可用时的回退方案。

    Attributes
    ----------
    max_size : int
        最大缓存条目数

    default_ttl : float
        默认过期时间（秒）
    """

    def __init__(self, max_size: int = 10000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "sets": 0}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            value, expires_at = self._cache[key]

            # 检查过期
            if expires_at > 0 and time.time() > expires_at:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # LRU: 移动到末尾
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """设置缓存值"""
        async with self._lock:
            # 计算过期时间
            expires_at = 0.0
            if ttl is not None and ttl > 0:
                expires_at = time.time() + ttl
            elif self.default_ttl > 0:
                expires_at = time.time() + self.default_ttl

            # 如果已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 检查容量，LRU 驱逐
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, expires_at)
            self._stats["sets"] += 1
            return True

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        value = await self.get(key)
        return value is not None

    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            "backend": "memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "hit_rate": round(hit_rate, 3),
        }


# ============================================================================
# Redis Cache
# ============================================================================

class RedisCache:
    """
    Redis 缓存实现。

    支持异步操作和多种数据类型。

    Attributes
    ----------
    url : str
        Redis 连接 URL

    prefix : str
        缓存键前缀
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        prefix: str = "ics:",  # intelligent-customer-service
    ):
        self.url = url
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.prefix = prefix
        self._redis = None
        self._pool = None
        self._connected = False
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}

    async def connect(self) -> bool:
        """连接到 Redis"""
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.url,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                decode_responses=False,  # 返回 bytes，便于处理二进制数据
            )

            # 测试连接
            await self._redis.ping()
            self._connected = True
            logger.info(f"[Cache] Connected to Redis: {self.url}")
            return True

        except ImportError:
            logger.error("[Cache] redis package not installed. Run: pip install redis")
            return False
        except Exception as e:
            logger.error(f"[Cache] Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """断开 Redis 连接"""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("[Cache] Disconnected from Redis")

    def _make_key(self, key: str, cache_type: CacheType = CacheType.GENERAL) -> str:
        """生成带前缀的缓存键"""
        return f"{self.prefix}{cache_type.value}:{key}"

    async def get(
        self,
        key: str,
        cache_type: CacheType = CacheType.GENERAL
    ) -> Optional[Any]:
        """获取缓存值"""
        if not self._connected:
            return None

        try:
            full_key = self._make_key(key, cache_type)
            data = await self._redis.get(full_key)

            if data is None:
                self._stats["misses"] += 1
                return None

            # 反序列化
            value = self._deserialize(data, cache_type)
            self._stats["hits"] += 1
            return value

        except Exception as e:
            logger.warning(f"[Cache] Redis get error: {e}")
            self._stats["errors"] += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: CacheType = CacheType.GENERAL
    ) -> bool:
        """设置缓存值"""
        if not self._connected:
            return False

        try:
            full_key = self._make_key(key, cache_type)

            # 序列化
            data = self._serialize(value, cache_type)

            if ttl and ttl > 0:
                await self._redis.setex(full_key, ttl, data)
            else:
                await self._redis.set(full_key, data)

            self._stats["sets"] += 1
            return True

        except Exception as e:
            logger.warning(f"[Cache] Redis set error: {e}")
            self._stats["errors"] += 1
            return False

    async def delete(self, key: str, cache_type: CacheType = CacheType.GENERAL) -> bool:
        """删除缓存值"""
        if not self._connected:
            return False

        try:
            full_key = self._make_key(key, cache_type)
            result = await self._redis.delete(full_key)
            return result > 0
        except Exception as e:
            logger.warning(f"[Cache] Redis delete error: {e}")
            return False

    async def exists(self, key: str, cache_type: CacheType = CacheType.GENERAL) -> bool:
        """检查键是否存在"""
        if not self._connected:
            return False

        try:
            full_key = self._make_key(key, cache_type)
            return await self._redis.exists(full_key) > 0
        except Exception as e:
            logger.warning(f"[Cache] Redis exists error: {e}")
            return False

    async def clear(self, cache_type: Optional[CacheType] = None) -> None:
        """清空缓存"""
        if not self._connected:
            return

        try:
            if cache_type:
                pattern = f"{self.prefix}{cache_type.value}:*"
            else:
                pattern = f"{self.prefix}*"

            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"[Cache] Cleared cache with pattern: {pattern}")
        except Exception as e:
            logger.warning(f"[Cache] Redis clear error: {e}")

    def _serialize(self, value: Any, cache_type: CacheType) -> bytes:
        """序列化值"""
        if cache_type == CacheType.EMBEDDING:
            # numpy array 使用 pickle
            return pickle.dumps(value)
        else:
            # 其他类型使用 JSON
            return json.dumps(value, ensure_ascii=False).encode('utf-8')

    def _deserialize(self, data: bytes, cache_type: CacheType) -> Any:
        """反序列化值"""
        if cache_type == CacheType.EMBEDDING:
            return pickle.loads(data)
        else:
            return json.loads(data.decode('utf-8'))

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            "backend": "redis",
            "connected": self._connected,
            "url": self.url,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "errors": self._stats["errors"],
            "hit_rate": round(hit_rate, 3),
        }


# ============================================================================
# Cache Manager (Unified Interface)
# ============================================================================

class CacheManager:
    """
    统一缓存管理器。

    自动选择 Redis 或内存缓存，并提供统一接口。

    Features
    --------
    - 自动检测 Redis 可用性
    - 支持回退到内存缓存
    - 统一的 API 接口
    - 分类缓存 (embedding, search, general)

    Example
    -------
    ```python
    cache = CacheManager()
    await cache.initialize()

    # 存储嵌入向量
    await cache.set_embedding("hash_key", np.array([...]))

    # 获取嵌入向量
    embedding = await cache.get_embedding("hash_key")

    # 搜索结果缓存
    await cache.set_search_results("query", results)
    ```
    """

    def __init__(self):
        self._redis_cache: Optional[RedisCache] = None
        self._memory_cache: Optional[MemoryCache] = None
        self._backend: CacheBackend = CacheBackend.MEMORY
        self._initialized = False
        self._settings = None

    async def initialize(self) -> None:
        """初始化缓存管理器"""
        if self._initialized:
            return

        from config.settings import settings
        self._settings = settings

        # 始终初始化内存缓存作为后备
        self._memory_cache = MemoryCache(
            max_size=10000,
            default_ttl=settings.REDIS_CACHE_TTL
        )

        # 尝试连接 Redis
        if settings.REDIS_CACHE_ENABLED:
            self._redis_cache = RedisCache(
                url=settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            )

            if await self._redis_cache.connect():
                self._backend = CacheBackend.REDIS
                logger.info("[Cache] Using Redis backend")
            else:
                logger.warning("[Cache] Redis unavailable, falling back to memory cache")
                self._backend = CacheBackend.MEMORY
        else:
            logger.info("[Cache] Redis disabled, using memory cache")

        self._initialized = True

    @property
    def backend(self) -> CacheBackend:
        """当前使用的缓存后端"""
        return self._backend

    @property
    def is_redis(self) -> bool:
        """是否使用 Redis"""
        return self._backend == CacheBackend.REDIS

    # ==================== 通用缓存接口 ====================

    async def get(self, key: str) -> Optional[Any]:
        """获取通用缓存值"""
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.get(key, CacheType.GENERAL)
        return await self._memory_cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置通用缓存值"""
        ttl = ttl or self._settings.REDIS_CACHE_TTL
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.set(key, value, ttl, CacheType.GENERAL)
        return await self._memory_cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.delete(key, CacheType.GENERAL)
        return await self._memory_cache.delete(key)

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.exists(key, CacheType.GENERAL)
        return await self._memory_cache.exists(key)

    # ==================== 嵌入缓存接口 ====================

    def _hash_content(self, content: str) -> str:
        """生成内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def get_embedding(self, content: str) -> Optional[np.ndarray]:
        """获取嵌入向量缓存"""
        key = self._hash_content(content)
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.get(key, CacheType.EMBEDDING)
        return await self._memory_cache.get(f"emb:{key}")

    async def set_embedding(self, content: str, embedding: np.ndarray) -> bool:
        """设置嵌入向量缓存"""
        key = self._hash_content(content)
        ttl = self._settings.REDIS_EMBEDDING_CACHE_TTL if self._settings else 86400
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.set(key, embedding, ttl, CacheType.EMBEDDING)
        return await self._memory_cache.set(f"emb:{key}", embedding, ttl)

    async def get_embeddings_batch(
        self,
        contents: List[str]
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """
        批量获取嵌入缓存。

        Returns
        -------
        Tuple of:
            - cached_embeddings: 命中的嵌入向量列表
            - cached_indices: 命中的原始索引列表
            - uncached_contents: 未命中的内容列表
            - uncached_indices: 未命中的原始索引列表
        """
        cached_embeddings = []
        cached_indices = []
        uncached_contents = []
        uncached_indices = []

        for i, content in enumerate(contents):
            embedding = await self.get_embedding(content)
            if embedding is not None:
                cached_embeddings.append(embedding)
                cached_indices.append(i)
            else:
                uncached_contents.append(content)
                uncached_indices.append(i)

        return cached_embeddings, cached_indices, uncached_contents, uncached_indices

    async def set_embeddings_batch(
        self,
        contents: List[str],
        embeddings: List[np.ndarray]
    ) -> None:
        """批量设置嵌入缓存"""
        for content, embedding in zip(contents, embeddings):
            await self.set_embedding(content, embedding)

    # ==================== 搜索结果缓存接口 ====================

    def _search_key(self, query: str, top_k: int = 5) -> str:
        """生成搜索缓存键"""
        content = f"{query.strip().lower()}:{top_k}"
        return hashlib.md5(content.encode()).hexdigest()

    async def get_search_results(
        self,
        query: str,
        top_k: int = 5
    ) -> Optional[List[Dict]]:
        """获取搜索结果缓存"""
        key = self._search_key(query, top_k)
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.get(key, CacheType.SEARCH)
        return await self._memory_cache.get(f"search:{key}")

    async def set_search_results(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5
    ) -> bool:
        """设置搜索结果缓存"""
        key = self._search_key(query, top_k)
        ttl = self._settings.REDIS_SEARCH_CACHE_TTL if self._settings else 300
        if self._backend == CacheBackend.REDIS:
            return await self._redis_cache.set(key, results, ttl, CacheType.SEARCH)
        return await self._memory_cache.set(f"search:{key}", results, ttl)

    # ==================== 管理接口 ====================

    async def clear_all(self) -> None:
        """清空所有缓存"""
        if self._backend == CacheBackend.REDIS:
            await self._redis_cache.clear()
        await self._memory_cache.clear()
        logger.info("[Cache] All caches cleared")

    async def clear_embeddings(self) -> None:
        """清空嵌入缓存"""
        if self._backend == CacheBackend.REDIS:
            await self._redis_cache.clear(CacheType.EMBEDDING)
        # Memory cache doesn't support type-based clearing easily

    async def clear_search(self) -> None:
        """清空搜索缓存"""
        if self._backend == CacheBackend.REDIS:
            await self._redis_cache.clear(CacheType.SEARCH)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self._backend == CacheBackend.REDIS:
            return self._redis_cache.get_stats()
        return self._memory_cache.get_stats()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        result = {
            "backend": self._backend.value,
            "initialized": self._initialized,
        }

        if self._backend == CacheBackend.REDIS and self._redis_cache:
            try:
                await self._redis_cache._redis.ping()
                result["redis_connected"] = True
            except Exception as e:
                result["redis_connected"] = False
                result["redis_error"] = str(e)
        else:
            result["memory_cache_size"] = len(self._memory_cache._cache)

        return result

    async def close(self) -> None:
        """关闭缓存连接"""
        if self._redis_cache:
            await self._redis_cache.disconnect()


# ============================================================================
# Global Instance
# ============================================================================

_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """
    获取全局缓存管理器实例。

    Returns
    -------
    CacheManager
        初始化的缓存管理器

    Example
    -------
    ```python
    cache = await get_cache_manager()
    await cache.set("key", "value")
    ```
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager


def get_cache_manager_sync() -> CacheManager:
    """
    同步获取缓存管理器（用于非异步上下文）。

    注意：调用此方法前需确保缓存已初始化。
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
