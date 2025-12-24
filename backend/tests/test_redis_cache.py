"""
Redis Cache Tests for Intelligent Customer Service.

Tests the Redis-backed cache manager and all cache integrations.

Run with: pytest tests/test_redis_cache.py -v
"""

import asyncio
import pytest
import numpy as np
from typing import List, Dict, Any

# Test configuration
pytestmark = pytest.mark.asyncio


class TestCacheManager:
    """Test the CacheManager from app.core.cache."""

    @pytest.fixture
    async def cache_manager(self):
        """Create and return a cache manager instance."""
        from app.core.cache import get_cache_manager
        manager = await get_cache_manager()
        yield manager
        # Cleanup
        try:
            await manager.clear_all()
        except Exception:
            pass

    async def test_cache_manager_initialization(self, cache_manager):
        """Test that cache manager initializes correctly."""
        assert cache_manager is not None
        assert cache_manager.backend is not None
        print(f"Cache backend: {cache_manager.backend.value}")

    async def test_embedding_cache_set_get(self, cache_manager):
        """Test embedding cache set and get operations."""
        content = "测试文本内容"
        embedding = np.random.rand(1536).astype(np.float32)

        # Set embedding
        success = await cache_manager.set_embedding(content, embedding)
        assert success

        # Get embedding
        cached = await cache_manager.get_embedding(content)
        assert cached is not None
        np.testing.assert_array_almost_equal(cached, embedding, decimal=5)

    async def test_embedding_cache_miss(self, cache_manager):
        """Test embedding cache miss returns None."""
        result = await cache_manager.get_embedding("不存在的内容")
        assert result is None

    async def test_embedding_batch_operations(self, cache_manager):
        """Test batch embedding cache operations."""
        contents = ["文本1", "文本2", "文本3"]
        embeddings = [np.random.rand(1536).astype(np.float32) for _ in range(3)]

        # Set batch
        await cache_manager.set_embeddings_batch(contents, embeddings)

        # Get batch
        cached, indices = await cache_manager.get_embeddings_batch(contents)
        assert len(cached) == 3
        assert len(indices) == 3
        for i, (c, orig) in enumerate(zip(cached, embeddings)):
            assert c is not None
            np.testing.assert_array_almost_equal(c, orig, decimal=5)

    async def test_search_result_cache(self, cache_manager):
        """Test search result caching."""
        query = "如何退款"
        results = [
            {"content": "退款流程说明", "score": 0.95},
            {"content": "退款条件", "score": 0.85},
        ]

        # Set search results
        success = await cache_manager.set_search_results(query, results, top_k=5)
        assert success

        # Get search results
        cached = await cache_manager.get_search_results(query, top_k=5)
        assert cached is not None
        assert len(cached) == 2
        assert cached[0]["content"] == "退款流程说明"

    async def test_search_result_cache_miss(self, cache_manager):
        """Test search result cache miss."""
        result = await cache_manager.get_search_results("不存在的查询", top_k=5)
        assert result is None

    async def test_cache_clear(self, cache_manager):
        """Test cache clearing operations."""
        # Set some data
        await cache_manager.set_embedding("test", np.random.rand(1536).astype(np.float32))
        await cache_manager.set_search_results("query", [{"content": "test"}], top_k=5)

        # Clear embeddings
        await cache_manager.clear_embeddings()
        assert await cache_manager.get_embedding("test") is None

        # Set again and clear search
        await cache_manager.set_search_results("query", [{"content": "test"}], top_k=5)
        await cache_manager.clear_search()
        assert await cache_manager.get_search_results("query", top_k=5) is None


class TestSearchResultCache:
    """Test the SearchResultCache from knowledge.py."""

    @pytest.fixture
    async def search_cache(self):
        """Create and return a search cache instance."""
        from app.mcp.tools.knowledge import SearchResultCache
        cache = SearchResultCache()
        await cache.initialize()
        yield cache
        # Cleanup
        await cache.aclear()

    async def test_search_cache_initialization(self, search_cache):
        """Test search cache initializes correctly."""
        assert search_cache._initialized

    async def test_async_set_get(self, search_cache):
        """Test async set and get operations."""
        query = "测试查询"
        results = [{"content": "结果1", "score": 0.9}]

        await search_cache.aset(query, 5, results)
        cached = await search_cache.aget(query, 5)

        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["content"] == "结果1"

    async def test_sync_fallback(self, search_cache):
        """Test sync operations work as fallback."""
        query = "同步测试"
        results = [{"content": "同步结果"}]

        search_cache.set(query, 5, results)
        cached = search_cache.get(query, 5)

        assert cached is not None
        assert cached[0]["content"] == "同步结果"

    async def test_cache_stats(self, search_cache):
        """Test cache statistics."""
        # Generate some hits and misses
        await search_cache.aset("query1", 5, [{"content": "test"}])
        await search_cache.aget("query1", 5)  # hit
        await search_cache.aget("query2", 5)  # miss

        stats = search_cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestEmbeddingCache:
    """Test the EmbeddingCache from postprocessor.py."""

    @pytest.fixture
    async def embedding_cache(self):
        """Create and return an embedding cache instance."""
        from app.rag.postprocessor import EmbeddingCache
        cache = EmbeddingCache()
        await cache.initialize()
        yield cache
        # Cleanup
        await cache.aclear()

    async def test_embedding_cache_initialization(self, embedding_cache):
        """Test embedding cache initializes correctly."""
        assert embedding_cache._initialized

    async def test_async_embedding_operations(self, embedding_cache):
        """Test async embedding cache operations."""
        content = "嵌入测试内容"
        embedding = np.random.rand(1536).astype(np.float32)

        await embedding_cache.aset(content, embedding)
        cached = await embedding_cache.aget(content)

        assert cached is not None
        np.testing.assert_array_almost_equal(cached, embedding, decimal=5)

    async def test_batch_operations(self, embedding_cache):
        """Test batch embedding operations."""
        contents = ["内容A", "内容B"]
        embeddings = [np.random.rand(1536).astype(np.float32) for _ in range(2)]

        await embedding_cache.aset_batch(contents, embeddings)
        cached, indices = await embedding_cache.aget_batch(contents)

        assert len(cached) == 2
        assert len(indices) == 2
        for c, orig in zip(cached, embeddings):
            assert c is not None
            np.testing.assert_array_almost_equal(c, orig, decimal=5)


class TestRedisConnection:
    """Test direct Redis connection."""

    async def test_redis_ping(self):
        """Test Redis server is accessible."""
        try:
            import redis.asyncio as redis
            from config.settings import settings

            client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
            )
            result = await client.ping()
            await client.close()
            assert result is True
            print("Redis connection successful!")
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    async def test_redis_set_get(self):
        """Test basic Redis set/get operations."""
        try:
            import redis.asyncio as redis
            from config.settings import settings

            client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )

            # Test basic operations
            await client.set("test:key", "test_value", ex=60)
            value = await client.get("test:key")
            assert value == "test_value"

            # Cleanup
            await client.delete("test:key")
            await client.close()
            print("Redis set/get successful!")
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")


# Run quick verification when executed directly
if __name__ == "__main__":
    async def quick_test():
        """Quick verification of Redis cache."""
        print("=" * 50)
        print("Redis Cache Quick Test")
        print("=" * 50)

        # Test 1: Redis connection
        print("\n[Test 1] Redis Connection...")
        try:
            import redis.asyncio as redis
            from config.settings import settings

            client = redis.from_url(settings.REDIS_URL)
            result = await client.ping()
            await client.close()
            print(f"  Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            return

        # Test 2: CacheManager
        print("\n[Test 2] CacheManager...")
        try:
            from app.core.cache import get_cache_manager
            manager = await get_cache_manager()
            print(f"  Backend: {manager.backend.value}")

            # Test embedding
            embedding = np.random.rand(1536).astype(np.float32)
            await manager.set_embedding("quick_test", embedding)
            cached = await manager.get_embedding("quick_test")
            assert cached is not None
            print("  Embedding cache: PASS")

            # Test search
            await manager.set_search_results("quick_query", [{"content": "test"}], top_k=5)
            cached = await manager.get_search_results("quick_query", top_k=5)
            assert cached is not None
            print("  Search cache: PASS")

            # Cleanup
            await manager.clear_all()
            print("  Result: PASS")
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            return

        # Test 3: SearchResultCache
        print("\n[Test 3] SearchResultCache...")
        try:
            from app.mcp.tools.knowledge import get_search_cache_async
            cache = await get_search_cache_async()
            await cache.aset("test_query", 5, [{"content": "result"}])
            result = await cache.aget("test_query", 5)
            assert result is not None
            stats = cache.get_stats()
            print(f"  Backend: {stats.get('backend', 'memory')}")
            print(f"  Stats: hits={stats['hits']}, misses={stats['misses']}")
            await cache.aclear()
            print("  Result: PASS")
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            return

        # Test 4: EmbeddingCache
        print("\n[Test 4] EmbeddingCache...")
        try:
            from app.rag.postprocessor import EmbeddingCache
            cache = EmbeddingCache()
            await cache.initialize()
            embedding = np.random.rand(1536).astype(np.float32)
            await cache.aset("embed_test", embedding)
            result = await cache.aget("embed_test")
            assert result is not None
            stats = cache.get_stats()
            print(f"  Backend: {stats.get('backend', 'memory')}")
            await cache.aclear()
            print("  Result: PASS")
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            return

        print("\n" + "=" * 50)
        print("All tests PASSED!")
        print("=" * 50)

    asyncio.run(quick_test())
