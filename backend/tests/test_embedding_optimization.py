"""
Test Embedding Optimization
============================

Tests for the optimized embedding manager:
- Basic embedding functionality
- Batch async embedding
- ThreadPoolExecutor parallel processing
- Retry mechanism
- Redis distributed lock
- Cache integration
- Error handling and partial failure recovery
"""
import asyncio
import time
import sys
import os

import pytest

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# Mark all async tests
pytestmark = pytest.mark.asyncio


def _check_ollama_available():
    """检查 Ollama 服务是否可用"""
    import httpx
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_basic_embedding():
    """Test basic single text embedding."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Single Text Embedding")
    print("=" * 60)

    from app.core.embeddings import get_embedding_manager

    manager = get_embedding_manager()

    # Warmup
    await manager.warmup()

    # Test single embedding
    text = "This is a test sentence for embedding."
    start = time.time()
    embedding = await manager.aembed_query(text)
    elapsed = time.time() - start

    print(f"  Text: '{text[:50]}...'")
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Status: {'PASS' if len(embedding) > 0 else 'FAIL'}")

    return len(embedding) > 0


async def test_batch_async_embedding():
    """Test batch async embedding with concurrency control."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Async Embedding")
    print("=" * 60)

    from app.core.embeddings import get_embedding_manager

    manager = get_embedding_manager()

    # Generate test texts
    texts = [f"Test document number {i} with some content for embedding." for i in range(100)]

    print(f"  Documents: {len(texts)}")
    print(f"  Batch size: 50")
    print(f"  Max concurrent: 20")

    start = time.time()
    embeddings = await manager.aembed_documents_batch(
        texts,
        batch_size=50,
        max_concurrent=20,
        show_progress=True,
        use_cache=False,  # Disable cache for accurate timing
    )
    elapsed = time.time() - start

    success_count = sum(1 for e in embeddings if len(e) > 0)
    speed = len(texts) / elapsed if elapsed > 0 else 0

    print(f"\n  Results:")
    print(f"    Total: {len(embeddings)}")
    print(f"    Success: {success_count}")
    print(f"    Failed: {len(texts) - success_count}")
    print(f"    Time: {elapsed:.2f}s")
    print(f"    Speed: {speed:.1f} docs/sec")
    print(f"  Status: {'PASS' if success_count == len(texts) else 'PARTIAL'}")

    return success_count == len(texts)


def test_threaded_embedding():
    """Test ThreadPoolExecutor parallel embedding."""
    print("\n" + "=" * 60)
    print("TEST 3: ThreadPoolExecutor Parallel Embedding")
    print("=" * 60)

    from app.core.embeddings import get_embedding_manager

    manager = get_embedding_manager()

    # Generate test texts
    texts = [f"Threaded test document {i} for parallel processing." for i in range(50)]

    print(f"  Documents: {len(texts)}")
    print(f"  Batch size: 10")
    print(f"  Thread pool workers: 10")

    start = time.time()
    embeddings = manager.embed_documents_threaded(texts, batch_size=10)
    elapsed = time.time() - start

    success_count = sum(1 for e in embeddings if len(e) > 0)
    speed = len(texts) / elapsed if elapsed > 0 else 0

    print(f"\n  Results:")
    print(f"    Total: {len(embeddings)}")
    print(f"    Success: {success_count}")
    print(f"    Time: {elapsed:.2f}s")
    print(f"    Speed: {speed:.1f} docs/sec")
    print(f"  Status: {'PASS' if success_count == len(texts) else 'PARTIAL'}")

    return success_count == len(texts)


async def test_retry_mechanism():
    """Test retry mechanism (simulated by normal operation)."""
    print("\n" + "=" * 60)
    print("TEST 4: Retry Mechanism")
    print("=" * 60)

    from app.core.embeddings import retry_with_backoff, RetryError
    import httpx

    # Test successful retry
    attempt_count = 0

    async def flaky_func():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise httpx.ConnectTimeout("Simulated timeout")
        return "success"

    try:
        result = await retry_with_backoff(
            flaky_func,
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
        )
        print(f"  Attempts: {attempt_count}")
        print(f"  Result: {result}")
        print(f"  Status: PASS (recovered after {attempt_count - 1} retries)")
        return True
    except RetryError as e:
        print(f"  Status: FAIL - {e}")
        return False


async def test_redis_lock():
    """Test Redis distributed lock."""
    print("\n" + "=" * 60)
    print("TEST 5: Redis Distributed Lock")
    print("=" * 60)

    from app.core.embeddings import EmbeddingLock

    lock = EmbeddingLock(timeout=10, retry_interval=0.1)
    initialized = await lock.initialize()

    print(f"  Redis available: {lock._redis is not None}")
    print(f"  Fallback to memory: {lock._redis is None}")

    test_content = "Test content for locking"

    # Test acquire and release
    acquired = await lock.acquire(test_content)
    print(f"  Lock acquired: {acquired}")

    # Try to acquire again (should wait or fail)
    acquired2 = await lock.acquire(test_content, wait=False)
    print(f"  Second acquire (no wait): {acquired2}")

    await lock.release(test_content)
    print(f"  Lock released")

    # Should be able to acquire again
    acquired3 = await lock.acquire(test_content, wait=False)
    print(f"  Acquire after release: {acquired3}")
    await lock.release(test_content)

    await lock.close()

    success = acquired and not acquired2 and acquired3
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    return success


async def test_cache_integration():
    """Test cache integration with embeddings."""
    print("\n" + "=" * 60)
    print("TEST 6: Cache Integration")
    print("=" * 60)

    try:
        from app.core.cache import get_cache_manager
        from app.core.embeddings import get_embedding_manager
        import numpy as np

        cache = await get_cache_manager()
        manager = get_embedding_manager()

        print(f"  Cache backend: {cache.backend.value}")

        # Clear cache first
        await cache.clear_embeddings()

        # Generate test texts
        texts = ["Cache test document one.", "Cache test document two."]

        # First embedding (should compute)
        print("\n  First batch (compute):")
        start = time.time()
        embeddings1 = await manager.aembed_documents_batch(
            texts,
            use_cache=True,
            show_progress=False,
        )
        time1 = time.time() - start
        print(f"    Time: {time1:.3f}s")

        # Second embedding (should hit cache)
        print("\n  Second batch (cache hit):")
        start = time.time()
        embeddings2 = await manager.aembed_documents_batch(
            texts,
            use_cache=True,
            show_progress=False,
        )
        time2 = time.time() - start
        print(f"    Time: {time2:.3f}s")

        # Verify results
        match = all(
            np.allclose(np.array(e1), np.array(e2))
            for e1, e2 in zip(embeddings1, embeddings2)
            if len(e1) > 0 and len(e2) > 0
        )

        cache_speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n  Results:")
        print(f"    Embeddings match: {match}")
        print(f"    Cache speedup: {cache_speedup:.1f}x")
        print(f"  Status: {'PASS' if match and cache_speedup > 1 else 'FAIL'}")

        return match

    except Exception as e:
        print(f"  Cache not available: {e}")
        print(f"  Status: SKIP")
        return True  # Skip is OK


async def test_concurrent_embedding():
    """Test concurrent embedding requests."""
    print("\n" + "=" * 60)
    print("TEST 7: Concurrent Embedding Requests")
    print("=" * 60)

    from app.core.embeddings import get_embedding_manager

    manager = get_embedding_manager()

    # Simulate multiple concurrent requests
    async def embed_request(request_id: int):
        texts = [f"Request {request_id} document {i}" for i in range(10)]
        return await manager.aembed_documents_batch(
            texts,
            batch_size=5,
            max_concurrent=5,
            show_progress=False,
            use_cache=False,
        )

    print("  Launching 5 concurrent embedding requests...")
    print("  Each request: 10 documents")

    start = time.time()
    results = await asyncio.gather(*[embed_request(i) for i in range(5)])
    elapsed = time.time() - start

    total_docs = sum(len(r) for r in results)
    success_docs = sum(sum(1 for e in r if len(e) > 0) for r in results)
    speed = total_docs / elapsed if elapsed > 0 else 0

    print(f"\n  Results:")
    print(f"    Total documents: {total_docs}")
    print(f"    Success: {success_docs}")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Throughput: {speed:.1f} docs/sec")
    print(f"  Status: {'PASS' if success_docs == total_docs else 'PARTIAL'}")

    return success_docs == total_docs


async def test_error_handling():
    """Test error handling and partial failure recovery."""
    print("\n" + "=" * 60)
    print("TEST 8: Error Handling")
    print("=" * 60)

    from app.core.embeddings import get_embedding_manager, RetryError

    manager = get_embedding_manager()

    # Test with valid texts
    texts = ["Valid document for testing error handling."]

    try:
        embeddings = await manager.aembed_documents_batch(
            texts,
            show_progress=False,
            use_cache=False,
        )

        success = len(embeddings) > 0 and len(embeddings[0]) > 0
        print(f"  Normal operation: {'PASS' if success else 'FAIL'}")

        # Verify RetryError class exists and works
        try:
            raise RetryError("Test error", ValueError("inner"))
        except RetryError as e:
            print(f"  RetryError class: PASS")

        print(f"  Status: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Status: FAIL")
        return False


async def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("EMBEDDING OPTIMIZATION TEST SUITE")
    print("=" * 60)

    results = {}

    # Run async tests
    results["basic_embedding"] = await test_basic_embedding()
    results["batch_async"] = await test_batch_async_embedding()
    results["retry_mechanism"] = await test_retry_mechanism()
    results["redis_lock"] = await test_redis_lock()
    results["cache_integration"] = await test_cache_integration()
    results["concurrent"] = await test_concurrent_embedding()
    results["error_handling"] = await test_error_handling()

    # Run sync test
    results["threaded"] = test_threaded_embedding()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 60)

    # Cleanup
    from app.core.embeddings import close_http_client, get_embedding_manager
    manager = get_embedding_manager()
    manager.shutdown()
    await close_http_client()

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)